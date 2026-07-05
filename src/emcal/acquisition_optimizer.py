"""AcquisitionOptimizer: the scipy-based acquisition/objective optimization loop used by
GPBODriver's per-iteration BO step (starting-point generation, scipy.optimize.minimize,
and the running-best-candidate scratch state).

Kept in its own module, separate from the pure-math acquisition.py (ExpectedImprovement),
for the same reason emulators.py stays separate from acquisition.py: this class
orchestrates GP/simulator calls and scipy, it doesn't implement the EI math itself.

Two collaborator references are handled with extra care (PHASE8_AUDIT.md §3.8's two
riskiest seams):

1. gp_emulator is never cached at construction and never assigned directly by this class
   -- GPBODriver's `gp_emulator` is a @property whose setter re-syncs
   `self.acq_optimizer.gp_emulator = value` (and asserts the identity holds) on every
   assignment, including the two sites where GPBODriver *replaces* (not just mutates) its
   own gp_emulator every restart. A construction-time or cached reference here would go
   silently stale starting on restart 2.
2. The rng.Generator is never stored on this class either -- it is passed explicitly into
   every method that needs it (optimize, _objective), so RNG streams can't desync from a
   restored checkpoint.
"""
import numpy as np
import warnings
import scipy.optimize as optimize
import scipy.spatial.distance as distance

from .data import CandidateSet
from .methods import GPBOMethod
from .enums import MethodName


def get_best_error(gp_emulator, method, exp_data, create_data_fn):
    """
    Helper function to calculate the best error and squared error calculations over x given the method.

    Parameters
    ----------
    gp_emulator: GPEmulator
        The trained GP emulator (ObjectiveGP or EmulatorGP)
    method: GPBOMethod
        Class containing GPBO method information
    exp_data: Data
        Experimental data containing at least exp_data.x_vals, exp_data.y_vals
    create_data_fn: callable
        Callable with signature (theta_array, get_y=...) -> Data, used to build the
        best-error Data instance from the winning theta (e.g.
        GPBODriver.create_data_instance_from_theta)

    Returns
    -------
    be_data: Data
        Contains best_error as an instance of the data class
    be_metrics: tuple(float, np.ndarray, np.ndarray)
        The min_SSE, param at min_SSE, and squared residuals
    """
    if method.is_emulator == False:
        # Type 1 best error is inferred from training data
        best_error, be_theta, train_idx = gp_emulator.calc_best_error()
        best_errors_x = None
        be_data = create_data_fn(be_theta.flatten(), get_y=False)
        be_data.y_vals = np.atleast_1d(
            gp_emulator.train_data.y_vals[train_idx]
        )
    else:
        # Type 2 best error must be calculated given the experimental data
        best_error, be_theta, best_errors_x, train_idx = (
            gp_emulator.calc_best_error(method, exp_data)
        )
        be_data = create_data_fn(be_theta.flatten(), get_y=False)
        be_data.y_vals = gp_emulator.train_data.y_vals[
            train_idx[0] : train_idx[1]
        ]

    be_metrics = best_error, be_theta, best_errors_x

    return be_data, be_metrics


class AcquisitionOptimizer:
    """
    Owns the scipy-based acquisition/SSE optimization loop for one BO iteration:
    starting-point generation, scipy.optimize.minimize, and the running-best-candidate
    scratch state (min_obj_class/min_obj_val/min_obj_prediction).

    Methods
    --------------
    __init__
    get_best_error()
    make_starting_points(best_error_metrics, rng_seed)
    optimize(opt_obj, rng, get_y=False, with_noise=False)
    """

    def __init__(self, cs_params, method, simulator, exp_data, ep_bias, sse_penalty,
                 sg_mc_samples, create_data_fn):
        """
        Parameters
        ----------
        cs_params: BOConfig
            Class containing the values associated with BOConfig
        method: GPBOMethod
            Class containing GPBO method information
        simulator: Simulator
            Class containing values of simulation parameters
        exp_data: Data
            Experimental data containing at least exp_data.x_vals, exp_data.y_vals
        ep_bias: ExplorationBias
            Class containing exploration parameter info
        sse_penalty: float
            The penalty _objective returns for a candidate theta that causes nan values
        sg_mc_samples: int
            Number of samples to use for the Tasmanian sparse grid or Monte Carlo approaches
        create_data_fn: callable
            Callable with signature (theta_array, get_y=..., with_noise=...) -> Data, used to
            build Data instances from a winning theta (e.g.
            GPBODriver.create_data_instance_from_theta)
        """
        self.cs_params = cs_params
        self.method = method
        self.simulator = simulator
        self.exp_data = exp_data
        self.ep_bias = ep_bias
        self.sse_penalty = sse_penalty
        self.sg_mc_samples = sg_mc_samples
        self.create_data_fn = create_data_fn
        # Re-synced explicitly by GPBODriver's gp_emulator property setter after every
        # gp_emulator replacement -- never set here at construction, never cached beyond
        # what the driver last assigned. See module docstring, risk #1.
        self.gp_emulator = None
        # Set once per iteration by make_starting_points, then deliberately REUSED across
        # both optimize() calls in that same iteration (PHASE8_AUDIT.md §3.8 risk #4) --
        # do not "fix" this reuse.
        self.opt_start_pts = None
        self.min_obj_class = None
        self.min_obj_val = None
        self.min_obj_prediction = None

    def get_best_error(self):
        """
        Helper function to calculate the best error and squared error calculations over x given the method.

        Returns
        -------
        be_data: Data
            Contains best_error as an instance of the data class
        be_metrics: tuple(float, np.ndarray, np.ndarray)
            The min_SSE, param at min_SSE, and squared residuals
        """
        return get_best_error(
            self.gp_emulator, self.method, self.exp_data, self.create_data_fn,
        )

    def make_starting_points(self, best_error_metrics, rng_seed):
        """
        Makes starting point for optimization with scipy

        Parameters
        -----------
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error, best error parameter set, and best_error_x values of the method. Hint: Use self.get_best_error()

        Returns
        --------
        starting_pts: np.ndarray
            Array of parameter set initializations for self.optimize
        """
        # Note: Could make this generate 2 sets of starting points based on whether you want to optimize sse or ei
        # For sparse grid and mc methods
        if self.method.uses_sparse_grid == True or self.method.uses_monte_carlo == True:
            starting_pts = self._gen_start_pts_mc_sparse(best_error_metrics, rng_seed)
        else:
            starting_pts = self._gen_start_pts_not_mc_sparse(rng_seed)

        return starting_pts

    def _gen_start_pts_mc_sparse(self, best_error_metrics, rng_seed):
        """
        Makes starting point for optimization with scipy if using sparse grid or Monte Carlo methods

        Parameters
        -----------
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error, best error parameter set, and best_error_x values of the method

        Returns
        --------
        starting_pts: np.ndarray
            Array of parameter set initializations for self.optimize
        """
        # Generate n LHS Theta vals
        num_mc_theta = 500
        theta_vals = self.simulator.generate_parameter_samples(num_mc_theta, rng_seed)

        # Add repeated theta_vals and experimental x values
        rep_theta_vals = np.repeat(theta_vals, len(self.exp_data.x_vals), axis=0)
        rep_x_vals = np.vstack([self.exp_data.x_vals] * num_mc_theta)

        # Create instance of Data Class
        sp_data = CandidateSet(
            rep_theta_vals,
            rep_x_vals,
            bounds_theta=self.simulator.bounds_theta_reg,
            bounds_x=self.simulator.bounds_x,
            sep_fact=self.cs_params.sep_fact,
        )

        # Evaluate GP mean and Var (This is the slowest step)
        feat_sp_data = self.gp_emulator.featurize_data(sp_data)
        pred = self.gp_emulator.predict(
            data=sp_data, featurized_data=feat_sp_data
        )

        # Evaluate GP SSE and SSE_Var (This is the 2nd slowest step). Reuses `pred` (predict()
        # is the slowest step) instead of re-reading data.gp_mean/data.gp_covar.
        sp_data_sse_mean, sp_data_sse_var = self.gp_emulator.predict_sse(
            data=sp_data, method=self.method, exp_data=self.exp_data, prediction=pred
        )

        # Note - Use Sparse grid EI for approximations
        # Evaluate EI using Sparse Grid or EI (This is relatively quick). Reuses `pred`
        # instead of re-reading data.gp_mean/data.gp_covar.
        method_3 = GPBOMethod(MethodName(3))
        sp_data_ei, iter_max_ei_terms = self.gp_emulator.expected_improvement(
            data=sp_data, exp_data=self.exp_data, ep_bias=self.ep_bias,
            best_error_metrics=best_error_metrics, method=method_3, gp_prediction=pred
        )

        ##Sort by min(-ei)
        # Create a list of tuples containing indices and values
        indexed_values = list(enumerate(-1 * sp_data_ei))  # argmin(-ei) = argmax(ei)

        # Sort the list of tuples based on values
        sorted_values = sorted(indexed_values, key=lambda x: x[1])

        # Extract the indices from the sorted list
        min_indices = [index for index, _ in sorted_values]
        # Sets the points in order based on the indices
        all_pts = theta_vals[min_indices]

        # Choose top retrain_gp points as starting points
        starting_pts = all_pts[: self.cs_params.reoptimize_obj + 1]

        return starting_pts

    def _gen_start_pts_not_mc_sparse(self, rng_seed):
        """
        Makes starting point for optimization with scipy if not using sparse grid or Monte Carlo methods

        Returns
        --------
        starting_pts: np.ndarray
            Array of parameter set initializations for self.optimize
        """
        # Create starting points equal to number of retrain_gp
        starting_pts = self.simulator.generate_parameter_samples(
            self.cs_params.reoptimize_obj + 1, rng_seed
        )

        return starting_pts

    def optimize(self, opt_obj, rng, get_y=False, with_noise=False):
        """
        Optimizes a function with scipy.optimize

        Parameters
        ----------
        opt_obj: str
            Which objective to calculate. neg_ei, E[SSE], or SSE
        rng: np.random.Generator
            The random number generator used by _objective's tie-breaking draw. Passed
            explicitly rather than stored on this class (see module docstring, risk #2).
        get_y: bool, default False
            Whether to return the y values of the optimized theta
        with_noise: bool, default False
            Whether to return the y values with noise

        Returns
        --------
        best_val: float
            The optimized value of the function
        best_theta: np.ndarray
            The parameter set corresponding to best_val

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        # These three track the best candidate seen so far across the scipy restarts
        # below, always updated together (see _objective) -- reset per call since a new
        # call means a new opt_obj/best_error_metrics, so any prior winner is stale.
        self.min_obj_class = None
        self.min_obj_val = None
        self.min_obj_prediction = None

        assert isinstance(opt_obj, str), "opt_obj must be string!"
        assert opt_obj in [
            "neg_ei",
            "E_sse",
            "sse",
        ], "opt_obj must be 'neg_ei', or 'sse'!"

        # Note use > because index 0 counts as 1 reoptimization
        if self.cs_params.reoptimize_obj > 50:
            warnings.warn("The objective will be reoptimized more than 50 times!")

        # Calc best error
        be_data, best_error_metrics = self.get_best_error()

        # Find bounds and arguments for function
        bnds = (
            self.simulator.bounds_theta_reg.T
        )  # Transpose bounds to work with scipy.optimize
        # Need to account for normalization here (make bounds array of [0,1]^dim_theta)

        ## Loop over each validation point/ a certain number of validation point thetas
        for i in range(self.cs_params.reoptimize_obj + 1):
            # Choose a random index of theta to start with
            theta_guess = self.opt_start_pts[i].flatten()

            # Initialize L-BFGS-B as default optimization method
            obj_opt_method = "L-BFGS-B"

            try:
                # Call scipy method to optimize EI given theta
                # Using L-BFGS-B instead of BFGS because it allowd for bounds
                best_result = optimize.minimize(
                    self._objective,
                    theta_guess,
                    bounds=bnds,
                    method=obj_opt_method,
                    args=(opt_obj, best_error_metrics, rng),
                )
            except ValueError:
                # If the intialized theta causes scipy.optimize to choose nan values, skip it
                pass

        best_val = self.min_obj_val
        best_class = self.min_obj_class
        best_prediction = self.min_obj_prediction

        best_class_simple = self.create_data_fn(
            self.min_obj_class.theta_vals[0], get_y=get_y, with_noise=with_noise
        )
        if get_y:
            best_class.y_vals = best_class_simple.y_vals

        return best_val, best_class, best_prediction

    def _objective(self, theta, opt_obj, best_error_metrics, rng):
        """
        Calculates either -ei, sse objective, or E[SSE] at a candidate parameter set value

        Parameters
        -----------
        theta: np.ndarray
            Array of theta values to optimize
        opt_obj: str
            Which objective to calculate. 'neg_ei', 'E_sse', or 'sse'
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error, best error parameter set, and best_error_x values of the method. Hint: Use self.get_best_error()
        rng: np.random.Generator
            The random number generator used for the SSE tie-breaking draw

        Returns
        --------
        obj: float, the value of the specified objective function for the given candidate parameter set

        Notes
        -----
        If there are nan values in theta, the objective function is set to 1 for neg_ei and self.sse_penalty for sse and E_sse

        """
        # Check if there are nan values in theta
        if np.isnan(theta).any():
            # If there are nan values, set neg ei to 1 (ei = -1)
            if opt_obj == "neg_ei":
                obj = 1
            # Set sse and lcb to self.sse_penalty
            else:
                obj = self.sse_penalty

        # If not, continue the algorithm normally
        else:
            candidate = CandidateSet(
                None,
                self.exp_data.x_vals,
                bounds_theta=self.simulator.bounds_theta_reg,
                bounds_x=self.simulator.bounds_x,
                sep_fact=self.cs_params.sep_fact,
            )

            # Create feature data for candidate point
            if self.method.is_emulator == False:
                candidate_theta_vals = theta.reshape(1, -1)
            else:
                candidate_theta_vals = np.repeat(
                    theta.reshape(1, -1), self.exp_data.n_x, axis=0
                )

            candidate.theta_vals = candidate_theta_vals
            self.gp_emulator.cand_data = candidate

            # Set candidate point feature data
            self.gp_emulator.feature_cand_data = self.gp_emulator.featurize_data(
                self.gp_emulator.cand_data
            )

            # Evaluate GP mean/ stdev at theta
            pred = self.gp_emulator.predict(target="cand")
            cand_mean, cand_var = pred

            # Evaluate SSE & SSE stdev at theta. Reuses `pred` instead of re-reading
            # data.gp_mean/data.gp_covar off gp_emulator.cand_data.
            if self.method.is_emulator == False:
                # For Type 1 GP, the sse and sse_var are directly inferred from the gp_mean and gp_var
                cand_sse_mean, cand_sse_var = self.gp_emulator.predict_sse(target="cand", prediction=pred)
            else:
                # For Type 2 GP, the sse and sse_var are calculated from the gp_mean, gp_var, and experimental data
                cand_sse_mean, cand_sse_var = self.gp_emulator.predict_sse(
                    target="cand", method=self.method, exp_data=self.exp_data, prediction=pred
                )

            # Calculate objective fxn
            if opt_obj == "sse":
                # Objective to minimize is log(sse) for LOG_CONVENTIONAL (B1), sse otherwise
                obj = cand_sse_mean
            elif opt_obj == "E_sse":
                # Objective to minimize is E[sse] for EXPECTED_SSE (A3)
                obj = cand_sse_mean + np.sum(cand_sse_var)
            else:
                # Otherwise objective is ei. Reuses `pred` instead of re-reading
                # data.gp_mean/data.gp_covar off gp_emulator.cand_data.
                if self.method.is_emulator == False:
                    ei_output = self.gp_emulator.expected_improvement(
                        target="cand", exp_data=self.exp_data, ep_bias=self.ep_bias,
                        best_error_metrics=best_error_metrics, gp_prediction=pred
                    )
                else:
                    ei_output = self.gp_emulator.expected_improvement(
                        target="cand",
                        exp_data=self.exp_data,
                        ep_bias=self.ep_bias,
                        best_error_metrics=best_error_metrics,
                        method=self.method,
                        sg_mc_samples=self.sg_mc_samples,
                        gp_prediction=pred,
                    )
                obj = -1 * ei_output[0]

            # min_obj_val is the value reported back to callers, which must be the raw
            # (unnegated) EI for neg_ei -- `obj` itself is the sign-flipped value scipy is
            # minimizing, not a usable EI. For sse/E_sse, `obj` IS the reported value, so
            # min_obj_val is set from `obj` directly at the bottom of this loop instead.
            cand_acq_val = ei_output[0] if opt_obj == "neg_ei" else None

            set_acq_val = True

            # Track the best candidate seen across all restarts (lower `obj` always wins,
            # whether `obj` is sse/E_sse directly or the negated EI). Exact ties are
            # broken differently by objective: plain "sse" breaks ties with a random draw
            # (sse is deterministic given theta, so a tie means truly equivalent
            # candidates); "E_sse"/"neg_ei" ties instead prefer whichever candidate sits
            # farther from the existing training data, biasing acquisition toward
            # exploration when the acquisition surface is flat.
            # Save candidate class if there is no current value
            if self.min_obj_class == None:
                self.min_obj_class = self.gp_emulator.cand_data
                self.min_obj_val = cand_acq_val
                self.min_obj_prediction = pred
            # The sse/lcb objective is smaller than what we have so far
            elif self.min_obj_val > obj and opt_obj != "neg_ei":
                self.min_obj_class = self.gp_emulator.cand_data
                self.min_obj_val = cand_acq_val
                self.min_obj_prediction = pred
            # The ei objective is larger than what we have so far
            elif self.min_obj_val * -1 > obj and opt_obj == "neg_ei":
                self.min_obj_class = self.gp_emulator.cand_data
                self.min_obj_val = cand_acq_val
                self.min_obj_prediction = pred
            # For SSE, if the objective is the same, randomly choose between the two (since sse is an objective fxn)
            elif (
                np.isclose(self.min_obj_val, obj, rtol=1e-7)
                and opt_obj == "sse"
            ):
                # random_number = rng.randint(0, 1)
                random_number = rng.integers(0,1)
                if random_number > 0:
                    self.min_obj_class = self.gp_emulator.cand_data
                    self.min_obj_val = cand_acq_val
                    self.min_obj_prediction = pred
                else:
                    set_acq_val = False
            # For EI/E_sse (acquisition fxns) switch to the value farthest from any training data
            elif np.isclose(self.min_obj_val, obj, rtol=1e-7):
                # Get the distance between the candidate and the current min_obj_class value and the training data
                dist_old = (
                    distance.cdist(
                        self.gp_emulator.train_data.theta_vals,
                        self.min_obj_class.theta_vals[0, :].reshape(1, -1),
                        metric="euclidean",
                    )
                    .ravel()
                    .max()
                )
                dist_new = (
                    distance.cdist(
                        self.gp_emulator.train_data.theta_vals,
                        self.gp_emulator.cand_data.theta_vals[0, :].reshape(1, -1),
                        metric="euclidean",
                    )
                    .ravel()
                    .max()
                )
                # If the distance of the new point is larger or equal to the old point, keep the new point
                if dist_new >= dist_old:
                    self.min_obj_class = self.gp_emulator.cand_data
                    self.min_obj_val = cand_acq_val
                    self.min_obj_prediction = pred
                else:
                    set_acq_val = False
            else:
                set_acq_val = False

            if set_acq_val and opt_obj != "neg_ei":
                self.min_obj_val = obj

        return obj
