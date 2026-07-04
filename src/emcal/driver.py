"""GPBODriver: orchestrates the Bayesian-optimization loop (data prep, GP fitting,
acquisition optimization, termination, and results assembly).
"""
import numpy as np
import pandas as pd
import scipy.optimize as optimize
import scipy.spatial.distance as distance
import copy
import warnings
import time
import pickle
import gzip
from itertools import combinations
from .enums import MethodName, GenMethod
from .methods import GPBOMethod
from .config import BOConfig
from .simulator import Simulator
from .data import Data, CandidateSet
from .exploration import ExplorationBias
from .emulators import GPEmulator, ObjectiveGP, EmulatorGP, build_gp_emulator
from .results import BOResults, build_iteration_row, ITERATION_COLUMNS


class GPBODriver:
    """
    The base class for running the GPBO Workflow

    Methods
    --------------
    __init__
    __gen_emulator()
    __get_best_error()
    __make_starting_opt_pts(best_error_metrics)
    __gen_start_pts_mc_sparse(best_error_metrics)
    __gen_start_pts_not_mc_sparse()
    __opt_with_scipy(opt_obj)
    __scipy_fxn(theta, opt_obj, best_error_metrics, beta)
    create_heat_map_param_data(n_points_set)
    __augment_train_data(theta_best_data)
    create_data_instance_from_theta(theta_array)
    __run_bo_iter(gp_model, iteration)
    __run_bo_to_term(gp_model)
    __run_bo_workflow()
    run()
    """

    # Class variables and attributes

    def __init__(
        self,
        cs_params,
        method,
        simulator,
        exp_data,
        sim_data,
        sim_sse_data,
        val_data,
        val_sse_data,
        gp_emulator,
        ep_bias,
        gen_meth_theta,
        backend=None,
    ):
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
            Experimental data containing at least exp_data.theta_vals, exp_data.x_vals, and exp_data.y_vals
        sim_data: Data
            Simulated data containing at least sim_data.theta_vals, sim_data.x_vals, and sim_data.y_vals
        sim_sse_data: Data
            Simulated objective data containing at least sim_sse_data.theta_vals, sim_sse_data.x_vals, and sim_sse_data.y_vals
        val_data: Data or None
            Validation data containing at least val_data.theta_vals, val_data.x_vals, and val_data.y_vals
        val_sse_data: Data or None
            Validation data containing at least val_sse_data.theta_vals, val_sse_data.x_vals, and val_sse_data.y_vals
        gp_emulator: GPEmulator
            Class containing gp_emulator data (set after training)
        ep_bias: ExplorationBias class
            Class containing exploration parameter info
        gen_meth_theta: GenMethod or None
            The method by which simulation data is generated. For heat map making
        backend: GPBackend or None, default None
            The GP backend passed through to __gen_emulator's build_gp_emulator call. If
            None, resolved via get_backend (gpflow by default) -- production behavior is
            unchanged. Tests inject a fake backend here.

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert isinstance(
            cs_params, BOConfig
        ), "cs_params must be instance of BOConfig"
        assert isinstance(
            method, GPBOMethod
        ), "method must be instance of GPBOMethod"
        assert isinstance(
            simulator, Simulator
        ), "simulator must be instance of Simulator"
        assert isinstance(exp_data, Data), "exp_data must be instance of Data"
        assert isinstance(sim_data, Data), "sim_data must be instance of Data"
        assert isinstance(sim_sse_data, Data), "sim_sse_data must be instance of Data"
        assert (
            isinstance(val_data, Data) or val_data is None
        ), "val_data must be instance of Data or None"
        assert (
            isinstance(val_sse_data, Data) or val_sse_data is None
        ), "val_sse_data must be instance of Data or None"
        assert (
            isinstance(gp_emulator, (ObjectiveGP, EmulatorGP))
            or gp_emulator is None
        ), "gp_emulator must be instance of ObjectiveGP, EmulatorGP, or None"
        assert isinstance(
            ep_bias, ExplorationBias
        ), "ep_bias must be instance of ExplorationBias"
        assert isinstance(
            gen_meth_theta, GenMethod
        ), "gen_meth_theta must be instance of GenMethod"

        # Constructor method
        self.cs_params = cs_params
        self.method = method
        self.simulator = simulator
        self.exp_data = exp_data
        self.sim_data = sim_data
        self.sim_sse_data = sim_sse_data
        self.val_data = val_data
        self.val_sse_data = val_sse_data
        self.gp_emulator = gp_emulator
        self.ep_bias = ep_bias
        self.gen_meth_theta = gen_meth_theta
        self._backend = backend
        self.bo_iter_term_frac = 0.3  # The fraction of iterations after which to terminate bo if no sse improvement is made
        self.sse_penalty = 1e7  # The penalty the __scipy_opt function gets for choosing nan theta values
        self.sg_mc_samples = 2000  # This can be changed at will
        # This object is used for optimization
        self.__min_obj_class = None
        # Companions to __min_obj_class, set together with it: the objective/acq value and the
        # GPPrediction of whichever candidate is currently winning. Read instead of
        # __min_obj_class.acq / re-deriving a prediction from __min_obj_class's Data fields.
        self.__min_obj_val = None
        self.__min_obj_prediction = None
        # self.reset_rng()

    def __make_BO_results_temp(self, results_df, why_term, max_ei_details_df, list_gp_emulator_class):
        "Makes BO results from minimum data"

        # Set results for all compiled iterations for that run
        bo_results_res = BOResults(
            None, None, self.exp_data, None, results_df, None, why_term, None
        )

        bo_results_GPs = BOResults(
            None,
            None,
            None,
            list_gp_emulator_class,
            None,
            max_ei_details_df,
            None,
            None,
        )
        return bo_results_res, bo_results_GPs


    def __gen_emulator(self):
        """
        Sets GP Emulator class with training data and validation data based on the method class instance

        Returns
        --------
        gp_emulator: GPEmulator
            Class for the GP emulator
        """
        # Build the right emulator (ObjectiveGP / EmulatorGP) via the shared factory.
        return build_gp_emulator(
            self.method,
            self.sim_data,
            self.sim_sse_data,
            self.val_data,
            self.val_sse_data,
            self.cs_params.kernel,
            self.cs_params.lenscl,
            self.cs_params.outputscl,
            self.cs_params.retrain_GP,
            self.cs_params.seed,
            self.cs_params.normalize,
            self.simulator.noise_std,
            self.exp_data.n_x,
            backend=self._backend,
        )

    def __get_best_error(self):
        """
        Helper function to calculate the best error and squared error calculations over x given the method.

        Returns
        -------
        be_data: Data
            Contains best_error as an instance of the data class
        be_metrics: tuple(float, np.ndarray, np.ndarray)
            The min_SSE, param at min_SSE, and squared residuals
        """

        if self.method.is_emulator == False:
            # Type 1 best error is inferred from training data
            best_error, be_theta, train_idx = self.gp_emulator.calc_best_error()
            best_errors_x = None
            be_data = self.create_data_instance_from_theta(
                be_theta.flatten(), get_y=False
            )
            be_data.y_vals = np.atleast_1d(
                self.gp_emulator.train_data.y_vals[train_idx]
            )
        else:
            # Type 2 best error must be calculated given the experimental data
            best_error, be_theta, best_errors_x, train_idx = (
                self.gp_emulator.calc_best_error(self.method, self.exp_data)
            )
            be_data = self.create_data_instance_from_theta(
                be_theta.flatten(), get_y=False
            )
            be_data.y_vals = self.gp_emulator.train_data.y_vals[
                train_idx[0] : train_idx[1]
            ]

        be_metrics = best_error, be_theta, best_errors_x

        return be_data, be_metrics

    def __make_starting_opt_pts(self, best_error_metrics, rng_seed):
        """
        Makes starting point for optimization with scipy

        Parameters
        -----------
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error, best error parameter set, and best_error_x values of the method. Hint: Use self.__get_best_error()

        Returns
        --------
        starting_pts: np.ndarray
            Array of parameter set initializations for self.__opt_with_scipy
        """
        # Note: Could make this generate 2 sets of starting points based on whether you want to optimize sse or ei
        # For sparse grid and mc methods
        if self.method.uses_sparse_grid == True or self.method.uses_monte_carlo == True:
            starting_pts = self.__gen_start_pts_mc_sparse(best_error_metrics, rng_seed)
        else:
            starting_pts = self.__gen_start_pts_not_mc_sparse(rng_seed)

        return starting_pts

    def __gen_start_pts_mc_sparse(self, best_error_metrics, rng_seed):
        """
        Makes starting point for optimization with scipy if using sparse grid or Monte Carlo methods

        Parameters
        -----------
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error, best error parameter set, and best_error_x values of the method

        Returns
        --------
        starting_pts: np.ndarray
            Array of parameter set initializations for self.__opt_with_scipy
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

        # Choose top retrain_GP points as starting points
        starting_pts = all_pts[: self.cs_params.reoptimize_obj + 1]

        return starting_pts

    def __gen_start_pts_not_mc_sparse(self, rng_seed):
        """
        Makes starting point for optimization with scipy if not using sparse grid or Monte Carlo methods

        Returns
        --------
        starting_pts: np.ndarray
            Array of parameter set initializations for self.__opt_with_scipy
        """
        # Create starting points equal to number of retrain_GP
        starting_pts = self.simulator.generate_parameter_samples(
            self.cs_params.reoptimize_obj + 1, rng_seed
        )

        return starting_pts

    def __opt_with_scipy(self, opt_obj, get_y = False, w_noise = False):
        """
        Optimizes a function with scipy.optimize

        Parameters
        ----------
        opt_obj: str
            Which objective to calculate. neg_ei, E[SSE], or SSE
        get_y: bool, default False
            Whether to return the y values of the optimized theta
        w_noise: bool, default False
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
        self.__min_obj_class = None
        self.__min_obj_val = None
        self.__min_obj_prediction = None

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
        be_data, best_error_metrics = self.__get_best_error()

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
                    self.__scipy_fxn,
                    theta_guess,
                    bounds=bnds,
                    method=obj_opt_method,
                    args=(opt_obj, best_error_metrics),
                )
            except ValueError:
                # If the intialized theta causes scipy.optimize to choose nan values, skip it
                pass

        best_val = self.__min_obj_val
        best_class = self.__min_obj_class
        best_prediction = self.__min_obj_prediction

        best_class_simple = self.create_data_instance_from_theta(
            self.__min_obj_class.theta_vals[0], get_y=get_y, w_noise=w_noise
        )
        if get_y:
            best_class.y_vals = best_class_simple.y_vals

        return best_val, best_class, best_prediction

    def __scipy_fxn(self, theta, opt_obj, best_error_metrics):
        """
        Calculates either -ei, sse objective, or E[SSE] at a candidate parameter set value

        Parameters
        -----------
        theta: np.ndarray
            Array of theta values to optimize
        opt_obj: str
            Which objective to calculate. 'neg_ei', 'E_sse', or 'sse'
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error, best error parameter set, and best_error_x values of the method. Hint: Use self.__get_best_error()

        Returns
        --------
        obj: float, the value of the specified objective function for the given candidate parameter set

        Notes
        -----
        If there are nan values in theta, the objective function is set to 1 for neg_ei and self.sse_penalty for sse and E_sse

        """
        rng = self.rng_set
        # Set seed
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

            cand_acq_val = ei_output[0] if opt_obj == "neg_ei" else None

            set_acq_val = True

            # Save candidate class if there is no current value
            if self.__min_obj_class == None:
                self.__min_obj_class = self.gp_emulator.cand_data
                self.__min_obj_val = cand_acq_val
                self.__min_obj_prediction = pred
            # The sse/lcb objective is smaller than what we have so far
            elif self.__min_obj_val > obj and opt_obj != "neg_ei":
                self.__min_obj_class = self.gp_emulator.cand_data
                self.__min_obj_val = cand_acq_val
                self.__min_obj_prediction = pred
            # The ei objective is larger than what we have so far
            elif self.__min_obj_val * -1 > obj and opt_obj == "neg_ei":
                self.__min_obj_class = self.gp_emulator.cand_data
                self.__min_obj_val = cand_acq_val
                self.__min_obj_prediction = pred
            # For SSE, if the objective is the same, randomly choose between the two (since sse is an objective fxn)
            elif (
                np.isclose(self.__min_obj_val, obj, rtol=1e-7)
                and opt_obj == "sse"
            ):
                # random_number = rng.randint(0, 1)
                random_number = rng.integers(0,1)
                if random_number > 0:
                    self.__min_obj_class = self.gp_emulator.cand_data
                    self.__min_obj_val = cand_acq_val
                    self.__min_obj_prediction = pred
                else:
                    set_acq_val = False
            # For EI/E_sse (acquisition fxns) switch to the value farthest from any training data
            elif np.isclose(self.__min_obj_val, obj, rtol=1e-7):
                # Get the distance between the candidate and the current min_obj_class value and the training data
                dist_old = (
                    distance.cdist(
                        self.gp_emulator.train_data.theta_vals,
                        self.__min_obj_class.theta_vals[0, :].reshape(1, -1),
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
                    self.__min_obj_class = self.gp_emulator.cand_data
                    self.__min_obj_val = cand_acq_val
                    self.__min_obj_prediction = pred
                else:
                    set_acq_val = False
            else:
                set_acq_val = False

            if set_acq_val and opt_obj != "neg_ei":
                self.__min_obj_val = obj

        return obj

    def create_heat_map_param_data(self, n_points_set=None):
        """
        Creates parameter sets that can be used to generate heat maps of data at any given iteration

        Parameters
        -----------
        n_points_set: int or None, default None
            The number of points to use per axis for creating heat maps. If None, the number of unique simulation points is used

        Returns
        --------
        heat_map_data_dict: dict
            Heat map data for each set of 2 parameters indexed by parameter name tuple ("param_1,param_2")
        """
        assert isinstance(
            self.gp_emulator, (ObjectiveGP, EmulatorGP)
        ), "self.gp_emulator must be instance of ObjectiveGP or EmulatorGP"
        assert isinstance(
            self.gp_emulator.gp_sim_data, Data
        ), "self.gp_emulator.gp_sim_data must be an instance of Data!"
        assert isinstance(
            self.gen_meth_theta, GenMethod
        ), "self.gen_meth_theta must be instance of GenMethod"
        assert isinstance(
            self.exp_data.x_vals, (np.ndarray)
        ), "self.exp_data.x_vals must be np.ndarray"
        assert (
            isinstance(n_points_set, int) or n_points_set is None
        ), "n_points_set must be None or int"

        # Create list of heat map theta data
        heat_map_data_dict = {}

        # Create a linspace for the number of dimensions and define number of points
        dim_list = np.linspace(
            0, self.simulator.dim_theta - 1, self.simulator.dim_theta
        )
        # Create a list of all combinations (without repeats e.g no (1,1), (2,2)) of dimensions of theta
        mesh_combos = np.array(list(combinations(dim_list, 2)), dtype=int)

        # Set x_vals
        norm_x_vals = self.exp_data.x_vals
        num_x = self.exp_data.n_x

        # If no number of points is set, use the length of the unique simulation thetas
        if n_points_set == None:
            # Use number of training theta for number of theta points
            n_thetas_points = len(self.gp_emulator.gp_sim_data.get_unique_theta())
            # Initialze meshgrid-like set of theta values at their true values
            # If points were generated with an LHS, the number of points per parameter is n_thetas_points for the meshgrid
            if self.gen_meth_theta.value == 1:
                n_points = n_thetas_points
            else:
                # For a meshgrid, the number of theta values/ parameter is n_thetas_points for the meshgrid ^(1/theta_dim)
                n_points = int((n_thetas_points) ** (1 / self.simulator.dim_theta))
        else:
            n_points = n_points_set

        # Ensure we will never generate more than 5000 pts per heat map
        # if self.method.is_emulator == True:
        if num_x * n_points**2 >= 5000:
            n_points = int(np.sqrt(5000 / (num_x)))

        # Meshgrid set always defined by n_points**2
        # Set thetas for meshgrid. Never use more than 10000 points
        theta_set = np.tile(np.array(self.simulator.theta_true), (n_points**2, 1))

        # Infer how many times to repeat theta and x values given that heat maps are meshgrid form by definition
        # The meshgrid of parameter values created below is symmetric, therefore, x is repeated by n_points**2 for a 2D meshgrid
        repeat_x = n_points**2  # Square because only 2 values at a time change
        x_vals = np.vstack([norm_x_vals] * repeat_x)
        repeat_theta = self.exp_data.n_x

        # Loop over all possible theta combinations of 2
        for i in range(len(mesh_combos)):
            # Create a copy of the true values to change the mehsgrid valus on
            theta_set_copy = np.copy(theta_set)
            # Set the indeces of theta_set for evaluation as each row of mesh_combos
            idcs = mesh_combos[i]
            # define name of parameter set as tuple ("param_1,param_2")
            data_set_name = (
                self.simulator.theta_true_names[idcs[0]],
                self.simulator.theta_true_names[idcs[1]],
            )

            # Create a meshgrid of values of the 2 selected values of theta and reshape to the correct shape
            # Assume that theta1 and theta2 have equal number of points on the meshgrid
            theta1 = np.linspace(
                self.simulator.bounds_theta_reg[0][idcs[0]],
                self.simulator.bounds_theta_reg[1][idcs[0]],
                n_points,
            )
            theta2 = np.linspace(
                self.simulator.bounds_theta_reg[0][idcs[1]],
                self.simulator.bounds_theta_reg[1][idcs[1]],
                n_points,
            )
            theta12_mesh = np.array(np.meshgrid(theta1, theta2))
            theta12_vals = np.array(theta12_mesh).T.reshape(-1, 2)

            # Set initial values for evaluation (true values) to meshgrid values
            theta_set_copy[:, idcs] = theta12_vals

            # Put values into instance of data class
            # Create data set based on emulator status
            if self.method.is_emulator == True:
                # Repeat the theta vals for Type 2 methods to ensure that theta and x values are in the correct form for evaluation with gp_emulator.eval_gp_mean_heat_map()
                theta_vals = np.repeat(theta_set_copy, repeat_theta, axis=0)
                data_set = CandidateSet(
                    theta_vals,
                    x_vals,
                    bounds_theta=self.simulator.bounds_theta_reg,
                    bounds_x=self.simulator.bounds_x,
                    sep_fact=self.cs_params.sep_fact,
                )
            else:
                data_set = CandidateSet(
                    theta_set_copy,
                    norm_x_vals,
                    bounds_theta=self.simulator.bounds_theta_reg,
                    bounds_x=self.simulator.bounds_x,
                    sep_fact=self.cs_params.sep_fact,
                )

            # Append data set to dictionary with name
            heat_map_data_dict[data_set_name] = data_set

        return heat_map_data_dict

    def __augment_train_data(self, theta_best_data):
        """
        Augments training data given a new data point

        Parameters
        ----------
        theta_best_data: Data
            The parameter set data associated with the optimal acquisition function value
        """
        # Augment training theta, x, and y/sse data
        self.gp_emulator.append_training_point(theta_best_data)

    def create_data_instance_from_theta(self, theta_array, get_y=True, w_noise = False):
        """
        Creates instance of Data from an np.ndarray parameter set

        Parameters
        ----------
        theta_array: np.ndarray
            Array of parameter values to turn into an instance of Data
        get_y: bool, default True
            Whether to calculate y values for theta_array
        w_noise: bool, default False
            Whether to add noise to the y values

        Returns
        --------
        theta_arr_data: Data
            Data class instance for the theta_array

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        rng = self.rng_set

        assert isinstance(theta_array, np.ndarray), "theta_array must be np.ndarray"
        assert len(theta_array.shape) == 1, "theta_array must be 1D"
        assert isinstance(
            self.exp_data.x_vals, (np.ndarray)
        ), "self.exp_data.x_vals must be np.ndarray"

        # Repeat the theta best array once for each x value
        # Need to repeat theta_best such that it can be evaluated at every x value in exp_data using simulator.evaluate_model
        theta_arr_repeated = np.repeat(
            theta_array.reshape(1, -1), self.exp_data.n_x, axis=0
        )
        # Add instance of Data class to theta_best
        theta_arr_data = CandidateSet(
            theta_arr_repeated,
            self.exp_data.x_vals,
            bounds_theta=self.simulator.bounds_theta_reg,
            bounds_x=self.simulator.bounds_x,
            sep_fact=self.cs_params.sep_fact,
        )
        if get_y:
            if w_noise:
                # Calculate y values and sse for theta_best with noise
                theta_arr_data.y_vals = self.simulator.evaluate_model(
                    theta_arr_data, self.simulator.noise_mean, self.simulator.noise_std, rng
                )
            else:
                # Calculate y values and sse for theta_best without noise
                theta_arr_data.y_vals = self.simulator.evaluate_model(
                    theta_arr_data, self.simulator.noise_mean, 0, rng
                )

        # Set the best data to be in sse form if using a type 1 GP
        if self.method.is_emulator == False:
            theta_arr_data = self.simulator.to_sse_data(
                self.method,
                theta_arr_data,
                self.exp_data,
                self.cs_params.sep_fact,
                not get_y,
            )

        return theta_arr_data

    def __run_bo_iter(self, iteration):
        """
        Runs a single GPBO iteration

        Parameters
        ----------
        iteration: int, The iteration of BO in progress

        Returns
        --------
        iter_df: pd.DataFrame
            Dataframe containing the results from the GPBO Workflow for iteration
        iter_max_ei_terms: pd.DataFrame or None
            Contains ei calculation terms for max ei parameter set if self.cs_params.save_data
        gp_emulator_curr: GPEmulator
            The class used for this iteration of the GPBO workflow
        """
        # Start timer
        # Initialize iter_max_ei df to None
        iter_max_ei_terms = None
        
        #Initialize the iterations seed with start_seed as a backup
        iter_seed = self.simulator.start_seed

        #Generate a random number for the seed to generate initial LHS samples with that is not the same as the sim or val seeds 
        if self.cs_params.seed is not None:
            for i in range(10):
                seed_init = self.rng_set.integers(1, 1e8)
                if seed_init not in [self.simulator.sim_seed, self.simulator.val_seed]:
                    iter_seed = seed_init
                    break
        else:
            iter_seed = None
        
        time_start = time.time()

        # Train GP model (this step updates the model to a trained model)
        self.gp_emulator.fit()

        # Calcuate best error
        best_err_data, best_error_metrics = self.__get_best_error()

        # Add not log best error to ep_bias
        if iteration == 0 or self.ep_bias.ep_enum.value == 4:
            # Since best error is squared when used in Jasrasaria calculations, the value will always be >=0
            self.ep_bias.best_error = best_error_metrics[0]

        # Calculate mean of var for validation set if using Jasrasaria heuristic
        if self.ep_bias.ep_enum.value == 4:
            # Calculate average gp mean and variance of the validation set
            val_pred = self.gp_emulator.predict(target="val")
            val_gp_mean, val_gp_var = val_pred
            # For emulator methods, the mean of the variance should come from the sse variance
            if self.method.is_emulator == True:
                # Redefine gp_mean and gp_var to be the mean and variane of the sse. Reuses
                # `val_pred` instead of re-reading data.gp_mean/data.gp_covar off gp_val_data.
                val_gp_mean, val_gp_var = self.gp_emulator.predict_sse(
                    target="val", method=self.method, exp_data=self.exp_data, prediction=val_pred
                )

            # Check for ln(sse) values
            # For 1B, propogate errors associated with an unlogged sse value
            val_gp_var = val_gp_var * np.exp(val_gp_mean) ** 2

            # Set mean of sse variance
            mean_of_var = np.average(val_gp_var)
            self.ep_bias.mean_of_var = mean_of_var

        # Set initial exploration bias and bo_iter
        if self.ep_bias.ep_enum.value == 2:
            self.ep_bias.bo_iter = iteration

        # Calculate new ep. Note. It is extemely important to do this AFTER setting the ep_max
        self.ep_bias.update()

        # Set Optimization starting points for this iteration
        self.opt_start_pts = self.__make_starting_opt_pts(best_error_metrics, iter_seed)

        # Call optimize E[SSE] or log(E[SSE]) objective function
        # Note if we didn't want actual sse values, we would have to set get_y_sse = False
        min_sse, min_theta_data, min_sse_prediction = self.__opt_with_scipy("sse", get_y = self.cs_params.get_y_sse, w_noise = self.cs_params.w_noise)

        # Call optimize EI acquistion fxn (If not using E[SSE])
        if self.method.method_name.value != 7:
            opt_acq, acq_theta_data, best_prediction = self.__opt_with_scipy("neg_ei", get_y = True, w_noise = self.cs_params.w_noise)
            if self.method.is_emulator == True:
                ei_args = dict(
                    data=acq_theta_data,
                    exp_data=self.exp_data,
                    ep_bias=self.ep_bias,
                    best_error_metrics=best_error_metrics,
                    method=self.method,
                    sg_mc_samples=self.sg_mc_samples,
                    gp_prediction=best_prediction,
                )
            else:
                ei_args = dict(
                    data=acq_theta_data,
                    exp_data=self.exp_data,
                    ep_bias=self.ep_bias,
                    best_error_metrics=best_error_metrics,
                    gp_prediction=best_prediction,
                )
        else:
            opt_acq, acq_theta_data, best_prediction = self.__opt_with_scipy("E_sse", get_y = True, w_noise = self.cs_params.w_noise)

        # If type 2, turn it into sse_data
        # Set the best data to be in sse form if using a type 2 GP and find the min sse
        if self.method.is_emulator == True:
            # Evaluate SSE & SSE stdev at max ei theta
            min_sse_theta_data = self.simulator.to_sse_data(
                self.method,
                min_theta_data,
                self.exp_data,
                self.cs_params.sep_fact,
                False,
            )
            acq_sse_theta_data = self.simulator.to_sse_data(
                self.method,
                min_theta_data,
                self.exp_data,
                self.cs_params.sep_fact,
                False,
            )

        # Otherwise the sse data is the original (scaled) data
        else:
            # Evaluate SSE & SSE stdev at max ei theta
            min_sse_theta_data = min_theta_data
            acq_sse_theta_data = acq_theta_data

        # Evaluate max EI terms at theta
        if self.cs_params.save_data and not self.method.method_name.value == 7:
            ei_max, iter_max_ei_terms = self.gp_emulator.expected_improvement(**ei_args)

        # Turn min_sse_sim value into a float (this makes analyzing data from csvs and dataframes easier)
        min_sse_gp = float(np.asarray(min_sse).item())
        opt_acq_sim = float(np.asarray(acq_sse_theta_data.y_vals).item())

        # calculate improvement if using Boyle's method to update the exploration bias
        # Improvement is true if the min sim sse found is lower than (not log) best error, otherwise it's false
        if min_sse_gp < best_error_metrics[0]:
            improvement = True
        else:
            improvement = False
        if self.ep_bias.ep_enum.value == 3:
            # Set ep improvement
            self.ep_bias.improvement = improvement

        # Create a copy of the GP Emulator Class for this iteration
        gp_emulator_curr = copy.deepcopy(self.gp_emulator)

        # Call __augment_train_data to append training data
        self.__augment_train_data(acq_theta_data)


        # Calc time/ iter
        time_end = time.time()
        time_per_iter = time_end - time_start

        # Create Results Pandas DataFrame for 1 iter
        num_exp_x = self.exp_data.n_x
        # Return SSE and not log(SSE) for 'Min Obj', 'sse_actual', 'sse_gp' when calculating MSE
        if self.cs_params.get_y_sse:
            min_sse_sim = float(np.asarray(min_sse_theta_data.y_vals).item())
        else:
            min_sse_sim = None

        iter_df = build_iteration_row(
            best_error_metrics[0],
            self.ep_bias.ep_curr,
            acq_theta_data.theta_vals[0],
            opt_acq,
            opt_acq_sim,
            min_sse_theta_data.theta_vals[0],
            min_sse_gp,
            min_sse_sim,
            time_per_iter,
            self.method.log_scaled,
            num_exp_x,
        )

        return iter_df, iter_max_ei_terms, gp_emulator_curr

    def __run_bo_to_term(self, run_num, job = None):
        """
        Runs GPBO to termination

        Params:
        -------
        gp_model: gpflow.models.GPR, GP emulator for workflow

        Returns
        --------
        iter_df: pd.DataFrame
            Dataframe containing the results from the GPBO Workflow for all iterations
        max_ei_details_df: pd.DataFrame
            Contains ei data for max ei parameter sets for each bo iter if self.cs_params.save_data
        list_gp_emulator_class: list(GPEmulator)
            The classes used for all iterations of the GPBO workflow
        why_term: str
            String containing reasons for bo algorithm termination

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert (
            0 < self.bo_iter_term_frac <= 1
        ), "self.bo_iter_term_frac must be between 0 and 1"
        # Initialize pandas dataframes
        results_df = pd.DataFrame(columns=ITERATION_COLUMNS)
        max_ei_details_df = pd.DataFrame()
        list_gp_emulator_class = []
        # Initialize count
        obj_counter = 0
        self.ep_bias.ep_curr = None

        cond1 = len(self.gpbo_res_GP) > 0
        cond2 = len(self.gpbo_res_GP) == run_num + 1
        cond3 = job is not None

        #Check for files in job for gpbo_res_simple and gpbo_res_GP
        if cond1 and cond2 and cond3:
            results_df = self.gpbo_res_simple[run_num].results_df
            self.ep_bias.ep_curr = self.gpbo_res_simple[run_num].results_df["alpha"].iloc[-1]
            #The obj_counter is set as why_term until termination happens
            obj_counter = self.gpbo_res_simple[run_num].why_term
            list_gp_emulator_class = self.gpbo_res_GP[run_num].list_gp_emulator_class

        #Start at the next iteration after data ends. If no data, start at 0
        iter_start = len(list_gp_emulator_class)

        # Initilize terminate flags
        acq_flag = False
        obj_flag = False
        terminate = False

        # Set why_term strings
        why_terms = ["acq", "obj", "max_budget"]

        # Do Bo iters while stopping criteria is not met
        while terminate == False:
            # Loop over number of max bo iters
            for i in range(iter_start, self.cs_params.bo_iter_tot, 1):
                # Output results of 1 bo iter and the emulator used to get the results
                iter_df, iter_max_ei_terms, gp_emulator_class = self.__run_bo_iter(
                    i
                ) 
                # Add results to dataframe
                results_df = pd.concat(
                    [results_df.astype(iter_df.dtypes), iter_df], ignore_index=True
                )
                if iter_max_ei_terms is not None:
                    max_ei_details_df = pd.concat(
                        [max_ei_details_df, iter_max_ei_terms]
                    )
                # At the first iteration
                if i == 0:
                    # improvement is defined as infinity on 1st iteration (something is always better than nothing)
                    improvement = np.inf
                elif results_df["sse_gp"].iloc[i] < float(
                    results_df["sse_gp"][:-1].min()
                ):
                    # And the improvement is defined as the difference between the last Min Obj Cum. and current Obj Min (unscaled)
                    if not self.method.log_scaled:
                        improvement = (
                            results_df["sse_gp"][:-1].min()
                            - results_df["sse_gp"].iloc[i]
                        )
                    else:
                        improvement = np.exp(
                            results_df["sse_gp"][:-1].min()
                        ) - np.exp(results_df["sse_gp"].iloc[i])
                # Otherwise
                else:
                    # And the improvement is defined as 0, since it must be non-negative
                    improvement = 0

                # Add gp emulator data from that iteration to list (before stopping criteria)
                list_gp_emulator_class.append(gp_emulator_class)
                # Call stopping criteria after 1st iteration and update improvement counter
                # If the improvement is negligible, add to counter
                if improvement < self.cs_params.obj_tol:
                    obj_counter += 1
                # Otherwise reset the counter
                else:
                    obj_counter = 0

                # set flag if opt acq. func val is less than the tolerance 3 times in a row
                if (
                    all(results_df["acq_value"].tail(3) < self.cs_params.acq_tol)
                    and i >= 4
                ):
                    acq_flag = True
                # set flag if small sse progress over 1/3 of total iteration budget
                if (
                    obj_counter
                    >= int(self.cs_params.bo_iter_tot * self.bo_iter_term_frac)
                    and i >= 4
                ):
                    obj_flag = True

                flags = [acq_flag, obj_flag]

                # Terminate if you meet 2 stopping criteria, hit the budget, or obj has not improved after 1/2 of iterations
                if i == self.cs_params.bo_iter_tot - 1:
                    terminate = True
                    why_term = why_terms[-1]
                    break
                elif flags.count(True) >= 2:
                    terminate = True
                    # Pull indecies of list that are true
                    term_flags = [
                        why_terms[index] for index, value in enumerate(flags) if value
                    ]
                    why_term = "-".join(term_flags)
                    break
                elif (
                    obj_counter >= int(self.cs_params.bo_iter_tot * 0.5)
                    and self.cs_params.bo_iter_tot >= 5
                ):
                    terminate = True
                    why_term = why_terms[1]
                    break
                # Continue if no stopping criteria are met
                else:
                    terminate = False
                
                #Save results
                #make a new list of emulator classes to save which includes a copy of the original list + newest GP object
                list_emulator_class_temp = copy.deepcopy(list_gp_emulator_class.copy())
                list_emulator_class_temp[-1] = copy.deepcopy(self.gp_emulator)
                #Make temporary BO results for this iter
                bo_results_res, bo_results_GPs = self.__make_BO_results_temp(results_df, obj_counter, max_ei_details_df, list_emulator_class_temp)
                # Add simulator class and save the rng seeds that are being used
                bo_results_res.simulator_class = copy.deepcopy(self.simulator)
                bo_results_GPs.driver_rng = copy.deepcopy(self.rng_set)
                bo_results_res.sim_rng = copy.deepcopy(self.simulator.rng_set)
                
                #Save results at each iteration so that if the job takes a while it can be continued
                if len(self.gpbo_res_simple) == len(self.gpbo_res_GP) != run_num + 1:
                    self.gpbo_res_simple.append(bo_results_res)
                    self.gpbo_res_GP.append(bo_results_GPs)
                else:
                    self.gpbo_res_simple[run_num] = bo_results_res
                    self.gpbo_res_GP[run_num] = bo_results_GPs

                #Save results
                if job is not None:
                    self.save_results_run(job)
                
        # Reset the index of the pandas df
        results_df = results_df.reset_index()

        # Fill Cumulative value columns based on results
        # Initialize cum columns as the same as the original columns
        results_df.rename(columns={"index": "bo_iter"}, inplace=True)
        results_df["bo_iter"] += 1
        results_df["method"] = self.method.report_name
        results_df["max_evals"] = len(results_df)
        results_df["theta_best_at_acq"] = results_df["theta_at_acq"]
        results_df["theta_best_gp"] = results_df["theta_at_min"]
        results_df["theta_best_actual"] = results_df["theta_at_min"]
        results_df["termination_reason"] = why_term
        results_df["total_time"] = float(results_df["time_per_iter"].sum())

        results_df["best_sse_gp"] = np.minimum.accumulate(results_df["sse_gp"])
        if self.cs_params.get_y_sse == True:
            results_df["best_sse_actual"] = np.minimum.accumulate(results_df["sse_actual"])
        else:
            results_df["best_sse_actual"] = None
        results_df["best_sse_at_acq"] = np.minimum.accumulate(results_df["sse_at_acq"])

        # Add cumulative values to the dataframe
        for i in range(len(results_df)):
            if i > 0:
                if (
                    results_df["best_sse_at_acq"].iloc[i]
                    >= results_df["best_sse_at_acq"].iloc[i - 1]
                ):
                    results_df.at[i, "theta_best_at_acq"] = (
                        results_df["theta_best_at_acq"].iloc[i - 1].copy()
                    )
                #If we are tracking actual values, update as normal, otherwise follow the same trend os the GP SSE
                if (self.cs_params.get_y_sse == True and
                    results_df["best_sse_actual"].iloc[i]
                    >= results_df["best_sse_actual"].iloc[i - 1] 
                ) or (self.cs_params.get_y_sse == False and results_df["best_sse_gp"].iloc[i]
                    >= results_df["best_sse_gp"].iloc[i - 1]):
                    results_df.at[i, "theta_best_actual"] = (
                        results_df["theta_best_actual"].iloc[i - 1].copy()
                    )
                if (
                    results_df["best_sse_gp"].iloc[i]
                    >= results_df["best_sse_gp"].iloc[i - 1]
                ):
                    results_df.at[i, "theta_best_gp"] = (
                        results_df["theta_best_gp"].iloc[i - 1].copy()
                    )

        # Create df for ei and add those results here
        if iter_max_ei_terms is not None:
            max_ei_details_df.columns = iter_max_ei_terms.columns.tolist()
            max_ei_details_df = max_ei_details_df.reset_index(drop=True)


        return results_df, max_ei_details_df, list_gp_emulator_class, why_term

    def __run_bo_workflow(self, run_num, job = None):
        """
        Runs a GPBO method through all bo iterations and reports the data for that run of the method

        Returns
        --------
        bo_results_res: BOResults
            Includes table of results, exp_Data and why term for the GPBO workflow
        bo_results_GPs: BOResults
            Includes the GP emulator classes used and max ei details for each iteration of the BO workflow

        Notes
        ------
        Two instances of BOResults are used since opening the GP files is often tedious and we may not need to open them to analyze the results
        """
        
        #If a results object for this run exists, load it
        cond1 = len(self.gpbo_res_GP) > 0
        cond2 = len(self.gpbo_res_GP) == run_num + 1
        cond3 = job is not None

        #If results exist for this run and is being saved, use the emulator class from the last iteration
        if cond1 and cond2 and cond3:
            self.gp_emulator = self.gpbo_res_GP[run_num].list_gp_emulator_class[-1]
            self.rng_set = self.gpbo_res_GP[run_num].driver_rng
            self.simulator.rng_set = self.gpbo_res_simple[run_num].sim_rng
        #If results do not exist for this run, initialize the emulator class
        else:
            #Reset driver rng at each run to update seed for driver class
            self.reset_rng()
            # Initialize gp_emualtor class
            gp_emulator = self.__gen_emulator()
            self.gp_emulator = gp_emulator

            # Choose training data
            train_data, test_data = self.gp_emulator.split_train_test(
                self.cs_params.sep_fact, self.cs_params.seed
            )

        ##Call bo_iter
        results_df, max_ei_details_df, list_gp_emulator_class, why_term = (
            self.__run_bo_to_term(run_num, job)
        )

        # # Set results for all compiled iterations for that run
        bo_results_res, bo_results_GPs = self.__make_BO_results_temp(results_df, why_term, max_ei_details_df, list_gp_emulator_class)

        # return bo_results_res, bo_results_GPs
        return bo_results_res, bo_results_GPs
    
    def reset_rng(self):
        """
        Resets the random number generator to the seed value
        """
        if self.cs_params.seed is not None:
            self.rng_set = np.random.default_rng(self.cs_params.seed)
        if self.simulator.sim_seed is not None:
            self.simulator.rng_set = np.random.default_rng(self.simulator.sim_seed)
    
    def run(self, job = None):
        """
        Runs multiple GPBO restarts

        Returns
        --------
        gpbo_res_simple: list(BOResults)
            Includes the most relevant results related to a set of BO iters for all restarts
        gpbo_res_GP: list(BOResults)
            Includes the GP emulator classes used and max ei details for each iteration of the BO workflow

        Notes
        ------
        gpbo_res_simple includes the Configuration, Simulator class, Experiment Data Results DataFrame, and termination criteria results
        """
        gpbo_res_simple = []
        gpbo_res_GP = []
        run_start = 0

        if job is not None:
            #Check for files in job for gpbo_res_simple and gpbo_res_GP
            # if os.path.exists("BO_Results.gz") and os.path.exists("BO_Results_GPs.gz"):
            if job.isfile("BO_Results.gz") and job.isfile("BO_Results_GPs.gz"):
                #Load the data from the files
                fileObj1 = gzip.open(job.fn("BO_Results.gz"), "rb")
                # fileObj1 = gzip.open("BO_Results.gz", "rb")
                gpbo_res_simple = pickle.load(fileObj1)
                fileObj1.close()
                fileObj2 = gzip.open(job.fn("BO_Results_GPs.gz"), "rb")
                # fileObj2 = gzip.open("BO_Results_GPs.gz", "rb")
                gpbo_res_GP = pickle.load(fileObj2)
                fileObj2.close()

        self.gpbo_res_simple = gpbo_res_simple
        self.gpbo_res_GP = gpbo_res_GP
        
        simulator_class = self.simulator
        configuration = {
            "DateTime String": self.cs_params.DateTime,
            "Method Name Enum Value": self.method.method_name.value,
            "Case Study Name": self.cs_params.cs_name,
            "Number of Parameters": len(self.simulator.theta_true_names),
            "Number of State Points": self.exp_data.n_x,
            "Exploration Bias Method Value": self.ep_bias.ep_enum.value,
            "Separation Factor": self.cs_params.sep_fact,
            "Normalize": self.cs_params.normalize,
            "Initial Kernel": self.cs_params.kernel,
            "Initial Lengthscale": self.cs_params.lenscl,
            "Initial Outputscale": self.cs_params.outputscl,
            "Retrain GP": self.cs_params.retrain_GP,
            "Reoptimize Obj": self.cs_params.reoptimize_obj,
            "Heat Map Points Generated": self.cs_params.gen_heat_map_data,
            "Max BO Iters": self.cs_params.bo_iter_tot,
            "Number of Workflow Restarts": self.cs_params.bo_run_tot,
            "Seed": self.cs_params.seed,
            "Acq Tolerance": self.cs_params.acq_tol,
            "MC SG Max Points": self.sg_mc_samples,
            "Obj Improvement Tolerance": self.cs_params.obj_tol,
            "Theta Generation Enum Value": self.gen_meth_theta.value,
            "Gen y with Noise": self.cs_params.w_noise,
            "Gen y for Minimized SSE": self.cs_params.get_y_sse,
        }

        #If some runs have already been completed
        if len(self.gpbo_res_simple) > 0:
            # Check if all of the iterations of that runs have been completed 
            if len(self.gpbo_res_GP[-1].list_gp_emulator_class) < self.cs_params.bo_iter_tot:
                #If not, complete the last run before continuing
                run_start = len(gpbo_res_simple) -1
            else:
                #If the run is complete, start from the next run
                run_start = len(gpbo_res_simple)

        #Get the seed based on the run number
        if self.cs_params.seed is not None:
            self.cs_params.seed += run_start

        #Complete remaining runs
        for i in range(run_start, self.cs_params.bo_run_tot, 1):
            #Run the bo workflow and get the results
            bo_results_res, bo_results_GPs = self.__run_bo_workflow(i, job)

            # Update the seed in configuration
            configuration["Seed"] = self.cs_params.seed
            # Add this copy of configuration with the new seed to the bo_results
            bo_results_res.configuration = configuration.copy()
            # # Add simulator class after rng changes (allows us to restart from the next run)
            bo_results_res.simulator_class = copy.deepcopy(simulator_class)
            # On the 1st iteration of the first run, create heat map data if we are actually generating the data
            if i == 0:
                if self.cs_params.gen_heat_map_data == True:
                    # Generate heat map data for each combination of parameter values stored in a dictionary
                    heat_map_data_dict = self.create_heat_map_param_data()
                    # Save these heat map values in the bo_results object
                    # Only store in first list entry to avoid repeated data which stays the same for each iteration.
                    bo_results_GPs.heat_map_data_dict = heat_map_data_dict

            #Save the results to the gpbo_res_simple and gpbo_res_GP lists.
            # The per-iteration loop in __run_bo_to_term normally appends an entry for this run,
            # but when bo_iter_tot == 1 the loop terminates before that append runs, leaving the
            # lists short. Append-or-assign handles both cases (fixes the bo_iter_tot==1 crash).
            if i < len(self.gpbo_res_simple):
                self.gpbo_res_simple[i] = bo_results_res
                self.gpbo_res_GP[i] = bo_results_GPs
            else:
                self.gpbo_res_simple.append(bo_results_res)
                self.gpbo_res_GP.append(bo_results_GPs)

            # #At each restart, resave gpbo_res_simple and gpbo_res_GP to the data file
            if job is not None:
                self.save_results_run(job)

            # Add 1 to the seed to get different seeds when the seeds are set at each restart
            if self.cs_params.seed is not None:
                self.cs_params.seed += 1

        return self.gpbo_res_simple, self.gpbo_res_GP

    def save_results_run(self, job):
        """
        Defines where to save data to and saves data accordingly

        Parameters
        ----------
        restart_bo_results: list of class instances of BO_results, The results of all restarts of the BO workflow for reproduction
        """
        ##Define a path for the data. (Use the name of the case study and date)
        #Get Date only from DateTime String
        savepath1 = job.fn("BO_Results.gz")
        # savepath1 = "BO_Results.gz"
        fileObj1 = gzip.open(savepath1, "wb", compresslevel=1)
        pickled_results1 = pickle.dump(self.gpbo_res_simple, fileObj1)
        fileObj1.close()

        savepath2 = job.fn("BO_Results_GPs.gz")
        # savepath2 = "BO_Results_GPs.gz"
        fileObj2 = gzip.open(savepath2, "wb", compresslevel=2)
        pickled_results2 = pickle.dump(self.gpbo_res_GP, fileObj2)
        fileObj2.close()
