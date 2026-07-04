"""ExpectedImprovement: the expected-improvement acquisition function, with the standard
(objective-GP) and emulator-GP computation strategies selected at construction.
"""
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
from scipy import integrate
import math
import copy
import matplotlib.pyplot as plt
from .exploration import ExplorationBias
from .data import Data
from .methods import GPBOMethod


class ExpectedImprovement:
    """
    The base class for acquisition functions
    Parameters

    Methods
    --------------
    __init__(*): Constructor method
    __set_sg_def(dim): Sets the sparse grid depth
    __set_rand_vars(mean, covar): Sets random variables for MC integration
    compute(): Calculates the expected improvement using the strategy selected at construction
        (standard objective-GP when method=None; the emulator variant for the given method otherwise)
    __calc_ei_emulator(gp_mean, gp_var, y_target): Calculates the expected improvement for the independence approx.
    __calc_ei_log_emulator(gp_mean, gp_var, y_target): Calculates the expected improvement for the log independence approx.
    __ei_approx_ln_term(epsilon, gp_mean, gp_stdev, y_target): Calculates the integral for the log independence approx.
    __calc_ei_sparse(gp_mean, gp_var, y_target): Calculates the expected improvement for the sparse grid method
    __get_sparse_grids(dim, output=1,depth=10, rule="gauss-hermite-odd", verbose = False): Gets the sparse grid
    __calc_ei_mc(gp_var, y_target): Calculates the expected improvement for the Monte Carlo method
    __bootstrap(pilot_sample, ns=100, alpha=0.05, seed = None): Bootstraps for the Monte Carlo method
    """

    def __init__(
        self,
        ep_bias,
        gp_mean,
        gp_covar,
        exp_data,
        best_error_metrics,
        set_seed,
        sg_mc_samples=2000,
        method=None,
    ):
        """
        Parameters
        ----------
        ep_bias: ExplorationBias
            Class with information of exploration bias parameter
        gp_mean: tensor
            The GP model's mean
        gp_covar: tensor
            The GP model's covariance
        exp_data: Data
            The experimental data to evaluate ei with
        best_error_metrics: tuple(float, np.ndarray, np.ndarray)
            The best error, best error parameter set, and best_error_x values of the method. Hint: use calc_best_error()
        set_seed: int or None
            Determines seed for randomizations. None if seed is random
        sg_mc_samples: int, default 2000
            The number of points to use for the Tasmanian sparse grid and Monte Carlo
        """
        assert len(gp_mean) == len(
            gp_covar
        ), "gp_mean and gp_covar must be arrays of the same length"
        assert len(gp_covar.shape) == 2, "gp_covar must be a 2D array"
        assert (
            isinstance(best_error_metrics, tuple) and len(best_error_metrics) == 3
        ), "best_error_metrics must be a tuple of length 3"
        assert all(
            isinstance(arr, np.ndarray) for arr in (gp_mean, gp_covar, exp_data.y_vals)
        ), "gp_mean, gp_var, and exp_data.y_vals must be ndarrays"
        assert isinstance(
            ep_bias, ExplorationBias
        ), "ep_bias must be instance of ExplorationBias"
        assert isinstance(exp_data, Data), "exp_data must be instance of Data"
        assert isinstance(
            best_error_metrics[0], (float, int)
        ), "best_error_metrics[0] must be float or int. Calculate with GPEmulator.calc_best_error()"
        assert isinstance(
            best_error_metrics[1], np.ndarray
        ), "best_error_metrics[1] must be np.ndarray"
        assert (
            isinstance(best_error_metrics[2], np.ndarray)
            or best_error_metrics[2] is None
        ), "best_error_metrics[2] must be np.ndarray (type 2 ei) or None (type 1 ei)"
        assert (
            isinstance(sg_mc_samples, int) or sg_mc_samples is None
        ), "sg_mc_samples must be int (MC and sparse grid) or None (other)"

        # Constructor method
        self.ep_bias = ep_bias
        self.gp_mean = gp_mean
        self.exp_data = exp_data
        self.seed = set_seed
        self.gp_covar = gp_covar
        self.gp_var = np.diag(gp_covar)
        self.best_error = best_error_metrics[0]
        self.best_error_x = best_error_metrics[2]
        self.samples_mc_sg = sg_mc_samples
        # Acquisition computation strategy, selected at construction: method=None uses the
        # standard (objective-GP) EI; a GPBOMethod selects the emulator EI variant it names.
        assert method is None or isinstance(
            method, GPBOMethod
        ), "method must be a GPBOMethod or None"
        self.method = method

        self.rng_rand = np.random.default_rng(self.seed)
        if self.seed is not None:
            self.rng_set = np.random.default_rng(self.seed)
        else:
            self.rng_set = self.rng_rand

    def __set_sg_def(self, dim):
        """
        Sets the sparse grid depth based on the maximum number of samples

        Parameters
        ----------
        dim: int
            The number of dimensions in the sparse grid

        Returns
        -------
        depth: int
            The depth of the sparse grid
        """
        import Tasmanian  # lazy: only needed for sparse-grid acquisition (optional 'sparsegrid' extra)

        depth = 0
        num_points = 0
        # Compute the maximum depth based on the budget
        while num_points <= self.samples_mc_sg:
            depth += 1
            # Generate the global grid with the current depth
            grid_p = Tasmanian.makeGlobalGrid(
                dim, 1, depth, "qphyperbolic", "gauss-hermite-odd"
            )

            # Get the number of points on the grid
            num_points = grid_p.getNumPoints()

            # Check if the number of points exceeds the budget
            if num_points > self.samples_mc_sg:
                if depth > 1:
                    depth -= 1
                break
        return depth

    def __set_rand_vars(self, mean=None, covar=None):
        """
        Sets random variables for MC integration

        Parameters
        ----------
        mean: np.ndarray or None, default None
            The mean of the random variables
        covar: np.ndarray or None, default None
            The covariance of the random variables

        Returns
        ---------
        random_vars: np.ndarray
            Array of multivariate normal random variables
        """
        rng = self.rng_set
        dim = len(self.exp_data.y_vals)
        mc_samples = self.samples_mc_sg  # Set 2000 MC samples

        eigvals, _ = np.linalg.eigh(covar)

        # Get random standard variables
        random_vars_stand = rng.multivariate_normal(
            np.zeros(dim), np.eye(dim), mc_samples
        )
        # If we have a mean and a variance
        if mean is not None or covar is not None:
            # Use the mvn function directly to get the random variables if matrix is Positive Definite
            if np.all(eigvals > 1e-7):
                random_vars = rng.multivariate_normal(
                    mean, np.real(covar), mc_samples, tol=1e-5, method="eigh"
                )
            # Otherwise, use the LDL decomposition
            else:
                lu, d, perm = scipy.linalg.ldl(
                    np.real(covar), lower=True
                )  # Use the lower part
                # Clip tiny-negative diagonal entries (numerically non-PSD GP covariance) to 0
                # before sqrt to avoid NaN samples; a no-op when the diagonal is non-negative.
                # Without this, large-magnitude objectives (e.g. CS12 Yield-Loss) produce NaN
                # Monte-Carlo EI, which made every acquisition restart fail. See refactor_notes.
                sqrt_d = np.sqrt(np.clip(np.diag(d), 0, None))[:, np.newaxis]
                random_vars = (
                    mean[:, np.newaxis] + lu[:, perm] @ (sqrt_d * random_vars_stand.T)
                ).T

        return random_vars

    def compute(self):
        """
        Compute expected improvement using the strategy selected at construction.

        Dispatches to the standard (objective-GP) calculation when no ``method`` was
        given, or to the emulator calculation for the ``method`` that was constructed.

        Returns
        -------
        ei : np.ndarray
        ei_term_df : pd.DataFrame
        """
        if self.method is None:
            return self.__compute_standard()
        return self.__compute_emulator()

    def __compute_standard(self):
        """
        Calculates expected improvement of type 1 (standard) GPBO given gp_mean, gp_var, and best_error data

        Returns
        -------
        ei: np.ndarray
            The expected improvement of the parameter set
        ei_term_df: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter set
        """
        columns = ["best_error", "z", "cdf", "pdf", "ei_term_1", "ei_term_2", "ei"]
        ei_term_df = pd.DataFrame(columns=columns)

        ei = np.zeros(len(self.gp_mean))

        for i in range(len(self.gp_mean)):
            pred_stdev = np.sqrt(self.gp_var[i])  # 1xn_test
            # Checks that all standard deviations are positive
            if pred_stdev > 0:
                # Calculates z-score based on Eq. 6b in Wang and Dowling (2022), COCHE
                z = (
                    self.best_error * self.ep_bias.ep_curr - self.gp_mean[i]
                ) / pred_stdev  # scaler
                # Calculates ei based on Eq. 6a in Wang and Dowling (2022), COCHE
                # Explotation term
                ei_term_1 = (
                    self.best_error * self.ep_bias.ep_curr - self.gp_mean[i]
                ) * norm.cdf(
                    z
                )  # scaler
                # Exploration Term
                ei_term_2 = pred_stdev * norm.pdf(z)  # scaler
                ei[i] = ei_term_1 + ei_term_2  # scaler

                # Create a temporary DataFrame for the current row
                row_data = pd.DataFrame(
                    [
                        [
                            self.best_error,
                            z,
                            norm.cdf(z),
                            norm.pdf(z),
                            ei_term_1,
                            ei_term_2,
                            ei[0],
                        ]
                    ],
                    columns=columns,
                )

            else:
                # Sets ei to zero if standard deviation is zero
                ei[i] = 0
                # Create a temporary DataFrame for the current row
                row_data = pd.DataFrame(
                    [[self.best_error, None, None, None, None, None, ei]],
                    columns=columns,
                )

            # Concatenate the temporary DataFrame with the main DataFrame
            ei_term_df = pd.concat(
                [ei_term_df.astype(row_data.dtypes), row_data], ignore_index=True
            )
        return ei, ei_term_df

    def __compute_emulator(self):
        """
        Calculates expected improvement of type 2 (emulator) GPBO for the constructed method

        Returns
        -------
        ei: np.ndarray
            The expected improvement of the parameter set
        ei_term_df: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter set

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        ei_term_df = pd.DataFrame()
        assert isinstance(
            self.best_error_x, np.ndarray
        ), "best_error_metrics[1] must be np.ndarray for type 2 ei calculations"
        assert isinstance(self.method, GPBOMethod), "method must be type GPBOMethod"
        # Num thetas = #gp mean pts/number of x_vals for Type 2
        num_thetas = int(len(self.gp_mean) / self.exp_data.n_x)
        # Define n as the number of x values
        n = self.exp_data.n_x
        # Initialize array of eis for eacch theta
        ei = np.zeros(num_thetas)

        # Loop over number of thetas in theta_val_set
        for i in range(num_thetas):  # 1 ei per theta and also 1 sse per theta
            # Get gp mean and var for each set of x values
            # for ei, ensure that a gp mean and gp_var corresponding to a certain theta are sent
            gp_mean_i = self.gp_mean[i * n : (i + 1) * n]
            gp_var_i = self.gp_var[i * n : (i + 1) * n]

            # Calculate ei for a given theta (ei for all x over each theta)

            if self.method.method_name.value == 3:  # 2A
                # Calculate ei for a given theta (ei for all x over each theta)
                ei[i], row_data = self.__calc_ei_emulator(
                    gp_mean_i, gp_var_i, self.exp_data.y_vals
                )

            elif self.method.method_name.value == 4:  # 2B
                ei[i], row_data = self.__calc_ei_log_emulator(
                    gp_mean_i, gp_var_i, self.exp_data.y_vals
                )

            elif self.method.method_name.value == 5:  # 2C
                ei[i], row_data = self.__calc_ei_sparse(
                    gp_mean_i, gp_var_i, self.exp_data.y_vals
                )

            elif self.method.method_name.value == 6:  # 2D
                ei[i], row_data = self.__calc_ei_mc(
                    gp_var_i, self.exp_data.y_vals
                )

            else:
                raise ValueError(
                    "method.method_name.value must be 3 (2A), 4 (2B), 5 (2C), or 6 (2D)"
                )

        # Concatenate the temporary DataFrame with the main DataFrame
        ei_term_df = pd.concat([ei_term_df, row_data], ignore_index=True)
        ei_term_df.columns = row_data.columns.tolist()

        return ei, ei_term_df

    def __calc_ei_emulator(self, gp_mean, gp_var, y_target):
        """
        Calculates the expected improvement of the emulator approach with an independence approximation (2A)

        Parameters
        ----------
        gp_mean: np.ndarray
            Model mean at state points (x) for a given parameter set
        gp_variance: np.ndarray
            Model variance at state points (x) for a given parameter set
        y_target: np.ndarray
            The expected value of the function from data or other source

        Returns
        -------
        ei_temp: np.ndarray
            The expected improvement for one parameter set
        row_data: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter set

        """
        # Create column names
        columns = [
            "bound_l",
            "bound_u",
            "cdf_l",
            "cdf_u",
            "eta_l",
            "eta_u",
            "psi_l",
            "psi_u",
            "ei_term1",
            "ei_term2",
            "ei_term3",
            "ei",
            "ei_total",
        ]

        # Initialize ei as all zeros
        ei = np.zeros(len(gp_var))
        # Create a mask for values where var > 0. Set a value of 1e-14?
        pos_stdev_mask = gp_var > 0

        # Assuming all standard deviations are not zero
        if np.any(pos_stdev_mask):
            # Get indices and values where stdev > 0
            valid_indices = np.where(pos_stdev_mask)[0]
            pred_stdev_val = np.sqrt(gp_var[valid_indices])
            gp_var_val = gp_var[valid_indices]
            gp_mean_val = gp_mean[valid_indices]
            y_target_val = y_target[valid_indices]
            best_errors_x = self.best_error_x[valid_indices]

            # If variance is close to zero this is important
            with np.errstate(divide="warn"):
                # Creates upper and lower bounds and described by Equation X in Manuscript
                bound_a = (
                    (y_target_val - gp_mean_val)
                    + np.sqrt(best_errors_x * self.ep_bias.ep_curr)
                ) / pred_stdev_val
                bound_b = (
                    (y_target_val - gp_mean_val)
                    - np.sqrt(best_errors_x * self.ep_bias.ep_curr)
                ) / pred_stdev_val
                bound_lower = np.minimum(bound_a, bound_b)
                bound_upper = np.maximum(bound_a, bound_b)

                # Creates EI terms in terms of Equation X in Manuscript
                ei_term1_comp1 = norm.cdf(bound_upper) - norm.cdf(bound_lower)
                ei_term1_comp2 = (best_errors_x * self.ep_bias.ep_curr) - (
                    y_target_val - gp_mean_val
                ) ** 2

                ei_term2_comp1 = 2 * (y_target_val - gp_mean_val) * pred_stdev_val
                ei_eta_upper = -np.exp(-(bound_upper**2) / 2) / np.sqrt(2 * np.pi)
                ei_eta_lower = -np.exp(-(bound_lower**2) / 2) / np.sqrt(2 * np.pi)
                ei_term2_comp2 = ei_eta_upper - ei_eta_lower

                ei_term3_comp1 = bound_upper * ei_eta_upper
                ei_term3_comp2 = bound_lower * ei_eta_lower

                ei_term3_comp3 = (1 / 2) * scipy.special.erf(bound_upper / np.sqrt(2))
                ei_term3_comp4 = (1 / 2) * scipy.special.erf(bound_lower / np.sqrt(2))

                ei_term3_psi_upper = ei_term3_comp1 + ei_term3_comp3
                ei_term3_psi_lower = ei_term3_comp2 + ei_term3_comp4

                ei_term1 = ei_term1_comp1 * ei_term1_comp2
                ei_term2 = ei_term2_comp1 * ei_term2_comp2
                ei_term3 = -gp_var_val * (ei_term3_psi_upper - ei_term3_psi_lower)

                # Set EI values of indecies where pred_stdev > 0
                ei[valid_indices] = ei_term1 + ei_term2 + ei_term3

            # The Ei is the sum of the ei at each value of x
            ei_temp = np.sum(ei)
            row_data_lists = pd.DataFrame(
                [
                    [
                        bound_lower,
                        bound_upper,
                        norm.cdf(bound_lower),
                        norm.cdf(bound_upper),
                        ei_eta_lower,
                        ei_eta_upper,
                        ei_term3_psi_lower,
                        ei_term3_psi_upper,
                        ei_term1,
                        ei_term2,
                        ei_term3,
                        ei,
                        ei_temp,
                    ]
                ],
                columns=columns,
            )
        else:
            ei_temp = 0
            row_data_lists = pd.DataFrame(
                [
                    [
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        ei_temp,
                    ]
                ],
                columns=columns,
            )

        row_data = row_data_lists.apply(
            lambda col: col.explode(ignore_index=True), axis=0
        ).reset_index(drop=True)

        return ei_temp, row_data

    def __calc_ei_log_emulator(self, gp_mean, gp_var, y_target):
        """
        Calculates the expected improvement of the emulator approach with a log-scaled independence approximation (2B)

        Parameters
        ----------
        gp_mean: np.ndarray
            Model mean at state points (x) for a given parameter set
        gp_variance: np.ndarray
            Model variance at state points (x) for a given parameter set
        y_target: np.ndarray
            The expected value of the function from data or other source

        Returns
        -------
        ei_temp: np.ndarray
            The expected improvement for one parameter set
        row_data: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter set
        """
        columns = [
            "best_error",
            "bound_l",
            "bound_u",
            "ei_term1",
            "ei_term2",
            "ei",
            "ei_total",
        ]

        # Initialize ei as all zeros
        ei = np.zeros(len(gp_var))

        # Create a mask for values where pred_stdev > 0
        pos_stdev_mask = gp_var > 0
        # best_errors_x_all = np.log(self.best_error_x)
        best_errors_x_all = np.log(np.where(self.best_error_x < 1e-16, 1e-16, self.best_error_x))

        # Assuming all standard deviations are not zero
        if np.any(pos_stdev_mask):
            # Get indices and values where stdev > 0
            valid_indices = np.where(pos_stdev_mask)[0]
            pred_stdev_val = np.sqrt(gp_var[valid_indices])
            gp_mean_val = gp_mean[valid_indices]
            y_target_val = y_target[valid_indices]
            best_errors_x = copy.deepcopy(best_errors_x_all)[valid_indices]
            # Important when stdev is close to 0
            with np.errstate(divide="warn"):
                # Creates upper and lower bounds and described by Alex Dowling's Derivation
                bound_a = (
                    (y_target_val - gp_mean_val)
                    + np.sqrt(np.exp(best_errors_x * self.ep_bias.ep_curr))
                ) / pred_stdev_val  # 1xn
                bound_b = (
                    (y_target_val - gp_mean_val)
                    - np.sqrt(np.exp(best_errors_x * self.ep_bias.ep_curr))
                ) / pred_stdev_val  # 1xn
                bound_lower = np.minimum(bound_a, bound_b)
                bound_upper = np.maximum(bound_a, bound_b)

                # Calculate EI
                args = (gp_mean_val, pred_stdev_val, y_target_val, self.ep_bias.ep_curr)
                ei_term_1 = (best_errors_x * self.ep_bias.ep_curr) * (
                    norm.cdf(bound_upper) - norm.cdf(bound_lower)
                )
                ei_term_2_out = np.array(
                    [
                        integrate.quad(
                            self.__ei_approx_ln_term, bl, bu, args=(gm, ps, yt)
                        )
                        for bl, bu, gm, ps, yt in zip(
                            bound_lower,
                            bound_upper,
                            gp_mean_val,
                            pred_stdev_val,
                            y_target_val,
                        )
                    ]
                )

                ei_term_2 = (-2) * ei_term_2_out[:, 0]
                term_2_abs_err = ei_term_2_out[:, 1]

                # Add ei values to correct indecies.
                ei[valid_indices] = ei_term_1 + ei_term_2

            # The Ei is the sum of the ei at each value of x
            ei_temp = np.sum(ei)
            row_data_lists = pd.DataFrame(
                [
                    [
                        best_errors_x,
                        bound_lower,
                        bound_upper,
                        ei_term_1,
                        ei_term_2,
                        ei,
                        ei_temp,
                    ]
                ],
                columns=columns,
            )
        else:
            ei_temp = 0
            row_data_lists = pd.DataFrame(
                [[best_errors_x_all, "N/A", "N/A", "N/A", "N/A", "N/A", ei_temp]],
                columns=columns,
            )

        row_data = row_data_lists.apply(
            lambda col: col.explode(ignore_index=True), axis=0
        ).reset_index(drop=True)

        return ei_temp, row_data

    def __ei_approx_ln_term(self, epsilon, gp_mean, gp_stdev, y_target):
        """
        Calculates the integrand of expected improvement intregral for the log independence approximation

        Parameters
        ----------
        epsilon: float
            The random variable over which we integrate
        gp_mean: np.ndarray
            GP model mean
        gp_stdev: np.ndarray
            GP model stdev
        y_target: np.ndarray
            The expected value of the function from data or other source

        Returns
        -------
        ei_term_2_integral: np.ndarray
            The expected improvement for term 2 of the GP model for method 2B
        """
        # Define inside term as the maximum of 1e-14 or abs((y_target - gp_mean - gp_stdev*epsilon))
        inside_term = max(1e-14, abs((y_target - gp_mean - gp_stdev * epsilon)))

        ei_term_2_integral = math.log(inside_term) * norm.pdf(epsilon)

        return ei_term_2_integral

    def __calc_ei_sparse(self, gp_mean, gp_var, y_target):
        """
        Calculates the expected improvement of the emulator approach with a sparse grid approach (2C)

        Parameters
        ----------
        gp_mean: np.ndarray
            Model mean at state points (x) for a given parameter set
        gp_var: np.ndarray
            Model variance at state points (x) for a given parameter set
        y_target: np.ndarray
            The expected value of the function from data or other source

        Returns
        -------
        ei_temp: np.ndarray
            The expected improvement for one parameter set
        row_data: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter set

        Notes
        -----
        To apply the sparse grid method on multiple parameter sets you must loop over each parameter set, calculate the posterior mean and variance, and then
        apply the sparse grid method to calculate EI for each parameter set.
        If the covariance matrix is not positive definite, the LDL decomposition is used instead of Cholesky factorization.
        """
        columns = ["best_error", "sse_temp", "improvement", "ei_total"]

        # Create a mask for values where pred_stdev >= 0 (Here approximation includes domain stdev >= 0)
        pos_stdev_mask = gp_var >= 0

        # Assuming all standard deviations are not zero
        if np.any(pos_stdev_mask):
            ndims = len(y_target)
            # Get indices and values where stdev > 0
            valid_indices = np.where(pos_stdev_mask)[0]
            gp_stdev_val = np.sqrt(gp_var[valid_indices])
            gp_mean_val = gp_mean[valid_indices]
            y_target_val = y_target[valid_indices]
            gp_mean_min_y = y_target_val - gp_mean_val

            # #Obtain Sparse Grid points and weights
            # Get maximum depth given number of points p
            sg_depth = self.__set_sg_def(ndims)
            points_p, weights_p = self.__get_sparse_grids(
                ndims, output=1, depth=sg_depth, rule="gauss-hermite-odd", verbose=False
            )

            # Diagonalize covariance matrix
            try:
                # As long as the covariance matrix is positive definite use Cholesky decomposition
                L = scipy.linalg.cholesky(np.real(self.gp_covar), lower=True)
            except:
                # If it is not, use LDL decomposition instead
                lu, d, perm = scipy.linalg.ldl(
                    np.real(self.gp_covar), lower=True
                )  # Use the upper part
                # Clip tiny-negative entries (numerically non-PSD GP covariance) before sqrt to
                # avoid NaN; a no-op when entries are already non-negative.
                L = lu[:, perm] @ np.diag(np.sqrt(np.clip(d, 0, None)))

            transformed_points = L @ points_p.T
            gp_random_vars = self.gp_mean[:, np.newaxis] + np.sqrt(2) * (
                transformed_points
            )
            sse_temp = np.sum((y_target[:, np.newaxis] - gp_random_vars) ** 2, axis=0)
            # Apply max operator (equivalent to max[(best_error*ep) - SSE_Temp,0])
            error_diff = self.best_error * self.ep_bias.ep_curr - sse_temp
            # Smooth max improvement function
            improvement = (0.5) * (error_diff + np.sqrt(error_diff**2 + 1e-7))

            # Calculate EI_temp using vectorized operations
            ei_temp = (np.pi ** (-ndims / 2)) * np.dot(weights_p, improvement)

        else:
            ei_temp = 0
            sse_temp = "N/A"
            improvement = "N/A"

        row_data_lists = pd.DataFrame(
            [[self.best_error, sse_temp, improvement, ei_temp]], columns=columns
        )
        row_data = row_data_lists.apply(
            lambda col: col.explode(ignore_index=True), axis=0
        ).reset_index(drop=True)

        return ei_temp, row_data

    def __get_sparse_grids(
        self, dim, output=1, depth=10, rule="gauss-hermite-odd", verbose=False
    ):
        """
        This function builds a sparse grid

        Parameters
        -----------
        dim: int
            Sparse grids dimension
        output: int, default 1
            Output level for function that would be interpolated
        depth: int, default 10
            Depth level. Controls density of abscissa points. Uses qphyperbolic level system
        rule: str, default 'gauss-hermite-odd'
            Quadrature rule
        verbose: bool, default False
            Determines Whether or not plot of sparse grid is shown

        Returns
        --------
        points_p: np.ndarray
            The sparse grid points
        weights_p: np.ndarray
            The Gauss-Hermite Quadrature Rule Weights

        Notes
        ------
        A figure shows a 2D sparse grid if verbose = True
        """
        import Tasmanian  # lazy: only needed for sparse-grid acquisition (optional 'sparsegrid' extra)

        # Get grid points and weights
        grid_p = Tasmanian.makeGlobalGrid(dim, output, depth, "qphyperbolic", rule)
        points_p = grid_p.getPoints()
        weights_p = grid_p.getQuadratureWeights()
        if verbose == True:
            # If verbose is true print the sparse grid
            for i in range(len(points_p)):
                plt.scatter(points_p[i, 0], points_p[i, 1])
                plt.title("Sparse Grid of " + rule.title(), fontsize=20)
                plt.xlabel(r"$ϵ$ Dimension 1", fontsize=20)
                plt.ylabel(r"$ϵ$ Dimension 2", fontsize=20)
            plt.show()
        return points_p, weights_p

    def __calc_ei_mc(self, gp_var, y_target):
        """
        Calculates the expected improvement of the emulator approach with a Monte Carlo approach (2D)

        Parameters
        ----------
        gp_variance: np.ndarray
            Model variance at state points x for a given parameter set
        y_target: np.ndarray
            The expected value of the function from data or other source

        Returns
        -------
        ei_mean: np.ndarray
            The expected improvement for one parameter set
        row_data: pd.DataFrame
            Pandas dataframe containing the values of calculations associated with ei for the parameter set

        Note
        -----
        To apply the Monte Carlo method on multiple parameter sets you must loop over each parameter set, calculate the posterior mean and variance, and then
        apply the MC method to calculate EI for each parameter set.
        """
        # Set column names
        columns = [
            "best_error",
            "sse_temp",
            "improvement",
            "ci_lower",
            "ci_upper",
            "ei_total",
        ]

        # Calc EI
        # Create a mask for values where pred_stdev >= 0 (Here approximation includes domain stdev >= 0)
        pos_stdev_mask = gp_var >= 0

        # Assuming all standard deviations are not zero
        if np.any(pos_stdev_mask):
            # Set random variables for MC integration
            self.random_vars = self.__set_rand_vars(self.gp_mean, self.gp_covar)
            sse_temp = np.sum(
                (y_target[:, np.newaxis].T - self.random_vars) ** 2, axis=1
            )
            error_diff = self.best_error * self.ep_bias.ep_curr - sse_temp
            # Smooth max improvement function
            improvement = (0.5) * (
                error_diff + np.sqrt(error_diff**2 + 1e-7)
            ).reshape(-1, 1)
            # Flatten improvement
            ei_temp = improvement.flatten()

        else:
            ei_temp = 0
            sse_temp = "N/A"
            improvement = "N/A"

        # Calc monte carlo integrand for each theta and add it to the total
        ei_mean = np.average(ei_temp)  # y.sum()/len(y)
        # Note: Domain for random variable is 0-1, so V for MC is 1

        # Perform bootstrapping
        ci_interval = self.__bootstrap(ei_temp, ns=100, alpha=0.05)

        ci_l = ci_interval[0]
        ci_u = ci_interval[1]

        row_data_lists = pd.DataFrame(
            [[self.best_error, sse_temp, improvement, ci_l, ci_u, ei_temp]],
            columns=columns,
        )
        row_data = row_data_lists.apply(
            lambda col: col.explode(ignore_index=True), axis=0
        ).reset_index(drop=True)

        return ei_mean, row_data

    def __bootstrap(self, pilot_sample, ns=100, alpha=0.05):
        """
        Bootstrapping code for Monte Carlo method. Generously provided by Ryan Smith.

        Parameters
        ----------
        pilot_sample: np.ndarray (n_samples x dim param set)
            The samples to perform bootstrapping on
        ns: int, default 100
            Number of bootstrapping samples
        alpha: float, default 0.05
            On interval (0,1). The level of significance associated with the bootstrapping
        set_seed: int or None, default None
            Seed associated with bootstrapping

        Returns
        --------
        ci_percentile: np.ndarray
            The confidence interval of the MC samples
        """
        # pilot_sample has one column per rv, one row per observation
        # alpha is the level of significance; 0.05 for 95% confidence interval
        quantiles = np.array([alpha * 0.5, 1.0 - alpha * 0.5])

        # Guard against a scalar/0-d pilot_sample. __calc_ei_mc sets ei_temp = 0 (a scalar)
        # in the degenerate case where every GP predictive variance is negative (numerically
        # non-PSD covariance, e.g. CS12 Yield-Loss); np.mean(0, axis=0) would raise AxisError
        # (a ValueError subclass) that was silently swallowed by the caller, leaving no valid
        # acquisition candidate and crashing the BO iteration. atleast_1d makes the bootstrap
        # return a degenerate [0, 0] CI instead. No-op for the normal (1-D array) case.
        pilot_sample = np.atleast_1d(pilot_sample)

        # Determine mean of all original samples and its shape
        theta_orig = np.mean(pilot_sample, axis=0)

        # Initialize bootstrap samples as zeros
        theta_bs = np.zeros(tuple([ns] + list(theta_orig.shape)))

        # Create bootstrap samples
        for ibs in range(ns):
            samples = self.rng_set.choice(
                pilot_sample, size=pilot_sample.shape[0], replace=True
            )
            theta_bs[ibs, ...] = np.mean(samples, axis=0)

        # percentile CI
        ci_percentile = np.quantile(theta_bs, quantiles, 0)

        # return theta_orig, theta_bs, CI_percentile
        return ci_percentile
