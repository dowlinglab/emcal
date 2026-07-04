"""BOResults: container for the results of a Bayesian-optimization run (trajectory
DataFrame, per-iteration GP emulators, experimental data, termination reason).
"""
import numpy as np
import pandas as pd
from .simulator import Simulator
from .data import Data
from .emulators import ObjectiveGP, EmulatorGP


# The single source of truth for GPBODriver's per-iteration results columns -- previously
# duplicated verbatim between __run_bo_iter (as the columns of the 1-row DataFrame
# build_iteration_row now assembles) and __run_bo_to_term (as the columns of the empty
# DataFrame results are concatenated into).
ITERATION_COLUMNS = [
    "best_error",
    "alpha",
    "theta_at_acq",
    "acq_value",
    "sse_at_acq",
    "mse_at_acq",
    "theta_at_min",
    "sse_gp",
    "sse_actual",
    "mse_gp",
    "mse_actual",
    "time_per_iter",
]


def build_iteration_row(
    best_error,
    ep_curr,
    acq_theta_vals,
    opt_acq,
    opt_acq_sim,
    min_sse_theta_vals,
    min_sse_gp,
    min_sse_sim,
    time_per_iter,
    log_scaled,
    num_exp_x,
):
    """
    Assembles the single-row results DataFrame for one BO iteration.

    Parameters
    ----------
    best_error: float
        The best error (sse) at the start of the iteration
    ep_curr: float
        The current exploration bias value
    acq_theta_vals: np.ndarray
        Parameter set that optimized the acquisition function
    opt_acq: float
        The optimized acquisition function value
    opt_acq_sim: float
        The (possibly log-scaled) simulated/actual objective value at acq_theta_vals
    min_sse_theta_vals: np.ndarray
        Parameter set that minimized the SSE/E[SSE] objective
    min_sse_gp: float
        The (possibly log-scaled) GP-predicted objective value at min_sse_theta_vals
    min_sse_sim: float or None
        The (possibly log-scaled) simulated/actual objective value at min_sse_theta_vals,
        or None if not generated (cs_params.get_y_sse is False)
    time_per_iter: float
        Wall-clock seconds spent on this iteration
    log_scaled: bool
        Whether best_error/opt_acq_sim/min_sse_gp/min_sse_sim are log-scaled (method.log_scaled)
    num_exp_x: int
        Number of experimental state points (exp_data.n_x), used to convert SSE to MSE

    Returns
    -------
    iter_df: pd.DataFrame
        Single-row DataFrame with columns ITERATION_COLUMNS
    """
    MSE_acq_obj_act = (
        np.exp(opt_acq_sim) / num_exp_x if log_scaled else opt_acq_sim / num_exp_x
    )
    if min_sse_sim is not None:
        MSE_obj_act = (
            np.exp(min_sse_sim) / num_exp_x if log_scaled else min_sse_sim / num_exp_x
        )
    else:
        MSE_obj_act = None

    MSE_obj_gp = np.exp(min_sse_gp) / num_exp_x if log_scaled else min_sse_gp / num_exp_x

    bo_iter_results = [
        best_error,
        float(ep_curr),
        acq_theta_vals,
        float(np.asarray(opt_acq).item()),
        opt_acq_sim,
        MSE_acq_obj_act,
        min_sse_theta_vals,
        min_sse_gp,
        min_sse_sim,
        MSE_obj_gp,
        MSE_obj_act,
        time_per_iter,
    ]
    iter_df = pd.DataFrame(columns=ITERATION_COLUMNS)
    iter_df.loc[0] = bo_iter_results
    return iter_df


class BOResults:
    """
    The base class for storing important BO Results

    Methods:
    --------
    __init__(*): Constructor method
    """

    # Class variables and attributes
    def __init__(
        self,
        configuration,
        simulator_class,
        exp_data_class,
        list_gp_emulator_class,
        results_df,
        max_ei_details_df,
        why_term,
        heat_map_data_dict,
    ):
        """
        Parameters
        ----------
        configuration: dict
            Dictionary containing the configuration of the BO algorithm
        simulator_class: Simulator
            Class containing the Simulator class information
        exp_data_class: Data
            The experimental data for the workflow
        list_gp_emulator_class: list(GPEmulator)
            Contains all gp_emulator information at each BO iter
        results_df: pd.DataFrame
            Dataframe including the values pertinent to BO for all BO runs
        max_ei_details_df: pd.DataFrame
            Dataframe including ei components of the best EI at each iter
        why_term: str
            String detailing the reason for algorithm termination
        heat_map_data_dict: dict
            Heat map data for each set of 2 parameters indexed by parameter names "param_1-param_2"
        """
        assert isinstance(configuration, dict) or configuration is None, "configuration must be a dictionary or None"
        assert isinstance(simulator_class, Simulator) or simulator_class is None, "simulator_class must be an instance of Simulator or None"
        assert isinstance(exp_data_class, Data) or exp_data_class is None, "exp_data_class must be an instance of Data or None"
        assert isinstance(list_gp_emulator_class, list) or list_gp_emulator_class is None, "list_gp_emulator_class must be a list or None"
        if list_gp_emulator_class is not None:
            assert all(isinstance(gp_emulator, (ObjectiveGP, EmulatorGP)) for gp_emulator in 
                   list_gp_emulator_class), "entries of list list_gp_emulator_class must be ObjectiveGP or EmulatorGP"
        assert isinstance(results_df, pd.DataFrame) or results_df is None, "results_df must be a pandas DataFrame or None"
        assert isinstance(max_ei_details_df, pd.DataFrame) or max_ei_details_df is None, "max_ei_details_df must be a pandas DataFrame or None"
        assert isinstance(why_term, (str, int)) or why_term is None, "why_term must be a string, int, or None"
        assert isinstance(heat_map_data_dict, dict) or heat_map_data_dict is None, "heat_map_data_dict must be a dictionary or None"
        # Constructor method
        self.configuration = configuration
        self.simulator_class = simulator_class
        self.exp_data_class = exp_data_class
        self.results_df = results_df
        self.max_ei_details_df = max_ei_details_df
        self.why_term = why_term
        self.list_gp_emulator_class = list_gp_emulator_class
        self.heat_map_data_dict = heat_map_data_dict
