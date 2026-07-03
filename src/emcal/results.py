"""BOResults: container for the results of a Bayesian-optimization run (trajectory
DataFrame, per-iteration GP emulators, experimental data, termination reason).
"""
import pandas as pd
from .simulator import Simulator
from .data import Data
from .emulators import ObjectiveGP, EmulatorGP


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
