# Import Dependencies
import os
import warnings
import numpy as np
import pandas as pd
import copy
from collections.abc import Iterable

# NOTE: signac is intentionally NOT imported here. This module operates on a "job-like"
# object that exposes `.sp` (statepoint), `.fn(name)` (path into a results workspace) and
# `.id`. A real signac job satisfies that interface, and so does the signac-free JobContext
# defined below — so analysis/plotting run with or without signac installed. The project
# passed to General_Analysis is likewise used only via its own methods (find_jobs/open_job),
# so no top-level signac dependency is required.


from .case_studies import *
from .GPBO_Classes_New import *

import pickle
import gzip
import json
import ast
import re


def load_gz(file_path):
    """
    Opens a .gz or .pickle file based on the extension

    Parameters
    ----------
    file_path: str
        The file path of the data

    Returns
    -------
    results: pickled object
        The results stored in the .pickle or .gz file

    Raises
    ------
    AssertionError
        If the file path is not a string
    ValueError
        If the file type is not .gz or .pickle
    """
    assert isinstance(file_path, str), "file_path must be a string"
    if file_path.endswith(".pickle") or file_path.endswith(".pkl"):
        with open(file_path, "rb") as fileObj:
            results = pickle.load(fileObj)
    elif file_path.endswith(".gz"):
        with gzip.open(file_path, "rb") as fileObj:
            results = pickle.load(fileObj)
    else:
        raise ValueError("File type must be .gz or .pickle!")

    return results


class _StatePoint(dict):
    """A dict that also supports attribute access, mimicking a signac job's ``.sp``
    (so both ``sp["cs_name_val"]`` and ``sp.cs_name_val`` work)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


class JobContext:
    """Signac-free stand-in for a signac job.

    Exposes the read-only interface the analysis/plotting code needs from a job —
    ``.sp`` (statepoint), ``.fn(name)`` (path into a results workspace directory), and
    ``.id`` — so results saved outside a signac workflow can still be analyzed/plotted.
    A real ``signac.job.Job`` satisfies the same interface, so the analysis methods accept
    either (see ``is_job_like``).

    Parameters
    ----------
    workspace_dir : str
        Directory containing this job's result files (e.g. ``BO_Results.gz``,
        ``BO_Results_GPs.gz``, ``signac_statepoint.json``).
    statepoint : dict, optional
        The job statepoint (configuration). Defaults to an empty dict.
    job_id : str, optional
        An identifier for the job (used only for labeling/output paths).
    """

    def __init__(self, workspace_dir, statepoint=None, job_id=None):
        self.workspace_dir = str(workspace_dir)
        self.sp = statepoint if isinstance(statepoint, _StatePoint) else _StatePoint(statepoint or {})
        self.id = job_id

    def fn(self, name=""):
        """Return the path to ``name`` inside this job's workspace directory."""
        return os.path.join(self.workspace_dir, name)

    def statepoint(self):
        return dict(self.sp)


def is_job_like(obj):
    """True if ``obj`` behaves like a signac job for this module's read-only use.

    Accepts both a real ``signac.job.Job`` and a :class:`JobContext` (duck-typed on
    ``.sp`` + ``.fn``), so callers do not need signac installed.
    """
    return hasattr(obj, "sp") and callable(getattr(obj, "fn", None))


class General_Analysis:
    """
    The base class for per-run/diagnostic GPBO workflow analysis (single job/JobContext or a
    fitted emulator). Multi-job/cross-method/benchmark analysis (least-squares and
    derivative-free baselines, cross-case-study aggregation) has moved to the archive repo.

    Methods
    --------------
    __init__(*args, **kwargs): Constructor method
    make_dir_name_from_criteria(dict_to_use, is_nested = False): Makes a directory string name from a criteria dictionary
    str_to_array_df_col(str_arr): Used to turn arrays from csvs loaded to pd dataframes from strings into arrays
    get_run_dataframe(job, save_csv = None): Get best data from jobs and optionally save the csvs for the data
    load_data(path): Loads data from a file based on the file extension
    save_data(data, save_path): Saves data to a file based on the file extension
    __z_choice_helper(z_choices, theta_true_data, data_type): Helper function to get the correct data and data names for plotting
    best_error(job): Computes/loads best-error metrics for a job's GP training data
    __preprocess_analyze(job, z_choice, data_type): Helper function to preprocess data for analysis
    objective_trajectories(job, z_choices): Compiles objective data for plotting
    parameter_trajectories(job, z_choice): Compiles parameter set data for plotting
    hyperparameter_trajectories(job): Compiles hyperparameter data for plotting
    __rebuild_cs(sp_data): Rebuilds the BOConfig instance from the job statepoint data
    gp_parity_data(job, run_num, bo_iter): Compiles parity plot data for plotting
    gp_heat_map_data(job, run_num, bo_iter, pair_id, get_ei = False): Compiles heat map data for plotting
    """

    # Class variables and attributes

    def __init__(self, criteria_dict, project, mode, save_csv):
        """
        Parameters
        ----------
        criteria_dict: dict
            Signac statepoints to consider for the job. Should include minimum of cs_name_val
        project: signac.project.Project
            The signac project to analyze
        mode: str
            The mode to analyze the data in ('act', 'acq', or 'gp')
        save_csv: bool
            Whether to save csvs

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        # Asserts
        assert isinstance(criteria_dict, dict), "criteria_dict must be a dictionary"
        # project is a signac project when using the signac workflow, or None for signac-free
        # use (operate on JobContext objects passed directly to the per-job analysis methods).
        assert project is None or hasattr(project, "find_jobs"), (
            "project must be a signac Project (with .find_jobs) or None"
        )
        assert isinstance(save_csv, bool), "save_csv must be a boolean"
        assert mode in ["act", "acq", "gp"], "mode must be 'act', 'acq', or 'gp'"
        # Collect unique statepoints of all jobs (only possible with a live signac project)
        if project is not None:
            statepoint_names = set()
            for job in project:
                statepoint_names.update(job.statepoint().keys())
            key_list = list(statepoint_names)
            self.sp_keys_valid = key_list
            assert (
                all(key in key_list for key in list(criteria_dict.keys())) == True
            ), "All keys in criteria_dict must be in project statepoints"
        else:
            self.sp_keys_valid = list(criteria_dict.keys())

        # Constructor method
        self.mode = mode
        self.criteria_dict = criteria_dict
        self.project = project
        self.save_csv = save_csv

    def make_dir_name_from_criteria(self, dict_to_use, is_nested=False):
        """
        Makes a directory string name from a criteria dictionary

        Parameters
        ----------
        dict_to_use: dict
            Dictionary to use to make directory name
        is_nested: bool, default False
            Whether the dictionary is nested or not

        Returns
        -------
        result_dir: str
            The directory name from the dictionary

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        -----
        For proper results, ALWAYS use this function with is_nested = False
        """
        assert isinstance(dict_to_use, dict), "dict_to_use must be a dictionary"
        assert isinstance(is_nested, bool), "is_nested must be a boolean"
        # Note, criteria dict is only checked when is_nested = False
        if not is_nested:
            assert (
                all(key in self.sp_keys_valid for key in list(dict_to_use.keys()))
                == True
            ), "All keys in criteria_dict must be in project statepoints"
        # Organize Dictionary keys and values sorted from lowest to highest
        sorted_dict = dict(
            sorted(dict_to_use.items(), key=lambda item: (item[0], item[1]))
        )

        # Make list of parts
        parts = []
        for key, value in sorted_dict.items():
            if isinstance(value, dict):
                # Recursively format nested dictionaries
                nested_path = self.make_dir_name_from_criteria(value, True)
                parts.append(f"{key.replace('$', '')}_{nested_path}")
            elif isinstance(value, list):
                # Format lists as a string without square brackets and commas
                list_str = "_".join(map(str, value))
                parts.append(f"{key.replace('$', '')}_{list_str}")
            else:
                parts.append(f"{key.replace('$', '')}_{value}")

        if is_nested:
            result_dir = "/".join(parts)
        else:
            project_name = (
                os.path.basename(self.project.fn("").rstrip("/"))
                if self.project is not None
                else "results"
            )
            result_dir = os.path.join(
                project_name, "Results_" + self.mode, "/".join(parts)
            )
        return result_dir

    def str_to_array_df_col(self, str_arr):
        """
        Converts a DataFrame column value (loaded from CSV, so array-valued columns come
        back as their str(np.ndarray) repr) back into a numeric np.ndarray.

        Parameters
        ----------
        str_arr: str, list, or np.ndarray
            The column value to convert. If not a str, it is passed through np.array()
            unchanged (list/np.ndarray inputs occur when the DataFrame was not round
            tripped through CSV).

        Returns
        -------
        array_from_str: np.ndarray or float
            The parsed array, or a scalar float if the array has exactly one element
        """
        if isinstance(str_arr, str):
            cleaned_str1 = re.sub(r"\s+", " ", str_arr.strip())
            cleaned_str = re.sub(r"(-?\d+\.\d*|\d+)\s+", r"\1, ", cleaned_str1)
            array_from_str = np.array(ast.literal_eval(f"[{cleaned_str}]"), dtype=float)
        elif isinstance(str_arr, (list)):
            # Return the original value if it isn't a string
            array_from_str = np.array(str_arr, dtype=float)
        else:
            array_from_str = np.array(str_arr, dtype=float)
        if len(array_from_str) == 1:
            array_from_str = array_from_str[0]

        return array_from_str

    def get_run_dataframe(self, job, save_csv=None):
        """
        Get best data from jobs and optionally save the csvs for the data

        Parameters
        ----------
        job: signac.job.Job
            The job to get data from
        save_csv: bool, default None
            Whether to save csvs. Set to the class default if None.

        Returns
        -------
        df_job: pd.DataFrame
            The dataframe of the data for the given job
        theta_true_data_w_bnds: tuple(dict, np.ndarray)
            Tuple of a dictionary of true parameter values and bounds for the parameters

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        Notes
        -----
        If save_csv is None, it is set to the class default
        """
        assert is_job_like(job), "job must be a signac job or a JobContext"
        assert save_csv == None or isinstance(
            save_csv, bool
        ), "save_csv must be a boolean or None"
        save_csv = self.save_csv if save_csv == None else save_csv
        # Initialize df for a single job
        df_job = pd.DataFrame()
        data_file = job.fn("BO_Results.gz")

        # Open the file and get the dataframe
        results = load_gz(data_file)

        # Get statepoint info
        with open(job.fn("signac_statepoint.json"), "r") as json_file:
            # Load the JSON data
            sp_data = json.load(json_file)

        # Find number of workflow restarts in that job
        tot_runs = results[0].configuration["Number of Workflow Restarts"]
        num_x_exp = results[0].exp_data_class.n_x
        # get theta_true from 1st run since it never changes within a case study
        theta_true = results[0].simulator_class.theta_true
        theta_true_names = results[0].simulator_class.theta_true_names
        theta_true_bnds = results[0].simulator_class.bounds_theta_reg
        theta_true_data = dict(zip(theta_true_names, theta_true))

        # Loop over runs in each job
        for run in range(tot_runs):
            # Read data as pd.df
            df_run = results[run].results_df
            # Add the EP enum value as a column
            col_vals = job.sp.ep_enum_val
            df_run["ep_method_val"] = EpSchedule(int(col_vals)).name
            # Set index as the first run in the job's run number + the run we're at in the job
            df_run["index"] = int(job.sp.bo_run_num + run)
            df_run["job_id"] = job.id

            # Set Run numbers as columns
            df_run.rename(columns={"index": "run_number"}, inplace=True)

            # Add run dataframe to job dataframe after
            df_job = pd.concat([df_job, df_run], ignore_index=False)

        # Reset index on job dataframe
        df_job = df_job.reset_index(drop=True)
        # Add cs name data to the dataframe
        df_job["cs_name_val"] = job.sp.cs_name_val
        df_job["cs_name"] = get_cs_class_from_val(job.sp.cs_name_val).name

        if save_csv:
            all_data_path = os.path.join(job.fn("analysis_data"), "tabulated_data.csv")
            theta_data_path = os.path.join(
                job.fn("analysis_data"), "true_param_data.json"
            )
            theta_bnds_path = os.path.join(
                job.fn("analysis_data"), "true_bnds_data.pkl"
            )
            self.save_data(df_job, all_data_path)
            self.save_data(theta_true_data, theta_data_path)
            self.save_data(theta_true_bnds, theta_bnds_path)

        true_data_w_bnds = (theta_true_data, theta_true_bnds)

        return df_job, true_data_w_bnds

    def load_data(self, path):
        """
        Loads data from a file based on the file extension

        Parameters
        ----------
        path: str, The file path of the data

        Returns
        -------
        found_data: bool
            Whether the data was found
        data: np.ndarray or pd.DataFrame or None
            The data from the file or None

        Raises
        ------
        ValueError: If the file type is not .gz, .pkl, .pickle, .npy, .csv, or .json
        """
        assert isinstance(path, str), "path_end must be str"
        # Split path into parts
        ext = os.path.splitext(path)[-1]
        assert ext in [
            ".csv",
            ".npy",
            ".pkl",
            ".pickle",
            ".gz",
            ".json",
        ], "File type not supported"
        # Based on extension, load in different ways
        # Check if csv already exists
        if os.path.exists(path):
            # If so, load the file
            if ext == ".csv":
                data = pd.read_csv(path, index_col=0)
            elif ext == ".npy":
                data = np.load(path, allow_pickle=True)
            elif ext == ".pkl" or ext == ".gz" or ext == ".pickle":
                data = load_gz(path)
            elif ext == ".json":
                with open(path, "r") as file:
                    data = json.load(file)
            else:
                raise ValueError("NOT a csv, json, npy, pkl, pickle, or gz file")
            return True, data
        else:
            return False, None

    def save_data(self, data, save_path):
        """
        Saves data to a file based on the file extension

        Parameters
        ----------
        data: Object
            The data to save
        save_path: str
            The file path to save the data

        Raises
        ------
        ValueError: If the file type is not .gz, .pkl, .pickle, .npy, .csv, or .json
        """
        assert isinstance(save_path, str), "path_end must be str"
        # Split path into parts
        ext = os.path.splitext(save_path)[-1]
        assert ext in [
            ".csv",
            ".npy",
            ".pkl",
            ".pickle",
            ".gz",
            ".json",
        ], "File type not supported"
        # Extract directory name
        dirname = os.path.dirname(save_path)
        # Make directory if it doesn't already exist
        os.makedirs(dirname, exist_ok=True)
        # Based on extension, save in different ways
        if ext == ".csv":
            data.to_csv(save_path)
        elif ext == ".npy":
            np.save(save_path, data)
        elif ext == ".json":
            with open(save_path, "w") as file:
                json.dump(data, file)
        elif ext == ".gz":
            with gzip.open(save_path, "wb", compresslevel=1) as file:
                data = pickle.dump(data, file)
        elif ext == ".pkl" or ext == ".pickle":
            with open(save_path, "wb") as file:
                data = pickle.dump(data, file)
        else:
            raise ValueError("NOT a csv, json, npy, pkl, pickle, or gz file")
        return

    def __z_choice_helper(self, z_choices, theta_true_data, data_type):
        """
        creates column and data names based on data type

        Parameters
        ----------
        z_choices: list(str)
            The choices of data to analyze
        theta_true_data: tuple(dict, np.ndarray)
            Tuples of a dictionary of true parameter values and bounds for the parameters
        data_type: str
            The type of data to analyze (parameter or objective data). Either 'objs' or 'params'.

        Returns
        -------
        col_name: list(str)
            The column names for the data
        data_names: list(str)
            The names of the data

        Raises
        ------
        AssertionError
            If the z_choices are not of the correct type

        """
        if self.mode == "act":
            obj_col_sse = "sse_actual"
            obj_col_sse_min = "best_sse_actual"
            param_sse = "theta_at_min"
            param_sse_min = "theta_best_actual"
        elif self.mode == "acq":
            obj_col_sse = "sse_at_acq"
            obj_col_sse_min = "best_sse_at_acq"
            param_sse = "theta_at_acq"
            param_sse_min = "theta_best_at_acq"
        elif self.mode == "gp":
            obj_col_sse = "sse_gp"
            obj_col_sse_min = "best_sse_gp"
            param_sse = "theta_at_min"
            param_sse_min = "theta_best_gp"

        if data_type == "objs":
            assert isinstance(z_choices, list), "z_choices must be list of string."
            assert all(
                isinstance(item, str) for item in z_choices
            ), "z_choices elements must be string"
            assert any(
                item in z_choices for item in ["acq", "min_sse", "sse"]
            ), "z_choices must contain at least 'min_sse', 'acq', or 'sse'"
            col_name = []
            data_names = []

            if self.mode == "gp":
                label_g = "\\tilde{\mathscr{L}}(\mathbf{"
            else:
                label_g = "\mathscr{L}(\mathbf{"

            for z_choice in z_choices:
                if "sse" == z_choice:
                    theta = "\\theta}^o" if self.mode != "acq" else "\\theta^*}"
                    col_name += [obj_col_sse]
                    data_names += [label_g + theta + ")"]
                if "min_sse" == z_choice:
                    theta = "\\theta}^{\prime}"
                    col_name += [obj_col_sse_min]
                    data_names += [label_g + theta + ")"]
                if "acq" == z_choice:
                    col_name += ["acq_value"]
                    data_names += ["\Xi(\mathbf{\\theta^*})"]

        elif data_type == "params":
            assert isinstance(z_choices, str), "z_choices must be a string"
            assert any(
                item == z_choices for item in ["acq", "min_sse", "sse"]
            ), "z_choices must be one of 'min_sse', 'acq', or 'sse'"
            data_names = list(theta_true_data.keys())
            if "min_sse" in z_choices:
                col_name = param_sse_min
            elif "sse" == z_choices:
                col_name = param_sse
            elif "acq" in z_choices:
                col_name = "theta_at_acq"
            else:
                warnings.warn("z_choices must be 'acq', 'sse', or 'min_sse'.")
        return col_name, data_names

    def best_error(self, job):
        """
        Gets the best (lowest) error and its corresponding parameter set for each restart
        of a job, loading from a per-job CSV cache if present (or computing and caching it
        from the pickled BO_Results_GPs.gz otherwise).

        Parameters
        ----------
        job: signac.job.Job
            The job to analyze

        Returns
        -------
        be_list: np.ndarray or None
            The best error for each restart, or None if BO_Results_GPs.gz doesn't exist
        be_theta_list: np.ndarray or None
            The parameter set at the best error for each restart, or None if
            BO_Results_GPs.gz doesn't exist
        """
        # Look for be data for job
        tab_data_path1 = os.path.join(job.fn("analysis_data"), "init_be_data.csv")
        tab_data_path2 = os.path.join(job.fn("analysis_data"), "init_be_theta_data.csv")

        found_data1, be_list = self.load_data(tab_data_path1)
        found_data2, be_theta_list = self.load_data(tab_data_path2)

        if not found_data1 or not found_data2 or self.save_csv:
            if os.path.exists(job.fn("BO_Results_GPs.gz")):
                smallest_file = job.fn("BO_Results_GPs.gz")
                # Open the statepoint of the job
                with open(job.fn("signac_statepoint.json"), "r") as json_file:
                    # Load the JSON data
                    sp_data = json.load(json_file)
                method = GPBOMethod(MethodName(sp_data["meth_name_val"]))
                # Open the smallest data file and pull the smallest sse_val from the training data
                results_GPs = load_gz(smallest_file)
                results = load_gz(job.fn("BO_Results.gz"))
                exp_data = results[
                    0
                ].exp_data_class  # Exp Data will not change between runs

                be_list = []
                be_theta_list = []
                for result in results_GPs:
                    gp_emulator = result.list_gp_emulator_class[0]
                    if method.is_emulator == False:
                        # Type 1 best error is inferred from training data
                        best_error, be_theta, train_idx = gp_emulator.calc_best_error()
                    else:
                        # Type 2 best error must be calculated given the experimental data
                        best_error, be_theta, best_errors_x, train_idx = (
                            gp_emulator.calc_best_error(method, exp_data)
                        )

                    if (
                        sp_data["meth_name_val"] == 2
                    ):  # or ( sp_data["meth_name_val"] == 4 and sp_data["cs_name_val"] not in [15, 16, 17]):
                        best_error = np.exp(best_error)
                    be_list.append(best_error)
                    be_theta_list.append(be_theta)
                be_list = np.array(be_list)
                be_theta_list = np.array(be_theta_list)
                df_be = pd.DataFrame(be_list, columns=["best_error"])
                df_be_theta = pd.DataFrame(be_theta_list)

                # Save the data
                self.save_data(df_be, tab_data_path1)
                self.save_data(df_be_theta, tab_data_path2)
            else:
                be_list = None
                be_theta_list = None
        else:
            be_list = be_list.to_numpy()
            be_theta_list = be_theta_list.to_numpy()

        return be_list, be_theta_list

    def __preprocess_analyze(self, job, z_choice, data_type):
        """
        Preprocesses data for analysis based on data type

        Parameters
        ----------
        job: signac.job.Job
            The job to analyze
        z_choice: list(str) or str
            The choices of data to analyze. One of 'min_sse', 'sse', or 'acq'
        data_type: str
            The type of data to analyze (parameter or objective data). Either 'objs' or 'params'.

        Returns
        -------
        df_job: pd.DataFrame
            The dataframe of the data for the given job
        data: np.ndarray
            The data for plotting
        data_true: np.ndarray or None
            The reference values of the data
        sp_data: dict
            The statepoint data for the job
        tot_runs: int
            The total number of runs in the job
        data_median: np.ndarray or None
            The median values of the reference data
        """
        # Look for data if it already exists, if not create it
        # Check if we have theta data and create it if not
        tab_data_path = os.path.join(job.fn("analysis_data"), "tabulated_data.csv")
        true_param_data_path = os.path.join(
            job.fn("analysis_data"), "true_param_data.json"
        )
        found_data1, df_job = self.load_data(tab_data_path)
        found_data2, theta_true_data = self.load_data(true_param_data_path)
        data_median = None
        if not found_data1 or not found_data2:
            df_job, theta_true_data_w_bnds = self.get_run_dataframe(
                job, save_csv=False
            )
            theta_true_data = theta_true_data_w_bnds[0]
        elif found_data1:
            columns_to_convert = [
                "theta_at_acq",
                "theta_at_min",
                "theta_best_gp",
                "theta_best_actual",
                "theta_best_at_acq",
            ]
            for col in columns_to_convert:
                df_job[col] = df_job[col].apply(self.str_to_array_df_col)
        # Get statepoint info
        with open(job.fn("signac_statepoint.json"), "r") as json_file:
            # Load the JSON data
            sp_data = json.load(json_file)
            tot_runs = sp_data["bo_runs_in_job"]
            max_iters = sp_data["bo_iter_tot"]

        if data_type == "objs":
            # No least-squares reference line signac-free: the NLS/derivative-free baseline
            # comparison lives in the archive repo, not here (see analysis module docstring).
            data_true = None
            data_median = None
            data = np.zeros((tot_runs, max_iters, len(z_choice)))

        elif data_type == "params":
            data_true = theta_true_data
            data = np.zeros((tot_runs, max_iters, len(list(theta_true_data.keys()))))

        # Sort df_job by run and iter
        df_job = df_job.sort_values(by=["run_number", "bo_iter"], ascending=True)

        return df_job, data, data_true, sp_data, tot_runs, data_median

    def objective_trajectories(self, job, z_choices):
        """
        Gets the data into an array for for plotting any comination of sse, log_sse, and ei

        Parameters
        ----------
        job: signac.job.Job
            The job to analyze
        z_choices: list(str) or str
            The choices of data to analyze. Contains a combination of 'min_sse', 'sse', or 'acq'

        Returns
        -------
        data: np.ndarray
            The data for plotting
        data_names: list(str)
            The names of the data
        data_true: dict or None
            The reference values of the data
        sp_data: dict
            The statepoint data for the job
        data_true_med: dict or None
            The median values of the reference data

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert is_job_like(job), "job must be a signac job or a JobContext"
        assert isinstance(
            z_choices, (Iterable, str)
        ), "z_choices must be Iterable or str"
        if isinstance(z_choices, str):
            z_choices = [z_choices]
        assert all(
            isinstance(item, str) for item in z_choices
        ), "z_choices elements must be str"
        for i in range(len(z_choices)):
            assert z_choices[i] in [
                "min_sse",
                "sse",
                "acq",
            ], "z_choices items must be 'min_sse', 'sse', or 'acq'"

        df_job, data, data_true_val, sp_data, tot_runs, data_true_med_val = (
            self.__preprocess_analyze(job, z_choices, "objs")
        )
        data_true = {}
        data_true_med = {}
        col_name, data_names = self.__z_choice_helper(z_choices, data_true, "objs")

        unique_run_nums = pd.unique(df_job["run_number"])
        # Loop over each choice
        for z in range(len(z_choices)):
            # Loop over runs
            for i, run in enumerate(unique_run_nums):
                # Make a df of only the data which meets that run criteria
                df_run = df_job[df_job["run_number"] == run]
                z_data = df_run[col_name[z]]
                # If sse in log choices, the "true data" is sse data from least squares
                if "sse" in z_choices[z]:
                    data_true[z_choices[z]] = data_true_val
                    data_true_med[z_choices[z]] = data_true_med_val
                    # If the z_choice is sse and the method has a log objective function value, un logscale data
                    # if sp_data["meth_name_val"] in [2, 4]:
                    if (
                        sp_data["meth_name_val"] == 2
                    ):  # or ( sp_data["meth_name_val"] == 4 and sp_data["cs_name_val"] not in [15, 16, 17]):
                        z_data = np.exp(z_data.values.astype(float))
                else:
                    data_true[z_choices[z]] = None
                    data_true_med[z_choices[z]] = None
                # Set data to be where it needs to go in the above data matrix
                data[i, : len(z_data), z] = z_data

        return data, data_names, data_true, sp_data, data_true_med

    def parameter_trajectories(self, job, z_choice):
        """
        Gets the data into an array for for plotting parameter values

        Parameters
        ----------
        job: signac.job.Job
            The job to analyze
        z_choice: str
            The choice of data to analyze. One of 'min_sse', 'sse', or 'acq'

        Returns
        -------
        data: np.ndarray
            The data for plotting
        data_names: list(str)
            The names of the data
        data_true: dict or None
            The reference values of the data
        sp_data: dict
            The statepoint data for the job

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert is_job_like(job), "job must be a signac job or a JobContext"
        assert isinstance(z_choice, (str)), "z_choice must be a str"
        assert z_choice in [
            "min_sse",
            "sse",
            "acq",
        ], "z_choice must be 'min_sse', 'sse', or 'acq'"

        df_job, data, data_true, sp_data, tot_runs, data_true_med = (
            self.__preprocess_analyze(job, z_choice, "params")
        )
        col_name, data_names = self.__z_choice_helper(z_choice, data_true, "params")
        # Loop over runs
        unique_run_nums = pd.unique(df_job["run_number"])
        for i, run in enumerate(unique_run_nums):
            # Make a df of only the data which meets that run criteria
            df_run = df_job[df_job["run_number"] == run]
            df_run_arry = np.array(
                [arr.tolist() for arr in df_run[col_name].to_numpy()]
            )
            for param in range(data.shape[-1]):
                z_data = df_run_arry[:, param]
                # Set data to be where it needs to go in the above data matrix
                data[i, : len(z_data), param] = z_data

        data_names = [element.replace("theta", "\\theta") for element in data_names]

        return data, data_names, data_true, sp_data

    def hyperparameter_trajectories(self, job):
        """
        Gets the data into an array for for plotting hyperparameters

        Parameters
        ----------
        job: signac.job.Job
            The job to analyze

        Returns
        -------
        data: np.ndarray
            The data for plotting
        data_names: list(str)
            The names of the data
        data_true: None
            The reference values of the data (None for hyperaprameters)
        sp_data: dict
            The statepoint data for the job

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        assert is_job_like(job), "job must be a signac job or a JobContext"
        data_true = None
        # Check for prexisting data
        hp_data_path = os.path.join(job.fn("analysis_data"), "hyperparam_data.npy")
        hp_name_path = os.path.join(job.fn("analysis_data"), "hp_name_data.json")
        found_data1, data = self.load_data(hp_data_path)
        found_data2, data_names = self.load_data(hp_name_path)

        # Get statepoint info
        with open(job.fn("signac_statepoint.json"), "r") as json_file:
            # Load the JSON data
            sp_data = json.load(json_file)
            tot_runs = sp_data["bo_runs_in_job"]
            max_iters = sp_data["bo_iter_tot"]

        if self.save_csv or (not found_data1 and not found_data2):
            loaded_results = load_gz(job.fn("BO_Results_GPs.gz"))
            dim_hps = (
                len(loaded_results[0].list_gp_emulator_class[0].trained_hyperparams[0])
                + 2
            )
            data = np.zeros((tot_runs, max_iters, dim_hps))
            data_names = [f"\\ell_{i}" for i in range(1, dim_hps + 1)]
            data_names[-2] = "\sigma"
            data_names[-1] = "\\tau"

            for j in range(tot_runs):
                run = loaded_results[j]
                for i in range(len(run.list_gp_emulator_class)):
                    # Extract the array and convert other elements to float
                    array_part = run.list_gp_emulator_class[i].trained_hyperparams[0]
                    rest_part = np.array(
                        run.list_gp_emulator_class[i].trained_hyperparams[1:],
                        dtype=float,
                    )
                    hp = np.concatenate([array_part, rest_part])
                    # Create the resulting array of shape (1, 10)
                    data[j, i, :] = hp

            if self.save_csv:
                self.save_data(data, hp_data_path)
                self.save_data(data_names, hp_name_path)

        return data, data_names, data_true, sp_data

    def __rebuild_cs(self, sp_data):
        """
        builds instance of BOConfig from saved file data

        Parameters
        ----------
        sp_data: dict
            The statepoint data for the job

        Returns
        -------
        cs_params: BOConfig
            The case study parameters
        method: GPBOMethod
            The method used
        gen_meth_theta: GenMethod
            The method used to generate theta values
        ep_enum: EpSchedule
            The method used to generate exploration bias values
        """
        method = GPBOMethod(MethodName(sp_data["meth_name_val"]))
        cs_name = (
            get_cs_class_from_val(sp_data["cs_name_val"]).name
            if "cs_name_val" in sp_data
            else "New_CS"
        )
        ep0 = sp_data["ep0"]
        sep_fact = sp_data["sep_fact"]
        normalize = sp_data["normalize"]
        kernel = Kernel(sp_data["kernel_enum_val"])
        lenscl = sp_data["lenscl"]
        outputscl = sp_data["outputscl"]
        retrain_gp = sp_data["retrain_gp"]
        reoptimize_obj = sp_data["reoptimize_obj"]
        gen_heat_map_data = sp_data["gen_heat_map_data"]
        bo_iter_tot = sp_data["bo_iter_tot"]
        bo_run_tot = sp_data["bo_run_tot"]
        save_data = False
        created_at = None
        seed = sp_data["seed"]
        obj_tol = sp_data["obj_tol"]
        ei_tol = sp_data["ei_tol"]
        gen_meth_theta = GenMethod(sp_data["gen_meth_theta"])
        ep_enum = EpSchedule(sp_data["ep_enum_val"])

        cs_params = BOConfig(
            cs_name,
            ep0,
            sep_fact,
            normalize,
            kernel,
            lenscl,
            outputscl,
            retrain_gp,
            reoptimize_obj,
            gen_heat_map_data,
            bo_iter_tot,
            bo_run_tot,
            save_data,
            created_at,
            seed,
            obj_tol,
            ei_tol,
        )

        return cs_params, method, gen_meth_theta, ep_enum

    def gp_parity_data(self, job, run_num, bo_iter):
        """
        Generates parity plot for testing data

        Parameters
        ----------
        job: signac.job.Job
            The job to analyze
        run_num: int
            The run number to analyze
        bo_iter: int
            The bo iteration to analyze

        Returns
        -------
        test_data_obj: Data
            The evaluated testing data for the given run and iteration

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
            If the run_num or bo_iter are out of bounds
        """
        assert is_job_like(job), "job must be a signac job or a JobContext"
        assert isinstance(run_num, (np.int64, int)), "run_num must be an int"
        assert isinstance(bo_iter, (np.int64, int)), "bo_iter must be an int"
        # Get Best Data
        # Check if data exists, if so, load it
        # Assert that heat map data does not aleady exist
        dir_name = os.path.join(
            job.fn(""),
            "analysis_data",
            "gp_evaluations",
            "run_" + str(run_num),
            "iter_" + str(bo_iter),
        )
        data_name = os.path.join(dir_name, "test_data.pkl")
        found_data1, test_data_obj = self.load_data(data_name)

        # Get statepoint_info
        # Get statepoint info
        with open(job.fn("signac_statepoint.json"), "r") as json_file:
            # Load the JSON data
            sp_data = json.load(json_file)
        bo_runs_in_job = sp_data["bo_runs_in_job"]
        bo_run_num_int = sp_data["bo_run_num"]
        run_idx = run_num - bo_run_num_int
        meth_name_val = sp_data["meth_name_val"]
        meth_name = MethodName(meth_name_val)
        method = GPBOMethod(meth_name)

        # Otherwise Generate it
        if self.save_csv or not found_data1:
            # Open file
            results = load_gz(job.fn("BO_Results.gz"))
            results_GP = load_gz(job.fn("BO_Results_GPs.gz"))
            assert len(results) > run_idx, "run_num is out of bounds"
            assert (
                len(results_GP[run_idx].list_gp_emulator_class) > bo_iter - 1
            ), "bo_iter is out of bounds"
            gp_object = copy.copy(
                results_GP[run_idx].list_gp_emulator_class[bo_iter - 1]
            )
            simulator = copy.copy(results[run_idx].simulator_class)
            if hasattr(simulator, "indeces_to_consider"):
                simulator.indices_to_consider = (
                    simulator.indeces_to_consider
                )  # For backwards compatibility
            exp_data = copy.copy(
                results[0].exp_data_class
            )  # Experimental data won't change

            # Get testing data if it doesn't exist
            if gp_object.test_data is None or len(gp_object.test_data.theta_vals) == 0:
                # Generate testing data if it doesn't exist
                # Get 10 num_theta points for testing
                num_x = exp_data.n_x
                dim_x = exp_data.x_dim
                use_x = int(num_x ** (1 / dim_x))
                # Make Data (For multi vs 1D X data)
                # For conventional methods, must use same x for testing data as exp values to get same results
                test_data_sim = simulator.generate_simulation_data(
                    10,
                    use_x,
                    GenMethod(1),
                    GenMethod(2),
                    1.0,
                    simulator.val_seed,
                    False,
                    x_vals=exp_data.x_vals,
                )

                if method.is_emulator == False:
                    test_data_sim = simulator.to_sse_data(
                        method, test_data_sim, exp_data, 1.0, False
                    )
                gp_object.test_data = test_data_sim
                gp_object.feature_test_data = gp_object.featurize_data(
                    gp_object.test_data
                )
                # This is gp_parity_data's legitimate output: predict() no longer mutates
                # gp_mean/gp_var/gp_covar onto Data itself, so attach all three explicitly
                # from the returned GPPrediction (previously .gp_covar came from predict()'s
                # internal write, which this attachment alone didn't cover).
                test_prediction = gp_object.predict(target="test")
                gp_object.test_data.gp_mean = test_prediction.mean
                gp_object.test_data.gp_var = test_prediction.variance
                gp_object.test_data.gp_covar = test_prediction.covariance

            test_data_obj = gp_object.test_data

            if self.save_csv:
                self.save_data(test_data_obj, data_name)

        return test_data_obj

    def gp_heat_map_data(self, job, run_num, bo_iter, pair_id, get_ei=False):
        """
        Generates/analyzes heat map data for the given run and iteration

        Parameters
        ----------
        job: signac.job.Job
            The job to analyze
        run_num: int
            The run number to analyze
        bo_iter: int
            The bo iteration to analyze
        pair_id: int or str
            The pair of parameters to analyze
        get_ei: bool, default False
            Whether to calculate the acquisition function

        Returns
        -------
        all_data: np.ndarray
            The data for plotting
        test_mesh: np.ndarray
            The meshgrid for the testing data
        param_info_dict: dict
            The parameter information for the given pair
        sp_data: dict
            The statepoint data for the job

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
            If the run_num or bo_iter are out of bounds
        ValueError
            pair_id is out of bounds or an invalid string
        """
        assert is_job_like(job), "job must be a signac job or a JobContext"
        assert isinstance(run_num, (np.int64, int)), "run_num must be an int"
        assert isinstance(bo_iter, (np.int64, int)), "bo_iter must be an int"
        assert isinstance(
            pair_id, (np.int64, int, str)
        ), "pair_id must be an int or str"
        assert isinstance(get_ei, bool), "get_ei must be a bool"

        # Assert that heat map data does not aleady exist
        dir_name = os.path.join(
            job.fn(""),
            "analysis_data",
            "gp_evaluations",
            "run_" + str(run_num),
            "iter_" + str(bo_iter),
            "pair_" + str(pair_id),
        )
        hm_path_name = os.path.join(dir_name, "hm_data.gz")
        hm_sse_path_name = os.path.join(dir_name, "hm_sse_data.gz")
        param_info_path = os.path.join(dir_name, "notable_param_info.pkl")
        try:
            found_data1, heat_map_data = self.load_data(hm_path_name)
        except:
            found_data1, heat_map_data = False, None
        found_data2, heat_map_sse_data = self.load_data(hm_sse_path_name)
        found_data3, param_info_dict = self.load_data(param_info_path)

        # Get statepoint info
        with open(job.fn("signac_statepoint.json"), "r") as json_file:
            # Load the JSON data
            sp_data = json.load(json_file)

        # Set run number to the index of the run number in the job by subtracting
        # The number of the first run in the job
        run_num -= sp_data["bo_run_num"]
        bo_iter -= 1
        cs_params, method, gen_meth_theta, ep_method = self.__rebuild_cs(sp_data)

        data_not_found = not found_data1 or not found_data2 or not found_data3
        # Initialize data_needs_ei as true
        data_needs_ei = True
        # If we don't need acq data, set data_needs_ei to False
        if get_ei == False:
            data_needs_ei = False
        # If we have all the data and we need to calculate acq, check if we have acq data
        elif not data_not_found and not self.save_csv and get_ei:
            # If we have all the data, we won't need to calculate ei
            if heat_map_sse_data.acq is not None:
                data_needs_ei = False

        # Generate driver class/ emulator class if data doesn't exist or we need to calculate acq
        if self.save_csv or data_not_found or data_needs_ei:
            loaded_results = load_gz(job.fn("BO_Results.gz"))
            loaded_results_GPs = load_gz(job.fn("BO_Results_GPs.gz"))
            assert len(loaded_results_GPs) > run_num, "run_num is out of bounds"
            assert (
                len(loaded_results_GPs[run_num].list_gp_emulator_class) > bo_iter
            ), "bo_iter is out of bounds"

            # Create Heat Map Data for a run and iter
            # Regeneate class objects
            gp_emulator = loaded_results_GPs[run_num].list_gp_emulator_class[bo_iter]
            exp_data = loaded_results[run_num].exp_data_class
            simulator = loaded_results[run_num].simulator_class
            if hasattr(simulator, "indeces_to_consider"):
                simulator.indices_to_consider = (
                    simulator.indeces_to_consider
                )  # For backwards compatibility
            ep_at_iter = (
                loaded_results[run_num].results_df["alpha"].iloc[bo_iter]
            )
            ep_bias = ExplorationBias(
                None, ep_at_iter, ep_method, None, None, None, None, None, None, None
            )
            driver = GPBODriver(
                cs_params,
                method,
                simulator,
                exp_data,
                gp_emulator.gp_sim_data,
                gp_emulator.gp_sim_data,
                gp_emulator.gp_val_data,
                gp_emulator.gp_val_data,
                gp_emulator,
                ep_bias,
                gen_meth_theta,
            )
            driver.reset_rng()

            # Get best error metrics (name-mangled private; class is GPBODriver so the
            # mangled attribute is _GPBODriver__get_best_error -- the old _GPBO_Driver__
            # spelling was stale after the class rename and raised AttributeError here).
            be_data, best_error_metrics = driver._GPBODriver__get_best_error()

        # Create heat map data if it doesn't exists
        hm_prediction = None
        sse_prediction = None
        if self.save_csv or data_not_found:
            if self.mode == "act":
                param_sse_min = "theta_best_actual"
            elif self.mode == "acq":
                param_sse_min = "theta_best_at_acq"
            elif self.mode == "gp":
                param_sse_min = "theta_best_gp"

            # Get important theta values
            theta_true = loaded_results[run_num].simulator_class.theta_true
            theta_opt = loaded_results[run_num].results_df[param_sse_min].iloc[bo_iter]
            theta_next = (
                loaded_results[run_num].results_df["theta_at_acq"].iloc[bo_iter]
            )
            train_theta = (
                loaded_results_GPs[run_num]
                .list_gp_emulator_class[bo_iter]
                .train_data.theta_vals
            )

            # Get specific heat map data or generate it
            num_x = exp_data.n_x
            n_points_set = len(driver.gp_emulator.gp_sim_data.get_unique_theta())
            if num_x * n_points_set**2 >= 5000:
                n_points_set = int(np.sqrt(5000 / num_x))
            loaded_results_GPs[0].heat_map_data_dict = (
                driver.create_heat_map_param_data(n_points_set)
            )
            heat_map_data_dict = loaded_results_GPs[0].heat_map_data_dict

            # Get pair ID
            if isinstance(pair_id, str):
                assert (
                    pair_id in loaded_results_GPs[0].heat_map_data_dict.keys()
                ), "pair_id is an invalid string"
                param_names = pair_id
            elif isinstance(pair_id, int):
                assert pair_id < len(
                    loaded_results_GPs[0].heat_map_data_dict.keys()
                ), "pair_id is out of bounds"
                param_names = list(loaded_results_GPs[0].heat_map_data_dict.keys())[
                    pair_id
                ]
            else:
                raise ValueError("Invalid pair_id!")

            # Initialize heat map data class
            heat_map_data_org = heat_map_data_dict[param_names]

            # Calculate GP mean and var for heat map data
            featurized_hm_data = gp_emulator.featurize_data(heat_map_data_org)
            try:
                hm_prediction = gp_emulator.predict(
                    data=heat_map_data_org, featurized_data=featurized_hm_data
                )
                hm_org_mean, hm_org_var = hm_prediction
            except:
                print(n_points_set)

            # Get index of param set and best error
            idcs_to_plot = [
                loaded_results[run_num].simulator_class.theta_true_names.index(name)
                for name in param_names
            ]

            # Set param info
            param_info_dict = {
                "true": theta_true,
                "min_sse": theta_opt,
                "opt_acq": theta_next,
                "train": train_theta,
                "names": param_names,
                "idcs": idcs_to_plot,
            }

            # If the emulator is a conventional method, create heat map data in emulator form to calculate y_vals
            if not method.is_emulator:
                # Make surrogate heat map data for full theta and x grid to calculate y_vals
                n_points = int(np.sqrt(heat_map_data_org.n_theta))
                repeat_x = n_points**2  # Square because only 2 values at a time change
                x_vals = np.vstack(
                    [exp_data.x_vals] * repeat_x
                )  # Repeat x_vals n_points**2 number of times
                repeat_theta = (
                    exp_data.n_x
                )  # Repeat theta len(x) number of times
                theta_vals = np.repeat(
                    heat_map_data_org.theta_vals, repeat_theta, axis=0
                )  # Create theta data repeated
                heat_map_data = Data(
                    theta_vals,
                    x_vals,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    simulator.bounds_theta_reg,
                    simulator.bounds_x,
                    cs_params.sep_fact,
                )
            else:
                heat_map_data = heat_map_data_org

            # Generate heat map data and sse heat map data sim y values (noiseless)
            heat_map_data.y_vals = simulator.evaluate_model(
                heat_map_data, 0, 0, simulator.rng_set
            )

            # Create sse data from regular y data
            heat_map_sse_data = simulator.to_sse_data(
                method, heat_map_data, exp_data, cs_params.sep_fact, y_to_sse=False
            )
            # Set the mean and variance to the correct heat map data object. gp_covar comes
            # from hm_prediction.covariance (always the full covariance -- predict() computes
            # it unconditionally) rather than re-reading heat_map_data_org.gp_covar off Data.
            if not method.is_emulator:
                heat_map_sse_data.gp_mean = hm_org_mean
                heat_map_sse_data.gp_var = hm_org_var
                heat_map_sse_data.gp_covar = hm_prediction.covariance
            else:
                heat_map_data.gp_mean = hm_org_mean
                heat_map_data.gp_var = hm_org_var
                heat_map_data.gp_covar = hm_prediction.covariance

            # Calculate SSE and SSE var. Reuses `hm_prediction` instead of re-reading
            # data.gp_mean/data.gp_covar (heat_map_sse_data.gp_mean/gp_covar above were set
            # from that same prediction; heat_map_data IS heat_map_data_org in the emulator
            # branch, so the prediction is on the exact object predict_sse operates on).
            if method.is_emulator == False:
                sse_prediction = gp_emulator.predict_sse(data=heat_map_sse_data, prediction=hm_prediction)
                heat_map_sse_data.sse, heat_map_sse_data.sse_var = sse_prediction
            else:
                sse_prediction = gp_emulator.predict_sse(
                    data=heat_map_data, method=method, exp_data=exp_data,
                    prediction=hm_prediction,
                )
                heat_map_sse_data.sse, heat_map_sse_data.sse_var = sse_prediction

        # Resolve sse mean/var: freshly computed above (sse_prediction), or -- if the block
        # above didn't run because the heat map data was already cached and only EI needed
        # recomputing -- already on heat_map_sse_data from the loaded cache
        # (self.load_data(hm_sse_path_name) near the top of this method).
        if sse_prediction is not None:
            sse_mean_val, sse_var_val = sse_prediction.mean, sse_prediction.variance
        else:
            sse_mean_val, sse_var_val = heat_map_sse_data.sse, heat_map_sse_data.sse_var

        # Get EI if needed. This operation can be expensive which is why it's optional
        acq_result = None
        if data_needs_ei:
            if method.method_name.value == 7:
                acq_result = sse_mean_val + np.sum(sse_var_val)
            elif method.is_emulator == False:
                # Reuses `hm_prediction` instead of re-reading data.gp_mean/data.gp_covar
                # (heat_map_sse_data.gp_mean/gp_covar above were set from that same prediction,
                # or -- cache-hit-without-heat-map-recompute -- hm_prediction is None here and
                # expected_improvement falls back to its own data.gp_* read, which in that
                # scenario already holds the cached values).
                acq_result = gp_emulator.expected_improvement(
                    data=heat_map_sse_data, exp_data=exp_data, ep_bias=ep_bias,
                    best_error_metrics=best_error_metrics, gp_prediction=hm_prediction,
                )[0]
            # In older data, sparse grid depth is not a set parameter. Therefore, we set the number of points to 2000
            # This will be irrelevant for non-MC and SG data anyway
            else:
                try:
                    sg_mc_samples = loaded_results[run_num].configuration[
                        "MC SG Max Points"
                    ]
                except:
                    sg_mc_samples = 2000

                # For SG and MC data, we must get the sse mean and covar for each point individually
                ei_vals = []
                for t_val in range(len(heat_map_sse_data.get_unique_theta())):
                    # Create feature data for candidate point
                    theta = heat_map_sse_data.theta_vals[t_val]
                    candidate_theta_vals = np.repeat(
                        theta.reshape(1, -1), exp_data.n_x, axis=0
                    )
                    candidate = Data(
                        None,
                        exp_data.x_vals,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        simulator.bounds_theta_reg,
                        simulator.bounds_x,
                        cs_params.sep_fact,
                    )
                    candidate.theta_vals = candidate_theta_vals
                    gp_emulator.cand_data = candidate
                    # Set candidate point feature data
                    gp_emulator.feature_cand_data = gp_emulator.featurize_data(
                        gp_emulator.cand_data
                    )
                    # Evaluate GP mean/ stdev at theta
                    cand_pred = gp_emulator.predict(target="cand")
                    cand_mean, cand_var = cand_pred
                    # For Type 2 GP, the sse and sse_var are calculated from the gp_mean, gp_var,
                    # and experimental data. Reuses `cand_pred` instead of re-reading
                    # data.gp_mean/data.gp_covar off gp_emulator.cand_data.
                    cand_sse_mean, cand_sse_var = gp_emulator.predict_sse(
                        target="cand", method=method, exp_data=exp_data, prediction=cand_pred
                    )
                    # Otherwise objective is ei. Reuses `cand_pred` instead of re-reading
                    # data.gp_mean/data.gp_covar off gp_emulator.cand_data.
                    ei_output = gp_emulator.expected_improvement(
                        target="cand", exp_data=exp_data, ep_bias=ep_bias, best_error_metrics=best_error_metrics,
                        method=method, sg_mc_samples=sg_mc_samples, gp_prediction=cand_pred,
                    )[0]
                    ei_vals.append(ei_output)

                acq_result = np.array(ei_vals)

            heat_map_sse_data.acq = acq_result

        # Save data if necessary
        if self.save_csv:
            self.save_data(heat_map_data, hm_path_name)
            self.save_data(heat_map_sse_data, hm_sse_path_name)
            self.save_data(param_info_dict, param_info_path)

        # Find the theta_vals in the given Data class to be only the 2D (varying) parts you want to plot
        theta_mesh_vals = heat_map_sse_data.theta_vals[:, param_info_dict["idcs"]]
        # Back out the number of theta points from the hm_sse_data
        theta_pts = int(np.sqrt(len(theta_mesh_vals)))
        # Create test mesh for that specific pair and set it as the new sse data theta vals.
        test_mesh = theta_mesh_vals.reshape(theta_pts, theta_pts, -1).T

        # Define sse_sim, sse_gp_mean, and sse_gp_var, and ei based on whether to report log scaled data
        sse_sim = heat_map_sse_data.y_vals
        sse_var = sse_var_val
        sse_mean = sse_mean_val

        # Reshape data to correct shape and add to list to return
        reshape_list = [sse_sim, sse_mean, sse_var]
        all_data = [var.reshape(theta_pts, theta_pts).T for var in reshape_list]
        if get_ei:  # and heat_map_sse_data.acq is not None
            # acq_result is set above if this call computed it; otherwise (cache hit with acq
            # already present, data_needs_ei False) it's still on heat_map_sse_data from the
            # loaded cache.
            final_acq = acq_result if acq_result is not None else heat_map_sse_data.acq
            try:
                acq_new = copy.deepcopy(final_acq)
                all_data += [acq_new.reshape(theta_pts, theta_pts).T]
            except:
                all_data += [final_acq.reshape(theta_pts, theta_pts).T]
        else:
            all_data += [None]

        return all_data, test_mesh, param_info_dict, sp_data

