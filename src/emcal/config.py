"""BOConfig: the Bayesian-optimization run configuration (a validated dataclass;
formerly CaseStudyParameters).
"""
import numpy as np
import warnings
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from .enums import Kernel


@dataclass(eq=False)
class BOConfig:
    """
    Configuration for a Bayesian-optimization run (formerly CaseStudyParameters).

    A plain keyword-friendly dataclass; all fields have defaults and are
    validated in __post_init__.

    Parameters
    ----------
    cs_name: str, default "New_Case_Study"
        Name associated with the case study being evaluated. An Enum is
        accepted and stored as its ``.name``.
    ep0: float or int, default 1.0
        Starting value for the exploration-bias parameter alpha.
    sep_fact: float or int, default 1.0
        Fraction of data used for training (in (0, 1]).
    normalize: bool, default True
        Standardize feature data (sklearn RobustScaler) if True.
    kernel: Kernel, default Kernel(1)
        GP kernel choice.
    lenscl: float, int, np.ndarray, list, or None, default None
        Lengthscale hyperparameter value; None trains it. Lists are stored as arrays.
    outputscl: float, int, or None, default None
        Outputscale value; None trains it. Must be > 0 if given.
    retrain_GP: int, default 25
        Number of GP (re)training rounds (0 = use initial hyperparameters).
    reoptimize_obj: int, default 25
        Number of acquisition/objective reoptimizations (0 = 1 optimization).
    gen_heat_map_data: bool, default False
        Generate validation data for heat maps.
    bo_iter_tot: int, default 10
        Maximum BO iterations per restart (> 0).
    bo_run_tot: int, default 1
        Total number of BO restarts (> 0).
    save_data: bool, default False
        Save EI data for the argmax(EI) theta.
    DateTime: str or None, default None
        Run timestamp; None fills in the current date/time.
    seed: int or None, default 1
        RNG seed (int >= 1) or None for random. (Formerly the ``set_seed`` argument.)
    obj_tol: float, default 1e-7
        Objective-difference termination tolerance (rho_1, >= 0).
    acq_tol: float, default 1e-7
        Acquisition-value termination tolerance (rho_2, >= 0).
    get_y_sse: bool, default False
        Compute the simulated y value when SSE is locally minimized.
    w_noise: bool, default False
        Include noise in the simulation data.

    Raises
    ------
    AssertionError
        If any input is of the wrong type or out of range.
    Warning
        If cs_name is not a string (it is coerced to one).
    """

    cs_name: str = "New_Case_Study"
    ep0: float = 1.0
    sep_fact: float = 1.0
    normalize: bool = True
    kernel: "Kernel" = Kernel(1)
    lenscl: object = None
    outputscl: object = None
    retrain_GP: int = 25
    reoptimize_obj: int = 25
    gen_heat_map_data: bool = False
    bo_iter_tot: int = 10
    bo_run_tot: int = 1
    save_data: bool = False
    DateTime: object = None
    seed: object = 1
    obj_tol: float = 1e-7
    acq_tol: float = 1e-7
    get_y_sse: bool = False
    w_noise: bool = False

    def __post_init__(self):
        # --- type / range validation (fail fast with clear messages) ---
        if not isinstance(self.cs_name, str):
            warnings.warn(
                "cs_name will be converted to string if it is not an instance of CS_name_enum"
            )
        assert isinstance(self.kernel, Enum), "kernel must be type Enum"
        assert all(
            isinstance(var, (float, int)) for var in [self.sep_fact, self.ep0]
        ), "sep_fact and ep0 must be float or int"
        assert all(
            isinstance(var, bool)
            for var in [self.normalize, self.gen_heat_map_data, self.save_data]
        ), "normalize, gen_heat_map_data, and save_data must be bool"
        assert all(
            isinstance(var, int)
            for var in [self.bo_iter_tot, self.bo_run_tot, self.retrain_GP, self.reoptimize_obj]
        ), "bo_iter_tot, bo_run_tot, retrain_GP, and reoptimize_obj must be int"
        assert self.seed is None or (
            isinstance(self.seed, int) and self.seed >= 1
        ), "seed must be int >= 1 or None"
        assert (
            isinstance(self.outputscl, (float, int)) or self.outputscl is None
        ), "outputscl must be float, int, or None"
        if self.outputscl is not None:
            assert self.outputscl > 0, "outputscl must be > 0 initially if it is not None"

        # Accept a list lengthscale by converting it to an array
        if isinstance(self.lenscl, list):
            self.lenscl = np.array(self.lenscl)

        assert isinstance(self.get_y_sse, bool), "get_y_sse must be bool"
        assert isinstance(self.w_noise, bool), "w_noise must be bool"
        assert (
            isinstance(self.lenscl, (float, int, np.ndarray)) or self.lenscl is None
        ), "lenscl must be float, int, np.ndarray, or None"
        if self.lenscl is not None:
            if isinstance(self.lenscl, (float, int)):
                assert self.lenscl > 0, "lenscl must be > 0 initially if lenscl is not None"
            else:
                assert all(
                    isinstance(var, (np.int64, np.float64, float, int)) for var in self.lenscl
                ), "All lenscl elements must float or int"
                assert all(
                    item > 0 for item in self.lenscl
                ), "lenscl elements must be > 0 initially if lenscl is not None"
        assert (
            0 < self.sep_fact <= 1
        ), "Separation factor must be between 0 and 1. Not including zero"
        assert all(
            var > 0 for var in [self.bo_iter_tot, self.bo_run_tot]
        ), "bo_iter_tot and bo_run_tot must be > 0"
        assert all(
            var >= 0 for var in [self.retrain_GP, self.reoptimize_obj]
        ), "retrain_GP and reoptimize_obj must be >= 0"
        assert (
            isinstance(self.DateTime, str) or self.DateTime is None
        ), "DateTime must be str or None"
        assert (
            isinstance(self.acq_tol, (float, int)) and self.acq_tol >= 0
        ), "acq_tol must be a positive float or integer"
        assert (
            isinstance(self.obj_tol, (float, int)) and self.obj_tol >= 0
        ), "obj_tol must be a positive float or integer"

        # --- derived / normalized values ---
        if self.DateTime is None:
            self.DateTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(self.cs_name, Enum):
            self.cs_name = self.cs_name.name
        else:
            self.cs_name = str(self.cs_name)
