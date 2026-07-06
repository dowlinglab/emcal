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
    lengthscale: float, int, np.ndarray, list, or None, default None
        Lengthscale hyperparameter value; None trains it. Lists are stored as arrays.
    outputscale: float, int, or None, default None
        Outputscale value; None trains it. Must be > 0 if given.
    retrain_gp: int, default 25
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
    created_at: str or None, default None
        Run timestamp; None fills in the current date/time.
    seed: int or None, default 1
        RNG seed (int >= 1) or None for random. (Formerly the ``set_seed`` argument.)
    obj_tol: float, default 1e-7
        Objective-difference termination tolerance (rho_1, >= 0).
    acq_tol: float, default 1e-7
        Acquisition-value termination tolerance (rho_2, >= 0).
    compute_y_sse: bool, default False
        Compute the simulated y value when SSE is locally minimized.
    with_noise: bool, default False
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
    lengthscale: object = None
    outputscale: object = None
    retrain_gp: int = 25
    reoptimize_obj: int = 25
    gen_heat_map_data: bool = False
    bo_iter_tot: int = 10
    bo_run_tot: int = 1
    save_data: bool = False
    created_at: object = None
    seed: object = 1
    obj_tol: float = 1e-7
    acq_tol: float = 1e-7
    compute_y_sse: bool = False
    with_noise: bool = False

    def __post_init__(self):
        """
        Validates field types/ranges immediately after dataclass construction.

        Raises
        ------
        AssertionError
            If any field is not of the correct type or value
        """
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
            for var in [self.bo_iter_tot, self.bo_run_tot, self.retrain_gp, self.reoptimize_obj]
        ), "bo_iter_tot, bo_run_tot, retrain_gp, and reoptimize_obj must be int"
        assert self.seed is None or (
            isinstance(self.seed, int) and self.seed >= 1
        ), "seed must be int >= 1 or None"
        assert (
            isinstance(self.outputscale, (float, int)) or self.outputscale is None
        ), "outputscale must be float, int, or None"
        if self.outputscale is not None:
            assert self.outputscale > 0, "outputscale must be > 0 initially if it is not None"

        # Accept a list lengthscale by converting it to an array
        if isinstance(self.lengthscale, list):
            self.lengthscale = np.array(self.lengthscale)

        assert isinstance(self.compute_y_sse, bool), "compute_y_sse must be bool"
        assert isinstance(self.with_noise, bool), "with_noise must be bool"
        assert (
            isinstance(self.lengthscale, (float, int, np.ndarray)) or self.lengthscale is None
        ), "lengthscale must be float, int, np.ndarray, or None"
        if self.lengthscale is not None:
            if isinstance(self.lengthscale, (float, int)):
                assert self.lengthscale > 0, "lengthscale must be > 0 initially if lengthscale is not None"
            else:
                assert all(
                    isinstance(var, (np.int64, np.float64, float, int)) for var in self.lengthscale
                ), "All lengthscale elements must float or int"
                assert all(
                    item > 0 for item in self.lengthscale
                ), "lengthscale elements must be > 0 initially if lengthscale is not None"
        assert (
            0 < self.sep_fact <= 1
        ), "Separation factor must be between 0 and 1. Not including zero"
        assert all(
            var > 0 for var in [self.bo_iter_tot, self.bo_run_tot]
        ), "bo_iter_tot and bo_run_tot must be > 0"
        assert all(
            var >= 0 for var in [self.retrain_gp, self.reoptimize_obj]
        ), "retrain_gp and reoptimize_obj must be >= 0"
        assert (
            isinstance(self.created_at, str) or self.created_at is None
        ), "created_at must be str or None"
        assert (
            isinstance(self.acq_tol, (float, int)) and self.acq_tol >= 0
        ), "acq_tol must be a positive float or integer"
        assert (
            isinstance(self.obj_tol, (float, int)) and self.obj_tol >= 0
        ), "obj_tol must be a positive float or integer"

        # --- derived / normalized values ---
        if self.created_at is None:
            self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(self.cs_name, Enum):
            self.cs_name = self.cs_name.name
        else:
            self.cs_name = str(self.cs_name)
