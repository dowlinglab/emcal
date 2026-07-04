"""Data: container for parameter (theta) / state-point (x) / objective (SSE) values and
the GP predictions attached to them during a run.
"""
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._utils import vector_to_1D_array


@dataclass
class GPPrediction:
    """Result of a GP prediction: posterior ``mean`` and its ``variance`` / ``covariance``.

    Introduced as part of the Data split (design Q3) to give predictions a typed result
    object instead of mutating them onto a ``Data`` instance. It stays fully
    backward-compatible with the historical ``(mean, var_or_covar)`` tuple return: it is
    iterable and indexable, so ``mean, var = emulator.predict(...)`` (and
    ``mean, covar = emulator.predict(..., covar=True)``) keep working unchanged, while
    ``.mean`` / ``.variance`` / ``.covariance`` provide named access.

    The second element of the legacy tuple was the covariance when ``covar=True`` was
    requested and the marginal variance otherwise; ``as_covar`` records that choice so the
    iteration protocol reproduces it exactly.
    """

    mean: Any
    variance: Any = None
    covariance: Any = None
    as_covar: bool = False

    def _second(self):
        """The legacy second tuple element (covariance if requested, else variance)."""
        return self.covariance if self.as_covar else self.variance

    def __iter__(self):
        yield self.mean
        yield self._second()

    def __getitem__(self, index):
        return (self.mean, self._second())[index]

    def __len__(self):
        return 2


class Data:
    """
    The base class for any Data used in this workflow
    Parameters

    Methods
    --------------
    __init__(*): Constructor method
    __get_unique(all_vals): Gets unique instances of a certain type of data
    get_unique_theta(): Defines the unique parameter sets from self.theta_vals
    get_unique_x(): Defines the unique state point data from self.x_vals
    n_theta (property): Defines the total number of parameter sets (self.theta_vals)
    theta_dim (property): Defines the total dimensions of the parameter sets (self.theta_vals)
    n_x (property): Defines the total number of state points (self.x_vals)
    x_dim (property): Defines the total dimensions of the state points (self.x_vals)
    train_test_idx_split(): Splits data into training and testing data
    """

    # Class variables and attributes

    def __init__(
        self,
        theta_vals,
        x_vals,
        y_vals,
        gp_mean,
        gp_var,
        sse,
        sse_var,
        acq,
        bounds_theta,
        bounds_x,
        sep_fact
    ):
        """
        Parameters
        ----------
        theta_vals: np.ndarray
            The array of parameter sets
        x_vals: np.ndarray
            Experimental state points (x data)
        y_vals: np.ndarray
            Experimental y data
        gp_mean: np.ndarray
            GP mean prediction values associated with theta_vals and x_vals
        gp_var: np.ndarray
            GP variance prediction values associated with theta_vals and x_vals
        sse: np.ndarray
            GP based sum of squared error values associated with theta_vals and x_vals
        sse_var: np.ndarray
            GP based variance of sum of squared error values associated with theta_vals and x_vals
        acq: np.ndarray
            Acquisition function values associated with theta_vals and x_vals
        bounds_theta: np.ndarray
            Bounds of theta
        bounds_x: np.ndarray
            Bounds of x
        sep_fact: float or int
            The separation factor that decides what percentage of data will be training data. Between 0 and 1.

        Raises
        ------
        AssertionError
            If any of the inputs are not of the correct type or value
        """
        list_vars = [theta_vals, x_vals, y_vals, gp_mean, gp_var, sse, acq]
        assert all(
            isinstance(var, np.ndarray) or var is None for var in list_vars
        ), "theta_vals, x_vals, y_vals, gp_mean, gp_var, sse, and ei must be np.ndarray, or None"
        assert (
            isinstance(sep_fact, (float, int)) or sep_fact is None
        ), "Separation factor must be float or int > 0 or None (exp_data)"
        if sep_fact is not None:
            assert (
                0 < sep_fact <= 1
            ), "sep_fact must be > 0 and less than or equal to 1!"
        # Constructor method
        self.theta_vals = theta_vals
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.gp_mean = gp_mean
        self.gp_var = gp_var
        self.gp_covar = None  # This is calculated later
        self.sse = sse
        self.sse_var = sse_var
        self.sse_covar = None  # This is calculated later
        self.acq = acq
        self.bounds_theta = bounds_theta
        self.bounds_x = bounds_x
        self.sep_fact = sep_fact

    def __get_unique(self, all_vals):
        """
        Gets unique instances of a certain type of data

        Parameters
        -----------
        all_vals: np.ndarray
            Array of parameters with duplicates

        Returns
        --------
        unique_vals: np.ndarray
            Array of parameters without duplicates
        """
        # Get unique indecies and use them to get the values
        unique_indexes = np.unique(all_vals, axis=0, return_index=True)[1]
        unique_vals = np.array([all_vals[index] for index in sorted(unique_indexes)])

        return unique_vals

    def get_unique_theta(self):
        """
        Defines the unique parameter sets from self.theta_vals

        Returns
        --------
        unique_theta_vals: np.ndarray
            Array of unique parameter sets

        Raises
        ------
        AssertionError
            If any self.theta_vals is not defined
        """
        assert self.theta_vals is not None, "self.theta_vals must be defined"
        # Get unique indecies and use them to get the values
        unique_theta_vals = self.__get_unique(self.theta_vals)
        return unique_theta_vals

    def get_unique_x(self):
        """
        Defines the unique state point data from self.x_vals

        Returns
        --------
        unique_x_vals: np.ndarray
            Array of unique state points

        Raises
        ------
        AssertionError
            If self.x_vals is not defined
        """
        assert self.x_vals is not None, "self.x_vals must be defined"
        # Get unique indecies and use them to get the values
        unique_x_vals = self.__get_unique(self.x_vals)
        return unique_x_vals

    @property
    def n_theta(self):
        """
        Defines the total number of parameter sets (self.theta_vals)

        Returns
        -------
        num_theta_data: int
            The number of parameter sets (self.theta_vals)

        Raises
        ------
        AssertionError
            If self.theta_vals is not defined
        """
        assert self.theta_vals is not None, "theta_vals must be defined"
        num_theta_data = len(self.theta_vals)

        return num_theta_data

    @property
    def theta_dim(self):
        """
        Defines the total dimensions of the parameter sets (self.theta_vals)

        Returns
        -------
        dim_theta_data: int
            The cardinality of the parameter sets (self.theta_vals)

        Raises
        ------
        AssertionError
            If self.theta_vals is not defined
        """
        assert self.theta_vals is not None, "self.theta_vals must be defined"
        if len(self.theta_vals) == 1:
            theta_vals = self.theta_vals.reshape(1, -1)
        else:
            theta_vals = self.theta_vals

        dim_theta_data = theta_vals.shape[1]

        return dim_theta_data

    @property
    def n_x(self):
        """
        Defines the total number of state point data (self.x_vals)

        Returns
        -------
        num_x_data: int
            The number of state points (self.x_vals)

        Raises
        ------
        AssertionError
            If self.x_vals is not defined
        """
        assert self.x_vals is not None, "self.x_vals must be defined"
        # Length is the number of data
        num_x_data = len(self.x_vals)

        return num_x_data

    @property
    def x_dim(self):
        """
        Defines the total dimensions of state point data (self.x_vals)

        Returns
        -------
        dim_x_data: int
            The cardinality of state point data (self.x_vals)

        Raises
        ------
        AssertionError
            If self.x_vals is not defined
        """
        assert self.x_vals is not None, "x_vals must be defined"
        # Get dim of x data
        dim_x_data = vector_to_1D_array(self.x_vals).shape[1]

        return dim_x_data

    def train_test_idx_split(self, rng_seed = None):
        """
        Splits data indices into training and testing indices

        Returns
        --------
        train_idx: np.ndarray
            The training theta data identifiers
        test_idx: np.ndarray
            The testing theta data identifiers

        Raises
        ------
        AssertionError
            If self.sep_fact or self.theta_vals are not defined

        Notes
        -----
        The training and testing data is split such that the number train_data is always rounded up. Ensures there is always training data

        """
        assert (
            self.sep_fact is not None
        ), "Data must have a separation factor that is not None!"
        assert self.theta_vals is not None, "data must have theta_vals"
        assert isinstance(rng_seed, int) or rng_seed is None, "rng_seed must be int or None"

        if rng_seed is not None:
            rng = np.random.default_rng(rng_seed)
        else:
            rng = np.random.default_rng()

        # Find number of unique thetas and calculate length of training data
        len_theta = len(self.get_unique_theta())
        len_train_idc = int(
            np.ceil(len_theta * self.sep_fact)
        )  # Ensure there will always be at least one training point by using np.ceil

        # Create an index for each theta
        all_idx = np.arange(0, len_theta)

        # Shuffle all_idx data in such a way that theta values will be randomized
        rng.shuffle(all_idx)

        # Set train test indeces
        train_idx = all_idx[:len_train_idc]
        test_idx = all_idx[len_train_idc:]

        return train_idx, test_idx


# ---------------------------------------------------------------------------
# Typed views of Data (refactor: design Q3, "Data split", step A).
#
# `Data` is a single overloaded container that plays ~8 roles. These thin
# subclasses give each role a small, self-documenting keyword constructor while
# remaining a `Data` in every respect: they delegate to `Data.__init__` with the
# same values, so `isinstance(x, Data)`, all inherited attributes/methods, and the
# stored arrays are byte-for-byte identical to the old positional calls. This is a
# pure readability change with no behavior change (golden-gated).
#
# The predictions (`gp_mean`/`gp_var`/`sse`/...) are still mutated onto these
# objects after construction, exactly as before; ending that pattern via a
# separate `GPPrediction` result is a later, separately-gated step (design Q3, C).
# ---------------------------------------------------------------------------


class ExperimentalData(Data):
    """Experimental observations: state points ``x_vals`` and responses ``y_vals``.

    For the synthetic case studies ``theta_vals`` carries the (repeated) true
    parameters, because the simulator reads it to *generate* ``y_vals``; for real
    calibration ``theta_vals`` is None (there is no ground-truth parameter set).
    Experimental data has no train/test split, so ``sep_fact`` is always None.
    """

    def __init__(self, x_vals, y_vals, *, theta_vals=None,
                 bounds_theta=None, bounds_x=None):
        super().__init__(
            theta_vals, x_vals, y_vals, None, None, None, None, None,
            bounds_theta, bounds_x, None,
        )


class SimulationData(Data):
    """Simulator evaluations ``(theta_vals, x_vals) -> y_vals`` used to train the
    GP emulator. Train/test splits are subsets of the same type.
    """

    def __init__(self, theta_vals, x_vals, y_vals, *, sse=None, sse_var=None,
                 acq=None, bounds_theta=None, bounds_x=None, sep_fact=None):
        super().__init__(
            theta_vals, x_vals, y_vals, None, None, sse, sse_var, acq,
            bounds_theta, bounds_x, sep_fact,
        )


class ObjectiveData(Data):
    """Objective (SSE) data per parameter set: ``theta_vals -> sse`` (+ ``sse_var``).

    Carries the experimental ``x_vals`` for reference but no per-point ``y_vals``.
    """

    def __init__(self, theta_vals, *, x_vals=None, sse=None, sse_var=None,
                 acq=None, bounds_theta=None, bounds_x=None, sep_fact=None):
        super().__init__(
            theta_vals, x_vals, None, None, None, sse, sse_var, acq,
            bounds_theta, bounds_x, sep_fact,
        )


class CandidateSet(Data):
    """A set of candidate/query points ``(theta_vals, x_vals)`` the GP is evaluated
    on during the BO loop (predictions are attached afterward). ``theta_vals`` may
    start as None and be filled in before prediction.
    """

    def __init__(self, theta_vals, x_vals, *, bounds_theta=None, bounds_x=None,
                 sep_fact=None):
        super().__init__(
            theta_vals, x_vals, None, None, None, None, None, None,
            bounds_theta, bounds_x, sep_fact,
        )
