"""Fast tests for the Data container, its typed subclasses, and GPPrediction.

No GP training here, so these run without the gpflow/Tasmanian extras.
"""
import numpy as np
import pytest

from emcal import (
    Data, ExperimentalData, SimulationData, ObjectiveData, CandidateSet, GPPrediction,
)


# --- typed subclasses ---------------------------------------------------------------
def _bounds():
    return np.array([[0.0, 1.0], [0.0, 1.0]]), np.array([[0.0, 1.0]])


def test_typed_subclasses_are_data_and_map_fields():
    bt, bx = _bounds()
    theta = np.zeros((3, 2))
    x = np.ones((3, 1))
    y = np.arange(3.0)

    exp = ExperimentalData(x, y, theta_vals=theta, bounds_theta=bt, bounds_x=bx)
    sim = SimulationData(theta, x, y, sse=None, bounds_theta=bt, bounds_x=bx, sep_fact=0.8)
    obj = ObjectiveData(theta, x_vals=x, sse=y, sse_var=y, bounds_theta=bt, bounds_x=bx, sep_fact=1.0)
    cand = CandidateSet(theta, x, bounds_theta=bt, bounds_x=bx, sep_fact=0.8)

    # Every typed view IS a Data (isinstance stays true, so all call sites keep working).
    for d in (exp, sim, obj, cand):
        assert isinstance(d, Data)

    # Field mapping matches the historical positional Data(...) calls.
    assert exp.sep_fact is None and exp.gp_mean is None and np.allclose(exp.y_vals, y)
    assert sim.sep_fact == 0.8 and sim.sse is None and np.allclose(sim.theta_vals, theta)
    assert obj.y_vals is None and np.allclose(obj.sse, y) and np.allclose(obj.sse_var, y)
    assert cand.theta_vals is not None and cand.y_vals is None and cand.sep_fact == 0.8


def test_typed_subclass_byte_identical_to_positional_data():
    # A typed subclass must produce the same attribute values as the equivalent raw
    # positional Data(...) it replaced (this is what keeps the golden byte-identical).
    bt, bx = _bounds()
    theta, x, y = np.zeros((2, 2)), np.ones((2, 1)), np.arange(2.0)
    raw = Data(theta, x, y, None, None, None, None, None, bt, bx, 0.75)
    typed = SimulationData(theta, x, y, bounds_theta=bt, bounds_x=bx, sep_fact=0.75)
    for attr in ("theta_vals", "x_vals", "y_vals", "gp_mean", "gp_var", "gp_covar",
                 "sse", "sse_var", "sse_covar", "acq", "bounds_theta", "bounds_x", "sep_fact"):
        rv, tv = getattr(raw, attr), getattr(typed, attr)
        if isinstance(rv, np.ndarray):
            assert np.array_equal(rv, tv), attr
        else:
            assert rv == tv, attr


def test_candidate_set_allows_none_theta():
    _, bx = _bounds()
    cand = CandidateSet(None, np.ones((4, 1)), bounds_x=bx, sep_fact=0.8)
    assert cand.theta_vals is None
    assert cand.n_x == 4


# --- Data helper methods ------------------------------------------------------------
def test_unique_and_dimension_properties():
    theta = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 2.0]])  # first two duplicate
    x = np.array([[0.0], [0.0], [5.0]])
    d = SimulationData(theta, x, np.zeros(3), bounds_theta=None, bounds_x=None, sep_fact=1.0)
    assert d.n_theta == 3
    assert d.theta_dim == 2
    assert d.n_x == 3
    assert d.x_dim == 1
    assert len(d.get_unique_theta()) == 2   # duplicate collapsed
    assert len(d.get_unique_x()) == 2


def test_train_test_split_is_seed_deterministic_and_covers_all():
    theta = np.arange(20.0).reshape(10, 2)
    d = SimulationData(theta, np.ones((10, 1)), np.zeros(10),
                       bounds_theta=None, bounds_x=None, sep_fact=0.7)
    tr1, te1 = d.train_test_idx_split(rng_seed=42)
    tr2, te2 = d.train_test_idx_split(rng_seed=42)
    assert np.array_equal(tr1, tr2) and np.array_equal(te1, te2)   # same seed -> same split
    # ceil(10 * 0.7) = 7 training points; partition covers all 10 unique thetas.
    assert len(tr1) == 7 and len(te1) == 3
    assert sorted(np.concatenate([tr1, te1])) == list(range(10))


def test_train_test_split_requires_sep_fact():
    d = SimulationData(np.zeros((4, 2)), np.ones((4, 1)), np.zeros(4),
                       bounds_theta=None, bounds_x=None, sep_fact=None)
    with pytest.raises(AssertionError):
        d.train_test_idx_split(rng_seed=0)


# --- GPPrediction -------------------------------------------------------------------
def test_gpprediction_named_access_and_legacy_unpack():
    mean = np.array([1.0, 2.0])
    var = np.array([0.1, 0.2])
    covar = np.eye(2)

    p = GPPrediction(mean, variance=var, covariance=covar, as_covar=False)
    # named access
    assert np.array_equal(p.mean, mean)
    assert np.array_equal(p.variance, var)
    assert np.array_equal(p.covariance, covar)
    # legacy 2-tuple unpack: covar=False -> variance is the second element
    m, v = p
    assert np.array_equal(m, mean) and np.array_equal(v, var)
    # indexing + len mirror the old tuple
    assert np.array_equal(p[0], mean) and np.array_equal(p[1], var) and len(p) == 2


def test_gpprediction_as_covar_switches_second_element():
    mean, var, covar = np.array([1.0]), np.array([0.1]), np.array([[0.1]])
    p = GPPrediction(mean, variance=var, covariance=covar, as_covar=True)
    _, second = p
    assert np.array_equal(second, covar)   # covar=True -> covariance is the second element
