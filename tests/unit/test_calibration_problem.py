"""Unit tests for the CalibrationProblem dataclass + case-study helpers (design Q1)."""
import numpy as np
import pytest

from emcal import CalibrationProblem, get_case_study, make_case_study_simulator

BUILT_INS = [1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17]


def _cubic(theta, x):
    return theta[0] * x + theta[1] * x**2 + x**3


def test_synthetic_problem_validates():
    p = CalibrationProblem(
        model=_cubic, param_names=["a", "b"], param_bounds=[(-2, 2), (-2, 2)],
        x_bounds=[(-2, 2)], true_params=np.array([1.0, -1.0]), name="cubic",
    )
    assert p.name == "cubic"
    assert p.true_params is not None


def test_real_calibration_problem_validates():
    x = np.linspace(-2, 2, 6)
    y = _cubic(np.array([1.0, -1.0]), x)
    p = CalibrationProblem(
        model=_cubic, param_names=["a", "b"], param_bounds=[(-2, 2), (-2, 2)],
        x_bounds=[(-2, 2)], experimental_data=(x, y),
    )
    assert p.experimental_data is not None
    assert p.true_params is None


def test_requires_exactly_one_mode():
    # Neither experimental_data nor true_params.
    with pytest.raises(AssertionError):
        CalibrationProblem(model=_cubic, param_names=["a", "b"],
                           param_bounds=[(-2, 2), (-2, 2)], x_bounds=[(-2, 2)])
    # Both provided.
    with pytest.raises(AssertionError):
        CalibrationProblem(model=_cubic, param_names=["a", "b"],
                           param_bounds=[(-2, 2), (-2, 2)], x_bounds=[(-2, 2)],
                           true_params=np.array([1.0, -1.0]),
                           experimental_data=(np.array([0.0]), np.array([0.0])))


def test_param_names_bounds_length_mismatch():
    with pytest.raises(AssertionError):
        CalibrationProblem(model=_cubic, param_names=["a", "b", "c"],
                           param_bounds=[(-2, 2), (-2, 2)], x_bounds=[(-2, 2)],
                           true_params=np.array([1.0, -1.0]))


def test_true_params_out_of_bounds():
    with pytest.raises(AssertionError):
        CalibrationProblem(model=_cubic, param_names=["a", "b"],
                           param_bounds=[(-2, 2), (-2, 2)], x_bounds=[(-2, 2)],
                           true_params=np.array([99.0, -1.0]))


def test_bad_bound_order():
    with pytest.raises(AssertionError):
        CalibrationProblem(model=_cubic, param_names=["a", "b"],
                           param_bounds=[(2, -2), (-2, 2)], x_bounds=[(-2, 2)],
                           true_params=np.array([1.0, -1.0]))


@pytest.mark.parametrize("cs", BUILT_INS)
def test_get_case_study_returns_valid_problem(cs):
    if cs in (2, 3):
        pytest.importorskip("pyomo")
    p = get_case_study(cs)
    assert isinstance(p, CalibrationProblem)
    assert len(p.param_names) == len(p.param_bounds)
    assert p.true_params is not None            # built-ins are synthetic benchmarks
    assert p.name


def test_make_case_study_simulator_builds_simulator():
    from emcal import GenMethod
    p = get_case_study(1)
    sim = make_case_study_simulator(p, 0, None, 1)
    exp = sim.generate_experimental_data(5, GenMethod(2), None, 0.01)
    # 5 experimental x points requested.
    assert exp.n_x == 5


def test_real_calibration_simulator_has_no_ground_truth():
    # Real-calibration problem -> Simulator with theta_ref None and theta_true None.
    x = np.linspace(-2, 2, 6)
    y = _cubic(np.array([1.0, -1.0]), x)
    p = CalibrationProblem(
        model=_cubic, param_names=["a", "b"], param_bounds=[(-2, 2), (-2, 2)],
        x_bounds=[(-2, 2)], experimental_data=(x, y),
    )
    sim = make_case_study_simulator(p, 0, None, 1)
    assert sim.theta_ref is None
    assert sim.theta_true is None
    # set_experimental_data builds exp data from the user's (x, y): no true theta.
    exp = sim.set_experimental_data(*p.experimental_data)
    assert exp.n_x == 6
    assert exp.theta_vals is None
    assert np.allclose(np.asarray(exp.y_vals).ravel(), y)
    assert sim.noise_std is not None          # a noise scale is filled in
