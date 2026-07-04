"""Fast orchestration tests for emulators.py (ObjectiveGP, EmulatorGP), driven by
FakeGPBackend -- no real GP training, deterministic canned mean/variance so the
GP-derived quantities (SSE, EI) are hand-computable.
"""
import numpy as np
import pytest
from scipy.stats import norm

from emcal.data import Data, ExperimentalData, SimulationData
from emcal.emulators import EmulatorGP, ObjectiveGP, build_gp_emulator
from emcal.enums import EpSchedule, Kernel, MethodName
from emcal.exploration import ExplorationBias
from emcal.methods import GPBOMethod

from _fakes import FakeGPBackend

BOUNDS_THETA = np.array([[0.0, 20.0], [0.0, 20.0]])
BOUNDS_X = np.array([[0.0, 1.0]])


# --- synthetic-data builders ---------------------------------------------------------

def _objective_sim_data(n_theta=6, sep_fact=0.7):
    theta_vals = np.arange(n_theta * 2, dtype=float).reshape(n_theta, 2)
    x_vals = np.array([[0.0]])  # unused by featurize_data; not indexed by train/test split
    y_vals = np.linspace(1.0, float(n_theta), n_theta)  # distinct SSE values per theta
    return SimulationData(theta_vals, x_vals, y_vals, bounds_theta=BOUNDS_THETA,
                           bounds_x=BOUNDS_X, sep_fact=sep_fact)


def _objective_gp(lenscl=None, outputscl=None, retrain_GP=1, normalize=False,
                   sep_fact=0.7, n_theta=6, mean_value=2.0, variance_value=0.25):
    backend = FakeGPBackend(mean_value=mean_value, variance_value=variance_value)
    gp_sim_data = _objective_sim_data(n_theta=n_theta, sep_fact=sep_fact)
    gp = ObjectiveGP(gp_sim_data, None, None, None, None, Kernel.MAT_52, lenscl, None,
                      outputscl, retrain_GP, 0, normalize,
                      backend=backend)
    gp.split_train_test(sep_fact, shuffle_seed=0)
    return gp, backend


def _emulator_sim_data(n_theta_unique=4, n_x=3, sep_fact=0.75):
    theta_unique = np.arange(n_theta_unique * 2, dtype=float).reshape(n_theta_unique, 2)
    theta_vals = np.repeat(theta_unique, n_x, axis=0)
    x_unique = np.linspace(0, 1, n_x).reshape(-1, 1)
    x_vals = np.tile(x_unique, (n_theta_unique, 1))
    y_vals = np.arange(len(theta_vals), dtype=float)
    data = SimulationData(theta_vals, x_vals, y_vals, bounds_theta=BOUNDS_THETA,
                           bounds_x=BOUNDS_X, sep_fact=sep_fact)
    return data, x_unique


def _emulator_gp(lenscl=None, outputscl=None, retrain_GP=1, normalize=False,
                  sep_fact=0.75, n_theta_unique=4, n_x=3, mean_value=1.0, variance_value=0.1):
    backend = FakeGPBackend(mean_value=mean_value, variance_value=variance_value)
    gp_sim_data, x_unique = _emulator_sim_data(n_theta_unique, n_x, sep_fact)
    gp = EmulatorGP(gp_sim_data, None, None, None, None, Kernel.MAT_52, lenscl, None,
                     outputscl, retrain_GP, 0, normalize,
                     backend=backend)
    gp.split_train_test(sep_fact, shuffle_seed=0)
    return gp, backend, x_unique


def _ep(ep_curr=1.0):
    return ExplorationBias(ep_curr, ep_curr, EpSchedule.CONSTANT,
                            None, None, None, None, None, None, None)


# --- featurize_data: theta-only (Type 1) vs theta+x (Type 2) -------------------------

def test_objective_featurize_data_uses_theta_only():
    gp, _ = _objective_gp()
    assert np.array_equal(gp.featurize_data(gp.train_data), gp.train_data.theta_vals)


def test_emulator_featurize_data_concatenates_theta_and_x():
    gp, _, _ = _emulator_gp()
    features = gp.featurize_data(gp.train_data)
    assert features.shape == (gp.train_data.n_theta, 3)  # 2 theta dims + 1 x dim
    assert np.array_equal(features[:, :2], gp.train_data.theta_vals)
    assert np.array_equal(features[:, 2:], gp.train_data.x_vals)


# --- split_train_test: shapes + feature arrays populated ------------------------------

def test_objective_split_train_test_partitions_all_rows():
    gp, _ = _objective_gp(n_theta=6, sep_fact=0.7)
    assert gp.train_data.n_theta + gp.test_data.n_theta == 6
    assert gp.feature_train_data.shape == (gp.train_data.n_theta, 2)
    assert gp.feature_test_data.shape == (gp.test_data.n_theta, 2)
    assert gp.train_data_init is not None


def test_emulator_split_train_test_partitions_by_unique_theta():
    gp, _, _ = _emulator_gp(n_theta_unique=4, n_x=3, sep_fact=0.75)
    n_train_theta = len(gp.train_data.get_unique_theta())
    n_test_theta = len(gp.test_data.get_unique_theta())
    assert n_train_theta + n_test_theta == 4
    assert gp.train_data.n_theta == n_train_theta * 3


# --- fit(): backend interaction across hyperparameter-guessing branches --------------

@pytest.mark.parametrize("retrain_GP", [0, 1, 2, 3])
def test_objective_fit_calls_backend_train_once_per_retrain(retrain_GP):
    gp, backend = _objective_gp(retrain_GP=retrain_GP)
    gp.fit()
    assert len(backend.train_calls) == retrain_GP
    assert gp.trained_hyperparams == backend.hyperparams
    assert gp.posterior is not None


@pytest.mark.parametrize("lenscl,outputscl", [
    (None, None), (1.0, 1.0), (np.array([1.0, 1.0]), None),
])
def test_objective_fit_handles_fixed_and_guessed_hyperparameters(lenscl, outputscl):
    gp, backend = _objective_gp(lenscl=lenscl, outputscl=outputscl, retrain_GP=1)
    gp.fit()
    assert len(backend.train_calls) == 1


def test_objective_fit_with_normalize_runs_and_predicts_finite():
    gp, _ = _objective_gp(normalize=True)
    gp.fit()
    pred = gp.predict(target="test")
    assert np.all(np.isfinite(pred.mean)) and np.all(np.isfinite(pred.variance))


def test_emulator_fit_calls_backend_train_once_per_retrain():
    gp, backend, _ = _emulator_gp(retrain_GP=2)
    gp.fit()
    assert len(backend.train_calls) == 2


# --- predict(): canned output, no Data mutation (locks in the 7C rewrite) ------------

def test_objective_predict_matches_canned_backend_and_does_not_mutate_data():
    gp, backend = _objective_gp(mean_value=2.0, variance_value=0.25)
    gp.fit()
    pred = gp.predict(target="test")

    n = gp.test_data.n_theta
    assert np.array_equal(pred.mean, np.full(n, 2.0))
    assert np.array_equal(pred.variance, np.full(n, 0.25))
    assert gp.test_data.gp_mean is None and gp.test_data.gp_var is None
    assert gp.test_data.gp_covar is None


def test_emulator_predict_matches_canned_backend_and_does_not_mutate_data():
    gp, backend, _ = _emulator_gp(mean_value=1.0, variance_value=0.1)
    gp.fit()
    pred = gp.predict(target="test")

    n = gp.test_data.n_theta
    assert np.array_equal(pred.mean, np.full(n, 1.0))
    assert gp.test_data.gp_mean is None and gp.test_data.gp_var is None


# --- predict_sse(): hand-computed from the canned mean/covar, no Data mutation -------

def test_objective_predict_sse_equals_predict_for_standard_gp():
    # For the objective GP the SSE *is* the GP output -- predict_sse should exactly match
    # predict() (Type 1: the GP already models the objective directly).
    gp, _ = _objective_gp()
    gp.fit()
    pred = gp.predict(target="test")
    sse_pred = gp.predict_sse(target="test")

    assert np.array_equal(sse_pred.mean, pred.mean)
    assert np.array_equal(sse_pred.variance, pred.variance)
    assert gp.test_data.sse is None and gp.test_data.sse_var is None


def test_emulator_predict_sse_matches_hand_derived_formula():
    # gp_mean is constant (mean_value) at every x; y_target is 0 -> per-theta SSE is
    # n_x * mean_value**2, and (for a single test theta) the variance formula reduces to
    # 2*trace(covar^2) + 4*residuals.T @ covar @ residuals with covar = variance_value*I.
    mean_value, variance_value, n_x = 1.0, 0.1, 3
    gp, backend, x_unique = _emulator_gp(mean_value=mean_value, variance_value=variance_value,
                                         n_theta_unique=4, n_x=n_x, sep_fact=0.75)
    gp.fit()
    exp_data = ExperimentalData(x_unique, np.zeros(n_x))
    method = GPBOMethod(MethodName.INDEPENDENCE)

    sse_pred = gp.predict_sse(target="test", method=method, exp_data=exp_data)

    n_test_theta = len(gp.test_data.get_unique_theta())
    assert n_test_theta == 1  # keeps the variance formula single-block (matches assumption)
    expected_sse = n_x * mean_value**2
    assert np.allclose(sse_pred.mean, expected_sse)
    residual = mean_value  # y_target - gp_mean = 0 - mean_value
    expected_var = 2 * n_x * variance_value**2 + 4 * n_x * (residual**2) * variance_value
    assert np.allclose(sse_pred.variance, expected_var)
    assert gp.test_data.sse is None and gp.test_data.sse_var is None


# --- calc_best_error(): hand-computed from train_data.y_vals -------------------------

def test_objective_calc_best_error_is_min_of_training_y():
    gp, _ = _objective_gp()
    best_error, be_theta, train_idx = gp.calc_best_error()

    assert best_error == gp.train_data.y_vals.min()
    assert train_idx == np.argmin(gp.train_data.y_vals)
    assert np.array_equal(be_theta, gp.train_data.theta_vals[train_idx])


def test_emulator_calc_best_error_is_min_sse_over_training_thetas():
    gp, _, x_unique = _emulator_gp()
    exp_data = ExperimentalData(x_unique, np.zeros(3))
    method = GPBOMethod(MethodName.INDEPENDENCE)

    best_error, be_theta, best_sq_error, org_idcs = gp.calc_best_error(method, exp_data)

    n_x = 3
    y_resh = gp.train_data.y_vals.reshape(-1, n_x)
    sse_per_theta = np.sum(y_resh**2, axis=1)  # exp y is 0, so sse = sum(y^2)
    assert np.isclose(best_error, sse_per_theta.min())
    assert best_sq_error.shape == (n_x,)


# --- expected_improvement(): matches the closed-form standard EI, no Data mutation ---

def test_objective_expected_improvement_matches_closed_form():
    mean_value, variance_value = 2.0, 0.25
    gp, _ = _objective_gp(mean_value=mean_value, variance_value=variance_value)
    gp.fit()
    best_error, be_theta, train_idx = gp.calc_best_error()
    ep = _ep(1.0)
    exp_data = ExperimentalData(np.array([[0.0]]), np.array([0.0]))

    ei, ei_df = gp.expected_improvement(target="test", exp_data=exp_data, ep_bias=ep,
                                         best_error_metrics=(best_error, be_theta, None))

    stdev = np.sqrt(variance_value)
    z = (best_error * ep.ep_curr - mean_value) / stdev
    expected = (best_error * ep.ep_curr - mean_value) * norm.cdf(z) + stdev * norm.pdf(z)
    assert np.isclose(ei[0], expected)
    assert gp.test_data.acq is None


def test_emulator_expected_improvement_is_finite_and_does_not_mutate_data():
    gp, _, x_unique = _emulator_gp()
    gp.fit()
    exp_data = ExperimentalData(x_unique, np.zeros(3))
    method = GPBOMethod(MethodName.INDEPENDENCE)
    best_error, be_theta, best_sq_error, _ = gp.calc_best_error(method, exp_data)
    ep = _ep(1.0)

    ei, ei_df = gp.expected_improvement(
        target="test", exp_data=exp_data, ep_bias=ep,
        best_error_metrics=(best_error, be_theta, best_sq_error), method=method,
    )

    assert np.all(np.isfinite(ei))
    assert gp.test_data.acq is None and gp.test_data.sse is None


# --- append_training_point(): grows training data by exactly one theta ---------------

def test_objective_append_training_point_grows_by_one_row():
    gp, _ = _objective_gp()
    n_before = gp.train_data.n_theta
    new_point = SimulationData(np.array([[99.0, 99.0]]), np.array([[0.0]]), np.array([42.0]),
                                bounds_theta=BOUNDS_THETA, bounds_x=BOUNDS_X, sep_fact=None)

    gp.append_training_point(new_point)

    assert gp.train_data.n_theta == n_before + 1
    assert gp.train_data.y_vals[-1] == 42.0
    assert gp.feature_train_data.shape[0] == n_before + 1


def test_emulator_append_training_point_grows_by_n_x_rows():
    gp, _, x_unique = _emulator_gp(n_x=3)
    n_before = gp.train_data.n_theta
    new_theta = np.repeat(np.array([[50.0, 50.0]]), 3, axis=0)
    new_point = SimulationData(new_theta, x_unique, np.array([9.0, 9.0, 9.0]),
                                bounds_theta=BOUNDS_THETA, bounds_x=BOUNDS_X, sep_fact=None)

    gp.append_training_point(new_point)

    assert gp.train_data.n_theta == n_before + 3
    assert gp.feature_train_data.shape[0] == n_before + 3


# --- build_gp_emulator(): factory dispatch by method.is_emulator ---------------------

def test_build_gp_emulator_picks_objective_gp_for_non_emulator_method():
    sim_sse_data = _objective_sim_data()
    method = GPBOMethod(MethodName.CONVENTIONAL)

    emu = build_gp_emulator(method, None, sim_sse_data, None, None, Kernel.MAT_52, None,
                            None, 1, 0, False, 0.1, 5, backend=FakeGPBackend())

    assert isinstance(emu, ObjectiveGP)
    assert emu._backend.hyperparams == FakeGPBackend().hyperparams


def test_build_gp_emulator_picks_emulator_gp_for_emulator_method():
    sim_data, _ = _emulator_sim_data()
    method = GPBOMethod(MethodName.INDEPENDENCE)

    emu = build_gp_emulator(method, sim_data, None, None, None, Kernel.MAT_52, None,
                            None, 1, 0, False, 0.1, 3, backend=FakeGPBackend())

    assert isinstance(emu, EmulatorGP)
