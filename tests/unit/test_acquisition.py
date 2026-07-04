"""Unit tests for ExpectedImprovement (acquisition.py): pure EI math, no GP backend."""
import math

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

from emcal.acquisition import ExpectedImprovement
from emcal.data import Data
from emcal.enums import EpSchedule, MethodName
from emcal.exploration import ExplorationBias
from emcal.methods import GPBOMethod


def _exp_data(y_vals):
    x = np.arange(len(y_vals), dtype=float).reshape(-1, 1)
    return Data(None, x, y_vals, None, None, None, None, None, None, None, None)


def _ep(ep_curr):
    return ExplorationBias(ep_curr, ep_curr, EpSchedule.CONSTANT,
                            None, None, None, None, None, None, None)


def _ei(gp_mean, gp_covar, y_vals, best_error, best_error_x, ep_curr=1.0,
        seed=0, sg_mc_samples=200, method=None, be_theta=None):
    ep = _ep(ep_curr)
    exp_data = _exp_data(y_vals)
    bem = (best_error, np.array([0.0]) if be_theta is None else be_theta, best_error_x)
    return ExpectedImprovement(ep, gp_mean, gp_covar, exp_data, bem, set_seed=seed,
                                sg_mc_samples=sg_mc_samples, method=method)


# --- __compute_standard (method=None): closed-form z-score/cdf/pdf, Eq. 6a/6b -------------

def test_compute_standard_matches_closed_form_eq6():
    gp_mean = np.array([2.0, 5.0])
    gp_covar = np.diag([4.0, 0.0])  # stdevs [2.0, 0.0]
    obj = _ei(gp_mean, gp_covar, np.zeros(2), best_error=3.0, best_error_x=None, ep_curr=1.5)

    ei, ei_df = obj.compute()

    stdev0 = 2.0
    z0 = (3.0 * 1.5 - 2.0) / stdev0
    expected0 = (3.0 * 1.5 - 2.0) * norm.cdf(z0) + stdev0 * norm.pdf(z0)
    assert np.isclose(ei[0], expected0)
    assert ei[1] == 0.0  # zero-stdev branch short-circuits to 0
    assert len(ei_df) == 2


# --- __calc_ei_emulator (method 3, independence approx / 2A) -----------------------------

def test_ei_emulator_matches_closed_form_for_exact_fit():
    # y_target == gp_mean collapses the independence-approx bounds to +-B, which reduces to
    # a closed form in norm.cdf/pdf (derived from the manuscript's independence-approx EI).
    best_error_x, ep_curr, var = 2.0, 1.5, 0.25
    method = GPBOMethod(MethodName.INDEPENDENCE)
    obj = _ei(np.array([5.0]), np.array([[var]]), np.array([5.0]),
              best_error=best_error_x, best_error_x=np.array([best_error_x]),
              ep_curr=ep_curr, method=method)

    ei, _ = obj.compute()

    B = math.sqrt(best_error_x * ep_curr) / math.sqrt(var)
    expected = (2 * norm.cdf(B) - 1) * best_error_x * ep_curr + var * (
        2 * B * norm.pdf(B) - 2 * norm.cdf(B) + 1
    )
    assert np.isclose(ei[0], expected, rtol=1e-9)


def test_ei_emulator_is_zero_when_variance_non_positive():
    method = GPBOMethod(MethodName.INDEPENDENCE)
    obj = _ei(np.array([5.0]), np.array([[0.0]]), np.array([5.0]),
              best_error=2.0, best_error_x=np.array([2.0]), method=method)

    ei, ei_df = obj.compute()

    assert ei[0] == 0.0
    assert ei_df["ei_total"].iloc[0] == 0


# --- __ei_approx_ln_term: pure integrand for the log independence approx (2B) -------------

def test_ei_approx_ln_term_is_zero_at_the_target():
    obj = _ei(np.array([1.0]), np.array([[1.0]]), np.array([1.0]), 1.0, np.array([1.0]))
    # inside_term = |y_target - gp_mean - gp_stdev*epsilon| = |1 - 1 - 2*0.5| = 1.0 -> log(1)=0
    value = obj._ExpectedImprovement__ei_approx_ln_term(0.5, 1.0, 2.0, 1.0)
    assert value == 0.0


def test_ei_approx_ln_term_matches_log_pdf_formula():
    obj = _ei(np.array([1.0]), np.array([[1.0]]), np.array([1.0]), 1.0, np.array([1.0]))
    value = obj._ExpectedImprovement__ei_approx_ln_term(1.0, 0.0, 1.0, 3.0)
    assert np.isclose(value, math.log(2.0) * norm.pdf(1.0))


# --- __calc_ei_log_emulator (method 4, log independence approx / 2B) ---------------------

def test_ei_log_emulator_cross_checked_against_independent_quadrature():
    # ei_term1 has the same +-B closed form as 2A (in log space); ei_term2 is cross-checked
    # by integrating the method's own (separately-tested above) integrand independently
    # here, rather than re-deriving its antiderivative by hand.
    best_error_x, ep_curr, var = 2.0, 1.5, 0.25
    method = GPBOMethod(MethodName.LOG_INDEPENDENCE)
    obj = _ei(np.array([5.0]), np.array([[var]]), np.array([5.0]),
              best_error=best_error_x, best_error_x=np.array([best_error_x]),
              ep_curr=ep_curr, method=method)

    ei, _ = obj.compute()

    be_log = math.log(best_error_x)
    stdev = math.sqrt(var)
    B = math.sqrt(math.exp(be_log * ep_curr)) / stdev
    term1 = be_log * ep_curr * (2 * norm.cdf(B) - 1)
    q, _ = integrate.quad(
        obj._ExpectedImprovement__ei_approx_ln_term, -B, B, args=(5.0, stdev, 5.0)
    )
    expected = term1 - 2 * q
    assert np.isclose(ei[0], expected, rtol=1e-6)


def test_ei_log_emulator_is_zero_when_variance_non_positive():
    method = GPBOMethod(MethodName.LOG_INDEPENDENCE)
    obj = _ei(np.array([5.0]), np.array([[0.0]]), np.array([5.0]),
              best_error=2.0, best_error_x=np.array([2.0]), method=method)

    ei, ei_df = obj.compute()

    assert ei[0] == 0.0
    assert ei_df["ei_total"].iloc[0] == 0


# --- __calc_ei_sparse (method 5, sparse-grid integrated EI / 2C) -------------------------

def test_ei_sparse_is_deterministic_and_nonnegative():
    best_error_x, ep_curr, var = 2.0, 1.5, 0.25
    method = GPBOMethod(MethodName.SPARSE_GRID)

    def _run():
        obj = _ei(np.array([5.0]), np.array([[var]]), np.array([5.0]),
                  best_error=best_error_x, best_error_x=np.array([best_error_x]),
                  ep_curr=ep_curr, seed=7, sg_mc_samples=200, method=method)
        return obj.compute()[0][0]

    ei_a, ei_b = _run(), _run()
    assert ei_a == ei_b  # Tasmanian's quadrature grid is deterministic, no RNG involved
    assert ei_a > 0


def test_ei_sparse_is_zero_when_variance_negative():
    # pos_stdev_mask uses `gp_var >= 0`; a genuinely negative (numerically non-PSD) variance
    # is the only way to make it all-False.
    method = GPBOMethod(MethodName.SPARSE_GRID)
    obj = _ei(np.array([5.0]), np.array([[-1.0]]), np.array([5.0]),
              best_error=2.0, best_error_x=np.array([2.0]), method=method)

    ei, ei_df = obj.compute()

    assert ei[0] == 0.0
    assert ei_df["ei_total"].iloc[0] == 0


# --- __calc_ei_mc (method 6, Monte Carlo integrated EI / 2D) + __bootstrap --------------

def test_ei_mc_is_deterministic_given_fixed_seed():
    best_error_x, ep_curr, var = 2.0, 1.5, 0.25
    method = GPBOMethod(MethodName.MONTE_CARLO)

    def _run():
        obj = _ei(np.array([5.0]), np.array([[var]]), np.array([5.0]),
                  best_error=best_error_x, best_error_x=np.array([best_error_x]),
                  ep_curr=ep_curr, seed=42, sg_mc_samples=500, method=method)
        return obj.compute()

    (ei_a, df_a), (ei_b, df_b) = _run(), _run()
    assert ei_a[0] == ei_b[0]  # same seed -> same MC draw -> exact reproduction
    assert df_a["ci_lower"].iloc[0] <= ei_a[0] <= df_a["ci_upper"].iloc[0]


def test_ei_mc_is_zero_when_variance_negative():
    method = GPBOMethod(MethodName.MONTE_CARLO)
    obj = _ei(np.array([5.0]), np.array([[-1.0]]), np.array([5.0]),
              best_error=2.0, best_error_x=np.array([2.0]), seed=1, method=method)

    ei, ei_df = obj.compute()

    assert ei[0] == 0.0
    assert ei_df["ei_total"].iloc[0] == 0


def test_ei_sparse_ldl_fallback_for_non_positive_definite_covariance():
    # __calc_ei_sparse diagonalizes self.gp_covar with Cholesky, falling back to LDL when
    # that fails; a non-PD covariance (eigenvalues 3, -1) forces the except branch.
    method = GPBOMethod(MethodName.SPARSE_GRID)
    non_pd_covar = np.array([[1.0, 2.0], [2.0, 1.0]])
    obj = _ei(np.array([5.0, 5.0]), non_pd_covar, np.array([5.0, 5.0]),
              best_error=2.0, best_error_x=np.array([2.0, 2.0]),
              sg_mc_samples=50, method=method)

    ei, _ = obj.compute()

    assert np.all(np.isfinite(ei))


def test_compute_emulator_rejects_method_outside_2a_2d_range():
    # method must be an emulator method (3-6); a non-emulator MethodName (e.g. CONVENTIONAL)
    # is structurally valid at construction (assert method is None or GPBOMethod) but
    # rejected by __compute_emulator's own dispatch.
    method = GPBOMethod(MethodName.CONVENTIONAL)
    obj = _ei(np.array([1.0]), np.array([[1.0]]), np.array([1.0]), 1.0,
              np.array([1.0]), method=method)

    with pytest.raises(ValueError, match="2A.*2B.*2C.*2D"):
        obj.compute()


def test_rng_set_falls_back_to_rng_rand_when_seed_is_none():
    obj = _ei(np.array([1.0]), np.array([[1.0]]), np.array([1.0]), 1.0, None, seed=None)
    assert obj.rng_set is obj.rng_rand


def test_bootstrap_handles_scalar_pilot_sample():
    # __calc_ei_mc can pass ei_temp=0 (a Python scalar, not an array) in the degenerate
    # case; np.mean(0, axis=0) would raise AxisError without the atleast_1d guard.
    obj = _ei(np.array([1.0]), np.array([[1.0]]), np.array([1.0]), 1.0, np.array([1.0]), seed=3)
    ci = obj._ExpectedImprovement__bootstrap(0, ns=10, alpha=0.05)
    assert np.array_equal(ci, np.array([0.0, 0.0]))


# --- __set_rand_vars: LDL fallback for a non-positive-definite covariance ----------------

def test_set_rand_vars_ldl_fallback_for_non_positive_definite_covariance():
    obj = _ei(np.array([1.0, 2.0]), np.eye(2), np.array([1.0, 2.0]), 1.0,
              np.array([1.0, 1.0]), seed=1, sg_mc_samples=50,
              method=GPBOMethod(MethodName.MONTE_CARLO))
    non_pd_covar = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3, -1

    random_vars = obj._ExpectedImprovement__set_rand_vars(np.array([1.0, 2.0]), non_pd_covar)

    assert random_vars.shape == (50, 2)
    assert np.all(np.isfinite(random_vars))
