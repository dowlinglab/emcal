"""Test-only fakes. Not part of the shipped package or the gp_backend registry.

FakeGPBackend implements the 7-method GPBackend ABC with opaque, deterministic stand-ins
for gpflow's model/posterior objects, so emulators.py/driver.py orchestration can be
exercised fast (no real GP training) via GPEmulator's `backend=` injection point.
"""
from types import SimpleNamespace

import numpy as np

from emcal.gp_backend.base import GPBackend


class FakeGPBackend(GPBackend):
    """A deterministic, no-training GP backend for fast orchestration tests.

    predict_f always returns a canned mean (constant `mean_value`, or `mean_fn(eval_points)`
    if given) and a diagonal covariance `variance_value * I` -- trivially SPD, so every
    point's predictive variance is exactly `variance_value` and all covariances are 0.
    That makes the GP-derived quantities downstream (SSE, EI, z-scores) hand-computable
    from `mean_value`/`variance_value` alone, without depending on any actual fitting.

    Call counts (train_calls, predict_f_calls, build_model_calls) are recorded so tests can
    assert *how* the orchestration code drives the backend (e.g. one train() per retrain),
    not just its numeric output.
    """

    name = "fake"

    def __init__(self, mean_value=1.0, variance_value=0.25, hyperparams=None, mean_fn=None):
        self.mean_value = mean_value
        self.variance_value = variance_value
        self.hyperparams = (
            hyperparams if hyperparams is not None else [np.array([1.0]), 0.01, 1.0]
        )
        self.mean_fn = mean_fn
        self.configure_calls = 0
        self.build_model_calls = []
        self.train_calls = []
        self.predict_f_calls = []

    def configure(self):
        self.configure_calls += 1

    def make_bounded_parameter(self, low, high, initial_value):
        # No transform needed: nothing outside this backend reads the returned value except
        # this backend's own build_model, so a plain pass-through is sufficient.
        return np.asarray(initial_value, dtype=float)

    def build_model(self, data, kernel_value, lenscls, tau, white_var,
                     fix_lengthscale, fix_outputscale, noise_variance=1e-5):
        model = SimpleNamespace(
            data=data, kernel_value=kernel_value, lenscls=lenscls, tau=tau,
            white_var=white_var, fix_lengthscale=fix_lengthscale,
            fix_outputscale=fix_outputscale, noise_variance=noise_variance,
        )
        self.build_model_calls.append(model)
        return model

    def train(self, model):
        self.train_calls.append(model)
        return True, 0.0

    def get_hyperparameters(self, model):
        return self.hyperparams

    def make_posterior(self, model):
        return SimpleNamespace(model=model)

    def predict_f(self, posterior, eval_points, full_cov=True):
        self.predict_f_calls.append(eval_points)
        n = len(eval_points)
        if self.mean_fn is not None:
            mean = np.asarray(self.mean_fn(eval_points), dtype=float)
        else:
            mean = np.full(n, self.mean_value, dtype=float)
        covar = self.variance_value * np.eye(n)
        return mean, covar
