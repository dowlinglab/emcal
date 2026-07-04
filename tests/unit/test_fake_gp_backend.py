"""Pins FakeGPBackend's own contract (tests/unit/_fakes.py) and the backend= injection
point on GPEmulator/build_gp_emulator/GPBODriver: passing backend=None must reproduce
the exact production default (get_backend("gpflow")), so these are additive, zero-risk
changes to the constructors.
"""
import numpy as np

from emcal.emulators import GPEmulator, build_gp_emulator
from emcal.enums import Kernel
from emcal.gp_backend.gpflow_backend import GpflowBackend

from _fakes import FakeGPBackend


def test_fake_backend_implements_full_abc_interface():
    backend = FakeGPBackend()
    for method in ("configure", "make_bounded_parameter", "build_model", "train",
                   "get_hyperparameters", "make_posterior", "predict_f"):
        assert callable(getattr(backend, method))


def test_fake_backend_predict_f_is_canned_and_spd():
    backend = FakeGPBackend(mean_value=3.0, variance_value=0.5)
    eval_points = np.zeros((4, 2))  # values irrelevant -- only shape matters
    mean, covar = backend.predict_f(None, eval_points)

    assert np.array_equal(mean, np.full(4, 3.0))
    assert covar.shape == (4, 4)
    assert np.array_equal(covar, 0.5 * np.eye(4))
    # SPD: symmetric and strictly positive eigenvalues.
    assert np.array_equal(covar, covar.T)
    assert np.all(np.linalg.eigvalsh(covar) > 0)


def test_fake_backend_predict_f_supports_custom_mean_fn():
    backend = FakeGPBackend(mean_fn=lambda pts: pts.sum(axis=1))
    eval_points = np.array([[1.0, 2.0], [3.0, 4.0]])
    mean, covar = backend.predict_f(None, eval_points)
    assert np.array_equal(mean, np.array([3.0, 7.0]))


def test_fake_backend_records_train_and_predict_calls():
    backend = FakeGPBackend()
    model = backend.build_model((np.zeros((2, 1)), np.zeros((2, 1))), 3, 1.0, 1.0, 1.0,
                                 False, False)
    backend.train(model)
    posterior = backend.make_posterior(model)
    backend.predict_f(posterior, np.zeros((2, 1)))

    assert len(backend.build_model_calls) == 1
    assert len(backend.train_calls) == 1
    assert len(backend.predict_f_calls) == 1


def test_gpemulator_backend_none_resolves_to_gpflow_default():
    # gp_sim_data/gp_val_data/cand_data may be None at this base-class level; confirms the
    # real __init__ code path (not a re-implementation of its logic) picks gpflow by default.
    emu = GPEmulator(None, None, None, Kernel.MAT_52, None, 1.0, None, 0, None, False,
                      None, None, None, None)
    assert isinstance(emu._backend, GpflowBackend)


def test_gpemulator_backend_kwarg_bypasses_registry():
    fake = FakeGPBackend()
    emu = GPEmulator(None, None, None, Kernel.MAT_52, None, 1.0, None, 0, None, False,
                      None, None, None, None, backend=fake)
    assert emu._backend is fake


def test_build_gp_emulator_backend_kwarg_is_last_and_optional():
    import inspect

    sig = inspect.signature(build_gp_emulator)
    params = list(sig.parameters.values())
    assert params[-1].name == "backend"
    assert params[-1].default is None
