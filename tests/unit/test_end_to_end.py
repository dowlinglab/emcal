"""A tiny end-to-end BO smoke test on the redesigned API.

Marked `slow` (it builds and trains GPs via gpflow). Run the fast tier with
`pytest -m "not slow"`.
"""
import numpy as np
import pytest

from emcal import (
    MethodName, EpSchedule, Kernel, GenMethod, GPBOMethod, ExplorationBias,
    BOConfig, GPBODriver, get_case_study, make_case_study_simulator,
    CalibrationProblem,
)


@pytest.mark.slow
def test_simple_linear_bo_completes():
    method = GPBOMethod(MethodName(7))          # E[SSE] emulator (analytic)
    problem = get_case_study(1)
    sim = make_case_study_simulator(problem, 0, None, 1)
    exp = sim.generate_experimental_data(5, GenMethod(2), None, 0.01)
    n = len(sim.indices_to_consider)
    simd = sim.generate_simulation_data(
        10 * n, 5, GenMethod(1), GenMethod(2), 1.0, 1, False, None, with_noise=False
    )
    ssed = sim.to_sse_data(method, simd, exp, 1.0, False)
    ep = ExplorationBias(1, None, EpSchedule(1), None, None, None, None, None, None, None)
    cfg = BOConfig(problem.name, 1, 1.0, True, Kernel(1), None, None, 3, 3, False,
                   3, 1, False, None, 1, 1e-7, 1e-7, True, False)
    driver = GPBODriver(cfg, method, sim, exp, simd, ssed, None, None, None, ep, GenMethod(1))
    res_simple, res_gp = driver.run(job=None)

    assert len(res_simple) == 1
    best = float(res_simple[0].results_df.tail(1)["best_sse_actual"].iloc[0])
    assert np.isfinite(best)
    # CS1/m7 is deterministic (compile=False); this is the committed golden value.
    assert best == pytest.approx(12.0973745816, abs=1e-6)


@pytest.mark.slow
def test_real_calibration_runs_end_to_end():
    # Real calibration: measured (x, y), no true_params. Should run and return a finite SSE.
    def model(theta, x):
        return theta[0] * x + theta[1] * x**2 + x**3

    rng = np.random.default_rng(0)
    x = np.linspace(-2.0, 2.0, 6)
    y = model(np.array([1.0, -1.0]), x) + rng.normal(0, 0.02, size=x.shape)
    problem = CalibrationProblem(
        model=model, param_names=["a", "b"], param_bounds=[(-2, 2), (-2, 2)],
        x_bounds=[(-2, 2)], experimental_data=(x, y), name="Real Cubic",
    )
    method = GPBOMethod(MethodName.EXPECTED_SSE)
    sim = make_case_study_simulator(problem, 0, None, 1)
    exp = sim.set_experimental_data(*problem.experimental_data)
    n = len(sim.indices_to_consider)
    simd = sim.generate_simulation_data(
        10 * n, 6, GenMethod.LHS, GenMethod.MESHGRID, 1.0, 1, False, None, with_noise=False
    )
    ssed = sim.to_sse_data(method, simd, exp, 1.0, False)
    ep = ExplorationBias(1, None, EpSchedule.CONSTANT, None, None, None, None, None, None, None)
    cfg = BOConfig(problem.name, kernel=Kernel.MAT_52, retrain_gp=5, reoptimize_obj=5,
                   bo_iter_tot=5, bo_run_tot=1, compute_y_sse=True)
    res_simple, _ = GPBODriver(cfg, method, sim, exp, simd, ssed,
                               None, None, None, ep, GenMethod.LHS).run(job=None)
    best = float(res_simple[0].results_df.tail(1)["best_sse_actual"].iloc[0])
    assert np.isfinite(best) and best >= 0.0
