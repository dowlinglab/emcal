#!/usr/bin/env python3
"""Example: REAL calibration — fit a model to measured (x, y) data, no ground truth.

Unlike the synthetic examples (which generate data from known `true_params`), here you
have measured observations and want the parameters that best explain them. You express
this with a `CalibrationProblem` that carries `experimental_data=(x, y)` and NO
`true_params`, then feed the measured data to the simulator via `set_experimental_data`.

Run:  python run_real_calibration.py
Needs: the 'gpflow' extra (GP backend). No signac, no cluster.
"""
import numpy as np

from emcal import (
    MethodName, EpSchedule, Kernel, GenMethod,
    GPBOMethod, ExplorationBias, BOConfig, GPBODriver,
    CalibrationProblem, make_case_study_simulator,
)


def my_model(theta, x):
    """A user model: y = theta_1 * x + theta_2 * x^2 + x^3. Signature is model(theta, x)."""
    return theta[0] * x + theta[1] * x**2 + x**3


def main():
    # --- 0. Pretend these are MEASURED data (here we fabricate them from a hidden
    #        parameter set + noise; a real user would load their own measurements).
    rng = np.random.default_rng(0)
    x_measured = np.linspace(-2.0, 2.0, 8)
    y_measured = my_model(np.array([1.0, -1.0]), x_measured) + rng.normal(0, 0.02, size=x_measured.shape)

    # --- 1. Define the calibration problem in REAL mode: experimental_data, no true_params.
    problem = CalibrationProblem(
        model=my_model,
        param_names=["a", "b"],
        param_bounds=[(-2.0, 2.0), (-2.0, 2.0)],
        x_bounds=[(-2.0, 2.0)],
        experimental_data=(x_measured, y_measured),
        name="Real Cubic",
    )

    # --- 2. Build the simulator (theta_ref is None -- no ground truth) and hand it the
    #        measured data directly via set_experimental_data (NOT generate_experimental_data,
    #        which would synthesize data from a true parameter set).
    method = GPBOMethod(MethodName.EXPECTED_SSE)     # analytic emulator acquisition
    sim = make_case_study_simulator(problem, 0, None, 1)
    exp_data = sim.set_experimental_data(*problem.experimental_data)

    # --- 3. Simulation (training) data for the GP emulator: sample parameters over the
    #        bounds and evaluate the model (no ground truth needed).
    n = len(sim.indices_to_consider)
    sim_data = sim.generate_simulation_data(
        10 * n, len(x_measured), GenMethod.LHS, GenMethod.MESHGRID, 1.0, 1, False, None, w_noise=False
    )
    sim_sse_data = sim.to_sse_data(method, sim_data, exp_data, 1.0, False)

    # --- 4. Exploration bias + BO config, then run (job=None => signac-free, in-memory).
    ep_bias = ExplorationBias(1, None, EpSchedule.CONSTANT, None, None, None, None, None, None, None)
    cfg = BOConfig(problem.name, kernel=Kernel.MAT_52, retrain_GP=10, reoptimize_obj=10,
                   bo_iter_tot=10, bo_run_tot=1, get_y_sse=True)
    driver = GPBODriver(cfg, method, sim, exp_data, sim_data, sim_sse_data,
                        None, None, None, ep_bias, GenMethod.LHS)
    results_simple, results_gp = driver.run(job=None)

    # --- 5. Inspect the best-fit parameters (there is no "true" answer to compare against).
    final = results_simple[0].results_df.tail(1)
    print(f"Real calibration of '{problem.name}' ({len(x_measured)} measured points).")
    print(f"  best SSE to data = {final['Min Obj Act Cum'].iloc[0]:.6g}")
    print(f"  best-fit parameters = {final['Theta Obj Act Cum'].iloc[0]}  (names: {problem.param_names})")
    return results_simple, results_gp


if __name__ == "__main__":
    main()
