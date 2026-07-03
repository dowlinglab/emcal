#!/usr/bin/env python3
"""Example: calibrate YOUR OWN model (not a built-in case study), signac-free.

The built-in ``run_*.py`` scripts calibrate the paper's case studies via
``get_case_study(n)``. This example shows the other entry point: define a
:class:`~emcal.CalibrationProblem` around your own ``model(theta, x)``
function and run emulator GPBO on it.

Mode shown here: **synthetic benchmark** — you supply ``true_params`` and the
package generates (noisy) "experimental" data from your model at those parameters,
so you can check that BO recovers them. This is the mode the paper studies use.

    Real calibration (you have measured (x, y), no known truth) is expressed as
        CalibrationProblem(model=f, param_names=..., param_bounds=...,
                           experimental_data=(x, y))
    and runs end-to-end via sim.set_experimental_data(x, y) -- see
    run_real_calibration.py.

Run:  python run_user_defined_problem.py
Needs: the 'gpflow' extra (GP backend). No signac, no cluster.
"""
import numpy as np

from emcal import CalibrationProblem, make_case_study_simulator
from emcal import (
    MethodName, EpSchedule, Kernel, GenMethod,
    GPBOMethod, ExplorationBias, BOConfig, GPBODriver,
)


def my_model(theta, x):
    """A user model: y = theta_1 * sin(x) + theta_2 * x. Signature is model(theta, x)."""
    return theta[0] * np.sin(x) + theta[1] * x


def main():
    # --- 1. Define the calibration problem around your own model. __post_init__ validates
    #        the bounds, the param/bounds lengths, and does a trial model(theta, x) call.
    true_params = np.array([2.0, -0.5])
    problem = CalibrationProblem(
        model=my_model,
        param_names=["amp", "slope"],
        param_bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        x_bounds=[(-3.0, 3.0)],
        true_params=true_params,     # synthetic-benchmark mode
        name="User Sine-Linear",
    )

    # --- 2. Build the simulator engine from the problem (same recipe as the built-ins).
    method = GPBOMethod(MethodName(7))          # E[SSE] emulator acquisition (analytic)
    simulator = make_case_study_simulator(problem, 0, None, 1)
    exp_data = simulator.generate_experimental_data(6, GenMethod(2), None, 0.01)
    n_params = len(simulator.indices_to_consider)
    sim_data = simulator.generate_simulation_data(
        10 * n_params, 6, GenMethod(1), GenMethod(2), 1.0, 1, False, None, w_noise=False
    )
    sim_sse_data = simulator.to_sse_data(method, sim_data, exp_data, 1.0, False)

    # --- 3. Exploration bias + BO config, then run (job=None => signac-free, in-memory).
    ep_bias = ExplorationBias(1, None, EpSchedule(1), None, None, None, None, None, None, None)
    cfg = BOConfig(
        problem.name, 1, 1.0, True, Kernel(1), None, None,
        10, 10, False, 10, 1, False, None, 1, 1e-7, 1e-7, True, False,
    )
    driver = GPBODriver(
        cfg, method, simulator, exp_data, sim_data, sim_sse_data,
        None, None, None, ep_bias, GenMethod(1),
    )
    gpbo_res_simple, gpbo_res_GP = driver.run(job=None)

    # --- 4. Inspect: did BO recover the true parameters?
    print(f"User-defined problem '{problem.name}'. True parameters: "
          f"{dict(zip(problem.param_names, true_params))}")
    for i, res in enumerate(gpbo_res_simple):
        final = res.results_df.tail(1)
        print(f"  restart {i}: best SSE = {final['best_sse_actual'].iloc[0]:.6g} "
              f"at theta = {final['theta_best_actual'].iloc[0]}  (why_term={res.why_term})")
    return gpbo_res_simple, gpbo_res_GP


if __name__ == "__main__":
    main()
