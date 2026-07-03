#!/usr/bin/env python3
"""Example / tutorial: Simple Linear case study (CS1), signac-free.

This is the *annotated* example — it writes out the full emulator-GPBO recipe inline so
you can see every step. The other ``run_*.py`` scripts do the same via the shared
``common.run_case_study`` helper.

Paper case study "Simple Linear" (Carlozo, Wang & Dowling, Ind. Eng. Chem. Res. 2025,
Table 2): model f(theta, x) = theta_1*x + theta_2*x**2 + x**3, true theta = [1, -1].

Run:  python run_simple_linear.py
Needs: the 'gpflow' extra (GP backend). No signac, no cluster.
"""
import numpy as np

from emcal import (
    MethodName, EpSchedule, Kernel, GenMethod,
    GPBOMethod, ExplorationBias, BOConfig, GPBODriver,
)
from emcal import get_case_study, make_case_study_simulator


def main():
    cs_num = 1

    # --- method: 7 = E[SSE], an emulator (Type 2) acquisition recommended in the paper.
    #     It is analytic (no Tasmanian needed). See common.METHODS for all 7 options.
    method = GPBOMethod(MethodName(7))

    # --- 1. Define the calibration problem, then build its simulator.
    #     get_case_study(n) returns a CalibrationProblem for a built-in paper benchmark.
    #     To calibrate your own model instead, build a CalibrationProblem around it
    #     (model=f, param_names=[...], param_bounds=[...], true_params=[...]) -- see
    #     run_user_defined_problem.py.
    #     make_case_study_simulator args: (problem, noise_mean, noise_std, seed);
    #     noise_std=None => noise is set automatically to ~1% of the median |y|.
    problem = get_case_study(cs_num)
    simulator = make_case_study_simulator(problem, 0, None, 1)

    # --- 2. Synthetic "experimental" data: num_x state points on a grid.
    exp_data = simulator.generate_experimental_data(5, GenMethod(2), None, 0.01)

    # --- 3. Simulation (training) data for the GP emulator: Latin-hypercube over parameters
    #     (10 x n_params points), grid over state points.
    n_params = len(simulator.indices_to_consider)
    sim_data = simulator.generate_simulation_data(
        10 * n_params, 5, GenMethod(1), GenMethod(2), 1.0, 1, False, None, w_noise=False
    )
    # Convert to the SSE representation the method expects.
    sim_sse_data = simulator.to_sse_data(method, sim_data, exp_data, 1.0, False)

    # --- 4. Exploration bias (constant alpha = 1: balanced explore/exploit).
    ep_bias = ExplorationBias(1, None, EpSchedule(1), None, None, None, None, None, None, None)

    # --- 5. Case study / BO parameters. Positional args (see BOConfig):
    #     name, ep0, sep_fact, normalize, kernel, lenscl, outputscl, retrain_GP,
    #     reoptimize_obj, gen_heat_map_data, bo_iter_tot, bo_run_tot, save_data, DateTime,
    #     set_seed, obj_tol, acq_tol, get_y_sse, w_noise.
    #     (retrain/reopt reduced to 10 so this runs in ~1-2 min; the paper used 25.)
    cs_params = BOConfig(
        problem.name, 1, 1.0, True, Kernel(1), None, None,
        10, 10, False, 10, 1, False, None, 1, 1e-7, 1e-7, True, False,
    )

    # --- 6. Driver and run. job=None => signac-free; results are returned in memory.
    driver = GPBODriver(
        cs_params, method, simulator, exp_data, sim_data, sim_sse_data,
        None, None, None, ep_bias, GenMethod(1),
    )
    gpbo_res_simple, gpbo_res_GP = driver.run(job=None)

    # --- 7. Inspect results.
    true = dict(zip(simulator.theta_true_names, simulator.theta_true))
    print(f"Simple Linear (CS1), method E[SSE]. True parameters: {true}")
    for i, res in enumerate(gpbo_res_simple):
        final = res.results_df.tail(1)
        print(f"  restart {i}: best SSE = {final['best_sse_actual'].iloc[0]:.6g} "
              f"at theta = {final['theta_best_actual'].iloc[0]}  (why_term={res.why_term})")
    return gpbo_res_simple, gpbo_res_GP


if __name__ == "__main__":
    main()
