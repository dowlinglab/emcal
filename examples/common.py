"""Shared helper for the emcal examples.

Encapsulates the minimal, signac-free recipe for running emulator GPBO on one of the
paper's case studies via ``GPBODriver.run(job=None)``. Each ``run_*.py``
example sets a case study + method and calls :func:`run_case_study`; see
``run_simple_linear.py`` for the same recipe written out inline (the teaching example).

All example runs use small budgets and fixed seeds so they finish quickly and
reproducibly — they demonstrate the API, they are not paper-scale studies.
"""
from emcal import (
    MethodName,
    EpSchedule,
    Kernel,
    GenMethod,
    GPBOMethod,
    ExplorationBias,
    BOConfig,
    GPBODriver,
)
from emcal import (
    get_case_study,
    make_case_study_simulator,
)

# Per-case-study metadata. Maps the case study number to its paper name, the number of
# experimental x points, fixed x-grids for the VLE studies, and whether it needs an ipopt
# solver (the Müller studies solve a Pyomo NLP). See the paper's Table 2.
CASE_STUDIES = {
    1:  dict(name="Simple Linear",       num_x=5,  x_vals=None, needs_ipopt=False),
    2:  dict(name="Muller x0",           num_x=5,  x_vals=None, needs_ipopt=True),
    3:  dict(name="Muller y0",           num_x=5,  x_vals=None, needs_ipopt=True),
    10: dict(name="Large Linear",        num_x=5,  x_vals=None, needs_ipopt=False),
    11: dict(name="BOD Curve",           num_x=10, x_vals=None, needs_ipopt=False),
    12: dict(name="Yield-Loss",          num_x=10, x_vals=None, needs_ipopt=False),
    13: dict(name="Log Logistic",        num_x=10, x_vals=None, needs_ipopt=False),
    14: dict(name="2D Log Logistic",     num_x=5,  x_vals=None, needs_ipopt=False),
    15: dict(name="Simple Multimodal",   num_x=10, x_vals=None, needs_ipopt=False),
    16: dict(name="Water-Glycerol VLE",  num_x=10,
             x_vals=[0.0, 0.1115, 0.2475, 0.4076, 0.5939, 0.8230, 0.9214, 0.9296, 0.985, 1.000],
             needs_ipopt=False),
    17: dict(name="ACN-Water VLE",       num_x=10,
             x_vals=[0.0087, 0.0269, 0.0568, 0.1556, 0.2749, 0.4449, 0.661, 0.8096, 0.9309, 0.9578],
             needs_ipopt=False),
}

# Method enum -> short description (see paper Section 2). 1-2 are standard GPBO (Type 1),
# 3-7 are emulator GPBO (Type 2).
METHODS = {
    1: "Conventional (standard GPBO)",
    2: "Log Conventional (standard GPBO)",
    3: "Independence (emulator)",
    4: "Log Independence (emulator)",
    5: "Sparse Grid (emulator; needs the 'sparsegrid' extra / Tasmanian)",
    6: "Monte Carlo (emulator)",
    7: "E[SSE] (emulator)",
}


def run_case_study(cs_num, method_val=4, iters=10, runs=1, seed=1,
                   retrain_gp=10, reopt_obj=10, verbose=True):
    """Run emulator GPBO on one case study, signac-free.

    Parameters
    ----------
    cs_num : int
        Case study number (a key of CASE_STUDIES).
    method_val : int
        Method enum value 1-7 (see METHODS). Default 4 (Log Independence, emulator).
    iters : int
        BO iterations per restart (use >= 2). Default 10.
    runs : int
        Number of restarts. Default 1.
    seed : int
        Random seed for the BO run (data generation uses a fixed seed of 1).
    retrain_gp, reopt_obj : int
        GP re-trainings and acquisition re-optimizations per iteration. Default 10 each
        (lighter than the paper's 25 so examples run in a couple of minutes; raise for
        better convergence).
    verbose : bool
        Print a short result summary.

    Returns
    -------
    (gpbo_res_simple, gpbo_res_GP) : tuple of lists of BOResults
        The per-restart results (see BOResults). ``run(job=None)`` keeps
        everything in memory — no files are written and signac is not involved.
    """
    if cs_num not in CASE_STUDIES:
        raise ValueError(f"Unknown case study {cs_num}. Options: {sorted(CASE_STUDIES)}")
    meta = CASE_STUDIES[cs_num]
    import numpy as np

    method = GPBOMethod(MethodName(method_val))
    kernel = Kernel(1)                 # Matern 5/2
    gen_meth_theta = GenMethod(1)       # Latin hypercube over parameters
    gen_meth_x = GenMethod(2)           # grid over state points
    x_vals = np.array(meta["x_vals"]) if meta["x_vals"] is not None else None

    # 1. Define the calibration problem, then build its simulator (fixed data-generation
    #    seed keeps data reproducible). get_case_study returns a CalibrationProblem for the
    #    built-in paper benchmarks; make_case_study_simulator builds the internal engine.
    problem = get_case_study(cs_num)
    simulator = make_case_study_simulator(problem, 0, None, 1)

    # 2. Synthetic experimental data + simulation (training) data
    exp_data = simulator.generate_experimental_data(meta["num_x"], gen_meth_x, x_vals, 0.01)
    num_theta = len(simulator.indices_to_consider) * 10
    sim_data = simulator.generate_simulation_data(
        num_theta, meta["num_x"], gen_meth_theta, gen_meth_x, 1.0, 1, False, x_vals, w_noise=False
    )
    sim_sse_data = simulator.to_sse_data(method, sim_data, exp_data, 1.0, False)

    # 3. Exploration bias (constant alpha = 1) and case study parameters
    ep_bias = ExplorationBias(1, None, EpSchedule(1), None, None, None, None, None, None, None)
    cs_params = BOConfig(
        problem.name, 1, 1.0, True, kernel, None, None,
        retrain_gp, reopt_obj, False, iters, runs, False, None, seed, 1e-7, 1e-7, True, False,
    )

    # 4. Driver + run (job=None => no signac, results returned in memory)
    driver = GPBODriver(
        cs_params, method, simulator, exp_data, sim_data, sim_sse_data,
        None, None, None, ep_bias, gen_meth_theta,
    )
    gpbo_res_simple, gpbo_res_GP = driver.run(job=None)

    if verbose:
        print(f"Case study {cs_num}: {meta['name']}  |  method {method_val}: {METHODS[method_val]}")
        true = dict(zip(simulator.theta_true_names, simulator.theta_true))
        print(f"  true parameters: {true}")
        for i, res in enumerate(gpbo_res_simple):
            final = res.results_df.tail(1)
            print(f"  restart {i}: best objective (best_sse_actual) = "
                  f"{final['best_sse_actual'].iloc[0]:.6g}  (why_term={res.why_term})")
    return gpbo_res_simple, gpbo_res_GP
