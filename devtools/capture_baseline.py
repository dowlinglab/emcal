#!/usr/bin/env python3
"""Capture golden regression baseline across all 11 case studies.

Runs GPBODriver.run(job=None) (signac-free) with a small fixed budget
and fixed seeds, recording the final best-theta / best-SSE per (case study, method).
Each run is isolated in try/except so one failure does not halt the sweep.

Mirrors the GPBO_nonoise statepoint config (init_gpbo_nonoise.py /
project_gpbononoise.py), including per-CS num_x_data and the fixed x-grids for the
VLE case studies (CS16, CS17).

Usage:
    python capture_baseline.py [--iters N] [--runs N] [--methods 1,4,5,7] [--cs 1,2,...]
Output: JSON to --out (default /tmp/gpbo_baseline/baseline.json), written incrementally.
"""
import sys, os, json, time, traceback, argparse
# emcal is pip-installed (src layout)
import warnings
warnings.simplefilter("ignore")

import numpy as np
from emcal.GPBO_Classes_New import (
    MethodName, EpSchedule, Kernel, GenMethod,
    GPBOMethod, ExplorationBias, BOConfig, GPBODriver,
)
from emcal.case_studies import (
    get_case_study, make_case_study_simulator,
)

# ---- per-CS config from init_gpbo_nonoise.py ----
NUM_X_DATA = {1: 5, 2: 5, 3: 5, 10: 5, 11: 10, 12: 10, 13: 10,
              14: 5, 15: 10, 16: 10, 17: 10}
X_VALS = {
    16: np.array([0.0, 0.1115, 0.2475, 0.4076, 0.5939, 0.8230, 0.9214, 0.9296, 0.985, 1.000]),
    17: np.array([0.0087, 0.0269, 0.0568, 0.1556, 0.2749, 0.4449, 0.661, 0.8096, 0.9309, 0.9578]),
}
ALL_CS = [1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17]

# Fixed statepoint defaults (nonoise)
EP0, EP_ENUM_VAL = 1, 1
SEP_FACT = 1.0
NORMALIZE = True
GEN_HEAT_MAP = False
NOISE_MEAN, NOISE_STD, NOISE_STD_PCT = 0, None, 0.01
KERNEL_ENUM_VAL = 1          # Matern 5/2
LENSCL = OUTPUTSCL = None
RETRAIN_GP = REOPT_OBJ = 25
SAVE_DATA = False
EI_TOL = OBJ_TOL = 1e-7
GEN_METH_THETA_I, GEN_METH_X_I = 1, 2   # LHS thetas, grid x
NUM_THETA_MULT = 10
GET_Y_SSE = True
GEN_Y_W_NOISE = False
SIM_SEED, RUN_SEED = 1, 1


def run_one(cs_val, meth_val, iters, runs):
    method = GPBOMethod(MethodName(meth_val))
    ep_enum = EpSchedule(EP_ENUM_VAL)
    kernel = Kernel(KERNEL_ENUM_VAL)
    gen_meth_theta = GenMethod(GEN_METH_THETA_I)
    gen_meth_x = GenMethod(GEN_METH_X_I)
    num_x_data = NUM_X_DATA[cs_val]
    x_vals = X_VALS.get(cs_val, None)

    problem = get_case_study(cs_val)
    simulator = make_case_study_simulator(problem, NOISE_MEAN, NOISE_STD, SIM_SEED)
    exp_data = simulator.generate_experimental_data(num_x_data, gen_meth_x, x_vals, NOISE_STD_PCT)
    ep_bias = ExplorationBias(EP0, None, ep_enum, None, None, None, None, None, None, None)

    num_theta_data = len(simulator.indices_to_consider) * NUM_THETA_MULT
    sim_data = simulator.generate_simulation_data(
        num_theta_data, num_x_data, gen_meth_theta, gen_meth_x,
        SEP_FACT, SIM_SEED, False, x_vals, with_noise=GEN_Y_W_NOISE,
    )
    sim_sse_data = simulator.to_sse_data(method, sim_data, exp_data, SEP_FACT, False)

    cs_name = problem.name
    cs_params = BOConfig(
        cs_name, EP0, SEP_FACT, NORMALIZE, kernel, LENSCL, OUTPUTSCL,
        RETRAIN_GP, REOPT_OBJ, GEN_HEAT_MAP, iters, runs,
        SAVE_DATA, None, RUN_SEED, OBJ_TOL, EI_TOL, GET_Y_SSE, GEN_Y_W_NOISE,
    )
    driver = GPBODriver(
        cs_params, method, simulator, exp_data, sim_data, sim_sse_data,
        None, None, None, ep_bias, gen_meth_theta,
    )
    t0 = time.time()
    gpbo_res_simple, _ = driver.run(job=None)
    elapsed = time.time() - t0

    runs_out = []
    for i, res in enumerate(gpbo_res_simple):
        final = json.loads(res.results_df.tail(1).to_json(orient="records"))[0]
        runs_out.append({
            "run": i,
            "why_term": str(getattr(res, "why_term", None)),
            "final_row": final,
        })
    return {
        "cs_val": cs_val, "cs_name": cs_name,
        "method_val": meth_val, "method": method.method_name.name,
        "emulator": bool(method.is_emulator),
        "num_x_data": num_x_data, "iters": iters, "runs": runs,
        "wall_sec": round(elapsed, 2), "runs_out": runs_out, "ok": True,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--methods", type=str, default="1,4,5,7")
    p.add_argument("--cs", type=str, default=",".join(map(str, ALL_CS)))
    p.add_argument("--out", type=str, default="/tmp/gpbo_baseline/baseline.json")
    args = p.parse_args()

    methods = [int(x) for x in args.methods.split(",")]
    cs_list = [int(x) for x in args.cs.split(",")]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    meta = {"iters": args.iters, "runs": args.runs, "methods": methods,
            "cs_list": cs_list, "sim_seed": SIM_SEED, "run_seed": RUN_SEED}
    results = []
    total = len(cs_list) * len(methods)
    n = 0
    for cs_val in cs_list:
        for meth_val in methods:
            n += 1
            tag = f"[{n}/{total}] CS{cs_val} method {meth_val}"
            print(f"{tag} ... ", flush=True)
            try:
                rec = run_one(cs_val, meth_val, args.iters, args.runs)
                bested = rec["runs_out"][0]["final_row"].get("best_sse_actual")
                print(f"{tag} OK  ({rec['wall_sec']}s)  bestSSE={bested}", flush=True)
            except Exception as e:
                rec = {"cs_val": cs_val, "method_val": meth_val, "ok": False,
                       "error": repr(e), "traceback": traceback.format_exc()}
                print(f"{tag} FAILED: {e!r}", flush=True)
            results.append(rec)
            # incremental write
            with open(args.out, "w") as f:
                json.dump({"meta": meta, "results": results}, f, indent=2, default=str)

    ok = sum(1 for r in results if r.get("ok"))
    print(f"\nDONE: {ok}/{len(results)} runs succeeded. Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
