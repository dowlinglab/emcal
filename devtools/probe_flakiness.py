#!/usr/bin/env python3
"""Probe GP-training flakiness and determinism on the gpbo-dev stack.

Runs a single (cs, method) config many times in ONE process and reports:
  - failure rate (the float/double dtype InvalidArgumentError)
  - the set of distinct final best-SSE values (determinism gauge)

Optionally applies a runtime monkeypatch (no repo edits):
  --patch nocompile : force gpflow.optimizers.Scipy.minimize(compile=False)
  --patch determinism : enable_op_determinism + fixed tf seed (+ nocompile)

Usage:
  python probe_flakiness.py --cs 1 --meth 1 --iters 3 --trials 6 --patch none
"""
import sys, json, time, argparse, traceback
# emcal is pip-installed (src layout)
import warnings; warnings.simplefilter("ignore")
import numpy as np


def apply_patch(kind):
    import gpflow, tensorflow as tf
    gpflow.config.set_default_float(np.float64)
    if kind in ("nocompile", "determinism"):
        _orig = gpflow.optimizers.Scipy.minimize
        def _patched(self, closure, variables, *a, **kw):
            kw["compile"] = False
            return _orig(self, closure, variables, *a, **kw)
        gpflow.optimizers.Scipy.minimize = _patched
    if kind == "determinism":
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as e:
            print("enable_op_determinism failed:", e)
        tf.random.set_seed(1)


def run_once(cs_val, meth_val, iters, runs):
    from emcal.GPBO_Classes_New import (
        MethodName, EpSchedule, Kernel, GenMethod,
        GPBOMethod, ExplorationBias, BOConfig, GPBODriver,
    )
    from emcal.case_studies import (
        get_case_study, make_case_study_simulator,
    )
    X_VALS = {
        16: np.array([0.0,0.1115,0.2475,0.4076,0.5939,0.8230,0.9214,0.9296,0.985,1.000]),
        17: np.array([0.0087,0.0269,0.0568,0.1556,0.2749,0.4449,0.661,0.8096,0.9309,0.9578]),
    }
    NUM_X = {1:5,2:5,3:5,10:5,11:10,12:10,13:10,14:5,15:10,16:10,17:10}
    method = GPBOMethod(MethodName(meth_val))
    kernel = Kernel(1); ep_enum = EpSchedule(1)
    gmt, gmx = GenMethod(1), GenMethod(2)
    nx = NUM_X[cs_val]; xv = X_VALS.get(cs_val)
    problem = get_case_study(cs_val)
    sim = make_case_study_simulator(problem, 0, None, 1)
    exp = sim.generate_experimental_data(nx, gmx, xv, 0.01)
    ep = ExplorationBias(1, None, ep_enum, None, None, None, None, None, None, None)
    nth = len(sim.indices_to_consider) * 10
    simd = sim.generate_simulation_data(nth, nx, gmt, gmx, 1.0, 1, False, xv, w_noise=False)
    ssed = sim.to_sse_data(method, simd, exp, 1.0, False)
    name = problem.name
    csp = BOConfig(name, 1, 1.0, True, kernel, None, None, 25, 25,
                              False, iters, runs, False, None, 1, 1e-7, 1e-7, True, False)
    drv = GPBODriver(csp, method, sim, exp, simd, ssed, None, None, None, ep, gmt)
    res, _ = drv.run(job=None)
    final = json.loads(res[0].results_df.tail(1).to_json(orient="records"))[0]
    return float(final.get("best_sse_actual"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cs", type=int, default=1)
    p.add_argument("--meth", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--patch", choices=["none","nocompile","determinism"], default="none")
    args = p.parse_args()

    apply_patch(args.patch)
    print(f"probe CS{args.cs} meth{args.meth} iters={args.iters} trials={args.trials} patch={args.patch}")
    vals, fails = [], 0
    for t in range(args.trials):
        t0 = time.time()
        try:
            v = run_once(args.cs, args.meth, args.iters, args.runs)
            vals.append(v)
            print(f"  trial {t}: bestSSE={v!r}  ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            fails += 1
            print(f"  trial {t}: FAIL {type(e).__name__}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"\nsummary patch={args.patch}: {len(vals)}/{args.trials} ok, {fails} failed")
    if vals:
        uniq = sorted(set(f"{v:.10g}" for v in vals))
        print(f"  distinct bestSSE values ({len(uniq)}): {uniq}")
        print(f"  spread: min={min(vals):.10g} max={max(vals):.10g} "
              f"range={max(vals)-min(vals):.3g}")


if __name__ == "__main__":
    main()
