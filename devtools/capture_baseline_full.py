#!/usr/bin/env python3
"""Comprehensive golden-baseline campaign across all case studies x methods.

For each (case study, method): run GPBO signac-free with fixed seeds, retrying on
transient GP-training failures, and save the FULL per-iteration results_df so
regression tests can check the whole trajectory (not just the final value).

Captures:
  - per-config JSON with the complete results_df (records orient) + metadata
  - a master summary.json (status, attempts, wall time, best SSE, best theta)

Stabilization (runtime monkeypatch, no repo edits) via --patch:
  none        : code as-is
  nocompile   : gpflow.optimizers.Scipy.minimize(compile=False)  [+ float64]
  determinism : nocompile + enable_op_determinism + tf seed

Determinism check: --recheck CS:METH,CS:METH re-runs those configs a 2nd time and
records whether the final best-SSE matches bit-for-bit.

Usage:
  python capture_baseline_full.py --methods 1,2,3,4,5,6,7 --iters 3 \
      --patch nocompile --retries 3 --out-dir emcal/devtools/baselines/full
"""
import sys, os, json, time, argparse, traceback
import warnings; warnings.simplefilter("ignore")  # emcal is pip-installed (src layout)
import numpy as np

ALL_CS = [1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17]
NUM_X_DATA = {1:5,2:5,3:5,10:5,11:10,12:10,13:10,14:5,15:10,16:10,17:10}
X_VALS = {
    16: np.array([0.0,0.1115,0.2475,0.4076,0.5939,0.8230,0.9214,0.9296,0.985,1.000]),
    17: np.array([0.0087,0.0269,0.0568,0.1556,0.2749,0.4449,0.661,0.8096,0.9309,0.9578]),
}
SIM_SEED = RUN_SEED = 1


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
            print("enable_op_determinism failed:", e, flush=True)
        tf.random.set_seed(1)


def build_and_run(cs_val, meth_val, iters, runs, retrain_gp=25, reopt_obj=25):
    from emcal.GPBO_Classes_New import (
        MethodName, EpSchedule, Kernel, GenMethod,
        GPBOMethod, ExplorationBias, BOConfig, GPBODriver,
    )
    from emcal.case_studies import (
        get_case_study, make_case_study_simulator,
    )
    method = GPBOMethod(MethodName(meth_val))
    kernel = Kernel(1); ep_enum = EpSchedule(1)
    gmt, gmx = GenMethod(1), GenMethod(2)
    nx = NUM_X_DATA[cs_val]; xv = X_VALS.get(cs_val)
    problem = get_case_study(cs_val)
    sim = make_case_study_simulator(problem, 0, None, SIM_SEED)
    exp = sim.generate_experimental_data(nx, gmx, xv, 0.01)
    ep = ExplorationBias(1, None, ep_enum, None, None, None, None, None, None, None)
    nth = len(sim.indices_to_consider) * 10
    simd = sim.generate_simulation_data(nth, nx, gmt, gmx, 1.0, SIM_SEED, False, xv, with_noise=False)
    ssed = sim.to_sse_data(method, simd, exp, 1.0, False)
    name = problem.name
    csp = BOConfig(name, 1, 1.0, True, kernel, None, None, retrain_gp, reopt_obj,
                              False, iters, runs, False, None, RUN_SEED, 1e-7, 1e-7, True, False)
    drv = GPBODriver(csp, method, sim, exp, simd, ssed, None, None, None, ep, gmt)
    res, _ = drv.run(job=None)
    out_runs = []
    for i, r in enumerate(res):
        df = r.results_df
        out_runs.append({
            "run": i,
            "why_term": str(getattr(r, "why_term", None)),
            "results_df": json.loads(df.to_json(orient="records")),
        })
    return {"cs_val": cs_val, "cs_name": name, "method_val": meth_val,
            "method": method.method_name.name, "emulator": bool(method.is_emulator),
            "num_x_data": nx, "iters": iters, "runs": runs, "runs_out": out_runs}


def final_best(rec):
    try:
        return float(rec["runs_out"][0]["results_df"][-1].get("best_sse_actual"))
    except Exception:
        return None


def run_with_retries(cs_val, meth_val, iters, runs, retries, retrain_gp=25, reopt_obj=25):
    last = None
    for attempt in range(1, retries + 1):
        t0 = time.time()
        try:
            rec = build_and_run(cs_val, meth_val, iters, runs, retrain_gp, reopt_obj)
            rec.update(ok=True, attempts=attempt, wall_sec=round(time.time()-t0, 2))
            return rec
        except Exception as e:
            last = {"cs_val": cs_val, "method_val": meth_val, "ok": False,
                    "attempts": attempt, "error": repr(e),
                    "traceback": traceback.format_exc(), "wall_sec": round(time.time()-t0, 2)}
            print(f"      attempt {attempt}/{retries} failed: {type(e).__name__}", flush=True)
    return last


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", type=str, default="1,2,3,4,5,6,7")
    p.add_argument("--cs", type=str, default=",".join(map(str, ALL_CS)))
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--retrain", type=int, default=25, help="retrain_gp per iteration")
    p.add_argument("--reopt", type=int, default=25, help="reoptimize_obj per iteration")
    p.add_argument("--patch", choices=["none","nocompile","determinism"], default="nocompile")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--recheck", type=str, default="1:1,11:1,1:5",
                   help="comma list CS:METH to re-run once for determinism check")
    p.add_argument("--out-dir", type=str, default="emcal/devtools/baselines/full")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip a config if its per-config JSON already exists with ok=True")
    p.add_argument("--order", choices=["declared", "fast", "slow"], default="declared",
                   help="config run order. 'fast' = ascending recorded wall_sec (fail-fast: "
                        "surfaces failures early before slow configs run); 'slow' = descending; "
                        "'declared' = cs x methods nested order. Timings read from --order-ref.")
    p.add_argument("--order-ref", type=str,
                   default="emcal/devtools/baselines/full",
                   help="dir of reference per-config JSONs (with wall_sec) used to order configs")
    args = p.parse_args()

    apply_patch(args.patch)
    methods = [int(x) for x in args.methods.split(",")]
    cs_list = [int(x) for x in args.cs.split(",")]
    recheck = set()
    for tok in args.recheck.split(","):
        if ":" in tok:
            c, m = tok.split(":"); recheck.add((int(c), int(m)))
    os.makedirs(args.out_dir, exist_ok=True)

    meta = {"iters": args.iters, "runs": args.runs, "methods": methods, "cs_list": cs_list,
            "patch": args.patch, "retries": args.retries, "retrain_gp": args.retrain,
            "reopt_obj": args.reopt, "sim_seed": SIM_SEED, "run_seed": RUN_SEED}
    # Build the (cs, method) work list, optionally ordered by recorded runtime.
    pairs = [(cs_val, meth_val) for cs_val in cs_list for meth_val in methods]
    if args.order != "declared":
        def _wall(cs_val, meth_val):
            ref = os.path.join(args.order_ref, f"cs{cs_val}_m{meth_val}.json")
            try:
                w = json.load(open(ref)).get("wall_sec")
                return float(w) if w is not None else float("inf")
            except Exception:
                return float("inf")  # unknown timing (e.g. new config) -> run last
        pairs.sort(key=lambda cm: _wall(*cm), reverse=(args.order == "slow"))
        log_order = ", ".join(f"CS{c}m{m}" for c, m in pairs[:5])
        print(f"order={args.order}: first few -> {log_order} ...", flush=True)

    meta["order"] = args.order
    summary = []
    total = len(pairs); n = 0
    t_start = time.time()
    for cs_val, meth_val in pairs:
            n += 1
            tag = f"[{n}/{total}] CS{cs_val} m{meth_val}"
            cfg_path = os.path.join(args.out_dir, f"cs{cs_val}_m{meth_val}.json")
            if args.skip_existing and os.path.exists(cfg_path):
                try:
                    prev = json.load(open(cfg_path))
                    if prev.get("ok"):
                        print(f"{tag} SKIP (already ok)", flush=True)
                        b = final_best(prev)
                        summary.append({"cs_val": cs_val, "method_val": meth_val, "ok": True,
                                        "attempts": prev.get("attempts"), "wall_sec": prev.get("wall_sec"),
                                        "best_sse": b, "skipped": True,
                                        "why_term": prev["runs_out"][0]["why_term"]})
                        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
                            json.dump({"meta": meta, "summary": summary}, f, indent=2, default=str)
                        continue
                except Exception:
                    pass
            print(f"{tag} ...", flush=True)
            rec = run_with_retries(cs_val, meth_val, args.iters, args.runs, args.retries,
                                   args.retrain, args.reopt)
            # write full per-config record
            cfg_path = os.path.join(args.out_dir, f"cs{cs_val}_m{meth_val}.json")
            with open(cfg_path, "w") as f:
                json.dump(rec, f, indent=2, default=str)
            best = final_best(rec) if rec.get("ok") else None
            s = {"cs_val": cs_val, "method_val": meth_val, "ok": rec.get("ok", False),
                 "attempts": rec.get("attempts"), "wall_sec": rec.get("wall_sec"),
                 "best_sse": best, "why_term": (rec["runs_out"][0]["why_term"] if rec.get("ok") else None)}
            # determinism recheck
            if (cs_val, meth_val) in recheck and rec.get("ok"):
                rec2 = run_with_retries(cs_val, meth_val, args.iters, args.runs, args.retries,
                                        args.retrain, args.reopt)
                best2 = final_best(rec2) if rec2.get("ok") else None
                s["recheck_best_sse"] = best2
                s["deterministic"] = (best is not None and best2 is not None and
                                      repr(best) == repr(best2))
                print(f"{tag} recheck: best={best!r} best2={best2!r} "
                      f"deterministic={s['deterministic']}", flush=True)
            print(f"{tag} ok={s['ok']} attempts={s['attempts']} best={best} ({s['wall_sec']}s)", flush=True)
            summary.append(s)
            with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
                json.dump({"meta": meta, "summary": summary}, f, indent=2, default=str)

    ok = sum(1 for s in summary if s["ok"])
    print(f"\nDONE {ok}/{len(summary)} ok in {round(time.time()-t_start)}s. Out: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
