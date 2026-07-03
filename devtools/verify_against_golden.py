#!/usr/bin/env python3
"""Rename-robust golden gate for the class redesign.

Runs configs through the CURRENT code (via capture_baseline_full, which is updated to the
current class/method/column names as the refactor proceeds) and compares the best-objective
number to the committed name-independent reference (golden_metrics.json). Because the
reference is pure numbers, this survives class/method/DataFrame-column renames — only
capture_baseline_full's `final_best` accessor tracks the current column name.

Usage:
  python verify_against_golden.py --configs 1:1,1:5,2:1,13:3   # specific cs:method
  python verify_against_golden.py --fast 12                    # 12 fastest configs (by golden wall_sec)
  python verify_against_golden.py --all                        # all 77 (slow)
Exit code 0 iff every run matches its golden number (exact repr; nocompile is deterministic).
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import capture_baseline_full as cap  # noqa: E402  (uses current names/columns)

GOLDEN_DIR = os.path.join(HERE, "baselines")
METRICS = json.load(open(os.path.join(GOLDEN_DIR, "golden_metrics.json")))["metrics"]


def _fast_configs(n):
    # order configs by recorded wall_sec from the committed baseline (fastest first)
    import glob
    timed = []
    for f in glob.glob(os.path.join(GOLDEN_DIR, "full", "cs*.json")):
        d = json.load(open(f))
        timed.append((d["cs_val"], d["method_val"], d.get("wall_sec") or 1e9))
    timed.sort(key=lambda t: t[2])
    return [(c, m) for c, m, _ in timed[:n]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", type=str, help="comma list of cs:method, e.g. 1:1,1:5")
    p.add_argument("--fast", type=int, help="verify the N fastest configs")
    p.add_argument("--all", action="store_true", help="verify all 77 configs")
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--retrain", type=int, default=3)
    p.add_argument("--reopt", type=int, default=3)
    args = p.parse_args()

    if args.configs:
        pairs = [tuple(int(x) for x in tok.split(":")) for tok in args.configs.split(",")]
    elif args.fast:
        pairs = _fast_configs(args.fast)
    elif args.all:
        pairs = _fast_configs(77)
    else:
        # default quick gate: covers all 7 methods (CS1) + a 4-param case + pyomo path
        pairs = [(1, m) for m in range(1, 8)] + [(13, 3), (2, 1)]

    cap.apply_patch("none")  # compile=False is baked into the code post-Phase-2a
    n_ok = n_bad = 0
    bad = []
    for cs, m in pairs:
        key = f"cs{cs}_m{m}"
        gold = METRICS.get(key, {}).get("best_obj")
        rec = cap.run_with_retries(cs, m, args.iters, 1, 2, args.retrain, args.reopt)
        new = cap.final_best(rec) if rec.get("ok") else None
        match = (new is not None and gold is not None and repr(new) == repr(gold))
        status = "MATCH" if match else "*** MISMATCH ***"
        print(f"  {key}: new={new!r} golden={gold!r} {status}", flush=True)
        if match:
            n_ok += 1
        else:
            n_bad += 1
            bad.append(key)
    print(f"\nRESULT: {n_ok} match, {n_bad} mismatch" + (f" -> {bad}" if bad else ""))
    sys.exit(0 if n_bad == 0 else 1)


if __name__ == "__main__":
    main()
