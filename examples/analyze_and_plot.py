#!/usr/bin/env python3
"""Example: run a case study, then analyze + plot the results — all without signac.

Demonstrates the signac-free analysis path added in the refactor: save BOResults to a
plain directory, wrap it in a ``JobContext`` (a lightweight stand-in for a signac job that
just exposes ``.sp`` and ``.fn()``), and use ``RunAnalysis`` to tabulate results.
Also plots best-SSE vs BO iteration with matplotlib.

Run (from this examples/ directory):  python analyze_and_plot.py
Needs: the 'gpflow' extra. No signac.
"""
import os
import json
import gzip
import pickle
import tempfile

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from common import run_case_study
from emcal.analysis import JobContext, RunAnalysis


def main():
    cs_num, method_val = 1, 7  # Simple Linear, E[SSE]

    # 1. Run the algorithm (signac-free; results in memory).
    res_simple, res_gp = run_case_study(
        cs_num, method_val=method_val, iters=10, runs=1, verbose=False
    )

    # 2. Save results to a plain directory. This mimics a signac job workspace layout
    #    (BO_Results.gz + signac_statepoint.json) but involves no signac.
    ws = tempfile.mkdtemp(prefix="gpbo_example_")
    with gzip.open(os.path.join(ws, "BO_Results.gz"), "wb") as f:
        pickle.dump(res_simple, f)
    with gzip.open(os.path.join(ws, "BO_Results_GPs.gz"), "wb") as f:
        pickle.dump(res_gp, f)
    sp = {"cs_name_val": cs_num, "meth_name_val": method_val,
          "ep_enum_val": 1, "bo_run_num": 1, "bo_run_tot": 1}
    with open(os.path.join(ws, "signac_statepoint.json"), "w") as f:
        json.dump(sp, f)

    # 3. Analyze via a JobContext (no signac). A real signac job would work here too.
    jc = JobContext(ws, sp, job_id="example")
    analyzer = RunAnalysis(
        {"cs_name_val": cs_num, "meth_name_val": method_val},
        project=None, mode="act", save_csv=False,
    )
    df, (theta_true, _bnds) = analyzer.get_run_dataframe(jc)
    print("True parameters:", theta_true)
    print("Tabulated results (tail):")
    print(df[["bo_iter", "best_sse_actual", "theta_best_actual"]].tail().to_string(index=False))

    # 4. Plot best-SSE-so-far vs BO iteration.
    fig, ax = plt.subplots()
    ax.plot(df["bo_iter"], df["best_sse_actual"], marker="o")
    ax.set_xlabel("BO iteration")
    ax.set_ylabel("Best SSE so far")
    ax.set_yscale("log")
    ax.set_title("Simple Linear (CS1): emulator-GPBO convergence")
    out = os.path.join(ws, "convergence.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("Saved convergence plot:", out)


if __name__ == "__main__":
    main()
