#!/usr/bin/env python3
"""Analysis-layer regression net (signac-free, via JobContext).

The golden gate (verify_against_golden.py) only checks the algorithm's best objective; it
does NOT cover the per-run analysis code that reads DataFrame columns by name. This script
adds that coverage so the analysis-layer column/method renames are safe: it regenerates a
small deterministic CS1 run, saves it as a JobContext workspace, drives the per-run analysis
methods, reduces each output to a stable numeric fingerprint, and compares to a committed
golden (analysis_golden.json). First run writes the golden.

Usage:  python verify_analysis.py          # compare to golden (exit 1 on mismatch)
        python verify_analysis.py --write   # (re)generate golden
"""
import os
import sys
import json
import gzip
import pickle
import tempfile
import argparse
import warnings

warnings.simplefilter("ignore")
import matplotlib
matplotlib.use("Agg")  # headless: must be set before any pyplot import (incl. via emcal.plotting)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "baselines", "analysis_golden.json")


def _fingerprint(obj):
    """Reduce an arbitrary analysis return value to a stable, JSON-able numeric fingerprint."""
    import pandas as pd
    if isinstance(obj, tuple):
        return [_fingerprint(o) for o in obj]
    if isinstance(obj, pd.DataFrame):
        # Drop wall-clock timing columns before summing -- they vary run-to-run and would make the
        # fingerprint nondeterministic. Matching on "time" survives the column rename
        # (Time/Iter -> time_per_iter, Total Run Time -> total_time).
        numeric = obj.select_dtypes(include=[np.number])
        numeric = numeric[[c for c in numeric.columns if "time" not in str(c).lower()]]
        num = numeric.to_numpy(dtype=float)
        # Record the exact column-name SET too: nansum is name-independent, so without this a
        # botched DataFrame column rename (drop/typo/collision) would not be caught. When the §7
        # column renames land, this list changes deliberately and the golden is regenerated.
        return {"kind": "df", "shape": list(obj.shape),
                "columns": sorted(str(c) for c in obj.columns),
                "nansum": round(float(np.nansum(num)), 6)}
    if isinstance(obj, np.ndarray):
        a = obj.astype(float) if obj.dtype != object else None
        return {"kind": "ndarray", "shape": list(obj.shape),
                "nansum": (round(float(np.nansum(a)), 6) if a is not None else None)}
    if isinstance(obj, (list,)):
        return {"kind": "list", "len": len(obj), "fp": [_fingerprint(o) for o in obj[:3]]}
    if isinstance(obj, (int, float, np.floating, np.integer)):
        return {"kind": "scalar", "val": round(float(obj), 6)}
    if isinstance(obj, dict):
        return {"kind": "dict", "keys": sorted(map(str, obj.keys()))}
    # Data (and its typed subclasses): fingerprint the attached prediction arrays so a
    # regression in predict/predict_sse (or the later GPPrediction rewrite that ends the
    # mutate-onto-Data pattern) is caught numerically. nansum per field is name/type-independent.
    if obj.__class__.__name__ in ("Data", "ExperimentalData", "SimulationData",
                                  "ObjectiveData", "CandidateSet"):
        def _ns(a):
            if a is None:
                return None
            arr = np.asarray(a)
            return round(float(np.nansum(arr.astype(float))), 6) if arr.dtype != object else None
        return {"kind": "Data",
                "fields": {f: _ns(getattr(obj, f, None)) for f in
                           ("theta_vals", "x_vals", "y_vals", "gp_mean", "gp_var",
                            "gp_covar", "sse", "sse_var", "sse_covar", "acq")}}
    if obj.__class__.__name__ == "Figure" and obj.__class__.__module__.startswith("matplotlib"):
        # Structural fingerprint (axes/lines/collections/patches counts), not pixel data --
        # stable across matplotlib versions/backends while still catching a plot silently
        # rendering nothing (or a REMOVE-analyzer branch quietly changing what's drawn).
        return {"kind": "figure", "n_axes": len(obj.axes),
                "n_lines": sum(len(ax.lines) for ax in obj.axes),
                "n_collections": sum(len(ax.collections) for ax in obj.axes),
                "n_patches": sum(len(ax.patches) for ax in obj.axes)}
    return {"kind": type(obj).__name__}


def _capture_shown_figure(plot_fn):
    """Call a Plotters method that ends in plt.show()/plt.close() (save_figs=False path) and
    return the Figure it produced, captured just before it would be closed."""
    import matplotlib.pyplot as plt
    captured = {}
    orig_close = plt.close

    def _capture(*a, **kw):
        captured["fig"] = plt.gcf()

    plt.close = _capture
    try:
        plot_fn()
    finally:
        plt.close = orig_close
    fig = captured.get("fig")
    if fig is not None:
        orig_close(fig)
    return fig


def build_fixture():
    """Regenerate a small deterministic CS1 (method 7) run and save as a JobContext workspace."""
    from emcal.GPBO_Classes_New import (
        MethodName, EpSchedule, Kernel, GenMethod,
        GPBOMethod, ExplorationBias, BOConfig, GPBODriver,
    )
    from emcal.case_studies import get_case_study, make_case_study_simulator

    method = GPBOMethod(MethodName(7))
    problem = get_case_study(1)
    sim = make_case_study_simulator(problem, 0, None, 1)
    exp = sim.generate_experimental_data(5, GenMethod(2), None, 0.01)
    ep = ExplorationBias(1, None, EpSchedule(1), None, None, None, None, None, None, None)
    simd = sim.generate_simulation_data(20, 5, GenMethod(1), GenMethod(2), 1.0, 1, False, None, w_noise=False)
    ssed = sim.to_sse_data(method, simd, exp, 1.0, False)
    csp = BOConfig(problem.name, 1, 1.0, True, Kernel(1), None, None,
                              3, 3, False, 3, 1, False, None, 1, 1e-7, 1e-7, True, False)
    drv = GPBODriver(csp, method, sim, exp, simd, ssed, None, None, None, ep, GenMethod(1))
    res_simple, res_gp = drv.run(job=None)

    ws = tempfile.mkdtemp(prefix="gpbo_analysis_fix_")
    with gzip.open(os.path.join(ws, "BO_Results.gz"), "wb") as f:
        pickle.dump(res_simple, f)
    with gzip.open(os.path.join(ws, "BO_Results_GPs.gz"), "wb") as f:
        pickle.dump(res_gp, f)
    # Statepoint keys the per-run analysis methods read. bo_runs_in_job / bo_iter_tot mirror the
    # fixture's bo_run_tot=1 and bo_iter_tot=3 above; parameter_trajectories/hyperparameter_trajectories need them to
    # size their (runs x iters x dim) arrays. w_noise is read by the objective-analysis path.
    sp = {"cs_name_val": 1, "meth_name_val": 7, "ep_enum_val": 1, "bo_run_num": 1,
          "bo_run_tot": 1, "kernel_enum_val": 1, "gen_meth_theta": 1,
          "bo_runs_in_job": 1, "bo_iter_tot": 3, "w_noise": False,
          # Full set of keys read by __rebuild_cs (gp_heat_map_data). These mirror the fixture's
          # BOConfig exactly, so the rebuilt config reproduces the saved run. (Adding them makes the
          # statepoint match the fixture; it also shifts the parameter_/hyperparameter_trajectories
          # fingerprints, which previously ran against a statepoint missing these keys.)
          "ep0": 1, "sep_fact": 1.0, "normalize": True, "lenscl": None, "outputscl": None,
          "retrain_GP": 3, "reoptimize_obj": 3, "gen_heat_map_data": False,
          "seed": 1, "obj_tol": 1e-7, "ei_tol": 1e-7}
    with open(os.path.join(ws, "signac_statepoint.json"), "w") as f:
        json.dump(sp, f)
    return ws


def run_analysis(ws):
    from emcal.analysis import JobContext, General_Analysis
    from emcal.plotting import Plotters
    jc = JobContext(ws, json.load(open(os.path.join(ws, "signac_statepoint.json"))), job_id="fix")
    ga = General_Analysis({"cs_name_val": 1, "meth_name_val": 7}, project=None,
                          mode="act", save_csv=False)
    # save_figs=False so these exercise only the show/close (per-job) path -- the save_figs=True
    # branch of plot_gp_fit calls the REMOVE-slated make_dir_name_from_criteria for path-naming
    # and is not part of what this guard is meant to protect going into the 7B trim.
    plotter = Plotters(ga, save_figs=False)
    # Plotters.plot_parameters -> parameter_trajectories -> __preprocess_analyze reads
    # sp_data["num_theta_multiplier"] (num_train_points = num_theta_multiplier * num_params),
    # a key the 8 existing checks' fixture never needed. That method reads the statepoint from
    # the workspace FILE (job.fn("signac_statepoint.json")), not from job.sp, so the augmented
    # key needs its own workspace dir (symlinked to the same result files) rather than a second
    # JobContext over the same `ws` -- this keeps the original statepoint file, and therefore
    # the 8 existing checks' fingerprints (which embed sp_data's key SET), byte-for-byte
    # unchanged. CS1 has 2 params and the fixture's simd above used 20 theta points -> 10.
    ws_plot_params = tempfile.mkdtemp(prefix="gpbo_analysis_fix_plotparams_")
    for fname in ("BO_Results.gz", "BO_Results_GPs.gz"):
        os.symlink(os.path.join(ws, fname), os.path.join(ws_plot_params, fname))
    sp_for_plot_params = dict(jc.sp)
    sp_for_plot_params["num_theta_multiplier"] = 10
    with open(os.path.join(ws_plot_params, "signac_statepoint.json"), "w") as f:
        json.dump(sp_for_plot_params, f)
    jc_plot_params = JobContext(ws_plot_params, sp_for_plot_params, job_id="fix_plot")
    out = {}
    # parameter_trajectories / hyperparameter_trajectories exercise the per-run column readers signac-free (the columns
    # they read must stay consistent with get_run_dataframe's df across renames). objective_trajectories
    # is intentionally recorded as ok=False: its "objs" path builds LS_Analysis, which needs a live
    # signac project for the least-squares reference values, so it cannot run signac-free. That is the
    # package/paper boundary, not a regression -- the golden simply pins that expectation.
    checks = {
        "get_run_dataframe": lambda: ga.get_run_dataframe(jc),
        "best_error": lambda: ga.best_error(jc),
        "hyperparameter_trajectories": lambda: ga.hyperparameter_trajectories(jc),
        "objective_trajectories": lambda: ga.objective_trajectories(jc, ["min_sse", "sse", "acq"]),
        "parameter_trajectories": lambda: ga.parameter_trajectories(jc, "min_sse"),
        # gp_parity_data re-runs the emulator's predict() on the held-out test split and reads the
        # predictions back off the Data object -- this is the analysis-layer consumer of the
        # mutate-predictions-onto-Data pattern, so it guards the GPPrediction rewrite (design Q3, C).
        "gp_parity_data": lambda: ga.gp_parity_data(jc, 1, 1),
        # gp_heat_map_data reconstructs the config and evaluates predict_sse on a heat-map grid,
        # reading the results back off Data; with get_ei it also runs expected_improvement. These
        # are the other analysis-layer consumers of the mutate-onto-Data pattern, so they guard the
        # GPPrediction rewrite (design Q3, C). Deterministic: they use the SAVED trained GP.
        "gp_heat_map_data": lambda: ga.gp_heat_map_data(jc, 1, 1, 0),
        "gp_heat_map_data_ei": lambda: ga.gp_heat_map_data(jc, 1, 1, 0, get_ei=True),
        # Plotting smoke checks (7B guard): these are the KEEP (per-job) Plotters entry points
        # that will survive the analysis.py/plotting.py trim. They had zero test coverage before
        # this, so this pins their structure (axes/lines/collections counts) ahead of the trim --
        # a regression here would otherwise only surface as a silent blank/broken plot.
        "plot_hyperparameters": lambda: _capture_shown_figure(lambda: plotter.plot_hyperparameters(jc)),
        "plot_parameters": lambda: _capture_shown_figure(
            lambda: plotter.plot_parameters(jc_plot_params, "min_sse")
        ),
        "plot_gp_fit": lambda: _capture_shown_figure(
            lambda: plotter.plot_gp_fit(jc, 1, 1, 0, ["sse_sim", "sse_mean"])
        ),
    }
    for name, fn in checks.items():
        try:
            out[name] = {"ok": True, "fp": _fingerprint(fn())}
        except Exception as e:
            out[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true", help="(re)generate the golden fingerprint file")
    args = p.parse_args()

    ws = build_fixture()
    result = run_analysis(ws)

    if args.write or not os.path.exists(GOLDEN):
        json.dump(result, open(GOLDEN, "w"), indent=2, default=str)
        print(f"WROTE golden -> {GOLDEN}")
        for k, v in result.items():
            print(f"  {k}: ok={v['ok']}" + ("" if v["ok"] else f"  {v['error']}"))
        return

    gold = json.load(open(GOLDEN))
    n_ok = n_bad = 0
    for name, res in result.items():
        g = gold.get(name)
        match = (g is not None and json.dumps(g, sort_keys=True) == json.dumps(res, sort_keys=True))
        print(f"  {name}: {'MATCH' if match else '*** MISMATCH ***'}"
              + ("" if res["ok"] else f"  ({res['error']})"))
        n_ok += int(match); n_bad += int(not match)
    print(f"\nRESULT: {n_ok} match, {n_bad} mismatch")
    sys.exit(0 if n_bad == 0 else 1)


if __name__ == "__main__":
    main()
