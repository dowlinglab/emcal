#!/usr/bin/env python3
"""Example: diagnose a GP emulator BEFORE running Bayesian optimization.

Emulator GPBO only works well when the GP is good -- accurate AND with trustworthy
uncertainty. The `diagnostics` module reports both, using GP best practices:
accuracy (R^2/RMSE/MAE/MAPE) and uncertainty calibration (standardized residuals,
reduced chi^2, interval coverage, NLPD), on a held-out split and via cross-validation,
plus parity / calibration / residual plots.

Run:  python inspect_gp_fit.py
Needs: the 'gpflow' extra (GP backend). No signac, no cluster.
"""
import matplotlib
matplotlib.use("Agg")   # headless-friendly

from emcal import (
    MethodName, Kernel, GenMethod, GPBOMethod, BOConfig,
    get_case_study, make_case_study_simulator, diagnostics,
)


def main():
    problem = get_case_study(1)                      # Simple Linear (CS1)
    method = GPBOMethod(MethodName.INDEPENDENCE)     # an emulator GP (Type-2)

    # Sample experimental + simulation (training) data -- the usual pre-BO steps.
    sim = make_case_study_simulator(problem, 0, None, 1)
    exp_data = sim.generate_experimental_data(5, GenMethod.MESHGRID, None, 0.01)
    n = len(sim.indices_to_consider)
    sim_data = sim.generate_simulation_data(
        10 * n, 5, GenMethod.LHS, GenMethod.MESHGRID, 1.0, 1, False, None, with_noise=False
    )
    sim_sse_data = sim.to_sse_data(method, sim_data, exp_data, 1.0, False)
    config = BOConfig(problem.name, kernel=Kernel.MAT_52, retrain_gp=25, reoptimize_obj=10)

    # 1. Single held-out split: accuracy + uncertainty calibration.
    emulator = diagnostics.fit_gp(
        method, sim, exp_data, sim_data, sim_sse_data, config, sep_fact=0.8
    )
    held_out = diagnostics.evaluate_gp(emulator)
    print(held_out.summary())

    # 2. Cross-validated (leave-N-out style) performance + calibration.
    cv = diagnostics.cross_validate_gp(
        method, sim, exp_data, sim_data, sim_sse_data, config,
        holdout_frac=0.2, n_repeats=5,
    )
    print()
    print(cv.summary())

    # 3. Diagnostic plots: parity (+/-1.96 sigma) | calibration curve | residual QQ.
    fig = held_out.plot_all()
    fig.savefig("gp_diagnostics.png", dpi=120, bbox_inches="tight")
    print("\n  diagnostic plots written to gp_diagnostics.png")
    return held_out, cv


if __name__ == "__main__":
    main()
