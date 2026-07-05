# Golden Regression Baseline

Captured 2026-07-01 on `gpbo-dev` (gpflow 2.9.1 / TF 2.15, macOS arm64). Re-verified bit-for-bit after Phase 2 (backend abstraction + bug fixes).

**Purpose:** deterministic reference values to verify the refactor preserves behavior. NOT paper-fidelity (reduced knobs for speed).

## Config (fixed)

- iters=3, retrain_gp=3, reoptimize_obj=3, runs=1; sim_seed=1, run_seed=1, kernel=Matern5/2, ep0=1
- GP optimizer: gpflow Scipy.minimize(compile=False) (deterministic). Full trajectories in `full/cs<CS>_m<M>.json`

## Result: 77/77 ok (CS12 m6 fixed in Phase 2d).

| CS | Case Study | Method | best obj | why_term | wall s | ok |
|----|-----------|--------|----------|----------|-------:|----|
| 1 | Simple Linear | A1 Conventional | 0.3664014051 | max_budget | 39.19 | Y |
| 1 | Simple Linear | A2 Log Conv | -2.4757204242 | max_budget | 11.46 | Y |
| 1 | Simple Linear | B1 Independence | 12.0973745816 | max_budget | 12.92 | Y |
| 1 | Simple Linear | B2 Log Ind | 12.0973745816 | max_budget | 18.52 | Y |
| 1 | Simple Linear | C2 Sparse Grid | 194.7279963893 | max_budget | 14.48 | Y |
| 1 | Simple Linear | D2 Monte Carlo | 0.0010876321 | max_budget | 24.99 | Y |
| 1 | Simple Linear | E[SSE] | 12.0973745816 | max_budget | 12.16 | Y |
| 2 | Muller x0 | A1 Conventional | 2.0916601644 | max_budget | 43.9 | Y |
| 2 | Muller x0 | A2 Log Conv | 0.9602997322 | max_budget | 33.43 | Y |
| 2 | Muller x0 | B1 Independence | 1.361848163 | max_budget | 280.13 | Y |
| 2 | Muller x0 | B2 Log Ind | 0.3806142744 | max_budget | 1215.36 | Y |
| 2 | Muller x0 | C2 Sparse Grid | 0.1480273007 | max_budget | 369.18 | Y |
| 2 | Muller x0 | D2 Monte Carlo | 0.6103078398 | max_budget | 294.62 | Y |
| 2 | Muller x0 | E[SSE] | 0.0535170919 | max_budget | 218.38 | Y |
| 3 | Muller y0 | A1 Conventional | 1.4142242801 | max_budget | 56.33 | Y |
| 3 | Muller y0 | A2 Log Conv | -0.2522268423 | max_budget | 19.48 | Y |
| 3 | Muller y0 | B1 Independence | 1.0443161125 | max_budget | 234.23 | Y |
| 3 | Muller y0 | B2 Log Ind | 1.0914842456 | max_budget | 1493.9 | Y |
| 3 | Muller y0 | C2 Sparse Grid | 1.2158307947 | max_budget | 317.37 | Y |
| 3 | Muller y0 | D2 Monte Carlo | 1.2158307947 | max_budget | 303.42 | Y |
| 3 | Muller y0 | E[SSE] | 0.9300084841 | max_budget | 248.44 | Y |
| 10 | Large Linear | A1 Conventional | 232.2363339569 | max_budget | 70.15 | Y |
| 10 | Large Linear | A2 Log Conv | 4.5278485301 | max_budget | 15.14 | Y |
| 10 | Large Linear | B1 Independence | 0.0003390547 | max_budget | 646.61 | Y |
| 10 | Large Linear | B2 Log Ind | 0.000338656 | max_budget | 1960.9 | Y |
| 10 | Large Linear | C2 Sparse Grid | 0.0003385464 | max_budget | 594.07 | Y |
| 10 | Large Linear | D2 Monte Carlo | 0.0003409221 | max_budget | 1114.0 | Y |
| 10 | Large Linear | E[SSE] | 0.0003384993 | max_budget | 579.36 | Y |
| 11 | BOD Curve | A1 Conventional | 0.2062147931 | max_budget | 15.85 | Y |
| 11 | BOD Curve | A2 Log Conv | 1.770443159 | max_budget | 9.24 | Y |
| 11 | BOD Curve | B1 Independence | 0.114070769 | max_budget | 26.56 | Y |
| 11 | BOD Curve | B2 Log Ind | 0.1140840968 | max_budget | 108.09 | Y |
| 11 | BOD Curve | C2 Sparse Grid | 0.1139098746 | max_budget | 75.59 | Y |
| 11 | BOD Curve | D2 Monte Carlo | 0.1139508902 | max_budget | 70.19 | Y |
| 11 | BOD Curve | E[SSE] | 0.1139029895 | max_budget | 63.43 | Y |
| 12 | Yield-Loss | A1 Conventional | 0.1217992959 | max_budget | 34.82 | Y |
| 12 | Yield-Loss | A2 Log Conv | 0.6384613241 | max_budget | 13.92 | Y |
| 12 | Yield-Loss | B1 Independence | 4.7583762611 | max_budget | 48.26 | Y |
| 12 | Yield-Loss | B2 Log Ind | 3.713943005 | max_budget | 71.69 | Y |
| 12 | Yield-Loss | C2 Sparse Grid | 4.6162780332 | max_budget | 38.77 | Y |
| 12 | Yield-Loss | D2 Monte Carlo | 4.6162780332 | max_budget | 52.11 | Y |
| 12 | Yield-Loss | E[SSE] | 26.1806776884 | max_budget | 44.51 | Y |
| 13 | Log Logistic | A1 Conventional | 2.760675211 | max_budget | 25.61 | Y |
| 13 | Log Logistic | A2 Log Conv | 0.886751012 | max_budget | 13.07 | Y |
| 13 | Log Logistic | B1 Independence | 0.0010634028 | max_budget | 75.04 | Y |
| 13 | Log Logistic | B2 Log Ind | 0.0020628604 | max_budget | 539.69 | Y |
| 13 | Log Logistic | C2 Sparse Grid | 0.0009248776 | max_budget | 109.97 | Y |
| 13 | Log Logistic | D2 Monte Carlo | 0.0008981294 | max_budget | 98.48 | Y |
| 13 | Log Logistic | E[SSE] | 0.0003606084 | max_budget | 107.29 | Y |
| 14 | 2D Log Logistic | A1 Conventional | 5.0062077671 | max_budget | 22.44 | Y |
| 14 | 2D Log Logistic | A2 Log Conv | 1.0425459767 | max_budget | 18.09 | Y |
| 14 | 2D Log Logistic | B1 Independence | 0.2603686003 | max_budget | 521.4 | Y |
| 14 | 2D Log Logistic | B2 Log Ind | 0.2372149747 | max_budget | 1608.64 | Y |
| 14 | 2D Log Logistic | C2 Sparse Grid | 0.0356183993 | max_budget | 446.09 | Y |
| 14 | 2D Log Logistic | D2 Monte Carlo | 0.1450102336 | max_budget | 408.15 | Y |
| 14 | 2D Log Logistic | E[SSE] | 0.2603686003 | max_budget | 395.3 | Y |
| 15 | Simple Multimodal | A1 Conventional | 503.8604077976 | max_budget | 25.29 | Y |
| 15 | Simple Multimodal | A2 Log Conv | 4.4095893499 | max_budget | 9.93 | Y |
| 15 | Simple Multimodal | B1 Independence | 90.1645922707 | max_budget | 38.49 | Y |
| 15 | Simple Multimodal | B2 Log Ind | 90.1645922707 | max_budget | 95.2 | Y |
| 15 | Simple Multimodal | C2 Sparse Grid | 0.0193800364 | max_budget | 64.73 | Y |
| 15 | Simple Multimodal | D2 Monte Carlo | 0.957872063 | max_budget | 66.94 | Y |
| 15 | Simple Multimodal | E[SSE] | 90.1645922707 | max_budget | 40.34 | Y |
| 16 | Water-Glycerol | A1 Conventional | 72270.2037731791 | max_budget | 16.29 | Y |
| 16 | Water-Glycerol | A2 Log Conv | 6.6608752017 | max_budget | 10.48 | Y |
| 16 | Water-Glycerol | B1 Independence | 452.0342926387 | max_budget | 43.28 | Y |
| 16 | Water-Glycerol | B2 Log Ind | 229.1555950409 | max_budget | 103.98 | Y |
| 16 | Water-Glycerol | C2 Sparse Grid | 200.463658675 | max_budget | 39.72 | Y |
| 16 | Water-Glycerol | D2 Monte Carlo | 368.2184466586 | max_budget | 31.71 | Y |
| 16 | Water-Glycerol | E[SSE] | 391.9018650667 | max_budget | 37.44 | Y |
| 17 | ACN-Water | A1 Conventional | 0.1319744021 | max_budget | 8.93 | Y |
| 17 | ACN-Water | A2 Log Conv | -1.7367693747 | max_budget | 7.97 | Y |
| 17 | ACN-Water | B1 Independence | 0.0826066477 | max_budget | 19.48 | Y |
| 17 | ACN-Water | B2 Log Ind | 0.0992754237 | max_budget | 165.09 | Y |
| 17 | ACN-Water | C2 Sparse Grid | 0.0963432201 | max_budget | 25.98 | Y |
| 17 | ACN-Water | D2 Monte Carlo | 0.0944765977 | max_budget | 36.52 | Y |
| 17 | ACN-Water | E[SSE] | 0.661392687 | max_budget | 29.42 | Y |
