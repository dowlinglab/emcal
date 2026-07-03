#!/usr/bin/env python3
"""Example: Water-Glycerol VLE (CS16), signac-free.

Paper case study "Water-Glycerol VLE" (Carlozo, Wang & Dowling, Ind. Eng. Chem. Res. 2025, Table 2).
UNIQUAC vapor-liquid equilibrium; uses fixed experimental x-grid.

Runs emulator GPBO (method 7 = E[SSE]) via the shared helper; see run_simple_linear.py
for the same recipe written out inline.

Run (from this examples/ directory):  python run_water_glycerol_vle.py
"""
from common import run_case_study

if __name__ == "__main__":
    run_case_study(cs_num=16, method_val=7, iters=10, runs=1)
