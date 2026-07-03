#!/usr/bin/env python3
"""Example: Large Linear (CS10), signac-free.

Paper case study "Large Linear" (Carlozo, Wang & Dowling, Ind. Eng. Chem. Res. 2025, Table 2).
5-parameter linear-in-parameters model (the highest-dimensional case study).

Runs emulator GPBO (method 7 = E[SSE]) via the shared helper; see run_simple_linear.py
for the same recipe written out inline.

Run (from this examples/ directory):  python run_large_linear.py
"""
from common import run_case_study

if __name__ == "__main__":
    run_case_study(cs_num=10, method_val=7, iters=10, runs=1)
