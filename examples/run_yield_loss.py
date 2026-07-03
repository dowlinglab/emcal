#!/usr/bin/env python3
"""Example: Yield-Loss (CS12), signac-free.

Paper case study "Yield-Loss" (Carlozo, Wang & Dowling, Ind. Eng. Chem. Res. 2025, Table 2).
Large-magnitude objective; exercises the Monte-Carlo numerics fix (see refactor_notes).

Runs emulator GPBO (method 7 = E[SSE]) via the shared helper; see run_simple_linear.py
for the same recipe written out inline.

Run (from this examples/ directory):  python run_yield_loss.py
"""
from common import run_case_study

if __name__ == "__main__":
    run_case_study(cs_num=12, method_val=7, iters=10, runs=1)
