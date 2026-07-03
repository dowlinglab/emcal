#!/usr/bin/env python3
"""Example: Muller x0 (CS2), signac-free.

Paper case study "Muller x0" (Carlozo, Wang & Dowling, Ind. Eng. Chem. Res. 2025, Table 2).
Solves a Pyomo model of the Müller potential — needs the 'muller' extra (ipopt on PATH).

Runs emulator GPBO (method 7 = E[SSE]) via the shared helper; see run_simple_linear.py
for the same recipe written out inline.

Run (from this examples/ directory):  python run_muller_x0.py
"""
from common import run_case_study

if __name__ == "__main__":
    run_case_study(cs_num=2, method_val=7, iters=10, runs=1)
