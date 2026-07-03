#!/usr/bin/env python3
"""Example: BOD Curve (CS11), signac-free.

Paper case study "BOD Curve" (Carlozo, Wang & Dowling, Ind. Eng. Chem. Res. 2025, Table 2).
Biochemical oxygen demand: f = theta_1*(1 - exp(-theta_2*x)).

Runs emulator GPBO (method 7 = E[SSE]) via the shared helper; see run_simple_linear.py
for the same recipe written out inline.

Run (from this examples/ directory):  python run_bod_curve.py
"""
from common import run_case_study

if __name__ == "__main__":
    run_case_study(cs_num=11, method_val=7, iters=10, runs=1)
