"""Enumerations for GPBO configuration: method (MethodName), GP kernel (Kernel),
sampling scheme (GenMethod), and exploration-bias schedule (EpSchedule).
"""
from enum import Enum


class MethodName(Enum):
    """
    GPBO method identifiers.

    Members are named for the method's role; the paper's Table-1 label (A1, B1,
    ...) is given in the comment beside each (Carlozo, Wang & Dowling, IECR 2025).

    Notes
    -----
    - The "conventional" methods fit a GP directly to the SSE objective (Type-1).
    - The others are emulator methods: the GP models the model output and the
      SSE / expected improvement is derived from it (Type-2).
    """

    CONVENTIONAL = 1      # A1: conventional GPBO, SSE objective
    LOG_CONVENTIONAL = 2  # B1: conventional GPBO, ln(SSE) objective
    INDEPENDENCE = 3      # A2: emulator GPBO, independence-approx. EI
    LOG_INDEPENDENCE = 4  # B2: emulator GPBO, log independence-approx. EI
    SPARSE_GRID = 5       # C2: emulator GPBO, sparse-grid integrated EI
    MONTE_CARLO = 6       # D2: emulator GPBO, Monte-Carlo integrated EI
    EXPECTED_SSE = 7      # A3: emulator GPBO, E[SSE] acquisition


class Kernel(Enum):
    """
    Base class for kernel choices

    Notes
    -------
    1 = Matern 52
    2 = Matern 32
    3 = RBF
    """

    MAT_52 = 1
    MAT_32 = 2
    RBF = 3


class GenMethod(Enum):
    """
    The base class for any GPBO Method names

    Notes
    -------
    1 = LHS
    2 = Meshgrid
    """

    LHS = 1
    MESHGRID = 2


class EpSchedule(Enum):
    """
    The base class for any Method for calculating the decay of the exploration parameter

    Notes
    -------
    1 = Constant
    2 = Decay
    3 = Boyle
    4 = Jasrasaria
    """

    CONSTANT = 1
    DECAY = 2
    BOYLE = 3
    JASRASARIA = 4
