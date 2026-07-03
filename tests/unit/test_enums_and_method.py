"""Unit tests for the enums and the GPBOMethod descriptor (redesigned API).

Imports go through the package top-level so these survive the planned module split.
"""
import pytest

from emcal import MethodName, Kernel, GenMethod, EpSchedule, GPBOMethod


def test_method_name_members_and_values():
    # Descriptive members map to the paper's Table-1 order (values 1-7, unchanged).
    assert [m.value for m in MethodName] == [1, 2, 3, 4, 5, 6, 7]
    assert MethodName(1) is MethodName.CONVENTIONAL
    assert MethodName(2) is MethodName.LOG_CONVENTIONAL
    assert MethodName(5) is MethodName.SPARSE_GRID
    assert MethodName(6) is MethodName.MONTE_CARLO
    assert MethodName(7) is MethodName.EXPECTED_SSE


def test_objective_enum_removed():
    # design Q2: the Objective enum was dropped in favour of GPBOMethod.log_scaled.
    import emcal

    assert not hasattr(emcal, "Objective")


# (method value, is_emulator, log_scaled, uses_sparse_grid, uses_monte_carlo, report_name)
_EXPECTED = [
    (1, False, False, False, False, "Conventional"),
    (2, False, True, False, False, "Log Conventional"),
    (3, True, False, False, False, "Independence"),
    (4, True, False, False, False, "Log Independence"),
    (5, True, False, True, False, "Sparse Grid"),
    (6, True, False, False, True, "Monte Carlo"),
    (7, True, False, False, False, "E[SSE]"),
]


@pytest.mark.parametrize("val,is_emu,log_scaled,sparse,mc,report", _EXPECTED)
def test_gpbomethod_properties(val, is_emu, log_scaled, sparse, mc, report):
    m = GPBOMethod(MethodName(val))
    assert m.is_emulator is is_emu
    assert m.log_scaled is log_scaled
    assert m.uses_sparse_grid is sparse
    assert m.uses_monte_carlo is mc
    assert m.report_name == report


def test_gpbomethod_rejects_non_enum():
    with pytest.raises(AssertionError):
        GPBOMethod(1)  # must be a MethodName instance


def test_other_enum_members():
    assert [k.name for k in Kernel] == ["MAT_52", "MAT_32", "RBF"]
    assert [g.name for g in GenMethod] == ["LHS", "MESHGRID"]
    assert [e.name for e in EpSchedule] == ["CONSTANT", "DECAY", "BOYLE", "JASRASARIA"]
