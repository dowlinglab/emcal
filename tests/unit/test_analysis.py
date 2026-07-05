"""Unit tests for emcal.analysis (per-run/diagnostic analysis, signac-free via JobContext)."""
import warnings

import pytest

from emcal import analysis as analysis_module
from emcal.analysis import RunAnalysis


def _ga(mode="act"):
    return RunAnalysis({"cs_name_val": 1}, project=None, mode=mode, save_csv=False)


def test_z_choice_helper_warns_not_nameerror_on_invalid_choice(monkeypatch):
    """Pins the analysis.py:559 bug fix: __z_choice_helper's else branch calls
    warnings.warn(...), but the module never imported `warnings` -- if that branch ever
    ran, it raised NameError instead of warning. The branch is normally unreachable in
    production (the method's own `any(...)` assert only lets 'acq'/'min_sse'/'sse' through,
    and all three are handled by the if/elif chain), so this test shadows the module-global
    `any` (a standard technique: LOAD_GLOBAL resolves against the function's __globals__,
    i.e. the module dict, before falling back to builtins) to exercise the defensive else
    branch directly and confirm it now warns cleanly instead of NameError'ing.

    The else branch never assigns `col_name` before the function returns, so an
    UnboundLocalError still follows the warning -- that's a separate, pre-existing gap in
    this dead branch, out of scope for this fix; asserted here only so the test pins exactly
    the fixed behavior (warns) without masking the unrelated one.
    """
    ga = _ga()
    monkeypatch.setattr(analysis_module, "any", lambda *a, **kw: True, raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(UnboundLocalError):
            ga._RunAnalysis__z_choice_helper("bogus", {"a": 1}, "params")

    assert len(caught) == 1
    assert "z_choices must be" in str(caught[0].message)


def test_make_dir_name_from_criteria_is_order_independent():
    """Pins the analysis.py:227-229 fix: `sorted_dict` was computed (per the docstring's
    intent, "Organize Dictionary keys and values sorted from lowest to highest") but the
    loop below it iterated the original, unsorted `dict_to_use` instead -- so two criteria
    dicts with identical content but different key-insertion order produced different
    directory names. Now both orders produce the same result.
    """
    ga_a = RunAnalysis({"b": 2, "a": 1}, project=None, mode="act", save_csv=False)
    ga_b = RunAnalysis({"a": 1, "b": 2}, project=None, mode="act", save_csv=False)

    name_a = ga_a.make_dir_name_from_criteria({"b": 2, "a": 1})
    name_b = ga_b.make_dir_name_from_criteria({"a": 1, "b": 2})

    assert name_a == name_b
    # Sanity: the sorted key ("a") appears before "b" in the resulting path.
    assert name_a.index("a_1") < name_a.index("b_2")
