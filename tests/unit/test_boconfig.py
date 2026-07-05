"""Unit tests for the BOConfig dataclass (formerly CaseStudyParameters)."""
import numpy as np
import pytest

from emcal import BOConfig, Kernel


def _valid_kwargs(**over):
    kw = dict(
        cs_name="CS1", ep0=1, sep_fact=1.0, normalize=True, kernel=Kernel(1),
        lenscl=None, outputscl=None, retrain_gp=3, reoptimize_obj=3,
        gen_heat_map_data=False, bo_iter_tot=3, bo_run_tot=1, save_data=False,
        created_at=None, seed=1, obj_tol=1e-7, acq_tol=1e-7, compute_y_sse=True,
        with_noise=False,
    )
    kw.update(over)
    return kw


def test_defaults_and_seed_field():
    c = BOConfig()
    assert c.seed == 1                 # set_seed -> seed field
    assert c.kernel == Kernel(1)
    assert c.created_at is not None      # filled in when None


def test_positional_construction_matches_fields():
    # 19 positional args (the order used across examples/devtools).
    c = BOConfig("CS1", 1, 1.0, True, Kernel(1), None, None, 3, 3, False,
                 3, 1, False, None, 5, 1e-7, 1e-7, True, False)
    assert c.cs_name == "CS1"
    assert c.seed == 5
    assert c.compute_y_sse is True
    assert c.with_noise is False


def test_lenscl_list_becomes_array():
    c = BOConfig(**_valid_kwargs(lenscl=[1.0, 2.0]))
    assert isinstance(c.lenscl, np.ndarray)
    assert list(c.lenscl) == [1.0, 2.0]


@pytest.mark.parametrize("bad", [
    dict(sep_fact=5.0),      # must be in (0, 1]
    dict(sep_fact=0.0),      # not including zero
    dict(bo_iter_tot=0),     # must be > 0
    dict(retrain_gp=-1),     # must be >= 0
    dict(seed=0),            # int >= 1 or None
    dict(outputscl=-1.0),    # > 0 if given
    dict(acq_tol=-1e-7),     # >= 0
])
def test_validation_rejects_bad_values(bad):
    with pytest.raises(AssertionError):
        BOConfig(**_valid_kwargs(**bad))
