"""Guards the module split: every class is importable from its own submodule, and the
GPBO_Classes_New facade re-exports the SAME objects (so old import paths keep working)."""
import importlib

import pytest

# submodule -> classes that must live there after the split
LAYOUT = {
    "enums": ["MethodName", "Kernel", "GenMethod", "EpSchedule"],
    "exploration": ["ExplorationBias"],
    "config": ["BOConfig"],
    "data": ["Data"],
    "methods": ["GPBOMethod"],
    "simulator": ["Simulator"],
    "acquisition": ["ExpectedImprovement"],
    "emulators": ["GPEmulator", "ObjectiveGP", "EmulatorGP"],
    "results": ["BOResults"],
    "driver": ["GPBODriver"],
}


@pytest.mark.parametrize("submodule,classes", list(LAYOUT.items()))
def test_class_lives_in_submodule(submodule, classes):
    mod = importlib.import_module(f"emcal.{submodule}")
    for cls in classes:
        assert hasattr(mod, cls), f"{cls} missing from emcal.{submodule}"


def test_facade_reexports_same_objects():
    # GPBO_Classes_New is now a thin facade; the objects must be identical (is) to the
    # ones defined in the real submodules, so any old `from .GPBO_Classes_New import X` works.
    facade = importlib.import_module("emcal.GPBO_Classes_New")
    for submodule, classes in LAYOUT.items():
        mod = importlib.import_module(f"emcal.{submodule}")
        for cls in classes:
            assert getattr(facade, cls) is getattr(mod, cls), f"facade {cls} != {submodule}.{cls}"


def test_case_study_helpers_in_case_studies_module():
    cs = importlib.import_module("emcal.case_studies")
    for name in ["CalibrationProblem", "get_case_study", "make_case_study_simulator"]:
        assert hasattr(cs, name)


def test_public_api_matches_all():
    import emcal
    for name in emcal.__all__:
        assert hasattr(emcal, name), f"__all__ lists {name} but it is not importable"
