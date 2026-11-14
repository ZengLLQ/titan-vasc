"""Automated regression tests verifying all standalone examples run cleanly."""

from pathlib import Path
import importlib.util
import sys
import pytest


def _run_example_file(filename: str) -> None:
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    script_path = root_dir / "examples" / filename
    spec = importlib.util.spec_from_file_location("example_module", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def test_example_01_execution() -> None:
    _run_example_file("01_healthy_vs_aneurysm_hemodynamics.py")


def test_example_02_execution() -> None:
    _run_example_file("02_hgo_hyperelastic_rupture_risk.py")


def test_example_03_execution() -> None:
    _run_example_file("03_wave_intensity_windkessel_coupling.py")


def test_example_04_execution() -> None:
    _run_example_file("04_patient_clinical_trial_simulation.py")
