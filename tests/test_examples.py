"""Automated regression tests verifying all standalone examples run cleanly."""

import importlib
import pytest


def test_example_01_execution() -> None:
    mod = importlib.import_module("examples.01_healthy_vs_aneurysm_hemodynamics")
    mod.main()


def test_example_02_execution() -> None:
    mod = importlib.import_module("examples.02_hgo_hyperelastic_rupture_risk")
    mod.main()


def test_example_03_execution() -> None:
    mod = importlib.import_module("examples.03_wave_intensity_windkessel_coupling")
    mod.main()


def test_example_04_execution() -> None:
    mod = importlib.import_module("examples.04_patient_clinical_trial_simulation")
    mod.main()
