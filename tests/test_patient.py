"""Unit tests for patient clinical profile representations and validation."""

import pytest

from titan_vasc.calibration.patient import PatientProfile


def test_patient_profile_properties() -> None:
    pat = PatientProfile(
        patient_id="PAT-TEST",
        age=65.0,
        sex="male",
        systolic_bp=140.0,
        diastolic_bp=90.0,
        max_aneurysm_diameter=0.052,
        healthy_aorta_diameter=0.020,
    )

    assert pat.pulse_pressure_mmhg == 50.0
    # MAP = 90 + 50 / 3 = 106.67 mmHg
    assert pytest.approx(pat.mean_arterial_pressure_mmhg, 0.01) == 106.666
    assert pytest.approx(pat.dilatation_ratio, 1e-4) == 2.6
    assert pat.systolic_pressure_pa > 18000.0


def test_patient_profile_validation_errors() -> None:
    with pytest.raises(ValueError):
        PatientProfile(age=-5.0)

    with pytest.raises(ValueError):
        # SBP must exceed DBP
        PatientProfile(systolic_bp=80.0, diastolic_bp=90.0)

    with pytest.raises(ValueError):
        # Max diameter cannot be smaller than healthy baseline
        PatientProfile(max_aneurysm_diameter=0.010, healthy_aorta_diameter=0.020)
