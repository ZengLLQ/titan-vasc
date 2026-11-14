"""Unit tests for clinical scoring algorithms and SVS practice guidelines."""

import pytest

from titan_vasc.calibration.patient import PatientProfile
from titan_vasc.calibration.scoring import (
    ClinicalRiskEngine,
    ClinicalScoringResult,
)


def test_uksat_risk_progression_by_diameter() -> None:
    pat_small = PatientProfile(max_aneurysm_diameter=0.038, smoking_status="never", sex="male")
    pat_large = PatientProfile(max_aneurysm_diameter=0.065, smoking_status="never", sex="male")

    risk_small = ClinicalRiskEngine.calculate_uksat_annual_risk(pat_small)
    risk_large = ClinicalRiskEngine.calculate_uksat_annual_risk(pat_large)

    assert risk_large > risk_small
    assert risk_small <= 1.0
    assert risk_large >= 15.0


def test_female_sex_and_smoking_hazard_multipliers() -> None:
    pat_male = PatientProfile(max_aneurysm_diameter=0.050, sex="male", smoking_status="never")
    pat_female = PatientProfile(max_aneurysm_diameter=0.050, sex="female", smoking_status="never")
    pat_smoker = PatientProfile(max_aneurysm_diameter=0.050, sex="male", smoking_status="current")

    risk_male = ClinicalRiskEngine.calculate_uksat_annual_risk(pat_male)
    risk_female = ClinicalRiskEngine.calculate_uksat_annual_risk(pat_female)
    risk_smoker = ClinicalRiskEngine.calculate_uksat_annual_risk(pat_smoker)

    # 3x risk in females at equivalent diameter
    assert risk_female > risk_male
    assert risk_smoker > risk_male


def test_svs_repair_indications() -> None:
    # 5.6 cm male (Threshold is 5.5 cm)
    pat_indicated_male = PatientProfile(max_aneurysm_diameter=0.056, sex="male")
    res_m = ClinicalRiskEngine.evaluate_svs_guidelines(pat_indicated_male)
    assert res_m.svs_repair_indicated is True
    assert res_m.recommended_surveillance_months == 0

    # 4.5 cm male (Below threshold)
    pat_surv = PatientProfile(max_aneurysm_diameter=0.045, sex="male")
    res_surv = ClinicalRiskEngine.evaluate_svs_guidelines(pat_surv)
    assert res_surv.svs_repair_indicated is False
    assert res_surv.recommended_surveillance_months == 6

    # 5.1 cm female (Threshold is 5.0 cm for females)
    pat_indicated_female = PatientProfile(max_aneurysm_diameter=0.051, sex="female")
    res_f = ClinicalRiskEngine.evaluate_svs_guidelines(pat_indicated_female)
    assert res_f.svs_repair_indicated is True


def test_evar_hostile_neck_contraindication() -> None:
    pat_hostile = PatientProfile(neck_angle_deg=75.0)
    res = ClinicalRiskEngine.evaluate_svs_guidelines(pat_hostile)
    assert res.evar_anatomically_suitable is False
    assert len(res.contraindications_evar) > 0
