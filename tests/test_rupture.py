"""Unit tests for aortic rupture risk assessment and Peak Wall Stress (PWS)."""

import numpy as np
import pytest

from titan_vasc.mechanics.rupture import (
    RuptureRiskEvaluator,
    RuptureRiskResult,
)


def test_membrane_stresses_calculation() -> None:
    evaluator = RuptureRiskEvaluator()
    z = np.linspace(0.0, 0.20, 21)
    dz = float(z[1] - z[0])
    radius = np.full(21, 0.015)
    wall_thickness = np.full(21, 0.0015)
    pressure = 16000.0  # 120 mmHg

    s_theta, s_z, s_vm, s_tresca = evaluator.compute_membrane_stresses(
        radius=radius,
        wall_thickness=wall_thickness,
        transmural_pressure=pressure,
        dz=dz,
    )

    # For straight cylinder: sigma_theta = P * R / h = 16000 * 0.015 / 0.0015 = 160 kPa
    expected_theta = 16000.0 * 0.015 / 0.0015
    np.testing.assert_allclose(s_theta, expected_theta, rtol=1e-3)
    # Axial stress is half circumferential: sigma_z = P * R / (2 * h) = 80 kPa
    np.testing.assert_allclose(s_z, expected_theta / 2.0, rtol=1e-3)
    assert np.all(s_vm > 0.0)


def test_raghavan_vorp_wall_strength() -> None:
    evaluator = RuptureRiskEvaluator(baseline_wall_strength=1.2e6)
    r_norm = np.full(10, 0.010)  # Healthy 20 mm
    r_dilated = np.full(10, 0.027) # Dilated 54 mm

    str_norm = evaluator.compute_local_wall_strength(r_norm, is_female=False)
    str_dilated = evaluator.compute_local_wall_strength(r_dilated, is_female=False)
    str_female = evaluator.compute_local_wall_strength(r_dilated, is_female=True)

    # Dilated tissue is mechanically weaker
    assert np.all(str_norm > str_dilated)
    # Female tissue is weaker due to structural sex disparity
    assert np.all(str_dilated > str_female)


def test_evaluate_rupture_risk_full() -> None:
    evaluator = RuptureRiskEvaluator()
    z = np.linspace(0.0, 0.20, 31)
    radius = 0.010 + 0.018 * np.sin(np.pi * z / 0.20)
    wall_thickness = np.full(31, 0.0015)

    res = evaluator.evaluate_rupture_risk(
        z=z,
        radius=radius,
        wall_thickness=wall_thickness,
        systolic_pressure=16500.0,
    )

    assert isinstance(res, RuptureRiskResult)
    assert res.peak_wall_stress > 0.0
    assert res.peak_rpi > 0.0
    assert 0.0 <= res.peak_rpi_location <= 0.20
    assert res.rupture_risk_equivalent_diameter > 0.0
    assert res.clinical_risk_tier in ["low", "moderate", "high", "critical"]
