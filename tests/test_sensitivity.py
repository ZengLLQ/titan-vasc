"""Unit tests for Monte Carlo uncertainty propagation and Morris sensitivity screening."""

import pytest

from titan_vasc.calibration.patient import PatientProfile
from titan_vasc.calibration.sensitivity import (
    RuptureSensitivityAnalyzer,
    UncertaintyQuantificationResult,
)


def test_monte_carlo_uncertainty_quantification() -> None:
    pat = PatientProfile(
        patient_id="UQ-TEST",
        age=70.0,
        max_aneurysm_diameter=0.052,
        systolic_bp=145.0,
    )
    analyzer = RuptureSensitivityAnalyzer(seed=123)

    uq_res = analyzer.monte_carlo_rupture_analysis(
        patient=pat,
        num_samples=40,
    )

    assert isinstance(uq_res, UncertaintyQuantificationResult)
    assert uq_res.pws_mean > 0.0
    assert uq_res.pws_std >= 0.0
    assert uq_res.pws_ci95[0] < uq_res.pws_ci95[1]
    assert uq_res.rpi_ci95[0] < uq_res.rpi_ci95[1]
    assert 0.0 <= uq_res.critical_rupture_probability <= 1.0

    # Morris parameter ranking
    assert len(uq_res.parameter_rankings) >= 3
    # Check that rankings are sorted descending
    effects = [eff for _, eff in uq_res.parameter_rankings]
    assert all(effects[i] >= effects[i + 1] for i in range(len(effects) - 1))
