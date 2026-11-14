"""Unit tests for physiological and pathological cardiac inflow waveform generators."""

import numpy as np
import pytest

from titan_vasc.boundary.inflow import (
    InflowWaveformFactory,
    FourierHarmonicInflow,
    CardiacInflowMetrics,
)


def test_resting_inflow_properties() -> None:
    wave, metrics = InflowWaveformFactory.resting_inflow(heart_rate=75.0, stroke_volume=70e-6)
    assert isinstance(metrics, CardiacInflowMetrics)
    assert metrics.heart_rate == 75.0
    assert np.isclose(metrics.cycle_period, 0.80)
    assert np.isclose(metrics.stroke_volume, 70e-6)
    assert np.isclose(metrics.cardiac_output, 5.25) # 75 * 70 mL = 5.25 L/min

    # Peak flow occurs during systole
    q_mid_sys = wave(metrics.ejection_time / 2.0)
    assert q_mid_sys > 0.0

    # Diastolic quiescence
    q_late_dia = wave(0.70)
    assert np.isclose(q_late_dia, 0.0)


def test_exercise_inflow_properties() -> None:
    wave, metrics = InflowWaveformFactory.exercise_inflow(heart_rate=140.0, stroke_volume=90e-6)
    assert metrics.cardiac_output > 10.0 # High cardiac output
    assert wave(0.05) > 0.0


def test_hypertensive_inflow_properties() -> None:
    wave, metrics = InflowWaveformFactory.hypertensive_inflow()
    assert metrics.ejection_time < 0.30
    assert wave(0.05) > 0.0


def test_fourier_harmonic_inflow() -> None:
    fourier = FourierHarmonicInflow(
        mean_flow=80e-6,
        a_coeffs=[20e-6, -10e-6],
        b_coeffs=[30e-6, 15e-6],
        period=0.8,
    )
    q_0 = fourier(0.0)
    q_half = fourier(0.4)
    assert isinstance(q_0, float)
    assert isinstance(q_half, float)
