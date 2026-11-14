"""Physiological and pathological cardiac inflow waveform generators.

Synthesizes realistic pulsatile flow waveforms entering the ascending or abdominal aorta:
- Normal resting adult
- Physical exercise / high-output state
- Isolated systolic hypertension
- Elderly stiffened aorta with rapid ejection
- Multi-harmonic Fourier series synthesis from clinical phase-contrast MRI (PC-MRI)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class CardiacInflowMetrics:
    """Quantitative hemodynamics properties of an inflow waveform.

    Attributes
    ----------
    heart_rate : float
        Cardiac frequency in beats per minute (bpm).
    cycle_period : float
        Duration of one cardiac cycle T = 60 / HR in seconds.
    stroke_volume : float
        Integral of forward ejection flow over one beat in m^3 (or mL / 1e6).
    cardiac_output : float
        Volumetric flow rate per minute (CO = HR * SV) in L/min.
    peak_systolic_flow : float
        Maximum instantaneous flow rate Q_peak in m^3/s.
    mean_flow_rate : float
        Time-averaged volumetric flow rate in m^3/s.
    ejection_time : float
        Duration of systolic ejection phase in seconds.
    """

    heart_rate: float
    cycle_period: float
    stroke_volume: float
    cardiac_output: float
    peak_systolic_flow: float
    mean_flow_rate: float
    ejection_time: float


class InflowWaveformFactory:
    """Factory generating standard and customized pulsatile aortic inflow functions."""

    @staticmethod
    def resting_inflow(
        heart_rate: float = 72.0,
        stroke_volume: float = 70.0e-6,
        systolic_fraction: float = 0.35,
    ) -> tuple[Callable[[float], float], CardiacInflowMetrics]:
        """Generate healthy adult resting aortic flow profile."""
        period = 60.0 / heart_rate
        t_sys = period * systolic_fraction
        # Peak flow for half-sine ejection: integral_0^t_sys Q_pk * sin(pi*t/t_sys) = (2/pi) * Q_pk * t_sys = SV
        q_peak = (stroke_volume * np.pi) / (2.0 * t_sys)

        def wave(t: float) -> float:
            t_phase = t % period
            if t_phase < t_sys:
                return float(q_peak * np.sin(np.pi * t_phase / t_sys))
            # Mild reverse flow at aortic valve closure (dicrotic wave)
            t_dia = t_phase - t_sys
            if t_dia < 0.12 * period:
                return float(-0.06 * q_peak * np.sin(np.pi * t_dia / (0.12 * period)))
            return 0.0

        metrics = CardiacInflowMetrics(
            heart_rate=heart_rate,
            cycle_period=period,
            stroke_volume=stroke_volume,
            cardiac_output=(stroke_volume * 1e3) * heart_rate,
            peak_systolic_flow=q_peak,
            mean_flow_rate=stroke_volume / period,
            ejection_time=t_sys,
        )
        return wave, metrics

    @staticmethod
    def exercise_inflow(
        heart_rate: float = 135.0,
        stroke_volume: float = 95.0e-6,
        systolic_fraction: float = 0.45,
    ) -> tuple[Callable[[float], float], CardiacInflowMetrics]:
        """Generate high-output exercise cardiovascular stress flow profile."""
        period = 60.0 / heart_rate
        t_sys = period * systolic_fraction
        q_peak = (stroke_volume * np.pi) / (2.0 * t_sys)

        def wave(t: float) -> float:
            t_phase = t % period
            if t_phase < t_sys:
                return float(q_peak * np.sin(np.pi * t_phase / t_sys))
            return 0.0

        metrics = CardiacInflowMetrics(
            heart_rate=heart_rate,
            cycle_period=period,
            stroke_volume=stroke_volume,
            cardiac_output=(stroke_volume * 1e3) * heart_rate,
            peak_systolic_flow=q_peak,
            mean_flow_rate=stroke_volume / period,
            ejection_time=t_sys,
        )
        return wave, metrics

    @staticmethod
    def hypertensive_inflow(
        heart_rate: float = 80.0,
        stroke_volume: float = 75.0e-6,
        systolic_fraction: float = 0.30,
    ) -> tuple[Callable[[float], float], CardiacInflowMetrics]:
        """Generate hypertensive sharp ejection flow profile with steep acceleration."""
        period = 60.0 / heart_rate
        t_sys = period * systolic_fraction
        q_peak = (stroke_volume * np.pi) / (2.0 * t_sys)

        def wave(t: float) -> float:
            t_phase = t % period
            if t_phase < t_sys:
                # Skewed peak: reaches max earlier in ejection (t_sys / 3)
                tau = t_phase / t_sys
                shape = np.sin(np.pi * (tau**0.7))
                return float(q_peak * 1.15 * shape)
            t_dia = t_phase - t_sys
            if t_dia < 0.10 * period:
                return float(-0.08 * q_peak * np.sin(np.pi * t_dia / (0.10 * period)))
            return 0.0

        metrics = CardiacInflowMetrics(
            heart_rate=heart_rate,
            cycle_period=period,
            stroke_volume=stroke_volume,
            cardiac_output=(stroke_volume * 1e3) * heart_rate,
            peak_systolic_flow=q_peak * 1.15,
            mean_flow_rate=stroke_volume / period,
            ejection_time=t_sys,
        )
        return wave, metrics


class FourierHarmonicInflow:
    """Reconstructs pulsatile inflow from measured Fourier coefficients.

    Q(t) = a_0 + sum_{n=1}^N [ a_n * cos(n * omega * t) + b_n * sin(n * omega * t) ]
    """

    def __init__(
        self,
        mean_flow: float,
        a_coeffs: list[float],
        b_coeffs: list[float],
        period: float = 0.8,
    ) -> None:
        self.a_0 = float(mean_flow)
        self.a_coeffs = np.array(a_coeffs, dtype=np.float64)
        self.b_coeffs = np.array(b_coeffs, dtype=np.float64)
        self.period = float(period)
        self.omega = 2.0 * np.pi / self.period

    def __call__(self, t: float) -> float:
        n_harmonics = len(self.a_coeffs)
        n_arr = np.arange(1, n_harmonics + 1, dtype=np.float64)
        cos_terms = np.cos(n_arr * self.omega * t)
        sin_terms = np.sin(n_arr * self.omega * t)
        flow = self.a_0 + np.sum(self.a_coeffs * cos_terms + self.b_coeffs * sin_terms)
        return float(flow)
