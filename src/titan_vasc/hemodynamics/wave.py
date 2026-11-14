"""Arterial wave propagation, tube law, and wave separation dynamics.

Implements:
- Nonlinear arterial wall distensibility and tube law
- Pulse wave velocity (PWV) formulation
- Wave Intensity Analysis (WIA)
- Forward and backward (reflected) wave decomposition
- Augmentation Index (AIx) and Wave Reflection Index (WRI)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TubeLaw:
    """Arterial wall elasticity and constitutive tube law.

    Relates transmural pressure P(z, t) to cross-sectional lumen area A(z, t):
        P(z, t) - P_0 = beta * (sqrt(A) - sqrt(A_0))

    where:
        beta = (4/3) * sqrt(pi) * h_0 * E / A_0

    Parameters
    ----------
    reference_radius : float
        Undeformed inner lumen radius R_0 in meters (default: 0.012 m = 12 mm for aorta).
    wall_thickness : float
        Undeformed arterial wall thickness h_0 in meters (default: 0.0015 m = 1.5 mm).
    young_modulus : float
        Young's elastic modulus E in Pascals (default: 0.8e6 Pa = 800 kPa).
    blood_density : float
        Blood density rho in kg/m^3 (default: 1060.0 kg/m^3).
    reference_pressure : float
        Reference physiological diastolic pressure P_0 in Pa (default: 10666.0 Pa = 80 mmHg).
    """

    reference_radius: float = 0.012
    wall_thickness: float = 0.0015
    young_modulus: float = 0.8e6
    blood_density: float = 1060.0
    reference_pressure: float = 10666.0

    @property
    def reference_area(self) -> float:
        """Reference undeformed cross-sectional area A_0 [m^2]."""
        return float(np.pi * (self.reference_radius**2))

    @property
    def beta(self) -> float:
        """Arterial stiffness coefficient beta [Pa/m]."""
        return float(
            (4.0 / 3.0)
            * np.sqrt(np.pi)
            * self.wall_thickness
            * self.young_modulus
            / self.reference_area
        )

    def pressure_from_area(self, area: float | np.ndarray) -> float | np.ndarray:
        """Compute transmural pressure P [Pa] from cross-sectional lumen area A [m^2]."""
        a = np.asarray(area, dtype=np.float64)
        p = self.reference_pressure + self.beta * (np.sqrt(a) - np.sqrt(self.reference_area))
        if np.isscalar(area):
            return float(p)
        return p

    def area_from_pressure(self, pressure: float | np.ndarray) -> float | np.ndarray:
        """Compute cross-sectional lumen area A [m^2] from transmural pressure P [Pa]."""
        p = np.asarray(pressure, dtype=np.float64)
        delta_p = p - self.reference_pressure
        sqrt_a = np.sqrt(self.reference_area) + (delta_p / self.beta)
        sqrt_a = np.maximum(sqrt_a, 1e-5)
        a = sqrt_a**2
        if np.isscalar(pressure):
            return float(a)
        return a

    def pulse_wave_velocity(self, area: float | np.ndarray) -> float | np.ndarray:
        """Compute local pulse wave velocity c(A) = sqrt( (beta / (2 * rho)) * sqrt(A) ) [m/s]."""
        a = np.asarray(area, dtype=np.float64)
        c = np.sqrt((self.beta / (2.0 * self.blood_density)) * np.sqrt(np.maximum(a, 1e-7)))
        if np.isscalar(area):
            return float(c)
        return c

    @property
    def reference_pwv(self) -> float:
        """Pulse wave velocity at reference diastolic area c_0 [m/s]."""
        return float(self.pulse_wave_velocity(self.reference_area))

    def distensibility(self, area: float | np.ndarray) -> float | np.ndarray:
        """Arterial distensibility D = (1/A) * (dA/dP) = 1 / (rho * c^2) [Pa^-1]."""
        c = self.pulse_wave_velocity(area)
        return 1.0 / (self.blood_density * (c**2))

    def compliance_per_unit_length(self, area: float | np.ndarray) -> float | np.ndarray:
        """Compliance per unit length C_L = dA/dP = A / (rho * c^2) [m^2/Pa]."""
        a = np.asarray(area, dtype=np.float64)
        c = self.pulse_wave_velocity(a)
        return a / (self.blood_density * (c**2))


@dataclass
class WaveSeparationResult:
    """Container for hemodynamic wave separation analysis."""

    time: np.ndarray
    pressure: np.ndarray
    flow_rate: np.ndarray
    velocity: np.ndarray
    p_forward: np.ndarray
    p_backward: np.ndarray
    q_forward: np.ndarray
    q_backward: np.ndarray
    wave_intensity: np.ndarray
    wave_intensity_forward: np.ndarray
    wave_intensity_backward: np.ndarray
    reflection_index: float
    augmentation_index: float


class WaveSeparator:
    """Performs forward and backward wave separation on pulsatile arterial waveforms.

    Based on the Method of Characteristics and Wave Intensity Analysis (Parker et al.):
        dP_+ = 0.5 * (dP + rho * c * dU)
        dP_- = 0.5 * (dP - rho * c * dU)
        dQ_+ = 0.5 * (dQ + (A / (rho * c)) * dP)
        dQ_- = 0.5 * (dQ - (A / (rho * c)) * dP)
        dI   = dP * dU = dI_+ + dI_-
    """

    def __init__(self, tube_law: TubeLaw | None = None) -> None:
        self.tube_law = tube_law if tube_law is not None else TubeLaw()

    def separate(
        self,
        time: np.ndarray,
        pressure: np.ndarray,
        flow_rate: np.ndarray,
        area: np.ndarray | float | None = None,
    ) -> WaveSeparationResult:
        """Decompose pressure and flow into forward and backward traveling waves.

        Parameters
        ----------
        time : np.ndarray
            Time array over cardiac cycle(s) [s].
        pressure : np.ndarray
            Arterial pressure waveform [Pa].
        flow_rate : np.ndarray
            Volumetric flow rate waveform [m^3/s].
        area : np.ndarray | float | None
            Lumen area array or scalar [m^2]. If None, calculated via tube law.

        Returns
        -------
        WaveSeparationResult
            Separated forward/backward pressures, flows, and reflection indices.
        """
        t = np.asarray(time, dtype=np.float64)
        p = np.asarray(pressure, dtype=np.float64)
        q = np.asarray(flow_rate, dtype=np.float64)

        if area is None:
            a = self.tube_law.area_from_pressure(p)
        elif np.isscalar(area):
            a = np.full_like(p, area, dtype=np.float64)
        else:
            a = np.asarray(area, dtype=np.float64)

        u = q / np.maximum(a, 1e-7)
        c = self.tube_law.pulse_wave_velocity(a)
        rho = self.tube_law.blood_density

        # Finite increments
        dp = np.gradient(p, t)
        du = np.gradient(u, t)
        dq = np.gradient(q, t)

        # Characteristic impedance Z_0 = rho * c / A
        z_0 = (rho * c) / np.maximum(a, 1e-7)

        # Rate of changes of forward and backward waves
        dp_forward_dt = 0.5 * (dp + rho * c * du)
        dp_backward_dt = 0.5 * (dp - rho * c * du)

        dq_forward_dt = 0.5 * (dq + (1.0 / z_0) * dp)
        dq_backward_dt = 0.5 * (dq - (1.0 / z_0) * dp)

        # Numerical integration using cumulative trapezoid rule
        p_f = np.zeros_like(p)
        p_b = np.zeros_like(p)
        q_f = np.zeros_like(q)
        q_b = np.zeros_like(q)

        dt = np.diff(t)
        for i in range(len(t) - 1):
            p_f[i + 1] = p_f[i] + 0.5 * (dp_forward_dt[i] + dp_forward_dt[i + 1]) * dt[i]
            p_b[i + 1] = p_b[i] + 0.5 * (dp_backward_dt[i] + dp_backward_dt[i + 1]) * dt[i]
            q_f[i + 1] = q_f[i] + 0.5 * (dq_forward_dt[i] + dq_forward_dt[i + 1]) * dt[i]
            q_b[i + 1] = q_b[i] + 0.5 * (dq_backward_dt[i] + dq_backward_dt[i + 1]) * dt[i]

        # Shift to reference baseline
        p_min = np.min(p)
        p_f += p_min * 0.5
        p_b += p_min * 0.5

        # Wave intensities: dI = dP * dU
        di_f = dp_forward_dt * (dp_forward_dt / (rho * c))
        di_b = dp_backward_dt * (-dp_backward_dt / (rho * c))
        di_net = dp * du

        # Wave Reflection Index: WRI = int(|dI_-|) / int(dI_+)
        int_f = float(np.trapezoid(np.maximum(di_f, 0.0), t))
        int_b = float(np.trapezoid(np.maximum(np.abs(di_b), 0.0), t))
        reflection_index = (int_b / int_f) if int_f > 1e-12 else 0.0

        # Augmentation Index: AIx = (P_peak - P_inflection) / Pulse_Pressure
        p_sys = float(np.max(p))
        p_dia = float(np.min(p))
        pulse_pressure = max(p_sys - p_dia, 1.0)

        # Identify shoulder / reflection arrival (first inflection after systolic rise)
        d2p = np.gradient(dp, t)
        inflection_idx = int(np.argmin(d2p[: len(t) // 2])) if len(t) > 4 else 0
        p_inflection = float(p[inflection_idx])
        augmentation_index = ((p_sys - p_inflection) / pulse_pressure) * 100.0

        return WaveSeparationResult(
            time=t,
            pressure=p,
            flow_rate=q,
            velocity=u,
            p_forward=p_f,
            p_backward=p_b,
            q_forward=q_f,
            q_backward=q_b,
            wave_intensity=di_net,
            wave_intensity_forward=di_f,
            wave_intensity_backward=di_b,
            reflection_index=float(reflection_index),
            augmentation_index=float(augmentation_index),
        )
