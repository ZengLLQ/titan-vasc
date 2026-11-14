"""Wall Shear Stress (WSS) and hemodynamic disturbance index metrics.

Computes time-resolved and cycle-averaged hemodynamic shear stress indices:
- Wall Shear Stress (WSS) vectors and magnitudes
- Time-Averaged Wall Shear Stress (TAWSS)
- Oscillatory Shear Index (OSI)
- Relative Residence Time (RRT)
- Endothelial Cell Activation Potential (ECAP)
- Axial WSS spatial gradients (WSSG)
- Endothelial atheroprone and thrombogenic risk classification
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np

from titan_vasc.hemodynamics.rheology import BloodRheologyModel, CarreauYasudaRheology


def _integrate_trapezoid(y: np.ndarray, x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Trapezoidal integration compatible with both NumPy 1.x and NumPy 2.x."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return getattr(np, "trapz")(y, x, axis=axis)


@dataclass
class HemodynamicDisturbanceResult:
    """Cycle-averaged hemodynamic wall shear disturbance indices.

    Attributes
    ----------
    z : np.ndarray
        Axial spatial coordinates in meters.
    tawss : np.ndarray
        Time-averaged wall shear stress in Pascals (Pa).
    osi : np.ndarray
        Oscillatory shear index (dimensionless, bounded [0, 0.5]).
    rrt : np.ndarray
        Relative residence time in Pa^-1.
    ecap : np.ndarray
        Endothelial cell activation potential in Pa^-1.
    wssg_mean : np.ndarray
        Mean axial wall shear stress gradient in Pa/m.
    atheroprone_mask : np.ndarray
        Boolean mask of endothelial sites at risk of intimal thickening (TAWSS < 0.4 Pa and OSI > 0.2).
    thrombogenic_mask : np.ndarray
        Boolean mask of high-shear sites predisposing to platelet shear activation (TAWSS > 7.0 Pa).
    """

    z: np.ndarray
    tawss: np.ndarray
    osi: np.ndarray
    rrt: np.ndarray
    ecap: np.ndarray
    wssg_mean: np.ndarray
    atheroprone_mask: np.ndarray
    thrombogenic_mask: np.ndarray


class WallShearStressCalculator:
    """Calculates instantaneous and time-averaged wall shear stress metrics from flow history.

    Parameters
    ----------
    rheology : BloodRheologyModel | None
        Non-Newtonian constitutive rheology model. Default: CarreauYasudaRheology.
    velocity_profile_factor : float
        Velocity profile shape factor zeta (zeta=2 for parabolic Poiseuille flow, zeta=9 for blunted plug flow).
        Default: 2.0.
    """

    def __init__(
        self,
        rheology: BloodRheologyModel | None = None,
        velocity_profile_factor: float = 2.0,
    ) -> None:
        self.rheology = rheology if rheology is not None else CarreauYasudaRheology()
        self.velocity_profile_factor = float(velocity_profile_factor)

    def instantaneous_shear_rate(
        self,
        area: np.ndarray,
        flow_rate: np.ndarray,
    ) -> np.ndarray:
        """Compute instantaneous wall shear rate gamma_dot_w [s^-1].

        Formulation:
            gamma_dot_w = (2 + zeta) * |Q| / (pi * R^3)
        where R = sqrt(A / pi).
        """
        a_safe = np.maximum(area, 1e-7)
        r = np.sqrt(a_safe / np.pi)
        shear_rate = (2.0 + self.velocity_profile_factor) * np.abs(flow_rate) / (np.pi * (r**3) + 1e-12)
        return shear_rate

    def instantaneous_wss(
        self,
        area: np.ndarray,
        flow_rate: np.ndarray,
    ) -> np.ndarray:
        """Compute signed instantaneous wall shear stress WSS [Pa].

        WSS = sign(Q) * mu(gamma_dot_w) * gamma_dot_w
        """
        shear_rate = self.instantaneous_shear_rate(area, flow_rate)
        viscosity = self.rheology.dynamic_viscosity(shear_rate)
        wss = np.sign(flow_rate) * viscosity * shear_rate
        return wss

    def compute_indices(
        self,
        time: np.ndarray,
        z: np.ndarray,
        area: np.ndarray,
        flow_rate: np.ndarray,
        cycle_start_time: float | None = None,
        cycle_end_time: float | None = None,
    ) -> HemodynamicDisturbanceResult:
        """Compute cycle-averaged hemodynamic disturbance indices across spatial domain.

        Parameters
        ----------
        time : np.ndarray
            Time array in seconds (length: M).
        z : np.ndarray
            Axial coordinate array in meters (length: N).
        area : np.ndarray
            Cross-sectional area history, shape (M, N).
        flow_rate : np.ndarray
            Volumetric flow rate history, shape (M, N).
        cycle_start_time : float | None
            Start time for cardiac cycle integration. If None, uses final cycle period.
        cycle_end_time : float | None
            End time for cardiac cycle integration. If None, uses final simulation time.

        Returns
        -------
        HemodynamicDisturbanceResult
            Complete computed hemodynamic disturbance profile.
        """
        if len(time) < 2:
            raise ValueError("Time array must contain at least two discrete samples.")

        total_time = float(time[-1] - time[0])
        if cycle_end_time is None:
            cycle_end_time = float(time[-1])
        if cycle_start_time is None:
            # Default to last 0.8 seconds (typical resting cardiac cycle) or full duration
            cycle_start_time = max(float(time[0]), cycle_end_time - 0.8)

        mask = (time >= cycle_start_time) & (time <= cycle_end_time)
        if np.sum(mask) < 2:
            mask = np.ones(len(time), dtype=bool)

        t_sub = time[mask]
        area_sub = area[mask, :]
        flow_sub = flow_rate[mask, :]
        dt = np.gradient(t_sub)

        num_steps, num_nodes = area_sub.shape
        wss_history = np.zeros((num_steps, num_nodes), dtype=np.float64)

        for step in range(num_steps):
            wss_history[step, :] = self.instantaneous_wss(area_sub[step, :], flow_sub[step, :])

        cycle_period = float(t_sub[-1] - t_sub[0])
        if cycle_period <= 0.0:
            cycle_period = 1e-3

        # Time-Averaged Wall Shear Stress (TAWSS): (1 / T) * integral(|WSS| dt)
        abs_wss = np.abs(wss_history)
        tawss = _integrate_trapezoid(abs_wss, t_sub, axis=0) / cycle_period

        # Vector time-average: (1 / T) * integral(WSS dt)
        mean_vector_wss = _integrate_trapezoid(wss_history, t_sub, axis=0) / cycle_period

        # Oscillatory Shear Index (OSI): 0.5 * (1 - |mean_vector_wss| / TAWSS)
        safe_tawss = np.maximum(tawss, 1e-6)
        ratio = np.abs(mean_vector_wss) / safe_tawss
        ratio = np.clip(ratio, 0.0, 1.0)
        osi = 0.5 * (1.0 - ratio)

        # Relative Residence Time (RRT): 1 / ((1 - 2 * OSI) * TAWSS) = 1 / |mean_vector_wss|
        safe_mean_vec = np.maximum(np.abs(mean_vector_wss), 1e-5)
        rrt = 1.0 / safe_mean_vec

        # Endothelial Cell Activation Potential (ECAP): OSI / TAWSS
        ecap = osi / safe_tawss

        # Axial Wall Shear Stress Gradient (WSSG): mean |d(WSS)/dz|
        dz = float(z[1] - z[0]) if len(z) > 1 else 1e-3
        wssg_history = np.abs(np.gradient(wss_history, dz, axis=1))
        wssg_mean = _integrate_trapezoid(wssg_history, t_sub, axis=0) / cycle_period

        # Risk classification masks
        atheroprone_mask = (tawss < 0.4) & (osi > 0.2)
        thrombogenic_mask = tawss > 7.0

        return HemodynamicDisturbanceResult(
            z=np.copy(z),
            tawss=tawss,
            osi=osi,
            rrt=rrt,
            ecap=ecap,
            wssg_mean=wssg_mean,
            atheroprone_mask=atheroprone_mask,
            thrombogenic_mask=thrombogenic_mask,
        )
