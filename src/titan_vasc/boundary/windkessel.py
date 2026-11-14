"""Lumped parameter Windkessel boundary models for terminal arterial impedance.

Implements 2-element (RC), 3-element (RCR / Westerhof), and 4-element (inertance L-RCR)
outflow impedance models with 4th-order Runge-Kutta numerical ODE integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass
class WindkesselState:
    """Instantaneous state of terminal Windkessel compartment."""

    time: float
    distal_pressure: float     # P(t) [Pa]
    flow_rate: float           # Q(t) [m^3/s]
    compliance_pressure: float # P_c(t) across storage capacitor [Pa]
    inertance_flow_rate: float # Q_L(t) through inductor [m^3/s]


class ThreeElementWindkessel:
    """3-Element Windkessel (RCR / Westerhof) terminal boundary condition.

    Circuit topology:
        Proximal characteristic resistance R_c in series with a parallel
        combination of arterial compliance C and peripheral resistance R_p:
            P(t) - P_venous = R_c * Q(t) + P_c(t)
            d(P_c)/dt = Q(t) / C - P_c / (R_p * C)

    Parameters
    ----------
    r_c : float
        Characteristic aortic impedance in Pa * s / m^3 (default: 1.1e8 Pa*s/m^3).
    r_p : float
        Distal peripheral resistance in Pa * s / m^3 (default: 1.2e9 Pa*s/m^3).
    compliance : float
        Distal arterial vascular compliance in m^3 / Pa (default: 1.0e-9 m^3/Pa).
    venous_pressure : float
        Central venous pressure in Pa (default: 666.0 Pa = 5 mmHg).
    initial_pressure : float
        Initial diastolic compliance capacitor pressure in Pa (default: 10666.0 Pa = 80 mmHg).
    """

    def __init__(
        self,
        r_c: float = 1.1e8,
        r_p: float = 1.2e9,
        compliance: float = 1.0e-9,
        venous_pressure: float = 666.0,
        initial_pressure: float = 10666.0,
    ) -> None:
        self.r_c = float(r_c)
        self.r_p = float(r_p)
        self.c = float(compliance)
        self.p_v = float(venous_pressure)
        self.initial_pressure = float(initial_pressure)
        self.p_c = float(initial_pressure - self.p_v)

    def reset(self) -> None:
        """Reset internal state capacitor to initial diastolic condition."""
        self.p_c = float(self.initial_pressure - self.p_v)

    def outflow_from_pressure(self, t: float, p_distal: float, dt: float = 1e-4) -> float:
        """Compute terminal volumetric flow rate Q_out(t) given distal luminal pressure P_distal.

        Uses analytical exponential decay or RK4 for capacitor voltage d(P_c)/dt.

        Parameters
        ----------
        t : float
            Current time [s].
        p_distal : float
            Distal boundary pressure from 1D arterial tree [Pa].
        dt : float
            Time step for state integration [s].

        Returns
        -------
        float
            Outflow rate Q_out [m^3/s].
        """
        # Current flow rate through R_c: Q = (P_distal - P_v - P_c) / R_c
        q_now = max(0.0, (p_distal - self.p_v - self.p_c) / self.r_c)

        # ODE: d(P_c)/dt = (P_distal - P_v - P_c) / (R_c * C) - P_c / (R_p * C)
        def dpc_dt(p_c_val: float) -> float:
            flow_in = (p_distal - self.p_v - p_c_val) / self.r_c
            flow_out = p_c_val / self.r_p
            return (flow_in - flow_out) / self.c

        # RK4 step
        k1 = dpc_dt(self.p_c)
        k2 = dpc_dt(self.p_c + 0.5 * dt * k1)
        k3 = dpc_dt(self.p_c + 0.5 * dt * k2)
        k4 = dpc_dt(self.p_c + dt * k3)

        self.p_c += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return float(q_now)

    def complex_impedance(self, omega: float | np.ndarray) -> complex | np.ndarray:
        """Compute frequency-domain terminal impedance Z(omega).

        Z(omega) = R_c + R_p / (1 + j * omega * R_p * C)
        """
        j = 1j
        parallel_rc = self.r_p / (1.0 + j * omega * self.r_p * self.c)
        return self.r_c + parallel_rc


class FourElementWindkessel:
    """4-Element Windkessel (L-RCR) adding blood fluid mass inertance L.

    Accounts for high-frequency inertial phase shifts between pressure and flow.
    Governing system:
        L * dQ/dt + R_c * Q + P_c = P(t) - P_v
        C * dP_c/dt + P_c / R_p = Q
    """

    def __init__(
        self,
        r_c: float = 1.0e8,
        r_p: float = 1.2e9,
        compliance: float = 1.0e-9,
        inertance: float = 2.5e6,
        venous_pressure: float = 666.0,
        initial_pressure: float = 10666.0,
    ) -> None:
        self.r_c = float(r_c)
        self.r_p = float(r_p)
        self.c = float(compliance)
        self.l = float(inertance)
        self.p_v = float(venous_pressure)
        self.p_c = float(initial_pressure - self.p_v)
        self.q = 0.0

    def reset(self) -> None:
        """Reset internal states."""
        self.p_c = float(10666.0 - self.p_v)
        self.q = 0.0

    def step_rk4(self, p_distal: float, dt: float) -> tuple[float, float]:
        """Advance system states (q, p_c) by dt under input pressure p_distal.

        Returns (q_new, p_c_new).
        """
        def derivatives(q_val: float, pc_val: float) -> tuple[float, float]:
            dq_dt = (p_distal - self.p_v - self.r_c * q_val - pc_val) / self.l
            dpc_dt = (q_val - pc_val / self.r_p) / self.c
            return dq_dt, dpc_dt

        k1_q, k1_pc = derivatives(self.q, self.p_c)
        k2_q, k2_pc = derivatives(self.q + 0.5 * dt * k1_q, self.p_c + 0.5 * dt * k1_pc)
        k3_q, k3_pc = derivatives(self.q + 0.5 * dt * k2_q, self.p_c + 0.5 * dt * k2_pc)
        k4_q, k4_pc = derivatives(self.q + dt * k3_q, self.p_c + dt * k3_pc)

        self.q += (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
        self.p_c += (dt / 6.0) * (k1_pc + 2.0 * k2_pc + 2.0 * k3_pc + k4_pc)
        return float(self.q), float(self.p_c)
