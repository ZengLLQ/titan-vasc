"""1D Pulsatile Hemodynamics and Arterial Wave Propagation Solvers.

Solves the 1D cross-sectional averaged nonlinear Navier-Stokes equations:
    dA/dt + dQ/dz = 0
    dQ/dt + d/dz(alpha * Q^2 / A) + (A / rho) * dP/dz + K_R * (Q / A) = 0

Coupled to nonlinear arterial tube law:
    P(A) = P_0 + beta * (sqrt(A) - sqrt(A_0))

Provides MacCormack and Richtmyer (two-step Lax-Wendroff) numerical integration schemes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal
import numpy as np

from titan_vasc.hemodynamics.rheology import BloodRheologyModel, CarreauYasudaRheology
from titan_vasc.hemodynamics.wave import TubeLaw


@dataclass
class ArterialSegment:
    """Geometric and mechanical definition of an arterial vessel segment.

    Parameters
    ----------
    length : float
        Arterial segment length in meters (default: 0.25 m = 25 cm for abdominal aorta).
    radius : float
        Reference lumen radius R_0 in meters (default: 0.010 m = 10 mm).
    wall_thickness : float
        Undeformed wall thickness h_0 in meters (default: 0.0015 m = 1.5 mm).
    young_modulus : float
        Young's modulus of arterial wall E in Pa (default: 8.0e5 Pa = 800 kPa).
    num_nodes : int
        Number of axial spatial discretization nodes (default: 51).
    alpha : float
        Coriolis momentum velocity profile correction coefficient (default: 1.1).
    blood_density : float
        Blood density rho in kg/m^3 (default: 1060.0 kg/m^3).
    reference_pressure : float
        Diastolic reference pressure in Pa (default: 10666.0 Pa = 80 mmHg).
    """

    length: float = 0.25
    radius: float = 0.010
    wall_thickness: float = 0.0015
    young_modulus: float = 8.0e5
    num_nodes: int = 51
    alpha: float = 1.1
    blood_density: float = 1060.0
    reference_pressure: float = 10666.0

    def __post_init__(self) -> None:
        self.tube_law = TubeLaw(
            reference_radius=self.radius,
            wall_thickness=self.wall_thickness,
            young_modulus=self.young_modulus,
            blood_density=self.blood_density,
            reference_pressure=self.reference_pressure,
        )
        self.z = np.linspace(0.0, self.length, self.num_nodes)
        self.dz = float(self.length / (self.num_nodes - 1))
        self.a_0 = np.full(self.num_nodes, self.tube_law.reference_area, dtype=np.float64)


@dataclass
class SimulationHistory:
    """Stores full time-resolved hemodynamics solution fields."""

    time: np.ndarray
    z: np.ndarray
    area: np.ndarray        # Shape: (num_steps, num_nodes)
    flow_rate: np.ndarray   # Shape: (num_steps, num_nodes)
    pressure: np.ndarray    # Shape: (num_steps, num_nodes)
    velocity: np.ndarray    # Shape: (num_steps, num_nodes)


class PulsatileFlowSolver:
    """Finite difference solver for 1D pulsatile blood flow in elastic arteries."""

    def __init__(
        self,
        segment: ArterialSegment | None = None,
        rheology: BloodRheologyModel | None = None,
        scheme: Literal["maccormack", "richtmyer"] = "maccormack",
        cfl: float = 0.7,
    ) -> None:
        self.segment = segment if segment is not None else ArterialSegment()
        self.rheology = rheology if rheology is not None else CarreauYasudaRheology()
        self.scheme = scheme
        self.cfl = cfl

    def compute_flux(self, area: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute conservative fluxes F(U) = [Q, alpha * Q^2 / A + (beta / (3 * rho)) * A^(3/2)]."""
        a_safe = np.maximum(area, 1e-7)
        alpha = self.segment.alpha
        beta = self.segment.tube_law.beta
        rho = self.segment.blood_density

        f_a = flow
        f_q = alpha * (flow**2) / a_safe + (beta / (3.0 * rho)) * (a_safe**1.5)
        return f_a, f_q

    def compute_source(self, area: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Compute viscous friction source term S_Q = -K_R * (Q / A)."""
        a_safe = np.maximum(area, 1e-7)
        r = np.sqrt(a_safe / np.pi)
        shear_rate = 4.0 * np.abs(flow) / (np.pi * (r**3) + 1e-12)
        mu = self.rheology.dynamic_viscosity(shear_rate)
        rho = self.segment.blood_density

        # Viscous resistance coefficient K_R = 8 * pi * mu / rho
        k_r = 8.0 * np.pi * mu / rho
        s_q = -k_r * (flow / a_safe)
        return s_q

    def compute_time_step(self, area: np.ndarray, flow: np.ndarray) -> float:
        """Calculate adaptive CFL time step dt = CFL * dz / max(|u| + c)."""
        a_safe = np.maximum(area, 1e-7)
        u = np.abs(flow / a_safe)
        c = self.segment.tube_law.pulse_wave_velocity(a_safe)
        max_speed = float(np.max(u + c))
        if max_speed < 1e-3:
            max_speed = 5.0
        dt = self.cfl * self.segment.dz / max_speed
        # Clamp to realistic physiological time-step bounds
        return float(np.clip(dt, 1e-6, 1e-3))

    def step_maccormack(
        self,
        area: np.ndarray,
        flow: np.ndarray,
        dt: float,
        q_in: float,
        q_out: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance solution by one time-step using the two-step MacCormack scheme."""
        dz = self.segment.dz
        f_a, f_q = self.compute_flux(area, flow)
        s_q = self.compute_source(area, flow)

        n = len(area)
        a_star = np.copy(area)
        q_star = np.copy(flow)

        # Predictor step: Forward difference for interior
        a_star[1 : n - 1] = area[1 : n - 1] - (dt / dz) * (f_a[2:n] - f_a[1 : n - 1])
        q_star[1 : n - 1] = (
            flow[1 : n - 1]
            - (dt / dz) * (f_q[2:n] - f_q[1 : n - 1])
            + dt * s_q[1 : n - 1]
        )

        # Inflow boundary at predictor
        q_star[0] = q_in
        c_0 = self.segment.tube_law.pulse_wave_velocity(area[0])
        # Extrapolate area from interior characteristic
        a_star[0] = area[0] - (dt / dz) * (flow[1] - flow[0])

        # Outflow boundary at predictor
        if q_out is not None:
            q_star[-1] = q_out
            a_star[-1] = area[-1] - (dt / dz) * (flow[-1] - flow[-2])
        else:
            # Non-reflecting characteristic outflow
            a_star[-1] = a_star[-2]
            q_star[-1] = q_star[-2]

        # Fluxes at predicted state
        f_a_star, f_q_star = self.compute_flux(a_star, q_star)
        s_q_star = self.compute_source(a_star, q_star)

        # Corrector step: Backward difference for interior
        a_new = np.copy(area)
        q_new = np.copy(flow)

        a_new[1 : n - 1] = 0.5 * (
            area[1 : n - 1]
            + a_star[1 : n - 1]
            - (dt / dz) * (f_a_star[1 : n - 1] - f_a_star[0 : n - 2])
        )
        q_new[1 : n - 1] = 0.5 * (
            flow[1 : n - 1]
            + q_star[1 : n - 1]
            - (dt / dz) * (f_q_star[1 : n - 1] - f_q_star[0 : n - 2])
            + dt * s_q_star[1 : n - 1]
        )

        # Boundaries at new time step
        q_new[0] = q_in
        a_new[0] = 0.5 * (area[0] + a_star[0] - (dt / dz) * (flow[1] - flow[0]))

        if q_out is not None:
            q_new[-1] = q_out
            a_new[-1] = 0.5 * (area[-1] + a_star[-1] - (dt / dz) * (flow[-1] - flow[-2]))
        else:
            a_new[-1] = a_new[-2]
            q_new[-1] = q_new[-2]

        return np.maximum(a_new, 1e-7), q_new

    def step_richtmyer(
        self,
        area: np.ndarray,
        flow: np.ndarray,
        dt: float,
        q_in: float,
        q_out: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance solution using two-step Richtmyer (Lax-Wendroff) scheme."""
        dz = self.segment.dz
        f_a, f_q = self.compute_flux(area, flow)
        s_q = self.compute_source(area, flow)

        n = len(area)
        # Half-step at cell centers (i + 1/2)
        a_half = 0.5 * (area[0 : n - 1] + area[1:n]) - (dt / (2.0 * dz)) * (
            f_a[1:n] - f_a[0 : n - 1]
        )
        q_half = (
            0.5 * (flow[0 : n - 1] + flow[1:n])
            - (dt / (2.0 * dz)) * (f_q[1:n] - f_q[0 : n - 1])
            + (dt / 4.0) * (s_q[0 : n - 1] + s_q[1:n])
        )

        f_a_half, f_q_half = self.compute_flux(a_half, q_half)
        s_q_half = self.compute_source(a_half, q_half)

        # Full step at cell nodes
        a_new = np.copy(area)
        q_new = np.copy(flow)

        a_new[1 : n - 1] = area[1 : n - 1] - (dt / dz) * (
            f_a_half[1 : n - 1] - f_a_half[0 : n - 2]
        )
        q_new[1 : n - 1] = (
            flow[1 : n - 1]
            - (dt / dz) * (f_q_half[1 : n - 1] - f_q_half[0 : n - 2])
            + (dt / 2.0) * (s_q_half[0 : n - 2] + s_q_half[1 : n - 1])
        )

        q_new[0] = q_in
        a_new[0] = area[0] - (dt / dz) * (flow[1] - flow[0])

        if q_out is not None:
            q_new[-1] = q_out
            a_new[-1] = area[-1] - (dt / dz) * (flow[-1] - flow[-2])
        else:
            a_new[-1] = a_new[-2]
            q_new[-1] = q_new[-2]

        return np.maximum(a_new, 1e-7), q_new

    def solve(
        self,
        duration: float = 0.8,
        inflow_fn: Callable[[float], float] | None = None,
        outflow_fn: Callable[[float, float], float] | None = None,
        record_interval: float = 0.005,
    ) -> SimulationHistory:
        """Run transient simulation over prescribed duration.

        Parameters
        ----------
        duration : float
            Simulation time in seconds (default: 0.8 s, one cardiac cycle).
        inflow_fn : Callable[[float], float] | None
            Inflow volumetric flow rate Q_in(t) [m^3/s].
        outflow_fn : Callable[[float, float], float] | None
            Outflow condition Q_out(t, P_out). If None, non-reflecting condition used.
        record_interval : float
            Time interval for recording output history [s].

        Returns
        -------
        SimulationHistory
            Simulated flow, pressure, velocity, and area over time.
        """
        if inflow_fn is None:
            inflow_fn = generate_cardiac_cycle_inflow()

        n = self.segment.num_nodes
        area = np.copy(self.segment.a_0)
        flow = np.zeros(n, dtype=np.float64)

        t = 0.0
        last_rec = -1.0

        times_list: list[float] = []
        area_list: list[np.ndarray] = []
        flow_list: list[np.ndarray] = []
        press_list: list[np.ndarray] = []
        vel_list: list[np.ndarray] = []

        step_fn = (
            self.step_maccormack
            if self.scheme == "maccormack"
            else self.step_richtmyer
        )

        while t < duration:
            # Record state
            if t - last_rec >= record_interval or last_rec < 0:
                p_now = self.segment.tube_law.pressure_from_area(area)
                u_now = flow / area
                times_list.append(t)
                area_list.append(np.copy(area))
                flow_list.append(np.copy(flow))
                press_list.append(np.copy(p_now))
                vel_list.append(np.copy(u_now))
                last_rec = t

            dt = self.compute_time_step(area, flow)
            if t + dt > duration:
                dt = duration - t

            q_in = inflow_fn(t)
            q_out = None
            if outflow_fn is not None:
                p_distal = float(self.segment.tube_law.pressure_from_area(area[-1]))
                q_out = outflow_fn(t, p_distal)

            area, flow = step_fn(area, flow, dt, q_in, q_out)
            t += dt

        # Final state record
        p_final = self.segment.tube_law.pressure_from_area(area)
        u_final = flow / area
        times_list.append(t)
        area_list.append(np.copy(area))
        flow_list.append(np.copy(flow))
        press_list.append(np.copy(p_final))
        vel_list.append(np.copy(u_final))

        return SimulationHistory(
            time=np.array(times_list, dtype=np.float64),
            z=np.copy(self.segment.z),
            area=np.array(area_list, dtype=np.float64),
            flow_rate=np.array(flow_list, dtype=np.float64),
            pressure=np.array(press_list, dtype=np.float64),
            velocity=np.array(vel_list, dtype=np.float64),
        )


def generate_cardiac_cycle_inflow(
    stroke_volume: float = 70e-6,
    heart_rate: float = 75.0,
    systolic_fraction: float = 0.35,
) -> Callable[[float], float]:
    """Generate physiological pulsatile aortic inflow flow rate waveform Q_in(t).

    Parameters
    ----------
    stroke_volume : float
        Stroke volume per beat in m^3 (default: 70 mL = 70e-6 m^3).
    heart_rate : float
        Heart rate in beats per minute (default: 75 bpm -> cycle period T = 0.8 s).
    systolic_fraction : float
        Fraction of cardiac cycle spent in ejection systole (default: 0.35).

    Returns
    -------
    Callable[[float], float]
        Pulsatile flow rate function Q_in(t) [m^3/s].
    """
    period = 60.0 / heart_rate
    t_systole = period * systolic_fraction

    # Peak systolic flow to match stroke volume: int_0^T_sys Q_peak * sin(pi * t / T_sys) dt = 2/pi * Q_peak * T_sys
    q_peak = (stroke_volume * np.pi) / (2.0 * t_systole)

    def inflow(t: float) -> float:
        t_phase = t % period
        if t_phase < t_systole:
            # Half-sine systolic ejection
            return float(q_peak * np.sin(np.pi * t_phase / t_systole))
        # Diastolic quiescence with mild early diastolic dicrotic wave
        t_diastole = t_phase - t_systole
        if t_diastole < 0.1 * period:
            # Small reverse flow / aortic valve closure notch
            return float(-0.05 * q_peak * np.sin(np.pi * t_diastole / (0.1 * period)))
        return 0.0

    return inflow
