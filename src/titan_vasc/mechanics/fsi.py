"""Coupled 1D-0D Fluid-Structure Interaction (FSI) and wall compliance feedback.

Couples hydrodynamic pressure waves to dynamic hyperelastic and viscoelastic wall deformation:
- Radial equation of motion with wall inertia and viscoelastic damping
- Dynamic compliance updating C(z, t) = dA/dP
- Pressure-area (P-A) phase lag and hysteresis energy loss loops
- Wall shear stress to hyperelastic deformation feedback
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from titan_vasc.mechanics.hyperelastic import HyperelasticMaterial, HolzapfelGasserOgdenModel


@dataclass
class DynamicWallState:
    """Instantaneous dynamic state of the viscoelastic arterial wall."""

    radius: np.ndarray           # Instantaneous lumen radius R(z, t) [m]
    radial_velocity: np.ndarray  # dR/dt [m/s]
    circumferential_stress: np.ndarray  # sigma_theta [Pa]
    effective_compliance: np.ndarray    # dA/dP [m^2 / Pa]
    dissipated_energy_density: float    # Hysteresis work loss [J / m^3]


class CoupledFSIEngine:
    """1D-0D coupled fluid-structure interaction solver for compliant aneurysmal arteries.

    Parameters
    ----------
    material : HyperelasticMaterial | None
        Constitutive model for vascular tissue. Default: HolzapfelGasserOgdenModel.
    wall_density : float
        Arterial wall tissue density in kg/m^3 (default: 1120.0 kg/m^3).
    viscoelastic_damping : float
        Kelvin-Voigt wall damping coefficient gamma in Pa*s/m (default: 250.0 Pa*s/m).
    external_tissue_pressure : float
        Perivascular retroperitoneal tissue pressure in Pa (default: 666.0 Pa = 5 mmHg).
    """

    def __init__(
        self,
        material: HyperelasticMaterial | None = None,
        wall_density: float = 1120.0,
        viscoelastic_damping: float = 250.0,
        external_tissue_pressure: float = 666.0,
    ) -> None:
        self.material = material if material is not None else HolzapfelGasserOgdenModel()
        self.wall_density = float(wall_density)
        self.damping = float(viscoelastic_damping)
        self.p_ext = float(external_tissue_pressure)

    def compute_static_pressure_area_curve(
        self,
        reference_radius: float,
        wall_thickness: float,
        pressure_range: tuple[float, float] = (5000.0, 25000.0),
        num_points: int = 50,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate static P-A equilibrium curve and differential compliance.

        Returns
        -------
        pressures : np.ndarray
            Pressure values [Pa].
        areas : np.ndarray
            Equilibrium cross-sectional areas [m^2].
        compliances : np.ndarray
            dA/dP [m^2 / Pa].
        """
        pressures = np.linspace(pressure_range[0], pressure_range[1], num_points)
        areas = np.zeros(num_points, dtype=np.float64)

        # Invert sigma_theta(lambda_th) = (P - P_ext) * R / h using iterative bisection
        for idx, p_val in enumerate(pressures):
            p_trans = max(100.0, p_val - self.p_ext)
            lam_min = 1.0
            lam_max = 1.6

            for _ in range(30):
                lam_mid = 0.5 * (lam_min + lam_max)
                stress_state = self.material.compute_biaxial_stress(
                    stretch_theta=lam_mid,
                    stretch_z=1.05,
                )
                r_mid = reference_radius * lam_mid
                # Incompressible thinning: h = h_0 / (lambda_th * lambda_z)
                h_mid = wall_thickness / (lam_mid * 1.05)
                internal_p_equil = (stress_state.cauchy_stress_theta * h_mid) / r_mid

                if internal_p_equil < p_trans:
                    lam_min = lam_mid
                else:
                    lam_max = lam_mid

            r_equil = reference_radius * 0.5 * (lam_min + lam_max)
            areas[idx] = np.pi * (r_equil**2)

        # Differential compliance dA/dP
        dp = float(pressures[1] - pressures[0])
        compliances = np.gradient(areas, dp)
        return pressures, areas, compliances

    def step_wall_dynamics(
        self,
        current_radius: np.ndarray,
        radial_velocity: np.ndarray,
        reference_radius: np.ndarray,
        wall_thickness: np.ndarray,
        fluid_pressure: np.ndarray,
        dt: float,
    ) -> DynamicWallState:
        """Advance arterial wall motion by dt under fluid pressure driving force.

        Radial momentum balance:
            rho_wall * h * d^2(R)/dt^2 + gamma * dR/dt = (P_fluid - P_ext) - (h / R) * sigma_theta

        Returns
        -------
        DynamicWallState
            Updated dynamic radius, velocity, wall stress, and instantaneous compliance.
        """
        r_ref = np.maximum(reference_radius, 1e-4)
        stretch_theta = np.maximum(1.0, current_radius / r_ref)

        # Hyperelastic Cauchy stress
        stress_state = self.material.compute_biaxial_stress(
            stretch_theta=stretch_theta,
            stretch_z=1.05,
        )
        sigma_theta = np.asarray(stress_state.cauchy_stress_theta, dtype=np.float64)

        # Current wall thickness accounting for finite Poisson thinning
        h_current = wall_thickness / (stretch_theta * 1.05)

        # Net radial force per unit luminal area
        p_trans = fluid_pressure - self.p_ext
        elastic_restoring_force = (h_current / current_radius) * sigma_theta
        damping_force = self.damping * radial_velocity

        inertial_mass = self.wall_density * h_current
        # Acceleration d^2(R)/dt^2
        radial_accel = (p_trans - elastic_restoring_force - damping_force) / np.maximum(inertial_mass, 1e-3)

        # Semi-implicit Euler-Cromer update
        new_velocity = radial_velocity + radial_accel * dt
        # Velocity damping limit to ensure physical stability
        new_velocity = np.clip(new_velocity, -1.5, 1.5)
        new_radius = current_radius + new_velocity * dt
        new_radius = np.maximum(reference_radius * 0.95, new_radius)

        # Tangent compliance: dA/dP = (2 * pi * R) / (dP/dR)
        # dP/dR ~ (h / R^2) * (d(sigma)/d(lambda) / R_0 - sigma / R)
        tangent_mod = np.asarray(stress_state.tangent_modulus_theta, dtype=np.float64)
        dp_dr = (h_current / new_radius) * (tangent_mod / r_ref)
        effective_compliance = (2.0 * np.pi * new_radius) / np.maximum(dp_dr, 1e2)

        # Viscous work dissipation density increment: gamma * (dR/dt)^2 * dt
        dissipated_energy = float(np.mean(self.damping * (new_velocity**2) * dt))

        return DynamicWallState(
            radius=new_radius,
            radial_velocity=new_velocity,
            circumferential_stress=sigma_theta,
            effective_compliance=effective_compliance,
            dissipated_energy_density=dissipated_energy,
        )
