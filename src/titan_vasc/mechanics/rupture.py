"""Aortic rupture risk assessment, peak wall stress, and rupture potential index (RPI).

Implements finite-membrane shell stress analysis and patient-specific wall strength models:
- Curved membrane circumferential and meridional wall stresses
- Von Mises and Tresca equivalent yield envelopes
- Raghavan-Vorp statistical patient-specific wall tensile strength distribution
- Rupture Potential Index (RPI = sigma_vm / sigma_strength)
- Rupture Risk Equivalent Diameter (RRED)
- Clinical risk tier categorization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass
class RuptureRiskResult:
    """Comprehensive biomechanical rupture risk evaluation.

    Attributes
    ----------
    z : np.ndarray
        Spatial axial coordinates [m].
    circumferential_stress : np.ndarray
        Circumferential Cauchy stress sigma_theta(z) in Pascals.
    meridional_stress : np.ndarray
        Meridional (axial) Cauchy stress sigma_z(z) in Pascals.
    von_mises_stress : np.ndarray
        Von Mises equivalent wall stress in Pascals.
    tresca_stress : np.ndarray
        Tresca maximum shear stress in Pascals.
    tensile_wall_strength : np.ndarray
        Local patient-specific wall tensile failure strength in Pascals.
    rupture_potential_index : np.ndarray
        RPI ratio (von_mises_stress / tensile_wall_strength).
    peak_wall_stress : float
        Maximum Von Mises stress observed across the aneurysm dome in Pascals.
    peak_rpi : float
        Maximum local Rupture Potential Index across the aorta.
    peak_rpi_location : float
        Axial coordinate z of maximum RPI in meters.
    rupture_risk_equivalent_diameter : float
        Equivalent idealized Laplace diameter in meters corresponding to observed PWS.
    clinical_risk_tier : Literal["low", "moderate", "high", "critical"]
        Stratified risk tier based on RPI and PWS.
    """

    z: np.ndarray
    circumferential_stress: np.ndarray
    meridional_stress: np.ndarray
    von_mises_stress: np.ndarray
    tresca_stress: np.ndarray
    tensile_wall_strength: np.ndarray
    rupture_potential_index: np.ndarray
    peak_wall_stress: float
    peak_rpi: float
    peak_rpi_location: float
    rupture_risk_equivalent_diameter: float
    clinical_risk_tier: Literal["low", "moderate", "high", "critical"]


class RuptureRiskEvaluator:
    """Calculates multidimensional wall stresses and evaluates local rupture criteria.

    Parameters
    ----------
    baseline_wall_strength : float
        Healthy reference aortic tensile strength in Pa (default: 1.2e6 Pa = 1.2 MPa).
    diameter_strength_slope : float
        Rate of strength decline per meter increase in aneurysm diameter (default: 1.5e7 Pa/m = 0.15 MPa/cm).
    ilt_strength_slope : float
        Rate of strength decline per meter of thrombus thickness due to hypoxia (default: 8.0e7 Pa/m = 0.08 MPa/mm).
    female_strength_penalty : float
        Absolute tensile strength reduction in female aorta (default: 1.8e5 Pa = 0.18 MPa).
    """

    def __init__(
        self,
        baseline_wall_strength: float = 1.2e6,
        diameter_strength_slope: float = 1.5e7,
        ilt_strength_slope: float = 8.0e7,
        female_strength_penalty: float = 1.8e5,
    ) -> None:
        self.base_strength = float(baseline_wall_strength)
        self.diameter_slope = float(diameter_strength_slope)
        self.ilt_slope = float(ilt_strength_slope)
        self.female_penalty = float(female_strength_penalty)

    def compute_membrane_stresses(
        self,
        radius: np.ndarray,
        wall_thickness: np.ndarray,
        transmural_pressure: float | np.ndarray,
        dz: float,
        effective_pressure_reduction: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute circumferential, meridional, Von Mises, and Tresca wall stresses.

        Formulation with axial curvature correction:
            sigma_theta = (P_eff * R / h) * [1 - (R * R'') / (2 * (1 + R'^2))]
            sigma_z = (P_eff * R / (2 * h)) * [1 + (R * R'') / (1 + R'^2)]

        Parameters
        ----------
        radius : np.ndarray
            Lumen radius R(z) [m].
        wall_thickness : np.ndarray
            Arterial wall thickness h(z) [m].
        transmural_pressure : float | np.ndarray
            Transmural luminal pressure in Pa (e.g. peak systolic pressure ~16000 Pa = 120 mmHg).
        dz : float
            Spatial grid spacing in meters.
        effective_pressure_reduction : np.ndarray | None
            Thrombus shielding factor (0 to 0.40).

        Returns
        -------
        sigma_theta, sigma_z, sigma_vm, tau_tresca
        """
        dr_dz = np.gradient(radius, dz)
        d2r_dz2 = np.gradient(dr_dz, dz)

        p_net = (
            transmural_pressure
            if isinstance(transmural_pressure, np.ndarray)
            else np.full_like(radius, transmural_pressure)
        )

        if effective_pressure_reduction is not None:
            p_net = p_net * (1.0 - effective_pressure_reduction)

        h_safe = np.maximum(wall_thickness, 1e-5)
        r_safe = np.maximum(radius, 1e-4)

        slope_sq = dr_dz**2
        curvature_factor = (r_safe * d2r_dz2) / (1.0 + slope_sq + 1e-12)

        # Circumferential stress with curvature modification
        sigma_theta = (p_net * r_safe / h_safe) * (1.0 - 0.5 * curvature_factor)
        # Axial / meridional stress
        sigma_z = (p_net * r_safe / (2.0 * h_safe)) * (1.0 + curvature_factor)

        # Von Mises equivalent stress: sqrt(s_th^2 - s_th * s_z + s_z^2)
        sigma_vm = np.sqrt(
            np.maximum(
                0.0,
                sigma_theta**2 - sigma_theta * sigma_z + sigma_z**2,
            )
        )

        # Tresca equivalent shear stress: max principal stress difference / 2
        tau_tresca = 0.5 * np.maximum(
            np.abs(sigma_theta - sigma_z),
            np.maximum(np.abs(sigma_theta), np.abs(sigma_z)),
        )

        return sigma_theta, sigma_z, sigma_vm, tau_tresca

    def compute_local_wall_strength(
        self,
        radius: np.ndarray,
        ilt_thickness: np.ndarray | None = None,
        is_female: bool = False,
        age: float = 68.0,
        family_history: bool = False,
    ) -> np.ndarray:
        """Estimate patient-specific local wall tensile strength distribution.

        Based on Raghavan & Vorp clinical biaxial rupture testing.
        """
        d_current = radius * 2.0
        d_healthy_ref = 0.020  # 20 mm reference aorta

        # Degradation due to dilatation
        strength = self.base_strength - self.diameter_slope * np.maximum(0.0, d_current - d_healthy_ref)

        # Degradation due to thrombus hypoxia
        if ilt_thickness is not None:
            strength -= self.ilt_slope * ilt_thickness

        # Biological covariates
        if is_female:
            strength -= self.female_penalty
        if family_history:
            strength -= 1.0e5  # Familial connective tissue defect penalty

        # Age-related loss of collagen cross-linking: -0.5% per year above 60
        age_excess = max(0.0, age - 60.0)
        strength *= max(0.65, 1.0 - 0.005 * age_excess)

        # Physiological bounds: aortic wall strength cannot fall below 0.35 MPa or exceed 2.5 MPa
        return np.clip(strength, 3.5e5, 2.5e6)

    def evaluate_rupture_risk(
        self,
        z: np.ndarray,
        radius: np.ndarray,
        wall_thickness: np.ndarray,
        systolic_pressure: float = 16000.0,
        ilt_thickness: np.ndarray | None = None,
        is_female: bool = False,
        age: float = 68.0,
        family_history: bool = False,
    ) -> RuptureRiskResult:
        """Perform comprehensive biomechanical rupture risk assessment.

        Parameters
        ----------
        z : np.ndarray
            Axial coordinates [m].
        radius : np.ndarray
            Lumen radius [m].
        wall_thickness : np.ndarray
            Wall thickness [m].
        systolic_pressure : float
            Peak systolic pressure in Pa (default: 16000 Pa = 120 mmHg; hypertensive: >18600 Pa = 140 mmHg).
        ilt_thickness : np.ndarray | None
            Intraluminal thrombus thickness [m].
        is_female : bool
            Patient sex.
        age : float
            Patient age in years.
        family_history : bool
            Positive family history of AAA rupture.

        Returns
        -------
        RuptureRiskResult
            Full rupture risk profile and clinical stratification.
        """
        dz = float(z[1] - z[0]) if len(z) > 1 else 0.001

        # Calculate stresses
        eff_reduction = None
        if ilt_thickness is not None:
            eff_reduction = np.clip(1.0 - np.exp(-65.0 * ilt_thickness), 0.0, 0.40)

        s_theta, s_z, s_vm, s_tresca = self.compute_membrane_stresses(
            radius=radius,
            wall_thickness=wall_thickness,
            transmural_pressure=systolic_pressure,
            dz=dz,
            effective_pressure_reduction=eff_reduction,
        )

        # Calculate tensile failure strength
        strength = self.compute_local_wall_strength(
            radius=radius,
            ilt_thickness=ilt_thickness,
            is_female=is_female,
            age=age,
            family_history=family_history,
        )

        # Rupture Potential Index (RPI)
        rpi = s_vm / strength

        peak_stress = float(np.max(s_vm))
        max_rpi_idx = int(np.argmax(rpi))
        peak_rpi = float(rpi[max_rpi_idx])
        peak_rpi_loc = float(z[max_rpi_idx])

        # Rupture Risk Equivalent Diameter (RRED):
        # Laplace inversion: sigma = P * R_eq / h_0 => D_eq = 2 * (sigma * h_0 / P_norm)
        p_norm = 16000.0   # 120 mmHg standard reference pressure
        h_ref = 0.0015     # 1.5 mm reference thickness
        rred = float(2.0 * (peak_stress * h_ref / p_norm))

        # Clinical tier classification
        if peak_rpi >= 0.85 or peak_stress >= 5.0e5:
            tier: Literal["low", "moderate", "high", "critical"] = "critical"
        elif peak_rpi >= 0.65 or peak_stress >= 4.0e5:
            tier = "high"
        elif peak_rpi >= 0.45 or peak_stress >= 3.0e5:
            tier = "moderate"
        else:
            tier = "low"

        return RuptureRiskResult(
            z=np.copy(z),
            circumferential_stress=s_theta,
            meridional_stress=s_z,
            von_mises_stress=s_vm,
            tresca_stress=s_tresca,
            tensile_wall_strength=strength,
            rupture_potential_index=rpi,
            peak_wall_stress=peak_stress,
            peak_rpi=peak_rpi,
            peak_rpi_location=peak_rpi_loc,
            rupture_risk_equivalent_diameter=rred,
            clinical_risk_tier=tier,
        )
