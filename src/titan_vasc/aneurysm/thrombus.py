"""Intraluminal Thrombus (ILT) biomechanics, deposition, and oxygen diffusion barrier.

Models the dual role of intraluminal thrombus in abdominal aortic aneurysms:
1. Mechanical stress shielding (cushioning and pressure attenuation on the wall)
2. Biochemical wall degradation (oxygen diffusion barrier causing transmural hypoxia and tensile strength reduction)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ThrombusState:
    """Quantitative state of intraluminal thrombus along the aortic domain.

    Attributes
    ----------
    z : np.ndarray
        Axial spatial coordinates in meters.
    thickness : np.ndarray
        Spatial thickness of intraluminal thrombus h_ilt(z) in meters.
    effective_pressure_reduction : np.ndarray
        Fraction of luminal blood pressure attenuated by ILT cushioning (0.0 to 0.45).
    wall_oxygen_tension : np.ndarray
        Partial pressure of oxygen pO2 reaching the media-adventitia wall in mmHg.
    is_hypoxic : np.ndarray
        Boolean mask of wall regions undergoing severe hypoxia (pO2 < 5 mmHg).
    strength_reduction_factor : np.ndarray
        Tensile wall strength degradation multiplier (1.0 = healthy, down to 0.50 = severe degradation).
    """

    z: np.ndarray
    thickness: np.ndarray
    effective_pressure_reduction: np.ndarray
    wall_oxygen_tension: np.ndarray
    is_hypoxic: np.ndarray
    strength_reduction_factor: np.ndarray


class IntraluminalThrombusModel:
    """Calculates thrombus accumulation kinetics, mechanical cushioning, and hypoxia.

    Parameters
    ----------
    oxygen_diffusivity : float
        Diffusivity of oxygen in fibrin thrombus D_O2 in m^2/s (default: 1.2e-9 m^2/s).
    oxygen_consumption_rate : float
        Volumetric metabolic oxygen consumption rate by thrombus leukocytes R in mmHg/s (default: 0.18 mmHg/s).
    luminal_po2 : float
        Arterial luminal oxygen partial pressure in mmHg (default: 95.0 mmHg).
    hypoxia_threshold_po2 : float
        Critical threshold for tissue hypoxia in mmHg (default: 5.0 mmHg).
    cushioning_coefficient : float
        Pressure attenuation factor per meter of thrombus thickness (default: 65.0 m^-1).
    """

    def __init__(
        self,
        oxygen_diffusivity: float = 1.2e-9,
        oxygen_consumption_rate: float = 0.18,
        luminal_po2: float = 95.0,
        hypoxia_threshold_po2: float = 5.0,
        cushioning_coefficient: float = 65.0,
    ) -> None:
        self.d_o2 = float(oxygen_diffusivity)
        self.r_o2 = float(oxygen_consumption_rate)
        self.luminal_po2 = float(luminal_po2)
        self.hypoxia_threshold_po2 = float(hypoxia_threshold_po2)
        self.cushioning_coef = float(cushioning_coefficient)

    def estimate_deposition_profile(
        self,
        z: np.ndarray,
        radius: np.ndarray,
        tawss: np.ndarray,
        rrt: np.ndarray,
        max_ilt_thickness: float = 0.012,
    ) -> np.ndarray:
        """Estimate steady-state ILT thickness distribution based on hemodynamics and dilatation.

        Thrombus forms preferentially where TAWSS is low (<0.4 Pa), RRT is high (>10 Pa^-1),
        and the lumen is dilated.

        Parameters
        ----------
        z : np.ndarray
            Axial coordinates [m].
        radius : np.ndarray
            Lumen radius profile [m].
        tawss : np.ndarray
            Time-averaged wall shear stress [Pa].
        rrt : np.ndarray
            Relative residence time [Pa^-1].
        max_ilt_thickness : float
            Maximum thrombus thickness in meters (default: 0.012 m = 12 mm).

        Returns
        -------
        np.ndarray
            Thickness profile h_ilt(z) in meters.
        """
        r_healthy = float(np.min(radius))
        dilatation = np.maximum(0.0, (radius - r_healthy) / r_healthy)

        # Hemodynamic deposition driver: low shear index (1 - min(1, TAWSS/0.4))
        low_shear_score = np.clip((0.4 - tawss) / 0.4, 0.0, 1.0)
        # Residence time driver: sigmoidal response to elevated RRT
        residence_score = 1.0 / (1.0 + np.exp(-0.25 * (rrt - 10.0)))

        thrombus_driver = dilatation * (0.6 * low_shear_score + 0.4 * residence_score)
        max_driver = np.max(thrombus_driver)

        if max_driver > 1e-6:
            normalized_profile = thrombus_driver / max_driver
            ilt_thickness = max_ilt_thickness * normalized_profile
        else:
            ilt_thickness = np.zeros_like(radius)

        return ilt_thickness

    def compute_oxygen_transport(
        self,
        ilt_thickness: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute transmural oxygen penetration and wall interface pO2 across ILT layer.

        Diffusion-reaction equation in flat-slab approximation:
            D_O2 * d^2(C)/dy^2 - R = 0
            C(0) = C_lumen
            dC/dy(h_ilt) = 0 (adventitial no-flux boundary or capillary bed limit)

        Interface wall pO2:
            pO2_wall = max(0.0, pO2_lumen - (R * h_ilt^2) / (2 * D_eff))

        Returns
        -------
        wall_po2 : np.ndarray
            Oxygen tension at wall interface [mmHg].
        is_hypoxic : np.ndarray
            Boolean array where wall_po2 < hypoxia_threshold_po2.
        """
        # Effective transport coefficient k_diff = R / (2 * D_eff) in units of mmHg / m^2
        # Calibrated so that ~5-6 mm of thrombus causes wall hypoxia (<5 mmHg)
        k_diff = 3.6e6   # mmHg / m^2

        wall_po2 = self.luminal_po2 - k_diff * (ilt_thickness**2)
        wall_po2 = np.maximum(0.0, wall_po2)
        is_hypoxic = wall_po2 < self.hypoxia_threshold_po2

        return wall_po2, is_hypoxic

    def compute_stress_shielding(
        self,
        ilt_thickness: np.ndarray,
    ) -> np.ndarray:
        """Compute mechanical stress shielding factor (pressure attenuation on arterial wall).

        P_wall = P_lumen * (1 - reduction_fraction)
        reduction_fraction = 1 - exp(-cushioning_coef * h_ilt)
        Capped at 40% attenuation based on experimental biaxial tissue measurements.
        """
        attenuation = 1.0 - np.exp(-self.cushioning_coef * ilt_thickness)
        return np.clip(attenuation, 0.0, 0.40)

    def evaluate_thrombus_state(
        self,
        z: np.ndarray,
        radius: np.ndarray,
        tawss: np.ndarray,
        rrt: np.ndarray,
        prescribed_thickness: np.ndarray | None = None,
    ) -> ThrombusState:
        """Evaluate full biomechanical and hypoxia state of intraluminal thrombus.

        Parameters
        ----------
        z : np.ndarray
            Axial coordinates [m].
        radius : np.ndarray
            Lumen radius [m].
        tawss : np.ndarray
            TAWSS profile [Pa].
        rrt : np.ndarray
            Relative residence time profile [Pa^-1].
        prescribed_thickness : np.ndarray | None
            Optional externally measured or prescribed thrombus thickness profile.

        Returns
        -------
        ThrombusState
            Integrated thrombus state evaluation.
        """
        if prescribed_thickness is not None:
            ilt_thickness = prescribed_thickness
        else:
            ilt_thickness = self.estimate_deposition_profile(z, radius, tawss, rrt)

        wall_po2, is_hypoxic = self.compute_oxygen_transport(ilt_thickness)
        shielding = self.compute_stress_shielding(ilt_thickness)

        # Strength reduction factor: hypoxia accelerates inflammatory proteolysis,
        # degrading tensile strength by up to 45% under thick thrombus
        strength_reduction = np.ones_like(ilt_thickness)
        for i, (thick_i, hyp_i) in enumerate(zip(ilt_thickness, is_hypoxic)):
            if hyp_i:
                # Severe hypoxia degrades strength significantly
                strength_reduction[i] = max(0.55, 1.0 - 0.45 * (thick_i / 0.015))
            else:
                strength_reduction[i] = max(0.85, 1.0 - 0.15 * (thick_i / 0.015))

        return ThrombusState(
            z=np.copy(z),
            thickness=ilt_thickness,
            effective_pressure_reduction=shielding,
            wall_oxygen_tension=wall_po2,
            is_hypoxic=is_hypoxic,
            strength_reduction_factor=strength_reduction,
        )
