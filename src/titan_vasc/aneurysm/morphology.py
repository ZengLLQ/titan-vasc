"""Abdominal Aortic Aneurysm (AAA) 3D and 1D geometric morphology models.

Implements parameterized geometric profiles for healthy and aneurysmal aorta:
- Fusiform symmetric aneurysm
- Asymmetric eccentric fusiform aneurysm
- Saccular focal aneurysm
- Proximal and distal neck angulation and length
- Tortuosity index and centerline curvature
- Lumen volume, wall volume, and external surface area
- Spatial principal curvatures (meridional kappa_1, circumferential kappa_2)
- Clinical diameter progression and expansion rate forecasting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass
class AneurysmGeometry:
    """Full quantitative morphology and spatial curvature profile of an aortic aneurysm.

    Attributes
    ----------
    z : np.ndarray
        Axial discretization coordinates [m].
    radius : np.ndarray
        Internal lumen radius R(z) [m].
    wall_thickness : np.ndarray
        Arterial wall thickness h(z) [m].
    cross_sectional_area : np.ndarray
        Internal lumen cross-sectional area A(z) = pi * R(z)^2 [m^2].
    max_diameter : float
        Maximum internal diameter D_max [m].
    healthy_diameter : float
        Baseline healthy vessel diameter D_0 [m].
    dilatation_ratio : float
        Ratio of maximum diameter to healthy baseline (D_max / D_0).
    aneurysm_length : float
        Axial extent of the aneurysmal dilatation [m].
    proximal_neck_length : float
        Length of healthy infrarenal neck preceding dilatation [m].
    proximal_neck_angle : float
        Infrarenal neck angulation relative to longitudinal axis [degrees].
    tortuosity_index : float
        Ratio of centerline path length to Euclidean end-to-end distance minus 1.
    lumen_volume : float
        Total internal lumen volume [m^3].
    wall_volume : float
        Total vascular wall tissue volume [m^3].
    surface_area : float
        Internal luminal endothelial surface area [m^2].
    meridional_curvature : np.ndarray
        Meridional (axial) curvature kappa_1(z) [m^-1].
    circumferential_curvature : np.ndarray
        Circumferential curvature kappa_2(z) [m^-1].
    """

    z: np.ndarray
    radius: np.ndarray
    wall_thickness: np.ndarray
    cross_sectional_area: np.ndarray
    max_diameter: float
    healthy_diameter: float
    dilatation_ratio: float
    aneurysm_length: float
    proximal_neck_length: float
    proximal_neck_angle: float
    tortuosity_index: float
    lumen_volume: float
    wall_volume: float
    surface_area: float
    meridional_curvature: np.ndarray
    circumferential_curvature: np.ndarray


class MorphologyGenerator:
    """Generates parameterized 1D and axisymmetrical AAA morphologies for simulation.

    Parameters
    ----------
    total_length : float
        Total length of the modeled aortic segment in meters (default: 0.25 m = 25 cm).
    healthy_radius : float
        Baseline healthy aortic radius R_0 in meters (default: 0.010 m = 10 mm, D = 20 mm).
    healthy_wall_thickness : float
        Healthy aortic wall thickness h_0 in meters (default: 0.0015 m = 1.5 mm).
    num_nodes : int
        Number of axial spatial grid nodes (default: 101).
    """

    def __init__(
        self,
        total_length: float = 0.25,
        healthy_radius: float = 0.010,
        healthy_wall_thickness: float = 0.0015,
        num_nodes: int = 101,
    ) -> None:
        self.total_length = float(total_length)
        self.healthy_radius = float(healthy_radius)
        self.healthy_wall_thickness = float(healthy_wall_thickness)
        self.num_nodes = int(num_nodes)
        self.z = np.linspace(0.0, self.total_length, self.num_nodes)
        self.dz = float(self.total_length / (self.num_nodes - 1))

    def create_fusiform_aneurysm(
        self,
        max_diameter: float = 0.055,
        aneurysm_center: float = 0.13,
        aneurysm_length: float = 0.08,
        neck_angle: float = 15.0,
        wall_thinning_factor: float = 0.7,
        asymmetry_factor: float = 0.0,
    ) -> AneurysmGeometry:
        """Create a fusiform abdominal aortic aneurysm geometry.

        Radius profile formulation:
            R(z) = R_0 + Delta_R * sin^2(pi * (z - z_start) / L_aaa)  for z in [z_start, z_end]

        Parameters
        ----------
        max_diameter : float
            Maximum internal diameter in meters (default: 0.055 m = 5.5 cm).
        aneurysm_center : float
            Axial center of aneurysm dilatation in meters (default: 0.13 m).
        aneurysm_length : float
            Axial length of dilated section in meters (default: 0.08 m = 8 cm).
        neck_angle : float
            Infrarenal neck angle in degrees (default: 15.0 deg).
        wall_thinning_factor : float
            Ratio of wall thickness at peak dilatation relative to healthy (default: 0.7).
        asymmetry_factor : float
            Asymmetry index between anterior/posterior expansion in [0, 1].

        Returns
        -------
        AneurysmGeometry
            Complete geometric and curvature specification.
        """
        r_0 = self.healthy_radius
        r_max = max_diameter / 2.0
        delta_r = max(0.0, r_max - r_0)

        z_start = aneurysm_center - aneurysm_length / 2.0
        z_end = aneurysm_center + aneurysm_length / 2.0

        radius = np.full(self.num_nodes, r_0, dtype=np.float64)
        wall_thickness = np.full(self.num_nodes, self.healthy_wall_thickness, dtype=np.float64)

        for i, zi in enumerate(self.z):
            if z_start <= zi <= z_end:
                xi = (zi - z_start) / aneurysm_length
                # Bell-shaped smooth expansion
                shape = np.sin(np.pi * xi) ** 2
                radius[i] = r_0 + delta_r * shape
                # Wall thinning over the dilated dome
                thinning = 1.0 - (1.0 - wall_thinning_factor) * shape
                wall_thickness[i] = self.healthy_wall_thickness * thinning

        return self._build_geometry_object(
            radius=radius,
            wall_thickness=wall_thickness,
            max_diameter=float(np.max(radius) * 2.0),
            aneurysm_length=aneurysm_length,
            proximal_neck_length=max(0.0, z_start),
            neck_angle=neck_angle,
            asymmetry_factor=asymmetry_factor,
        )

    def create_saccular_aneurysm(
        self,
        sac_diameter: float = 0.045,
        sac_center: float = 0.12,
        sac_length: float = 0.04,
        neck_angle: float = 20.0,
    ) -> AneurysmGeometry:
        """Create a focal saccular aneurysm morphology with asymmetric outpouching.

        Parameters
        ----------
        sac_diameter : float
            Total transverse diameter at saccular neck in meters (default: 0.045 m).
        sac_center : float
            Axial location of saccular center in meters (default: 0.12 m).
        sac_length : float
            Axial extent of saccular mouth in meters (default: 0.04 m).
        neck_angle : float
            Infrarenal neck angle in degrees.
        """
        r_0 = self.healthy_radius
        r_max = sac_diameter / 2.0
        delta_r = max(0.0, r_max - r_0)

        radius = np.full(self.num_nodes, r_0, dtype=np.float64)
        wall_thickness = np.full(self.num_nodes, self.healthy_wall_thickness, dtype=np.float64)

        sigma_z = sac_length / 3.5
        for i, zi in enumerate(self.z):
            dist = zi - sac_center
            gaussian_bulge = np.exp(-(dist**2) / (2.0 * sigma_z**2))
            radius[i] = r_0 + delta_r * gaussian_bulge
            # Severe localized thinning at saccular apex
            wall_thickness[i] = self.healthy_wall_thickness * (1.0 - 0.45 * gaussian_bulge)

        return self._build_geometry_object(
            radius=radius,
            wall_thickness=wall_thickness,
            max_diameter=float(np.max(radius) * 2.0),
            aneurysm_length=sac_length,
            proximal_neck_length=max(0.0, sac_center - sac_length / 2.0),
            neck_angle=neck_angle,
            asymmetry_factor=0.85,
        )

    def create_healthy_aorta(self) -> AneurysmGeometry:
        """Create normal non-aneurysmal tapering abdominal aorta geometry."""
        # Slight physiological taper: 10 mm proximal to 8.5 mm distal
        taper_radius = np.linspace(self.healthy_radius, self.healthy_radius * 0.85, self.num_nodes)
        wall_thickness = np.linspace(
            self.healthy_wall_thickness, self.healthy_wall_thickness * 0.9, self.num_nodes
        )

        return self._build_geometry_object(
            radius=taper_radius,
            wall_thickness=wall_thickness,
            max_diameter=float(taper_radius[0] * 2.0),
            aneurysm_length=0.0,
            proximal_neck_length=self.total_length,
            neck_angle=0.0,
            asymmetry_factor=0.0,
        )

    def _build_geometry_object(
        self,
        radius: np.ndarray,
        wall_thickness: np.ndarray,
        max_diameter: float,
        aneurysm_length: float,
        proximal_neck_length: float,
        neck_angle: float,
        asymmetry_factor: float,
    ) -> AneurysmGeometry:
        """Helper to compute integral volumes, surfaces, curvatures, and tortuosity."""
        cross_sectional_area = np.pi * (radius**2)

        # Derivatives for curvature and surface integration
        dr_dz = np.gradient(radius, self.dz)
        d2r_dz2 = np.gradient(dr_dz, self.dz)

        # Principal curvatures:
        # kappa_1 = meridional curvature along axial profile
        # kappa_2 = circumferential curvature around cross-section
        denom = (1.0 + dr_dz**2) ** 1.5
        meridional_curvature = -d2r_dz2 / np.maximum(denom, 1e-12)
        circumferential_curvature = 1.0 / (radius * np.sqrt(1.0 + dr_dz**2) + 1e-12)

        # Lumen volume: integral(pi * R^2 dz)
        def _trapz(y: np.ndarray, x: np.ndarray) -> float:
            if hasattr(np, "trapezoid"):
                return float(np.trapezoid(y, x))
            return float(getattr(np, "trapz")(y, x))

        lumen_volume = _trapz(cross_sectional_area, self.z)

        # Wall tissue volume: integral(pi * ((R + h)^2 - R^2) dz)
        outer_area = np.pi * ((radius + wall_thickness) ** 2)
        wall_volume = _trapz(outer_area - cross_sectional_area, self.z)

        # Luminal surface area: integral(2 * pi * R * sqrt(1 + (dR/dz)^2) dz)
        ds = 2.0 * np.pi * radius * np.sqrt(1.0 + dr_dz**2)
        surface_area = _trapz(ds, self.z)

        # Centerline tortuosity calculation (idealized model from neck angle and asymmetry)
        straight_distance = self.total_length
        # Neck angle introduces curvilinear centerline deviation
        angle_rad = np.radians(neck_angle)
        effective_path_length = straight_distance * (1.0 + 0.5 * (1.0 - np.cos(angle_rad)) + 0.05 * asymmetry_factor)
        tortuosity_index = float((effective_path_length / straight_distance) - 1.0)

        healthy_diameter = float(self.healthy_radius * 2.0)
        dilatation_ratio = float(max_diameter / healthy_diameter)

        return AneurysmGeometry(
            z=np.copy(self.z),
            radius=radius,
            wall_thickness=wall_thickness,
            cross_sectional_area=cross_sectional_area,
            max_diameter=max_diameter,
            healthy_diameter=healthy_diameter,
            dilatation_ratio=dilatation_ratio,
            aneurysm_length=aneurysm_length,
            proximal_neck_length=proximal_neck_length,
            proximal_neck_angle=neck_angle,
            tortuosity_index=tortuosity_index,
            lumen_volume=lumen_volume,
            wall_volume=wall_volume,
            surface_area=surface_area,
            meridional_curvature=meridional_curvature,
            circumferential_curvature=circumferential_curvature,
        )


class GrowthForecaster:
    """Predicts future longitudinal aneurysm diameter progression based on clinical models.

    Calibrated against UK Small Aneurysm Trial (UKSAT) and RESCUE natural history data.
    """

    @staticmethod
    def annual_expansion_rate(
        current_diameter: float,
        is_smoker: bool = True,
        is_hypertensive: bool = True,
        is_female: bool = False,
    ) -> float:
        """Estimate annual expansion rate in meters per year.

        Clinical baseline (UKSAT):
            - D < 4.0 cm: ~2.0 mm/year
            - 4.0 <= D < 5.0 cm: ~3.3 mm/year
            - 5.0 <= D < 6.0 cm: ~5.0 mm/year
            - D >= 6.0 cm: ~8.0 mm/year
        Smoking adds +35%, uncontrolled hypertension adds +20%, female adds +15% expansion rate.
        """
        d_cm = current_diameter * 100.0

        if d_cm < 4.0:
            base_mm_yr = 2.0
        elif d_cm < 5.0:
            base_mm_yr = 3.3
        elif d_cm < 6.0:
            base_mm_yr = 5.0
        else:
            base_mm_yr = 8.0 + (d_cm - 6.0) * 2.0

        multiplier = 1.0
        if is_smoker:
            multiplier += 0.35
        if is_hypertensive:
            multiplier += 0.20
        if is_female:
            multiplier += 0.15

        rate_mm_yr = base_mm_yr * multiplier
        return float(rate_mm_yr * 1e-3)

    @classmethod
    def forecast_trajectory(
        cls,
        initial_diameter: float,
        years: float = 5.0,
        dt_years: float = 0.1,
        is_smoker: bool = True,
        is_hypertensive: bool = True,
        is_female: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute time series of aneurysm expansion over multi-year horizon."""
        t_arr = np.arange(0.0, years + dt_years, dt_years)
        d_arr = np.zeros(len(t_arr), dtype=np.float64)
        d_arr[0] = initial_diameter

        for i in range(1, len(t_arr)):
            rate = cls.annual_expansion_rate(
                current_diameter=d_arr[i - 1],
                is_smoker=is_smoker,
                is_hypertensive=is_hypertensive,
                is_female=is_female,
            )
            d_arr[i] = d_arr[i - 1] + rate * dt_years

        return t_arr, d_arr
