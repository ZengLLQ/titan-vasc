"""Hyperelastic constitutive models for healthy and aneurysmal vascular walls.

Implements nonlinear continuum mechanics strain energy density functions:
- Incompressible Neo-Hookean baseline
- Two-parameter Mooney-Rivlin rubber-like continuum
- Fung-type exponential orthotropic arterial constitutive model
- Holzapfel-Gasser-Ogden (HGO) structural anisotropic model with collagen dispersion
- Cauchy stress tensor computation under finite deformation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class BiaxialStressState:
    """Cauchy stress state resulting from biaxial planar deformation.

    Attributes
    ----------
    stretch_theta : float | np.ndarray
        Circumferential stretch ratio lambda_theta (L / L_0).
    stretch_z : float | np.ndarray
        Longitudinal (axial) stretch ratio lambda_z (L_z / L_z0).
    cauchy_stress_theta : float | np.ndarray
        Circumferential Cauchy stress sigma_theta [Pa].
    cauchy_stress_z : float | np.ndarray
        Axial Cauchy stress sigma_z [Pa].
    tangent_modulus_theta : float | np.ndarray
        Incremental circumferential tangent modulus d(sigma_theta)/d(lambda_theta) [Pa].
    """

    stretch_theta: float | np.ndarray
    stretch_z: float | np.ndarray
    cauchy_stress_theta: float | np.ndarray
    cauchy_stress_z: float | np.ndarray
    tangent_modulus_theta: float | np.ndarray


class HyperelasticMaterial(ABC):
    """Abstract base class for hyperelastic vascular wall material models."""

    @abstractmethod
    def strain_energy_density(
        self,
        stretch_theta: float | np.ndarray,
        stretch_z: float | np.ndarray,
    ) -> float | np.ndarray:
        """Compute strain energy density W [J/m^3 = Pa]."""
        pass

    @abstractmethod
    def compute_biaxial_stress(
        self,
        stretch_theta: float | np.ndarray,
        stretch_z: float | np.ndarray,
    ) -> BiaxialStressState:
        """Calculate Cauchy stresses under incompressibility constraint lambda_r = 1 / (lambda_theta * lambda_z)."""
        pass


class NeoHookeanArterialWall(HyperelasticMaterial):
    """Neo-Hookean isotropic hyperelastic model: W = C10 * (I1 - 3)."""

    def __init__(self, c10: float = 8.5e4) -> None:
        self.c10 = float(c10)

    def strain_energy_density(
        self,
        stretch_theta: float | np.ndarray,
        stretch_z: float | np.ndarray,
    ) -> float | np.ndarray:
        stretch_r = 1.0 / (stretch_theta * stretch_z)
        i1 = stretch_theta**2 + stretch_z**2 + stretch_r**2
        return self.c10 * (i1 - 3.0)

    def compute_biaxial_stress(
        self,
        stretch_theta: float | np.ndarray,
        stretch_z: float | np.ndarray,
    ) -> BiaxialStressState:
        stretch_r = 1.0 / (stretch_theta * stretch_z)
        # Incompressible plane stress (sigma_rr = 0):
        # sigma_theta = 2 * C10 * (lambda_theta^2 - lambda_r^2)
        # sigma_z = 2 * C10 * (lambda_z^2 - lambda_r^2)
        sigma_theta = 2.0 * self.c10 * (stretch_theta**2 - stretch_r**2)
        sigma_z = 2.0 * self.c10 * (stretch_z**2 - stretch_r**2)
        tangent = 4.0 * self.c10 * stretch_theta + 4.0 * self.c10 * (stretch_r**2 / stretch_theta)

        return BiaxialStressState(
            stretch_theta=stretch_theta,
            stretch_z=stretch_z,
            cauchy_stress_theta=sigma_theta,
            cauchy_stress_z=sigma_z,
            tangent_modulus_theta=tangent,
        )


class MooneyRivlinArterialWall(HyperelasticMaterial):
    """Mooney-Rivlin hyperelastic model: W = C10 * (I1 - 3) + C01 * (I2 - 3)."""

    def __init__(self, c10: float = 6.2e4, c01: float = 2.1e4) -> None:
        self.c10 = float(c10)
        self.c01 = float(c01)

    def strain_energy_density(
        self,
        stretch_theta: float | np.ndarray,
        stretch_z: float | np.ndarray,
    ) -> float | np.ndarray:
        stretch_r = 1.0 / (stretch_theta * stretch_z)
        i1 = stretch_theta**2 + stretch_z**2 + stretch_r**2
        i2 = stretch_theta**(-2) + stretch_z**(-2) + stretch_r**(-2)
        return self.c10 * (i1 - 3.0) + self.c01 * (i2 - 3.0)

    def compute_biaxial_stress(
        self,
        stretch_theta: float | np.ndarray,
        stretch_z: float | np.ndarray,
    ) -> BiaxialStressState:
        stretch_r = 1.0 / (stretch_theta * stretch_z)
        # sigma_i = 2 * (lambda_i^2 - lambda_r^2) * (C10 + C01 * lambda_k^2)
        sigma_theta = 2.0 * (self.c10 + self.c01 * stretch_z**2) * (stretch_theta**2 - stretch_r**2)
        sigma_z = 2.0 * (self.c10 + self.c01 * stretch_theta**2) * (stretch_z**2 - stretch_r**2)
        tangent = 4.0 * (self.c10 + self.c01 * stretch_z**2) * (stretch_theta + stretch_r**2 / stretch_theta)

        return BiaxialStressState(
            stretch_theta=stretch_theta,
            stretch_z=stretch_z,
            cauchy_stress_theta=sigma_theta,
            cauchy_stress_z=sigma_z,
            tangent_modulus_theta=tangent,
        )


class FungExponentialArterialWall(HyperelasticMaterial):
    """Orthotropic Fung-type exponential arterial wall strain energy model.

    Formulation:
        W = (c / 2) * (exp(Q) - 1)
        Q = b_theta * E_theta^2 + b_z * E_z^2 + 2 * b_thetaz * E_theta * E_z
    where Green-Lagrange strains are E_i = 0.5 * (lambda_i^2 - 1).
    """

    def __init__(
        self,
        c: float = 2.6e4,
        b_theta: float = 12.5,
        b_z: float = 8.2,
        b_thetaz: float = 2.8,
    ) -> None:
        self.c = float(c)
        self.b_theta = float(b_theta)
        self.b_z = float(b_z)
        self.b_thetaz = float(b_thetaz)

    def strain_energy_density(
        self,
        stretch_theta: float | np.ndarray,
        stretch_z: float | np.ndarray,
    ) -> float | np.ndarray:
        e_theta = 0.5 * (stretch_theta**2 - 1.0)
        e_z = 0.5 * (stretch_z**2 - 1.0)
        q = (
            self.b_theta * (e_theta**2)
            + self.b_z * (e_z**2)
            + 2.0 * self.b_thetaz * e_theta * e_z
        )
        return 0.5 * self.c * (np.exp(np.clip(q, -50.0, 50.0)) - 1.0)

    def compute_biaxial_stress(
        self,
        stretch_theta: float | np.ndarray,
        stretch_z: float | np.ndarray,
    ) -> BiaxialStressState:
        e_theta = 0.5 * (stretch_theta**2 - 1.0)
        e_z = 0.5 * (stretch_z**2 - 1.0)
        q = (
            self.b_theta * (e_theta**2)
            + self.b_z * (e_z**2)
            + 2.0 * self.b_thetaz * e_theta * e_z
        )
        exp_q = np.exp(np.clip(q, -50.0, 50.0))

        # 2nd Piola-Kirchhoff: S_theta = dW/dE_theta = c * exp(Q) * (b_theta * E_theta + b_thetaz * E_z)
        s_theta = self.c * exp_q * (self.b_theta * e_theta + self.b_thetaz * e_z)
        s_z = self.c * exp_q * (self.b_z * e_z + self.b_thetaz * e_theta)

        # Cauchy stress: sigma = lambda^2 * S
        sigma_theta = (stretch_theta**2) * s_theta
        sigma_z = (stretch_z**2) * s_z

        # Tangent modulus
        tangent = (
            2.0 * stretch_theta * s_theta
            + (stretch_theta**3)
            * self.c
            * exp_q
            * (self.b_theta + (self.b_theta * e_theta + self.b_thetaz * e_z) ** 2)
        )

        return BiaxialStressState(
            stretch_theta=stretch_theta,
            stretch_z=stretch_z,
            cauchy_stress_theta=sigma_theta,
            cauchy_stress_z=sigma_z,
            tangent_modulus_theta=tangent,
        )


class HolzapfelGasserOgdenModel(HyperelasticMaterial):
    """Holzapfel-Gasser-Ogden (HGO) structural anisotropic arterial constitutive model.

    Accounts for non-symmetric collagen fiber dispersion in media and adventitia.

    Formulation:
        W = (c / 2) * (I1 - 3) + (k1 / (2 * k2)) * sum_{i=4,6} [ exp(k2 * E_i^2) - 1 ]
    where:
        E_i = kappa * (I1 - 3) + (1 - 3 * kappa) * (I_i - 1)
    and fibers are active only in tension (E_i > 0).

    Parameters
    ----------
    c : float
        Isotropic matrix stiffness parameter in Pa (default: 3.2e4 Pa = 32 kPa).
    k1 : float
        Fiber stress scaling coefficient in Pa (default: 4.8e5 Pa = 480 kPa for healthy, up to 1.5e6 Pa for aneurysm).
    k2 : float
        Fiber exponential stiffening exponent (default: 18.0 dimensionless).
    kappa : float
        Fiber dispersion parameter in [0, 1/3] (0 = perfectly aligned, 1/3 = isotropic, default: 0.14).
    fiber_angle_deg : float
        Fiber orientation angle gamma relative to circumferential axis (default: 42.0 degrees).
    """

    def __init__(
        self,
        c: float = 3.2e4,
        k1: float = 4.8e5,
        k2: float = 18.0,
        kappa: float = 0.14,
        fiber_angle_deg: float = 42.0,
    ) -> None:
        self.c = float(c)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.kappa = float(np.clip(kappa, 0.0, 1.0 / 3.0))
        self.gamma = float(np.radians(fiber_angle_deg))

    def strain_energy_density(
        self,
        stretch_theta: float | np.ndarray,
        stretch_z: float | np.ndarray,
    ) -> float | np.ndarray:
        stretch_r = 1.0 / (stretch_theta * stretch_z)
        i1 = stretch_theta**2 + stretch_z**2 + stretch_r**2

        # Invariants I4 and I6 for two symmetric fiber families:
        # a_01 = (cos gamma, sin gamma, 0), a_02 = (cos gamma, -sin gamma, 0)
        cos_g = np.cos(self.gamma)
        sin_g = np.sin(self.gamma)
        i4 = (stretch_theta**2) * (cos_g**2) + (stretch_z**2) * (sin_g**2)

        w_iso = 0.5 * self.c * (i1 - 3.0)

        # Generalized structural strain: E_4 = kappa * (I1 - 3) + (1 - 3 * kappa) * (I4 - 1)
        e4 = self.kappa * (i1 - 3.0) + (1.0 - 3.0 * self.kappa) * (i4 - 1.0)

        # Active only in tension
        e4_tens = np.maximum(0.0, e4)
        # Sum over two identical fiber families (factor of 2)
        w_aniso = (self.k1 / self.k2) * (np.exp(np.clip(self.k2 * (e4_tens**2), -50.0, 50.0)) - 1.0)

        return w_iso + w_aniso

    def compute_biaxial_stress(
        self,
        stretch_theta: float | np.ndarray,
        stretch_z: float | np.ndarray,
    ) -> BiaxialStressState:
        stretch_r = 1.0 / (stretch_theta * stretch_z)
        i1 = stretch_theta**2 + stretch_z**2 + stretch_r**2

        cos_g = np.cos(self.gamma)
        sin_g = np.sin(self.gamma)
        i4 = (stretch_theta**2) * (cos_g**2) + (stretch_z**2) * (sin_g**2)

        # Isotropic matrix stress contribution
        sigma_iso_theta = self.c * (stretch_theta**2 - stretch_r**2)
        sigma_iso_z = self.c * (stretch_z**2 - stretch_r**2)

        # Anisotropic fiber stress contribution
        e4 = self.kappa * (i1 - 3.0) + (1.0 - 3.0 * self.kappa) * (i4 - 1.0)
        e4_tens = np.maximum(0.0, e4)

        exp_fiber = np.exp(np.clip(self.k2 * (e4_tens**2), -50.0, 50.0))
        # dW_aniso / dE4 = 2 * k1 * E4 * exp(k2 * E4^2)
        dw_de4 = 2.0 * self.k1 * e4_tens * exp_fiber

        # Derivatives with respect to stretches
        de4_dlambda_theta = 2.0 * stretch_theta * (self.kappa + (1.0 - 3.0 * self.kappa) * (cos_g**2))
        de4_dlambda_z = 2.0 * stretch_z * (self.kappa + (1.0 - 3.0 * self.kappa) * (sin_g**2))

        sigma_aniso_theta = dw_de4 * de4_dlambda_theta * 0.5 * stretch_theta
        sigma_aniso_z = dw_de4 * de4_dlambda_z * 0.5 * stretch_z

        sigma_theta = sigma_iso_theta + 2.0 * sigma_aniso_theta
        sigma_z = sigma_iso_z + 2.0 * sigma_aniso_z

        # Tangent modulus approximation
        tangent = (
            2.0 * self.c * stretch_theta
            + 4.0 * self.k1 * exp_fiber * (de4_dlambda_theta**2)
            + 2.0 * self.c * (stretch_r**2 / stretch_theta)
        )

        return BiaxialStressState(
            stretch_theta=stretch_theta,
            stretch_z=stretch_z,
            cauchy_stress_theta=sigma_theta,
            cauchy_stress_z=sigma_z,
            tangent_modulus_theta=tangent,
        )
