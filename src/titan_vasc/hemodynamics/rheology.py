"""Non-Newtonian blood rheology models for computational hemodynamics.

Implements shear-thinning behavior of human blood, including:
- Newtonian baseline model
- Carreau-Yasuda generalized Newtonian model
- Casson yield-stress model
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


class BloodRheologyModel(ABC):
    """Abstract base class for blood constitutive rheology models."""

    @abstractmethod
    def dynamic_viscosity(self, shear_rate: float | np.ndarray) -> float | np.ndarray:
        """Calculate dynamic viscosity mu [Pa*s] given shear rate gamma_dot [s^-1]."""
        raise NotImplementedError

    def kinematic_viscosity(
        self, shear_rate: float | np.ndarray, density: float = 1060.0
    ) -> float | np.ndarray:
        """Calculate kinematic viscosity nu [m^2/s] = mu / rho."""
        return self.dynamic_viscosity(shear_rate) / density


@dataclass(frozen=True)
class NewtonianRheology(BloodRheologyModel):
    """Newtonian blood rheology model assuming constant dynamic viscosity.

    Parameters
    ----------
    viscosity : float
        Constant dynamic viscosity mu_0 in Pa*s (default: 0.0035 Pa*s = 3.5 mPa*s).
    """

    viscosity: float = 0.0035

    def dynamic_viscosity(self, shear_rate: float | np.ndarray) -> float | np.ndarray:
        """Return constant viscosity for any shear rate."""
        if isinstance(shear_rate, np.ndarray):
            return np.full_like(shear_rate, self.viscosity, dtype=np.float64)
        return float(self.viscosity)


@dataclass(frozen=True)
class CarreauYasudaRheology(BloodRheologyModel):
    """Carreau-Yasuda generalized Newtonian shear-thinning blood model.

    Constitutive equation:
        mu(gamma_dot) = mu_inf + (mu_0 - mu_inf) * [1 + (lambda * gamma_dot)^a]^((n - 1) / a)

    Default parameters represent physiological human blood (hematocrit ~45%):
        mu_0   = 0.056 Pa*s   (zero-shear limiting viscosity)
        mu_inf = 0.00345 Pa*s (infinite-shear limiting viscosity)
        lam    = 3.313 s      (relaxation time parameter)
        n      = 0.3568       (power-law index < 1 indicates shear-thinning)
        a      = 2.0          (Yasuda transition parameter)
    """

    mu_0: float = 0.056
    mu_inf: float = 0.00345
    lam: float = 3.313
    n: float = 0.3568
    a: float = 2.0

    def dynamic_viscosity(self, shear_rate: float | np.ndarray) -> float | np.ndarray:
        """Compute shear-rate dependent dynamic viscosity in Pa*s."""
        gamma = np.asarray(shear_rate, dtype=np.float64)
        abs_gamma = np.abs(gamma)
        # Avoid zero or negative values inside power
        base = 1.0 + (self.lam * abs_gamma) ** self.a
        exponent = (self.n - 1.0) / self.a
        mu = self.mu_inf + (self.mu_0 - self.mu_inf) * (base**exponent)

        if np.isscalar(shear_rate):
            return float(mu)
        return mu


@dataclass(frozen=True)
class CassonRheology(BloodRheologyModel):
    """Casson plastic/pseudoplastic blood model with yield stress.

    Constitutive equation:
        sqrt(tau) = sqrt(tau_y) + sqrt(mu_inf * gamma_dot)  for tau > tau_y
        mu(gamma_dot) = [sqrt(mu_inf) + sqrt(tau_y / gamma_dot)]^2

    Parameters
    ----------
    tau_yield : float
        Yield stress tau_y in Pa (default: 0.004 Pa).
    mu_inf : float
        Asymptotic high-shear viscosity in Pa*s (default: 0.00345 Pa*s).
    gamma_min : float
        Cutoff low shear rate to prevent infinite viscosity at rest (default: 1e-4 s^-1).
    """

    tau_yield: float = 0.004
    mu_inf: float = 0.00345
    gamma_min: float = 1e-4

    def dynamic_viscosity(self, shear_rate: float | np.ndarray) -> float | np.ndarray:
        """Compute Casson effective dynamic viscosity in Pa*s."""
        gamma = np.asarray(shear_rate, dtype=np.float64)
        abs_gamma = np.maximum(np.abs(gamma), self.gamma_min)
        sqrt_term = np.sqrt(self.mu_inf) + np.sqrt(self.tau_yield / abs_gamma)
        mu = sqrt_term**2

        if np.isscalar(shear_rate):
            return float(mu)
        return mu


def estimate_wall_shear_rate(flow_rate: float | np.ndarray, radius: float | np.ndarray) -> float | np.ndarray:
    """Estimate nominal wall shear rate under parabolic laminar flow.

    Formula:
        gamma_dot_w = 4 * |Q| / (pi * R^3)

    Parameters
    ----------
    flow_rate : float | np.ndarray
        Volumetric flow rate Q in m^3/s.
    radius : float | np.ndarray
        Lumen radius R in meters.

    Returns
    -------
    float | np.ndarray
        Wall shear rate in s^-1.
    """
    q_abs = np.abs(np.asarray(flow_rate, dtype=np.float64))
    r = np.maximum(np.asarray(radius, dtype=np.float64), 1e-6)
    gamma_w = 4.0 * q_abs / (np.pi * (r**3))
    if np.isscalar(flow_rate) and np.isscalar(radius):
        return float(gamma_w)
    return gamma_w
