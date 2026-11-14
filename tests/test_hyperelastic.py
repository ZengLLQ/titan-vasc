"""Unit tests for vascular hyperelastic constitutive models."""

import numpy as np
import pytest

from titan_vasc.mechanics.hyperelastic import (
    BiaxialStressState,
    NeoHookeanArterialWall,
    MooneyRivlinArterialWall,
    FungExponentialArterialWall,
    HolzapfelGasserOgdenModel,
)


def test_neo_hookean_wall() -> None:
    mat = NeoHookeanArterialWall(c10=8.0e4)
    # At zero stretch (lambda = 1.0), stress must be zero
    state_0 = mat.compute_biaxial_stress(1.0, 1.0)
    assert np.isclose(state_0.cauchy_stress_theta, 0.0)
    assert np.isclose(state_0.cauchy_stress_z, 0.0)

    # In tension (lambda > 1.0), stress must be positive
    state_tens = mat.compute_biaxial_stress(1.10, 1.05)
    assert state_tens.cauchy_stress_theta > 0.0
    assert state_tens.tangent_modulus_theta > 0.0


def test_mooney_rivlin_wall() -> None:
    mat = MooneyRivlinArterialWall(c10=6.0e4, c01=2.0e4)
    state_0 = mat.compute_biaxial_stress(1.0, 1.0)
    assert np.isclose(state_0.cauchy_stress_theta, 0.0)

    state_tens = mat.compute_biaxial_stress(1.15, 1.05)
    assert state_tens.cauchy_stress_theta > 0.0


def test_fung_exponential_wall() -> None:
    mat = FungExponentialArterialWall(c=2.5e4, b_theta=12.0, b_z=8.0, b_thetaz=2.5)
    w_0 = mat.strain_energy_density(1.0, 1.0)
    assert np.isclose(w_0, 0.0)

    state = mat.compute_biaxial_stress(1.10, 1.05)
    assert state.cauchy_stress_theta > 0.0
    assert state.cauchy_stress_z > 0.0


def test_hgo_anisotropic_model() -> None:
    mat = HolzapfelGasserOgdenModel(
        c=3.0e4,
        k1=5.0e5,
        k2=18.0,
        kappa=0.14,
        fiber_angle_deg=45.0,
    )
    # Energy at undeformed state is 0
    w_0 = mat.strain_energy_density(1.0, 1.0)
    assert np.isclose(w_0, 0.0)

    # Biaxial extension
    state = mat.compute_biaxial_stress(1.12, 1.05)
    assert state.cauchy_stress_theta > 0.0
    assert state.cauchy_stress_z > 0.0
    assert state.tangent_modulus_theta > 0.0

    # Non-symmetric fiber engagement: larger stretch produces exponential stiffening
    state_large = mat.compute_biaxial_stress(1.20, 1.05)
    tangent_ratio = state_large.tangent_modulus_theta / state.tangent_modulus_theta
    assert tangent_ratio > 1.5  # Distinct nonlinear fiber stiffening
