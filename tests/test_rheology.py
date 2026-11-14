"""Unit tests for non-Newtonian blood rheology models."""

import numpy as np
import pytest

from titan_vasc.hemodynamics.rheology import (
    NewtonianRheology,
    CarreauYasudaRheology,
    CassonRheology,
    estimate_wall_shear_rate,
)


def test_newtonian_rheology_constant() -> None:
    model = NewtonianRheology(viscosity=0.0035)
    assert model.dynamic_viscosity(0.0) == 0.0035
    assert model.dynamic_viscosity(100.0) == 0.0035
    rates = np.array([0.0, 10.0, 100.0, 1000.0])
    np.testing.assert_allclose(model.dynamic_viscosity(rates), 0.0035)


def test_carreau_yasuda_shear_thinning() -> None:
    model = CarreauYasudaRheology(
        mu_0=0.056,
        mu_inf=0.00345,
        lam=3.313,
        n=0.3568,
        a=2.0,
    )
    mu_low = model.dynamic_viscosity(0.01)
    mu_mid = model.dynamic_viscosity(50.0)
    mu_high = model.dynamic_viscosity(2000.0)

    # Monotonic shear thinning
    assert mu_low > mu_mid > mu_high
    assert np.isclose(mu_low, 0.056, atol=0.005)
    assert np.isclose(mu_high, 0.00345, atol=0.001)


def test_casson_yield_stress() -> None:
    model = CassonRheology(tau_yield=0.004, mu_inf=0.00345)
    mu_10 = model.dynamic_viscosity(10.0)
    mu_500 = model.dynamic_viscosity(500.0)
    assert mu_10 > mu_500
    assert mu_500 > 0.00345


def test_estimate_wall_shear_rate() -> None:
    q = 100e-6  # 100 mL/s
    r = 0.010   # 10 mm
    gamma_dot = estimate_wall_shear_rate(q, r)
    expected = (4.0 * q) / (np.pi * (r**3))
    assert np.isclose(gamma_dot, expected)
