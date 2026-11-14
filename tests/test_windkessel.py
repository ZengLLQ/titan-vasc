"""Unit tests for lumped parameter Windkessel terminal impedance models."""

import numpy as np
import pytest

from titan_vasc.boundary.windkessel import (
    ThreeElementWindkessel,
    FourElementWindkessel,
)


def test_three_element_windkessel_flow_and_decay() -> None:
    wk = ThreeElementWindkessel(
        r_c=1.0e8,
        r_p=1.0e9,
        compliance=1.0e-9,
        venous_pressure=666.0,
        initial_pressure=10666.0,
    )

    # Elevated systolic pressure generates outward flow
    q1 = wk.outflow_from_pressure(t=0.0, p_distal=16000.0, dt=1e-3)
    assert q1 > 0.0

    # During prolonged diastole at low pressure, flow gradually decays
    for step in range(50):
        q_dia = wk.outflow_from_pressure(t=step * 1e-3, p_distal=10000.0, dt=1e-3)
    assert q_dia < q1


def test_three_element_windkessel_impedance() -> None:
    wk = ThreeElementWindkessel(r_c=1.0e8, r_p=1.0e9, compliance=1.0e-9)

    # Zero frequency (DC limit): Z(0) = R_c + R_p
    z_dc = wk.complex_impedance(0.0)
    assert np.isclose(z_dc.real, 1.1e9)
    assert np.isclose(z_dc.imag, 0.0)

    # High frequency (infinity limit): Z(inf) -> R_c
    z_hf = wk.complex_impedance(1e5)
    assert np.isclose(z_hf.real, 1.0e8, rtol=1e-2)


def test_four_element_windkessel_step() -> None:
    wk4 = FourElementWindkessel(r_c=1.0e8, r_p=1.0e9, compliance=1.0e-9, inertance=2.0e6)
    q, pc = wk4.step_rk4(p_distal=15000.0, dt=1e-4)

    assert q >= 0.0
    assert pc > 0.0
