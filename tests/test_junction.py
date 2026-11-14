"""Unit tests for aortic bifurcation branching junctions and wave reflection."""

import numpy as np
import pytest

from titan_vasc.boundary.junction import (
    AorticBifurcationJunction,
    BifurcationJunctionState,
)


def test_bifurcation_wave_reflection_and_transmission() -> None:
    junction = AorticBifurcationJunction(blood_density=1060.0)

    # Well-matched bifurcation: area ratio eta ~ 1.15
    a_parent = np.pi * 0.010**2
    a_daughter = np.pi * 0.0076**2

    state = junction.evaluate_wave_reflection(
        area_parent=a_parent,
        pwv_parent=6.0,
        area_left_iliac=a_daughter,
        pwv_left_iliac=7.0,
        area_right_iliac=a_daughter,
        pwv_right_iliac=7.0,
        flow_parent=100e-6,
    )

    assert isinstance(state, BifurcationJunctionState)
    assert -1.0 <= state.reflection_coefficient <= 1.0
    # Transmission T = 1 + Gamma
    assert np.isclose(state.transmission_coefficient, 1.0 + state.reflection_coefficient)
    assert state.area_ratio > 1.0
    assert state.minor_head_loss >= 0.0


def test_bifurcation_flow_distribution() -> None:
    junction = AorticBifurcationJunction()
    q_parent = 100e-6
    # Symmetrical daughters
    q_l, q_r = junction.distribute_flows(q_parent, admittance_left=1e-8, admittance_right=1e-8)
    assert np.isclose(q_l, 50e-6)
    assert np.isclose(q_r, 50e-6)
    assert np.isclose(q_l + q_r, q_parent)
