"""Unit tests for Intraluminal Thrombus (ILT) deposition, shielding, and hypoxia barrier."""

import numpy as np
import pytest

from titan_vasc.aneurysm.thrombus import (
    ThrombusState,
    IntraluminalThrombusModel,
)


def test_thrombus_deposition_profile() -> None:
    model = IntraluminalThrombusModel()
    z = np.linspace(0.0, 0.20, 21)
    # Healthy ends (10 mm), dilated center (25 mm)
    radius = 0.010 + 0.015 * np.sin(np.pi * z / 0.20)
    tawss = 1.0 - 0.8 * np.sin(np.pi * z / 0.20)  # Dip in center
    rrt = 1.0 + 15.0 * np.sin(np.pi * z / 0.20)   # Spike in center

    ilt = model.estimate_deposition_profile(z, radius, tawss, rrt)
    assert len(ilt) == 21
    # Thrombus forms thickest at the dilated center where shear is low and residence time is high
    assert ilt[10] > ilt[0]
    assert np.all(ilt >= 0.0)


def test_oxygen_transport_and_hypoxia() -> None:
    model = IntraluminalThrombusModel()
    thicknesses = np.array([0.0, 0.002, 0.005, 0.010])  # 0, 2, 5, 10 mm
    po2, is_hypoxic = model.compute_oxygen_transport(thicknesses)

    assert po2[0] == model.luminal_po2
    # pO2 declines with thrombus thickness
    assert po2[0] > po2[1] > po2[2] >= po2[3]
    # 10 mm thrombus should induce severe hypoxia
    assert is_hypoxic[-1] is True or po2[-1] < 10.0


def test_stress_shielding_bounds() -> None:
    model = IntraluminalThrombusModel()
    thick = np.array([0.0, 0.005, 0.020])
    shielding = model.compute_stress_shielding(thick)

    assert shielding[0] == 0.0
    # Capped at 40% attenuation
    assert np.all(shielding <= 0.4001)
    assert shielding[2] >= shielding[1]


def test_evaluate_thrombus_state() -> None:
    model = IntraluminalThrombusModel()
    z = np.linspace(0.0, 0.20, 21)
    radius = np.full(21, 0.015)
    tawss = np.full(21, 0.3)
    rrt = np.full(21, 12.0)

    state = model.evaluate_thrombus_state(z, radius, tawss, rrt)
    assert isinstance(state, ThrombusState)
    assert len(state.thickness) == 21
    assert len(state.wall_oxygen_tension) == 21
    assert len(state.strength_reduction_factor) == 21
    assert np.all(state.strength_reduction_factor <= 1.0)
