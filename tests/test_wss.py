"""Unit tests for Wall Shear Stress and hemodynamic disturbance index metrics."""

import numpy as np
import pytest

from titan_vasc.hemodynamics.wss import (
    WallShearStressCalculator,
    HemodynamicDisturbanceResult,
)


def test_instantaneous_wss() -> None:
    calc = WallShearStressCalculator()
    area = np.array([np.pi * 0.010**2, np.pi * 0.020**2])
    flow = np.array([100e-6, 100e-6])

    wss = calc.instantaneous_wss(area, flow)
    assert len(wss) == 2
    # Smaller lumen radius experiences much higher shear rate and higher WSS: gamma ~ 1 / R^3
    assert wss[0] > wss[1]


def test_compute_indices_and_masks() -> None:
    calc = WallShearStressCalculator()
    time = np.linspace(0.0, 0.8, 80)
    z = np.linspace(0.0, 0.20, 21)

    # 80 time steps, 21 spatial nodes
    area = np.full((80, 21), np.pi * 0.010**2)
    # Sinusoidal flow with mild reversal at node 10
    flow = np.zeros((80, 21))
    for t_idx, t_val in enumerate(time):
        flow[t_idx, :] = 80e-6 * np.sin(2.0 * np.pi * t_val / 0.8)

    res = calc.compute_indices(time=time, z=z, area=area, flow_rate=flow)

    assert isinstance(res, HemodynamicDisturbanceResult)
    assert len(res.tawss) == 21
    assert len(res.osi) == 21
    assert len(res.rrt) == 21
    assert len(res.ecap) == 21

    # OSI should be strictly in [0, 0.5]
    assert np.all(res.osi >= 0.0)
    assert np.all(res.osi <= 0.5001)

    # TAWSS must be non-negative
    assert np.all(res.tawss >= 0.0)
