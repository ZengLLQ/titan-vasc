"""Unit tests for 1D-0D coupled fluid-structure interaction (FSI) engine."""

import numpy as np
import pytest

from titan_vasc.mechanics.fsi import (
    DynamicWallState,
    CoupledFSIEngine,
)


def test_static_pressure_area_curve() -> None:
    fsi = CoupledFSIEngine()
    pressures, areas, compliances = fsi.compute_static_pressure_area_curve(
        reference_radius=0.010,
        wall_thickness=0.0015,
        pressure_range=(6000.0, 20000.0),
        num_points=15,
    )

    assert len(pressures) == 15
    assert len(areas) == 15
    # Area must monotonically increase with pressure
    assert np.all(np.diff(areas) > 0.0)
    # Compliance dA/dP must be positive
    assert np.all(compliances > 0.0)


def test_step_wall_dynamics() -> None:
    fsi = CoupledFSIEngine()
    r_ref = np.full(10, 0.010)
    h_ref = np.full(10, 0.0015)

    r_curr = np.copy(r_ref)
    vel_curr = np.zeros(10)
    p_fluid = np.full(10, 14000.0)
    dt = 1e-4

    state = fsi.step_wall_dynamics(
        current_radius=r_curr,
        radial_velocity=vel_curr,
        reference_radius=r_ref,
        wall_thickness=h_ref,
        fluid_pressure=p_fluid,
        dt=dt,
    )

    assert isinstance(state, DynamicWallState)
    assert len(state.radius) == 10
    assert len(state.radial_velocity) == 10
    assert len(state.circumferential_stress) == 10
    # Positive pressure jump causes positive radial velocity
    assert np.all(state.radial_velocity >= 0.0)
    assert state.dissipated_energy_density >= 0.0
