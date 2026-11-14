"""Unit tests for 1D pulsatile Navier-Stokes finite difference flow solver."""

import numpy as np
import pytest

from titan_vasc.hemodynamics.solver import (
    ArterialSegment,
    PulsatileFlowSolver,
    SimulationHistory,
    generate_cardiac_cycle_inflow,
)


def test_arterial_segment_initialization() -> None:
    seg = ArterialSegment(length=0.20, radius=0.012, num_nodes=41)
    assert seg.length == 0.20
    assert seg.radius == 0.012
    assert seg.num_nodes == 41
    assert len(seg.z) == 41
    assert np.isclose(seg.z[-1], 0.20)
    assert len(seg.a_0) == 41
    assert np.isclose(seg.a_0[0], np.pi * 0.012**2)


def test_solver_flux_and_source() -> None:
    seg = ArterialSegment(length=0.10, radius=0.010, num_nodes=21)
    solver = PulsatileFlowSolver(segment=seg, scheme="maccormack")

    area = np.copy(seg.a_0)
    flow = np.full(21, 50e-6)

    f_a, f_q = solver.compute_flux(area, flow)
    assert len(f_a) == 21
    assert len(f_q) == 21
    np.testing.assert_allclose(f_a, flow)

    s_q = solver.compute_source(area, flow)
    assert len(s_q) == 21
    # Viscous dissipation should oppose flow (negative for positive Q)
    assert np.all(s_q < 0.0)


def test_solver_time_step_cfl() -> None:
    seg = ArterialSegment(length=0.25, radius=0.010, num_nodes=51)
    solver = PulsatileFlowSolver(segment=seg, cfl=0.7)

    area = np.copy(seg.a_0)
    flow = np.full(51, 80e-6)
    dt = solver.compute_time_step(area, flow)

    assert 1e-6 <= dt <= 1e-3


def test_maccormack_and_richtmyer_step() -> None:
    seg = ArterialSegment(length=0.10, radius=0.010, num_nodes=21)
    solver = PulsatileFlowSolver(segment=seg)

    area = np.copy(seg.a_0)
    flow = np.zeros(21)
    dt = 1e-4
    q_in = 60e-6

    # Test MacCormack
    a_mac, q_mac = solver.step_maccormack(area, flow, dt, q_in=q_in)
    assert len(a_mac) == 21
    assert len(q_mac) == 21
    assert q_mac[0] == q_in
    assert np.all(a_mac > 0.0)

    # Test Richtmyer
    a_rich, q_rich = solver.step_richtmyer(area, flow, dt, q_in=q_in)
    assert len(a_rich) == 21
    assert len(q_rich) == 21
    assert q_rich[0] == q_in
    assert np.all(a_rich > 0.0)


def test_full_transient_solution() -> None:
    seg = ArterialSegment(length=0.15, radius=0.010, num_nodes=21)
    solver = PulsatileFlowSolver(segment=seg, scheme="maccormack")

    inflow_fn = generate_cardiac_cycle_inflow(stroke_volume=60e-6, heart_rate=80.0)
    history = solver.solve(duration=0.2, inflow_fn=inflow_fn, record_interval=0.01)

    assert isinstance(history, SimulationHistory)
    assert len(history.time) >= 10
    assert history.area.shape == (len(history.time), 21)
    assert history.flow_rate.shape == (len(history.time), 21)
    assert history.pressure.shape == (len(history.time), 21)
    assert history.velocity.shape == (len(history.time), 21)
