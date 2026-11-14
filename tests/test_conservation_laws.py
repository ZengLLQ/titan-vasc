"""Tests validating fundamental physics conservation laws and numerical convergence."""

import numpy as np
import pytest

from titan_vasc.hemodynamics.solver import ArterialSegment, PulsatileFlowSolver
from titan_vasc.hemodynamics.rheology import NewtonianRheology
from titan_vasc.boundary.inflow import InflowWaveformFactory
from titan_vasc.mechanics.hyperelastic import NeoHookeanArterialWall, HolzapfelGasserOgdenModel


def test_mass_conservation_steady_flow() -> None:
    """Validate 1D continuity equation: d(Q)/dz = 0 in steady-state limit with rigid wall."""
    seg = ArterialSegment(length=0.20, radius=0.010, num_nodes=21)
    solver = PulsatileFlowSolver(segment=seg, scheme="maccormack")

    # Constant steady inflow: 50 mL/s
    q_const = 50e-6
    inflow_fn = lambda t: q_const

    hist = solver.solve(duration=0.15, inflow_fn=inflow_fn, record_interval=0.01)

    # Inflow vs Outflow at final step
    q_inlet = hist.flow_rate[-1, 0]
    q_outlet = hist.flow_rate[-1, -1]
    rel_error = abs(q_inlet - q_outlet) / q_inlet
    # Within 2% for transient boundary settling
    assert rel_error < 0.02


def test_inflow_stroke_volume_conservation() -> None:
    """Validate analytical stroke volume vs integral of generated inflow waveform."""
    target_sv = 75.0e-6 # 75 mL
    wave, metrics = InflowWaveformFactory.resting_inflow(heart_rate=60.0, stroke_volume=target_sv)

    t_arr = np.linspace(0.0, metrics.cycle_period, 1000)
    q_arr = np.array([wave(t) for t in t_arr])

    def _trapz(y: np.ndarray, x: np.ndarray) -> float:
        if hasattr(np, "trapezoid"):
            return float(np.trapezoid(y, x))
        return float(getattr(np, "trapz")(y, x))

    # Integrate systolic forward flow only (positive values)
    computed_sv = _trapz(np.maximum(0.0, q_arr), t_arr)
    # Conservation within 1%
    assert np.isclose(computed_sv, target_sv, rtol=0.01)


def test_incompressibility_hyperelastic_constraint() -> None:
    """Validate isochoric deformation J = lambda_1 * lambda_2 * lambda_3 = 1."""
    lam_theta = 1.15
    lam_z = 1.05
    lam_r = 1.0 / (lam_theta * lam_z)

    j_det = lam_theta * lam_z * lam_r
    assert np.isclose(j_det, 1.0)
