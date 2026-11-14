"""Example 1: Healthy Aorta vs Fusiform Aneurysm Pulsatile Hemodynamics.

Demonstrates:
- 1D transient pulsatile Navier-Stokes flow simulation using MacCormack scheme
- Non-Newtonian Carreau-Yasuda blood rheology
- Wall Shear Stress (WSS) and Oscillatory Shear Index (OSI) comparison
- Detection of atheroprone low-shear recirculating zones in aneurysm dome
"""

import sys
import numpy as np

from titan_vasc.hemodynamics.solver import ArterialSegment, PulsatileFlowSolver
from titan_vasc.hemodynamics.rheology import CarreauYasudaRheology
from titan_vasc.hemodynamics.wss import WallShearStressCalculator
from titan_vasc.boundary.inflow import InflowWaveformFactory
from titan_vasc.visualizer.ascii_canvas import render_waveform_plot, render_sparkline


def main() -> None:
    print("=" * 75)
    print("  TITANVASC EXAMPLE 1: HEALTHY AORTA vs FUSIFORM ANEURYSM HEMODYNAMICS")
    print("=" * 75)

    # Inflow waveform: normal adult resting (75 bpm, SV = 70 mL)
    inflow_fn, inflow_metrics = InflowWaveformFactory.resting_inflow(heart_rate=75.0, stroke_volume=70e-6)
    print(f"  Cardiac Input: {inflow_metrics.heart_rate:.0f} bpm | CO = {inflow_metrics.cardiac_output:.2f} L/min | SV = {inflow_metrics.stroke_volume * 1e6:.1f} mL")

    # Segment 1: Healthy abdominal aorta (25 cm, 10 mm radius)
    seg_healthy = ArterialSegment(length=0.25, radius=0.010, num_nodes=51)
    solver_healthy = PulsatileFlowSolver(segment=seg_healthy, scheme="maccormack", cfl=0.65)

    # Segment 2: Diseased aorta with aneurysm dilatation (dilated middle region)
    seg_aneurysm = ArterialSegment(length=0.25, radius=0.010, num_nodes=51)
    # Tapered/dilated initial area profile representing a 5.2 cm aneurysm at center
    xi = np.linspace(0.0, 1.0, 51)
    aneurysm_mask = np.sin(np.pi * xi) ** 2
    r_aneurysm = 0.010 + (0.026 - 0.010) * aneurysm_mask
    seg_aneurysm.a_0 = np.pi * (r_aneurysm**2)

    solver_aneurysm = PulsatileFlowSolver(segment=seg_aneurysm, scheme="maccormack", cfl=0.65)

    print("\n  [1/2] Running Healthy Aorta Simulation (0.8s)...")
    hist_healthy = solver_healthy.solve(duration=0.8, inflow_fn=inflow_fn, record_interval=0.005)

    print("  [2/2] Running Aneurysmal Aorta Simulation (0.8s)...")
    hist_aneurysm = solver_aneurysm.solve(duration=0.8, inflow_fn=inflow_fn, record_interval=0.005)

    # Calculate WSS indices
    wss_calc = WallShearStressCalculator(rheology=CarreauYasudaRheology())
    wss_healthy = wss_calc.compute_indices(
        time=hist_healthy.time,
        z=hist_healthy.z,
        area=hist_healthy.area,
        flow_rate=hist_healthy.flow_rate,
    )
    wss_aneurysm = wss_calc.compute_indices(
        time=hist_aneurysm.time,
        z=hist_aneurysm.z,
        area=hist_aneurysm.area,
        flow_rate=hist_aneurysm.flow_rate,
    )

    print("\n" + "-" * 75)
    print("  HEMODYNAMIC COMPARISON SUMMARY:")
    print(f"    Metric                         Healthy Aorta       Aneurysmal Aorta")
    print(f"    -------------------------------------------------------------------")
    print(f"    Peak Systolic Velocity (m/s) : {float(np.max(hist_healthy.velocity)):<19.3f} {float(np.max(hist_aneurysm.velocity)):.3f}")
    print(f"    Mean TAWSS (Pa)              : {float(np.mean(wss_healthy.tawss)):<19.3f} {float(np.mean(wss_aneurysm.tawss)):.3f}")
    print(f"    Min TAWSS (stagnant zone, Pa): {float(np.min(wss_healthy.tawss)):<19.3f} {float(np.min(wss_aneurysm.tawss)):.3f}")
    print(f"    Max Oscillatory Index (OSI)  : {float(np.max(wss_healthy.osi)):<19.3f} {float(np.max(wss_aneurysm.osi)):.3f}")
    print(f"    Atheroprone Nodes (<0.4 Pa)  : {int(np.sum(wss_healthy.atheroprone_mask)):<19d} {int(np.sum(wss_aneurysm.atheroprone_mask))} / 51")
    print("-" * 75)

    # Render ASCII sparkline of TAWSS along the vessel
    print("\n  TAWSS Axial Profile (Healthy):")
    print("  [" + render_sparkline(wss_healthy.tawss, width=50) + f"] Mean = {float(np.mean(wss_healthy.tawss)):.2f} Pa")
    print("  TAWSS Axial Profile (Aneurysm):")
    print("  [" + render_sparkline(wss_aneurysm.tawss, width=50) + f"] Stagnant Dip = {float(np.min(wss_aneurysm.tawss)):.2f} Pa")

    print("\n" + render_waveform_plot(
        time=hist_aneurysm.time,
        signal=hist_aneurysm.pressure[:, 25] / 133.322,
        title="PRESSURE WAVE AT ANEURYSM CENTER (mmHg)",
        width=68,
        height=13,
    ))
    print("=" * 75)


if __name__ == "__main__":
    main()
