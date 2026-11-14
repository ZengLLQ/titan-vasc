"""Example 3: Wave Intensity Analysis (WIA) and 3-Element Windkessel Coupling.

Demonstrates:
- Coupling of 1D wave propagation to lumped parameter Windkessel outflow
- Forward and backward wave decomposition: dP_+ and dP_-
- Wave Intensity Analysis: dI = dP * dU
- Quantification of pathological reflected waves originating from aneurysmal bifurcation
"""

import numpy as np

from titan_vasc.hemodynamics.solver import ArterialSegment, PulsatileFlowSolver
from titan_vasc.hemodynamics.wave import WaveSeparator
from titan_vasc.boundary.windkessel import ThreeElementWindkessel
from titan_vasc.boundary.inflow import InflowWaveformFactory
from titan_vasc.visualizer.ascii_canvas import render_waveform_plot


def main() -> None:
    print("=" * 75)
    print("  TITANVASC EXAMPLE 3: WAVE INTENSITY ANALYSIS & WINDKESSEL COUPLING")
    print("=" * 75)

    # 1. Setup arterial segment with distal Windkessel
    segment = ArterialSegment(length=0.25, radius=0.011, num_nodes=51)
    wk = ThreeElementWindkessel(r_c=1.1e8, r_p=1.2e9, compliance=1.2e-9)

    inflow_fn, _ = InflowWaveformFactory.resting_inflow(heart_rate=70.0, stroke_volume=72e-6)

    def outflow_fn(t: float, p_distal: float) -> float:
        return wk.outflow_from_pressure(t, p_distal, dt=1e-4)

    solver = PulsatileFlowSolver(segment=segment, scheme="maccormack", cfl=0.65)

    print("  Running transient Windkessel-coupled pulsatile simulation (1.2s)...")
    history = solver.solve(duration=1.2, inflow_fn=inflow_fn, outflow_fn=outflow_fn, record_interval=0.005)

    # 2. Extract midpoint waveforms for Wave Intensity Analysis (WIA)
    mid_idx = 25
    time_arr = history.time
    p_mid = history.pressure[:, mid_idx]
    u_mid = history.velocity[:, mid_idx]
    a_mid = history.area[:, mid_idx]

    separator = WaveSeparator(tube_law=segment.tube_law)
    # Estimate local pulse wave velocity c = sqrt(A / (rho * C))
    c_pwv = float(segment.tube_law.pulse_wave_velocity(np.mean(a_mid)))

    wia_result = separator.separate(
        time=time_arr,
        pressure=p_mid,
        flow_rate=history.flow_rate[:, mid_idx],
        area=a_mid,
    )

    print("\n" + "-" * 75)
    print(f"  WAVE INTENSITY ANALYSIS METRICS (Aorta Midpoint, PWV = {c_pwv:.2f} m/s):")
    print(f"    Peak Forward Pressure (P+)    : {float(np.max(wia_result.p_forward)) / 133.322:.1f} mmHg")
    print(f"    Peak Backward Pressure (P-)   : {float(np.max(wia_result.p_backward)) / 133.322:.1f} mmHg")
    print(f"    Augmentation Index (AIx)      : {wia_result.augmentation_index * 100:.1f}%")
    print(f"    Wave Reflection Index (WRI)   : {wia_result.reflection_index:.3f}")
    print(f"    Peak Forward Wave Intensity   : {float(np.max(wia_result.wave_intensity_forward)):.2e} W/m^2")
    print(f"    Peak Backward Wave Intensity  : {float(np.max(wia_result.wave_intensity_backward)):.2e} W/m^2")
    print("-" * 75)

    # Plot total vs forward pressure wave
    plot_ascii = render_waveform_plot(
        time=time_arr,
        signal=p_mid / 133.322,
        title="TOTAL AORTIC PRESSURE (mmHg)",
        width=68,
        height=13,
    )
    print(plot_ascii)
    print("=" * 75)


if __name__ == "__main__":
    main()
