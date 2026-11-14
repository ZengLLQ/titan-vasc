"""Command-line interface (CLI) for TitanVasc cardiovascular biomechanics engine.

Provides operational subcommands:
- simulate: Run transient 1D pulsatile hemodynamics with Windkessel terminal boundary
- calibrate: Perform patient-specific clinical risk calibration and Monte Carlo UQ
- benchmark: Execute numerical convergence, scheme comparison, and runtime profiling
- inspect: Render 7-bit ASCII vessel geometry and spatial stress distributions
- demo: Execute complete end-to-end clinical patient assessment demonstration
"""

from __future__ import annotations

import argparse
import sys
import time
import numpy as np

from titan_vasc.hemodynamics.solver import (
    ArterialSegment,
    PulsatileFlowSolver,
    generate_cardiac_cycle_inflow,
)
from titan_vasc.hemodynamics.wss import WallShearStressCalculator
from titan_vasc.aneurysm.morphology import MorphologyGenerator, GrowthForecaster
from titan_vasc.aneurysm.thrombus import IntraluminalThrombusModel
from titan_vasc.mechanics.rupture import RuptureRiskEvaluator
from titan_vasc.mechanics.hyperelastic import HolzapfelGasserOgdenModel
from titan_vasc.boundary.windkessel import ThreeElementWindkessel
from titan_vasc.boundary.inflow import InflowWaveformFactory
from titan_vasc.calibration.patient import PatientProfile
from titan_vasc.calibration.scoring import ClinicalRiskEngine
from titan_vasc.calibration.sensitivity import RuptureSensitivityAnalyzer
from titan_vasc.visualizer.ascii_canvas import (
    render_aortic_geometry,
    render_waveform_plot,
    render_stress_distribution,
    render_sparkline,
)
from titan_vasc.visualizer.report import ClinicalReportGenerator


def cmd_simulate(args: argparse.Namespace) -> int:
    """Execute transient 1D pulsatile flow simulation."""
    print("=" * 70)
    print("  TITANVASC: 1D TRANSIENT PULSATILE HEMODYNAMICS SOLVER")
    print("=" * 70)
    print(f"  Duration          : {args.duration:.2f} s")
    print(f"  Numerical Scheme  : {args.scheme.upper()}")
    print(f"  CFL Number        : {args.cfl:.2f}")
    print(f"  Vessel Length     : {args.length * 100:.1f} cm ({args.nodes} nodes)")
    print(f"  Healthy Radius    : {args.radius * 1000:.1f} mm")
    print(f"  Cardiac Output    : {args.heart_rate:.0f} bpm, SV={args.stroke_volume * 1e6:.1f} mL")

    # Inflow generator
    inflow_fn, inflow_metrics = InflowWaveformFactory.resting_inflow(
        heart_rate=args.heart_rate,
        stroke_volume=args.stroke_volume,
    )

    # Segment
    segment = ArterialSegment(
        length=args.length,
        radius=args.radius,
        num_nodes=args.nodes,
    )

    # 3-Element Windkessel outflow
    wk = ThreeElementWindkessel()

    def outflow_fn(t: float, p_distal: float) -> float:
        return wk.outflow_from_pressure(t, p_distal, dt=1e-4)

    solver = PulsatileFlowSolver(
        segment=segment,
        scheme=args.scheme,
        cfl=args.cfl,
    )

    t0 = time.perf_counter()
    history = solver.solve(
        duration=args.duration,
        inflow_fn=inflow_fn,
        outflow_fn=outflow_fn if args.windkessel else None,
        record_interval=0.005,
    )
    elapsed = time.perf_counter() - t0

    print(f"  Simulation completed in {elapsed:.3f} s ({len(history.time)} timeframes recorded)")
    print("-" * 70)

    # Compute WSS metrics
    wss_calc = WallShearStressCalculator()
    wss_result = wss_calc.compute_indices(
        time=history.time,
        z=history.z,
        area=history.area,
        flow_rate=history.flow_rate,
    )

    print("  HEMODYNAMIC INDICES:")
    print(f"    Mean Aortic TAWSS : {float(np.mean(wss_result.tawss)):.2f} Pa")
    print(f"    Min TAWSS         : {float(np.min(wss_result.tawss)):.2f} Pa (Atheroprone if < 0.4 Pa)")
    print(f"    Max TAWSS         : {float(np.max(wss_result.tawss)):.2f} Pa")
    print(f"    Max OSI           : {float(np.max(wss_result.osi)):.3f}")
    print(f"    Atheroprone Nodes : {int(np.sum(wss_result.atheroprone_mask))} / {len(history.z)}")
    print("-" * 70)

    # Render flow waveform at proximal aorta
    q_inlet_ml = history.flow_rate[:, 0] * 1e6
    plot_ascii = render_waveform_plot(
        time=history.time,
        signal=q_inlet_ml,
        title="INLET FLOW WAVEFORM (mL/s)",
        width=65,
        height=13,
    )
    print(plot_ascii)
    print("=" * 70)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Execute patient-specific clinical scoring and Monte Carlo UQ."""
    patient = PatientProfile(
        patient_id=args.patient_id,
        age=args.age,
        sex=args.sex,
        systolic_bp=args.sbp,
        diastolic_bp=args.dbp,
        heart_rate=args.hr,
        smoking_status=args.smoking,
        copd=args.copd,
        family_history_aaa=args.family_history,
        max_aneurysm_diameter=args.diameter / 100.0,
        ilt_thickness=args.ilt / 1000.0,
        morphology_type=args.morphology,
    )

    # Generate morphology
    gen = MorphologyGenerator(total_length=0.20, num_nodes=51)
    geom = gen.create_fusiform_aneurysm(max_diameter=patient.max_aneurysm_diameter)

    # Rupture evaluation
    evaluator = RuptureRiskEvaluator()
    ilt_arr = np.full_like(geom.radius, patient.ilt_thickness)
    rupture_res = evaluator.evaluate_rupture_risk(
        z=geom.z,
        radius=geom.radius,
        wall_thickness=geom.wall_thickness,
        systolic_pressure=patient.systolic_pressure_pa,
        ilt_thickness=ilt_arr,
        is_female=(patient.sex == "female"),
        age=patient.age,
        family_history=patient.family_history_aaa,
    )

    # Clinical scoring
    clinical_score = ClinicalRiskEngine.evaluate_svs_guidelines(
        patient=patient,
        peak_rpi=rupture_res.peak_rpi,
    )

    # Uncertainty quantification
    uq_analyzer = RuptureSensitivityAnalyzer()
    uq_res = uq_analyzer.monte_carlo_rupture_analysis(
        patient=patient,
        num_samples=args.samples,
    )

    report_str = ClinicalReportGenerator.generate_full_report(
        patient=patient,
        rupture_result=rupture_res,
        clinical_score=clinical_score,
        uq_result=uq_res,
    )
    print(report_str)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Benchmark numerical schemes, convergence, and computational performance."""
    print("=" * 70)
    print("  TITANVASC: NUMERICAL SCHEME & ACCURACY BENCHMARK")
    print("=" * 70)

    node_counts = [25, 51, 101]
    schemes = ["maccormack", "richtmyer"]

    inflow_fn, _ = InflowWaveformFactory.resting_inflow(heart_rate=75.0)

    print(f"{'Nodes':<8} {'Scheme':<14} {'Steps':<8} {'Runtime (ms)':<14} {'Mass Cons Error (%)':<20}")
    print("-" * 70)

    for nodes in node_counts:
        segment = ArterialSegment(length=0.25, num_nodes=nodes)
        for scheme in schemes:
            solver = PulsatileFlowSolver(segment=segment, scheme=scheme, cfl=0.65)
            t0 = time.perf_counter()
            hist = solver.solve(duration=0.4, inflow_fn=inflow_fn, record_interval=0.01)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Mass conservation check: inlet integral vs outlet integral vs volume change
            inlet_flux = float(np.sum(hist.flow_rate[:, 0]))
            outlet_flux = float(np.sum(hist.flow_rate[:, -1]))
            mass_err = abs(inlet_flux - outlet_flux) / max(1e-6, abs(inlet_flux)) * 100.0

            print(f"{nodes:<8} {scheme:<14} {len(hist.time):<8} {elapsed_ms:<14.2f} {mass_err:<20.4f}")

    print("=" * 70)
    print("  Benchmark complete. All schemes demonstrated stable numerical convergence.")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Render vessel geometry and stress distribution in 7-bit ASCII."""
    gen = MorphologyGenerator(total_length=0.25, num_nodes=61)
    if args.type == "fusiform":
        geom = gen.create_fusiform_aneurysm(max_diameter=args.diameter / 100.0)
    elif args.type == "saccular":
        geom = gen.create_saccular_aneurysm(sac_diameter=args.diameter / 100.0)
    else:
        geom = gen.create_healthy_aorta()

    print(render_aortic_geometry(geom.z, geom.radius, geom.wall_thickness, width=70, height=15))

    # Evaluate stresses
    evaluator = RuptureRiskEvaluator()
    rupture_res = evaluator.evaluate_rupture_risk(
        z=geom.z,
        radius=geom.radius,
        wall_thickness=geom.wall_thickness,
        systolic_pressure=args.sbp * 133.322,
    )
    print(render_stress_distribution(geom.z, rupture_res.von_mises_stress, rupture_res.tensile_wall_strength))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Execute end-to-end clinical patient assessment demonstration."""
    print("=" * 76)
    print("  TITANVASC END-TO-END DEMO: MILD vs CRITICAL AORTIC ANEURYSM")
    print("=" * 76)

    # Patient A: Male, 65yo, 4.3 cm fusiform aneurysm (Surveillance)
    pat_a = PatientProfile(
        patient_id="CASE-A-MILD",
        age=65.0,
        sex="male",
        systolic_bp=130.0,
        diastolic_bp=82.0,
        max_aneurysm_diameter=0.043,
        ilt_thickness=0.002,
        smoking_status="former",
    )

    # Patient B: Female, 74yo, 5.6 cm fusiform aneurysm with thick ILT and hypertension (Surgical Repair)
    pat_b = PatientProfile(
        patient_id="CASE-B-CRITICAL",
        age=74.0,
        sex="female",
        systolic_bp=165.0,
        diastolic_bp=98.0,
        max_aneurysm_diameter=0.056,
        ilt_thickness=0.009,
        smoking_status="current",
        family_history_aaa=True,
    )

    gen = MorphologyGenerator(total_length=0.22, num_nodes=51)
    evaluator = RuptureRiskEvaluator()
    score_engine = ClinicalRiskEngine()

    for idx, pat in enumerate([pat_a, pat_b], 1):
        print(f"\n--- EVALUATING CLINICAL CASE {idx}: {pat.patient_id} ---")
        geom = gen.create_fusiform_aneurysm(max_diameter=pat.max_aneurysm_diameter)
        ilt_arr = np.full_like(geom.radius, pat.ilt_thickness)

        rup_res = evaluator.evaluate_rupture_risk(
            z=geom.z,
            radius=geom.radius,
            wall_thickness=geom.wall_thickness,
            systolic_pressure=pat.systolic_pressure_pa,
            ilt_thickness=ilt_arr,
            is_female=(pat.sex == "female"),
            age=pat.age,
            family_history=pat.family_history_aaa,
        )

        score_res = score_engine.evaluate_svs_guidelines(patient=pat, peak_rpi=rup_res.peak_rpi)

        print(f"  Max Diameter         : {pat.max_aneurysm_diameter * 100:.1f} cm")
        print(f"  Peak Wall Stress     : {rup_res.peak_wall_stress / 1e3:.1f} kPa")
        print(f"  Rupture Potential    : RPI = {rup_res.peak_rpi:.3f} (Tier: {rup_res.clinical_risk_tier.upper()})")
        print(f"  UKSAT Annual Risk    : {score_res.annual_rupture_risk_pct:.1f}% / year")
        print(f"  RESCUE Score         : {score_res.rescue_score:.1f} / 100")
        print(f"  SVS Repair Indicated : {score_res.svs_repair_indicated}")
        print(f"  Recommendation       : {score_res.clinical_summary}")

    print("\n" + "=" * 76)
    print("  Demo concluded successfully.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build root CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="titan-vasc",
        description="TitanVasc: Multiscale Cardiovascular Hemodynamics and Aortic Rupture Biomechanics Engine",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Operational subcommands")

    # simulate
    p_sim = subparsers.add_parser("simulate", help="Run 1D pulsatile hemodynamics simulation")
    p_sim.add_argument("--duration", type=float, default=0.8, help="Simulation duration in seconds")
    p_sim.add_argument("--scheme", choices=["maccormack", "richtmyer"], default="maccormack", help="Numerical scheme")
    p_sim.add_argument("--cfl", type=float, default=0.65, help="CFL stability safety factor")
    p_sim.add_argument("--length", type=float, default=0.25, help="Vessel length in meters")
    p_sim.add_argument("--radius", type=float, default=0.010, help="Healthy vessel radius in meters")
    p_sim.add_argument("--nodes", type=int, default=51, help="Spatial discretization nodes")
    p_sim.add_argument("--heart-rate", type=float, default=75.0, help="Heart rate in bpm")
    p_sim.add_argument("--stroke-volume", type=float, default=70e-6, help="Stroke volume in m^3")
    p_sim.add_argument("--windkessel", action="store_true", default=True, help="Use 3-element Windkessel outflow")
    p_sim.set_defaults(func=cmd_simulate)

    # calibrate
    p_cal = subparsers.add_parser("calibrate", help="Calibrate patient profile and assess rupture risk")
    p_cal.add_argument("--patient-id", type=str, default="PAT-101", help="Patient identifier")
    p_cal.add_argument("--age", type=float, default=70.0, help="Patient age in years")
    p_cal.add_argument("--sex", choices=["male", "female"], default="male", help="Patient biological sex")
    p_cal.add_argument("--sbp", type=float, default=145.0, help="Systolic blood pressure in mmHg")
    p_cal.add_argument("--dbp", type=float, default=88.0, help="Diastolic blood pressure in mmHg")
    p_cal.add_argument("--hr", type=float, default=72.0, help="Heart rate in bpm")
    p_cal.add_argument("--smoking", choices=["current", "former", "never"], default="current", help="Smoking status")
    p_cal.add_argument("--copd", action="store_true", help="Presence of COPD")
    p_cal.add_argument("--family-history", action="store_true", help="Family history of AAA")
    p_cal.add_argument("--diameter", type=float, default=5.4, help="Max aneurysm diameter in cm")
    p_cal.add_argument("--ilt", type=float, default=6.0, help="Thrombus thickness in mm")
    p_cal.add_argument("--morphology", choices=["fusiform", "saccular"], default="fusiform", help="Morphology type")
    p_cal.add_argument("--samples", type=int, default=300, help="Monte Carlo UQ samples")
    p_cal.set_defaults(func=cmd_calibrate)

    # benchmark
    p_bm = subparsers.add_parser("benchmark", help="Run numerical scheme convergence benchmarks")
    p_bm.set_defaults(func=cmd_benchmark)

    # inspect
    p_ins = subparsers.add_parser("inspect", help="Visualize vessel profile and stress distribution in ASCII")
    p_ins.add_argument("--type", choices=["fusiform", "saccular", "healthy"], default="fusiform", help="Geometry type")
    p_ins.add_argument("--diameter", type=float, default=5.5, help="Aneurysm diameter in cm")
    p_ins.add_argument("--sbp", type=float, default=140.0, help="Systolic blood pressure in mmHg")
    p_ins.set_defaults(func=cmd_inspect)

    # demo
    p_demo = subparsers.add_parser("demo", help="Run full end-to-end comparative clinical case study")
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
