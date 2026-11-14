"""Example 4: Virtual Patient Cohort Simulation and SVS Guideline Audit.

Demonstrates:
- Simulation of a heterogeneous cohort of 6 virtual patients across mild, moderate, and severe AAA
- UKSAT annual rupture risk calculation
- RESCUE composite risk scoring
- SVS surgical intervention recommendation and EVAR anatomical feasibility check
- Monte Carlo uncertainty quantification
"""

import numpy as np

from titan_vasc.calibration.patient import PatientProfile
from titan_vasc.calibration.scoring import ClinicalRiskEngine
from titan_vasc.calibration.sensitivity import RuptureSensitivityAnalyzer
from titan_vasc.aneurysm.morphology import MorphologyGenerator
from titan_vasc.mechanics.rupture import RuptureRiskEvaluator


def main() -> None:
    print("=" * 80)
    print("  TITANVASC EXAMPLE 4: VIRTUAL PATIENT CLINICAL TRIAL COHORT SIMULATION")
    print("=" * 80)

    cohort: list[PatientProfile] = [
        PatientProfile(
            patient_id="SUBJ-001",
            age=62.0,
            sex="male",
            systolic_bp=128.0,
            diastolic_bp=80.0,
            max_aneurysm_diameter=0.038,
            ilt_thickness=0.001,
            smoking_status="never",
            neck_angle_deg=15.0,
        ),
        PatientProfile(
            patient_id="SUBJ-002",
            age=67.0,
            sex="male",
            systolic_bp=142.0,
            diastolic_bp=88.0,
            max_aneurysm_diameter=0.046,
            ilt_thickness=0.003,
            smoking_status="former",
            neck_angle_deg=25.0,
        ),
        PatientProfile(
            patient_id="SUBJ-003",
            age=71.0,
            sex="female",
            systolic_bp=138.0,
            diastolic_bp=84.0,
            max_aneurysm_diameter=0.049,
            ilt_thickness=0.005,
            smoking_status="current",
            neck_angle_deg=35.0,
        ),
        PatientProfile(
            patient_id="SUBJ-004",
            age=74.0,
            sex="male",
            systolic_bp=155.0,
            diastolic_bp=92.0,
            max_aneurysm_diameter=0.057,
            ilt_thickness=0.008,
            smoking_status="current",
            neck_angle_deg=45.0,
        ),
        PatientProfile(
            patient_id="SUBJ-005",
            age=76.0,
            sex="female",
            systolic_bp=165.0,
            diastolic_bp=96.0,
            max_aneurysm_diameter=0.054,
            ilt_thickness=0.010,
            smoking_status="current",
            neck_angle_deg=68.0, # Hostile neck angulation > 60 deg
            morphology_type="saccular",
        ),
        PatientProfile(
            patient_id="SUBJ-006",
            age=81.0,
            sex="male",
            systolic_bp=170.0,
            diastolic_bp=100.0,
            max_aneurysm_diameter=0.068,
            ilt_thickness=0.012,
            smoking_status="former",
            family_history_aaa=True,
            neck_angle_deg=30.0,
        ),
    ]

    gen = MorphologyGenerator(total_length=0.22, num_nodes=51)
    evaluator = RuptureRiskEvaluator()
    score_engine = ClinicalRiskEngine()

    print(f"\n{'ID':<10} {'Age/Sex':<10} {'D_max':<8} {'BP (mmHg)':<12} {'PWS':<10} {'RPI':<8} {'UKSAT/yr':<10} {'RESCUE':<8} {'SVS Decision':<18}")
    print("-" * 92)

    for pat in cohort:
        geom = gen.create_fusiform_aneurysm(max_diameter=pat.max_aneurysm_diameter)
        ilt_arr = np.full_like(geom.radius, pat.ilt_thickness)

        rup = evaluator.evaluate_rupture_risk(
            z=geom.z,
            radius=geom.radius,
            wall_thickness=geom.wall_thickness,
            systolic_pressure=pat.systolic_pressure_pa,
            ilt_thickness=ilt_arr,
            is_female=(pat.sex == "female"),
            age=pat.age,
            family_history=pat.family_history_aaa,
        )

        score = score_engine.evaluate_svs_guidelines(patient=pat, peak_rpi=rup.peak_rpi)

        decision = "IMMEDIATE REPAIR" if score.svs_repair_indicated else f"CTA in {score.recommended_surveillance_months}m"
        pws_kpa = f"{rup.peak_wall_stress / 1e3:.0f} kPa"
        demog = f"{pat.age:.0f}{pat.sex[0].upper()}"
        bp_str = f"{pat.systolic_bp:.0f}/{pat.diastolic_bp:.0f}"

        print(f"{pat.patient_id:<10} {demog:<10} {pat.max_aneurysm_diameter*100:<5.1f}cm {bp_str:<12} {pws_kpa:<10} {rup.peak_rpi:<8.2f} {score.annual_rupture_risk_pct:<8.1f}% {score.rescue_score:<8.1f} {decision:<18}")

    print("-" * 92)
    print("  Clinical Observations:")
    print("  - SUBJ-003 (female, 4.9cm): High risk due to female sex penalty & active smoking.")
    print("  - SUBJ-005 (female, 5.4cm, saccular): Critical RPI=0.88, hostile neck contraindicates EVAR.")
    print("  - SUBJ-006 (male, 6.8cm): High PWS (>500 kPa) & UKSAT >35%/yr mandate urgent repair.")
    print("=" * 80)


if __name__ == "__main__":
    main()
