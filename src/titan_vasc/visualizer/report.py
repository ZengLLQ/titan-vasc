"""Executive clinical hemodynamic and rupture risk report generator.

Outputs an exhaustive, publication-grade diagnostic report in 100% pure 7-bit ASCII.
"""

from __future__ import annotations

from titan_vasc.calibration.patient import PatientProfile
from titan_vasc.calibration.scoring import ClinicalScoringResult
from titan_vasc.calibration.sensitivity import UncertaintyQuantificationResult
from titan_vasc.mechanics.rupture import RuptureRiskResult
from titan_vasc.hemodynamics.wss import HemodynamicDisturbanceResult
from titan_vasc.visualizer.ascii_canvas import (
    render_aortic_geometry,
    render_stress_distribution,
    render_waveform_plot,
)


class ClinicalReportGenerator:
    """Generates complete diagnostic and surgical recommendation reports in 7-bit ASCII."""

    @staticmethod
    def generate_full_report(
        patient: PatientProfile,
        rupture_result: RuptureRiskResult,
        clinical_score: ClinicalScoringResult,
        wss_result: HemodynamicDisturbanceResult | None = None,
        uq_result: UncertaintyQuantificationResult | None = None,
    ) -> str:
        """Compile comprehensive multi-subsystem diagnostic report."""
        lines: list[str] = []
        sep = "=" * 76
        sub_sep = "-" * 76

        lines.append(sep)
        lines.append("  TITANVASC: CARDIOVASCULAR HEMODYNAMICS & BIOMECHANICAL RUPTURE REPORT")
        lines.append("             Computational Aortic Aneurysm Clinical Assessment")
        lines.append(sep)
        lines.append(f"  Patient ID      : {patient.patient_id:<20} Date / Cycle : Active Assessment")
        lines.append(f"  Age / Sex       : {patient.age:.0f} yrs / {patient.sex.upper():<13} Smoking      : {patient.smoking_status.upper()}")
        lines.append(f"  Blood Pressure  : {patient.systolic_bp:.0f}/{patient.diastolic_bp:.0f} mmHg (MAP={patient.mean_arterial_pressure_mmhg:.1f} mmHg, PP={patient.pulse_pressure_mmhg:.1f} mmHg)")
        lines.append(f"  Heart Rate      : {patient.heart_rate:.0f} bpm               COPD / Fam Hx: {patient.copd} / {patient.family_history_aaa}")
        lines.append(sub_sep)

        # Section 1: Morphology
        d_max_cm = patient.max_aneurysm_diameter * 100.0
        d_healthy_cm = patient.healthy_aorta_diameter * 100.0
        lines.append("  [1] MORPHOLOGICAL & ANATOMICAL CHARACTERISTICS")
        lines.append(f"      Maximum Diameter (D_max) : {d_max_cm:.2f} cm (Healthy ref: {d_healthy_cm:.2f} cm)")
        lines.append(f"      Dilatation Ratio (chi)   : {patient.dilatation_ratio:.2f}x baseline diameter")
        lines.append(f"      Aneurysm Sac Length      : {patient.aneurysm_length * 100.0:.1f} cm")
        lines.append(f"      Infrarenal Neck Angle    : {patient.neck_angle_deg:.1f} deg")
        lines.append(f"      Intraluminal Thrombus    : {patient.ilt_thickness * 1000.0:.1f} mm maximum thickness")
        lines.append(f"      Geometry Configuration   : {patient.morphology_type.upper()}")
        lines.append("")

        # ASCII geometry diagram
        geom_ascii = render_aortic_geometry(
            z=rupture_result.z,
            radius=rupture_result.circumferential_stress * 0.0 + (patient.healthy_aorta_diameter / 2.0),
            wall_thickness=rupture_result.circumferential_stress * 0.0 + 0.0015,
            width=72,
            height=13,
        )
        lines.append(geom_ascii)
        lines.append(sub_sep)

        # Section 2: Biomechanical Wall Stress & Rupture Criteria
        lines.append("  [2] BIOMECHANICAL WALL STRESS & RUPTURE RISK ASSESSMENT")
        lines.append(f"      Peak Wall Stress (PWS)   : {rupture_result.peak_wall_stress / 1e3:.1f} kPa (Von Mises)")
        lines.append(f"      Peak Rupture Potential   : RPI = {rupture_result.peak_rpi:.3f} (Failure at >= 1.0)")
        lines.append(f"      Critical Failure Site    : z = {rupture_result.peak_rpi_location * 100.0:.1f} cm")
        lines.append(f"      Risk Equivalent Diameter : RRED = {rupture_result.rupture_risk_equivalent_diameter * 100.0:.2f} cm")
        lines.append(f"      Clinical Risk Tier       : [{rupture_result.clinical_risk_tier.upper()}]")
        lines.append("")

        # ASCII stress distribution
        stress_dist_ascii = render_stress_distribution(
            z=rupture_result.z,
            stress=rupture_result.von_mises_stress,
            strength=rupture_result.tensile_wall_strength,
            width=72,
        )
        lines.append(stress_dist_ascii)
        lines.append(sub_sep)

        # Section 3: Hemodynamic Disturbance
        if wss_result is not None:
            lines.append("  [3] CYCLE-AVERAGED HEMODYNAMIC DISTURBANCE (WSS & OSI)")
            lines.append(f"      Mean TAWSS across Aorta  : {float(np.mean(wss_result.tawss)):.2f} Pa (Normal: 1.0 - 1.5 Pa)")
            lines.append(f"      Minimum TAWSS (Stagnant) : {float(np.min(wss_result.tawss)):.2f} Pa (Atheroprone if < 0.4 Pa)")
            lines.append(f"      Peak Oscillatory Index   : OSI_max = {float(np.max(wss_result.osi)):.3f} (Max theoretical: 0.500)")
            lines.append(f"      Atheroprone Wall Length  : {float(np.sum(wss_result.atheroprone_mask)) * float(wss_result.z[1] - wss_result.z[0]) * 100.0:.1f} cm")
            lines.append(sub_sep)

        # Section 4: Uncertainty Quantification
        if uq_result is not None:
            lines.append("  [4] MONTE CARLO UNCERTAINTY QUANTIFICATION (500 Realizations)")
            lines.append(f"      PWS 95% Confidence Int.  : [{uq_result.pws_ci95[0]/1e3:.1f} kPa, {uq_result.pws_ci95[1]/1e3:.1f} kPa]")
            lines.append(f"      RPI 95% Confidence Int.  : [{uq_result.rpi_ci95[0]:.3f}, {uq_result.rpi_ci95[1]:.3f}]")
            lines.append(f"      Imminent Rupture Prob.   : P(RPI >= 1.0) = {uq_result.critical_rupture_probability * 100.0:.2f}%")
            lines.append("      Parameter Sensitivity Rank (Morris mu*):")
            for rank_idx, (pname, peffect) in enumerate(uq_result.parameter_rankings[:3], 1):
                lines.append(f"        {rank_idx}. {pname:<26} : mu* = {peffect:.4f}")
            lines.append(sub_sep)

        # Section 5: Clinical Decisions & SVS Guidelines
        lines.append("  [5] CLINICAL SCORING & SVS PRACTICE GUIDELINES RECOMMENDATION")
        lines.append(f"      UKSAT Annual Rupture Risk: {clinical_score.annual_rupture_risk_pct:.1f}% / year")
        lines.append(f"      UKSAT 5-Year Cumulative  : {clinical_score.five_year_rupture_risk_pct:.1f}%")
        lines.append(f"      Composite RESCUE Score   : {clinical_score.rescue_score:.1f} / 100")
        lines.append(f"      SVS Elective Repair      : {'*** INDICATED ***' if clinical_score.svs_repair_indicated else 'Not Indicated at Present'}")
        if not clinical_score.svs_repair_indicated:
            lines.append(f"      Surveillance Protocol    : CTA or Duplex Ultrasound in {clinical_score.recommended_surveillance_months} months")
        lines.append(f"      Standard EVAR Feasibility: {'ANATOMICALLY SUITABLE' if clinical_score.evar_anatomically_suitable else 'CONTRAINDICATED / COMPLEX'}")
        if not clinical_score.evar_anatomically_suitable:
            for contra in clinical_score.contraindications_evar:
                lines.append(f"        - Alert: {contra}")
            lines.append("        - Suggestion: Consider Fenestrated/Branched EVAR (F/B-EVAR) or Open Surgical Repair")

        lines.append(sep)
        lines.append("  [END OF CLINICAL REPORT - TITANVASC COMPUTATIONAL ENGINE]")
        lines.append(sep)
        return "\n".join(lines)
