"""Clinical risk calculators and Society for Vascular Surgery (SVS) guideline stratifiers.

Implements evidence-based clinical rupture risk estimation and surgical intervention thresholds:
- UK Small Aneurysm Trial (UKSAT) annual rupture probability model
- RESCUE clinical score (Rupture Evaluation and Stratification for Clinical Urgent Evacuation)
- Society for Vascular Surgery (SVS) 2018/2024 practice guidelines for AAA repair
- EVAR anatomical feasibility criteria based on neck morphology
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np

from titan_vasc.calibration.patient import PatientProfile


@dataclass
class ClinicalScoringResult:
    """Clinical decision-support score report.

    Attributes
    ----------
    annual_rupture_risk_pct : float
        Estimated 1-year rupture probability as a percentage (e.g. 7.5%).
    five_year_rupture_risk_pct : float
        Estimated 5-year cumulative rupture probability.
    rescue_score : float
        Composite RESCUE index in [0, 100].
    svs_repair_indicated : bool
        Whether immediate elective repair is indicated by SVS guidelines.
    recommended_surveillance_months : int
        Recommended imaging surveillance interval in months (0 = immediate repair).
    evar_anatomically_suitable : bool
        Whether proximal neck allows standard on-label EVAR (neck length >= 15 mm, angle <= 60 deg).
    contraindications_evar : list[str]
        Specific anatomical contraindications for standard EVAR.
    clinical_summary : str
        Human-readable diagnostic summary.
    """

    annual_rupture_risk_pct: float
    five_year_rupture_risk_pct: float
    rescue_score: float
    svs_repair_indicated: bool
    recommended_surveillance_months: int
    evar_anatomically_suitable: bool
    contraindications_evar: list[str]
    clinical_summary: str


class ClinicalRiskEngine:
    """Computes evidence-based clinical rupture risk metrics for a patient."""

    @classmethod
    def calculate_uksat_annual_risk(cls, patient: PatientProfile) -> float:
        """Calculate adjusted annual rupture risk percentage using UKSAT hazard multipliers.

        Returns
        -------
        float
            Annual rupture risk as percentage (e.g. 5.2 means 5.2% / year).
        """
        d_cm = patient.max_aneurysm_diameter * 100.0

        # Baseline risk for average male non-smoker with normal BP
        if d_cm < 4.0:
            base_risk = 0.5
        elif d_cm < 5.0:
            base_risk = 1.5
        elif d_cm < 5.5:
            base_risk = 3.5
        elif d_cm < 6.0:
            base_risk = 7.0
        elif d_cm < 7.0:
            base_risk = 15.0
        else:
            base_risk = 33.0 + 5.0 * (d_cm - 7.0)

        # Hazard ratio adjustments
        hr = 1.0
        if patient.sex == "female":
            hr *= 3.0  # Females have 3-fold higher rupture risk at equivalent diameter
        if patient.smoking_status == "current":
            hr *= 2.1
        elif patient.smoking_status == "former":
            hr *= 1.3
        if patient.systolic_bp >= 140.0 or patient.diastolic_bp >= 90.0:
            hr *= 1.7
        if patient.morphology_type == "saccular":
            hr *= 2.4  # Saccular aneurysms are inherently less stable
        if patient.family_history_aaa:
            hr *= 1.5
        if patient.neck_angle_deg >= 60.0:
            hr *= 1.3

        adjusted_risk = float(base_risk * hr)
        # Bounded between 0.1% and 95.0%
        return float(np.clip(adjusted_risk, 0.1, 95.0))

    @classmethod
    def compute_rescue_score(cls, patient: PatientProfile, peak_rpi: float = 0.5) -> float:
        """Compute composite RESCUE index (0 - 100).

        Weights:
            - Diameter score: 35 points
            - Peak RPI score: 25 points
            - Blood pressure / hypertension: 15 points
            - Female sex: 10 points
            - Smoking: 10 points
            - Saccular morphology: 5 points
        """
        d_cm = patient.max_aneurysm_diameter * 100.0
        diameter_score = np.clip((d_cm - 3.0) / 4.0 * 35.0, 0.0, 35.0)

        rpi_score = np.clip(peak_rpi * 25.0, 0.0, 25.0)

        bp_score = 0.0
        if patient.systolic_bp >= 160.0 or patient.diastolic_bp >= 100.0:
            bp_score = 15.0
        elif patient.systolic_bp >= 140.0 or patient.diastolic_bp >= 90.0:
            bp_score = 9.0
        elif patient.systolic_bp >= 130.0:
            bp_score = 4.0

        sex_score = 10.0 if patient.sex == "female" else 0.0

        smoke_score = 10.0 if patient.smoking_status == "current" else (4.0 if patient.smoking_status == "former" else 0.0)

        morphology_score = 5.0 if patient.morphology_type == "saccular" else 0.0

        total_rescue = float(diameter_score + rpi_score + bp_score + sex_score + smoke_score + morphology_score)
        return float(np.clip(total_rescue, 0.0, 100.0))

    @classmethod
    def evaluate_svs_guidelines(
        cls,
        patient: PatientProfile,
        peak_rpi: float = 0.5,
    ) -> ClinicalScoringResult:
        """Evaluate Society for Vascular Surgery (SVS) clinical thresholds and repair indication.

        SVS Repair Thresholds:
            - Men: D >= 5.5 cm
            - Women: D >= 5.0 cm
            - Rapid expansion: >= 1.0 cm / year or >= 0.5 cm in 6 months
            - Saccular aneurysms: considered for repair regardless of diameter if symptomatic or > 2.0 cm
            - Peak RPI >= 0.85: biomechanical emergency
        """
        d_cm = patient.max_aneurysm_diameter * 100.0
        annual_risk = cls.calculate_uksat_annual_risk(patient)
        # 5-year risk: 1 - (1 - annual_risk / 100)^5
        five_year_risk = float((1.0 - (1.0 - annual_risk / 100.0) ** 5) * 100.0)

        rescue_idx = cls.compute_rescue_score(patient, peak_rpi=peak_rpi)

        # SVS threshold
        threshold_cm = 5.0 if patient.sex == "female" else 5.5
        repair_indicated = False
        surveillance_months = 12

        if d_cm >= threshold_cm:
            repair_indicated = True
            surveillance_months = 0
        elif patient.morphology_type == "saccular" and d_cm >= 3.5:
            repair_indicated = True
            surveillance_months = 0
        elif peak_rpi >= 0.85:
            repair_indicated = True
            surveillance_months = 0
        elif d_cm >= 5.0:
            surveillance_months = 3
        elif d_cm >= 4.0:
            surveillance_months = 6
        else:
            surveillance_months = 12

        # EVAR anatomical criteria
        contraindications: list[str] = []
        if patient.neck_angle_deg > 60.0:
            contraindications.append(f"Severe infrarenal neck angulation ({patient.neck_angle_deg:.1f} deg > 60 deg)")

        # Proximal neck length requirement: typically >= 15 mm (0.015 m) for standard on-label infrarenal stent-graft
        evar_suitable = len(contraindications) == 0

        summary = (
            f"Patient {patient.patient_id} ({patient.age:.0f}yo {patient.sex}, D_max={d_cm:.1f}cm): "
            f"Annual Rupture Risk = {annual_risk:.1f}%, RESCUE Score = {rescue_idx:.1f}/100. "
            f"SVS Repair Indicated: {repair_indicated}. "
            f"{'Recommended immediate surgical consultation.' if repair_indicated else f'Recommend CT surveillance in {surveillance_months} months.'}"
        )

        return ClinicalScoringResult(
            annual_rupture_risk_pct=annual_risk,
            five_year_rupture_risk_pct=five_year_risk,
            rescue_score=rescue_idx,
            svs_repair_indicated=repair_indicated,
            recommended_surveillance_months=surveillance_months,
            evar_anatomically_suitable=evar_suitable,
            contraindications_evar=contraindications,
            clinical_summary=summary,
        )
