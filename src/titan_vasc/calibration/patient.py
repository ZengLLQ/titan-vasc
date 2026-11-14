"""Patient demographic, hemodynamic, and vascular morphology clinical profiles.

Defines standardized patient representations for personalized hemodynamic and rupture risk analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PatientProfile:
    """Clinical, anatomical, and physiological characteristics of an individual patient.

    Attributes
    ----------
    patient_id : str
        Unique clinical identifier.
    age : float
        Patient age in years (default: 68.0).
    sex : Literal["male", "female"]
        Biological sex.
    systolic_bp : float
        Systolic blood pressure in mmHg (default: 135.0 mmHg).
    diastolic_bp : float
        Diastolic blood pressure in mmHg (default: 85.0 mmHg).
    heart_rate : float
        Resting heart rate in beats per minute (default: 72.0 bpm).
    smoking_status : Literal["current", "former", "never"]
        Cigarette smoking status.
    copd : bool
        Presence of chronic obstructive pulmonary disease.
    family_history_aaa : bool
        Confirmed family history of AAA or aortic dissection in first-degree relatives.
    max_aneurysm_diameter : float
        Maximum transverse outer aneurysm diameter in meters (e.g. 0.052 m = 5.2 cm).
    healthy_aorta_diameter : float
        Normal non-dilated infrarenal aortic diameter in meters (default: 0.020 m = 2.0 cm).
    aneurysm_length : float
        Longitudinal extent of the aneurysm sac in meters (default: 0.075 m = 7.5 cm).
    neck_angle_deg : float
        Infrarenal neck angulation in degrees (default: 20.0 deg).
    ilt_thickness : float
        Maximum thickness of intraluminal thrombus in meters (default: 0.005 m = 5.0 mm).
    morphology_type : Literal["fusiform", "saccular"]
        Morphological classification.
    """

    patient_id: str = "PAT-001"
    age: float = 68.0
    sex: Literal["male", "female"] = "male"
    systolic_bp: float = 135.0
    diastolic_bp: float = 85.0
    heart_rate: float = 72.0
    smoking_status: Literal["current", "former", "never"] = "former"
    copd: bool = False
    family_history_aaa: bool = False
    max_aneurysm_diameter: float = 0.052
    healthy_aorta_diameter: float = 0.020
    aneurysm_length: float = 0.075
    neck_angle_deg: float = 20.0
    ilt_thickness: float = 0.005
    morphology_type: Literal["fusiform", "saccular"] = "fusiform"

    def __post_init__(self) -> None:
        if self.age <= 0.0:
            raise ValueError(f"Patient age must be positive: {self.age}")
        if self.systolic_bp <= self.diastolic_bp:
            raise ValueError(
                f"Systolic BP ({self.systolic_bp}) must exceed diastolic BP ({self.diastolic_bp})"
            )
        if self.max_aneurysm_diameter < self.healthy_aorta_diameter:
            raise ValueError("Max aneurysm diameter must be >= healthy aorta diameter")

    @property
    def pulse_pressure_mmhg(self) -> float:
        """Pulse pressure PP = SBP - DBP in mmHg."""
        return self.systolic_bp - self.diastolic_bp

    @property
    def mean_arterial_pressure_mmhg(self) -> float:
        """Mean arterial pressure MAP = DBP + 1/3 * (SBP - DBP) in mmHg."""
        return self.diastolic_bp + (self.systolic_bp - self.diastolic_bp) / 3.0

    @property
    def systolic_pressure_pa(self) -> float:
        """Systolic pressure in Pascals (1 mmHg = 133.322 Pa)."""
        return self.systolic_bp * 133.322

    @property
    def diastolic_pressure_pa(self) -> float:
        """Diastolic pressure in Pascals."""
        return self.diastolic_bp * 133.322

    @property
    def map_pa(self) -> float:
        """Mean arterial pressure in Pascals."""
        return self.mean_arterial_pressure_mmhg * 133.322

    @property
    def dilatation_ratio(self) -> float:
        """Ratio of maximum aneurysm diameter to healthy reference aorta."""
        return self.max_aneurysm_diameter / self.healthy_aorta_diameter
