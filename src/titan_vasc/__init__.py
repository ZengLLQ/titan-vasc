"""TitanVasc: Multiscale Cardiovascular Hemodynamics and Aortic Rupture Biomechanics Engine.

A high-performance computational medical engineering framework coupling:
- 1D Pulsatile cross-sectional averaged Navier-Stokes arterial flow
- Non-Newtonian blood rheology (Carreau-Yasuda, Casson)
- Wave intensity analysis (WIA) and characteristic wave separation
- Wall shear stress disturbance indices (TAWSS, OSI, RRT, ECAP)
- Abdominal aortic aneurysm (AAA) 3D morphology and ECM matrix remodeling
- Intraluminal thrombus (ILT) deposition, mechanical cushioning, and hypoxia
- Anisotropic hyperelastic constitutive modeling (Holzapfel-Gasser-Ogden structural dispersion)
- Finite membrane wall stress and Rupture Potential Index (RPI)
- Lumped parameter Windkessel outflow boundaries (2-element, 3-element, 4-element L-RCR)
- Patient-specific clinical risk calibration (UKSAT, RESCUE score, SVS guidelines)
- Monte Carlo uncertainty quantification and Morris global sensitivity analysis
- 100% pure 7-bit ASCII terminal visualizer and clinical reporting
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "TitanVasc Contributors"

from titan_vasc.hemodynamics.solver import (
    ArterialSegment,
    PulsatileFlowSolver,
    SimulationHistory,
    generate_cardiac_cycle_inflow,
)
from titan_vasc.hemodynamics.rheology import (
    BloodRheologyModel,
    NewtonianRheology,
    CarreauYasudaRheology,
    CassonRheology,
)
from titan_vasc.hemodynamics.wave import (
    TubeLaw,
    WaveSeparator,
    WaveSeparationResult,
)
from titan_vasc.hemodynamics.wss import (
    HemodynamicDisturbanceResult,
    WallShearStressCalculator,
)
from titan_vasc.aneurysm.morphology import (
    AneurysmGeometry,
    MorphologyGenerator,
    GrowthForecaster,
)
from titan_vasc.aneurysm.remodeling import (
    ECMCompositionState,
    VascularRemodelingModel,
)
from titan_vasc.aneurysm.thrombus import (
    ThrombusState,
    IntraluminalThrombusModel,
)
from titan_vasc.mechanics.hyperelastic import (
    BiaxialStressState,
    HyperelasticMaterial,
    NeoHookeanArterialWall,
    MooneyRivlinArterialWall,
    FungExponentialArterialWall,
    HolzapfelGasserOgdenModel,
)
from titan_vasc.mechanics.rupture import (
    RuptureRiskResult,
    RuptureRiskEvaluator,
)
from titan_vasc.mechanics.fsi import (
    DynamicWallState,
    CoupledFSIEngine,
)
from titan_vasc.boundary.windkessel import (
    WindkesselState,
    ThreeElementWindkessel,
    FourElementWindkessel,
)
from titan_vasc.boundary.inflow import (
    CardiacInflowMetrics,
    InflowWaveformFactory,
    FourierHarmonicInflow,
)
from titan_vasc.boundary.junction import (
    BifurcationJunctionState,
    AorticBifurcationJunction,
)
from titan_vasc.calibration.patient import PatientProfile
from titan_vasc.calibration.scoring import (
    ClinicalScoringResult,
    ClinicalRiskEngine,
)
from titan_vasc.calibration.sensitivity import (
    UncertaintyQuantificationResult,
    RuptureSensitivityAnalyzer,
)
from titan_vasc.visualizer.ascii_canvas import (
    AsciiCanvas,
    render_sparkline,
    render_waveform_plot,
    render_aortic_geometry,
    render_stress_distribution,
)
from titan_vasc.visualizer.report import ClinicalReportGenerator

__all__ = [
    "__version__",
    "__author__",
    "ArterialSegment",
    "PulsatileFlowSolver",
    "SimulationHistory",
    "generate_cardiac_cycle_inflow",
    "BloodRheologyModel",
    "NewtonianRheology",
    "CarreauYasudaRheology",
    "CassonRheology",
    "TubeLaw",
    "WaveSeparator",
    "WaveSeparationResult",
    "HemodynamicDisturbanceResult",
    "WallShearStressCalculator",
    "AneurysmGeometry",
    "MorphologyGenerator",
    "GrowthForecaster",
    "ECMCompositionState",
    "VascularRemodelingModel",
    "ThrombusState",
    "IntraluminalThrombusModel",
    "BiaxialStressState",
    "HyperelasticMaterial",
    "NeoHookeanArterialWall",
    "MooneyRivlinArterialWall",
    "FungExponentialArterialWall",
    "HolzapfelGasserOgdenModel",
    "RuptureRiskResult",
    "RuptureRiskEvaluator",
    "DynamicWallState",
    "CoupledFSIEngine",
    "WindkesselState",
    "ThreeElementWindkessel",
    "FourElementWindkessel",
    "CardiacInflowMetrics",
    "InflowWaveformFactory",
    "FourierHarmonicInflow",
    "BifurcationJunctionState",
    "AorticBifurcationJunction",
    "PatientProfile",
    "ClinicalScoringResult",
    "ClinicalRiskEngine",
    "UncertaintyQuantificationResult",
    "RuptureSensitivityAnalyzer",
    "AsciiCanvas",
    "render_sparkline",
    "render_waveform_plot",
    "render_aortic_geometry",
    "render_stress_distribution",
    "ClinicalReportGenerator",
]
