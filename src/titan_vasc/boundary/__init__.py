"""Boundary module: Windkessel terminal impedance, cardiac inflow, and bifurcation junctions."""

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

__all__ = [
    "WindkesselState",
    "ThreeElementWindkessel",
    "FourElementWindkessel",
    "CardiacInflowMetrics",
    "InflowWaveformFactory",
    "FourierHarmonicInflow",
    "BifurcationJunctionState",
    "AorticBifurcationJunction",
]
