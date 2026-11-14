"""Visualizer module: pure 7-bit ASCII canvas, waveform plotting, and clinical reporting."""

from titan_vasc.visualizer.ascii_canvas import (
    AsciiCanvas,
    render_sparkline,
    render_waveform_plot,
    render_aortic_geometry,
    render_stress_distribution,
)
from titan_vasc.visualizer.report import ClinicalReportGenerator

__all__ = [
    "AsciiCanvas",
    "render_sparkline",
    "render_waveform_plot",
    "render_aortic_geometry",
    "render_stress_distribution",
    "ClinicalReportGenerator",
]
