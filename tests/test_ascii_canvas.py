"""Unit tests for 7-bit ASCII canvas, sparklines, and terminal visualizations."""

import numpy as np
import pytest

from titan_vasc.visualizer.ascii_canvas import (
    AsciiCanvas,
    render_sparkline,
    render_waveform_plot,
    render_aortic_geometry,
    render_stress_distribution,
)


def _assert_pure_7bit_ascii(text: str) -> None:
    """Verify that every character in the string is strictly in 7-bit ASCII (0-127)."""
    for idx, ch in enumerate(text):
        assert ord(ch) < 128, f"Non-ASCII character '{ch}' (ord={ord(ch)}) found at index {idx}"


def test_ascii_canvas_primitives() -> None:
    canvas = AsciiCanvas(width=30, height=10)
    canvas.draw_box(0, 0, 30, 10)
    canvas.draw_line(2, 2, 25, 8)
    canvas.draw_text(5, 1, "TEST TITLE")
    output = canvas.render()

    _assert_pure_7bit_ascii(output)
    assert len(output.splitlines()) == 10
    assert "TEST TITLE" in output


def test_render_sparkline() -> None:
    vals = np.array([1.0, 3.0, 5.0, 10.0, 7.0, 2.0])
    spark = render_sparkline(vals, width=20)
    _assert_pure_7bit_ascii(spark)
    assert len(spark) > 0


def test_render_waveform_plot() -> None:
    time = np.linspace(0.0, 1.0, 50)
    signal = np.sin(2 * np.pi * time)
    plot_str = render_waveform_plot(time, signal, title="SINE WAVE", width=50, height=12)

    _assert_pure_7bit_ascii(plot_str)
    assert "SINE WAVE" in plot_str
    assert "t=0.00s" in plot_str


def test_render_aortic_geometry() -> None:
    z = np.linspace(0.0, 0.20, 31)
    radius = 0.010 + 0.015 * np.sin(np.pi * z / 0.20)
    wall_thickness = np.full(31, 0.0015)

    geom_str = render_aortic_geometry(z, radius, wall_thickness, width=60, height=15)
    _assert_pure_7bit_ascii(geom_str)
    assert "INFRARENAL ABDOMINAL AORTA" in geom_str


def test_render_stress_distribution() -> None:
    z = np.linspace(0.0, 0.20, 25)
    stress = np.full(25, 3.5e5)
    strength = np.full(25, 8.0e5)

    dist_str = render_stress_distribution(z, stress, strength, width=65)
    _assert_pure_7bit_ascii(dist_str)
    assert "SPATIAL WALL STRESS" in dist_str
