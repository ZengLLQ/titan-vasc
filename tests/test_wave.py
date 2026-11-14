"""Unit tests for arterial tube law and Wave Intensity Analysis (WIA)."""

import numpy as np
import pytest

from titan_vasc.hemodynamics.wave import (
    TubeLaw,
    WaveSeparator,
    WaveSeparationResult,
)


def test_tube_law_pwv_and_pressure() -> None:
    tl = TubeLaw(
        reference_radius=0.010,
        wall_thickness=0.0015,
        young_modulus=8.0e5,
        blood_density=1060.0,
        reference_pressure=10666.0,
    )
    a_0 = tl.reference_area
    assert np.isclose(a_0, np.pi * 0.010**2)

    # Pressure at reference area should equal reference pressure
    p_ref = tl.pressure_from_area(a_0)
    assert np.isclose(p_ref, 10666.0)

    # Area from pressure inversion
    a_calc = tl.area_from_pressure(10666.0)
    assert np.isclose(a_calc, a_0, rtol=1e-5)

    # Higher pressure expands area
    p_high = 16000.0
    a_high = tl.area_from_pressure(p_high)
    assert a_high > a_0

    # Pulse wave velocity c = sqrt((beta / (2 * rho)) * A^(1/4))
    c_0 = tl.pulse_wave_velocity(a_0)
    # Typical human aortic PWV is in 5 - 12 m/s range
    assert 4.0 <= c_0 <= 15.0


def test_wave_separator_decomposition() -> None:
    tl = TubeLaw(reference_radius=0.010)
    separator = WaveSeparator(tube_law=tl)

    time = np.linspace(0.0, 0.8, 100)
    # Synthesize forward traveling pulse
    p_pulse = 10666.0 + 4000.0 * np.sin(np.pi * time / 0.3) * (time < 0.3)
    c_0 = tl.pulse_wave_velocity(tl.reference_area)
    q_pulse = (p_pulse - 10666.0) * tl.reference_area / (1060.0 * c_0)

    res = separator.separate(time=time, pressure=p_pulse, flow_rate=q_pulse)

    assert isinstance(res, WaveSeparationResult)
    assert len(res.p_forward) == 100
    assert len(res.p_backward) == 100
    assert len(res.wave_intensity) == 100
    # Reflection index should be reasonably bounded [0, 1]
    assert 0.0 <= res.reflection_index <= 1.0
