"""Unit tests for AAA geometric morphology generation and expansion forecasting."""

import numpy as np
import pytest

from titan_vasc.aneurysm.morphology import (
    AneurysmGeometry,
    MorphologyGenerator,
    GrowthForecaster,
)


def test_fusiform_aneurysm_geometry() -> None:
    gen = MorphologyGenerator(total_length=0.25, healthy_radius=0.010, num_nodes=51)
    geom = gen.create_fusiform_aneurysm(max_diameter=0.055, aneurysm_length=0.08)

    assert isinstance(geom, AneurysmGeometry)
    assert len(geom.z) == 51
    assert len(geom.radius) == 51
    assert np.isclose(geom.max_diameter, 0.055, rtol=1e-2)
    assert np.isclose(geom.healthy_diameter, 0.020)
    assert geom.dilatation_ratio > 2.5
    assert geom.lumen_volume > 0.0
    assert geom.wall_volume > 0.0
    assert geom.surface_area > 0.0
    assert len(geom.meridional_curvature) == 51
    assert len(geom.circumferential_curvature) == 51


def test_saccular_aneurysm_geometry() -> None:
    gen = MorphologyGenerator(total_length=0.20, num_nodes=41)
    geom = gen.create_saccular_aneurysm(sac_diameter=0.048, sac_length=0.04)

    assert isinstance(geom, AneurysmGeometry)
    assert geom.max_diameter > 0.045
    assert geom.tortuosity_index >= 0.0


def test_healthy_aorta_taper() -> None:
    gen = MorphologyGenerator(total_length=0.25, healthy_radius=0.010, num_nodes=51)
    geom = gen.create_healthy_aorta()

    assert geom.dilatation_ratio == 1.0
    # Proximal radius should be >= distal radius due to physiological taper
    assert geom.radius[0] >= geom.radius[-1]


def test_growth_forecaster() -> None:
    # 4.5 cm aneurysm expansion rate
    rate_mild = GrowthForecaster.annual_expansion_rate(0.045, is_smoker=False)
    # 6.5 cm aneurysm expansion rate
    rate_large = GrowthForecaster.annual_expansion_rate(0.065, is_smoker=True)

    # Larger aneurysms expand faster
    assert rate_large > rate_mild
    assert 0.001 <= rate_mild <= 0.010

    t_arr, d_arr = GrowthForecaster.forecast_trajectory(
        initial_diameter=0.045, years=3.0, is_smoker=True
    )
    assert len(t_arr) == len(d_arr)
    # Diameter must strictly increase over time
    assert np.all(np.diff(d_arr) > 0.0)
