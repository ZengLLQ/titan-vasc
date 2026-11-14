"""Unit tests for vascular extracellular matrix remodeling and degradation kinetics."""

import numpy as np
import pytest

from titan_vasc.aneurysm.remodeling import (
    ECMCompositionState,
    VascularRemodelingModel,
)


def test_mmp_activity_calculation() -> None:
    model = VascularRemodelingModel()
    mmp2_norm, mmp9_norm = model.compute_mmp_activity(tawss=1.2, intraluminal_thrombus_thickness=0.0)
    mmp2_hypox, mmp9_hypox = model.compute_mmp_activity(tawss=0.2, intraluminal_thrombus_thickness=0.008)

    # Low WSS and thick thrombus upregulate MMPs
    assert mmp2_hypox > mmp2_norm
    assert mmp9_hypox > mmp9_norm


def test_remodeling_step_elastin_degradation() -> None:
    model = VascularRemodelingModel()
    base_modulus = 0.5 * (model.e_elastin + model.e_collagen)
    init_state = ECMCompositionState(
        elastin_fraction=1.0,
        collagen_fraction=1.0,
        vsmc_density_fraction=1.0,
        mmp2_concentration=1.5,
        mmp9_concentration=0.4,
        wall_thickness=0.0015,
        effective_tangent_modulus=base_modulus,
    )

    next_state = model.step(
        state=init_state,
        wall_stress=2.5e5,
        dt_months=1.0,
        tawss=0.3,
        ilt_thickness=0.004,
    )

    # Elastin degrades irreversibly
    assert next_state.elastin_fraction < init_state.elastin_fraction
    # Stiffening occurs due to collagen fraction increasing relative to elastin
    assert next_state.effective_tangent_modulus > init_state.effective_tangent_modulus


def test_longitudinal_simulation() -> None:
    model = VascularRemodelingModel()
    t_arr, states = model.simulate_longitudinal_remodeling(
        months=12.0,
        dt_months=1.0,
        initial_wall_stress=2.0e5,
    )

    assert len(t_arr) == 13
    assert len(states) == 13
    assert states[-1].elastin_fraction < states[0].elastin_fraction
