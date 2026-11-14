"""Vascular wall biomechanical remodeling, enzyme kinetics, and elastin degradation.

Models biochemical and structural remodeling of the aortic extracellular matrix (ECM):
- Matrix metalloproteinases (MMP-2, MMP-9) activation
- Elastin fragmentation and irreversible enzymatic degradation
- Collagen turnover, strain-induced synthesis, and cross-linking
- Vascular smooth muscle cell (VSMC) phenotypic switching and apoptosis
- Progressive wall thinning and stiffening under cyclic arterial hypertension
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ECMCompositionState:
    """Concentration and structural metrics of the aortic extracellular matrix.

    Attributes
    ----------
    elastin_fraction : float
        Current mass fraction of intact functional elastin (relative to healthy baseline = 1.0).
    collagen_fraction : float
        Current mass fraction of fibrillar collagen (relative to healthy baseline = 1.0).
    vsmc_density_fraction : float
        Current viable vascular smooth muscle cell density fraction.
    mmp2_concentration : float
        Active MMP-2 enzyme concentration in nmol/L.
    mmp9_concentration : float
        Active MMP-9 enzyme concentration in nmol/L.
    wall_thickness : float
        Current remodeled arterial wall thickness in meters.
    effective_tangent_modulus : float
        Current effective incremental tangent elastic modulus E in Pascals.
    """

    elastin_fraction: float
    collagen_fraction: float
    vsmc_density_fraction: float
    mmp2_concentration: float
    mmp9_concentration: float
    wall_thickness: float
    effective_tangent_modulus: float


class VascularRemodelingModel:
    """Coupled biochemical and mechanical differential equations for ECM remodeling.

    Governing equations:
        d(M_e)/dt = - k_e * [MMP_active] * M_e
        d(M_c)/dt = k_syn * max(0, sigma_wall - sigma_homeo) - k_deg_c * [MMP_active] * M_c
        d(rho_vsmc)/dt = - k_apop * (1 + alpha_inflam * [MMP_9]) * rho_vsmc

    Parameters
    ----------
    initial_thickness : float
        Healthy reference wall thickness h_0 in meters (default: 0.0015 m).
    homeostatic_wall_stress : float
        Target homeostatic circumferential tensile stress sigma_homeo in Pa (default: 1.2e5 Pa = 120 kPa).
    elastin_degradation_rate : float
        Rate constant k_e for MMP-mediated elastin lysis [L / (nmol * month)].
    collagen_synthesis_rate : float
        Rate constant k_syn for mechanosensitive collagen deposition [month^-1 / Pa].
    collagen_degradation_rate : float
        Rate constant k_deg_c for collagen turnover [L / (nmol * month)].
    vsmc_apoptosis_rate : float
        Basal VSMC apoptotic rate constant k_apop [month^-1].
    """

    def __init__(
        self,
        initial_thickness: float = 0.0015,
        homeostatic_wall_stress: float = 1.2e5,
        elastin_degradation_rate: float = 0.045,
        collagen_synthesis_rate: float = 8.5e-7,
        collagen_degradation_rate: float = 0.025,
        vsmc_apoptosis_rate: float = 0.015,
    ) -> None:
        self.h_0 = float(initial_thickness)
        self.sigma_homeo = float(homeostatic_wall_stress)
        self.k_e = float(elastin_degradation_rate)
        self.k_syn = float(collagen_synthesis_rate)
        self.k_deg_c = float(collagen_degradation_rate)
        self.k_apop = float(vsmc_apoptosis_rate)

        # Baseline elastic moduli: Elastin is compliant (0.6 MPa), Collagen is stiff (100 MPa)
        self.e_elastin = 0.6e6
        self.e_collagen = 65.0e6

    def compute_mmp_activity(
        self,
        inflammation_score: float = 1.0,
        tawss: float = 1.0,
        intraluminal_thrombus_thickness: float = 0.0,
    ) -> tuple[float, float]:
        """Compute active MMP-2 and MMP-9 concentrations based on hemodynamic and inflammatory stimuli.

        Low TAWSS (<0.4 Pa) and thick thrombus (>3 mm) upregulate MMP-9 via localized adventitial hypoxia.
        """
        # Baseline physiological concentrations in healthy vascular tissue
        base_mmp2 = 1.5   # nmol/L
        base_mmp9 = 0.4   # nmol/L

        # Hemodynamic shear modulation: low WSS induces monocyte adhesion and MMP release
        shear_factor = 1.0 + max(0.0, (0.4 - tawss) / 0.4) * 2.5 if tawss < 0.4 else 1.0

        # Thrombus hypoxia factor: each mm of ILT induces adventitial hypoxia and proteolytic secretion
        ilt_mm = intraluminal_thrombus_thickness * 1000.0
        hypoxia_factor = 1.0 + 0.35 * min(15.0, ilt_mm)

        active_mmp2 = float(base_mmp2 * inflammation_score * (1.0 + 0.3 * (shear_factor - 1.0)))
        active_mmp9 = float(base_mmp9 * inflammation_score * shear_factor * hypoxia_factor)

        return active_mmp2, active_mmp9

    def step(
        self,
        state: ECMCompositionState,
        wall_stress: float,
        dt_months: float,
        tawss: float = 1.0,
        ilt_thickness: float = 0.0,
        inflammation_score: float = 1.0,
    ) -> ECMCompositionState:
        """Advance ECM remodeling forward by dt_months using explicit Runge-Kutta 2nd order.

        Parameters
        ----------
        state : ECMCompositionState
            Current ECM biochemical state.
        wall_stress : float
            Current peak circumferential wall stress in Pa.
        dt_months : float
            Time step in months.
        tawss : float
            Local time-averaged wall shear stress in Pa.
        ilt_thickness : float
            Local intraluminal thrombus thickness in meters.
        inflammation_score : float
            Systemic/local inflammatory multiplier (1.0 = normal, 2.5 = high).

        Returns
        -------
        ECMCompositionState
            Updated remodeled state.
        """
        mmp2, mmp9 = self.compute_mmp_activity(
            inflammation_score=inflammation_score,
            tawss=tawss,
            intraluminal_thrombus_thickness=ilt_thickness,
        )
        total_mmp = mmp2 + 1.8 * mmp9

        def derivatives(
            e_frac: float, c_frac: float, vsmc_frac: float
        ) -> tuple[float, float, float]:
            # Irreversible elastin degradation
            de_dt = -self.k_e * total_mmp * e_frac

            # Collagen synthesis triggered by tensile excess over homeostatic setpoint
            stress_excess = max(0.0, wall_stress - self.sigma_homeo)
            dc_dt = self.k_syn * stress_excess - self.k_deg_c * total_mmp * c_frac

            # VSMC apoptosis enhanced by MMP-9 and mechanical overload
            stress_overload_factor = 1.0 + max(0.0, (wall_stress - 3.0e5) / 1.0e5)
            dvsmc_dt = -self.k_apop * (1.0 + 0.15 * mmp9) * stress_overload_factor * vsmc_frac

            return de_dt, dc_dt, dvsmc_dt

        # RK2 midpoint integration
        k1_e, k1_c, k1_v = derivatives(
            state.elastin_fraction, state.collagen_fraction, state.vsmc_density_fraction
        )

        mid_e = max(0.01, state.elastin_fraction + 0.5 * dt_months * k1_e)
        mid_c = max(0.10, state.collagen_fraction + 0.5 * dt_months * k1_c)
        mid_v = max(0.05, state.vsmc_density_fraction + 0.5 * dt_months * k1_v)

        k2_e, k2_c, k2_v = derivatives(mid_e, mid_c, mid_v)

        new_e = float(np.clip(state.elastin_fraction + dt_months * k2_e, 0.005, 1.2))
        new_c = float(np.clip(state.collagen_fraction + dt_months * k2_c, 0.05, 3.5))
        new_v = float(np.clip(state.vsmc_density_fraction + dt_months * k2_v, 0.01, 1.0))

        # Remodeled wall thickness: proportional to structural protein content
        total_protein = (new_e * 0.35 + new_c * 0.65)
        remodeled_thickness = float(self.h_0 * total_protein)

        # Effective tangent modulus: rule of mixtures between compliant elastin and stiff collagen
        total_mass = new_e + new_c + 1e-12
        e_effective = float((new_e * self.e_elastin + new_c * self.e_collagen) / total_mass)

        return ECMCompositionState(
            elastin_fraction=new_e,
            collagen_fraction=new_c,
            vsmc_density_fraction=new_v,
            mmp2_concentration=mmp2,
            mmp9_concentration=mmp9,
            wall_thickness=remodeled_thickness,
            effective_tangent_modulus=e_effective,
        )

    def simulate_longitudinal_remodeling(
        self,
        months: float = 36.0,
        dt_months: float = 0.5,
        initial_wall_stress: float = 1.8e5,
        tawss: float = 0.3,
        ilt_thickness: float = 0.004,
        inflammation_score: float = 1.5,
    ) -> tuple[np.ndarray, list[ECMCompositionState]]:
        """Simulate longitudinal ECM degradation over a multi-year clinical progression.

        Returns
        -------
        time_arr : np.ndarray
            Time array in months.
        states : list[ECMCompositionState]
            State trajectory over time.
        """
        time_arr = np.arange(0.0, months + dt_months, dt_months)
        current_state = ECMCompositionState(
            elastin_fraction=1.0,
            collagen_fraction=1.0,
            vsmc_density_fraction=1.0,
            mmp2_concentration=1.5,
            mmp9_concentration=0.4,
            wall_thickness=self.h_0,
            effective_tangent_modulus=(0.35 * self.e_elastin + 0.65 * self.e_collagen),
        )

        states: list[ECMCompositionState] = [current_state]

        for _ in range(1, len(time_arr)):
            # As wall thins, wall stress increases proportionally: sigma ~ 1 / h
            current_stress = initial_wall_stress * (self.h_0 / max(current_state.wall_thickness, 1e-5))
            current_state = self.step(
                state=current_state,
                wall_stress=current_stress,
                dt_months=dt_months,
                tawss=tawss,
                ilt_thickness=ilt_thickness,
                inflammation_score=inflammation_score,
            )
            states.append(current_state)

        return time_arr, states
