"""Arterial branching junctions, bifurcation hemodynamics, and wave reflection.

Models the distal aortic bifurcation into left and right common iliac arteries (CIA):
- Wave admittance matching Y_i = A_i / (rho * c_i)
- Reflection coefficient Gamma and transmission coefficient T
- Womersley / Murray area ratio eta = (A_1 + A_2) / A_0
- Conservation of mass and dynamic pressure loss across branching nodes
- Pathological reflection amplification in aneurysmal disease
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class BifurcationJunctionState:
    """Quantitative wave transmission and pressure-flow state at an arterial bifurcation.

    Attributes
    ----------
    area_ratio : float
        Combined daughter area to parent area ratio eta = (A_1 + A_2) / A_0.
    admittance_parent : float
        Acoustic wave admittance of parent aorta Y_0 [m^4 * s / kg].
    admittance_daughters : float
        Summed admittance of daughter iliac arteries Y_1 + Y_2.
    reflection_coefficient : float
        Pressure wave reflection coefficient Gamma in [-1, +1].
    transmission_coefficient : float
        Pressure wave transmission coefficient T = 1 + Gamma.
    minor_head_loss : float
        Dynamic pressure loss Delta_P_loss [Pa] across the bifurcation flow divider.
    """

    area_ratio: float
    admittance_parent: float
    admittance_daughters: float
    reflection_coefficient: float
    transmission_coefficient: float
    minor_head_loss: float


class AorticBifurcationJunction:
    """Solves mass conservation and characteristic wave matching across the aortic bifurcation.

    Parameters
    ----------
    blood_density : float
        Fluid density rho in kg/m^3 (default: 1060.0 kg/m^3).
    loss_coefficient : float
        Minor dynamic loss coefficient K_loss (default: 0.15).
    """

    def __init__(self, blood_density: float = 1060.0, loss_coefficient: float = 0.15) -> None:
        self.rho = float(blood_density)
        self.k_loss = float(loss_coefficient)

    def evaluate_wave_reflection(
        self,
        area_parent: float,
        pwv_parent: float,
        area_left_iliac: float,
        pwv_left_iliac: float,
        area_right_iliac: float,
        pwv_right_iliac: float,
        flow_parent: float = 0.0,
    ) -> BifurcationJunctionState:
        """Compute characteristic reflection and transmission coefficients.

        Formulation:
            Y_i = A_i / (rho * c_i)
            Gamma = (Y_0 - (Y_1 + Y_2)) / (Y_0 + (Y_1 + Y_2))
            T = 2 * Y_0 / (Y_0 + (Y_1 + Y_2)) = 1 + Gamma

        Parameters
        ----------
        area_parent : float
            Distal aortic cross-sectional area A_0 [m^2].
        pwv_parent : float
            Aortic pulse wave velocity c_0 [m/s].
        area_left_iliac : float
            Left common iliac area A_1 [m^2].
        pwv_left_iliac : float
            Left common iliac PWV c_1 [m/s].
        area_right_iliac : float
            Right common iliac area A_2 [m^2].
        pwv_right_iliac : float
            Right common iliac PWV c_2 [m/s].
        flow_parent : float
            Instantaneous flow rate Q_0 [m^3/s].

        Returns
        -------
        BifurcationJunctionState
            Wave reflection and junction characteristics.
        """
        y_0 = area_parent / (self.rho * pwv_parent)
        y_1 = area_left_iliac / (self.rho * pwv_left_iliac)
        y_2 = area_right_iliac / (self.rho * pwv_right_iliac)

        y_daughters = y_1 + y_2

        denom = y_0 + y_daughters
        if denom < 1e-15:
            gamma = 0.0
            t_trans = 1.0
        else:
            gamma = (y_0 - y_daughters) / denom
            t_trans = (2.0 * y_0) / denom

        area_ratio = (area_left_iliac + area_right_iliac) / max(1e-7, area_parent)

        # Dynamic head loss: Delta_P = 0.5 * K_loss * rho * u_0^2
        u_parent = flow_parent / max(1e-7, area_parent)
        head_loss = 0.5 * self.k_loss * self.rho * (u_parent**2)

        return BifurcationJunctionState(
            area_ratio=float(area_ratio),
            admittance_parent=float(y_0),
            admittance_daughters=float(y_daughters),
            reflection_coefficient=float(gamma),
            transmission_coefficient=float(t_trans),
            minor_head_loss=float(head_loss),
        )

    def distribute_flows(
        self,
        q_parent: float,
        admittance_left: float,
        admittance_right: float,
    ) -> tuple[float, float]:
        """Divide parent inflow between daughter iliac vessels based on admittance fractions.

        Returns (q_left, q_right).
        """
        total_admittance = admittance_left + admittance_right
        if total_admittance < 1e-15:
            return 0.5 * q_parent, 0.5 * q_parent

        frac_left = admittance_left / total_admittance
        frac_right = admittance_right / total_admittance

        return q_parent * frac_left, q_parent * frac_right
