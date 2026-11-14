"""Example 2: Anisotropic Hyperelastic Wall Stress and Rupture Potential Analysis.

Demonstrates:
- Holzapfel-Gasser-Ogden (HGO) structural anisotropic constitutive modeling
- Influence of non-symmetric collagen fiber dispersion on arterial compliance
- Patient-specific Peak Wall Stress (PWS) and Rupture Potential Index (RPI)
- Male vs female rupture threshold disparity under identical geometric dimensions
"""

import numpy as np

from titan_vasc.aneurysm.morphology import MorphologyGenerator
from titan_vasc.mechanics.hyperelastic import HolzapfelGasserOgdenModel
from titan_vasc.mechanics.rupture import RuptureRiskEvaluator
from titan_vasc.calibration.patient import PatientProfile
from titan_vasc.visualizer.ascii_canvas import render_stress_distribution, render_aortic_geometry


def main() -> None:
    print("=" * 75)
    print("  TITANVASC EXAMPLE 2: HGO HYPERELASTIC CONSTITUTIVE & RUPTURE RISK")
    print("=" * 75)

    # 1. Hyperelastic constitutive response
    hgo_healthy = HolzapfelGasserOgdenModel(c=3.2e4, k1=4.8e5, k2=18.0, kappa=0.14, fiber_angle_deg=42.0)
    hgo_aneurysm = HolzapfelGasserOgdenModel(c=4.5e4, k1=1.2e6, k2=24.0, kappa=0.08, fiber_angle_deg=49.0)

    stretches = np.linspace(1.0, 1.25, 20)
    stress_healthy = [hgo_healthy.compute_biaxial_stress(s, 1.05).cauchy_stress_theta for s in stretches]
    stress_aneurysm = [hgo_aneurysm.compute_biaxial_stress(s, 1.05).cauchy_stress_theta for s in stretches]

    print("\n  [1] CONSTITUTIVE HYPERELASTIC STRESS-STRETCH RESPONSE:")
    print(f"    Stretch lambda_theta : 1.05       1.10       1.15       1.20       1.25")
    print(f"    Healthy Wall (kPa)   : {stress_healthy[4]/1e3:<10.1f} {stress_healthy[8]/1e3:<10.1f} {stress_healthy[12]/1e3:<10.1f} {stress_healthy[16]/1e3:<10.1f} {stress_healthy[19]/1e3:<10.1f}")
    print(f"    Diseased Wall (kPa)  : {stress_aneurysm[4]/1e3:<10.1f} {stress_aneurysm[8]/1e3:<10.1f} {stress_aneurysm[12]/1e3:<10.1f} {stress_aneurysm[16]/1e3:<10.1f} {stress_aneurysm[19]/1e3:<10.1f}")
    print("    Note: Diseased aneurysmal tissue exhibits steep fiber engagement and collagen stiffening.")

    # 2. Geometry generation: 5.4 cm fusiform aneurysm
    gen = MorphologyGenerator(total_length=0.22, healthy_radius=0.010, num_nodes=51)
    geom = gen.create_fusiform_aneurysm(max_diameter=0.054, aneurysm_length=0.08, wall_thinning_factor=0.65)

    print("\n" + render_aortic_geometry(geom.z, geom.radius, geom.wall_thickness, width=72, height=13))

    # 3. Rupture risk comparison between Male and Female patients under identical geometry
    evaluator = RuptureRiskEvaluator()
    systolic_p_pa = 145.0 * 133.322 # 145 mmHg systolic BP
    ilt_thickness = np.full_like(geom.radius, 0.006) # 6 mm ILT

    res_male = evaluator.evaluate_rupture_risk(
        z=geom.z,
        radius=geom.radius,
        wall_thickness=geom.wall_thickness,
        systolic_pressure=systolic_p_pa,
        ilt_thickness=ilt_thickness,
        is_female=False,
        age=71.0,
    )
    res_female = evaluator.evaluate_rupture_risk(
        z=geom.z,
        radius=geom.radius,
        wall_thickness=geom.wall_thickness,
        systolic_pressure=systolic_p_pa,
        ilt_thickness=ilt_thickness,
        is_female=True,
        age=71.0,
    )

    print("\n  [2] BIOMECHANICAL RUPTURE EVALUATION (5.4 cm Aneurysm at 145 mmHg):")
    print(f"    Patient Cohort        Peak Wall Stress    Wall Strength     Peak RPI    Risk Tier")
    print(f"    ---------------------------------------------------------------------------------")
    print(f"    Male (71yo, D=5.4cm)   : {res_male.peak_wall_stress/1e3:<7.1f} kPa      {float(np.min(res_male.tensile_wall_strength))/1e3:<7.1f} kPa       {res_male.peak_rpi:<7.3f}    [{res_male.clinical_risk_tier.upper()}]")
    print(f"    Female (71yo, D=5.4cm) : {res_female.peak_wall_stress/1e3:<7.1f} kPa      {float(np.min(res_female.tensile_wall_strength))/1e3:<7.1f} kPa       {res_female.peak_rpi:<7.3f}    [{res_female.clinical_risk_tier.upper()}]")
    print("-" * 75)
    print("  Notice: Female aorta has ~180 kPa lower baseline tensile strength.")
    print("  Consequently, the female patient reaches critical rupture risk (RPI >= 0.85)")
    print("  at 5.4 cm, validating why SVS guidelines recommend repair at 5.0 cm for women.")

    print("\n" + render_stress_distribution(
        geom.z,
        res_female.von_mises_stress,
        res_female.tensile_wall_strength,
        width=72,
    ))
    print("=" * 75)


if __name__ == "__main__":
    main()
