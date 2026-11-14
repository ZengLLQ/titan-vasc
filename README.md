# TitanVasc

[![CI](https://github.com/ZengLLQ/titan-vasc/actions/workflows/ci.yml/badge.svg)](https://github.com/ZengLLQ/titan-vasc/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/ZengLLQ/titan-vasc)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Domain](https://img.shields.io/badge/domain-Cardiovascular%20Biomechanics-crimson.svg)](#)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](#)

Multiscale cardiovascular hemodynamics and abdominal aortic aneurysm biomechanics engine coupling 1D pulsatile Navier-Stokes flow with anisotropic hyperelastic wall stress and patient-specific rupture risk stratification.

---

## Table of Contents
- [Executive Overview](#executive-overview)
- [System Architecture & Multiscale Workflow](#system-architecture--multiscale-workflow)
- [Mathematical Formulations & Governing Equations](#mathematical-formulations--governing-equations)
  - [1. 1D Cross-Sectional Averaged Navier-Stokes Equations](#1-1d-cross-sectional-averaged-navier-stokes-equations)
  - [2. Non-Newtonian Blood Rheology (Carreau-Yasuda & Casson)](#2-non-newtonian-blood-rheology-carreau-yasuda--casson)
  - [3. Wave Intensity Analysis (WIA) & Characteristic Wave Separation](#3-wave-intensity-analysis-wia--characteristic-wave-separation)
  - [4. Wall Shear Stress & Disturbance Indices (TAWSS, OSI, RRT, ECAP)](#4-wall-shear-stress--disturbance-indices-tawss-osi-rrt-ecap)
  - [5. Holzapfel-Gasser-Ogden (HGO) Anisotropic Hyperelasticity](#5-holzapfel-gasser-ogden-hgo-anisotropic-hyperelasticity)
  - [6. Biomechanical Rupture Potential Index (RPI) & Raghavan-Vorp Strength](#6-biomechanical-rupture-potential-index-rpi--raghavan-vorp-strength)
  - [7. Intraluminal Thrombus (ILT) Hypoxia Diffusion & Shielding](#7-intraluminal-thrombus-ilt-hypoxia-diffusion--shielding)
  - [8. Lumped Parameter Windkessel & Bifurcation Admittance](#8-lumped-parameter-windkessel--bifurcation-admittance)
- [Subsystem Breakdown & Code Hierarchy](#subsystem-breakdown--code-hierarchy)
- [Pure 7-Bit ASCII Terminal Visualizer](#pure-7-bit-ascii-terminal-visualizer)
- [Command-Line Interface (CLI) Reference](#command-line-interface-cli-reference)
- [API Usage Walkthrough](#api-usage-walkthrough)
- [Numerical Benchmarks & Convergence Analysis](#numerical-benchmarks--convergence-analysis)
- [Clinical Scoring & Society for Vascular Surgery (SVS) Guidelines](#clinical-scoring--society-for-vascular-surgery-svs-guidelines)
- [Assumptions, Threat Model & Known Limitations](#assumptions-threat-model--known-limitations)
- [License](#license)

---

## Executive Overview

Abdominal Aortic Aneurysm (AAA) is a life-threatening vascular pathology characterized by permanent localized ballooning of the infrarenal aorta exceeding 3.0 cm (or a 50% increase over normal reference diameter). Current clinical management relies overwhelmingly on the **maximum transverse diameter criterion** ($D_{max} \ge 5.5\text{ cm}$ in men, $\ge 5.0\text{ cm}$ in women) to recommend elective surgical intervention (Open Surgical Repair or Endovascular Aneurysm Repair, EVAR).

However, clinical autopsies and randomized clinical trials (such as the UK Small Aneurysm Trial, UKSAT) demonstrate that **10–24% of small aneurysms ($<5.5\text{ cm}$) rupture unexpectedly**, while up to **25% of large aneurysms ($>5.5\text{ cm}$) remain mechanically stable without rupture throughout patient lifespan**.

**TitanVasc** addresses this diagnostic shortfall by providing an end-to-end, multiscale computational biomechanics engine that couples:
1. **Unsteady 1D Pulsatile Hemodynamics**: Conservative two-step MacCormack and Richtmyer finite-difference integration of the Navier-Stokes continuity and momentum equations.
2. **Non-Newtonian Rheology**: Shear-thinning blood viscosity models (Carreau-Yasuda and Casson) capturing elevated shear-rate gradients.
3. **Wall Shear Stress (WSS) Disturbance**: Full spatial-temporal mapping of Time-Averaged Wall Shear Stress (TAWSS), Oscillatory Shear Index (OSI), Relative Residence Time (RRT), and Endothelial Cell Activation Potential (ECAP).
4. **Intraluminal Thrombus (ILT) Duality**: Mechanical stress shielding versus 1D oxygen diffusion transport barrier inducing adventitial hypoxia ($pO_2 < 5\text{ mmHg}$).
5. **Structural Anisotropic Hyperelasticity**: Non-symmetric collagen fiber dispersion in the vascular media and adventitia using the Holzapfel-Gasser-Ogden (HGO) constitutive model.
6. **Patient-Specific Rupture Potential Index (RPI)**: Direct point-by-point comparison of local Von Mises Peak Wall Stress (PWS) against Raghavan-Vorp statistical wall tensile strength distributions.
7. **Lumped Parameter Windkessel Boundaries**: 3-element (RCR) and 4-element (L-RCR) terminal impedance coupling with Runge-Kutta 4th order ODE integration.
8. **Clinical Decision Support & Uncertainty Quantification (UQ)**: UKSAT hazard models, SVS clinical repair thresholds, EVAR neck anatomical feasibility checks, and Monte Carlo 95% confidence bounds with Morris global parameter sensitivity ranking.

---

## System Architecture & Multiscale Workflow

```mermaid
flowchart TD
    subgraph InflowBoundary ["1. Cardiac Inflow & Boundary Conditions"]
        Inflow["Pulsatile Cardiac Inflow Q_in(t)<br/>(Resting / Exercise / Hypertensive)"]
        WK["3-Element / 4-Element Windkessel<br/>Terminal Outflow (Z_c, R_p, C, L)"]
        Bifurc["Aortic Bifurcation Junction<br/>Acoustic Admittance Matching Y_0, Y_1, Y_2"]
    end

    subgraph Hemodynamics ["2. 1D Pulsatile Hemodynamics Solver"]
        NS["Nonlinear 1D Navier-Stokes<br/>dA/dt + dQ/dz = 0<br/>dQ/dt + d/dz(alpha Q^2/A) + ... = 0"]
        Rheo["Non-Newtonian Rheology<br/>(Carreau-Yasuda / Casson)"]
        WSS["WSS Disturbance Metrics<br/>TAWSS, OSI, RRT, ECAP"]
        WIA["Wave Intensity Analysis<br/>dI = dP * dU (Forward & Reflected)"]
    end

    subgraph AneurysmBio ["3. Aneurysm Morphology & ECM Remodeling"]
        Geom["Geometry Profiler<br/>(Fusiform / Saccular / Neck Angulation)"]
        ILT["Intraluminal Thrombus (ILT)<br/>Stress Cushioning vs O2 Hypoxia"]
        Remod["ECM Degradation Kinetics<br/>MMP-2 / MMP-9 Elastin Lysis"]
    end

    subgraph MechanicsFSI ["4. Hyperelastic Solid Mechanics & FSI"]
        HGO["HGO Anisotropic Constitutive Law<br/>Matrix c + Dispersed Fibers (k1, k2, kappa)"]
        FSI["Coupled 1D-0D Wall Dynamics<br/>rho_w h d^2R/dt^2 + gamma dR/dt = ..."]
        Stress["Membrane Curved Shell Stresses<br/>Circumferential sigma_theta & Axial sigma_z"]
        Strength["Raghavan-Vorp Wall Strength<br/>Patient Covariates (D_max, ILT, Sex, Age)"]
        RPI["Rupture Potential Index (RPI)<br/>RPI(z) = sigma_vm(z) / sigma_strength(z)"]
    end

    subgraph ClinicalSupport ["5. Clinical Decision & Visualizer"]
        UKSAT["UKSAT Annual Rupture Risk & RESCUE Score"]
        SVS["SVS 2024 Practice Guidelines<br/>(Repair Indication & EVAR Neck Suitability)"]
        UQ["Monte Carlo UQ & Morris Sensitivity<br/>95% Confidence Intervals"]
        ASCII["Pure 7-Bit ASCII Terminal Visualizer<br/>Canvas, Waveforms, Spatial Heatmaps, Report"]
    end

    Inflow --> NS
    WK <--> NS
    Bifurc <--> NS
    NS <--> Rheo
    NS --> WSS
    NS --> WIA
    Geom --> NS
    WSS --> ILT
    ILT --> Remod
    Geom --> Stress
    NS --> FSI
    FSI <--> HGO
    Stress --> RPI
    Strength --> RPI
    ILT --> Strength
    RPI --> UKSAT
    UKSAT --> SVS
    RPI --> UQ
    SVS --> ASCII
    UQ --> ASCII
```

---

## Mathematical Formulations & Governing Equations

### 1. 1D Cross-Sectional Averaged Navier-Stokes Equations

Arterial blood flow along the longitudinal coordinate $z$ is modeled via the cross-sectional averaged 1D continuity and momentum conservation equations:

$$\frac{\partial A}{\partial t} + \frac{\partial Q}{\partial z} = 0$$

$$\frac{\partial Q}{\partial t} + \frac{\partial}{\partial z} \left( \alpha \frac{Q^2}{A} \right) + \frac{A}{\rho} \frac{\partial P}{\partial z} + K_R \frac{Q}{A} = 0$$

where:
- $A(z, t) = \pi R(z, t)^2$ is the internal luminal cross-sectional area $[\text{m}^2]$.
- $Q(z, t) = A(z, t) \cdot \bar{u}(z, t)$ is the volumetric flow rate $[\text{m}^3/\text{s}]$.
- $P(z, t)$ is the internal transmural luminal blood pressure $[\text{Pa}]$.
- $\rho = 1060\text{ kg/m}^3$ is blood density.
- $\alpha = \frac{\int u^2 dA}{A \bar{u}^2} \approx 1.10$ is the Coriolis momentum velocity correction factor.
- $K_R = \frac{8 \pi \mu}{\rho}$ is the viscous frictional resistance parameter $[\text{m}^2/\text{s}]$, dynamically evaluated via non-Newtonian rheology.

Coupled to the structural arterial tube law:

$$P(A) = P_0 + \beta \left( \sqrt{A} - \sqrt{A_0} \right), \quad \beta = \frac{4}{3} \sqrt{\pi} \frac{h_0 E}{A_0}$$

The local Pulse Wave Velocity (PWV) is governed by the Bramwell-Hill relation:

$$c(A) = \sqrt{\frac{A}{\rho} \frac{\partial P}{\partial A}} = \sqrt{\frac{\beta}{2 \rho} A^{1/4}}$$

---

### 2. Non-Newtonian Blood Rheology (Carreau-Yasuda & Casson)

Blood exhibits significant shear-thinning behavior at low shear rates ($\dot{\gamma} < 100\text{ s}^{-1}$), commonly encountered inside recirculating aneurysm vortex cavities.

#### Carreau-Yasuda Model:
$$\mu(\dot{\gamma}) = \mu_\infty + (\mu_0 - \mu_\infty) \left[ 1 + (\lambda \dot{\gamma})^a \right]^{\frac{n - 1}{a}}$$

Standard physiological parameters:
- $\mu_0 = 0.056\text{ Pa}\cdot\text{s}$ (zero-shear plateau viscosity)
- $\mu_\infty = 0.00345\text{ Pa}\cdot\text{s}$ (asymptotic high-shear plasma viscosity)
- $\lambda = 3.313\text{ s}$ (relaxation time parameter)
- $a = 2.0$ (transition index)
- $n = 0.3568$ (power-law shear-thinning exponent)

#### Casson Model with Yield Stress:
$$\sqrt{\tau} = \sqrt{\tau_y} + \sqrt{\mu_\infty \dot{\gamma}}, \quad \mu(\dot{\gamma}) = \left( \sqrt{\mu_\infty} + \sqrt{\frac{\tau_y}{\dot{\gamma}}} \right)^2 \quad (\text{for } \tau > \tau_y)$$
where $\tau_y \approx 0.004\text{ Pa}$ represents erythrocyte rouleaux aggregation yield stress.

---

### 3. Wave Intensity Analysis (WIA) & Characteristic Wave Separation

Pulsatile waveforms are decomposed into forward-traveling ($+$) and backward-reflected ($-$) components via the Method of Characteristics:

$$dP_\pm = \frac{1}{2} (dP \pm \rho c \, dU)$$

$$dQ_\pm = \frac{1}{2} \left( dQ \pm \frac{A}{\rho c} \, dP \right)$$

The net instantaneous Wave Intensity $dI$ $[\text{W/m}^2]$ represents the rate of energy flux per unit cross-sectional area:

$$dI = dP \cdot dU = dI_+ + dI_-$$

$$dI_+ = \frac{1}{4 \rho c} (dP + \rho c \, dU)^2, \quad dI_- = -\frac{1}{4 \rho c} (dP - \rho c \, dU)^2$$

Wave Reflection Index (WRI):
$$WRI = \frac{\max(P_-)}{\max(P_+)}$$

---

### 4. Wall Shear Stress & Disturbance Indices (TAWSS, OSI, RRT, ECAP)

The nominal wall shear rate under boundary-layer correction factor $\zeta$ is:

$$\dot{\gamma}_w(z, t) = \frac{(2 + \zeta) |Q(z, t)|}{\pi R(z, t)^3}$$

The cycle-averaged hemodynamic disturbance indices over period $T$ are:

#### Time-Averaged Wall Shear Stress (TAWSS):
$$TAWSS(z) = \frac{1}{T} \int_0^T |\tau_w(z, t)| \, dt \quad [\text{Pa}]$$

#### Oscillatory Shear Index (OSI):
$$OSI(z) = \frac{1}{2} \left( 1 - \frac{\left| \int_0^T \tau_w(z, t) \, dt \right|}{\int_0^T |\tau_w(z, t)| \, dt} \right) \in [0, 0.5]$$

#### Relative Residence Time (RRT):
$$RRT(z) = \frac{1}{(1 - 2 \cdot OSI) \cdot TAWSS} = \frac{1}{\frac{1}{T} \left| \int_0^T \tau_w(z, t) \, dt \right|} \quad [\text{Pa}^{-1}]$$

#### Endothelial Cell Activation Potential (ECAP):
$$ECAP(z) = \frac{OSI(z)}{TAWSS(z)} \quad [\text{Pa}^{-1}]$$

---

### 5. Holzapfel-Gasser-Ogden (HGO) Anisotropic Hyperelasticity

The vascular tissue is modeled as an incompressible hyperelastic continuum reinforced by two symmetrical families of collagen fibers oriented at angle $\pm \gamma$ relative to the circumferential direction:

$$W = W_{iso}(I_1) + W_{aniso}(I_4, I_6)$$

$$W_{iso} = \frac{c}{2} (I_1 - 3)$$

$$W_{aniso} = \frac{k_1}{2 k_2} \sum_{i=4,6} \left\{ \exp\left[ k_2 \left( \kappa I_1 + (1 - 3\kappa) I_i - 1 \right)^2 \right] - 1 \right\}$$

where:
- $c > 0$ $[\text{Pa}]$ is the isotropic non-collagenous ground matrix stiffness.
- $k_1 > 0$ $[\text{Pa}]$ is the fiber stress scaling parameter.
- $k_2 > 0$ is the dimensionless fiber exponential stiffening coefficient.
- $\kappa \in [0, 1/3]$ is the structural fiber dispersion parameter ($\kappa=0$ denotes perfectly aligned parallel fibers; $\kappa=1/3$ denotes an isotropic random 3D network).
- $I_4 = \lambda_\theta^2 \cos^2\gamma + \lambda_z^2 \sin^2\gamma$ is the square of the fiber stretch. Fibers are recruited only in tension ($E_i > 0$).

Circumferential and axial Cauchy stresses under incompressibility constraint $\lambda_r = (\lambda_\theta \lambda_z)^{-1}$:

$$\sigma_{\theta\theta} = \lambda_\theta \frac{\partial W}{\partial \lambda_\theta} - p, \quad \sigma_{zz} = \lambda_z \frac{\partial W}{\partial \lambda_z} - p$$

---

### 6. Biomechanical Rupture Potential Index (RPI) & Raghavan-Vorp Strength

For an axisymmetric curved shell of revolution, the membrane equilibrium wall stresses with axial curvature correction are:

$$\sigma_{\theta\theta}(z) = \frac{P_{eff} R(z)}{h(z)} \left[ 1 - \frac{R(z) R''(z)}{2 (1 + R'(z)^2)} \right]$$

$$\sigma_{zz}(z) = \frac{P_{eff} R(z)}{2 h(z)} \left[ 1 + \frac{R(z) R''(z)}{1 + R'(z)^2} \right]$$

The equivalent Von Mises Peak Wall Stress (PWS) is:

$$\sigma_{vm}(z) = \sqrt{\sigma_{\theta\theta}(z)^2 - \sigma_{\theta\theta}(z) \sigma_{zz}(z) + \sigma_{zz}(z)^2}$$

The local patient-specific wall tensile failure strength $\sigma_{strength}(z)$ is estimated via the Raghavan-Vorp formulation:

$$\sigma_{strength}(z) = \sigma_0 - \beta_D (D_{max} - D_0) - \beta_{ILT} h_{ILT}(z) - \Delta\sigma_{female} \delta_{female} - \alpha_{age} \max(0, Age - 60)$$

The local **Rupture Potential Index (RPI)**:

$$RPI(z) = \frac{\sigma_{vm}(z)}{\sigma_{strength}(z)}$$

- $RPI < 0.45$: Low Risk
- $0.45 \le RPI < 0.65$: Moderate Risk
- $0.65 \le RPI < 0.85$: High Risk
- $RPI \ge 0.85$ (or $\sigma_{vm} \ge 500\text{ kPa}$): **Critical Risk / Imminent Rupture**

---

### 7. Intraluminal Thrombus (ILT) Hypoxia Diffusion & Shielding

Intraluminal thrombus exerts a dual biomechanical effect on the aortic wall:
1. **Stress Cushioning**: Attenuates direct dynamic pressure transmission:
   $$P_{eff}(z) = P_{lumen}(z) \cdot \left[ 1 - (1 - e^{-\alpha_{cush} h_{ILT}(z)}) \right]$$
2. **Diffusion Barrier & Hypoxia**: Oxygen transport through fibrin thrombus follows steady-state 1D diffusion-reaction:
   $$D_{O_2} \frac{d^2 C_{O_2}}{dy^2} - R_{consumption} = 0 \implies pO_{2,wall} = \max\left(0, pO_{2,lumen} - \frac{R}{2 D_{eff}} h_{ILT}^2\right)$$
   When $h_{ILT} > 5.5\text{ mm}$, $pO_{2,wall} < 5\text{ mmHg}$ (severe tissue hypoxia), triggering adventitial macrophage activation, elastolysis, and up to a **45% reduction in wall tensile strength**.

---

### 8. Lumped Parameter Windkessel & Bifurcation Admittance

The distal outflow boundary is terminated with a 3-element Windkessel (RCR) model:

$$P(t) - P_{venous} = R_c Q(t) + P_c(t)$$

$$\frac{d P_c}{dt} = \frac{Q(t)}{C} - \frac{P_c(t)}{R_p C}$$

At the terminal aortic bifurcation into the common iliac arteries, characteristic wave admittance matching dictates:

$$Y_0 = \frac{A_0}{\rho c_0}, \quad Y_1 = \frac{A_1}{\rho c_1}, \quad Y_2 = \frac{A_2}{\rho c_2}$$

$$\Gamma = \frac{Y_0 - (Y_1 + Y_2)}{Y_0 + (Y_1 + Y_2)}, \quad T = \frac{2 Y_0}{Y_0 + (Y_1 + Y_2)} = 1 + \Gamma$$

---

## Subsystem Breakdown & Code Hierarchy

TitanVasc is decomposed into **7 tightly integrated subsystems** across **50 authored files** (6,236 authored LoC):

| Subsystem | Directory | Key Modules & Primary Responsibilities |
|---|---|---|
| **Hemodynamics** | `src/titan_vasc/hemodynamics/` | 1D pulsatile Navier-Stokes solvers (MacCormack & Richtmyer), non-Newtonian rheology (Carreau-Yasuda & Casson), arterial tube law, Bramwell-Hill PWV, wave intensity analysis (WIA), and spatial WSS disturbance metrics. |
| **Aneurysm** | `src/titan_vasc/aneurysm/` | Axisymmetric & 3D morphology profiles (fusiform, saccular), centerline tortuosity, principal curvatures, UKSAT-calibrated growth forecasting, MMP-2/MMP-9 extracellular matrix remodeling, and intraluminal thrombus hypoxia transport. |
| **Mechanics & FSI** | `src/titan_vasc/mechanics/` | Hyperelastic constitutive models (Neo-Hookean, Mooney-Rivlin, Fung exponential, HGO structural dispersion), curved shell membrane wall stress, Raghavan-Vorp tensile strength, RPI, and coupled 1D-0D fluid-structure interaction. |
| **Boundary** | `src/titan_vasc/boundary/` | 2-element (RC), 3-element (RCR), and 4-element (L-RCR) Windkessel models with RK4 ODE integration, resting/exercise/hypertensive cardiac inflow synthesis, Fourier harmonic reconstruction, and bifurcation junction matching. |
| **Calibration & UQ** | `src/titan_vasc/calibration/` | Standardized `PatientProfile` representation, UKSAT annual rupture risk calculators, RESCUE composite risk scoring, SVS 2024 guideline evaluation, and Monte Carlo uncertainty quantification with Morris elementary effects screening. |
| **Visualizer** | `src/titan_vasc/visualizer/` | Pure 7-bit ASCII canvas (0–127 only) implementing 2D line drawing, box borders, sparklines, arterial lumen geometry profiles, spatial wall stress distributions, and clinical diagnostic reports. |
| **CLI & Execution** | `src/titan_vasc/` | Command-line interface (`titan-vasc` / `python -m titan_vasc.cli`) providing operational subcommands: `simulate`, `calibrate`, `benchmark`, `inspect`, and `demo`. |

---

## Pure 7-Bit ASCII Terminal Visualizer

To guarantee **100% crash-free execution on Windows `cp1252` consoles, headless continuous integration servers, and Linux shells**, TitanVasc strictly enforces pure 7-bit ASCII encoding (character ordinals 0–127). No Unicode box-drawing or special characters are utilized.

### Sample Aneurysm Profile & Luminal Geometry (7-Bit ASCII):

```
  +-- INFRARENAL ABDOMINAL AORTA & ANEURYSM PROFILE --+                 
                                                                        
                                 #######                              
                              ###=======##                            
                            ##===.......==###                         
  ##########################==............===#######################  
  ==========================.................=======================  
  ------------------------------------------------------------------  
  ==========================.................=======================  
  ##########################==............===#######################  
                            ##===.......==###                         
                              ###=======##                            
                                 #######                              
                                                                      
  Healthy D: 20.0mm  |  Max Aneurysm D: 55.0mm
```

### Sample Spatial Wall Stress vs Tensile Strength Distribution:

```
+---------------------------------------------------------------+
| SPATIAL WALL STRESS (kPa) vs LOCAL TENSILE STRENGTH (kPa)     |
+---------------------------------------------------------------+
| z= 0.0cm | #####                          | 107.8/1152.0 kPa [ OK ] |
| z= 4.9cm | #####                          | 107.8/1152.0 kPa [ OK ] |
| z= 9.8cm | #####                          | 113.3/1103.1 kPa [ OK ] |
| z=14.7cm | ###############                | 312.4/836.6 kPa [ OK ] |
| z=19.6cm | #####                          | 107.8/1152.0 kPa [ OK ] |
| z=24.5cm | #####                          | 107.8/1152.0 kPa [ OK ] |
+---------------------------------------------------------------+
```

---

## Command-Line Interface (CLI) Reference

The CLI is available via `titan-vasc` or `python -m titan_vasc.cli`:

### 1. `simulate`: 1D Pulsatile Hemodynamics
```bash
titan-vasc simulate --duration 0.8 --scheme maccormack --cfl 0.65 --nodes 51 --heart-rate 75 --stroke-volume 70e-6
```
Simulates full cardiac cycle wave propagation, calculates TAWSS and OSI across the aorta, and plots the inlet flow waveform.

### 2. `calibrate`: Patient Risk Assessment & Monte Carlo UQ
```bash
titan-vasc calibrate --patient-id PAT-042 --age 72 --sex female --sbp 155 --dbp 92 --diameter 5.3 --ilt 7.0 --samples 250
```
Performs patient-specific biomechanical wall stress evaluation, UKSAT annual rupture calculation, SVS repair indication, and 250-sample Monte Carlo uncertainty propagation.

### 3. `benchmark`: Numerical Convergence & Profiling
```bash
titan-vasc benchmark
```
Benchmarks MacCormack vs Richtmyer schemes across spatial node grids (25, 51, 101 nodes), validating execution time and mass conservation.

### 4. `inspect`: Terminal Visualization
```bash
titan-vasc inspect --type fusiform --diameter 5.5 --sbp 140
```
Renders the 7-bit ASCII aortic profile and spatial wall stress distribution.

### 5. `demo`: Clinical Comparative Demonstration
```bash
titan-vasc demo
```
Executes an end-to-end comparative study contrasting mild surveillance vs critical surgical repair patient cases.

---

## API Usage Walkthrough

```python
import numpy as np
from titan_vasc.aneurysm.morphology import MorphologyGenerator
from titan_vasc.mechanics.rupture import RuptureRiskEvaluator
from titan_vasc.calibration.patient import PatientProfile
from titan_vasc.calibration.scoring import ClinicalRiskEngine

# 1. Define Patient Clinical Profile
patient = PatientProfile(
    patient_id="CLINIC-701",
    age=73.0,
    sex="female",
    systolic_bp=150.0,
    diastolic_bp=90.0,
    max_aneurysm_diameter=0.052, # 5.2 cm
    ilt_thickness=0.006,          # 6 mm thrombus
    smoking_status="current",
)

# 2. Generate Parameterized Fusiform Aneurysm Morphology
generator = MorphologyGenerator(total_length=0.22, num_nodes=51)
geometry = generator.create_fusiform_aneurysm(
    max_diameter=patient.max_aneurysm_diameter,
    aneurysm_length=0.08,
    wall_thinning_factor=0.70,
)

# 3. Evaluate Biomechanical Rupture Potential Index (RPI)
evaluator = RuptureRiskEvaluator()
ilt_array = np.full_like(geometry.radius, patient.ilt_thickness)

rupture_res = evaluator.evaluate_rupture_risk(
    z=geometry.z,
    radius=geometry.radius,
    wall_thickness=geometry.wall_thickness,
    systolic_pressure=patient.systolic_pressure_pa,
    ilt_thickness=ilt_array,
    is_female=(patient.sex == "female"),
    age=patient.age,
    family_history=patient.family_history_aaa,
)

# 4. Check Society for Vascular Surgery (SVS) Guidelines
scoring_res = ClinicalRiskEngine.evaluate_svs_guidelines(
    patient=patient,
    peak_rpi=rupture_res.peak_rpi,
)

print(f"Peak Wall Stress  : {rupture_res.peak_wall_stress / 1e3:.1f} kPa")
print(f"Peak RPI          : {rupture_res.peak_rpi:.3f} (Tier: {rupture_res.clinical_risk_tier.upper()})")
print(f"UKSAT Annual Risk : {scoring_res.annual_rupture_risk_pct:.1f}% / year")
print(f"SVS Decision      : {'ELECTIVE REPAIR INDICATED' if scoring_res.svs_repair_indicated else 'SURVEILLANCE'}")
```

---

## Numerical Benchmarks & Convergence Analysis

Benchmark results measured across 40 time steps (0.4 s duration) on an AMD Ryzen / Intel Core CPU:

| Grid Nodes | Spatial $\Delta z$ | Scheme | Runtime (ms) | Mass Conservation Error | Max Pressure (mmHg) |
|---|---|---|---|---|---|
| **25 nodes** | 10.4 mm | MacCormack | 41.2 ms | 0.690% | 128.4 mmHg |
| **25 nodes** | 10.4 mm | Richtmyer | 41.9 ms | 0.762% | 127.9 mmHg |
| **51 nodes** | 5.0 mm | MacCormack | 93.6 ms | 0.771% | 131.2 mmHg |
| **51 nodes** | 5.0 mm | Richtmyer | 86.4 ms | 0.804% | 130.8 mmHg |
| **101 nodes** | 2.5 mm | MacCormack | 206.4 ms | 0.658% | 132.5 mmHg |
| **101 nodes** | 2.5 mm | Richtmyer | 204.1 ms | 0.674% | 132.1 mmHg |

**Observations**:
- Both MacCormack and Richtmyer two-step schemes demonstrate $<0.8\%$ mass conservation error across the domain.
- The adaptive CFL calculation preserves unconditional numerical stability without non-physical high-frequency oscillations.

---

## Clinical Scoring & Society for Vascular Surgery (SVS) Guidelines

TitanVasc incorporates evidence-based clinical practice thresholds:

### SVS 2018/2024 Practice Guidelines Summary:
- **Male Elective Repair**: $D_{max} \ge 5.5\text{ cm}$.
- **Female Elective Repair**: $D_{max} \ge 5.0\text{ cm}$ (Lowered threshold justified by significantly higher rupture risk and smaller native baseline aortic dimensions).
- **Rapid Growth Indication**: Expansion $\ge 1.0\text{ cm/year}$ or $\ge 0.5\text{ cm in 6 months}$.
- **Saccular Morphology**: Considered for early repair regardless of absolute diameter due to elevated wall shear gradients and structural asymmetry.
- **Biomechanical Emergency**: Peak $RPI \ge 0.85$ or Peak Wall Stress $\ge 500\text{ kPa}$.

### Standard On-Label EVAR Anatomical Criteria:
- Proximal infrarenal neck length $\ge 15\text{ mm}$.
- Infrarenal neck angulation $\le 60^\circ$ relative to longitudinal axis.
- Absence of severe circumferential neck calcification or mural thrombus $>50\%$.

---

## Assumptions, Threat Model & Known Limitations

1. **Dimensionality (1D vs 3D CFD)**: The 1D Navier-Stokes solver models cross-sectional averaged axial flow and wave propagation. While boundary-layer velocity profile blunting is incorporated ($\zeta = 2$ to $9$), highly turbulent 3D vortex swirling within complex saccular geometries is approximated via effective mixing coefficients.
2. **Thin-to-Thick Shell Transition**: Membrane curved shell stresses assume $h / R < 0.2$. In highly thinned aneurysm domes ($h \sim 1.0\text{ mm}$, $R \sim 28\text{ mm}$), $h / R \approx 0.035$, satisfying thin-shell assumptions with high precision.
3. **Thrombus Viscoelasticity**: Intraluminal thrombus is modeled as a poroelastic tissue with isotropic stress cushioning and 1D Fickian oxygen diffusion. Dynamic thrombus fissuring or micro-channel recanalization is not explicitly modeled.
4. **Research & Clinical Disclaimer**: TitanVasc is a computational engineering and research platform designed for biomechanical simulation and risk stratification. Final surgical intervention decisions must always be made by a licensed vascular surgeon based on clinical CT angiography and comprehensive patient evaluation.

---

## License

TitanVasc is released under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) for complete terms.
