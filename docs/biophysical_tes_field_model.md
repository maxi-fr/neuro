# Biophysical tES Field Models: From Phenomenological Surrogates to Cortical-Normal Superposition

**Date:** 2026-08-04
**Scope:** Comparative analysis of transcranial electrical stimulation (tES) spatial projection models ($\gamma$) used in whole-brain neural mass dynamics (Jansen-Rit) and model predictive control (MPC).

---

## 1. Executive Summary

In whole-brain neurostimulation modeling, the spatial projection vector/matrix $\boldsymbol{\gamma}$ maps external electrode currents ($u_k$ in mA) to local perturbation voltages ($U_{\text{tES}, i}$ in mV) acting on postsynaptic potentials of cortical pyramidal cell populations:

$$\mathbf{U}_{\text{tES}}(t) = \boldsymbol{\gamma}^T \mathbf{u}(t)$$

Two distinct modeling methodologies are implemented in this repository:

1. **Yu et al. (2024) Phenomenological Model**: Uses electric field magnitude $\|\mathbf{E}\||$ combined with a spatial median voltage step function $\text{sign}(V - V_{\text{med}})$.
2. **Biophysically Accurate Cortical-Normal Model ($E_{\text{normal}}$)**: Uses cable-theoretic somatodendritic polarization rules driven by the directional electric field component normal to the cortical surface ($\mathbf{E} \cdot \hat{\mathbf{n}}$).

---

## 2. Yu et al. (2024) Phenomenological Model

### Mathematical Formulation

From Yu et al. (*Nonlinear Dynamics*, 2024, Eq. 8):

$$\gamma_i = \|\mathbf{E}(x_i, y_i, z_i)\| \cdot \text{sign}\Big(V(x_i, y_i, z_i) - V_{\text{med}}\Big)$$

where:

* $(x_i, y_i, z_i)$ is the 3D MNI RAS coordinate of region centroid $i$.
* $\|\mathbf{E}(x_i, y_i, z_i)\|$ is the magnitude of the 3D electric field vector.
* $V(x_i, y_i, z_i)$ is the electrostatic potential.
* $V_{\text{med}} = \frac{V_{\text{max}} + V_{\text{min}}}{2}$ is the spatial median voltage over the head volume.

### Rationale & Design Goals

* **Designed for Single Cathode-Anode Pairs**: Developed for a 1-cathode / 1-anode configuration (e.g., CP5 cathode, Ex8 anode).
* **Qualitative Polarity Rule**: Ensures nodes near the cathodal electrode receive a negative sign ($\gamma_i < 0$, inducing hyperpolarization/inhibition) while nodes near the anodal electrode receive a positive sign ($\gamma_i > 0$, inducing depolarization/excitation).
* **Atlas Compatibility**: Can be evaluated directly at 3D point centroids without requiring high-resolution cortical surface meshes or local normal orientation vectors ($\hat{\mathbf{n}}$).

### Limitations for Multi-Electrode Control

* **Non-linear Superposition**: Because of the vector norm $\|\mathbf{E}\||$ and the step function $\text{sign}(\cdot)$, the scalar $\gamma$ does not obey exact vector superposition across overlapping electrode fields:
  $$\|\mathbf{E}_1 + \mathbf{E}_2\| \neq \|\mathbf{E}_1\| + \|\mathbf{E}_2\|$$
* **Spatial Discontinuity**: The $V_{\text{med}}$ threshold creates an artificial step boundary across the mid-voltage plane of the head.

---

## 3. Biophysically Accurate Cortical-Normal Model ($E_{\text{normal}}$)

### Microscopic Biophysical Basis

Cortical pyramidal neurons are geometrically oriented **perpendicular (normal) to the cortical surface** (somas in deep cortical layers, apical dendrites extending vertically toward layer I).

According to neuronal cable theory and sub-threshold polarization laws (Bikson et al., 2004; Rahman et al., 2013):

$$\Delta V_m = \lambda \cdot (\mathbf{E} \cdot \hat{\mathbf{n}})$$

where:

* $\Delta V_m$ is the steady-state somatic membrane potential deflection (mV).
* $\lambda$ is the neuronal polarization length ($\approx 0.2 - 0.5\text{ mm}$).
* $\mathbf{E}$ is the electric field vector ($\text{V/m}$).
* $\hat{\mathbf{n}}$ is the unit vector normal to the cortical surface (pointing outward from white matter to scalp).

```text
                  Scalp Surface
       ───────────────────────────────────
                      │  ▲
                      │  │  n_hat (Outward Cortical Normal)
                      ▼  │
               Apical Dendrite
                      │
                 Pyramidal
                   Cell
                      │
                    Soma
       ───────────────────────────────────
                  White Matter
```

### Mathematical Formulation

For a discrete cortical region $i$ containing $M_i$ cortical surface vertices:

$$\gamma_{k, i} = \frac{\lambda}{M_i} \sum_{v \in \text{Region } i} \mathbf{E}_k(v) \cdot \hat{\mathbf{n}}(v)$$

### Physical Properties & Advantages

1. **Strict Linear Superposition**: Dot products preserve linear vector field superposition exactly:
   $$\mathbf{E}_{\text{total}} \cdot \hat{\mathbf{n}} = \left(\sum_k u_k \mathbf{E}_k\right) \cdot \hat{\mathbf{n}} = \sum_k u_k (\mathbf{E}_k \cdot \hat{\mathbf{n}})$$
   This ensures compatibility with linear and nonlinear Model Predictive Control (MPC) and multi-channel optimization.
2. **Continuous Biophysical Polarity**:
   * **Inward Field ($\mathbf{E} \cdot \hat{\mathbf{n}} < 0$)**: Current enters dendrites and exits soma $\rightarrow$ **Somatic Depolarization (Anodal / Excitatory)**.
   * **Outward Field ($\mathbf{E} \cdot \hat{\mathbf{n}} > 0$)**: Current exits dendrites and enters soma $\rightarrow$ **Somatic Hyperpolarization (Cathodal / Inhibitory)**.
3. **Kirchhoff's Current Law (KCL) Compatibility**: Guarantees zero-net-current excitation in multi-electrode montages.

---

## 4. Comparison Table

| Property | Yu et al. (2024) Model | Cortical-Normal Model ($E_{\text{normal}}$) | Analytical Point Source |
| :--- | :--- | :--- | :--- |
| **Mathematical Driving Function** | $\|\mathbf{E}\| \cdot \text{sign}(V - V_{\text{med}})$ | $\lambda \cdot (\mathbf{E} \cdot \hat{\mathbf{n}})$ | $100 / \sqrt{r^2 + s^2}$ |
| **Required Mesh Geometry** | 3D Voxel Grid ($V, E$) | Cortical Surface + Normals ($\hat{\mathbf{n}}$) | Region Centroid Coordinates |
| **Linear Superposition** | Approximate (Channel-wise) | **Strictly Linear** | Strictly Linear |
| **Polarity Boundary** | Artificial ($V_{\text{med}}$ step) | Smooth / Continuous ($\mathbf{E} \cdot \hat{\mathbf{n}}$) | None (All-positive distance) |
| **Biophysical Mechanism** | Phenomenological scalar | Cable-theoretic somatodendritic | Distance-decay heuristic |
| **Primary Use Case** | Replicating Yu 2024 paper | MPC & multi-electrode control | Fast analytical prototyping |

---

## 5. Usage in Codebase

### A. Generating Yu et al. (2024) ROAST $\gamma$ Matrix (MATLAB)

```matlab
% Generates K x 76 gamma matrix using Yu et al. (2024) Eq. 8
montages = { ...
    {'TP7', -1, 'Ex8', 1}, ...  % Channel 1
    {'CP5', -1, 'Ex8', 1}    ...  % Channel 2
};
[gamma, metadata] = generate_roast_gamma(montages, ...
    'elecType', 'pad', ...
    'elecSize', [50 30 3], ...
    'outputFile', 'data/roast_gamma.mat');
```

### B. Python Integration (`Connectome`)

```python
from neuro.connectome import Connectome

# Load the precomputed ROAST 3D leadfield (63 channels x 76 nodes x 3).
# reduction_method picks the E_i -> U_tes_i equation; 'cortical_normal' is E dot n.
conn_field = Connectome.from_config({
    "speed": 50.0,
    "K": 0.60,
    "gamma_model": "roast_3d",
    "leadfield_path": "data/roast_leadfield_3d.npz",
    "reduction_method": "cortical_normal",
})
```

The control vector is per-channel current over `conn_field.control_channel_labels` (the 62 scalp
electrodes followed by the `Ex8` return), and must sum to zero. `target_electrode` does not apply
to `roast_3d`; it selects electrodes for the `analytical` Coulomb model only.

---

## 6. Potential Limitations & Failure Modes of $E_{\text{normal}}$

While biophysically more realistic than scalar magnitude heuristics, the cortical-normal model introduces specific challenges when integrated into macro-scale whole-brain network models:

### 6.1 The Sulcal Cancellation Artifact

In coarse connectome models (76 regions), a single cortical region spans several square centimetres containing folded sulci:

* On the left bank of a sulcus, the cortical normal $\hat{\mathbf{n}}$ points left; on the right bank, it points right.
* A uniform electric field $\mathbf{E}$ induces **opposite somatic polarizations** on opposite banks of the same sulcus ($\mathbf{E} \cdot \hat{\mathbf{n}} > 0$ vs $< 0$).
* Averaging $\mathbf{E} \cdot \hat{\mathbf{n}}$ over the entire region can cause these opposing fields to **cancel to near zero**, underestimating the localized stimulation drive experienced by folded cortical tissue.

### 6.2 Subcortical & Mesial Foci (Ill-Defined Normals)

In Temporal Lobe Epilepsy (TLE), primary seizure onset zones (EZ) are located in deep subcortical and mesial temporal structures—such as the **Hippocampus (`lHC`)** and **Amygdala (`lAMYG`)**.

* Unlike the cerebral neocortex, subcortical nuclear masses do not have a 2D sheet architecture or a single outward cortical normal vector $\hat{\mathbf{n}}$.
* Assigning a cortical normal $\hat{\mathbf{n}}$ to deep subcortical nodes is mathematically ill-defined.

### 6.3 Mesh Sensitivity & Sign Inversion

* Near gyral crowns and sulcal beds, cortical surface normals change direction by $180^\circ$ over millimetre distances.
* Numerical mesh noise, registration errors, or low vertex resolution can accidentally invert $\hat{\mathbf{n}} \rightarrow -\hat{\mathbf{n}}$, flipping the predicted stimulation effect from **inhibitory (cathodal)** to **excitatory (anodal)**.

### 6.4 Neglect of Axonal Gradient Activation

* $E_{\text{normal}}$ only models somatic polarization of vertically aligned pyramidal dendrites.
* Electric fields also activate bent axons and white-matter fiber tracts via the **activating function** (spatial gradient along the axon path, $\frac{\partial E_s}{\partial s}$). $E_{\text{normal}}$ omits axonal tract driving forces.
