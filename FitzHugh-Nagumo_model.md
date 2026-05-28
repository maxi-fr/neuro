# FitzHugh-Nagumo Whole-Brain Plant — Mathematical Reference

## Overview

The plant models a network of **N = 80** cortical regions (HCP structural connectome). Each region evolves according to the FitzHugh-Nagumo (FHN) equations, coupled via delayed diffusive connections derived from white-matter tract lengths. `NativeFHNPlant` is a pure-NumPy reimplementation that matches `FHNPlant` (neurolib) to near machine precision.

---

## 1. Local FHN Dynamics

Each node $i \in \{1, \ldots, N\}$ has two state variables:

| Variable | Role |
|----------|------|
| $v_i(t)$ | Fast excitatory variable (membrane voltage analogue) |
| $w_i(t)$ | Slow recovery variable |

The continuous-time ODEs are:

$$
\dot{v}_i = -\alpha v_i^3 + \beta v_i^2 + \gamma v_i - w_i + I_i^{\text{coup}}(t) + \eta_i^v(t) + v_{\text{ext}}
$$

$$
\dot{w}_i = \frac{v_i - \delta - \varepsilon\, w_i}{\tau} + \eta_i^w(t) + w_{\text{ext}}
$$

### Default parameters

| Symbol | Value | Meaning |
|--------|-------|---------|
| $\alpha$ | 3.0 | Cubic term — controls excitability threshold |
| $\beta$ | 4.0 | Quadratic term — N-shaped nullcline asymmetry |
| $\gamma$ | −1.5 | Linear term |
| $\delta$ | 0.0 | $w$-nullcline offset |
| $\varepsilon$ | 0.5 | Recovery rate coupling |
| $\tau$ | 20.0 ms | Recovery time constant |
| $v_{\text{ext}}$ | 1.0 | Constant external drive (puts nodes near limit cycle) |
| $w_{\text{ext}}$ | 0.0 | External drive on recovery variable |

The $v$-nullcline ($\dot{v} = 0$) is a cubic, and the $w$-nullcline ($\dot{w} = 0$) is linear. Their intersection determines whether the node is quiescent or oscillating; with $v_{\text{ext}} = 1.0$ the system operates near a limit cycle.

---

## 2. Structural Connectivity & Propagation Delays

The connectome supplies two matrices:

- **$C \in \mathbb{R}^{N \times N}$** — normalised structural connectivity weights (self-connections zeroed out)
- **$D \in \mathbb{R}^{N \times N}$** — fibre lengths in mm

Propagation delay from node $j$ to node $i$ in discrete timesteps:

$$
\tau_{ij}^{\text{steps}} = \text{round}\!\left(\frac{D_{ij}}{v_s \cdot \Delta t}\right), \quad v_s = 20\;\text{mm/ms}
$$

The **history window** must span the maximum delay plus one:

$$
s = \max_{i,j}\bigl(\tau_{ij}^{\text{steps}}\bigr) + 1
$$

---

## 3. Diffusive Coupling

Node $i$ receives input from all other nodes $j$ whose axons connect to it. The coupling is **diffusive** — it is proportional to the difference between the (delayed) source activity and the current target activity:

$$
I_i^{\text{coup}}(t) = K_{\text{gl}} \sum_{j=1}^{N} C_{ij} \bigl[v_j(t - \tau_{ij}) - v_i(t - \Delta t)\bigr]
$$

| Symbol | Value | Meaning |
|--------|-------|---------|
| $K_{\text{gl}}$ | 0.6 | Global coupling gain |

In the implementation this is vectorised over all $N$ nodes simultaneously. At timestep $k$ (absolute buffer index $i = s + k$):

```text
delayed_indices[i, j] = i - τ_steps[i,j] - 1
delayed_v[i, j]       = vs[j, delayed_indices[i,j]]
vs_input[i]           = K_gl * Σ_j ( C[i,j] * (delayed_v[i,j] - vs[i, i-1]) )
```

The `- 1` offset in the index ensures the source is evaluated at one step before the delay boundary, exactly matching neurolib's convention.

---

## 4. Ornstein-Uhlenbeck Noise

Each node has two independent OU processes $(\eta^v_i, \eta^w_i)$ that provide coloured noise with zero mean and time constant $\tau_{\text{ou}}$:

$$
\eta^v_i(t + \Delta t) = \eta^v_i(t) - \frac{\eta^v_i(t)}{\tau_{\text{ou}}} \Delta t + \sigma_{\text{ou}}\sqrt{\Delta t}\;\xi^v_i(t)
$$

$$
\eta^w_i(t + \Delta t) = \eta^w_i(t) - \frac{\eta^w_i(t)}{\tau_{\text{ou}}} \Delta t + \sigma_{\text{ou}}\sqrt{\Delta t}\;\xi^w_i(t)
$$

where $\xi^v_i, \xi^w_i \sim \mathcal{N}(0, 1)$ are i.i.d. at each step, and:

| Symbol | Value | Meaning |
|--------|-------|---------|
| $\tau_{\text{ou}}$ | 5.0 ms | Noise correlation time |
| $\sigma_{\text{ou}}$ | 0.05 (default) | Noise amplitude |

The stationary variance is $\sigma_{\text{ou}}^2 \cdot \tau_{\text{ou}} / 2$. Setting $\sigma_{\text{ou}} = 0$ gives a fully deterministic system.

> **Implementation detail:** neurolib pre-generates the full noise array for each chunk *before* the integration loop, then resets the RNG seed at the start of every chunk. `NativeFHNPlant` replicates this exactly — that's what makes multi-step trajectories reproducible and numerically identical between the two classes.

---

## 5. Forward Euler Integration

All ODEs are discretised with forward (explicit) Euler at step $\Delta t$:

$$
v_i(t + \Delta t) = v_i(t) + \Delta t \cdot \dot{v}_i(t)
$$

$$
w_i(t + \Delta t) = w_i(t) + \Delta t \cdot \dot{w}_i(t)
$$

where $\dot{v}_i, \dot{w}_i$ are evaluated at time $t$ using the full expressions from §1–§3. The OU update (§4) happens **after** the state update, using the pre-generated noise sample for that step.

---

## 6. State Buffer Layout

To handle delays efficiently, the implementation maintains a rolling buffer:

```text
vs : (N, s + n_steps)
     |<--- history --->|<--- new steps --->|
     0               s-1  s          s+n_steps-1
```

At step $k$ (absolute index $i = s + k$), the past value $v_j(t - \tau_{ij})$ is fetched from column $i - \tau_{ij}^{\text{steps}} - 1$. After integration, only the last $s$ columns are retained as the history window for the next chunk.

---

## 7. Initial Conditions

```text
v_i(0) ~ Uniform(0, 0.05)   × 0.05
w_i(0) ~ Uniform(0, 0.05)   × 0.05
```

The history window is initialised by broadcasting the single IC value across all $s$ past timesteps — equivalent to assuming the node was at its initial state for all $t < 0$.

---

## 8. Synthetic EEG Projection

A random lead-field matrix $L \in \mathbb{R}^{M \times N}$ maps node activity to $M$ EEG channels:

$$
\mathbf{e}(t) = L\, \mathbf{v}(t), \quad L_{ki} = \frac{\tilde{L}_{ki}}{\|\tilde{L}_{k\cdot}\|_2}
$$

where $\tilde{L} \sim \mathcal{N}(0,1)$ and rows are $\ell_2$-normalised. This is purely a forward observation model — it does not feed back into the dynamics.

---

## 9. Complete Discretised System (summary)

For each timestep $k = 0, 1, \ldots, T-1$:

$$
\boxed{
\begin{aligned}
I_i^k &= K_{\text{gl}} \textstyle\sum_j C_{ij}\!\left[v_j^{k-\tau_{ij}} - v_i^{k-1}\right] \\[4pt]
v_i^{k+1} &= v_i^k + \Delta t\!\left(-\alpha (v_i^k)^3 + \beta (v_i^k)^2 + \gamma v_i^k - w_i^k + I_i^k + \eta_i^{v,k} + v_{\text{ext}}\right) \\[4pt]
w_i^{k+1} &= w_i^k + \Delta t\!\left(\frac{v_i^k - \delta - \varepsilon w_i^k}{\tau} + \eta_i^{w,k} + w_{\text{ext}}\right) \\[4pt]
\eta_i^{v,k+1} &= \eta_i^{v,k}\!\left(1 - \tfrac{\Delta t}{\tau_{\text{ou}}}\right) + \sigma_{\text{ou}}\sqrt{\Delta t}\;\xi_i^{v,k} \\[4pt]
\eta_i^{w,k+1} &= \eta_i^{w,k}\!\left(1 - \tfrac{\Delta t}{\tau_{\text{ou}}}\right) + \sigma_{\text{ou}}\sqrt{\Delta t}\;\xi_i^{w,k} \\[4pt]
\mathbf{e}^k &= L\,\mathbf{v}^k
\end{aligned}
}
$$
