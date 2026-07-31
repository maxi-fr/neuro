# KCL & the collapse of seizure suppression — investigation note

> **Superseded (2026-07-31).** The root cause identified below — "`gamma` is not a physical
> current→field conversion" — was a *symptom*. `_compute_gamma` was differencing TVB's
> unit-vector sensor rows against millimetre region centroids in a permuted axis frame, which
> placed every electrode at the centre of the head and made all `gamma` rows collinear. The
> measurements in this note are correct; their interpretation is not, and no `gamma` rebuild
> was needed. See [`tes_field_geometry.md`](tes_field_geometry.md).

**Status:** diagnosis only, nothing fixed. Handoff for a future agent.
**Date:** 2026-07-27
**Question that started it:** "The commit that added the KCL check (`f540bd3`, *stimulation needs to follow KCL*) may have made control harder — can the controller still inhibit seizures?"

## TL;DR

Pre-KCL, the MPC controllers reached **≈97–98 %** seizure suppression. Post-KCL, the (linear) controller reaches only **≈5 %**. The KCL constraint *itself* is not the deep cause and is physically correct — the real root cause is that **`gamma` (the current→brain projection) channels ~98.5 % of its authority into the common-mode / net-current direction that KCL forbids.** The old 97 % was achieved by injecting large *net current* (mean `|sum(u)| ≈ 3.2`), which is physically impossible for a tES montage and is an artifact of a non-physical `gamma`. KCL closed that pathway in both the plant and the controllers, leaving only the ~1.5 % of authority that lives in the physically-realizable zero-sum subspace — hence the collapse.

So: **the controller still *technically* inhibits the seizure, but only marginally, and fixing KCL is the wrong move. The fix is a physically-valid `gamma` (and/or montage), then re-identify the predictors and re-run the nonlinear MPC.**

## What KCL commit `f540bd3` changed

Kirchhoff's current law (`sum(u) = 0`, a montage injects no net current) was enforced in **three** places:

- **Plant physics** — `_assert_zero_sum_current(u)` in [`src/neuro/jansen_rit.py`](../src/neuro/jansen_rit.py) (guarded by the new dynamics flag `enforce_zero_sum_current=True`). The plant now *rejects* any command with non-zero net current.
- **All MPC controllers** — `_sum_to_zero(u_vars)` added as an equality constraint (`lbg = ubg = 0`) in [`src/neuro/control.py`](../src/neuro/control.py) for `MPCController`, `LinearMPCController` (both `dense`/`sparse`), `NarxMPCController`, `LinearNarxMPCController`.
- **Excitation for system-ID** — `build_input_schedule` now projects each input row onto the zero-sum subspace (subtract the across-electrode mean), so predictors trained afterward only ever see zero-sum stimulation.

`gamma` was **not** changed (see memory note *kirchhoff-stim-constraint*: "electrode-level only, gamma unchanged"). Old models/data were archived under `artifacts/pre_kirchhoff_wrong_stim/` and must be regenerated.

## Experiments run (all reproducible)

### 1. Matched closed-loop suppression, post-KCL (linear controller)

Same plant (seed 69, `configs/simulation/l1_before.yaml`: excited Jansen–Rit, selected 25-ch montage, 3 electrodes `CP5/T7/F9`, `LinearMPCController`, `w_u=0`, `w_u_l1=0`, `u_max=3`, `horizon=20`, `dense`). Three variants share the identical noise realization. Metric = mean-square EEG over the steady window `t > 1 s` (seizure power). Suppression is relative to the uncontrolled baseline.

| variant | steady EEG ms | suppression | mean\|u\| | max\|u\| | net current |
| --- | --- | --- | --- | --- | --- |
| uncontrolled seizure | 263.5 | — | 0 | 0 | 0 |
| **KCL controller (current HEAD)** | 250.8 | **4.8 %** | 1.94 | 3 (saturating) | ~0 (2e-16) |
| KCL disabled (plant + controller) | 210.3 | 20.2 % | 2.87 | 3 | 3.0 |

The "KCL disabled" variant is a counterfactual: it monkeypatches `neuro.control._sum_to_zero` to emit identically-zero (non-binding) rows **and** builds the plant with `enforce_zero_sum_current=False`. It is *physically impossible* (net current) — included only to isolate the constraint's effect.

Decomposition of the logged control into zero-sum (`P = I - 11ᵀ/3`) vs common-mode parts:

- `kcl`: mean‖u_zerosum‖ = 4.12, mean‖u_commonmode‖ = 0.00.
- `no_kcl`: mean‖u_zerosum‖ = 4.72, mean‖u_commonmode‖ = 1.61, mean net current = −0.15.

The extra suppression without KCL comes partly from a larger reachable zero-sum projection (box corners) but mostly from the common-mode/net-current pathway.

### 2. Predictor control-response direction (`nn_predictor_2026-07-21_22-54-57`, linear depth-0)

Jacobian `B` of the 20-step predicted-EEG trajectory w.r.t. a constant control (exact, since the predictor is affine):

- Singular values `[5.235, 2.890, 8.3e-10]` — **rank 2**. The near-zero singular direction is exactly common-mode `[0.577, 0.577, 0.577]`.
- **100 % of the identified authority lies in the zero-sum subspace.**

Interpretation: the predictor was trained on zero-sum excitation (post-KCL `build_input_schedule`), so the common-mode direction is *unidentifiable*; the min-norm `lstsq` fit (see commit `b83f3b1`) zeroes it. The predictor is a faithful model of the weak zero-sum physics — it is not itself the bug.

### 3. `gamma` current→brain map (the root cause)

`gamma` is built in [`src/neuro/connectome.py`](../src/neuro/connectome.py) `_compute_gamma`: per electrode an **all-positive, unit-peak Gaussian** over region centroids, `exp(-dist²/2·spread²)`, each row normalized independently. Node drive is `n(u) = gammaᵀ u`.

For the 3-electrode montage (`CP5/T7/F9`, `spread=20`):

- All entries ≥ 0 (monopolar); per-electrode mass ≈ 7.0 each.
- **SVD of the current→node map `gammaᵀ`: singular values `[3.1314, 0.0445, 0.0105]`.** The dominant direction (σ=3.13) is **common-mode** (`|sum| = √3`); the two zero-sum directions are ~70× and ~300× weaker.
- **Only 1.5 % of the current→brain authority survives inside the zero-sum subspace.**
- Node-drive norm: common-mode `[1,1,1] → 3.13`, zero-sum `[1,−1,0] → 0.019`, `[1,1,−2] → 0.043`.
- The three `gamma` rows are **near-collinear in node space** (pairwise cosine 0.9994–0.9999) — their differences (any zero-sum combination) almost perfectly cancel.

This is *the* mechanism: KCL removes the one direction (`common-mode`) that carries virtually all of this `gamma`'s authority.

### 4. Confirmation from the archived pre-KCL runs

`artifacts/pre_kirchhoff_wrong_stim/curriculum_experiment/sims/` (full 62-ch, **2 electrodes**), steady-window metric, baseline steady EEG ms = 248.9:

| run | suppression | mean\|sum(u)\| (net current) | max\|u\| |
| --- | --- | --- | --- |
| linear_curriculum_seed69 | 98.2 % | 3.22 | 3 |
| nonlinear_curriculum_seed69 | 96.9 % | 3.38 | 3 |
| nonlinear_onestep_seed69 | 96.9 % | 3.43 | 3 |

Every ~97 % result drove a **large net current** (both electrodes saturated toward the same sign). That is exactly the KCL-forbidden, `gamma`-amplified pathway. This is the definitive confirmation that pre-KCL success depended on physically impossible net current.

## Hypotheses — verdicts

- **(User) `gamma` is not a physical current→field conversion — PRIMARY ROOT CAUSE (confirmed).** All-positive, independently-normalized, near-collinear Gaussian kernels make the current→brain map ~98.5 % common-mode. A real tES field is divergence-free/dipolar; a zero-sum montage should produce a structured, non-cancelling field. Here it produces almost nothing.
- **(User) The predictor was retrained on KCL-active data — TRUE, but a downstream symptom, not the cause.** The predictor correctly lost its common-mode response (unidentifiable under zero-sum excitation). Re-training it differently cannot recover authority that is not physically present under KCL. Note also the predictor tested is *linear* — the wrong tool for the nonlinear 97 % task (see below).
- **(Added) Apples-to-oranges controller/montage.** The 97 % figure is the **nonlinear softplus+PCA25** controller on the **full montage with 2 electrodes + net current**; the 5 % test is **linear, selected-25ch, 3 electrodes, zero-sum**. Before quantifying "the regression," the *actual* 97 % config must be re-run post-KCL. (Memory: only softplus+PCA25+single-shooting hit 97 %; full-62-no-PCA fails at ~10 %.)
- **(Added) Linear vs nonlinear operating point.** A linear predictor identified on seizing trajectories cannot predict the out-of-seizure regime that suppression drives toward. Even with good `gamma`, expect the nonlinear controller to be required.
- **(Added) `u_max` / current scaling cannot rescue a near-null direction.** With zero-sum authority ~70× weaker (σ 0.04 vs 3.13) and `|u|` already saturating at 3, rescaling `u_max` alone won't recover suppression.
- **(Added) Metric / plant parity.** Pre- vs post-KCL runs differ in montage (62 vs 25 ch), electrode count (2 vs 3), and baseline power (248.9 vs 263.5). Standardize the comparison before trusting magnitudes.
- **(Added) Excitation identifiability.** With a fixed `gamma`, zero-sum excitation barely perturbs the plant, so any post-KCL system-ID has poor control-response SNR. A better `gamma` also improves identifiability.

## Recommended next steps (ordered — NOT yet implemented)

1. **Establish the true regression.** Regenerate the nonlinear softplus+PCA25 predictor on post-KCL (zero-sum) excitation and re-run its single-shooting MPC on a plant matched to the pre-KCL 97 % run. Confirm whether nonlinear+full-montage also collapses under KCL (expected: yes, given the `gamma` finding).
2. **Fix `gamma` to a physically-valid current→field model** — the crux:
   - *Minimum:* model the montage as genuine source/sink geometry (bipolar/multipolar current dipoles) so zero-sum currents produce structured, non-cancelling node drive. Stop normalizing each electrode independently to unit peak; that erases relative geometry and forces near-collinear rows.
   - *Better:* use a real quasi-static tES lead field (Laplace with head conductivities), consistent with the EEG gain `L` already in the connectome. Success criterion: the dominant singular directions of `gammaᵀ` should be **zero-sum**, not common-mode.
3. **Re-place / re-parameterize the montage** so its zero-sum differential reach actually covers the seizing nodes (the `A = 3.6 / 3.4` excited regions in the config). Add a diagnostic: `gamma`-differential energy over the seizing nodes.
4. **Only after 2–3 give real zero-sum authority,** regenerate predictors (linear and nonlinear) on post-KCL excitation and re-run the MPC sweep.
5. **Add an identifiability check** on the excitation: confirm the zero-sum probe actually excites the new zero-sum pathway with adequate SNR before training.

## Reproduction

Post-KCL matched experiment (variant 1) — standalone script (paste into repo root):

```python
import copy
import casadi as ca, numpy as np
import neuro.control as ctrl
from simulate.config import load_config
from simulate.simulation import Simulation

BASE = load_config("configs/simulation/l1_before.yaml")
orig = ctrl._sum_to_zero

def run(cfg, disable_kcl=False):
    ctrl._sum_to_zero = (lambda uv: ca.vertcat(*[ca.sum1(u) * 0 for u in uv])) if disable_kcl else orig
    sim = Simulation.from_config(cfg); sim.run("out", prefix="log"); sim.export_results("out", prefix="log")
    ctrl._sum_to_zero = orig
    d = np.load("out/log.npz"); y, u = d["y_mea"], d["u"]
    t = np.linspace(0, BASE["t_end"], y.shape[0]); s = t > 1.0
    return np.mean(y[s] ** 2), np.mean(np.abs(u)), np.max(np.abs(u.sum(1)))

base = copy.deepcopy(BASE); base["controller"] = {"class_path": "neuro.control.ZeroController", "dt": 1e-4, "n_u": 3}
nokcl = copy.deepcopy(BASE); nokcl["dynamics"]["enforce_zero_sum_current"] = False
# run(base), run(copy.deepcopy(BASE)), run(nokcl, disable_kcl=True)
```

`gamma` SVD (variant 3):

```python
import numpy as np
from neuro.connectome import Connectome
G = Connectome.from_config({"speed": 50.0, "target_electrode": ["CP5", "T7", "F9"], "gamma_spread": 20.0, "K": 0.5357}).gamma
print(np.linalg.svd(G.T, compute_uv=False))  # -> [3.1314, 0.0445, 0.0105]; dominant dir is common-mode
```

## Pointers

- KCL constraint code: `neuro.control._sum_to_zero`, `neuro.jansen_rit._assert_zero_sum_current`, `build_input_schedule` (zero-sum projection).
- `gamma` construction: `neuro.connectome._compute_gamma`.
- Archived pre-KCL 97 % runs: `artifacts/pre_kirchhoff_wrong_stim/curriculum_experiment/sims/`.
- Only current post-KCL predictor: `artifacts/nn_predictor_2026-07-21_22-54-57/` (linear, selected-25ch).
- Related memories: *kirchhoff-stim-constraint*, *mpc-control-response-identification* ("suppression is gated by the under-identified tES→EEG response"), *nonlinear-mpc-softplus-pca-singleshooting* (the 97 % config), *l1-weight-scale-closed-loop*.
