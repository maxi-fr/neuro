# Predictor rewrite: JAX/Equinox -> PyTorch

Implementation plan. Each part is one commit and leaves the suite green.
Written against `1f088fa`. **Delete this file when Part 7 lands.**

## Context for a cold start

The NN predictor is a one-step MLP `f(y_hist, u_hist) -> y_next` unrolled autoregressively over a
horizon. It is trained in JAX/Equinox, saved as a 3-file artifact, and re-expressed as a CasADi
symbolic graph so IPOPT can optimise through it inside the MPC.

Read before starting: `src/neuro/nn_training.py`, `src/neuro/prediction.py`,
`src/neuro/nn_predictor_casadi.py`, `src/neuro/artifacts.py`, `docs/nn_predictor_training.md`.

### Locked decisions

| | |
|---|---|
| Scope | Framework + process. The model class stays an AR-unrolled one-step MLP. `control.py` is not touched. |
| Oracle | torch rollout == CasADi bridge to 1e-10. No golden training curve; retuning is expected. |
| Artifact | Single `.npz`, framework-free NumPy weights. No backward compat with `.eqx`. |
| Losses | MSE + PSD. FC loss deleted. |
| PSD | Bit-faithful Welch replica, pinned against `scipy.signal.welch`. |
| Layout | `src/neuro/predictor/` — `artifact.py`, `data.py` (no torch); `module.py`, `losses.py`, `train.py` (torch). |
| API | Pure `train(cfg, files, *, seed_offset) -> TrainingResult`; `result.save(dir)`; scripts plot. |
| Dtype | Explicit `float64`. Never `torch.set_default_dtype`. |
| Deps | Keep `cpu`/`cu128` extras and the index split. Drop `torchvision` and `optax`. |

### Invariants that hold at every commit

1. **`neuro.predictor.artifact`, `neuro.predictor.data`, `neuro.nn_predictor_casadi`, `neuro.control`
   and `neuro.artifacts` never import torch.** This is what makes the control path immune to the
   rewrite. Part 2 adds a test that enforces it.
2. **`control.py`, `transforms.py`, `filtering.py`, `esn_training.py` are not edited** (except one
   import line in `esn_training.py`, Part 1).
3. Everything is `float64` end to end. CasADi and IPOPT are double precision and the oracle
   tolerance is 1e-10.
4. `pre-commit run --all-files` passes (ruff check + format, ty, pytest, marimo, markdownlint).

### Part 0 — before anything

Delete `docs/torch_migration.md`. It is superseded; several of its recommendations were overruled
(it proposed cutting the PSD loss, a throwaway JAX->torch weight loader, a global default dtype, a
subpackage literally named `torch/`, and it claims `MPCController` reaches past the `SymbolicModel`
seam — that was fixed in `1f088fa`).

---

## Part 1 — Framework-free artifact and data module

**JAX still does the training.** This part only changes what the artifact *is* and who reads it.
Doing it first is what makes the CasADi oracle work: `tests/test_nn_predictor_casadi.py` keeps
pinning CasADi == JAX across this commit, so when Part 2 shows torch == CasADi you get
torch == CasADi == JAX transitively.

### Files

Create `src/neuro/predictor/__init__.py`, `artifact.py`, `data.py`.
Edit `prediction.py`, `nn_training.py`, `nn_predictor_casadi.py`, `artifacts.py`,
`closed_loop_eval.py`, `esn_training.py`, `config.py`, and 6 test files.

### `predictor/artifact.py` — no torch, no jax

```python
Activation = Literal["relu", "tanh", "softplus"]

@dataclass(frozen=True)
class MLPArtifact:
    layers: tuple[tuple[FloatArray, FloatArray], ...]  # (W (out, in), b (out,)) forward order
    activation: Activation
    n_y: int            # past EEG steps
    n_u: int            # past control steps
    horizon: int
    n_channels: int     # model-space channels: latent k under PCA, else raw C
    n_controls: int
    dt: float
    downsample: int
    y_pipeline: Pipeline
    u_pipeline: Pipeline
```

Carry over unchanged from `prediction.py`: `model_type` (now `"mlp"`, written explicitly),
`n_eeg_channels`, `priming_steps`, `encode`, `decode`, `prime`.

Replace, because they used Equinox:

- `is_linear` — was `artifact.model.model.depth == 0`, becomes `len(self.layers) == 1`.
- `rollout(state, u_future)` — reimplement the AR loop in NumPy. It must reproduce
  `AutoregressivePredictor.__call__` (`prediction.py:62-83`) exactly: shift `u_window`, concatenate
  `[y_window.flatten(), u_window.flatten()]`, apply the MLP, shift `y_window`. Note the current
  version rebuilds an `AutoregressivePredictor` on every call — do not carry that over.
- `forward_1step(y_flat, u_flat) -> y_next` — the NumPy MLP forward. Activation applies to every
  layer except the last (matching `_mlp_forward_ca`). Part 6 needs this to accept a batch dim.

### Single-`.npz` serialisation

Path convention is unchanged: configs give a suffix-less stem (`artifacts/x/model`), and
`save`/`load` apply `.with_suffix(".npz")`.

```text
meta            0-d "<U" array holding json.dumps(...)   # loads without allow_pickle
layer.0.weight  (out, in)
layer.0.bias    (out,)
layer.1.weight  ...
y.0.center / y.0.scale / y.1.basis / y.1.mean   # Pipeline.array_dict("y")
u.0.center / u.0.scale                          # Pipeline.array_dict("u")
```

`meta` holds `model_type`, `activation`, `n_y`, `n_u`, `horizon`, `n_channels`, `n_controls`,
`n_eeg_channels`, `dt`, `downsample`, `n_layers`, and the `y_pipeline`/`u_pipeline` step tags
(rebuild with `Pipeline.from_serialized`, which already exists).

Store the JSON as `np.array(json.dumps(meta))` — a 0-d unicode array, **not** an object array, so
`np.load` needs no `allow_pickle`. Read it back with `json.loads(str(npz["meta"]))`.

### `artifacts.py` — dispatch

`load_any_artifact` currently identifies an MLP by the *absence* of a `model_type` key. Replace with:

```python
p = Path(artifact_path)
npz = p.with_suffix(".npz")
if npz.exists():                      # MLP: single npz carrying meta
    ... read meta, dispatch on model_type
return ESNArtifact.load(p)            # ESN: still .json + .scalers.npz + .weights.npz
```

No collision: the ESN writes `model.weights.npz`, so `model.npz` is unambiguous.

### `predictor/data.py` — no torch, no jax

Move verbatim from `nn_training.py`: `load_trajectory`, `split_data_files`,
`extract_windows_flattened`, `apply_to_blocks`, `build_dataset_for_trajectory`,
`transform_features`, `prepare_datasets`.

Delete `reshape_to_trajectory` — it is a one-line wrapper over `.reshape`.

**Fix the wart while moving it:** `prepare_datasets` discards the raw validation trajectories, which
is why `train_and_save_predictor` hand-inlines a second copy of the loading loop at
`nn_training.py:955-960`. Return a `Datasets` dataclass carrying `X_train`, `Y_train`, `X_val`,
`Y_val`, `val_trajs`, `n_channels`, `n_controls`.

`esn_training.py:8` imports `load_trajectory` and `split_data_files` from `nn_training` — repoint to
`neuro.predictor.data`. That is the only edit to that file.

### `nn_predictor_casadi.py`

- Delete `_extract_mlp_layers` entirely.
- `_layers` becomes `self.artifact.layers` (drop the `cached_property`).
- `self.artifact.model.activation` -> `self.artifact.activation`.
- `is_linear` -> `self.artifact.is_linear`.
- Drop `import equinox as eqx`.

`_mlp_forward_ca` and everything symbolic is unchanged.

### Temporary shims (deleted in Part 5)

- `prediction.py` keeps only the Equinox `AutoregressivePredictor`. `MLPArtifact` is gone from it.
- `nn_training.py` grows `_artifact_from_eqx(model, dt, downsample, y_pipeline, u_pipeline)` that
  pulls `(weight, bias)` out of the eqx MLP into NumPy and builds the new `MLPArtifact`.
- `closed_loop_eval.py:61`: `trial_dir / "model.eqx"` -> `trial_dir / "model.npz"`.
  Also `tests/test_closed_loop_eval.py:34`.

### `config.py`

`ModelConfig.activation: Literal["relu", "tanh", "softplus"] = "relu"`. It is currently a bare `str`
and a typo surfaces at MPC construction time, potentially hours into a sweep.

### Test fixtures

Six files build artifacts by constructing an `eqx.nn.MLP` with a `jax.random.PRNGKey`. Replace with
random NumPy `(W, b)` arrays — the fixtures get *shorter*. Files: `test_prediction.py`,
`test_nn_predictor_casadi.py`, `test_mpc_controller.py`, `test_linear_mpc_controller.py`,
`test_metrics.py`, `test_esn.py` (which has an `MLPArtifact` case at line 278).

### Verify

- Full suite green. `test_nn_predictor_casadi.py` (CasADi == JAX) must still pass — it is the anchor
  the whole strategy rests on.
- **Add a temporary test** (delete in Part 5): NumPy `MLPArtifact.rollout` == Equinox
  `AutoregressivePredictor` rollout to 1e-10 on random windows. This is the one commit where both
  implementations coexist, so it is the only chance to check the NumPy AR loop directly.
- Train one model end to end with the existing script; confirm the `.npz` loads and the MPC runs.

### Gotchas

- Window layout is restated in six places. `X` is
  `[y_past (n_y*C) | u_past (n_u*m) | u_future (horizon*m)]`, but the *model input* is
  `[y_window | u_window]` only — `u_future` is consumed one step at a time by the rollout.
- `n_channels` means *model-space* channels (latent `k` when PCA is on), not raw EEG. Raw EEG is
  `n_eeg_channels`. Getting these backwards is the most likely bug in this part.
- `AutoregressivePredictor` shifts `u_window` **before** the MLP call, so the newest control is
  already in the window when predicting `y_next`.

---

## Part 2 — torch module and the oracle

### Files

Create `src/neuro/predictor/module.py` and `tests/test_predictor_module.py`.

### `predictor/module.py`

```python
class AutoregressiveMLP(nn.Module):
    # nn.ModuleList of nn.Linear, kept explicit -- export reads it back
    # buffers: decode_basis (k, C), decode_mean (C,)   [optional, PCA only]
    def forward(self, x: Tensor) -> Tensor:   # (B, in) -> (B, horizon * n_channels)
```

Rules:

- Build every `nn.Linear` with `dtype=torch.float64`. No `set_default_dtype`.
- Activation on all layers except the last.
- `register_buffer` the PCA decode arrays here so `losses.py` and `train.py` do not thread
  `decode_basis`/`decode_mean` through their signatures (the JAX version threaded them through five).
- `to_artifact(dt, downsample, y_pipeline, u_pipeline) -> MLPArtifact` — `state_dict()` tensors to
  float64 NumPy.
- `from_artifact(art) -> AutoregressiveMLP` — needed by the oracle test and by any reload.
- The rollout is a Python `for` over the horizon. BPTT depth equals the horizon.

### Tests

**Oracle** — the load-bearing test:

```text
art  = random MLPArtifact (nontrivial standardizers, with and without PCA)
sym  = NNSymbolicModel(art)
mod  = AutoregressiveMLP.from_artifact(art)

drive sym with f_step / f_out over `horizon` steps using RAW u
drive mod with the same windows using u_pipeline.transform(u)
assert allclose(..., atol=1e-10)
```

The u convention is the trap: `NNSymbolicModel.predict_output` z-scores raw `u` *internally*
(`nn_predictor_casadi.py:199`), while the torch module expects control already in model space, as
`transform_features` produces it. Parametrise over `latent_dim in (None, k)`, `depth in (0, 2)` and
all three activations.

**Seam guard** — must run in a subprocess, because by then pytest has already imported torch via
other test modules:

```python
subprocess.run([sys.executable, "-c",
    "import sys, neuro.control; assert 'torch' not in sys.modules"], check=True)
```

---

## Part 3 — Losses

### Files

Create `src/neuro/predictor/losses.py` and `tests/test_predictor_losses.py`.
Delete `tests/test_nn_losses.py`. Edit `config.py` and 7 config YAMLs.

### `predictor/losses.py`

**`curriculum_mask(epoch, horizon, max_steps, start_epoch, end_epoch) -> Tensor`** — port
`curriculum_state` and fold in `_step_mask_selector`. The closure existed only so `filter_jit` would
not recompile as the trusted rollout length grew; in eager torch it is a plain function returning a
prefix-of-ones mask. `tests/test_curriculum.py` has no JAX in it — keep it, repoint the import.

**`welch_psd(x, nperseg, fs=1.0)`** — bit-faithful to `scipy.signal.welch` /
`jax.scipy.signal.welch` under the call the trainer makes:
`welch(pool, nperseg=horizon, noverlap=0, axis=-1)`. That means matching all the defaults:

- `detrend="constant"` — subtract the per-segment mean **before** windowing.
- `window="hann"`, periodic — `torch.hann_window(nperseg, periodic=True)`.
- `scaling="density"` — `scale = 1 / (fs * (win ** 2).sum())`.
- `return_onesided=True` — double every bin except DC, and except Nyquist when `nperseg` is even.
- `average="mean"` over segments.

With `noverlap=0` the segmentation is just `reshape(C, batch, horizon)`. Everything else is the
scaling convention, and that is the whole reason this needs a test.

**`predictor_loss(pred_traj, true_traj, step_mask, w_psd) -> (total, dict)`** — port `compute_loss`
minus FC. Keep `psd_gate = step_mask[-1]` (PSD only contributes at full curriculum length) and the
`horizon > 1` guard. Decode to channel space via the module's buffers, not via parameters.

Delete `_masked_fc`.

### Config

Remove `w_fc` from `TrainingConfig` and delete the `w_fc: 0.0` line from all 7 configs that set it.
`StrictConfig` has `extra="forbid"` and `tests/test_example_configs.py` validates every YAML under
`configs/`, so the schema change fails those 7 tests until the YAMLs are edited in the same commit.

### Tests

- `welch_psd` vs `scipy.signal.welch` to 1e-10. Parametrise over even and odd `nperseg` — the
  Nyquist-doubling branch only exists for even.
- **PCA-decode gradient property** (ported from the surviving half of `test_nn_losses.py`): with an
  orthonormal basis, the gradient of decoded-space MSE is a scalar multiple of the gradient of
  latent-space MSE. This is what justifies rolling out in latent space while scoring in channel
  space. Becomes a `torch.autograd` test.

---

## Part 4 — Training loop

### Files

Create `src/neuro/predictor/train.py` and `tests/test_predictor_train.py`.
Edit `scripts/run_nn_predictor.py`, `scripts/sweep_nn_predictor.py`, `config.py`.

### `predictor/train.py`

```python
@dataclass(frozen=True)
class TrainingResult:
    artifact: MLPArtifact
    train_losses: list[float]
    val_losses: list[float]
    rollout: RolloutNMSE                  # from artifacts.py, unchanged
    val_trajs: list[tuple[FloatArray, FloatArray]]
    du_sensitivity: float
    def save(self, artifact_dir: Path) -> None:   # model.npz + training_stats.json

def train(cfg: NNPredictorConfig, data_files: list[str], *, seed_offset: int = 0) -> TrainingResult
```

`train` does **no I/O** — no `matplotlib`, no writes. It returns; the caller persists and plots.

Port from `train_model`, replacing 18 positional parameters with the config object:

| Current | Torch |
|---|---|
| `optax.adamw` + `cosine_decay_schedule` | `torch.optim.AdamW` + `CosineAnnealingLR` |
| `eqx.filter_value_and_grad` / `filter_jit` | `loss.backward()` |
| `jax.vmap` in `predict_batch` | delete — torch batches natively |
| `get_dataloaders` | keep the index-shuffle, **seeded**. Do not adopt `DataLoader`. |
| `_warm_start_linear_model` | keep the NumPy `lstsq`; `eqx.tree_at` becomes `layer.weight.copy_()` |

**`best_model = model` becomes a bug.** JAX modules are immutable so that line captured a snapshot;
a torch module is mutable, so it would alias the live model and early stopping would silently return
the *last* model rather than the best. Use `copy.deepcopy(model.state_dict())`.

**Seed the shuffle.** `nn_training.py:210` uses a bare `np.random.default_rng()`, so training is not
reproducible from `training.seed` even in principle. Use `training.seed + seed_offset` for both the
shuffle and `torch.manual_seed`. No `use_deterministic_algorithms` — pointless on CPU.

**`du_sensitivity`**: `torch.autograd.functional.jacobian` of the rollout w.r.t. the future control
tensor, Frobenius norm, averaged over a **fixed subsample** of validation windows (a full Jacobian is
`(N*C, N*m)` = 3100x400 at `horizon=50, C=62` — do not run it on every window). Record it in
`training_stats.json` next to `nmse_rollout`. It makes "predicts EEG well, ignores stimulation"
visible in seconds instead of after a closed-loop sweep. MLP-only; the ESN gets no comparable number.

`TrainingConfig.device: Literal["cpu", "cuda"] = "cpu"`.

### Scripts

- `run_nn_predictor.py`: `train` -> `result.save(dir)` -> render `loss_curve.png` and
  `comparison.png`. `plot_training_curves` and the `plot_multistep_predictions` call move here out of
  the training module.
- `sweep_nn_predictor.py`: `train` -> `result.save(trial_dir)` -> return `result.rollout.pooled`.
  Sweeps skip plotting. The `optuna.TrialPruned`-on-NaN handling is unchanged.

`training_stats.json` keys: `train_loss`, `val_loss`, `nmse_rollout`, `nmse_rollout_per_step`,
`du_sensitivity`. The old `mse` key (teacher-forced) is dropped along with `evaluate_model`.

### Tests — `tests/test_predictor_train.py`

Smoke test, tiny and fast: 2 synthetic trajectories, 3 epochs, `hidden=4`, `horizon=3`. Assert the
final training loss is below the first, `result.rollout.pooled` is finite, and a save/load round trip
predicts identically. Nothing between "dataset built" and "artifact saved" is currently reachable
from `tests/`, and this is the whole point of the new `train()` shape.

---

## Part 5 — Delete the JAX path

- Delete `src/neuro/nn_training.py` and `src/neuro/prediction.py`.
- Delete the Part 1 shims: `_artifact_from_eqx`, and the temporary NumPy-vs-Equinox rollout test.
- `pyproject.toml`: drop `optax` from `dependencies`; drop `torchvision` from both extras and from
  `tool.uv.sources`. Keep the `cpu`/`cu128` extras, the two `[[tool.uv.index]]` blocks and the
  conflict declaration.
- `uv lock` and re-sync with `uv sync --extra cpu`.

**`jax`, `jaxlib` and `equinox` will still be installed** — they arrive transitively via
`tvboptim` -> `diffrax` -> `equinox` -> `jax`. That is expected. The goal is that no `neuro` module
imports them, not that they leave the lockfile. Verify with
`grep -rn "import jax\|import equinox\|import optax" src scripts tests` returning nothing.

`notebooks/sweep_analysis.py:209` imports `jax.numpy` inside a cell. It keeps working, since JAX is
still installed. Leave it.

---

## Part 6 — Batched rollout

The Optuna objective's inner loop is a double Python loop over trajectories x window starts in
`accumulate_rollout_errors` (`artifacts.py:38-65`), one `prime` + `rollout` call per window.

### Files

Edit `predictor/artifact.py`, `esn.py`, `artifacts.py`. Add tests.

### Contract

Add `rollout_many(states, u_futures) -> (B, steps, n_eeg_channels)` to both artifacts, and
`prime_many` alongside it. `accumulate_rollout_errors` stacks the whole `t0` grid for a trajectory
and makes one call per trajectory.

- **MLP**: the AR loop already carries a leading batch dim naturally — `y_window` becomes
  `(B, n_y, C)`, the matmuls become `x @ W.T + b`.
- **ESN**: `ESNPredictor.readout` and `teacher_step` use `np.r_[h, 1.0]` and `np.r_[z, v, 1.0]`,
  which are scalar-only. Batched forms concatenate along the last axis, and `w_res @ h` becomes
  `h @ w_res.T` (`w_res` is a `scipy.sparse.csr_matrix`; sparse-dense with `h` as `(B, N)` works, but
  check the orientation). `prime` runs a washout loop — batch it the same way.

Everything stays NumPy. Neither artifact may import torch.

### Verify

`rollout_many` == a loop of `rollout` to 1e-12, for both families, with and without PCA.
`test_esn.py` pins the current scalar behaviour — it must stay green untouched.

---

## Part 7 — Documentation

Rewrite `docs/nn_predictor_training.md`. It is the source-of-truth description of this pipeline and
is JAX/Equinox/Optax-specific throughout. Update: the framework, the single-`.npz` artifact, the
loss set (FC gone), the removed `w_fc` knob, the new `train()` shape and `TrainingResult`,
`du_sensitivity`, and the `device` field. The problem statement and indexing-convention sections
(sections 1 and 1.1) are framework-independent and should survive as they are.

Delete this file.

---

## Sequencing

| Part | Green at end | Blocks |
|---|---|---|
| 0 Delete migration doc | yes | — |
| 1 Artifact + data (JAX still trains) | yes | everything |
| 2 torch module + oracle + seam guard | yes | 3, 4 |
| 3 Losses + config | yes | 4 |
| 4 train() + scripts + smoke test | yes | 5 |
| 5 Delete JAX, prune deps | yes | — |
| 6 Batched rollout (MLP + ESN) | yes | — |
| 7 Docs | yes | — |

Parts 6 and 7 are independent of each other and of 5, but both want 4 landed first.

Do not start a part before the previous one is green. The value of this ordering is that a
regression is bisectable to a single named cause; a big-bang branch throws that away and there is no
golden training curve to fall back on.
