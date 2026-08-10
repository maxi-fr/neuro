# Operations notes — the `roast_3d` plant

**Date:** 2026-08-10
**Scope:** the montage, the artifacts and commands, the traps, and what is still not done.
Physics and benchmark results: [`tes_field_geometry.md`](tes_field_geometry.md).
Why the MPC fails and what to change: [`mpc_next_steps.md`](mpc_next_steps.md).

## The montage

`[TP9, CP5, Ex8]` with **`[+2.0, 0.0, -2.0]` mA**: positive current at TP9, returning at Ex8.

`CP5` sits at zero current deliberately. Its leadfield row is ~4x weaker than TP9's and points
mostly at *contralateral* `rTCI`, so it contributes nothing to suppression — but it stays in the
montage as the second control DOF the MPC and the L1-sparsity work need (`n_controls >= 3`).

An exhaustive scan of the whole reachable 2-DOF space at +/-4 mA on every electrode confirms
nothing better exists with these three electrodes.

## Gotchas that cost time

1. **16 GB RAM ceiling.** Seven parallel IPOPT MPC workers OOM-kill; use 3. (That ceiling was set
   under multiple shooting at ~1.4 GB a worker; single shooting holds ~400-500 MB, so 7 may well
   fit — untested.) Seven parallel qpOASES-dense (linear) workers are fine. Killed pools leave
   ~1.4 GB `multiprocessing.spawn` orphans behind — check for those before blaming anything else.
2. **`shooting_depth` is the segment size, not a mode flag.** `1` is full multiple shooting (a
   state root at every step) and is intractable here; `>= horizon` is single shooting. Leave it
   unset — it defaults to `horizon` — or set it to the horizon. Getting this wrong cost an 8.4 h
   run that produced nothing and a retracted conclusion.
3. **`n_steps: null` in the predictor configs.** Truncation silently discards the seizure, which is
   the part of the trajectory the identification needs. Keep it `null` (whole trajectory).
4. **Wait on process exit, not on file size.** Watching for `sim_024.npz` to shrink fired while
   the export was still *writing* it, which started the split against in-flight files (one
   truncated copy, one orphaned temp; both recovered). Poll `wmic process ... get commandline`.
5. **`analytical` + `EX8` is miscalibrated.** Every config shipped `EX8` while the docs were tuned
   for `EX_NECK`; the commanded `[-0.5, -0.5, +1.0]` drives `lHC` **anodally** and ignites the
   contralateral hemisphere (L=3, **R=30**). The pre-`roast_3d` montage results never reproduced
   from the committed configs. Moving to `roast_3d`, referenced to Ex8 by construction, resolves it.

## Artifacts and data

| path | what |
| --- | --- |
| `data/experiment_excited_roast/{train,test}` | 22 + 3 x 20 s, `roast_3d` plant, RAS `[10,50,200]` ms holds, verified |
| `data/experiment_excited/` | the **old** `analytical` set, 6.8 GB, kept — delete when confident |
| `artifacts/roast/linear_full/model.eqx` | depth-0 predictor (trained; MPC 0/7) |
| `artifacts/roast/nonlinear_full/model.eqx` | depth-1 softplus, no PCA (trained; MPC 0/7) |
| `configs/nn_predictor/roast/{linear,nonlinear}_full.yaml` | identification configs |
| `configs/simulation/roast/{linear,nonlinear}_full_mpc.yaml` | closed-loop configs |
| `configs/simulation/threshold_control.yaml` | the working 6/7 result |

Regenerate the dataset with:

```bash
uv run python scripts/run_simulation.py configs/simulation/experiment_excited.yaml \
  --output-dir data/experiment_excited_roast --compress
```

then split 22/3. Note `run_simulation.py` strips the `x` state array from every npz only *after*
all trials finish, so the run is not done when the last file appears.

## Not done

- **The 62-channel ROAST leadfield.** The overnight run died on `O1` with a zero-byte `_e.pos`;
  that turned out to be **transient** (it re-solved cleanly to 33.7 MB), so the full run is
  viable: ~20 h, ~21 GB, ~53 electrodes left with 9 already cached.

  ```bash
  uv run python scripts/run_roast_matlab.py --output-file data/roast_leadfield_3d_full.mat
  ```

  It would not change the depth conclusion — no scalp montage exceeds ~0.2 V/m at mesial temporal
  — but it would allow a real montage *search* rather than accepting the given three. It also
  emits **`leadfield_V`**, which `yu_dynamic` requires and the current NPZ lacks; that is probably
  the stronger reason to run it.
- **The plant's `stimulation` default** is still `_NullConfig`, not `roast_3d`. `jansen_rit_baseline`,
  `jansen_rit_seizure` and `experiment` omit a stimulation block and pair `ZeroController` with
  `n_u: 1`; defaulting them to a 3-control model breaks them. "No stimulation block" therefore
  still means no stimulation, and `roast_3d` is the default *model of record* — every config that
  stimulates uses it. Flipping the plant default means fixing those three configs first.
- **Duty cycle.** The working threshold policy runs at 95 %, not the 20 % of the `analytical` era.
  That is inherent to a propagation block without lasting effects; implementing Yu §2.4 is what
  would change it.
