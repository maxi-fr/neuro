---
name: experiment-packaging
description: Use this skill when designing, executing, or concluding an experiment, model comparison, hyperparameter sweep(s), or benchmark in the repository.
---

# Experiment Organization and Packaging Runbook

This skill defines the repository standard for designing, running, and concluding experiments, benchmarks, and model comparisons. The goal is complete reproducibility, zero contamination of canonical configurations, and zero stray files left across the repository.

---

## 1. Zero-Contamination Boundaries

Every directory in the repository has a single, strictly enforced role:

| Directory | Role & Policy | Prohibited Contents |
| :--- | :--- | :--- |
| `artifacts/<experiment>/` | **Permanent Home**: Self-contained directory holding all configs, models, plots, and `report.md`. | Ephemeral temp files or multi-gigabyte raw simulation rollouts. |
| `configs/` | **Production Only**: Only for base simulation environments (e.g. `uncontrolled.yaml`, `threshold_control.yaml`, `jansen_rit_oracle_mpc.yaml`, `healthy_reference.yaml`). | **NEVER** put trial variants, comparison suites, or sweep templates here. |
| `docs/` | **Permanent Specifications & ADRs**: Domain biology, physics, equations, and architecture decisions. | **NEVER** put experiment summaries, sweep results, or benchmark notes here. |
| `results/` | **Ephemeral Staging Ground**: High-volume intermediate rollouts during active execution. Gitignored. | **NEVER** leave files in `results/` after concluding an experiment. Must be purged. |
| `data/` | **Shared Static Inputs**: Static connectomes, head models, reference spectra. | Experiment outputs or trained models. |

---

## 2. Experiment Directory Structure

Every experiment must live under `artifacts/<experiment_name>/`.

### Self-Contained Configs (Immunity to Upstream Changes)

Never point `comparison_manifest.yaml` or experiment scripts to configs in `configs/simulation/` or `configs/nn_predictor/`.
Always copy the baseline config into `artifacts/<experiment_name>/base_simulation.yaml` (or `base_model.yaml`). This guarantees that future modifications to production defaults cannot invalidate or alter past experiment results.

---

## 3. Report Structure (`report.md`)

Every experiment must contain a `report.md` created from containing the following sections:

1. `## Motivation`
2. `## Experimental Setup`
3. `## Results`
4. `## Visualizations` (optional)
5. `## Actionable Recommendation / Decision`

---

## 4. Lifecycle Protocol

### Phase 1: Initialization (Step 0)

1. Create `artifacts/<experiment_name>/`.
2. Snapshot any base configs into the folder.
3. Create variant directories with their local `config.yaml` files.
4. If comparing multiple arms, set up `comparison_manifest.yaml` pointing only to local files within the experiment folder.

### Phase 2: Execution & Staging

- Direct output files (checkpoints, metrics, plots) straight into their respective `artifacts/<experiment_name>/<variant>/` directory whenever feasible.
- If an experiment produces high-volume intermediate simulations, stage them under `results/<experiment_name>/`.

### Phase 3: Packaging & Staging Cleanup

- Move final model weights (`model.npz`), summary statistics, and plots into `artifacts/<experiment_name>/`.
- Delete the staging directory `results/<experiment_name>/`. `results/` must be left clean.
- Write `report.md` summarizing the outcomes.

### Phase 4: Validation Gate

Make sure that no configs or other artifacts pertaining to the experiment remain outside of the folder.
