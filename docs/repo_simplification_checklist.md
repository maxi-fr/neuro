# Repository simplification checklist

The repository audit identified the items below. Checked items are implemented
and verified; unchecked items remain open. The November 2026 Predictor families,
Loss comparisons, sparse MPC formulations and solver benchmarks stay supported.

## Approved cleanup

- [x] Remove obsolete Predictor aliases and helper reexports from
  `src/neuro/predictor/inference.py`, plus unused `forward()` and
  `with_provenance()` methods in `base.py`. Preserve consumed class exports and
  checkpoint loading.
- [x] Remove unused seizure convenience functions: `ez_pz_indices`,
  `SpreadProfile.recruited_by` and `spread_profile_from_states`. Keep the LFP
  measurement and recruitment interfaces used by analysis.
- [x] Remove the unused `comparison.format_cost_report` presentation helper;
  preserve numerical Cost decomposition and aggregation coverage.
- [x] Remove `OODIndex.global_distances`; preserve K-NN distances, calibrated
  percentiles and the OOD investigation workflow.
- [x] Resolve healthy/seizure regimes once at Plant construction, using the
  selected Connectome. Make `JansenRitParams.from_config` numeric-only and migrate
  callers of the old parameter shorthand. Verify gain and fixed-seed parity.
- [x] Require canonical `projection_E` in Field Projection loaders. Inventory
  retained assets and migrate legacy keys before deleting loader fallbacks.
  Preserve numeric arrays and projected Stimulation Drives. Remove the Yu-only
  potential-field converter interface; retain previously recorded `projection_V`
  arrays in existing assets.
- [x] Remove Dynamic Yu stimulation, its configuration schema, factory branch,
  dedicated tests and documentation describing it as supported. Keep ROAST,
  Analytical and Null stimulation.
- [x] Remove generic custom-loss training (`loss_fn` and arbitrary batch
  sequences). Use the configured Loss and `TrajectoryWindowDataset` interfaces;
  retain checkpoint eligibility, early stopping, standardizer and timing coverage.

## Interface and duplication improvements

- [ ] Give training Rollouts one explicit interface: call
  `rollout_standardized()` or the Predictor's callable interface, and let the
  inherited trajopt `rollout()` handle trajectories. Remove argument probing and
  dynamic Trainer dispatch after migrating callers.
- [ ] Consolidate applied-current and solver scoring from `comparison.py`,
  `mixed_integer_comparison.py`, `ood.py` and `run_view.py`. Use logged timestamps
  for Delivered Charge, preserve distinct failure policies and distinguish
  actual solve attempts from Warm-up Period and plan playback.
- [ ] Resolve the Plant's implicit stimulation default: requiring an explicit
  stimulation model or using Null stimulation would remove file-dependent
  ROAST-to-Null fallback behavior. This requires a behavior decision.
- [ ] Move the stimulation factory out of `stimulation/__init__.py` and make
  that file empty, updating consumers to import from the owning module.
- [ ] Remove the trajopt controller's write-only `_u_guess` and `_active_guess`
  bookkeeping; retain the real warm start owned by trajopt. CasADi's same-named
  fields are used and must stay.
- [ ] Remove or define the ignored `TrajOptMPCController.solve_state`
  `time_limit_s` argument, after deciding the supported solver-limit interface.

## Module splits

- [ ] Split `predictor/train.py` around preparation, fitting and scoring. Share
  repeated post-fit scoring within each prediction target while retaining the
  small training dispatch interface and all fit choices.
- [ ] Split `ood.py` into index calculations, prediction-error analysis,
  reporting and branch experiments; retain its active notebook consumers.
- [ ] Separate ensemble generation from scoring and archive access in
  `ensembles.py`; preserve manifests, stored shapes and cache reload behavior.

## Historical workflows requiring a decision

- [ ] Decide whether to retain `evaluate_observable_free_run`. Current training
  uses `evaluate_spectral_free_run`, but an archived Sweep script still calls
  the former; migrate that script before removing the helper.
- [ ] Decide whether archived MINLP benchmarks must remain runnable before
  removing the optional Gurobi/PySCIPOpt/Pyomo dependencies. Experiment-local
  `uv run --with` dependencies are an alternative to the shared `minlp` group.

## Verification

- [x] Establish green focused tests before changing retained behavior.
- [x] Run focused tests after each approved cleanup, preserving meaningful
  assertions for the retained interfaces.
- [x] Run the full test suite and lint, formatting and type checks; record any
  pre-existing failures separately from regressions.

## Implementation notes

- The retained `data/roast_field_projection_3d_3ch.mat` asset used `leadfield_3d`.
  Its HDF5 dataset link was renamed to `projection_E`; numeric datasets, shapes
  and dtypes were checked unchanged. Its electric field matches the existing
  three-channel NPZ, and checked montage Stimulation Drives are unchanged.
- Regime normalization now keeps the requested regime instead of expanding its
  regional gains early. Existing numeric gain configurations remain accepted.
  Provenance hashing is unchanged, so the different normalized representation
  can produce a different fingerprint from a historical expanded snapshot.

## Verification results

- Full suite: `uv run pytest -n 4` — 943 passed, 8 skipped. The skips are seven
  opt-in slow tests and one test requiring an unavailable saved run.
- Whole-repository Ruff lint, changed-file formatting and type checks, checklist
  Markdown lint, and `git diff --check` pass.
- Whole-repository type checking still reports six pre-existing errors in
  `src/neuro/predictor/dmd.py` and `tests/test_dmd.py`.
- Whole-repository formatting still flags three unchanged thesis figure scripts:
  `stft_geometry.py`, `stft_resolution.py` and `receding_horizon.py`.
