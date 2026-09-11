# TODO list

* Test cost functions with JR model as the MPC predictor!!!

* get a clear overview over the predictors and their hyperparameters!!
  * clean up all the old config files etc.Run

* figure out font typography etc.

* don't discard frequency 0Hz bin, could hold valuable information

## Refactors

* potential estimator refactor: move State Absorption/Priming (the "lift") out of the model into the estimator, so the model is pure `x_{k+1} = f(x_k, u_k)`. The lift would become a shared function used by both the runtime estimator and the evaluation free_run. Deferred for now: the model owns the window, the estimator emits one native measurement (y_k / o_k).

## Not urgent

* MPC metric: solver iterations, pred error along horizon

* EEG sensors shouldnt run at 10kHz, find realistic value (maybe just same as MPC)

* investigate running controller slower than predictor model. Less decision variables but still prediction model with high resolution
  * Idea: dynamics faster than input can affect

* reservoir computer w. hopf nodes

### Efficiency

* possible to remove torch.cat from AutoregressiveMLP.forward? for efficiency?
  > in other branch: perf/predictor-rollout-optimization

* Direct GPU Vectorized Slicing for `TrajectoryWindowDataset`:
  * Prototype benchmarked in `scratch/benchmark_slicing.py`.
  * Packs continuous standardized trajectories directly into GPU VRAM (`cuda:0`) and gathers mini-batches via strided tensor indexing without host-to-device transfers or Python item loops.
  * Delivers 450k+ samples/s data throughput and an additional ~7–16% end-to-end training speedup.
  * Trade-off: requires the continuous dataset to fit in GPU VRAM (takes ~334 MB for 800 trajectories of 8s, but scales with dataset duration/trials). Best suited for massive sweeps where dataset fits comfortably in VRAM.

## Other

## Simulate package

* allow for cross component config validation (like i've done here)

## MPC package (trajopt)

* make fully yaml configurable - also cost functions, constraints, etc.
* compare to MPC implementation here
* Add output function y = g(x). Would simplify cost fucntions etc.

## python-project-template

* Needs updating with the stuff from here
