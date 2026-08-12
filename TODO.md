# TODO list

## Refactors

* how does training data generation differ between MLP and ESN?

## Other

* write chapter on Jansen-Rit model
* check out ROAST leadfield matrix: Ex8 is all zeros - that seems weird

## Simulate package

* potential bug in logging: `MPCControllerLog gained n_iter and capped (defaulted, so the warm-up path and the linear MPC are untouched). capped is Maximum_Iterations_Exceeded specifically, so success / capped / other-failure is now a clean trichotomy. I avoided logging the status string: simulate's logger sizes each buffer from the first value's dtype, so a <U15 first status would silently truncate longer ones.`
