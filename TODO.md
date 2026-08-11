# TODO list

* write chapter on Jansen-Rit model
* Optuna sweep with closed-loop MPC performance as metric - this should show if the other losses are helpful
* Investigate average reciprocal leadfield (was not so insightful)

* generally seperate stuff more: into own folders even

* For simulate package
  * potential bug in logging: `MPCControllerLog gained n_iter and capped (defaulted, so the warm-up path and the linear MPC are untouched). capped is Maximum_Iterations_Exceeded specifically, so success / capped / other-failure is now a clean trichotomy. I avoided logging the status string: simulate's logger sizes each buffer from the first value's dtype, so a <U15 first status would silently truncate longer ones.`
