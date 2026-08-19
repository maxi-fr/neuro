# TODO list

## Refactors

* MPC metric: solver iterations, pred error along horizon

* MPC solver: SQP with IPOPT as a fallback
  > see [sqp ipopt investigation](knowledge-base/Notes/sqp_ipopt_fallback_benchmark.md)
  * Follow-up: Bring back the narx version and see if partial MS works better

* split up control.py into multiple files (make it a subpackage or smt)

* EEG sensors shouldnt run at 10kHz, find realistic value (maybe just same as MPC)

* investigate running controller slower than predictor model. Less decision variables but still prediction model with high resolution
  * Idea: dynamics faster than input can affect

* what about the effect of filter on spectrum losses/cost

* what about predicting the metric directly?

### Efficiency

* possible to remove torch.cat from AutoregressiveMLP.forward? for efficiency?
* training in float32?
* multiple shooting doesnt work, single shooting does - but takes many solver iterations. Maybe there is a middle ground. Investigate [Done]
* multiple welch spectrums after another can surely be optimized [in_progress]

## Other

* write chapter on Jansen-Rit model
* look into unified vocabulary - matt pocock skills

## Simulate package

* allow for cross component config validation (like i've done here)

## MPC package

* investigate if it could be used well here
* investigate what parts could maybe be simplified, etc.
* make fully yaml configurable - also cost functions, constraints, etc.
* compare to MPC implementation here

## python-project-template

* Needs updating with the stuff from here
