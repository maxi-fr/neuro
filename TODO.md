# TODO list

## Refactors

* MPC metric: solver iterations, pred error along horizon

* MPC solver: SQP with IPOPT as a fallback

* EEG sensors shouldnt run at 10kHz, find realistic value (maybe just same as MPC)

* move _spread_profile_from_trajectory to seizure

* Potentially: full rewrite
  * at least of the predictor part
  * move from jax to pytorch
  * list every feature and remove what is not necessary

## Other

* write chapter on Jansen-Rit model
* check out ROAST leadfield matrix: Ex8 is all zeros - that seems weird
* look into unified vocabulary - matt pocock skills

## Simulate package

## MPC package

* investigate if it could be used well here
* investigate what parts could maybe be simplified, etc.
* make fully yaml configurable - also cost functions, constraints, etc.
* compare to MPC implementation here
