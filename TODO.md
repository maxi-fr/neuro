# TODO list

## Refactors

* MPC metric: solver iterations, pred error along horizon

* MPC solver: SQP with IPOPT as a fallback

* EEG sensors shouldnt run at 10kHz, find realistic value (maybe just same as MPC)

* move _spread_profile_from_trajectory to seizure

* training 0% CPU usage? (No i think not, 40-50% for the python)

### Efficiency

* possible to remove torch.cat from AutoregressiveMLP.forward? for efficiency?
* training in float32?
* multiple shooting doesnt work, single shooting does - but takes many solver iterations. Maybe there is a middle ground. Investigate
* multiple welch spectrums after another can surely be optimized

## Other

* write chapter on Jansen-Rit model
* look into unified vocabulary - matt pocock skills

## Simulate package

## MPC package

* investigate if it could be used well here
* investigate what parts could maybe be simplified, etc.
* make fully yaml configurable - also cost functions, constraints, etc.
* compare to MPC implementation here

## python-project-template

* Needs updating with the stuff from here
