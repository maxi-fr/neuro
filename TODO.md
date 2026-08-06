# TODO list

* write chapter on Jansen-Rit model
* Optuna sweep with closed-loop MPC performance as metric - this should show if the other losses are helpful
* Investigate average reciprocal leadfield (was not so insightful)
* Why can I not create a Leadfield matrix for stimulation?

* split the EEG forward operator out of connectome.py into eeg.py, so EEGMeasurement stops
  constructing a whole Connectome to read one matrix
* generally seperate stuff more: into own folders even

* For simulate package
  * Change logging to be fully customizable (no CoreLog)
  * Make allowed python version to be lower (doesnt have to be 3.13)
