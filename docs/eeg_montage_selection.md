# EEG Montage Selection

Why the seizure-control pipeline observes EEG through a **25-channel montage** rather than
TVB's full 62-channel array, and how that montage was chosen. The montage is the set of scalp
electrodes the plant is measured through (`EEGMeasurement.selected_channels`); it fixes the output
dimension `C = n_channels` seen by the NN predictor and the MPC.

Source of truth:

- Measurement / channel subsetting & forward operator (EEG gain `L`): [`src/neuro/eeg.py`](../src/neuro/eeg.py)
- Wired into: [`configs/simulation/jansen_rit_seizure_excited.yaml`](../configs/simulation/jansen_rit_seizure_excited.yaml)
  (data generation) and closed-loop configs under [`configs/simulation/meeting_seven/`](../configs/simulation/meeting_seven/)
  (e.g., `selected_mpc.yaml`).

---

## 1. Motivation

The full EEG forward operator maps the 76 Jansen–Rit regions to **62 scalp channels**
(`L` has shape `(62, 76)`). 62 outputs is more state than the MPC needs to carry: the predictor's
per-step EEG vector and the MPC's tracked output both scale with `C`. Reducing `C` shrinks the
predictor and the control problem, provided the retained channels still *see* the seizure.

TVB ships no small scalp montage — the only EEG datasets are 62- and 65-channel
(`eeg_unitvector_62` / `projection_eeg_62_surface_16k` and `eeg_brainstorm_65` /
`projection_eeg_65_surface_16k`; the rest are MEG/sEEG). So the reduction is done by
**subselecting rows of the existing 62-channel gain**, not by loading a different layout.

## 2. The seizure focus is left mesial-temporal

The seizure is induced by raising the excitatory gain `A` above its baseline of 3.25 on a handful
of regions (see the `params.A` vector in the seizure configs):

| index | region | `A`  | role |
| :---: | ------ | :--: | ---- |
| 40 | `lAMYG` (amygdala)         | 3.6 | epileptogenic zone |
| 47 | `lHC` (hippocampus)        | 3.6 | epileptogenic zone |
| 62 | `lPHC` (parahippocampal)   | 3.6 | epileptogenic zone |
| 69 | `lTCI` (inferior temporal) | 3.4 | propagation zone |
| 72 | `lTCV` (ventral temporal)  | 3.4 | propagation zone |

This is a **left mesial temporal lobe** focus — the configuration of mesial temporal lobe epilepsy
(MTLE). MTLE is the classic blind spot of the standard international 10–20 montage: deep, basal
sources project weakly onto the scalp and mostly onto the **inferior-temporal** electrodes, which
the 19-channel 10–20 set does not include. Clinically this is why an inferior/anterior-temporal
chain is added for temporal-lobe cases.

## 3. Montage selection metric

Each region `r` contributes column `L[:, r]` to the scalp signal. For a candidate electrode subset
`S`, define the **EZ lead-field energy retained** as the Frobenius norm of the selected rows over
the EZ/PZ columns, relative to all 62 rows:

```text
energy(S) = ‖L[S, EZ]‖_F / ‖L[:, EZ]‖_F ,   EZ = {40, 47, 62, 69, 72}
```

This is the fraction of the focus's scalp footprint the montage keeps (1.0 = lossless, all 62
channels). Reproduce with:

```python
import numpy as np
from neuro.connectome import Connectome

c = Connectome.from_config({"speed": 50.0})
gain, chan = c.gain, list(c.channel_labels)
ez = [40, 47, 62, 69, 72]

def energy(sel):
    idx = [chan.index(e) for e in sel]
    return np.linalg.norm(gain[idx][:, ez]) / np.linalg.norm(gain[:, ez])
```

## 4. Results

The generic 19-channel 10–20 set keeps only ~half of the focus's signal, because the single
strongest electrodes for the deep regions (`TP9` for `lPHC`/`lTCI`) are **not** in it. Adding the
left inferior-temporal chain recovers most of the loss:

| montage | # ch | added to 10–20 | EZ energy |
| ------- | :--: | -------------- | :-------: |
| 10–20 (generic)          | 19 | —                            | 51% |
| + minimal                | 21 | `TP9, TP7`                   | 75% |
| + left ring              | 23 | `TP9, TP7, CP5, FC5`         | 80% |
| **+ left ring + posterior** | **25** | `TP9, TP7, CP5, FC5, P5, PO7` | **84%** |
| + right homologs         | 27 | left ring + `TP10, TP8, CP6, FC6` | 80% |

The focus is unilateral-left, so the right-hemisphere homologs add channels without much EZ gain.
The **25-channel** option is the chosen operating point: near-maximal EZ coverage at a channel
count that is still a ~2.5× reduction from 62.

### Chosen montage

The 19 standard 10–20 electrodes plus the six left inferior-temporal additions:

```yaml
selected_channels: [Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8,
                    P7, P3, Pz, P4, P8, O1, O2, TP9, TP7, CP5, FC5, P5, PO7]
```

## 5. Consistency requirement

`C = n_channels` is **not** a config field — the training pipeline infers it from the simulated
data (`n_channels = y.shape[1]`). The montage must therefore be identical everywhere the EEG is
produced or consumed, or the dimensions will not line up:

1. **Data generation** — `EEGMeasurement.selected_channels` in the data-gen config
   (`jansen_rit_seizure_excited.yaml`, propagated to every trial by
   [`scripts/generate_experiment.py`](../scripts/generate_experiment.py)) fixes the training data at
   `(T, 25)`.
2. **Predictor** — retraining picks up `n_channels = 25` from that data automatically; the NN
   config (`n_y`, `n_u`) is unaffected (those are history lengths, not channel counts).
3. **Closed-loop plant** — the MPC config's `EEGMeasurement` must use the *same* 25-channel
   `selected_channels` so the plant measurement matches the retrained predictor's input.

Changing the montage means regenerating the training data **and** retraining the predictor; an
artifact trained on a different channel count is dimension-incompatible with the plant.

## 6. Caveat — energy is necessary, not sufficient

Retained lead-field energy bounds how much of the seizure *can* reach the montage; it does not
guarantee the seizure state is recoverable. Even the full 62-channel scalp gain is a weak window
into amygdala/hippocampus (deep sources attenuate through the skull), and the EZ signal still has to
be disentangled from other regions hitting the same electrodes. Prior reduced-model work found that
reconstruction-oriented channel picks tend to under-serve the deep EZ, and that a small linear
montage loses a large fraction of the (genuinely high-dimensional) seizure EEG variance. The
25-channel montage is chosen to *maximise* the focus's scalp footprint under a channel budget, not
to make the deep sources fully observable.
