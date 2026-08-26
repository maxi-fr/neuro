# Observable-Space Predictor

## Problem Statement

The controller forecasts the raw EEG waveform with an autoregressive MLP and then reduces that
forecast into observable space (trailing mean-square power, log-spectrogram) to score its training
Losses and its MPC Costs. The waveform is an intermediate: every Loss and Cost that matters
operates on windowed reductions of it, not on the waveform itself. Training the waveform Predictor
against observable-space Losses works, but it keeps a full differentiable STFT reduction inside the
training Loss and a second, duplicated reduction inside the MPC Cost, and it makes the spectral
hinge reach into the model for a decode and for an extra transition step.

Build a Predictor that predicts the Observable directly, one Frame per step, and let the MPC read
the predicted Frames. This deletes the waveform-to-spectrum reduction from the observable training
Loss and from the observable Cost, and makes every Cost a pure function of the state it is handed.

## Solution

An observable-space Predictor: the autoregressive MLP core, re-aimed so one step advances one Frame
of the STFT log-power Observable. Training scores it with a curriculum MSE directly on predicted
standardized log-power Frames, with no differentiable STFT. At runtime a new Estimator reduces the
raw EEG stream into log-power Frames at the hop rate; the controller steps the Predictor one Frame
at a time and holds the Control Current over the hop; the MPC Cost reads predicted Frames straight
off the state. The waveform Predictor stays, because the autoregressive MLP core is indifferent to
whether its state vector carries raw EEG or log-power Frames. The waveform Costs are refactored onto
the same model-free contract as part of this work.

## Implementation Decisions

1. **The Observable is Frame-Kernel-smoothed STFT log-power, DC-free.** One Frame is the per-
   (channel, bin) log-power of a Segment, reduced with the shared STFT geometry: periodic Hann,
   density-scaled, no per-segment detrend, DC excluded, band slice, bin pooling, Frame Kernel
   applied to power before the log, then the log floor. The Frame Kernel is part of the Observable
   rather than a Loss-only smoother, so a Frame's sample support is
   ``(kernel_width - 1) * hop + segment``.

2. **One canonical reduction function.** All Observable reduction math lives as a pure NumPy
   function in the spectral module, parameterized by the Observable geometry. The offline dataset
   reduction, the online Estimator and the healthy-envelope builder call that one function, so the
   three agree to machine precision. The torch and jax twins that remain for the waveform path are
   parity-tested against it rather than being independent conventions.

3. **Decoupled, pure autoregressive MLP core.** The torch module and its jax runtime adapter are a
   single vector model advancing one position per call, with a residual skip fitting the one-step
   delta over ``n_outputs`` dimensions: ``n_channels`` for the waveform kind,
   ``n_channels * n_values`` for the observable kind. The core is agnostic to whether a position is
   a sample or a Frame. The two runtime adapters differ only in what one position means and in the
   geometry they carry.

4. **Frame-rate MPC, with the control-support rule.** The model steps one hop per call; the
   controller's update period is the hop; the Control Current is held constant over it. The Control
   Horizon is a fixed integer number of Frames, so the physical lookahead is
   ``(horizon - 1) * hop + segment`` samples. One model step takes the single control held over that
   hop, and a Frame's dependence on the earlier controls that shaped its Segment comes from the
   past-control window. Validation therefore enforces
   ``n_u >= kernel_width - 1 + ceil(segment / hop)``, without which the model cannot see part of the
   control that produced the Frame it predicts.

5. **Standardization lives in log-power space, per output.** The observable model trains and steps
   in standardized log-power, with center and scale fitted per output (channel x bin) on the
   training Frames. The standardizer is checkpointed as module buffers and is invisible at the
   runtime boundary: State Absorption takes raw Frames and the free run returns raw Frames.

6. **The Estimator owns the reduction.** A new multi-rate Estimator low-passes at plant rate,
   decimates, buffers Segments and emits raw log-power Frames at the hop rate by calling the
   canonical reduction. It carries no learned parameters. Its latency before the first Frame is
   ``segment + (kernel_width - 1) * hop`` decimated samples plus the low-pass group delay; it
   returns an unprimed NaN Frame until then. Between hops it holds the last Frame, so correctness
   depends on the controller sampling it exactly once per hop -- see decision 14. The waveform
   Estimator is unchanged.

7. **Costs are model-free.** Every MPC Cost is constructed from geometry, the healthy envelope and
   center/scale arrays at build time. No Cost holds a model instance or calls a model method. This
   is a refactor of the existing waveform Costs as much as a constraint on the new one: the waveform
   spectral hinge currently decodes through the model and recovers the terminal output with an extra
   transition step, and both go away. The refactored hinge scores the Frames the stage trajectory
   carries; the terminal knot's contribution is expressed as an explicit terminal Cost rather than
   synthesized by stepping the model.

8. **The output decode is removed from the Predictor.** The state carries the Observable, so a Cost
   slices the state and applies a decode baked from center/scale at build time. The decode method
   leaves the runtime protocol and the concrete adapters entirely.

9. **The Predictor seam stays two minimal mirrored protocols.** A training-side protocol and a
   runtime-side protocol bridge through the numpy exchange checkpoint, with persistence declared
   once. The runtime surface is the one-step transition, State Absorption, the readiness check, the
   unprimed initial state, the batched free run and the Priming Steps count, plus the checkpoint
   round-trip. No output decode.

10. **State Absorption is in Frames.** The observable model absorbs one raw Frame plus the control
    held over that hop. Priming Steps are counted in Frames, ``max(n_y, n_u)``. The Warm-up Period a
    closed-loop run needs is the Estimator's latency plus ``priming_steps * hop`` decimated samples.

11. **Loss configuration stays a polymorphic bag of terms.** Losses remain composable YAML entries
    with no discriminator block. On the observable path the curriculum MSE scores predicted
    standardized Frames directly, and the Loss context's rate is the Frame rate rather than the
    sample rate, so a Span declared in seconds resolves to Frames. Validation rejects the Loss terms
    that carry their own reduction (the STFT and mean-square terms) on the observable kind, since a
    differentiable STFT of log-power Frames is meaningless.

12. **The observable Trainer reports a validation loss, and the Ridge arm generalizes.** An
    observable Sweep trial ranks on validation Frame MSE or on the sweep-level closed-loop
    objective, and objective validation runs against the observable candidate set for the observable
    kind. The Ridge-Fittable design and readout-install path widens from ``n_channels`` to
    ``n_outputs``, so a depth-0 observable MLP is Ridge-Fittable and the closed-form arm serves both
    kinds.

13. **The healthy envelope is built at the model's Frame grid.** The envelope builder takes the
    Observable geometry and quantiles over already-banded, already-pooled, already-smoothed Frames,
    rather than writing a per-bin envelope that the Cost pools afterwards. Pooling a per-bin
    quantile is not the quantile of pooled power: averaging bins shrinks the variance, so a pooled
    envelope is systematically too permissive and the hinge under-fires. The envelope records the
    geometry it was measured at.

14. **All geometry is config, is checkpointed, and is cross-validated.** Segment, hop, band,
    pooling, Frame Kernel, MLP shape and training knobs are config values. The Observable geometry
    is also written into the checkpoint, so validation compares the runtime config against what the
    model was trained with rather than only against other config. The rules are: the control-support
    inequality of decision 4; controller dt equal to ``hop * downsample * plant_dt``, so exactly one
    fresh Frame is absorbed per tick; estimator dt equal to plant dt; band and pooling against the
    envelope's recorded geometry; channel count against the envelope; standardizer length against
    the output width; sampling rate against the envelope; curriculum Span against the Frames that
    Span holds; sweep key overlap and objective validity.

15. **The Cost menu.** The observable problem carries a one-sided log-power hinge against the
    healthy envelope, L1 control, quadratic control, control box bounds and the Kirchhoff equality,
    all model-free and YAML-configurable. There is no two-sided quadratic state Cost: the envelope
    is an upper quantile, and tracking it two-sided would drive healthy-typical power up toward it.

## Testing Decisions

A good test asserts external behavior at a seam a caller actually uses, on tiny synthetic modules
with random weights. It does not reach into private state and does not assert trained accuracy.

- **Estimator against the offline reduction.** Fed one raw trajectory, the Estimator emits exactly
  the Frames the training pipeline reduces for targets -- same convention, band, pooling, Frame
  Kernel, log floor and hop grid -- and the first Frame appears at the latency decision 6 states.
  Prior art: the filtering and spectral tests.
- **Predictor protocol parity, torch against jax.** The checkpoint round-trips so the jax adapter's
  free run equals the torch rollout to floating-point tolerance, for both kinds; the runtime surface
  behaves; the output decode is absent from the protocol. Prior art: the predictor module, predictor
  protocol, batched rollout and checkpoint reader tests.
- **Model-free Cost.** The one-sided hinge, fed predicted Frames, matches a NumPy reference of the
  same computation and constructs without a model instance. The same seam covers the refactored
  waveform hinge, whose value must not move under the refactor. Prior art: the cost, metric-loss and
  MPC tests.
- **MPC problem and controller loop.** The observable problem builder wires the hinge, box bounds,
  Kirchhoff and a Frame-count Control Horizon; the controller absorbs Frames, holds zero control
  through Priming, then emits finite controls. Prior art: the MPC and control tests.
- **Config validation.** The cross-consistency rules reject invalid configs and accept valid ones,
  including the control-support inequality and the controller/hop alignment. Prior art: the config
  and example-config tests.
- **Training, Loss and checkpoint round-trip.** The curriculum MSE over Frames ramps its trusted
  prefix correctly; training produces a model whose save and load round-trip weights, geometry and
  standardizer, for both kinds and for both the gradient-descent and Ridge arms. Prior art: the
  predictor training, unified trainer, ridge trainer and predictor loss tests.

## Out of Scope

- Designing the Sweep itself (objective selection, trial ordering, result ranking). Only that
  geometry, shape and training knobs are config values, are cross-validated, and have an objective
  the observable Trainer can report.
- The waveform Estimator refactor that would move State Absorption and Priming into a shared lift so
  the model is a pure state transition. Noted in TODO.md, deferred.
- Changing the two-engine split: torch trains, jax and trajopt run, the numpy checkpoint bridges.
- Closed-loop performance benchmarks and Seizure Burden gates. Performance stays the job of probe
  scripts and closed-loop evaluation.
- Removing the waveform Predictor. It stays alongside the observable Predictor. Refactoring it onto
  the model-free Cost contract is in scope; deleting it is not.

## Further Notes

- **The intervention changes, not only the model.** Frame-rate MPC drops the actuation update from
  the sample grid to the hop grid, and the stimulus becomes a Control Current held constant for a
  full hop. At the deployed geometry that is roughly 50 Hz down to a few Hz. This is a deliberate
  answer to the open question in TODO.md about running the controller slower than the Predictor, and
  it should be read as a change to the experiment rather than a refactor.
- **The Observable is phase-blind.** Log-power discards phase, so any phase-locked mechanism such as
  cycle cancellation is outside what this Predictor can represent. If such a mechanism turns out to
  matter, the answer is a different Observable, not a different model.
- **Estimator degrees of freedom set the noise floor.** An unpooled, unsmoothed periodogram bin is
  chi-squared with two degrees of freedom, so its log has a standard deviation near 1.28 nats and a
  mean about 0.58 nats below the log of the true power. Bin pooling and the Frame Kernel are what
  buy the degrees of freedom back. A geometry with too few will leave the curriculum MSE dominated
  by estimator noise and the model will learn to predict the mean. This is the main thing a geometry
  Sweep should be watched for.
- **The geometry-specific envelope constrains the Sweep.** Because the envelope is quantiled at the
  model's Frame grid, a Sweep over segment, hop, band, pooling or Frame Kernel needs an envelope
  rebuilt per trial. Sweeping training knobs alone does not.
- **DC power is deliberately excluded.** Reintroducing it is a separate Observable or a Cost change,
  not a model change.
- **The observable model is not a separate architecture.** It is the autoregressive MLP with a wider
  per-step output and Frame granularity.
- **Glossary.** Estimator is load-bearing throughout this spec but is not yet a CONTEXT.md term; it
  appears only in the Predictor entry's avoid-list. Add it while doing this work.
- **Done means:** both kinds train, checkpoint and round-trip; the Estimator matches the offline
  reduction to machine precision; the refactored waveform hinge reports an unchanged value; a
  closed-loop run with the observable problem primes and emits finite controls; and the observable
  Predictor beats a persistence baseline on held-out Frame MSE. Closed-loop suppression performance
  is not a gate here.
