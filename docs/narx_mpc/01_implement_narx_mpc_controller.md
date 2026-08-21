# 01 — Re-introduce NARX Nonlinear MPC Controller

**What to build:** Restore the original output-lifted NARX MPC implementation from historical commit `fb163e2` (prior to its deletion in `cc63909`) into a dedicated file (./src/neuro/control/narx_mpc.py), adapting it surgically to run against today's modern architecture:

- Utilize the modular (./src/neuro/control/solvers.py) backends (`IpoptMPCSolver`, `SqpMPCSolver`, `SqpFallbackMPCSolver`).
- Align state absorption and priming with [`NNSymbolicModel`](./src/neuro/nn_predictor_casadi.py).
- Emit [`MPCControllerLog`](./src/neuro/control/nonlinear_mpc.py) per step.
- Support modern cost functions: $w_y$ (EEG power), $w_{y,\text{terminal}}$, $w_u$ (effort), $w_{u,\text{L1}}$ (sparse stimulation), and $w_{\text{psd}}$ (spectral envelope hinge penalty).

**Blocked by:** None — can start immediately.

## Acceptance criteria

- [x] `NarxMPCController` is restored in (./src/neuro/control/narx_mpc.py) based on the original implementation from commit `fb163e2`.
- [x] The symbolic output-lifted NLP formulation (`NarxMPCNlp.build`) lifts decision variables $(u_0 \dots u_{H-1})$ and $(y_0 \dots y_{H-1})$ (plus L1 epigraph slacks when active), enforcing output defect equalities $y_k - \phi(y_{\text{win}}, u_{\text{win}}) = 0$ alongside Kirchhoff Current Law $\sum u_k = 0$.
- [x] Solver integration leverages [`neuro.control.solvers`](./src/neuro/control/solvers.py) supporting `solver="sqp_fallback"` (default), `solver="sqp"`, and `solver="ipopt"`.
- [x] Objective functions support $w_y$ (EEG power, normalized by horizon $1/H$), $w_{y,\text{terminal}}$, $w_u$ (effort), $w_{u,\text{L1}}$ (sparse effort), and $w_{\text{psd}}$ (spectral envelope hinge penalty).
- [x] State management cleanly absorbs incoming measurements via `model.absorb` and maintains warm-up until $n_y$ samples are buffered.
- [x] `validate_simulation_config` in (./src/neuro/validation.py) recognizes `NarxMPCController` as a predictive controller.
- [x] Historical test suite from commit `fb163e2` is restored and adapted in (./tests/test_narx_mpc_controller.py), asserting mathematical equivalence to single shooting, bounds compliance, KCL adherence, L1 sparsity, spectral penalty evaluation, solver fallback on iteration cap or error, non-MLP model rejection, and full closed-loop `Simulation` execution.
- [x] All pre-commit quality gates (`ruff check`, `ruff format`, `ty check`, `pytest`) pass cleanly.
