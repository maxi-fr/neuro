# 04 — Delete the CasADi MPC stack

**What to build:** Nothing new — this is the contraction step. Once every simulation config that
needs an MPC controller can run on the trajopt-based controller (ticket 03), remove the CasADi
stack it replaces: the hand-built NLP builder, the three solver wrappers, and the CasADi-specific
model bridges. Two of those bridges (`nn_predictor_casadi.py`, `observable_casadi.py`, and
`artifacts.py::build_symbolic_model`) are already slated for removal by
the unified-predictor spec regardless.

**Blocked by:** 03 — no config may still depend on the CasADi controller stack when this lands.

## Acceptance criteria

- [ ] `nonlinear_mpc.py`, `linear_mpc.py`, `nlp.py`, `solvers.py`, and `narx_mpc.py` are deleted.
- [ ] `nn_predictor_casadi.py`, `observable_casadi.py`, and
      `artifacts.py::build_symbolic_model` are deleted.
- [ ] No remaining config or code path references the deleted modules.
- [ ] `src/neuro/transforms.py`'s CasADi usage is confirmed non-load-bearing outside the controller
      before CasADi is removed from `pyproject.toml` (it is a direct dependency, not purely a
      controller concern).
- [ ] CasADi is removed from `pyproject.toml` and the lockfile is updated.
- [ ] Full test suite and `pre-commit run --all-files` pass with CasADi absent.
- [ ] Commit the changes. Fix any pre-commit hook errors
