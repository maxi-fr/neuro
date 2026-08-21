# 02 — NARX Output-Lifted MPC Benchmark & Solver Evaluation

**What to build:** An offline open-loop and online closed-loop benchmark suite evaluating the performance and stability of the NARX output-lifted MPC formulation against Single Shooting ($D = 50$) and Full-State Multiple Shooting ($D = 25$) baselines across IPOPT, standalone SQP (`qpoases`, `osqp`, `qrqp`), and hybrid SQP+IPOPT fallback.

This ticket adapts and utilizes the existing benchmark scripts in `scratch/`:

- [`scratch/benchmark_sqp_ipopt.py`]: Adapts open-loop NLP solver benchmarks to include the NARX output-lifted formulation alongside single and multiple shooting.
- [`scratch/sim_closed_loop_sqp.py`](file:///C:/Users/frank/closed-loop-neurostimulation/scratch/sim_closed_loop_sqp.py): Adapts closed-loop simulation benchmark on Jansen-Rit connectome to compare NARX MPC controller against full-state MPC and unassisted IPOPT baselines.

Results and insights will be documented in `knowledge-base/Notes/` as a follow-up to [`knowledge-base/Notes/sqp_ipopt_fallback_benchmark.md`](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Notes/sqp_ipopt_fallback_benchmark.md).

**Blocked by:** 01 — Re-introduce NARX Nonlinear MPC Controller

## Acceptance criteria

- [x] [`scratch/benchmark_sqp_ipopt.py`](file:///C:/Users/frank/closed-loop-neurostimulation/scratch/benchmark_sqp_ipopt.py) is adapted to evaluate offline open-loop NLP solves across held-out primed test states for:
  - Single Shooting ($D = 50$, $150$ variables)
  - Full-State Multiple Shooting ($D = 25$, $1,110$ variables with $960$ state defects)
  - NARX Output-Lifted Multiple Shooting ($3,250$ variables with $3,100$ compact output defects)
- [x] Median and P90 solve times, iterations, and convergence rates are benchmarked across IPOPT, SQP (`qpoases`, `osqp`, `qrqp`), and SQP+IPOPT fallback.
- [x] [`scratch/sim_closed_loop_sqp.py`](file:///C:/Users/frank/closed-loop-neurostimulation/scratch/sim_closed_loop_sqp.py) is adapted to run closed-loop simulation on Jansen-Rit connectome, comparing `NarxMPCController` against pure IPOPT and standard MPC.
- [x] Seizure burden reduction, delivered electrical charge, fallback frequency, and per-step solve times are measured.
- [x] Benchmark documentation in `knowledge-base/Notes/` ([`knowledge-base/Notes/narx_mpc_benchmark.md`](file:///C:/Users/frank/closed-loop-neurostimulation/knowledge-base/Notes/narx_mpc_benchmark.md)) captures findings, comparison tables, and execution instructions.
