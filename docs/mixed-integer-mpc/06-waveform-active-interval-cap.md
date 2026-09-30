# 06: Waveform active-interval cap

**What to build:** Let the waveform controller use Bonmin to limit enabled stimulation intervals within each Control Horizon.

**Blocked by:** 03: Waveform CNN.

## Acceptance criteria

- [x] One binary activation decision per Control Horizon step is shared by all electrodes. A disabled step has zero Control Current; an enabled step obeys the existing per-electrode bounds and Kirchhoff balance.
- [x] A configured horizon-local cap limits the number of enabled steps for both waveform MLP and CNN Predictors while preserving their selected continuous Costs.
- [x] Logs expose the planned binary decisions, applied and planned physical Control Currents, objective, solver status, success, and solve time.
- [x] A successful integer plan is shifted into the next initial guess. A failed solve issues no new Control Current and exposes the failure.
- [x] Configured-controller tests check the cap and physical constraints on returned plans, including a zero-cap case.
