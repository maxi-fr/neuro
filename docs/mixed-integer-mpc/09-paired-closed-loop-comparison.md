# 09: Paired closed-loop comparison

**What to build:** Compare continuous and mixed-integer controllers on paired Plant runs using the agreed Seizure Suppression target and stimulation measures.

**Blocked by:** 08: Active-interval penalty.

## Acceptance criteria

- [x] Continuous, capped, and penalized controller configurations run with the same Plant setup for seeds 7000, 7001, 7002, 7004, and 7005 over 12 seconds.
- [x] Results report final seizing regions per seed and whether at least four seeds finish with fewer than five seizing regions.
- [x] Results report enabled-step count, applied nonzero-current steps under a stated numerical threshold, delivered-current duty cycle in seconds, charge, energy, solver construction and per-update solve times, and solver failures.
- [x] A failed solve stops its seed's run and preserves the solver status in the comparison result.
- [x] The comparison distinguishes a horizon-local activation cap from realized duty cycle over the entire run and makes no claim of global optimality or real-time performance.

The energy measure is the squared-current integral in mA²·s. It is an energy proxy; electrical energy in joules requires an electrode resistance that the Plant configuration does not supply. The experiment package is in `artifacts/mixed_integer/03_casadi_paired_comparison/`.
