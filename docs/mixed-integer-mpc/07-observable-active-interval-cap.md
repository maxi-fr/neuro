# 07: Observable active-interval cap

**What to build:** Apply the Bonmin horizon-local active-interval cap to Observable MLP and CNN controllers on their native Frame grid.

**Blocked by:** 05: Observable CNN; 06: Waveform active-interval cap.

## Acceptance criteria

- [ ] Configuration enables the cap with either Observable Predictor architecture and the existing Observable Cost options.
- [ ] Each binary decision controls all electrodes for one native Predictor step; a disabled step has zero physical Control Current.
- [ ] Returned plans respect the cap, per-electrode bounds, and explicit Kirchhoff balance.
- [ ] The controller reports binary plans and solver outcomes, shifts successful plans, and issues no new Control Current after a failed solve.
