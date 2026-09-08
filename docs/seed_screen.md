# Held-out seed screen for closed-loop comparisons

The comparison manifests in `configs/comparison/` run every arm on the same seeds, so the seeds have
to be fixed once and justified once. Block 7000+ is reserved for this: it is not used for training,
sweeps, or any Predictor identification, so a controller cannot have seen it.

## Reproducing it

```bash
uv run python scripts/run_comparison.py configs/comparison/seed_screen.yaml \
    --output-dir results/comparison/seed_screen --workers 8
```

That is `uncontrolled.yaml` at `t_end = 12.0` over seeds 7000-7019, scored by the same
`neuro.comparison.spread_metrics` reduction the comparison tables use — so the numbers below are the
same quantities the tables report, not a parallel measurement. The table is reproduced here rather
than referenced because `results/` is gitignored.

Onsets are censored at the run length, so `t_pz` or `t_left_half` reading **12.0** means *never
recruited*, not *recruited at the end*. The `*_recruited` counts say which of the two it was.

## Results

| seed | burden | EZ | PZ | healthy | t_pz | t_left_half | frac_left |
| ---- | ------ | -- | -- | ------- | ---- | ----------- | --------- |
| 7000 | 0.124  | 3  | 2  | 27      | 9.50 | 10.00       | 0.921     |
| 7001 | 0.327  | 3  | 2  | 22      | 2.50 | 2.50        | 0.921     |
| 7002 | 0.283  | 3  | 2  | 26      | 1.00 | 4.00        | 0.921     |
| 7003 | 0.059  | 3  | 1  | 0       | 9.25 | *12.00*     | 0.237     |
| 7004 | 0.257  | 3  | 2  | 28      | 5.00 | 5.50        | 0.921     |
| 7005 | 0.359  | 3  | 2  | 23      | 1.50 | 1.75        | 0.921     |
| 7006 | 0.328  | 3  | 2  | 21      | 2.25 | 2.50        | 0.921     |
| 7007 | 0.237  | 3  | 2  | 23      | 5.75 | 6.25        | 0.921     |
| 7008 | 0.357  | 3  | 2  | 24      | 2.00 | 2.00        | 0.921     |
| 7009 | 0.199  | 3  | 2  | 27      | 0.75 | 7.50        | 0.921     |
| 7010 | 0.186  | 3  | 2  | 24      | 7.50 | 7.75        | 0.921     |
| 7011 | 0.331  | 3  | 2  | 26      | 2.00 | 2.75        | 0.921     |
| 7012 | 0.169  | 3  | 2  | 25      | 7.75 | 8.25        | 0.921     |
| 7013 | 0.058  | 3  | 1  | 1       | 2.25 | *12.00*     | 0.184     |
| 7014 | 0.053  | 3  | 2  | 1       | *12.00* | *12.00*  | 0.132     |
| 7015 | 0.069  | 3  | 2  | 28      | *12.00* | *12.00*  | 0.263     |
| 7016 | 0.384  | 3  | 2  | 24      | 1.00 | 1.00        | 0.921     |
| 7017 | 0.348  | 3  | 2  | 24      | 1.50 | 2.25        | 0.921     |
| 7018 | 0.276  | 3  | 2  | 27      | 4.00 | 4.75        | 0.921     |
| 7019 | 0.215  | 3  | 2  | 26      | 1.50 | 7.00        | 0.921     |

*Italic* onsets are censored: the zone was never recruited.

## What the screen found

- **The EZ is not informative.** All 20 seeds recruit all 3 EZ regions, always at 0.5 s. With
  `A_EZ = 3.6` the EZ is autonomously hyper-excitable, so it ignites regardless of the noise
  realisation. A seed criterion phrased on EZ recruitment selects nothing, and the EZ onset column
  was dropped from `spread_metrics` for the same reason.
- **Spread extent is bimodal, not a spectrum.** `frac_left` is 0.921 on 16 seeds and 0.13-0.26 on the
  other four (7003, 7013, 7014, 7015). There are no intermediate seeds. Either the seizure escapes
  the EZ and takes most of the left hemisphere, or it fizzles. An acceptance *band* on `frac_left`
  (0.3-0.9, say) would reject all 20 seeds.
- **The real variation is in timing.** Among the 16 spreading seeds, `t_pz` runs 0.75-9.5 s,
  `t_left_half` runs 1.0-10.0 s, and Seizure Burden runs 0.124-0.384. That is the axis a controller
  is actually scored against: a fast seizure gives it less warning than a slow one.
- Contralateral spread is essentially absent, so the right-hemisphere fraction was dropped from
  `spread_metrics` too: nothing to select on there either.
- Seed 7015 is the odd one out: 28 healthy regions are above threshold at the end, but almost none
  of them register a *sustained* onset, so nothing propagates in the sense the metrics measure.

## The criterion applied

A seed is admissible when all three hold on the uncontrolled run:

1. all 3 EZ regions recruited,
2. all 2 PZ regions recruited,
3. `t_left_half` below the run length -- the seizure leaves the seed zones inside the window.

Seeds 7003 and 7013 fail (2) and (3); 7014 and 7015 fail (3). The five seeds are then taken as the
**first five in seed order** that pass, not the five with the most interesting burdens: picking on
the outcome being measured would bias every arm's score by construction. That gives **7000, 7001,
7002, 7004, 7005**, whose `t_left_half` spans 1.75-10.0 s -- a genuine mix of fast and slow spread
rather than five replicates of one seizure.

These are the seeds written into `configs/comparison/controllers.yaml` and
`configs/comparison/costs.yaml`; changing them changes every number in the comparison tables, so
they should stay fixed.
