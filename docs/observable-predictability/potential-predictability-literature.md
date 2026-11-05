# Published estimators for full-state potential predictability

## Which equations to adopt

Use Schubert et al. (2002), §3a, Eqs. (1)–(3). They estimate signal variance and within-ensemble noise variance from balanced ensembles and report the ratio of signal variance to total variance. The same finite-ensemble estimator is printed particularly clearly in Matsumura et al. (2010), §2, Eqs. (1)–(4), which credits Rowell et al. (1995). Rowell (1998), §2, Eq. (2), uses this signal-to-total ratio for potential predictability and refers to the 1995 paper for the component estimators. [Schubert et al. (2002)](https://cims.nyu.edu/~gerber/pages/documents/schubert_etal2002.pdf) · [Matsumura et al. (2010)](https://eprints.lib.hokudai.ac.jp/repo/huscap/all/43127/JMSJ88-1_43-62.pdf) · [Rowell (1998)](https://journals.ametsoc.org/abstract/journals/clim/11/2/1520-0442_1998_011_0109_apspwa_2.0.co_2.xml)

At a fixed lookahead and starting phase, let \(y_{ir}\) be one scalar Observable component from snapshot \(i\) and replay \(r\). Let \(n\) be the number of snapshots, \(m\) the number of independent replays per snapshot, \(\bar y_i\) the replay mean and \(\bar y\) the grand mean. Substitute these for the seasons and ensemble members in Schubert et al. The within-snapshot variance, corrected between-snapshot variance, and published ratio are

\[
\begin{aligned}
\widehat V_N &= \frac{1}{n(m-1)}
\sum_{i=1}^n\sum_{r=1}^m (y_{ir}-\bar y_i)^2,\\
\widehat V_S &= \frac{1}{n-1}\sum_{i=1}^n(\bar y_i-\bar y)^2
-\frac{\widehat V_N}{m},\\
\widehat P_B &= \frac{\widehat V_S}{\widehat V_S+\widehat V_N}.
\end{aligned}
\]

The subtraction removes the noise variance still present in a finite replay mean. Under independent snapshots and independent replays conditional on each snapshot, the variance-component estimates are unbiased. The ratio itself is a plug-in estimate. A negative signal estimate can occur by sampling variation when the signal is weak, even though the population signal variance is nonnegative. A zero total makes the ratio undefined. Report the estimation convention and uncertainty if the estimated ratio falls near or below zero. [Schubert et al., Eqs. (1)–(3)](https://cims.nyu.edu/~gerber/pages/documents/schubert_etal2002.pdf) · [Matsumura et al., Eqs. (1)–(4)](https://eprints.lib.hokudai.ac.jp/repo/huscap/all/43127/JMSJ88-1_43-62.pdf)

## Why it applies to the Plant replay

DelSole (2004), §2, describes a stochastic forecast conditioned on an exactly known initial state. His example gives a forecast distribution with nonzero spread because future randomness remains. This supports the replay setup. The paper is by Timothy DelSole alone. The accessible May 2003 draft also discusses pairwise forecast dispersion around its Eq. (22). The indexed published version uses different equation numbering, and direct full-text access returned HTTP 403. Cite the published paper for the forecast setup and Schubert et al. for the estimator. [DelSole (2004)](https://doi.org/10.1175/1520-0469%282004%29061%3C2425%3APAITPI%3E2.0.CO%3B2) · [Author draft](https://aquadocs.org/server/api/core/bitstreams/1bb61395-6cb9-4c16-80d0-8bd97d26eb13/content)

Tang, Chen, and Yan (2014), §2, Eqs. (9)–(13), index the same empirical decomposition by initial condition and ensemble member. Their Eq. (13) names both signal-to-total ratio and signal-to-noise ratio. Their displayed empirical decomposition lacks the finite-ensemble subtraction above. Appendix a relates the square root of the population signal-to-total ratio to the correlation of the conditional mean forecast with an independent realization. Rowell’s Appendix b distinguishes that from the correlation between two independent individual realizations and gives the finite-ensemble mean correlation in Eq. (A22). For an ideal squared-error point Predictor, report the variance fraction itself. [Tang et al. (2014)](https://doi.org/10.1175/JCLI-D-12-00654.1) · [Rowell (1998)](https://journals.ametsoc.org/abstract/journals/clim/11/2/1520-0442_1998_011_0109_apspwa_2.0.co_2.xml)

In the simulation, the within-snapshot component is the minimum MSE under squared error when the Predictor knows the Plant snapshot and required pre-branch history, the future input is fixed, and future noise is unknown. The variance fraction expresses the gain over predicting the phase-specific mean at that lead. It depends on how the starting snapshots are sampled. The neuroscience analogue uses repeated stimuli to estimate explainable signal power from repeated neural responses. Sahani and Linden’s Eq. (1) likewise subtracts the finite-repeat noise contribution from the variance of trial means, with a time-bin variance convention suited to their experiment. [DelSole (2004)](https://doi.org/10.1175/1520-0469%282004%29061%3C2425%3APAITPI%3E2.0.CO%3B2) · [Sahani and Linden (NeurIPS 2002), §2, Eq. (1)](https://proceedings.neurips.cc/paper/2002/file/7b4773c039d539af17c883eb9283dd14-Paper.pdf)

## Applying it to our Observables

- Use zero-current continuations and hold the Plant snapshot and causal pre-branch history fixed across replays. Compute the published scalar estimator at each lead and starting phase for each component. Report phase-specific curves so differences between phase means do not inflate within-phase predictability.
- For a single curve per vector-valued Observable, sum the signal variances and the total variances across components, then divide. This is a variance-weighted aggregation of the scalar ratio and matches componentwise MSE with equal squared-error weights. It is the project’s aggregation choice. Keep the square root of mean within-snapshot noise variance alongside it as the native-unit error floor.
- Use the same valid components in both sums. The cited estimator assumes a balanced set of replays. If a component is missing, specify the eligible complete groups or justify a different estimator. For intervals, resample independent Plant seeds together with their replay groups. Keep leads that reuse pre-branch history marked on the plot.
- The legacy repo helper already computes the empirical within-to-total decomposition with population divisors and without the finite-ensemble subtraction. The scored archive stores replicate spread, but lacks the componentwise snapshot means or their between-snapshot variance. Calculating the published ratio requires recovering or regenerating the relevant Observable replays, or retaining the missing sufficient statistics during scoring. [Legacy variance ratio](../../src/neuro/metrics.py) · [Scored archive serialization](../../src/neuro/predictability_archive.py)
