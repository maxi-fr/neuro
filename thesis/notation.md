# Mathematical notation conventions

This agreement applies to the thesis, symbol list, and mathematical figure sources. It replaces the previous count grammar. The goals are one count family and equations that show mathematical relationships with minimal bookkeeping.

## Counts

Every count uses lowercase `n`, with a short subscript naming the counted quantity where needed. Layer selection uses a parenthesized superscript. Descriptive labels are upright; object and variable selectors are italic. Define meanings and counting units in prose rather than encoding categories such as `dim`, `layer`, `param`, or `score` in every symbol.

| Meaning | Notation |
| --- | --- |
| Regions, physical channels, electrodes, retained bins | $n_{\mathrm{reg}},n_{\mathrm{chan}},n_{\mathrm{elec}},n_{\mathrm{bin}}$ |
| State, observable, waveform, current dimensions | $n_x,n_o,n_y,n_u$ |
| History and regressor dimensions | $n_h,n_z$ |
| Observation and current history lengths | $n_{\mathrm{past}\,o},n_{\mathrm{past}\,u}$ |
| Prediction horizon | $n_{\mathrm{pred}}$ |
| Generic scored prediction length | $n_{\mathrm{score}}$ |
| Scored waveform samples and spectral frames | $n_{\mathrm{wave}},n_{\mathrm{spec}}$ |
| Segment, hop, smoothing, support lengths | $n_{\mathrm{seg}},n_{\mathrm{hop}},n_{\mathrm{smooth}},n_{\mathrm{support}}$ |
| MLP and CNN depths | $n_{\mathrm{MLP}},n_{\mathrm{CNN}}$ |
| MLP units in layer ell | $n^{(\ell)}$ |
| CNN channels and time positions in layer ell | $n_{\mathrm{chan}}^{(\ell)},n_{\mathrm{time}}^{(\ell)}$ |
| Temporal and frequency filter sizes | $n_{\mathrm{filter},t},n_{\mathrm{filter},f}$ |
| Temporal receptive field | $n_{\mathrm{RF}}$ |
| Training examples and trainable parameters | $n_{\mathrm{train}},n_\theta$ |
| Integer conduction lag | $n_{\mathrm{delay},ii'}$ |
| Full SVD rank | $n_{\mathrm{rank}}$ |

Retained rank $\rho$ remains a truncation selector. Layer selection always uses a parenthesized superscript, including counts: $n_{\mathrm{chan}}^{(\ell)}$, $n_{\mathrm{time}}^{(\ell)}$, and the MLP width $n^{(\ell)}$. The input layer uses $(0)$; the preceding layer uses $(\ell-1)$. Subscripts identify the counted quantity or its constituent coordinates. Physical counts and coordinate dimensions remain distinct concepts: state equalities such as $n_y=n_{\mathrm{chan}}$ and $n_u=n_{\mathrm{elec}}$ explicitly where applicable.

## Equation presentation

Main equations relate vectors, arrays, and mappings. Expand entries only when their indexing explains a relationship needed by the text. Explicitly suppress a fixed prediction step or example index in architecture definitions. Retain reference-step and example indices in dataset construction and extracted training pairs.

Use $\mathbf W^{(n_{\mathrm{MLP}})}$ and $\mathbf b^{(n_{\mathrm{MLP}})}$ for the MLP output layer, preserving the affine-layer numbering. State the output dimension directly rather than nesting a layer-count expression inside a width selector. Hidden layers use $\mathbf a^{(\ell)}$, $\mathbf W^{(\ell)}$, and $\mathbf b^{(\ell)}$.

Present CNN layers at array level using $\star$ for convolution between one filter and one input channel, with summation over input channels written separately. Use $c$ and $c'$ as output- and input-channel indices, respectively. Layer superscripts select arrays and layer-dependent counts; channel indices remain undecorated. Define filter support, unit strides, valid temporal extent, frequency padding, and bias broadcasting locally. An expanded entry formula is optional and must contribute more than implementation bookkeeping. A channel $\mathbf X_c^{(\ell)}$ is its full temporal or time-frequency array. Layer-zero channels correspond to physical EEG channels, with $n_{\mathrm{chan}}^{(0)}=n_{\mathrm{chan}}$; subsequent layers have their own channel counts. Restore the prediction-step index when connecting arrays to predictor inputs and outputs.

Original Fourier-bin labels are $m\in\mathcal M$. Describe CNN input histories by their array shapes and axis meanings rather than component equations or a second frequency-bin index. Time runs from oldest to newest, and retained frequency bins are in ascending order. The head receives final feature-array entries in fixed order followed by current history and candidate current; flattening needs no separately named vector or operation.

## Time and decorations

Use arguments for continuous-time signals and indices for discrete sequences. Plant integration, waveform, spectral-frame, predictor, and prediction-offset indices are $\kappa,r,q,k,j$. Define $k=r$ or $k=q$ when specializing the predictor. Time zero is the plant trajectory origin.

Use $i,i'$ for regions, $c$ for physical channels, $p$ for electrodes, $\nu$ for training examples, $k$ for their reference steps, $\gamma$ for optimization iteration, and $\beta,\beta'$ for window/filter offsets when needed. In dataset construction, $k$ may differ between examples; its dependence on $\nu$ is left implicit.

Predictions use $\mathbf o_{j\mid k}$ with $\mathbf o_{0\mid k}=\mathbf o_k$. The conditional index identifies prediction. Hats denote estimates; tildes denote standardization. The Heun intermediate state is $\mathbf x_{\kappa+1,i}^{(\mathrm p)}$ and basis vectors are $\mathbf e_i$. A superscript star selects an optimizer; parenthesized $(\rho)$ denotes truncation.

## Signals, arrays, and training

Keep the related family $\mathbf y_r$, $\mathbf Y_{q,c}$, $\boldsymbol{\mathcal Y}_{q,c}$, with scalar entries $y_{r,c}$, $Y_{q,c,m}$, $\mathcal Y_{q,c,m}$. Collect channels and retained bins in ascending order to form $\boldsymbol{\mathcal Y}_q$. Bold denotes vectors and arrays; specify their shape. Uppercase scalar array entries and calligraphic scalar losses are valid.

The preprocessed waveform $\mathbf y_r$ differs from raw $\mathbf y_{\mathrm{EEG}}(t)$. The simulated scalp EEG has no calibrated physical amplitude unit. Apply the logarithm directly to its numerical power-density values; no physical reference normalization is introduced.

The standardized shared regressor is $\mathbf z_k=[\widetilde{\mathbf h}_k^{\mathsf T}\;\widetilde{\mathbf u}_k^{\mathsf T}]^{\mathsf T}$, with $n_z=n_h+n_u$. The CNN-head input is $\mathbf z_k^{\mathrm{CNN}}$. Future input and target matrices $\mathbf U_k^{(\nu)}$, $\mathbf O_k^{(\nu)}$ have chronological columns.

Use $\mathcal L$ for training losses and stage costs, qualified locally. Use $\mathcal J$ for finite-horizon control cost; a horizon subscript is unnecessary unless comparing horizons. A bar denotes an average with domain and weights defined locally. Preserve transformation markers in means such as $\overline{\widetilde{\boldsymbol{\mathcal Y}}}$.

Centered DMD targets use $\mathbf T$ and centered regressors $\boldsymbol\Omega$. State snapshots use `prev` and `next`. Left/right singular vectors use $\mathbf V_{\mathrm L},\mathbf V_{\mathrm R}$ and singular values $\sigma_a$. Training standard deviations use $\sigma_{\mathrm o,a},\sigma_{\mathrm u,b}$. Structural connectivity $\mathbf W$ and layer weights $\mathbf W^{(\ell)}$ need no extra qualifier. Keep established biological parameters; locally defined reuse with indices is acceptable.

## Delays and availability

Keep physical delay $d_{ii'}=\ell_{ii'}/c$ and quantized delay $d_{ii'}^{[\Delta t_{\mathrm p}]}=\Delta t_{\mathrm p}n_{\mathrm{delay},ii'}$ distinct, with integer lag $n_{\mathrm{delay},ii'}=\operatorname{round}(d_{ii'}/\Delta t_{\mathrm p})$. Use the integer lag in discrete coupling.

Waveform spectral scoring requires $n_{\mathrm{spec}}n_{\mathrm{hop}}\le n_{\mathrm{pred}}$; waveform scoring requires $n_{\mathrm{wave}}\le n_{\mathrm{pred}}$. Spectral-output scoring requires $n_{\mathrm{spec}}\le n_{\mathrm{pred}}$.

Without earlier frames, the first smoothed frame has $q\ge n_{\mathrm{smooth}}-1$ and a complete observation history ends no earlier than $q=n_{\mathrm{smooth}}+n_{\mathrm{past}\,o}-2$. Past-current availability is separate.

## Verification

Check symbol-list entries, equations, and figure labels against this agreement. Compile into `out/`; keep temporary verification artifacts in `tmp/`. Check changed equations and figures for both layout and mathematical meaning. Reconcile unfinished MPC, experiment, conclusion, and appendix material as substantive equations are written.
