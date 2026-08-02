import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full", app_title="Supervisor Update — Seventh meeting")


@app.cell(hide_code=True)
def imports():
    """imports definition."""
    from pathlib import Path

    import marimo as mo

    notebook_dir = Path(__file__).parent
    fig_dir = notebook_dir.parent / "docs" / "figures"
    return fig_dir, mo


@app.cell(hide_code=True)
def slide_title(mo):
    """slide_title definition."""
    mo.md(r"""
    # Closed-loop Neurostimulation
    ## Seventh meeting

    **Max Frankenberg** · 8 July 2026
    """)
    return


@app.cell(hide_code=True)
def slide_losses(mo):
    """slide_losses definition."""
    _header = mo.md("## Training losses: spectral (PSD) & connectivity (FC)")

    _col_math = mo.md(
        r"""
    All terms evaluated in **standardized EEG-channel space**: latent predictions are
    decoded ($\hat{\mathbf{z}} \to \hat{\mathbf{y}}$) first, then **pooled over the whole batch**.

    **MSE curriculum** (recap) — anneal $\alpha: 1 \to 0$
    $$\mathcal{L}_\text{MSE} = \alpha\,\mathcal{L}_1 + (1-\alpha)\,\mathcal{L}_H$$

    **PSD loss** — Welch PSD per channel (`nperseg` $= H$), log-space MSE

    $$\mathcal{L}_\text{PSD} = \frac{1}{C\,F}\sum_{c=1}^{C}\sum_{f}\Big(\log \hat{S}_c(f) - \log S_c(f)\Big)^2$$

    **FC loss** — functional connectivity = channel–channel correlation matrix

    $$\mathrm{FC}(Y) = \operatorname{corr}(Y),\qquad \mathcal{L}_\text{FC} = \frac{1}{C^2}\big\lVert \mathrm{FC}(\hat{Y}) - \mathrm{FC}(Y)\big\rVert_F^2$$

    **Combined**
    $$\mathcal{L} = \mathcal{L}_\text{MSE} + w_\text{PSD}\,\mathcal{L}_\text{PSD} + w_\text{FC}\,\mathcal{L}_\text{FC}$$
    """
    )

    _col_notes = mo.md(
        r"""
    ### What changed
    - **Refactored** PSD and FC losses.
    - Now computed over the **entire batch** (pooled channel $\times$ time), not per-sample.
        - `pred_pool`: reshape $(B, H, C) \to (C,\, B\!\cdot\!H)$ before Welch / `corrcoef`.

    ### Findings
    - Spectral / FC match **improves** with the pooled version.
    - **But not necessary** — MSE curriculum alone already predicts and controls well.
    """
    )

    _body = mo.hstack([_col_math, _col_notes], widths=[3, 2], gap=2, align="start")
    mo.vstack([_header, _body])
    return


@app.cell(hide_code=True)
def slide_eeg(mo):
    """slide_eeg definition."""
    _header = mo.md(
        r"""
    ## Reducing the EEG output dimension
    Forward operator $L \in \mathbb{R}^{62 \times 76}$ maps 76 regions $\to$ 62 scalp channels.
    Shrinking $C$ shrinks **both** the predictor and the MPC's tracked output.
    """
    )

    _col_manual = mo.md(
        r"""
    ### 1 — Manual montage (10–20 + epilepsy)
    * standard 10-20 electrode setup + some epilepsy specific ones
    * TVB eeg projection matrix has 62 channels
        * represents a 10-10 configuration
        * Contains the entire 10-20 system (19 channels) + more

    <small>
    Fp1 Fp2 · F7 F3 Fz F4 F8 · T7 C3 Cz C4 T8 · P7 P3 Pz P4 P8 · O1 O2
    · **TP9 TP7 CP5 FC5 P5 PO7**
    </small>
    """
    )

    _col_pca = mo.md(
        r"""
    ### 2 — PCA latent projection
    Project the 62-channel EEG onto $k = 25$ principal components; **train and control in
    latent space**, decode back for the cost.

    $$\mathbf{z}_t = W^\top(\mathbf{y}_t - \bar{\mathbf{y}}), \qquad
      \hat{\mathbf{y}}_t = W\,\mathbf{z}_t + \bar{\mathbf{y}}$$
    with $W \in \mathbb{R}^{62 \times 25}$ fixed (differentiable) inverse projection.

    - The MLP predicts $\hat{\mathbf{z}}_{t+1}$; the decode $W$ is baked into the artifact.
    - MPC runs on latent dynamics, decodes to raw EEG only inside its cost term.
    - Keeps the full 62-channel information at $C_\text{model} = 25$.
    """
    )

    _body = mo.hstack([_col_manual, _col_pca], widths=[1, 1], gap=2, align="start")
    mo.vstack([_header, _body])
    return


@app.cell(hide_code=True)
def slide_mpc(mo):
    """slide_mpc definition."""
    _header = mo.md(
        r"""
    ## MPC formulations for (N)ARX models
    The NARX state is a **shift register** — every past input/output stacked:
    $\mathbf{x}_k = \big[\,\mathbf{y}_{k-1};\dots;\mathbf{y}_{k-n_y};\;
      \mathbf{u}_{k-1};\dots;\mathbf{u}_{k-n_u}\,\big] \in \mathbb{R}^{n_y c_y + n_u c_u}$

    """
    )

    _formulations = mo.md(
        r"""
    | | decision variables | idea | verdict |
    | --- | --- | --- | --- |
    | **Multiple shooting** | $n_\xi = H(n_y c_y + n_u c_u)$ | lift full shift-register state at every node; most rows are literal *copies* | too big + ill-conditioned |
    | **Output-lifted NARX** | $n_\xi = H(c_y + c_u)$ | lift only the new output, one defect $\mathbf{y}_k - \phi(\cdot) = 0$ per node; banded Jacobian | works, still slow |
    | **Single shooting** | $n_\xi = H\,c_u = 40$ | condense **all** states out, roll model forward, keep only controls | fast, chosen |

    - Output-lifted defect: $\;\mathbf{y}_k - \phi(\mathbf{y}_{k-1},\dots,\mathbf{y}_{k-n_y},\,
      \mathbf{u}_{k-1},\dots,\mathbf{u}_{k-n_u}) = 0$ — indices before the horizon are **measured past** (fixed).
    - Single shooting's $n_\xi = H\,c_u$ is **independent of channel count** $c_y$.
    - Each posed for **linear** and **nonlinear** MPC.
    """
    )

    mo.vstack([_header, _formulations])
    return


@app.cell(hide_code=True)
def slide_solver(mo):
    """slide_solver definition."""
    _header = mo.md(
        r"""
    ## Solver comparison
    """
    )

    _col_linear = mo.md(
        r"""
    ### Linear MPC — convex QP
    | solver | median solve |
    | ------ | -----------: |
    | OSQP (lifted, ill-cond.) | ~1900 ms |
    | **qpOASES condensed** (`dense`) | **147 ms** |

    OSQP never converges (hits 4000-iter cap)
    """
    )

    _col_channels = mo.md(
        r"""
    ### Effect on channel reduction on single shooting
    | config | solve | ms / iter |
    | ------ | ----: | --------: |
    | full (62 ch) | 0.53 s | 38 |
    | full PCA (25) | 0.50 s | 19 |
    | selected (25 ch) | 0.50 s | 20 |
    """
    )

    _nonlinear = mo.md(
        r"""
    ### Nonlinear MPC — 62 channels, IPOPT (30 s cap), $n_y{=}15,\,n_u{=}10,\,H{=}20$
    | formulation | decision vars | build | solve | iters | status |
    | ----------- | ------------: | ----: | ----: | ----: | ------ |
    | multiple shooting (`depth=1`) | 18 090 | 10.4 s | 33 s (cap) | **0** | CpuTime exceeded |
    | multiple shooting (`depth=5`) | 2 890 | 7.1 s | 33 s (cap) | 7 | CpuTime exceeded |
    | output-lifted NARX | 1 280 | 2.6 s | 11.7 s | 13 | converged |
    | **single shooting (`depth=20`)** | **40** | 0.05 s | **0.53 s** | 14 | **converged** |

    Multiple shooting can't finish **one** iteration of the 18 090-var problem in 30 s.
    Single shooting is **~22× faster than output-lifted** — same solution $\mathbf{u}_0 = [3, 3]$.
    """
    )

    _body = mo.hstack([_col_linear, _col_channels], widths=[1, 1], gap=2, align="start")
    mo.vstack([_header, _body, _nonlinear])
    return


@app.cell(hide_code=True)
def slide_suppression(fig_dir, mo):
    """slide_suppression definition."""
    _fig_path = fig_dir / "suppression_check.png"
    _fig = (
        mo.image(_fig_path.read_bytes(), width=760)
        if _fig_path.exists()
        else mo.md("> *[missing `docs/figures/suppression_check.png`]*")
    )

    _table = mo.md(
        r"""
    ### Suppression sanity check
    Closed loop vs. a **matched no-stim baseline** (same `seed: 69` noise).
    Score = mean-square EEG over the steady window $t \in [1, 5]\,\text{s}$.

    - **Full-EEG controllers surpress the seizure** (~97%), linear *and* nonlinear.
    - **25-channel "selected" largely fails**.
    """
    )

    mo.hstack([_table, _fig], widths=[1, 1], gap=2, align="center")
    return


if __name__ == "__main__":
    app.run()
