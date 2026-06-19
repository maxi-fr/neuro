import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium", app_title="Stage 1 Single-node Jansen-Rit")


@app.cell
def _():

    import marimo as mo
    import numpy as np
    from matplotlib import pyplot as plt

    from neuro.jansen_rit import JansenRitParams, lfp, simulate_network
    from utils.plotting import plot_psd

    DT_DEFAULT = 1e-4  # noqa: N806
    return (
        DT_DEFAULT,
        JansenRitParams,
        lfp,
        mo,
        np,
        plot_psd,
        plt,
        simulate_network,
    )


@app.cell
def _(mo):
    mo.md("""
    # 🧠 Stage 1 — Single-node Jansen-Rit

    One isolated, uncontrolled node of the Yu et al. 2024 whole-brain epilepsy
    model (their Eqs. 1-6), integrated with our own hand-rolled scheme. The
    excitatory gain **A** drives the fixed-point ↔ limit-cycle transition:

    - **A = 3.25** (healthy) and **A = 3.4** (PZ) are **stable fixed points** — a
      lone node stays quiescent; with noise it shows only a low-amplitude
      (peak-to-peak ≈ 2) background.
    - **A = 3.6** (EZ) is a **sustained spike-wave limit cycle** (~3 Hz, the
      seizure rhythm of this epilepsy model — not 8-12 Hz alpha).

    Parameters follow Table 1 (`I = 90`, `a = 100`, `b = 50`, `v0 = 6`, …); the
    only free knob is the white-noise std `sigma`, tuned so the background sits near
    amplitude 2 (calibrating the closed-loop threshold of 5). Everything is in SI
    seconds with `dt = 1e-4 s` (0.1 ms, matching the Stage 7 TVB reference).
    """)
    return


@app.cell
def _(DT_DEFAULT, JansenRitParams, lfp, simulate_network):
    def run(a_gain, *, sigma=None, duration=20.0, seed=7, deterministic=False):
        """Simulate one node and return ``(t, y, x)`` for gain ``a_gain``."""
        base = JansenRitParams()
        noise = 0.0 if deterministic else (base.sigma if sigma is None else sigma)
        params = JansenRitParams(A=a_gain, sigma=noise)
        t, x_traj = simulate_network(params=params, connectome=None, duration=duration, dt=DT_DEFAULT, seed=seed)
        x = x_traj[:, 0, :]  # drop the singleton node dim -> (6, n_samples)
        return t, lfp(x), x

    return (run,)


@app.cell
def _(DT_DEFAULT, np, plt, run):
    # Checkpoint: noisy background (A=3.25) vs seizure limit cycle (A=3.6).
    _fig, _axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, layout="constrained")
    for _ax, _a, _color, _name in (
        (_axes[0], 3.25, "#1f77b4", "A = 3.25  (healthy background)"),
        (_axes[1], 3.6, "#d62728", "A = 3.6  (EZ seizure)"),
    ):
        _t, _y, _ = run(_a, duration=20.0)
        _ax.plot(_t, _y, color=_color, linewidth=0.9)
        _ax.set_title(f"{_name}   |   p2p = {np.ptp(_y[int(5.0 / DT_DEFAULT) :]):.1f}", fontsize=11)
        _ax.set_ylabel("y = x₂ - x₃")
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.grid(visible=True, linestyle="--", alpha=0.3)
    _axes[1].set_xlabel("time (s)")
    _fig
    return


@app.cell
def _(DT_DEFAULT, plt, run):
    # Phase plane: lfp y = x2 - x3 against its derivative x5 - x6.
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    _n_drop = int(8.0 / DT_DEFAULT)
    for _ax, _a, _color, _kind in (
        (_axes[0], 3.25, "#1f77b4", "fixed point"),
        (_axes[1], 3.6, "#d62728", "closed orbit"),
    ):
        _t, _y, _x = run(_a, duration=20.0, deterministic=True)
        _ax.plot(_x[1, _n_drop:] - _x[2, _n_drop:], _x[4, _n_drop:] - _x[5, _n_drop:], color=_color, linewidth=0.8)
        _ax.set_title(f"A = {_a}  ({_kind})", fontsize=11)
        _ax.set_xlabel("y = x₂ - x₃")
        _ax.set_ylabel("ẏ = x₅ - x₆")
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.grid(visible=True, linestyle="--", alpha=0.3)
    _fig
    return


@app.cell
def _(DT_DEFAULT, np, plt, run):
    # Bifurcation: steady-state peak-to-peak amplitude of y as A is swept.
    _a_values = np.arange(3.0, 4.0001, 0.05)
    _n_drop = int(5.0 / DT_DEFAULT)
    _amps = []
    for _a in _a_values:
        _, _y, _ = run(_a, duration=14.0, deterministic=True)
        _amps.append(np.ptp(_y[_n_drop:]))
    _fig, _ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    _ax.plot(_a_values, _amps, "-o", color="#2ca02c", markersize=3)
    for _a, _label, _c in ((3.25, "healthy", "#1f77b4"), (3.4, "PZ", "#ff7f0e"), (3.6, "EZ", "#d62728")):
        _ax.axvline(_a, color=_c, linestyle="--", alpha=0.7, label=f"{_label} (A={_a})")
    _ax.set_title("Bifurcation: oscillation onset as A rises (deterministic)", fontsize=12)
    _ax.set_xlabel("excitatory gain A")
    _ax.set_ylabel("steady-state p2p amplitude")
    _ax.legend(loc="upper left", framealpha=0.9)
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.grid(visible=True, linestyle="--", alpha=0.3)
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 🎛️ Interactive single node
    """)
    return


@app.cell
def _(mo):
    a_slider = mo.ui.slider(3.0, 4.0, 0.01, value=3.6, label="A (excitatory gain)")
    sigma_slider = mo.ui.slider(0.0, 1000.0, 50.0, value=500.0, label="noise sigma")
    duration_slider = mo.ui.slider(5, 60, 5, value=20, label="duration (s)")
    seed_slider = mo.ui.slider(0, 50, 1, value=7, label="seed")
    deterministic_switch = mo.ui.checkbox(value=False, label="deterministic (RK4, no noise)")
    mo.vstack([a_slider, sigma_slider, duration_slider, seed_slider, deterministic_switch])
    return (
        a_slider,
        deterministic_switch,
        duration_slider,
        seed_slider,
        sigma_slider,
    )


@app.cell
def _(
    DT_DEFAULT,
    a_slider,
    deterministic_switch,
    duration_slider,
    np,
    plot_psd,
    plt,
    run,
    seed_slider,
    sigma_slider,
):
    _t, _y, _ = run(
        a_slider.value,
        sigma=sigma_slider.value,
        duration=float(duration_slider.value),
        seed=int(seed_slider.value),
        deterministic=deterministic_switch.value,
    )
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    _axes[0].plot(_t, _y, color="#9467bd", linewidth=0.8)
    _axes[0].set_title(f"A = {a_slider.value:.2f}   |   p2p = {np.ptp(_y[int(5.0 / DT_DEFAULT) :]):.1f}", fontsize=11)
    _axes[0].set_xlabel("time (s)")
    _axes[0].set_ylabel("y = x₂ - x₃")
    _axes[0].spines[["top", "right"]].set_visible(False)
    _axes[0].grid(visible=True, linestyle="--", alpha=0.3)

    plot_psd(_y[np.newaxis, :], DT_DEFAULT * 1000.0, max_freq=20.0, nperseg=8192, ax=_axes[1])
    _fig
    return


if __name__ == "__main__":
    app.run()
