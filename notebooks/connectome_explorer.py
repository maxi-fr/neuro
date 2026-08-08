import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium", app_title="Stage 0 Connectome Explorer")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from matplotlib import pyplot as plt

    from neuro.connectome import Connectome
    from neuro.eeg import build_eeg_gain

    return Connectome, build_eeg_gain, mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # 🧠 Stage 0 — Connectome & EEG Forward Operator

    Loads the structural backbone for the Yu et al. 2024 closed-loop tES model
    from **The Virtual Brain**: region connectivity, conduction delays, and the
    hand-rolled region → sensor EEG gain matrix **L**.

    All **76** TVB regions are kept (the paper's 74 = 76 minus {`lCC`, `rCC`}). **L**
    collapses the surface projection to regions by **summing** vertex columns,
    matching TVB's region-level EEG monitor.
    """)
    return


@app.cell
def _(Connectome, build_eeg_gain):
    connectome = Connectome.from_config({})
    gain, channel_labels = build_eeg_gain()
    channel_index = {label: idx for idx, label in enumerate(channel_labels)}
    return channel_index, connectome, gain


@app.cell
def _(channel_index, connectome, gain, mo, np):
    _ez_pz = ("lHC", "lPHC", "lAMYG", "lTCI", "lTCV")
    _named_channels = ("CP5", "CP6", "PO3", "P1", "P3", "F3", "F5", "AF3", "O1")
    _off_diag = connectome.delays[~np.eye(connectome.delays.shape[0], dtype=bool)]
    _max_sane_delay_ms = 200.0

    def _badge(*, ok: bool, label: str) -> str:
        """_badge definition."""
        return f"- {'✅' if ok else '❌'} {label}"

    _checks = [
        _badge(
            ok=connectome.weights.shape == (76, 76), label=f"Region count = 76 (weights {connectome.weights.shape})"
        ),
        _badge(ok=gain.shape == (62, 76), label=f"EEG gain L shape = (62, 76) (got {gain.shape})"),
        _badge(ok=np.isfinite(gain).all(), label="L is finite (no NaN/inf)"),
        _badge(
            ok=all(r in connectome.region_index for r in _ez_pz),
            label=f"EZ/PZ regions present: {', '.join(_ez_pz)}",
        ),
        _badge(
            ok=all(c in channel_index for c in _named_channels),
            label=f"Named channels present: {', '.join(_named_channels)}",
        ),
        _badge(
            ok=bool((np.diag(connectome.delays) == 0).all()) and 0 < _off_diag.max() < _max_sane_delay_ms,
            label=f"Delays in ms range (max {_off_diag.max():.2f} ms, zero diagonal)",
        ),
        _badge(ok=bool((connectome.weights >= 0).all()), label="Weights nonnegative"),
        _badge(
            ok=int(connectome.hemispheres.sum()) > 0 and int((~connectome.hemispheres).sum()) > 0,
            label=f"Hemisphere split: {int(connectome.hemispheres.sum())} right / "
            f"{int((~connectome.hemispheres).sum())} left",
        ),
    ]
    mo.md("## ✅ Verification\n" + "\n".join(_checks))
    return


@app.cell
def _(connectome, plt):
    _fig, _ax = plt.subplots(figsize=(6, 5))
    _im = _ax.imshow(connectome.weights, cmap="viridis", aspect="auto")
    _ax.set_title("Connection weights (hemispheric blocks)")
    _ax.set_xlabel("source region")
    _ax.set_ylabel("target region")
    _fig.colorbar(_im, ax=_ax)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(connectome, plt):
    _fig, _ax = plt.subplots(figsize=(6, 5))
    _im = _ax.imshow(connectome.delays, cmap="viridis", aspect="auto")
    _ax.set_title(f"Conduction delays (ms), speed = {connectome.speed:g} mm/ms")
    _ax.set_xlabel("source region")
    _ax.set_ylabel("target region")
    _fig.colorbar(_im, ax=_ax, label="ms")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(gain, np, plt):
    _fig, _ax = plt.subplots(figsize=(9, 4))
    _vmax = float(np.abs(gain).max())
    _im = _ax.imshow(gain, cmap="RdBu_r", aspect="auto", vmin=-_vmax, vmax=_vmax)
    _ax.set_title("EEG leadfield matrix L (62 channels x 76 regions)")
    _ax.set_xlabel("region")
    _ax.set_ylabel("channel")
    _fig.colorbar(_im, ax=_ax)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(connectome, mo):
    _region_rows = [
        {"index": idx, "region": label, "hemisphere": "R" if hemi else "L"}
        for idx, (label, hemi) in enumerate(zip(connectome.region_labels, connectome.hemispheres, strict=True))
    ]
    _channel_rows = [{"index": idx, "channel": label} for idx, label in enumerate(connectome.channel_labels)]
    mo.hstack(
        [
            mo.vstack([mo.md("### Regions (76)"), mo.ui.table(_region_rows, page_size=15)]),
            mo.vstack([mo.md("### EEG channels (62)"), mo.ui.table(_channel_rows, page_size=15)]),
        ],
        widths="equal",
        gap=2,
    )
    return


if __name__ == "__main__":
    app.run()
