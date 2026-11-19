from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style

FIGURES = Path(__file__).resolve().parent
THESIS = FIGURES.parents[1]
REPO = THESIS.parent
CELLS_DIR = REPO / "artifacts" / "transcription_benchmark" / "cells"

PREDICTORS = (
    ("spectral_dmdc", r"Spectral DMDc ($n_o = 744$)"),
    ("spectral_cnn", r"Spectral CNN ($n_o = 744$)"),
    ("waveform_mlp", r"Waveform MLP ($n_o = 62$)"),
)
TRANSCRIPTIONS = (
    ("single_shooting", "Single shooting", "C0", "-", "o"),
    ("narx", "NARX", "C1", "--", "s"),
    ("multiple_shooting", "Multiple shooting", "C2", ":", "^"),
)
HORIZONS_S = (0.30, 0.60, 1.02, 1.50)


def _cell_metrics(predictor: str, horizon_s: float, transcription: str) -> tuple[float | None, float | None]:
    """Return ``(converged_median_s, converged_pct)`` from the cell JSON, or ``(None, None)`` if skipped."""
    path = CELLS_DIR / f"{predictor}_h{round(horizon_s * 100):03d}_{transcription}.json"
    cell: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if cell["status"] in ("skipped_infeasible", "skipped_oom"):
        return None, None
    ol = cell["open_loop"]
    conv_times = [float(r["wall_s"]) for r in ol if r["success"]]
    if cell["status"] == "timeout_single_iter" and not ol:
        return None, 0.0
    if not ol:
        return None, None
    pct = 100.0 * len(conv_times) / len(ol)
    med = statistics.median(conv_times) if conv_times else None
    return med, pct


def save_figure() -> Path:
    """Plot converged open-loop solve time and convergence rate across predictors and horizons."""
    fig, axes = plt.subplots(
        2,
        3,
        figsize=style.figsize(height_in=3.1),
        sharex=True,
        sharey="row",
        layout="constrained",
    )

    for col, (pred_key, title) in enumerate(PREDICTORS):
        ax_time = axes[0, col]
        ax_conv = axes[1, col]
        for tr_key, label, color, ls, marker in TRANSCRIPTIONS:
            times: list[tuple[float, float]] = []
            convs: list[tuple[float, float]] = []
            for h_s in HORIZONS_S:
                med, pct = _cell_metrics(pred_key, h_s, tr_key)
                if med is not None:
                    times.append((h_s, med))
                if pct is not None:
                    convs.append((h_s, pct))
            if times:
                ax_time.plot(
                    [h for h, _ in times],
                    [t for _, t in times],
                    color=color,
                    linestyle=ls,
                    marker=marker,
                    markersize=3.5,
                    label=label,
                )
            if convs:
                ax_conv.plot(
                    [h for h, _ in convs],
                    [c for _, c in convs],
                    color=color,
                    linestyle=ls,
                    marker=marker,
                    markersize=3.5,
                    label=label,
                )

        ax_time.set_title(title)
        ax_time.set_yscale("log")
        ax_time.set_ylim(0.2, 200.0)
        ax_time.grid(axis="both")
        ax_conv.set_ylim(-6.0, 108.0)
        ax_conv.set_yticks((0, 50, 100))
        ax_conv.set_xticks(HORIZONS_S, [f"{h:.2f}" for h in HORIZONS_S])
        ax_conv.set_xlabel("Control horizon (s)")
        ax_conv.grid(axis="both")

    axes[0, 0].set_ylabel("Solve time (s)")
    axes[1, 0].set_ylabel(r"Converged (\%)")
    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, handlelength=2.0)

    out = FIGURES / "transcription_benchmark.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    """Generate the appendix transcription benchmark figure."""
    out = save_figure()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
