from __future__ import annotations

import json
import math
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from neuro.predictor.checkpoint import load_checkpoint
from neuro.predictor.data import load_trajectory, reduce_trajectory_to_frames
from neuro.predictor.evaluation import evaluate_spectral_free_run
from neuro.predictor.inference import ObservableCNNModel, ObservableMLPModel, inference_from_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style

if TYPE_CHECKING:
    from neuro.config import StftGeometry
    from neuro.types import FloatArray

FIGURES = Path(__file__).resolve().parent
THESIS = FIGURES.parents[1]
TABLES = THESIS / "tables" / FIGURES.name
REPO = THESIS.parent
CAMPAIGN = REPO / "artifacts" / "sweep_2026_11"
STUDIES = (
    ("Observable MLP", "observable_mlp/architecture", "observable_mlp", "val_loss"),
    ("Observable CNN", "cnn/observable/architecture", "cnn/observable", "val_log_mse"),
    ("Observable Hankel-DMDc", "observable_dmd/hop3", "observable_dmd", "val_loss"),
    ("Waveform MLP", "waveform_mlp/architecture", "waveform_mlp", "val_loss"),
    ("Waveform CNN", "cnn/waveform/architecture", "cnn/waveform", "val_loss"),
)
OBSERVABLE_MODELS = (ObservableMLPModel, ObservableCNNModel)
FS = 50
RAW_DT = 0.0001
DOWNSAMPLE = 200
HORIZON = 17


def read_study(directory: Path) -> tuple[int, float, int, int, dict[str, float]]:
    """Read the best completed trial, counts, and DMDc settings from a frozen study."""
    db = directory / "nn_predictor_sweep.db"
    with closing(sqlite3.connect(f"{db.as_uri()}?mode=ro&immutable=1", uri=True)) as conn:
        attempted, completed = conn.execute("SELECT COUNT(*), SUM(state = 'COMPLETE') FROM trials").fetchone()
        trial, value, trial_id = conn.execute(
            """
            SELECT trials.number, trial_values.value, trials.trial_id
            FROM trials JOIN trial_values USING (trial_id)
            WHERE trials.state = 'COMPLETE' AND trial_values.objective = 0
            ORDER BY trial_values.value, trials.number LIMIT 1
            """
        ).fetchone()
        dmd = {
            name: json.loads(distribution)["attributes"]["choices"][int(index)]
            for name, index, distribution in conn.execute(
                "SELECT param_name, param_value, distribution_json FROM trial_params "
                "WHERE trial_id = ? AND param_name IN ('dmd_energy', 'dmd_lambda')",
                (trial_id,),
            )
        }
        if dmd:
            (rank,) = conn.execute(
                "SELECT value_json FROM trial_user_attributes WHERE trial_id = ? AND key = 'retained_rank'",
                (trial_id,),
            ).fetchone()
            dmd["rank"] = json.loads(rank)
    return trial, value, completed, attempted, dmd


def describe_settings(meta: dict[str, Any], arrays: dict[str, FloatArray], dmd: dict[str, float]) -> str:
    """Describe the selected neural structure or DMDc identification settings in LaTeX."""
    if dmd:
        energy_percent = 100 * dmd["dmd_energy"]
        return (
            rf"Retained rank {dmd['rank']:.0f}, {energy_percent:.0f}\% energy cutoff, \(\lambda={dmd['dmd_lambda']:g}\)"
        )
    if meta["model_type"] == "mlp":
        output = "residual output" if meta["residual"] else "direct output"
        depth = meta["depth"]
        layers = "One hidden layer" if depth == 1 else f"{depth} hidden layers"
        return f"{layers} of {meta['hidden_size']} units, {meta['activation']}, {output}"
    filters = " then ".join(str(arrays[f"conv.{i}.weight"].shape[0]) for i in range(meta["n_convs"]))
    kernel_shape = arrays["conv.0.weight"].shape[2:]
    kernel = (
        f"{kernel_shape[0]}-sample kernel"
        if len(kernel_shape) == 1
        else rf"\({kernel_shape[0]}\times{kernel_shape[1]}\) kernel"
    )
    head = f"{arrays['head.0.weight'].shape[0]}-unit hidden head" if meta["head_depth"] else "linear readout"
    return f"{filters} filters, {kernel}, {meta['activation']}, {head}"


def load_recordings(names: list[str], geometry: StftGeometry) -> list[tuple[FloatArray, FloatArray]]:
    """Load filtered EEG and check that each current holds across a full scoring Frame."""
    trajectories = []
    hop = geometry.n_hop
    for name in names:
        currents, eeg = load_trajectory(str(REPO / name), None, DOWNSAMPLE, RAW_DT)
        first_endpoint = geometry.sample_support_steps(FS) - 1
        intervals = (len(currents) - first_endpoint) // hop
        held = currents[first_endpoint : first_endpoint + intervals * hop].reshape(intervals, hop, -1)
        if not np.all(held == held[:, :1]):
            msg = f"Current changes within a scoring Frame in {name}."
            raise ValueError(msg)
        trajectories.append((currents, eeg))
    return trajectories


def persistence_mse(
    trajectories: list[tuple[FloatArray, FloatArray]], geometry: StftGeometry, start: int
) -> tuple[FloatArray, int]:
    """Score the last measured Frame repeated over the same forecast windows as the fitted models."""
    sq_error = np.zeros(HORIZON)
    n_windows = 0
    n_values = 0
    for _, eeg in trajectories:
        frames = reduce_trajectory_to_frames(eeg, geometry, FS)
        origins = np.arange(start, len(frames) - HORIZON)
        current = frames[origins]
        for ahead in range(1, HORIZON + 1):
            sq_error[ahead - 1] += np.square(frames[origins + ahead] - current).sum()
        n_windows += len(origins)
        n_values += current.size
    return sq_error / n_values, n_windows


def print_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    """Print a Markdown table to stdout."""
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()


def save_table(path: Path, columns: str, headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    """Write a generated booktabs fragment for inclusion in the chapter."""
    lines = [
        "% Generated by offline_predictor_benchmark.py.",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        *(" & ".join(row) + r" \\" for row in rows),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_lookahead_plot(step_s: float, labels: tuple[str, ...], scores: list[FloatArray]) -> None:
    """Plot matched log-power error on the common physical lookahead grid."""
    fig, ax = plt.subplots(figsize=style.figsize(height_in=2.65), layout="constrained")
    lookaheads = step_s * np.arange(1, HORIZON + 1)
    styles = (
        ("0.40", ":", ""),
        ("C0", "-", "o"),
        ("C1", "-", "s"),
        ("C2", "-", "^"),
        ("C0", "--", "o"),
        ("C1", "--", "s"),
    )
    for label, score, (color, linestyle, marker) in zip(labels, scores, styles, strict=True):
        ax.plot(
            lookaheads, score, color=color, linestyle=linestyle, marker=marker, markersize=2.5, markevery=3, label=label
        )
    ax.set(xlabel=r"Lookahead $h$ (s)", ylabel="Log-power MSE")
    ax.grid(axis="y")
    handles, names = ax.get_legend_handles_labels()
    order = (1, 4, 2, 5, 3, 0)
    ax.legend(
        [handles[i] for i in order],
        [names[i] for i in order],
        loc="lower left",
        bbox_to_anchor=(0, 1.02, 1, 0),
        ncols=3,
        mode="expand",
        frameon=False,
    )
    fig.savefig(FIGURES / "offline_predictor_lookahead.pdf")
    plt.close(fig)


def publish_benchmark(configurations: list[tuple[str, ...]], step_s: float, scores: list[FloatArray]) -> None:
    """Print all lookahead scores and write the compact thesis tables and plot."""
    rows = [("All lookaheads", *(f"{score.mean():.6f}" for score in scores))]
    rows.extend((f"{(j + 1) * step_s:.2f}", *(f"{score[j]:.6f}" for score in scores)) for j in range(HORIZON))
    labels = ("Persistence", *(label for label, *_ in STUDIES))
    print_table(("Lookahead (s)", *labels), rows)

    TABLES.mkdir(parents=True, exist_ok=True)
    save_table(
        TABLES / "offline_predictor_selection.tex",
        r"@{}llp{0.48\linewidth}@{}",
        ("Predictor", "EEG history", "Selected settings"),
        configurations,
    )
    save_table(
        TABLES / "offline_predictor_matched_scores.tex",
        "@{}lr@{}",
        ("Predictor", "Mean log-power MSE"),
        [(label, f"{score.mean():.4f}") for label, score in zip(labels, scores, strict=True)],
    )
    save_lookahead_plot(step_s, labels, scores)


def main() -> None:
    """Print the benchmark results and generate the thesis figure and tables."""
    selection = []
    configurations = []
    models = []
    reference_manifest = CAMPAIGN / STUDIES[0][2] / "data_manifest.json"
    recordings = json.loads(reference_manifest.read_text(encoding="utf-8"))["validation_files"]
    for label, directory, manifest_dir, objective in STUDIES:
        study = CAMPAIGN / directory
        trial, value, completed, attempted, dmd = read_study(study)
        checkpoint = study / f"trial_{trial}" / "model.npz"
        meta, arrays = load_checkpoint(checkpoint)
        model = inference_from_checkpoint(meta, arrays)
        if isinstance(model, OBSERVABLE_MODELS):
            geometry = model.geometry
            samples = geometry.sample_support_steps(FS) + (model.n_y - 1) * geometry.n_hop
            history = f"{model.n_y} {'frame' if model.n_y == 1 else 'frames'}"
        else:
            samples = model.n_y
            history = f"{model.n_y} samples"
        configurations.append((label, f"{history} ({samples / FS:.2f} s)", describe_settings(meta, arrays, dmd)))
        manifest_path = CAMPAIGN / manifest_dir / "data_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(model, OBSERVABLE_MODELS) and manifest["validation_files"] != recordings:
            msg = f"Validation recordings differ from the reference manifest in {manifest_path}."
            raise ValueError(msg)
        models.append(model)
        selection.append(
            (
                label,
                f"{completed}/{attempted}",
                str(trial),
                str(model.n_y),
                str(model.n_u),
                manifest["data_path"],
                objective,
                f"{value:.6f}",
                checkpoint.relative_to(REPO).as_posix(),
            )
        )
    print("## Archived search results")
    print_table(
        (
            "Predictor",
            "Completed / attempted",
            "Trial",
            "n_y",
            "n_u",
            "Fit dataset",
            "Objective",
            "Value",
            "Checkpoint",
        ),
        selection,  # ty: ignore[invalid-argument-type]
    )

    reference = models[0]
    if not isinstance(reference, OBSERVABLE_MODELS):
        msg = "The first study must supply the reference Observable geometry."
        raise TypeError(msg)
    geometry = reference.geometry
    hop = geometry.n_hop
    support = geometry.sample_support_steps(FS)
    start = max(
        model.priming_steps
        if isinstance(model, OBSERVABLE_MODELS)
        else math.ceil((model.priming_steps - (support - 1)) / hop)
        for model in models
    )
    trajectories = load_recordings(recordings, geometry)
    persistence, n_windows = persistence_mse(trajectories, geometry, start)
    scores = [persistence]
    for model in models:
        result = evaluate_spectral_free_run(model, trajectories, HORIZON, geometry=geometry, fs=FS, start_frame=start)
        scores.append(result.per_step)

    step_s = hop / FS
    print("## Matched validation log-power MSE")
    print(f"{len(recordings)} recordings and {n_windows} common forecast windows.")
    print(f"Recording list: {reference_manifest.relative_to(REPO).as_posix()}")
    print(f"First origin: zero-based Frame {start}. Horizon: {HORIZON} Frames ({HORIZON * step_s:.2f} s).")
    print(f"Spectral geometry: {geometry.model_dump_json()}")
    print("Each row averages squared error over forecast origins, channels and frequency bins.")
    print()
    publish_benchmark(configurations, step_s, scores)


if __name__ == "__main__":
    main()
