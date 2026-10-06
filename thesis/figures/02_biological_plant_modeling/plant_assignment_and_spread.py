from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, PathPatch
from matplotlib.path import Path as MplPath

if TYPE_CHECKING:
    from matplotlib.axes import Axes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style

from neuro.connectome import Connectome
from neuro.seizure import EZ_REGIONS, PZ_REGIONS, spread_profile_from_lfp

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
LOG = ROOT / "artifacts/baseline_comparison/runs/uncontrolled_s7000/log.npz"


def _groups(connectome: Connectome) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint EZ, PZ, left-background, and right-background masks."""
    labels = connectome.region_labels
    ez = np.isin(labels, EZ_REGIONS)
    pz = np.isin(labels, PZ_REGIONS)
    left = ~connectome.hemispheres
    return ez, pz, left & ~ez & ~pz, ~left


def _draw_head_inset(ax: Axes, *, frontal: bool) -> None:
    """Draw a small head silhouette in the same projection as the atlas panel."""
    inset = ax.inset_axes((0.02, 0.02, 0.23, 0.29), facecolor="white", zorder=5)
    inset.set(xlim=(0, 1), ylim=(0, 1))
    inset.set_aspect("equal")
    inset.axis("off")
    inset.patch.set_visible(True)
    inset.patch.set_facecolor("white")
    stroke = "#4f4f4f"
    if frontal:
        inset.add_patch(Ellipse((0.5, 0.57), 0.62, 0.76, fill=False, edgecolor=stroke, linewidth=1))
        inset.add_patch(Ellipse((0.17, 0.55), 0.11, 0.2, fill=False, edgecolor=stroke, linewidth=0.8))
        inset.add_patch(Ellipse((0.83, 0.55), 0.11, 0.2, fill=False, edgecolor=stroke, linewidth=0.8))
        inset.plot((0.42, 0.58), (0.65, 0.65), linestyle="none", marker=".", color=stroke, markersize=2)
        inset.plot((0.5, 0.5), (0.58, 0.47), color=stroke, linewidth=0.7)
        inset.plot((0.43, 0.57), (0.38, 0.38), color=stroke, linewidth=0.7)
        return

    vertices = [
        (0.17, 0.52),
        (0.22, 0.78),
        (0.42, 0.9),
        (0.65, 0.88),
        (0.75, 0.76),
        (0.76, 0.64),
        (0.91, 0.57),
        (0.76, 0.51),
        (0.78, 0.4),
        (0.69, 0.27),
        (0.6, 0.19),
        (0.48, 0.22),
        (0.37, 0.18),
        (0.22, 0.34),
        (0.17, 0.52),
    ]
    codes = [MplPath.MOVETO, *([MplPath.LINETO] * (len(vertices) - 2)), MplPath.CLOSEPOLY]
    inset.add_patch(PathPatch(MplPath(vertices, codes), fill=False, edgecolor=stroke, linewidth=1))
    inset.add_patch(Ellipse((0.43, 0.57), 0.15, 0.2, fill=False, edgecolor=stroke, linewidth=0.8))
    inset.plot(0.69, 0.68, marker=".", color=stroke, markersize=2)


def plot_assignment(connectome: Connectome) -> None:
    """Plot regional excitability on atlas centroids with head-view insets."""
    ez, pz, left_bg, right_bg = _groups(connectome)
    centres = connectome.centres
    fig, axes = plt.subplots(
        1,
        2,
        figsize=style.figsize(height_in=3.1),
        gridspec_kw={"width_ratios": (165, 130)},
        constrained_layout=True,
    )
    categories = (
        (right_bg, "Right background", "#b8b8b8", 13),
        (left_bg, "Left background", "#808080", 13),
        (pz, "Propagation zone", "#df8b25", 48),
        (ez, "Epileptogenic zone", "#a22948", 48),
    )
    views = (
        (0, 2, "Sagittal projection", (-80, 85)),
        (1, 2, "Frontal projection", (-65, 65)),
    )
    names = ("Anterior coordinate / mm", "Left coordinate / mm", "Superior coordinate / mm")
    for ax, (x_dim, y_dim, title, x_limits) in zip(axes, views, strict=True):
        for mask, label, color, size in categories:
            ax.scatter(
                centres[mask, x_dim],
                centres[mask, y_dim],
                s=size,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                label=label,
                zorder=3,
            )
        ax.set(xlabel=names[x_dim], ylabel=names[y_dim], title=title, xlim=x_limits, ylim=(-65, 55))
        ax.grid(visible=True, linewidth=0.4, alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_aspect("equal", adjustable="box")
        _draw_head_inset(ax, frontal=x_dim == 1)
    axes[0].legend(loc="upper left", fontsize=style.FONT_LEGEND, frameon=True)
    fig.savefig(OUT / "ez_pz_assignment.pdf")
    plt.close(fig)


def plot_spread(connectome: Connectome) -> None:
    """Plot cumulative regional recruitment from the saved uncontrolled run."""
    with np.load(LOG) as log:
        times = log["dynamics.t"]
        lfp = log["dynamics.lfp"]
    dt = float(np.median(np.diff(times)))
    profile = spread_profile_from_lfp(lfp.T, dt)
    ez, pz, left_bg, right_bg = _groups(connectome)
    fig, ax = plt.subplots(figsize=style.figsize(height_in=2.8), constrained_layout=True)
    curves = (
        (ez, "Epileptogenic zone (3)", "#a22948"),
        (pz, "Propagation zone (2)", "#df8b25"),
        (left_bg, "Other left regions (33)", "#366b9a"),
        (right_bg, "Right regions (38)", "#555555"),
    )
    for mask, label, color in curves:
        count = np.count_nonzero(profile.onsets[mask, None] <= profile.times[None, :], axis=0)
        ax.step(profile.times, count, where="post", label=label, color=color, linewidth=1.2)
    ax.axvline(10.5, color="black", linestyle=":", linewidth=0.8)
    ax.set(xlim=(0, 12), ylim=(0, 34), xlabel="Time / s", ylabel="Recruited regions")
    ax.set_xticks(np.arange(0, 13, 2))
    ax.grid(visible=True, linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True)
    fig.savefig(OUT / "uncontrolled_propagation.pdf")
    plt.close(fig)


def main() -> None:
    """Generate the assignment and uncontrolled-spread vector figures."""
    connectome = Connectome.from_config({"speed": 50.0, "K": 0.60})
    plot_assignment(connectome)
    plot_spread(connectome)


if __name__ == "__main__":
    main()
