from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style

from neuro.connectome import Connectome

OUT = Path(__file__).resolve().parent / "connectome_matrices.pdf"


def main() -> None:
    """Plot the plant's structural weights and connected-pair delays."""
    connectome = Connectome.from_config({"speed": 50.0, "K": 0.60})
    connected = connectome.weights > 0
    panels = (
        (connectome.weights, "Connectivity weight", "Weight", "Blues", 3.0),
        (connectome.delays, "Conduction delay", "Delay / ms", "Oranges", 3.0),
    )
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(height_in=3.55), constrained_layout=True)
    for ax, (values, title, bar_label, palette, maximum) in zip(axes, panels, strict=True):
        color_map = plt.get_cmap(palette).copy()
        color_map.set_bad("white")
        image = ax.imshow(np.ma.masked_where(~connected, values), cmap=color_map, vmin=0, vmax=maximum)
        ax.set(title=title, xlabel="Sending region index", ylabel="Receiving region index")
        ax.set_xticks([0, 19, 38, 57, 75])
        ax.set_yticks([0, 19, 38, 57, 75])
        ax.set_xticks(np.arange(9.5, 76, 10), minor=True)
        ax.set_yticks(np.arange(9.5, 76, 10), minor=True)
        ax.grid(which="minor", color="#777777", linewidth=0.35, alpha=0.5)
        ax.axhline(37.5, color="#333333", linewidth=0.75)
        ax.axvline(37.5, color="#333333", linewidth=0.75)
        fig.colorbar(image, ax=ax, label=bar_label, fraction=0.046, pad=0.025)
    fig.savefig(OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
