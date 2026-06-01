import datetime
import json
import math
import pickle
import subprocess
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt


class ThesisPlotSaver:
    """For saving plots and metadata."""

    def __init__(self, textwidth_pt: float = 418.25, base_dir: str = "plots") -> None:
        """
        Initialize the saver with LaTeX document dimensions and global styles.

        Parameters
        ----------
        textwidth_pt: The exact width of your LaTeX text block in points.

        base_dir: The root directory where all plot folders will be created.
        """
        self.textwidth_pt: float = textwidth_pt
        self.base_dir: Path = Path(base_dir)
        self._configure_matplotlib()

    def _configure_matplotlib(self) -> None:
        """Set up the global matplotlib rcParams for crisp, native LaTeX output."""
        mpl.rcParams.update(
            {
                "backend": "pgf",
                "pgf.texsystem": "pdflatex",
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": [],
                "font.sans-serif": [],
                "font.monospace": [],
                "font.size": 11,
                "axes.labelsize": 11,
                "legend.fontsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "pgf.preamble": f"{r'\usepackage{amsmath}'}\n{r'\usepackage{amssymb}'}",
            }
        )

    def _get_git_commit(self) -> str:
        """Query the local git repository for the current commit hash identifier."""
        try:
            commit_hash: str = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],  # noqa: S607
                    stderr=subprocess.DEVNULL,
                )
                .decode("ascii")
                .strip()
            )

        except (subprocess.SubprocessError, FileNotFoundError):
            return "not_a_git_repository"
        else:
            return commit_hash

    def calculate_dimensions(self, fraction: float = 1.0, subplots: tuple[int, int] = (1, 1)) -> tuple[float, float]:
        """Compute the golden-ratio dimensions for a plot to guarantee font sizes don't distort when embedded into LaTeX."""
        fig_width_pt: float = self.textwidth_pt * fraction
        inches_per_pt: float = 1 / 72.27
        golden_ratio: float = (math.sqrt(5) - 1) / 2

        fig_width_in: float = fig_width_pt * inches_per_pt
        fig_height_in: float = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
        return (fig_width_in, fig_height_in)

    def save(self, fig: plt.Figure, name: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Save the figure object, PGF graphic, and JSON metadata into a target folder.

        If the directory 'plots/<name>' already exists, it auto-increments: 'plots/<name>_01'.
        """
        target_dir: Path = self.base_dir / name
        final_name: str = name

        if target_dir.exists():
            counter: int = 1
            while True:
                final_name = f"{name}_{counter:02d}"
                target_dir = self.base_dir / final_name
                if not target_dir.exists():
                    break
                counter += 1

        target_dir.mkdir(parents=True, exist_ok=True)
        base_filepath: Path = target_dir / final_name

        if metadata is None:
            metadata = {}

        metadata["git_commit"] = self._get_git_commit()
        metadata["generated_at"] = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")

        json_path: Path = base_filepath.with_suffix(".json")

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        pkl_path: Path = base_filepath.with_suffix(".pkl")
        with pkl_path.open("wb") as f:
            pickle.dump(fig, f)

        pgf_path: Path = base_filepath.with_suffix(".pgf")
        fig.savefig(pgf_path, bbox_inches="tight")

        png_path: Path = base_filepath.with_suffix(".png")
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
