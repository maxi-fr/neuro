from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Measured after \maketitle with the document options in thesis.tex (170 mm).
TEXTWIDTH_PT = 483.69688
INCHES_PER_PT = 1.0 / 72.27
TEXTWIDTH_IN = TEXTWIDTH_PT * INCHES_PER_PT  # ~6.693 inches
# PGF passes these values to \fontsize in TeX points, matching the TikZ styles.
FONT_LARGE = 9.0  # Main figure header
FONT_TITLE = 9.0  # Subplot title (e.g. "(a) Excitability...")
FONT_LABEL = 9.0  # Axis labels (e.g. "Time / s", "$y_\\mathrm{LFP}$ / mV")
FONT_TICK = 8.0  # Tick labels (numbers on axes)
FONT_LEGEND = 8.0  # Legend entries
FONT_ANNOTATION = 8.0  # In-plot descriptions, arrows, and labels
FONT_SMALL = 8.0  # Fine details, tight index tags (e.g. "$q = 0$")


def setup_style() -> None:
    """Render fixed-size figures with the thesis's pdfLaTeX fonts and 9/8 pt text."""
    mpl.use("pgf")

    plt.rcParams.update(
        {
            # Use the same font setup as tudapub, including Computer Modern math.
            "font.family": "serif",
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage{tudafonts}\usepackage{amsmath,amssymb}",
            # Sizing hierarchy
            "font.size": FONT_LABEL,
            "figure.titlesize": FONT_LARGE,
            "axes.titlesize": FONT_TITLE,
            "axes.labelsize": FONT_LABEL,
            "legend.fontsize": FONT_LEGEND,
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            # Plot styling
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.35,
            "lines.linewidth": 0.8,
            # Output export
            "savefig.bbox": None,
        }
    )


def figsize(
    width_ratio: float = 1.0,
    height_in: float | None = None,
    aspect_ratio: float = 0.6,
) -> tuple[float, float]:
    """Size the complete PDF canvas, including labels and margins, for native inclusion.

    Parameters
    ----------
    width_ratio : float, default=1.0
        Fraction of textwidth to occupy (1.0 = full text width, ~6.69 in).
    height_in : float, optional
        Target height in inches. If omitted, aspect_ratio is used.
    aspect_ratio : float, default=0.6
        Height / width ratio. Default 0.6 produces ~4.0 in height at full width.

    Returns
    -------
    tuple[float, float]
        Width and height in inches.
    """
    w = TEXTWIDTH_IN * width_ratio
    h = height_in if height_in is not None else w * aspect_ratio
    return (w, h)


# Automatically apply style on import
setup_style()
