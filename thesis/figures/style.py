from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

TEXTWIDTH_PT = 483.69687
INCHES_PER_PT = 1.0 / 72.27
TEXTWIDTH_IN = TEXTWIDTH_PT * INCHES_PER_PT  # ~6.693 inches
FONT_DIR = Path(__file__).parent / "fonts"

# Standard font size hierarchy (in points) matching thesis typography (11pt base)
FONT_LARGE = 10.0  # Suptitle / main figure header
FONT_TITLE = 9.0  # Subplot title (e.g. "(a) Excitability...")
FONT_LABEL = 9.0  # Axis labels (e.g. "Time / s", "$y_\\mathrm{LFP}$ / mV")
FONT_TICK = 8.0  # Tick labels (numbers on axes)
FONT_LEGEND = 7.5  # Legend entries
FONT_ANNOTATION = 7.5  # In-plot descriptions, arrows, and labels
FONT_SMALL = 6.5  # Fine details, tight index tags (e.g. "$q = 0$")



def _register_fonts() -> None:
    """Register font files located in thesis/figures/fonts/."""
    if FONT_DIR.is_dir():
        for font_file in FONT_DIR.glob("*.otf"):
            fm.fontManager.addfont(str(font_file))


def setup_style() -> None:
    """Apply thesis styling defaults to matplotlib.

    Uses XCharter serif font extracted from the thesis, standard font size hierarchy,
    refined line/grid weights, and tight bounding box export.
    """
    _register_fonts()

    plt.rcParams.update({
        # Typography matching tudapub (XCharter / serif)
        "font.family": "serif",
        "font.serif": ["XCharter", "Charter", "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
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
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def figsize(
    width_ratio: float = 1.0,
    height_in: float | None = None,
    aspect_ratio: float = 0.6,
) -> tuple[float, float]:
    """Calculate figure dimensions in inches matching thesis column width.

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
