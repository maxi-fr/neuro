"""Generate shooting.pdf: direct single shooting against direct multiple shooting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neuro.types import FloatArray  # noqa: TC001 -- runtime import keeps the script runnable standalone

DT = 0.1
HORIZON = 9
N_INTERVALS = 3
A, B = 0.93, 0.35
GAIN = 0.40
U_MAX = 1.0
X0 = 1.5
GAPS = (0.55, -0.45)
OUT = Path(__file__).with_suffix(".pdf")


def _inputs() -> FloatArray:
    """Input iterate shared by both panels, clipped to the bound ``U_MAX``.

    Returns
    -------
    u : ``(HORIZON,)`` inputs of the drawn solver iterate.
    """
    x = X0
    u = np.empty(HORIZON)
    for j in range(HORIZON):
        u[j] = np.clip(-GAIN * x, -U_MAX, U_MAX)
        x = A * x + B * u[j]
    return u


def _rollout(x0: float, u: FloatArray) -> FloatArray:
    """Simulate the scalar linear model from ``x0`` under the inputs ``u``.

    Returns
    -------
    x : ``(len(u) + 1,)`` states including the initial one.
    """
    x = np.empty(len(u) + 1)
    x[0] = x0
    for j, u_j in enumerate(u):
        x[j + 1] = A * x[j] + B * u_j
    return x


def _pieces(u: FloatArray) -> list[FloatArray]:
    """Per-interval state pieces of a non-converged multiple-shooting iterate.

    Each interval starts at its own decision variable, offset from the previous piece's end
    by the corresponding entry of ``GAPS``, so the continuity constraints are still violated.

    Returns
    -------
    pieces : ``N_INTERVALS`` arrays of shape ``(HORIZON // N_INTERVALS + 1,)``.
    """
    length = HORIZON // N_INTERVALS
    pieces = []
    start = X0
    for i in range(N_INTERVALS):
        piece = _rollout(start, u[i * length : (i + 1) * length])
        pieces.append(piece)
        if i < len(GAPS):
            start = piece[-1] + GAPS[i]
    return pieces


def main() -> None:
    """Write the two-panel figure contrasting single with multiple shooting."""
    u = _inputs()
    t = np.arange(HORIZON + 1) * DT
    single = _rollout(X0, u)
    pieces = _pieces(u)
    length = HORIZON // N_INTERVALS

    fig, (ax_s, ax_m) = plt.subplots(2, 1, figsize=(6.2, 4.4), sharex=True, constrained_layout=True)

    ax_s.plot(t, single, lw=1.6, color="C0", marker="o", ms=3.5, label="predicted trajectory")
    ax_s.plot(t[0], single[0], marker="s", ms=6, color="C3", ls="none", label=r"$\hat{\mathbf{x}}_k$")
    ax_s.set_title("Direct single shooting", fontsize=9, loc="left")
    ax_s.annotate(
        "only the inputs are decision variables,\n"
        "so the dynamics hold at every iterate and\n"
        "any iterate is a valid trajectory",
        xy=(0.97, 0.94),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7,
    )

    for i, piece in enumerate(pieces):
        t_piece = t[i * length : i * length + length + 1]
        ax_m.plot(t_piece, piece, lw=1.6, color="C0", marker="o", ms=3.5)
        ax_m.plot(
            t_piece[0],
            piece[0],
            marker="o",
            ms=7,
            mfc="white",
            mec="C2",
            mew=1.4,
            ls="none",
            label=r"decision variable $\mathbf{x}_{j|k}$" if i == 0 else None,
        )
    for i, gap in enumerate(GAPS):
        t_node = t[(i + 1) * length]
        end = pieces[i][-1]
        ax_m.annotate(
            "",
            xy=(t_node, end + gap),
            xytext=(t_node, end),
            arrowprops={"arrowstyle": "<->", "lw": 1.0, "color": "C3"},
        )
    ax_m.plot(t[0], X0, marker="s", ms=6, color="C3", ls="none", label=r"$\hat{\mathbf{x}}_k$")
    ax_m.set_title("Direct multiple shooting", fontsize=9, loc="left")
    ax_m.annotate(
        "gaps: continuity equality constraints, closed only at\nconvergence; the drawn iterate is not a valid trajectory",
        xy=(t[length], pieces[0][-1] + GAPS[0]),
        xytext=(0.97, 0.94),
        textcoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7,
        color="C3",
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "C3"},
    )
    for t_node in t[length::length][: N_INTERVALS - 1]:
        ax_m.axvline(t_node, ls=":", lw=0.7, color="C2")

    for ax in (ax_s, ax_m):
        ax.set_ylabel("State $x$ / a.u.")
        ax.legend(fontsize=7, loc="lower left")
        ax.set_ylim(-0.6, 2.8)
        ax.grid(visible=True, lw=0.4, alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_xlim(0.0, HORIZON * DT)

    ax_m.set_xlabel("Time / s")

    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
