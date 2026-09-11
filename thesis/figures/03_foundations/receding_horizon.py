"""Generate receding_horizon.pdf: the receding horizon principle over three control steps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neuro.types import FloatArray  # noqa: TC001 -- runtime import keeps the script runnable standalone

DT = 0.1
HORIZON = 8
N_STEPS = 3
N_TOTAL = 20
A, B = 0.95, 0.12
GAIN = 1.2
U_MAX = 1.0
Y_REF = 1.0
Y0 = 0.0
DISTURBANCE = (
    -0.11,
    0.08,
    -0.13,
    0.06,
    -0.1,
    0.11,
    -0.07,
    0.05,
    -0.05,
    0.04,
    -0.03,
    0.03,
    -0.02,
    0.02,
    -0.01,
    0.01,
    0.0,
    0.0,
    0.0,
    0.0,
)
U_SS = Y_REF * (1.0 - A) / B
STRIDE = 3
COLORS = ("C1", "C2", "C4")
OUT = Path(__file__).with_suffix(".pdf")


def _plan(y0: float) -> tuple[FloatArray, FloatArray]:
    """Input sequence and predicted trajectory of one horizon, from state ``y0``.

    The planner combines the steady-state input holding the reference with proportional
    feedback, so it saturates at the bound while the error is large and settles on the
    reference without offset. It uses the nominal, disturbance-free model, which is why its
    prediction departs from what the plant then does.

    Returns
    -------
    u : ``(HORIZON,)`` planned inputs, clipped to the bound ``U_MAX``.
    y : ``(HORIZON + 1,)`` predicted states including the initial one.
    """
    u = np.empty(HORIZON)
    y = np.empty(HORIZON + 1)
    y[0] = y0
    for j in range(HORIZON):
        u[j] = np.clip(U_SS + GAIN * (Y_REF - y[j]), -U_MAX, U_MAX)
        y[j + 1] = A * y[j] + B * u[j]
    return u, y


def _closed_loop() -> tuple[FloatArray, FloatArray, list[tuple[int, FloatArray, FloatArray]]]:
    """Realised trajectory, applied inputs, and the first ``N_STEPS`` plans.

    Returns
    -------
    y : ``(N_TOTAL + 1,)`` realised states under the disturbed plant.
    u : ``(N_TOTAL,)`` applied inputs, the first element of each plan.
    plans : the ``(u, y)`` pair of each of the first ``N_STEPS`` horizons.
    """
    y = [Y0]
    u: list[float] = []
    plans = []
    for k in range(N_TOTAL):
        u_plan, y_plan = _plan(y[-1])
        if k % STRIDE == 0 and k // STRIDE < N_STEPS:
            plans.append((k, u_plan, y_plan))
        u.append(u_plan[0])
        y.append(A * y[-1] + B * u_plan[0] + DISTURBANCE[k])
    return np.array(y), np.array(u), plans


def main() -> None:
    """Write three successive plans with the scalar input limits marked explicitly."""
    y_real, u_real, plans = _closed_loop()
    t_real = np.arange(len(y_real)) * DT

    fig, (ax_y, ax_u) = plt.subplots(
        2, 1, figsize=(6.2, 4.2), sharex=True, height_ratios=(2, 1), constrained_layout=True
    )

    for i, (k, u_plan, y_plan) in enumerate(plans):
        t_plan = (k + np.arange(HORIZON + 1)) * DT
        colour = COLORS[i]
        ax_y.plot(t_plan, y_plan, ls="--", lw=1.0, color=colour, marker="o", ms=2.5, label=f"plan at $k = {k}$")
        ax_u.step(t_plan[:-1], u_plan, where="post", ls="--", lw=1.0, color=colour)
        ax_y.axvline(k * DT, ls=":", lw=0.7, color=colour)
        ax_u.axvline(k * DT, ls=":", lw=0.7, color=colour)

    ax_y.plot(t_real, y_real, lw=1.8, color="C0", marker="o", ms=3.5, label="realised")
    ax_u.step(t_real[:-1], u_real, where="post", lw=1.8, color="C0")

    ax_y.axhline(Y_REF, lw=0.8, ls="-.", color="0.4")
    ax_y.text(0.02, Y_REF, "reference", fontsize=7, color="0.4", va="bottom")
    ax_y.set_ylabel("State $x$ / a.u.")
    ax_y.set_ylim(-0.12, 1.42)
    ax_y.legend(fontsize=7, ncols=2, loc="upper right")
    ax_y.annotate(
        "horizon of length $H$",
        xy=(HORIZON * DT, plans[0][2][-1]),
        xytext=(HORIZON * DT + 0.06, 0.26),
        fontsize=7,
        color=COLORS[0],
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": COLORS[0]},
    )

    for bound in (-U_MAX, U_MAX):
        ax_u.axhline(bound, ls=":", lw=0.9, color="C3")
    ax_u.text(0.02, U_MAX, r"$u_\mathrm{max}$", fontsize=8, color="C3", va="bottom")
    ax_u.text(0.02, -U_MAX, r"$-u_\mathrm{max}$", fontsize=8, color="C3", va="top")
    ax_u.set_ylabel("Input $u$ / a.u.")
    ax_u.set_xlabel("Time / s")
    ax_u.set_ylim(-1.5 * U_MAX, 1.5 * U_MAX)

    for ax in (ax_y, ax_u):
        ax.grid(visible=True, lw=0.4, alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_xlim(0.0, t_real[-1])

    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
