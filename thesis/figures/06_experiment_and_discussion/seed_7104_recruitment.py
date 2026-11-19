import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from experiments.observable_predictability.archive import load_evaluation_bundle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style

FIGURES = Path(__file__).resolve().parent
ARTIFACTS = FIGURES.parents[1] / "artifacts"
CAMPAIGN = FIGURES.parents[2] / "artifacts"
WAVEFORM = CAMPAIGN / "sweep_2026_11/waveform_matched_geometry/final/waveform_mlp/evaluation/runs/waveform_mlp_s7104/trace.npz"
SPECTRAL = CAMPAIGN / "sweep_2026_11/observable_benchmark/runs/obs_cnn_s7104/log.npz"


def plot_seed_7104() -> None:
    """Show the handover interval and situate the forecast against measured seizure state."""
    fig, (recruitment, penalty) = plt.subplots(1, 2, layout="constrained", figsize=style.figsize(height_in=2.75))
    endpoints = np.arange(1, 3.01, 0.25)
    with np.load(WAVEFORM) as trace:
        np.testing.assert_allclose(trace["spread_t"][:len(endpoints)] + 0.5, endpoints)
        waveform_counts = (trace["spread_ptp"][:, :len(endpoints)] > 5).sum(axis=0)
    with np.load(SPECTRAL) as trace:
        lfp = trace["dynamics.lfp"][:30001]
        stops = np.rint(endpoints * 10000).astype(int)
        spectral_counts = [(np.ptp(lfp[stop - 10000:stop], axis=0) > 5).sum() for stop in stops]
    recruitment.axvspan(1.56, 1.62, color="0.88", zorder=0)
    recruitment.plot(endpoints, waveform_counts, ".-", color="C1", label="Waveform MLP")
    recruitment.plot(endpoints, spectral_counts, ".-", color="C0", label="Spectral CNN")
    for time, color, marker in ((1.56, "C2", "o"), (1.62, "C4", "s")):
        with np.load(ARTIFACTS / f"seed_7104_runaway_check/runaway_7104_switch_{time:.2f}.npz") as trace:
            t = trace["endpoints"]
            counts = (trace["ptp"] > 5).sum(axis=1)
        show = (t >= np.floor(time * 4) / 4) & (t <= endpoints[-1])
        recruitment.plot(t[show], counts[show], "--", color=color, marker=marker, markersize=3.5,
                         markerfacecolor="white", label=f"CNN from {time:.2f} s")
    recruitment.set(title="Closed-loop recruitment", xlabel="Simulation time (s)", ylabel="Seizing regions",
                    xticks=[1, 1.5, 2, 2.5, 3], xlim=(0.96, 3.04), ylim=(0, 40), yticks=[0, 10, 20, 30, 40])
    recruitment.legend(loc="upper left", frameon=False)

    reference = load_evaluation_bundle(CAMPAIGN / "observable_predictability_expanded")
    points = reference.seizure_relevance["waveform"].finite_pairs()
    ceiling = np.quantile(points[points[:, 0] <= 7 / 76, 1], 0.95)
    data = np.genfromtxt(ARTIFACTS / "seed_7104_penalty_state/penalty_frames.csv",
                         delimiter=",", names=True, dtype=None, encoding="utf-8")
    plan = data[(data["plan"] == "waveform_plan") & np.isclose(data["decision_s"], 1.98)]
    assert len(plan) == 17
    state = plan["seizure_state"]
    penalty.scatter(*points.T, s=1.6, alpha=0.13, color="0.55", linewidth=0, rasterized=True)
    penalty.axhline(ceiling, linestyle=":", color="0.35", label="Cost ceiling")
    penalty.plot(state, plan["predicted_cost"], "o", markersize=3.8, color="C3", label="Forecast")
    penalty.plot(state, plan["replayed_cost"], "o", markersize=3.8, color="C0", label="Plant replay")
    point = plan[np.isclose(plan["endpoint_s"], 2.52)][0]
    x = point["seizure_state"]
    low, high = point["predicted_cost"], point["replayed_cost"]
    penalty.plot([x, x], [low, high], color="0.35", linewidth=0.7, zorder=1)
    penalty.annotate("2.52 s", (x, np.sqrt(low * high)), xytext=(-4, 0), textcoords="offset points",
                     ha="right", va="center", fontsize=style.FONT_ANNOTATION)
    penalty.set(title="Waveform plan issued at 1.98 s", xlabel="Seizure state in replay",
                ylabel="60 ms waveform cost", yscale="log", xlim=(0.018, 0.47), xticks=[0.1, 0.2, 0.3, 0.4])
    penalty.legend(loc="upper left", frameon=False)
    for ax in (recruitment, penalty):
        ax.set_axisbelow(True)
        ax.grid(axis="y")
    fig.savefig(FIGURES / "seed_7104_recruitment.pdf", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plot_seed_7104()
