"""
Lambda sweep figure of the thesis (lambda_ceiling_all), from
results/knn/<dataset>/knn_ceiling_results.csv. Every trained catalyzer is scored by an
exhaustive search in its space, recall@10 against the original-space ground truth. One panel
per dataset, one curve per output dimension, and the peak of each curve, the selected
lambda*, is marked and labeled.

Usage (from the repository root): python plots/generate_lambda_ceiling_figure.py
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from common import DIM_COLORS, ORIGINAL_DIM, RESULTS_ROOT, save

DATASETS = {"bigann": "BigANN-1M", "deep1b": "Deep1B-1M", "laion": "LAION 1M subset"}


def load_ceiling(ds):
    """Catalyzer rows with their lambda, sorted, and the recall of the original space."""
    df = pd.read_csv(RESULTS_ROOT / "knn" / ds / "knn_ceiling_results.csv")
    original = df[df["space"] == "original"]["recall@10"].iloc[0]
    cat = df[df["space"].str.startswith("catalyzer")].copy()
    # catalyzer64_0_0002 -> 0.0002
    cat["lambda"] = cat["space"].map(lambda s: float(s.split("_", 1)[1].replace("_", ".")))
    return cat.sort_values(["dim", "lambda"]), original


def main():
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(16, 24))
    dims_seen = set()
    for ax, (ds, title) in zip(axes, DATASETS.items()):
        cat, original = load_ceiling(ds)
        dims = sorted(cat["dim"].unique())
        dims_seen |= set(dims)
        peaks = {d: cat.loc[cat[cat["dim"] == d]["recall@10"].idxmax()] for d in dims}
        # Rank the curves by peak height, so the lambda* labels alternate above and below
        # the peaks and stay off the neighboring curve.
        rank = {d: i for i, d in enumerate(sorted(dims, key=lambda d: peaks[d]["recall@10"], reverse=True))}

        for dim in dims:
            sub = cat[cat["dim"] == dim]
            color = DIM_COLORS[dim]
            ax.plot(sub["lambda"], sub["recall@10"], marker="o", linewidth=2.8, markersize=10,
                    color=color, label=f"Catalyzer {dim}D")
            peak = peaks[dim]
            ax.plot(peak["lambda"], peak["recall@10"], marker="o", markersize=18, color=color,
                    markeredgecolor="black", markeredgewidth=2.0, linestyle="none", zorder=5)
            above = rank[dim] % 2 == 0
            ax.annotate(
                f"{peak['lambda']:g}",
                xy=(peak["lambda"], peak["recall@10"]),
                xytext=(0, 16 if above else -16), textcoords="offset points",
                ha="center", va="bottom" if above else "top", fontsize=26, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"),
            )

        ax.axhline(original, color="black", linestyle="--", linewidth=2.4)
        ax.set_xscale("log")
        ax.set_ylim(0.2, 1.05)
        ax.set_title(f"{title}, original {ORIGINAL_DIM[ds]}D", fontsize=34)
        ax.set_xlabel("Regularization parameter λ (log scale)", fontsize=30)
        ax.set_ylabel("Recall@10", fontsize=32)
        ax.tick_params(axis="both", labelsize=34)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    # One legend above the panels, because the curves span the full width of each panel
    handles = [Line2D([], [], color=DIM_COLORS[d], marker="o", linewidth=2.8, markersize=10,
                      label=f"Catalyzer {d}D") for d in sorted(dims_seen)]
    handles.append(Line2D([], [], color="black", linestyle="--", linewidth=2.4, label="Original space"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, fontsize=28, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.945], h_pad=3.0)
    save(fig, "lambda_ceiling_all")


if __name__ == "__main__":
    main()
