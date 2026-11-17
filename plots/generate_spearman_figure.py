"""
Spearman figures of the thesis, partition quality against index performance, one build
configuration at a time:

  spearman_consistency_recall_bigann
  spearman_consistency_recall_deep1b
  spearman_consistency_recall_laion

For every space of a partitioning index at one build configuration, the mean fractional
k-NN consistency is paired with the recall@10 reached at a fixed budget of distance
computations. The Spearman rank correlation over the spaces of that configuration answers
whether the space with higher consistency also gives higher recall. Configurations are kept
apart, because a larger bucket count lowers consistency and raises recall at a fixed budget.
The recall at the budget is interpolated on the measured recall-cost curve, linearly in
log10(distance computations). HNSW has no buckets and is left out. The correlations are
also printed, at the budget of the figure and at a second budget.

Usage (from the repository root): python plots/generate_spearman_figure.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr

from common import DIM_COLORS, ORIGINAL_DIM, load_results, output_dim, save, space_name

# Budget of the figure. 10 000 is 1 % of the 1M databases. On LAION every curve starts
# above 14 000, and 70 000 is the budget closest to a measured setting in every configuration.
BUDGETS = {"bigann": 10_000, "deep1b": 10_000, "laion": 70_000}
# Budgets of the printed robustness check
SWEEP = {"bigann": [10_000, 20_000], "deep1b": [10_000, 20_000], "laion": [70_000, 250_000]}

# (panel title, index, result CSVs, build configuration column). n_clusters is the bin
# count per level of NeuralLSH, 100 in both LAION runs, whose leaf counts differ slightly.
INDEXES = [
    ("IVFFlat",            "ivfflat", "ivf_results_*.csv",  "nlist"),
    ("LMI",                "lmi",     "lmi_results_*.csv",  "n_buckets"),
    ("NeuralLSH",          "nlsh",    "nlsh_results_*.csv", "n_clusters"),
    ("Neural Partitioner", "np",      "np_results_*.csv",   "n_bins"),
]

# Marker and line style per build configuration, smallest first. An index with two
# configurations uses the first and the last, so a triangle always marks the largest.
MARKERS = {1: ["o"], 2: ["o", "^"], 3: ["o", "s", "^"]}
LINESTYLES = {1: ["-"], 2: ["-", ":"], 3: ["-", "--", ":"]}


def recall_at_budget(curve, budget):
    """Recall@10 at the budget, interpolated linearly in log10(distance computations).
    None when the budget lies outside the measured range, so the point is dropped."""
    curve = curve.sort_values("dist_computations")
    x = curve["dist_computations"].to_numpy(dtype=float)
    y = curve["recall@10"].to_numpy(dtype=float)
    if budget < x.min() or budget > x.max():
        return None
    return float(np.interp(np.log10(budget), np.log10(x), y))


def collect(ds, index, pattern, config_col, budget):
    """One (consistency, recall at the budget) point per space and build configuration."""
    res = load_results(index, ds, pattern)
    cons = load_results(index, ds, "knn_consistency_*.csv")
    res["space"] = res["space"].map(space_name)
    cons["space"] = cons["space"].map(space_name)

    rows, dropped = [], []
    for (space, config), group in res.groupby(["space", config_col]):
        recall = recall_at_budget(group, budget)
        match = cons[(cons["space"] == space) & (cons[config_col] == config)]
        if recall is None or match.empty:
            dropped.append(f"{space}/{config}" + (" (no consistency row)" if recall is not None else ""))
            continue
        rows.append({"space": space, "dim": output_dim(space), "config": config,
                     "consistency": float(match["knn_consistency_mean"].iloc[0]), "recall": recall})
    if dropped:
        print(f"  [warn] {ds}: budget outside measured range, dropped: {', '.join(dropped)}")
    return pd.DataFrame(rows)


def per_config_rho(df):
    """Spearman over the spaces of each build configuration, smallest configuration first."""
    return [spearmanr(df[df["config"] == c]["consistency"], df[df["config"] == c]["recall"])[0]
            for c in sorted(df["config"].unique())]


def budget_sweep(ds):
    print(f"\n{ds} budget sweep (rho per configuration, smallest first):")
    for budget in SWEEP[ds]:
        line = []
        for title, index, pattern, config_col in INDEXES:
            rhos = per_config_rho(collect(ds, index, pattern, config_col, budget))
            line.append(f"{title} " + " / ".join(f"{r:+.2f}" for r in rhos))
        print(f"  {budget:>7,}: " + " | ".join(line))


def plot(ds):
    budget = BUDGETS[ds]
    # A small canvas, so the fonts print close to the body-text size at the text width
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 12.5))
    dims_present = set()

    for ax, (title, index, pattern, config_col) in zip(axes.flatten(), INDEXES):
        df = collect(ds, index, pattern, config_col, budget)
        dims_present |= set(df["dim"])
        configs = sorted(df["config"].unique())
        rhos = per_config_rho(df)
        for k, config in enumerate(configs):
            sub = df[df["config"] == config].sort_values("dim", key=lambda d: d.replace(0, 10**6))
            # One line per configuration through its spaces in order of output dimension
            ax.plot(sub["consistency"] * 100, sub["recall"], color="0.55",
                    ls=LINESTYLES[len(configs)][k], lw=2.4, zorder=1)
            # Hollow markers, so two spaces on the same point both stay visible
            ax.scatter(sub["consistency"] * 100, sub["recall"], s=260,
                       marker=MARKERS[len(configs)][k], facecolors="none",
                       edgecolors=[DIM_COLORS[d] for d in sub["dim"]], linewidth=3.0, zorder=3)
        print(f"{ds} {title}: rho per configuration " + " / ".join(f"{r:+.2f}" for r in rhos)
              + f"  (n per configuration = {len(df) // len(configs)})")

        # rho in the title, with a typographic minus sign
        ax.set_title(f"{title}\n$\\rho$ = " + " / ".join(f"{r:+.2f}".replace("-", "−") for r in rhos),
                     fontsize=28)
        ax.tick_params(axis="both", labelsize=26)
        ax.xaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 5, 10]))
        ax.grid(True, ls="--", alpha=0.5)

    fig.supxlabel(r"$k$-NN consistency (%)", fontsize=30, y=0.005)
    # Thousands separated by a thin space, as in the thesis text
    fig.supylabel(f"Recall@10 at {budget:,} distance computations".replace(",", " "), fontsize=30, x=0.015)

    # Two centered legends above the panels: the spaces of the dataset, and below them the
    # marker per build configuration. More than three spaces wrap into two rows.
    spaces = [Line2D([0], [0], marker="o", ls="", ms=14, mfc="none", mec=DIM_COLORS[d], mew=3.0,
                     label=f"Original {ORIGINAL_DIM[ds]}D" if d == 0 else f"Catalyzer {d}D")
              for d in sorted(dims_present, key=lambda v: 10**6 if v == 0 else v)]
    configs = [Line2D([0], [0], marker=m, ls=s, ms=13, lw=2.4, mfc="none", mew=3.0, color="0.45", label=text)
               for m, s, text in [("o", "-", "smallest"), ("s", "--", "middle"), ("^", ":", "largest")]]
    rows = 1 if len(spaces) <= 3 else 2
    leg1 = fig.legend(handles=spaces, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                      ncol=-(-len(spaces) // rows), fontsize=24, frameon=False, columnspacing=1.2)
    fig.canvas.draw()
    y0 = leg1.get_window_extent().y0 / fig.bbox.height
    leg2 = fig.legend(handles=configs, loc="upper center", bbox_to_anchor=(0.5, y0),
                      ncol=len(configs), fontsize=24, frameon=False, columnspacing=1.2,
                      title="Build configuration", title_fontsize=24)
    fig.canvas.draw()
    y1 = leg2.get_window_extent().y0 / fig.bbox.height

    plt.tight_layout(rect=[0.02, 0.03, 1, y1 - 0.01], h_pad=2.0, w_pad=2.0)
    # tight_layout leaves a wide gap above the shared x label, so the label is moved up
    # to just under the lowest tick labels.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bottom = min(ax.get_tightbbox(renderer).y0 for ax in axes[1]) / fig.bbox.height
    fig._supxlabel.set_verticalalignment("top")
    fig._supxlabel.set_y(bottom - 0.01)
    save(fig, f"spearman_consistency_recall_{ds}")


def main():
    for ds in BUDGETS:
        print(f"\n===== {ds}, budget {BUDGETS[ds]:,} =====")
        plot(ds)
        budget_sweep(ds)


if __name__ == "__main__":
    main()
