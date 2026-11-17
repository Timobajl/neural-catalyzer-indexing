"""
End-to-end figures of the thesis, recall@10 against the time per query in a batch of
10 000 queries, one panel per index at the build configuration of its recall figure:

  e2e_recall_vs_latency_bigann
  e2e_recall_vs_latency_deep1b
  e2e_recall_vs_latency_laion

The time per query is the catalyzer forward pass (results/inference/inference_times.csv,
batch of 10 000), plus the inference of the router for LMI, NeuralLSH and the Neural
Partitioner, plus the search. The router and search times are the totals over the
10 000 queries divided by 10 000.

Usage (from the repository root): python plots/generate_e2e_figure.py
"""

import matplotlib.pyplot as plt
import pandas as pd

import generate_hnsw_figures
import generate_ivfflat_figures
import generate_lmi_figures
import generate_nlsh_figures
import generate_np_figures
from common import DIM_COLORS, ORIGINAL_DIM, RESULTS_ROOT, output_dim, save

plt.rcParams.update({"font.size": 24})

N_QUERIES = 10_000
# (panel title, rows of the recall figure, effort column, router inference counted)
INDEXES = [
    ("IVFFlat",            generate_ivfflat_figures.results, "nprobe",   False),
    ("HNSW",               generate_hnsw_figures.results,    "efSearch", False),
    ("LMI",                generate_lmi_figures.results,     "nprobe",   True),
    ("NeuralLSH",          generate_nlsh_figures.results,    "nprobe",   True),
    ("Neural Partitioner", generate_np_figures.results,      "nprobe",   True),
]


def catalyzer_us(ds):
    """Catalyzer forward pass per query (us) by output dimension, in a batch of 10 000."""
    df = pd.read_csv(RESULTS_ROOT / "inference" / "inference_times.csv")
    df = df[(df["batch_size"] == N_QUERIES) & (df["space"].str.split("_").str[0] == ds)]
    return dict(zip(df["dim"], df["per_query_us"]))


def curves(ds, results, effort, router):
    """{output dimension: rows sorted by effort, with the time per query in column us}."""
    cat_us = catalyzer_us(ds)
    df = results(ds).copy()
    df["dim"] = df["space"].map(output_dim)
    router_us = df["inference_time"] / N_QUERIES * 1e6 if router else 0.0
    df["us"] = df["dim"].map(lambda d: 0.0 if d == 0 else cat_us[d]) + router_us + df["search_time"] / N_QUERIES * 1e6
    return {d: g.sort_values(effort) for d, g in df.groupby("dim")}


def plot(ds):
    # Five index panels in a 3 x 2 grid, the legend in the sixth cell
    fig, axes = plt.subplots(3, 2, figsize=(12, 15), sharey=True)
    axes = axes.flatten()
    for i, (title, results, effort, router) in enumerate(INDEXES):
        ax = axes[i]
        data = curves(ds, results, effort, router)
        # Catalyzer spaces by output dimension, the original last
        for d in sorted(data, key=lambda v: 10**6 if v == 0 else v):
            label = f"Original {ORIGINAL_DIM[ds]}D" if d == 0 else f"Catalyzer {d}D"
            ax.plot(data[d]["us"], data[d]["recall@10"], marker="o", markersize=9, linewidth=2.8,
                    color=DIM_COLORS[d], label=label)
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title, fontsize=30)
        ax.set_xlabel("Mean time per query\nin a batch (µs)", fontsize=24)
        if i % 2 == 0:
            ax.set_ylabel("Recall@10", fontsize=24)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=20)

    axes[5].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].legend(handles, labels, loc="center", fontsize=26, frameon=True)
    plt.tight_layout()
    save(fig, f"e2e_recall_vs_latency_{ds}")


def main():
    for ds in ("bigann", "deep1b", "laion"):
        plot(ds)


if __name__ == "__main__":
    main()
