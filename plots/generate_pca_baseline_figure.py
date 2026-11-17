"""
PCA baseline figure of the thesis (pca_baseline_bigann). On BigANN-1M the catalyzer at its
selected lambda* is compared with PCA to the same output dimension, with and without the
l2-normalization that the catalyzer applies to its output. All three are scored by an
exhaustive search, recall@10 against the original-space ground truth.

Reads results/knn/bigann/knn_ceiling_results.csv (catalyzer) and
results/knn/bigann/knn_ceiling_results_pca.csv (PCA).

Usage (from the repository root): python plots/generate_pca_baseline_figure.py
"""

import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS_ROOT, save

FOLDER = RESULTS_ROOT / "knn" / "bigann"

# (series, legend label, color, marker)
SERIES = [
    ("catalyzer", "Catalyzer (at λ*)", "tab:green",  "o"),
    ("pca",       "PCA",               "tab:orange", "s"),
    ("pca_norm",  "PCA + ℓ2-norm",     "tab:purple", "^"),
]


def load():
    """Recall@10 per output dimension of the three series, and of the original space."""
    df = pd.read_csv(FOLDER / "knn_ceiling_results.csv")
    original = df[df["space"] == "original"]["recall@10"].iloc[0]
    cat = df[df["space"].str.startswith("catalyzer")]
    # The best lambda per dimension is the peak of the lambda sweep, lambda*
    best = cat.loc[cat.groupby("dim")["recall@10"].idxmax()].set_index("dim")["recall@10"]

    pca = pd.read_csv(FOLDER / "knn_ceiling_results_pca.csv")
    pca = pca[pca["space"].str.startswith("pca")].astype({"pca_dim": int})
    norm = pca["space"].str.endswith("_norm")
    series = {
        "catalyzer": best.sort_index(),
        "pca": pca[~norm].set_index("pca_dim")["recall@10"].sort_index(),
        "pca_norm": pca[norm].set_index("pca_dim")["recall@10"].sort_index(),
    }
    return series, original


def main():
    series, original = load()
    fig, ax = plt.subplots(figsize=(16, 8))
    for key, label, color, marker in SERIES:
        s = series[key]
        ax.plot(s.index, s.values, marker=marker, linewidth=2.8, markersize=14, color=color, label=label)
    ax.axhline(original, color="black", linestyle="--", linewidth=2.4, label="Original (128D)")

    dims = sorted(series["catalyzer"].index)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Output dimension", fontsize=30)
    ax.set_ylabel("Recall@10", fontsize=28)
    ax.tick_params(axis="both", labelsize=34)
    ax.grid(True, which="major", ls="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=28)
    plt.tight_layout()
    save(fig, "pca_baseline_bigann")


if __name__ == "__main__":
    main()
