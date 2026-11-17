"""
k-NN consistency figures of the thesis on BigANN-1M, one per partitioning index, from
results/<index>/bigann/knn_consistency_*.csv: the mean fractional k-NN consistency of every
space at each build configuration.

  ivfflat_knn_consistency_bigann   per nlist
  lmi_knn_consistency_bigann       per number of buckets
  nlsh_knn_consistency_bigann      per tree
  np_knn_consistency_bigann        per tree

Usage (from the repository root): python plots/generate_knn_consistency_bigann.py
"""

import matplotlib.pyplot as plt

from common import SPACES, hue_order, load_results, palette, save, space_name

DS = "bigann"


def thousands(value):
    """1000 -> '1 000' with a thin space, as in the thesis text."""
    return f"{int(value):,}".replace(",", " ")


def tree(side):
    """16 -> '16×16', the two-level tree as in the thesis text."""
    return f"{side}×{side}"


def plot_lines(df, config_col, config_text, x_label, name):
    """One line per space over the build configurations."""
    df["space"] = df["space"].map(space_name)
    configs = sorted(df[config_col].unique())
    positions = range(len(configs))
    fig, ax = plt.subplots(figsize=(16, 6))
    for space in hue_order(df, DS):
        sub = df[df["space"] == space].set_index(config_col)
        ys = [sub.loc[c, "knn_consistency_mean"] * 100 for c in configs]
        ax.plot(positions, ys, marker="o", linewidth=2.8, markersize=12,
                color=palette(DS)[space], label=SPACES[DS][space][1])
    ax.set_xticks(list(positions))
    ax.set_xticklabels([config_text(c) for c in configs])
    ax.set_xlim(-0.25, len(configs) - 0.75)
    ax.set_xlabel(x_label, fontsize=32)
    ax.set_ylabel(r"$k$-NN consistency (%)", fontsize=32)
    ax.tick_params(axis="both", labelsize=34)
    ax.grid(True, ls="--", alpha=0.5)
    # Legend above the axes, so it does not cover the lines
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              fontsize=28, frameon=False, columnspacing=1.5)
    plt.tight_layout()
    save(fig, name)


def main():
    plot_lines(load_results("ivfflat", DS, "knn_consistency_*.csv"), "nlist", thousands,
               "Number of clusters (nlist)", "ivfflat_knn_consistency_bigann")
    plot_lines(load_results("lmi", DS, "knn_consistency_*.csv"), "n_buckets", thousands,
               "Number of buckets (n_buckets)", "lmi_knn_consistency_bigann")
    plot_lines(load_results("nlsh", DS, "knn_consistency_*.csv"), "n_clusters", tree,
               "Tree configuration", "nlsh_knn_consistency_bigann")
    np_df = load_results("np", DS, "knn_consistency_*.csv")
    np_df["side"] = (np_df["n_bins"] ** 0.5).round().astype(int)
    plot_lines(np_df, "side", tree, "Tree configuration", "np_knn_consistency_bigann")


if __name__ == "__main__":
    main()
