"""
Neural Partitioner figures of the thesis, from results/np/<dataset>/np_results_*.csv at the
smallest tree (16 x 16 on the 1M datasets, 100 x 100 on LAION 100M):

  np_recall_vs_distcomp_bigann        recall@10 vs distance computations
  np_recall_vs_distcomp_full_deep1b   recall@1, @5, @10 and @30 vs distance computations
  np_recall_vs_distcomp_full_laion    the same on LAION 100M

Usage (from the repository root): python plots/generate_np_figures.py
"""

from common import load_results, plot_recall_vs_distcomp, plot_recall_vs_distcomp_full, space_name


def results(ds):
    df = load_results("np", ds, "np_results_*.csv")
    df["space"] = df["space"].map(space_name)
    return df[df["n_bins"] == df["n_bins"].min()]


def main():
    plot_recall_vs_distcomp(results("bigann"), "bigann", "np_recall_vs_distcomp_bigann")
    for ds in ("deep1b", "laion"):
        plot_recall_vs_distcomp_full(results(ds), ds, f"np_recall_vs_distcomp_full_{ds}")


if __name__ == "__main__":
    main()
