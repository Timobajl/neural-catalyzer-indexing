"""
IVFFlat figures of the thesis, from results/ivfflat/<dataset>/ivf_results_*.csv at the
smallest nlist (1 000 on the 1M datasets, 10 106 on LAION 100M):

  ivfflat_recall_vs_distcomp_bigann        recall@10 vs distance computations
  ivfflat_recall_vs_distcomp_full_deep1b   recall@1, @5, @10 and @30 vs distance computations
  ivfflat_recall_vs_distcomp_full_laion    the same on LAION 100M

Usage (from the repository root): python plots/generate_ivfflat_figures.py
"""

from common import load_results, plot_recall_vs_distcomp, plot_recall_vs_distcomp_full


def results(ds):
    df = load_results("ivfflat", ds, "ivf_results_*.csv")
    return df[df["nlist"] == df["nlist"].min()]


def main():
    plot_recall_vs_distcomp(results("bigann"), "bigann", "ivfflat_recall_vs_distcomp_bigann")
    for ds in ("deep1b", "laion"):
        plot_recall_vs_distcomp_full(results(ds), ds, f"ivfflat_recall_vs_distcomp_full_{ds}")


if __name__ == "__main__":
    main()
