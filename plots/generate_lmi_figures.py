"""
LMI figures of the thesis, from results/lmi/<dataset>/lmi_results_*.csv at the smallest
number of buckets (1 000 on the 1M datasets, 10 106 on LAION 100M):

  lmi_recall_vs_distcomp_bigann        recall@10 vs distance computations
  lmi_recall_vs_distcomp_full_deep1b   recall@1, @5, @10 and @30 vs distance computations
  lmi_recall_vs_distcomp_full_laion    the same on LAION 100M

Usage (from the repository root): python plots/generate_lmi_figures.py
"""

from common import common_nprobe, load_results, plot_recall_vs_distcomp, plot_recall_vs_distcomp_full


def results(ds):
    df = load_results("lmi", ds, "lmi_results_*.csv")
    return df[df["n_buckets"] == df["n_buckets"].min()]


def main():
    plot_recall_vs_distcomp(results("bigann"), "bigann", "lmi_recall_vs_distcomp_bigann")
    for ds in ("deep1b", "laion"):
        plot_recall_vs_distcomp_full(common_nprobe(results(ds)), ds, f"lmi_recall_vs_distcomp_full_{ds}")


if __name__ == "__main__":
    main()
