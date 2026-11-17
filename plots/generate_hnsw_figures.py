"""
HNSW figures of the thesis, from results/hnsw/<dataset>/hnsw_results_*.csv at M = 32
(recall differs from M = 16 and 48 by at most 0.2 percentage points):

  hnsw_recall_vs_distcomp_bigann        recall@10 vs distance computations
  hnsw_recall_vs_distcomp_full_deep1b   recall@1, @5, @10 and @30 vs distance computations
  hnsw_recall_vs_distcomp_full_laion    the same on LAION 100M

Usage (from the repository root): python plots/generate_hnsw_figures.py
"""

from common import load_results, plot_recall_vs_distcomp, plot_recall_vs_distcomp_full


def results(ds):
    df = load_results("hnsw", ds, "hnsw_results_*.csv")
    return df[df["M"] == 32]


def main():
    plot_recall_vs_distcomp(results("bigann"), "bigann", "hnsw_recall_vs_distcomp_bigann")
    for ds in ("deep1b", "laion"):
        plot_recall_vs_distcomp_full(results(ds), ds, f"hnsw_recall_vs_distcomp_full_{ds}")


if __name__ == "__main__":
    main()
