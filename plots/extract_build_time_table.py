"""
Prints the rows of the build-time table of the thesis (total index build time in seconds),
at the build configuration of the recall figures of each index:

  IVFFlat    train_time + add_time
  HNSW       add_time
  LMI        train_time (clustering, training and the assignment of every vector)
  NeuralLSH  rank_precompute_time + nlsh_build_time, on LAION 100M also the assignment of
             the full base and the sorted bucket index, which run outside nlsh_build_time
  NP         root k'-NN matrix + train_time + add_time; train_time starts after the k'-NN
             matrix, so its time is taken from the run logs

Usage (from the repository root): python plots/extract_build_time_table.py
"""

import generate_hnsw_figures
import generate_ivfflat_figures
import generate_lmi_figures
import generate_nlsh_figures
import generate_np_figures

# (row label, rows of the recall figure)
INDEXES = {
    "ivfflat": ("IVFFlat", generate_ivfflat_figures.results),
    "hnsw":    ("HNSW",    generate_hnsw_figures.results),
    "lmi":     ("LMI",     generate_lmi_figures.results),
    "nlsh":    ("NLSH",    generate_nlsh_figures.results),
    "np":      ("NP",      generate_np_figures.results),
}
DATASETS = {"bigann": "BigANN-1M", "deep1b": "Deep1B-1M", "laion": "LAION 100M"}

# Table columns: Original, 16D, 64D, 96D or 128D, 768D, at the selected lambda
COLUMNS = {
    "bigann": ["original", "catalyzer16_0_01", "catalyzer64_0_0002", "catalyzer128_0_0001", None],
    "deep1b": ["original", "catalyzer16_0_02", "catalyzer64_0_0002", "catalyzer96_0_0002", None],
    "laion":  ["original", "catalyzer16_0_1", "catalyzer64_0_0002", "catalyzer128_0_001", "catalyzer768_0_0005"],
}

# NeuralLSH on LAION 100M: assignment of the full base + sorted bucket index, in seconds
NLSH_LAION_EXTRA_S = {
    "original":            741.0 + 12.6,  # log of job 20171040
    "catalyzer16_0_1":     423.4 + 10.7,  # log of job 23622012
    "catalyzer64_0_0002":  424.2 + 10.7,  # log of job 23622017
    "catalyzer128_0_001":  502.7 + 11.1,  # log of job 23622020
    "catalyzer768_0_0005": 628.9 + 9.3,   # log of job 20171041
}

# Neural Partitioner: root k'-NN matrix of the 16 x 16 runs (1M datasets) and the 100 x 100
# runs (LAION 100M), in seconds, the first "kNN done in Xs" after the label in the run log
NP_ROOT_KNN_S = {
    "bigann": {  # log of job 25349340
        "original": 274.0, "catalyzer16_0_01": 472.0, "catalyzer32_0_0007": 429.0,
        "catalyzer64_0_0002": 227.0, "catalyzer128_0_0001": 333.0,
    },
    "deep1b": {  # log of job 25349341
        "original": 322.0, "catalyzer16_0_02": 274.0, "catalyzer32_0_005": 337.0,
        "catalyzer64_0_0002": 259.0, "catalyzer96_0_0002": 252.0,
    },
    "laion": {
        "catalyzer16_0_1": 280.0,       # log of job 25352258
        "catalyzer32_0_001": 338.0,     # log of job 25352258
        "catalyzer64_0_0002": 413.0,    # log of job 25352258
        "catalyzer128_0_001": 462.0,    # log of job 25352258
        "catalyzer768_0_0005": 1004.0,  # log of job 25352259
        "original": 934.0,              # log of job 25352260
    },
}


def build_time(index, ds, row):
    if index == "ivfflat":
        return row["train_time"] + row["add_time"]
    if index == "hnsw":
        return row["add_time"]
    if index == "lmi":
        return row["train_time"]
    if index == "nlsh":
        t = row["rank_precompute_time"] + row["nlsh_build_time"]
        return t + (NLSH_LAION_EXTRA_S[row["space"]] if ds == "laion" else 0.0)
    return NP_ROOT_KNN_S[ds][row["space"]] + row["train_time"] + row["add_time"]


def fmt(v):
    """Seconds as printed in the table: one decimal below 100, else a thin-space thousands."""
    return f"{v:.1f}" if v < 100 else f"{int(v + 0.5):,}".replace(",", "\\,")


def main():
    for index, (label, results) in INDEXES.items():
        if index != "ivfflat":
            print("    \\midrule")
        for ds, ds_label in DATASETS.items():
            df = results(ds)
            cells = []
            for space in COLUMNS[ds]:
                rows = df[df["space"] == space] if space else None
                if space is None:
                    cells.append("---")
                elif rows.empty:
                    cells.append("MISSING")
                else:
                    cells.append(fmt(build_time(index, ds, rows.iloc[0])))
            print(f"    {label:<8} & {ds_label:<11} & " + " & ".join(cells) + " \\\\")


if __name__ == "__main__":
    main()
