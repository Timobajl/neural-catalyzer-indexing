"""
Shared parts of the figure scripts: style, colors and legend labels of the vector spaces,
loading of the result CSVs, space names, saving, and the two recall figure types.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 18})

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE.parent / "results"
OUTPUT_DIR = HERE / "output"

# Spaces in plotting order: (color, legend label)
SPACES = {
    "bigann": {
        "catalyzer16_0_01":    ("tab:red",    "Catalyzer 16D"),
        "catalyzer32_0_0007":  ("tab:orange", "Catalyzer 32D"),
        "catalyzer64_0_0002":  ("tab:green",  "Catalyzer 64D"),
        "catalyzer128_0_0001": ("tab:purple", "Catalyzer 128D"),
        "original":            ("black",      "Original 128D"),
    },
    "deep1b": {
        "catalyzer16_0_02":    ("tab:red",    "Catalyzer 16D"),
        "catalyzer32_0_005":   ("tab:orange", "Catalyzer 32D"),
        "catalyzer64_0_0002":  ("tab:green",  "Catalyzer 64D"),
        "catalyzer96_0_0002":  ("tab:blue",   "Catalyzer 96D"),
        "original":            ("black",      "Original 96D"),
    },
    "laion": {
        "catalyzer16_0_1":     ("tab:red",    "Catalyzer 16D"),
        "catalyzer32_0_001":   ("tab:orange", "Catalyzer 32D"),
        "catalyzer64_0_0002":  ("tab:green",  "Catalyzer 64D"),
        "catalyzer128_0_001":  ("tab:purple", "Catalyzer 128D"),
        "catalyzer768_0_0005": ("tab:cyan",   "Catalyzer 768D"),
        "original":            ("black",      "Original 768D"),
    },
}
RECALLS = ["recall@1", "recall@5", "recall@10", "recall@30"]

# Color per output dimension, 0 is the original space (the colors of SPACES)
DIM_COLORS = {16: "tab:red", 32: "tab:orange", 64: "tab:green", 96: "tab:blue",
              128: "tab:purple", 768: "tab:cyan", 0: "black"}
ORIGINAL_DIM = {"bigann": 128, "deep1b": 96, "laion": 768}


def space_name(space):
    """Space name as in SPACES. NeuralLSH and the Neural Partitioner add the tree
    (original_16x16), and the Neural Partitioner names the original space
    bigann_orig_16x16 and the LAION 768D space catalyzer_d768_l0.0005."""
    if "_orig" in space:
        return "original"
    m = re.match(r"catalyzer_d(\d+)_l(\d+)\.(\d+)$", space)
    if m:
        return "catalyzer{}_{}_{}".format(*m.groups())
    return re.sub(r"_\d+x\d+$", "", space)


def output_dim(space):
    """Output dimension of a space name from SPACES, 0 for the original."""
    return 0 if space == "original" else int(re.match(r"catalyzer(\d+)_", space).group(1))


def load_results(index, ds, pattern):
    """All rows of the CSVs matching pattern in results/<index>/<ds>/."""
    folder = RESULTS_ROOT / index / ds
    csvs = sorted(folder.glob(pattern))
    if not csvs:
        raise FileNotFoundError(f"no {pattern} in {folder}")
    return pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)


def common_nprobe(df):
    """Rows at the nprobe values that every space has, so all curves end at the same x."""
    shared = set.intersection(*[set(df[df["space"] == s]["nprobe"]) for s in df["space"].unique()])
    return df[df["nprobe"].isin(shared)]


def palette(ds):
    return {space: color for space, (color, _) in SPACES[ds].items()}


def hue_order(df, ds):
    present = set(df["space"])
    return [space for space in SPACES[ds] if space in present]


def legend_labels(ds, spaces):
    return [SPACES[ds][s][1] if s in SPACES[ds] else s for s in spaces]


def save(fig, name):
    OUTPUT_DIR.mkdir(exist_ok=True)
    for ext, kwargs in (("pdf", {}), ("png", {"dpi": 150})):
        path = OUTPUT_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"  wrote {path.name}")
    plt.close(fig)


def plot_recall_vs_distcomp(df, ds, name):
    """Recall@10 vs distance computations in one panel."""
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.lineplot(
        data=df, x="dist_computations", y="recall@10",
        hue="space", palette=palette(ds), hue_order=hue_order(df, ds),
        marker="o", linewidth=2.8, markersize=10, ax=ax,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Distance computations per query (log scale)", fontsize=32)
    ax.set_ylabel("Recall@10", fontsize=36)
    ax.tick_params(axis="both", labelsize=34)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    # Legend above the axes, so it does not cover the curves
    handles, spaces = ax.get_legend_handles_labels()
    ax.legend(handles, legend_labels(ds, spaces),
              loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              fontsize=28, frameon=False, columnspacing=1.5)
    plt.tight_layout()
    save(fig, name)


def plot_recall_vs_distcomp_full(df, ds, name):
    """Recall@1, @5, @10 and @30 vs distance computations in a 2 x 2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(28, 22))
    for ax, metric in zip(axes.flatten(), RECALLS):
        sns.lineplot(
            data=df, x="dist_computations", y=metric,
            hue="space", palette=palette(ds), hue_order=hue_order(df, ds),
            marker="o", linewidth=3.4, markersize=14, ax=ax,
        )
        ax.set_xscale("log")
        ax.set_xlabel("")
        ax.set_ylabel(metric.replace("recall", "Recall"), fontsize=48)
        ax.tick_params(axis="both", labelsize=48)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.get_legend().remove()

    # One x label and one legend for all four panels. The legend wraps to two rows
    # when there are more than five spaces (LAION).
    fig.supxlabel("Distance computations per query (log scale)", fontsize=52, y=0.02)
    handles, spaces = axes[0, 0].get_legend_handles_labels()
    n_cols = len(handles) if len(handles) <= 5 else (len(handles) + 1) // 2
    fig.legend(handles, legend_labels(ds, spaces),
               loc="upper center", bbox_to_anchor=(0.5, 0.985),
               ncol=n_cols, fontsize=36, frameon=False)

    plt.subplots_adjust(wspace=0.28, hspace=0.35)
    plt.tight_layout(rect=[0, 0.04, 1, 0.88], pad=3.5, w_pad=4.0, h_pad=5.0)
    # tight_layout leaves a wide gap above the shared x label, so the label is
    # moved up to just under the lowest tick labels.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bottom = min(a.get_tightbbox(renderer).y0 for a in axes[1]) / fig.bbox.height
    fig._supxlabel.set_verticalalignment("top")
    fig._supxlabel.set_y(bottom - 0.01)
    save(fig, name)
