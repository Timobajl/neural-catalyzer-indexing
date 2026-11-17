"""
Illustration of the lambda trade-off of the Neural Catalyzer (nc_lambda_tradeoff).

The figure illustrates the behavior described by Sablayrolles et al. (Figure 2 and Sec. 3.3):
lambda = 0 keeps the neighbors together but does not cover the output space, lambda -> infinity
covers it uniformly but does not keep the neighbors, and intermediate values trade one for the other.
No network is trained, every panel is constructed inside the unit disk:
  - input:          four Gaussian clusters
  - uniform set:    the same number of points spread evenly over the disk
                    (dart throwing with a minimum distance)
  - matching:       every input point gets one point of the uniform set by an optimal assignment
                    (minimum total squared distance), so each cluster owns the region next to it
  - lambda = 0:     clusters tightened towards their centers, most of the disk stays empty
  - lambda = 0.01:  points moved T_SMALL of the way to their matched uniform point
  - lambda = 0.1:   the matched points swapped with nearby ones (MIX_PROB), so the regions blend
                    at their borders, and points moved T_LARGE of the way to them
  - lambda -> inf:  the uniform set with a random matching, the disk is covered evenly and the
                    neighborhoods are lost
The two middle panels get a small random offset (JITTER_*), so the cover is not too regular.
The background is the hexagonal lattice, as in the original figure.

Usage (from the repository root): python plots/generate_nc_lambda_tradeoff_figure.py
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.optimize import linear_sum_assignment


plt.rcParams.update({"font.size": 22})

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

SEED = 0
N_PER = 60                                  # points per cluster
CENTERS = [(-0.55, 0.45), (0.25, 0.55), (0.6, -0.1), (-0.25, -0.45)]
COLORS = ["#d64532", "#f2c037", "#199d6b", "#2f6fdd"]
SPREAD = 0.11                               # standard deviation of each input cluster
SHRINK = 0.75                               # lambda = 0: cluster size relative to the input
T_SMALL, T_LARGE = 0.4, 0.92                # lambda = 0.01 and 0.1: share of the way to the uniform set
R_DISK = 0.95                               # uniform points stay inside this radius
MIN_DIST = 0.085                            # minimum distance between uniform points
MIX_PROB = 0.35                             # lambda = 0.1: chance that a point swaps its target ...
MIX_K = 6                                   # ... with one of this many nearest uniform points
JITTER_SMALL, JITTER_LARGE = 0.015, 0.035   # lambda = 0.01 and 0.1: random offset
R_MAX = 0.97                                # offset points are pulled back inside this radius


def make_clusters(rng):
    pts, lab = [], []
    for k, c in enumerate(CENTERS):
        pts.append(np.asarray(c) + SPREAD * rng.standard_normal((N_PER, 2)))
        lab.append(np.full(N_PER, k))
    return np.concatenate(pts), np.concatenate(lab)


def uniform_disk(n, rng):
    pts = []
    while len(pts) < n:
        r = R_DISK * math.sqrt(rng.random())
        a = 2 * math.pi * rng.random()
        p = np.array([r * math.cos(a), r * math.sin(a)])
        if all(np.hypot(*(p - q)) >= MIN_DIST for q in pts):
            pts.append(p)
    return np.array(pts)


def shrink(x, lab):
    out = x.copy()
    for k in range(len(CENTERS)):
        m = lab == k
        c = x[m].mean(axis=0)
        out[m] = c + SHRINK * (x[m] - c)
    return out


def mix_borders(u, match, rng):
    """Swap targets between nearby uniform points, so the regions blend where they meet."""
    owner = np.empty(len(u), dtype=int)
    owner[match] = np.arange(len(u))
    d = ((u[:, None, :] - u[None, :, :]) ** 2).sum(axis=2)
    nbr = np.argsort(d, axis=1)[:, 1:MIX_K + 1]
    for j in rng.permutation(len(u)):
        if rng.random() < MIX_PROB:
            k = nbr[j, rng.integers(MIX_K)]
            owner[j], owner[k] = owner[k], owner[j]
    out = np.empty_like(match)
    out[owner] = np.arange(len(u))
    return out


def jitter(p, sigma, rng):
    q = p + sigma * rng.standard_normal(p.shape)
    r = np.linalg.norm(q, axis=1, keepdims=True)
    return q * np.minimum(1.0, R_MAX / r)


def build_panels():
    rng = np.random.default_rng(SEED)
    x, lab = make_clusters(rng)
    u = uniform_disk(len(x), rng)
    cost = ((x[:, None, :] - u[None, :, :]) ** 2).sum(axis=2)
    _, match = linear_sum_assignment(cost)
    v = u[match]                                # region-keeping uniform target
    w = u[mix_borders(u, match, rng)]           # same cover, regions blended at their borders
    c = shrink(x, lab)
    panels = [
        ("input", x),
        (r"$\lambda = 0$", c),
        (r"$\lambda = 0.01$", jitter((1 - T_SMALL) * c + T_SMALL * v, JITTER_SMALL, rng)),
        (r"$\lambda = 0.1$", jitter((1 - T_LARGE) * c + T_LARGE * w, JITTER_LARGE, rng)),
        (r"$\lambda \to \infty$", u[rng.permutation(len(u))]),
    ]
    return panels, lab


def hex_lattice(ax, step=0.25):
    clip = Circle((0, 0), 1, transform=ax.transData)
    for k in range(-8, 9):
        for ang in (0, 60, 120):
            t = math.radians(ang)
            d = np.array([math.cos(t), math.sin(t)])
            p0 = np.array([-d[1], d[0]]) * k * step * math.sqrt(3) / 2
            seg = np.array([p0 - 2 * d, p0 + 2 * d])
            ax.plot(seg[:, 0], seg[:, 1], color="0.85", lw=0.5, zorder=0, clip_path=clip)


def save(fig, name):
    # The same as in common.py, which this script does not import, because its seaborn style would
    # change this figure
    OUTPUT_DIR.mkdir(exist_ok=True)
    for ext, kwargs in (("pdf", {}), ("png", {"dpi": 150})):
        path = OUTPUT_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"  wrote {path.name}")
    plt.close(fig)


def main():
    panels, lab = build_panels()
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
    for ax, (name, p) in zip(axes, panels):
        hex_lattice(ax)
        ax.add_patch(Circle((0, 0), 1, fill=False, lw=1.6, color="black", zorder=3))
        ax.scatter(p[:, 0], p[:, 1], c=[COLORS[k] for k in lab], s=14, zorder=2, edgecolors="none")
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(-1.08, 1.08)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(name, y=-0.22)
    for a, b in zip(axes[:-1], axes[1:]):
        fig.add_artist(FancyArrowPatch((a.get_position().x1, 0.55), (b.get_position().x0, 0.55),
                                       transform=fig.transFigure, arrowstyle="->", color="0.6",
                                       mutation_scale=12))
    save(fig, "nc_lambda_tradeoff")


if __name__ == "__main__":
    main()
