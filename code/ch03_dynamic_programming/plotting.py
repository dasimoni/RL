"""Small shared plotting helpers for Chapter 03 (matplotlib, Agg backend)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
DPI = 110

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": DPI,
})

# A small, colour-blind-friendly categorical palette (Okabe-Ito).
C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]


def save(fig, name: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure: {os.path.relpath(path)}")
    return path


def grid_values(ax, values, shape, title="", fmt="{:.1f}", cmap="viridis", desc=None,
                arrows=None, arrow_chars=None, fontsize=9, vmin=None, vmax=None):
    """Heat map of a value function on a grid, with numbers and optional greedy arrows.

    arrows: list (per state) of action-index arrays (all tied maximisers) or None.
    desc:   optional 2-D array of map characters (FrozenLake); holes/goal are labelled.
    """
    V = np.asarray(values, dtype=float).reshape(shape)
    im = ax.imshow(V, cmap=cmap, vmin=vmin, vmax=vmax)
    lo = np.nanmin(V) if vmin is None else vmin
    hi = np.nanmax(V) if vmax is None else vmax
    for (r, c), v in np.ndenumerate(V):
        s = r * shape[1] + c
        txt = fmt.format(v)
        if desc is not None and desc[r, c] in "HG":
            txt = desc[r, c]
        elif arrows is not None and arrows[s] is not None and len(arrows[s]) > 0:
            txt = txt + "\n" + "".join(arrow_chars[a] for a in arrows[s])
        light = (v - lo) / (hi - lo + 1e-12) > 0.6
        ax.text(c, r, txt, ha="center", va="center", fontsize=fontsize,
                color="black" if light else "white")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    return im
