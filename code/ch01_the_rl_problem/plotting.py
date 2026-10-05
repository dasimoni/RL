"""Shared figure style for the Chapter 01 scripts (no RL logic here).

Colors: categorical slots in a fixed order (blue, orange, aqua, ...), a
single-hue blue ramp for magnitudes, and a blue <-> red diverging ramp with a
neutral gray midpoint for signed quantities such as v_pi of the random policy.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # never open windows; scripts only save PNGs
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#c43a3a", "#f0a3a2", "#f0efec", "#86b6ef", "#1c5cab"])

DPI = 110


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": DPI, "savefig.dpi": DPI, "font.size": 10,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "legend.frameon": False,
        "axes.prop_cycle": matplotlib.cycler(color=[BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]),
    })


def _text_color(rgba) -> str:
    r, g, b = rgba[:3]
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else INK


def grid_heatmap(ax, values: np.ndarray, title: str, *, diverging: bool = False,
                 fmt: str = "{:.1f}", fontsize: int = 10, vmin=None, vmax=None):
    """Draw an n x n grid of numbers as a labelled heatmap (row 0 at the top)."""
    if diverging:
        lim = np.max(np.abs(values)) if vmax is None else vmax
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        cmap = DIVERGING
    else:
        norm = matplotlib.colors.Normalize(vmin=values.min() if vmin is None else vmin,
                                           vmax=values.max() if vmax is None else vmax)
        cmap = SEQ_BLUE
    im = ax.imshow(values, cmap=cmap, norm=norm)
    n_r, n_c = values.shape
    for i in range(n_r):
        for j in range(n_c):
            ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center",
                    fontsize=fontsize, color=_text_color(cmap(norm(values[i, j]))))
    ax.set_xticks(np.arange(-0.5, n_c, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_r, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)
    return im
