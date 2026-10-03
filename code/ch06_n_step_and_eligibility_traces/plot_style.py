"""Shared matplotlib styling for the Chapter 06 figures.

A fixed categorical colour order (validated for colour-vision deficiency, assigned
in order and never cycled), a single-hue sequential ramp for *ordered* parameters
such as n or lambda, thin lines and recessive grids. Scripts also vary markers or
line styles so that no plot relies on colour alone.
"""
import matplotlib

matplotlib.use("Agg")  # headless backend: we only ever write PNG files
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots, always assigned in this order.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Sequential blue ramp (light -> dark) for ordered parameters (n = 1, 2, 4, ... or lambda).
BLUES = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
         "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
GREY = "#898781"
INK = "#0b0b0b"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]


def ramp(k: int) -> list[str]:
    """k colours spread evenly along the sequential ramp (k <= 10)."""
    if k == 1:
        return [BLUES[5]]
    idx = [round(i * (len(BLUES) - 1) / (k - 1)) for i in range(k)]
    return [BLUES[i] for i in idx]


def setup():
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 110,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.edgecolor": "#c3c2b7",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=C),
        "legend.frameon": False,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
        "text.color": INK,
        "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
    })
    return plt
