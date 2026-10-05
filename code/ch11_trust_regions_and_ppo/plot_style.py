"""Shared matplotlib styling for the Chapter 11 figures (copied from Chapter 00 so the chapter folder is self-contained).

Kept deliberately tiny: a fixed categorical colour order (validated for
colour-vision deficiency), thin lines, recessive grids. Every script also uses
line styles / markers as a secondary encoding so no plot relies on colour alone.
"""
import matplotlib

matplotlib.use("Agg")  # headless backend: we only ever write PNG files
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots, always assigned in this order (never cycled past 8).
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GREY = "#8a8985"
INK = "#0b0b0b"


def setup():
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 110,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.edgecolor": "#b9b8b3",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=C),
        "legend.frameon": False,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
    })
    return plt


def kfmt(ax, axis="x"):
    """Show large step counts as 0, 50k, 100k, ... so tick labels never collide."""
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1000:g}k")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(f)
