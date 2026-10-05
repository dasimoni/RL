"""Backup diagrams: dynamic programming vs Monte Carlo (Sec. 2.6).

Draws (a) the DP backup for v_pi -- one step deep, all actions and all successor states;
(b) the Monte Carlo backup -- one sampled trajectory, all the way to termination; and
(c) the two dimensions along which methods differ (S&B Sec. 8.13): width (expected vs
sample) and depth (bootstrapped one step vs full return).

    python code/ch04_monte_carlo/backup_diagrams.py [--quick]
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
R = 0.17


def state(ax, x, y, highlight=False):
    ax.add_patch(Circle((x, y), R, fc="#ffe08a" if highlight else "white", ec="k", lw=1.5, zorder=3))


def action(ax, x, y):
    ax.add_patch(Circle((x, y), R * 0.55, fc="k", ec="k", zorder=3))


def terminal(ax, x, y):
    ax.add_patch(Rectangle((x - R, y - R), 2 * R, 2 * R, fc="#bbbbbb", ec="k", lw=1.5, zorder=3))


def edge(ax, x0, y0, x1, y1, color="k", lw=1.2):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=1)


def draw(path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.6), gridspec_kw=dict(width_ratios=[1.3, 0.8, 1.2]))
    # (a) DP: expected update over every action and every successor state, one step deep.
    ax = axes[0]
    state(ax, 0, 3, True)
    for ax_x in (-1.6, 0, 1.6):
        edge(ax, 0, 3, ax_x, 2)
        action(ax, ax_x, 2)
        for dx in (-0.45, 0.45):
            edge(ax, ax_x, 2, ax_x + dx, 1)
            state(ax, ax_x + dx, 1)
    ax.text(0, 3.45, r"$s$", ha="center", fontsize=13)
    ax.text(2.25, 2, r"$a \sim \pi(\cdot|s)$", fontsize=10, va="center")
    ax.text(1.65, 0.55, r"$s', r \sim p(\cdot|s,a)$", fontsize=10)
    ax.text(0, -0.25, "update uses the model:\n"
            r"$V(s) \leftarrow \sum_a \pi(a|s)\sum_{s',r} p(s',r|s,a)\,[r+\gamma V(s')]$"
            "\nwide (all branches), shallow (one step), bootstraps on $V(s')$",
            ha="center", va="top", fontsize=9)
    ax.set_title("(a) Dynamic programming backup", fontsize=11)
    ax.set_xlim(-2.6, 3.3)
    ax.set_ylim(-1.6, 3.8)
    # (b) MC: one sampled trajectory down to the terminal state.
    ax = axes[1]
    y = 3.4
    state(ax, 0, y, True)
    ax.text(-0.45, y, r"$S_t$", ha="right", va="center", fontsize=12)
    for k in range(4):
        edge(ax, 0, y, 0, y - 0.45)
        action(ax, 0, y - 0.45)
        edge(ax, 0, y - 0.45, 0, y - 0.9)
        y -= 0.9
        if k < 3:
            state(ax, 0, y)
            ax.text(0.3, y + 0.22, rf"$R_{{t+{k + 1}}}$", fontsize=9)
    terminal(ax, 0, y)
    ax.text(0.3, y + 0.22, r"$R_T$", fontsize=9)
    ax.text(0.45, y - 0.05, "terminal", fontsize=9, va="center")
    ax.text(0, y - 0.45, "update uses one sampled\nepisode, no model:\n"
            r"$V(S_t) \leftarrow \mathrm{avg}\,[G_t]$" "\n"
            r"$G_t = R_{t+1} + \gamma R_{t+2} + \cdots$" "\nnarrow (one branch), deep (to the end),\nno bootstrapping",
            ha="center", va="top", fontsize=9)
    ax.set_title("(b) Monte Carlo backup", fontsize=11)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 3.8)
    # (c) The two dimensions.
    ax = axes[2]
    ax.add_patch(Rectangle((0, 0), 1, 1, fc="#f5f5f5", ec="k"))
    pts = {"Temporal-\ndifference (Ch. 05)": (0.17, 0.12), "Dynamic\nprogramming (Ch. 03)": (0.83, 0.12),
           "Monte Carlo\n(this chapter)": (0.17, 0.88), "Exhaustive search": (0.83, 0.88)}
    for label, (x, yy) in pts.items():
        ax.plot(x, yy, "o", ms=10, color="C3" if "Monte" in label else "C0")
        ax.text(x, yy + (0.07 if yy < 0.5 else -0.07), label, ha="center",
                va="bottom" if yy < 0.5 else "top", fontsize=9)
    ax.annotate("", xy=(1.0, -0.08), xytext=(0.0, -0.08), arrowprops=dict(arrowstyle="->"))
    ax.text(0.5, -0.12, "width of update: sample  ->  expected (needs a model)", ha="center", va="top", fontsize=9)
    ax.annotate("", xy=(-0.2, 1.0), xytext=(-0.2, 0.0), arrowprops=dict(arrowstyle="->"))
    ax.text(-0.24, 0.5, "depth of update: one step (bootstrap)  ->  full return", rotation=90,
            ha="right", va="center", fontsize=9)
    ax.set_xlim(-0.45, 1.12)
    ax.set_ylim(-0.3, 1.1)
    ax.set_title("(c) Where the methods sit", fontsize=11)
    for a in axes:
        a.set_aspect("equal")
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    print("Backup diagrams (no randomness; seed not applicable)")
    if args.quick:
        print("quick mode: nothing to compute, no figure written")
        return
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    path = os.path.join(HERE, "figures", "backup_diagrams.png")
    draw(path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
