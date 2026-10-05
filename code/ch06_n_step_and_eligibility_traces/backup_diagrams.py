"""Backup diagrams for n-step methods and the weights of the lambda-return (schematics).

Chapter 06, Sections 2, 3, 6 and 8. Nothing is learned here; the script draws, in the
style of Sutton & Barto (2018, Figures 7.1, 7.3 and 12.2):

  (a) the backup diagrams of n-step TD for state values, from TD(0) to Monte Carlo:
      open circles are states, small filled dots are actions, a grey square is the
      terminal state; the bottom state of each diagram is the one we bootstrap from;
  (b) 3-step SARSA, 3-step Expected SARSA and 3-step Tree Backup for action values:
      in Tree Backup every action *not* taken is a leaf, weighted by pi(a|s), and the
      path continues only through the action actually taken, weighted by pi(A|S);
  (c) the weights (1 - lambda) lambda^(n-1) that the lambda-return (Eq. 6.17) puts on the
      n-step returns in an episode that ends T - t = 10 steps after t; the weight of all
      n >= T - t lands on the full return G_t, which receives lambda^(T-t-1).

The script prints the weights in (c) and checks that they sum to one.

Outputs (full mode): figures/backup_diagrams.png

Run:  python code/ch06_n_step_and_eligibility_traces/backup_diagrams.py [--quick]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
LAM, STEPS_LEFT = 0.8, 10          # panel (c): lambda and T - t
DY = 0.5                           # vertical distance between a state and an action node


def lambda_weights(lam, steps_left):
    """Weights of G_{t:t+1}, ..., G_{t:t+steps_left-1} and of the full return G_t (Eq. 6.17)."""
    n = np.arange(1, steps_left)
    w_n = (1 - lam) * lam ** (n - 1)
    w_full = lam ** (steps_left - 1)
    return n, w_n, w_full


# --------------------------------------------------------------------------- drawing helpers
def state(ax, x, y, ink, face, r=0.13):
    import matplotlib.patches as mp
    ax.add_patch(mp.Circle((x, y), r, facecolor=face, edgecolor=ink, lw=1.4, zorder=3))


def action(ax, x, y, ink, r=0.06):
    import matplotlib.patches as mp
    ax.add_patch(mp.Circle((x, y), r, facecolor=ink, edgecolor=ink, zorder=3))


def terminal(ax, x, y, grey, s=0.2):
    import matplotlib.patches as mp
    ax.add_patch(mp.Rectangle((x - s / 2, y - s / 2), s, s, facecolor=grey, edgecolor=grey, zorder=3))


def edge(ax, x0, y0, x1, y1, ink, ls="-"):
    ax.plot([x0, x1], [y0, y1], color=ink, lw=1.1, ls=ls, zorder=1)


def chain(ax, x, n_steps, ink, face, grey, gap_after=None, end="state"):
    """State-value chain S, A, S, A, ..., with an optional vertical ellipsis."""
    y = 0.0
    state(ax, x, y, ink, face)
    for k in range(n_steps):
        if gap_after is not None and k == gap_after:
            ax.text(x, y - 0.42, "⋮", ha="center", va="center", fontsize=14, color=ink)
            y -= 0.85
            state(ax, x, y, ink, face)
        edge(ax, x, y, x, y - DY, ink)
        action(ax, x, y - DY, ink)
        edge(ax, x, y - DY, x, y - 2 * DY, ink)
        y -= 2 * DY
        if k == n_steps - 1 and end == "terminal":
            terminal(ax, x, y, grey)
        else:
            state(ax, x, y, ink, face)
    return y


def fan(ax, x, y, ink, face, n_actions=3, taken=None, spread=0.35, leaf_label=None):
    """Branch from the state at (x, y) to all actions; return the x of the taken one."""
    xs = x + spread * (np.arange(n_actions) - (n_actions - 1) / 2)
    for k, xa in enumerate(xs):
        edge(ax, x, y, xa, y - DY, ink)
        action(ax, xa, y - DY, ink)
        if leaf_label and k != taken:
            ax.text(xa, y - DY - 0.17, leaf_label, ha="center", va="top", fontsize=7, color="#52514e")
    return xs[taken] if taken is not None else None


def q_diagram(ax, x, kind, ink, face, n=3):
    """3-step backups for action values: 'sarsa', 'esarsa' or 'tree'."""
    y = 0.0
    state(ax, x, y, ink, face)
    edge(ax, x, y, x, y - DY, ink)
    action(ax, x, y - DY, ink)                      # (S_t, A_t): the pair being updated
    y -= DY
    for k in range(1, n + 1):
        edge(ax, x, y, x, y - DY, ink)               # transition to S_{t+k}
        y -= DY
        state(ax, x, y, ink, face)
        last = k == n
        if kind == "tree":
            fan(ax, x, y, ink, face, taken=None if last else 1, leaf_label="π")
            y -= DY
        elif kind == "esarsa" and last:
            fan(ax, x, y, ink, face, taken=None, leaf_label="π")
            y -= DY
        else:                                         # sampled action (SARSA, or ES before the end)
            edge(ax, x, y, x, y - DY, ink)
            action(ax, x, y - DY, ink)
            y -= DY
    return y


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="print the weights only, no figure")
    parser.add_argument("--seed", type=int, default=0, help="unused (nothing is random); kept for uniformity")
    args = parser.parse_args()
    t0 = time.time()
    print(f"Backup diagrams and lambda-return weights | seed={args.seed} (no randomness) "
          f"lambda={LAM} T-t={STEPS_LEFT}")
    n, w_n, w_full = lambda_weights(LAM, STEPS_LEFT)
    print("  n:      " + " ".join(f"{k:6d}" for k in n) + "   full return G_t")
    print("  weight: " + " ".join(f"{w:6.3f}" for w in w_n) + f"   {w_full:6.3f}")
    total = w_n.sum() + w_full
    print(f"  sum of weights = {total:.12f}; mean lookahead (untruncated) 1/(1-lambda) = {1 / (1 - LAM):g}")
    assert abs(total - 1.0) < 1e-12
    print(f"elapsed {time.time() - t0:.2f} s")
    if args.quick:
        return

    from plot_style import setup, C, GREY, INK
    plt = setup()
    face = plt.rcParams["axes.facecolor"]
    fig = plt.figure(figsize=(15, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[5.0, 4.2, 4.4], wspace=0.22)

    # (a) n-step TD for state values
    ax = fig.add_subplot(gs[0])
    labels = ["1-step\nTD(0)", "2-step", "3-step", "n-step", "∞-step\n(Monte Carlo)"]
    for i, lab in enumerate(labels):
        x = 1.1 * i
        if i < 3:
            yb = chain(ax, x, i + 1, INK, face, GREY)
        elif i == 3:
            yb = chain(ax, x, 3, INK, face, GREY, gap_after=2)
        else:
            yb = chain(ax, x, 3, INK, face, GREY, gap_after=2, end="terminal")
        ax.text(x, -4.55, lab, ha="center", va="top", fontsize=9, color=INK)
    ax.text(-0.45, 0, r"$S_t$", ha="right", va="center", fontsize=9, color="#52514e")
    ax.set_xlim(-0.7, 5.0); ax.set_ylim(-5.15, 0.35)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) n-step TD backups (state values)", fontsize=10)

    # (b) action values
    ax = fig.add_subplot(gs[1])
    for i, (kind, lab) in enumerate([("sarsa", "3-step\nSARSA"), ("esarsa", "3-step\nExpected SARSA"),
                                     ("tree", "3-step\nTree Backup")]):
        q_diagram(ax, 1.5 * i, kind, INK, face)
        ax.text(1.5 * i, -4.55, lab, ha="center", va="top", fontsize=9, color=INK)
    ax.text(-0.3, -DY, r"$S_t,A_t$", ha="right", va="center", fontsize=9, color="#52514e")
    ax.set_xlim(-0.8, 3.6); ax.set_ylim(-5.15, 0.35)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(b) action values; leaves weighted by π", fontsize=10)

    # (c) lambda-return weights
    ax = fig.add_subplot(gs[2])
    ax.bar(n, w_n, color=C[0], width=0.7, label=r"$(1-\lambda)\lambda^{n-1}$ on $G_{t:t+n}$")
    ax.bar([STEPS_LEFT], [w_full], color=C[1], width=0.7,
           label=rf"$\lambda^{{T-t-1}}$ = {w_full:.3f} on the full return $G_t$")
    n_tail = np.arange(STEPS_LEFT + 1, 17)
    ax.bar(n_tail, (1 - LAM) * LAM ** (n_tail - 1), color="none", edgecolor=GREY, ls=":", width=0.7,
           label="weights of $n > T-t$ in the untruncated sum\n(all weight of $n \\geq T-t$ lands on $G_t$)")
    ax.set_xticks(list(n) + [STEPS_LEFT]); ax.set_xticklabels([str(k) for k in n] + ["T−t"])
    ax.set_xlabel("n (number of rewards before bootstrapping)")
    ax.set_ylabel("weight in the λ-return")
    ax.set_title(f"(c) λ-return weights, λ = {LAM}, episode ends at T = t + {STEPS_LEFT}", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 0.24)

    path = os.path.join(FIG_DIR, "backup_diagrams.png")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
