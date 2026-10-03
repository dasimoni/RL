"""Two explanatory diagrams for Chapter 12 (no learning, no randomness).

  dpg_intuition.png          Section 2: the deterministic policy gradient in one picture.  For one
                             state, the critic Q(s, .) is a curve over actions; the actor's current
                             action mu_theta(s) sits somewhere on it, and the DPG moves it uphill
                             along dQ/da (Eq. 12.3) instead of searching for argmax_a Q.
  actor_critic_dataflow.png  Sections 3-6: what flows where in DDPG, TD3 and SAC: the replay buffer,
                             the target computation, the critic loss, the actor loss whose gradient
                             passes *through* the critic, and the Polyak copies, with equation numbers.

Run (from the repository root):
  python code/ch12_continuous_control_actor_critic/diagrams.py          # writes both figures
  python code/ch12_continuous_control_actor_critic/diagrams.py --quick  # draws them, writes nothing
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import FIG_DIR
from plot_style import C, GREY, INK, setup


def dpg_intuition(plt):
    """Q(s, .) for one state, the current action, the tangent and the uphill step."""
    a = np.linspace(-1, 1, 400)
    q = lambda x: -1.6 * (x - 0.35) ** 2 + 0.12 * np.sin(6 * x) - 2.0      # an arbitrary smooth critic
    dq = lambda x: -3.2 * (x - 0.35) + 0.72 * np.cos(6 * x)
    mu = -0.45
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.plot(a, q(a), color=C[0], lw=2.2, label="critic  Q_w(s, a)  for one fixed state s")
    xs = np.linspace(mu - 0.3, mu + 0.3, 10)
    ax.plot(xs, q(mu) + dq(mu) * (xs - mu), color=C[1], ls="--", lw=1.6,
            label="tangent: slope dQ/da at a = mu_theta(s)")
    ax.plot([mu], [q(mu)], "o", color=C[1], ms=8, zorder=5)
    ax.annotate("current action mu_theta(s)", (mu, q(mu)), xytext=(-0.42, q(mu) - 0.75),
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    # the uphill step along the action axis
    step = 0.25
    ax.add_patch(FancyArrowPatch((mu, q(mu) + 0.08), (mu + step, q(mu + step) + 0.08), arrowstyle="-|>",
                                 mutation_scale=16, color=C[2], lw=2.2,
                                 connectionstyle="arc3,rad=-0.25", zorder=6))
    ax.text(-0.97, q(0.35) + 0.75, "DPG: move the action uphill,\n"
            "delta a  proportional to  dQ/da;\nthe actor changes theta so that\n"
            "mu_theta(s) moves that way (chain rule)", fontsize=8.5, color=C[2], va="top")
    a_star = a[np.argmax(q(a))]
    ax.axvline(a_star, color=GREY, ls=":", lw=1.2)
    ax.text(a_star + 0.03, q(a_star) - 1.45, "argmax_a Q(s, a):\nwhat DQN would need\n"
            "(an inner optimization\nfor every state)", fontsize=8.5, color=GREY)
    ax.set_xlabel("action a  (normalized to [-1, 1])")
    ax.set_ylabel("Q_w(s, a)")
    ax.set_xlim(-1, 1)
    ax.set_ylim(q(-1) - 0.2, q(a_star) + 0.9)
    ax.set_title("The deterministic policy gradient in one state")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def _box(ax, xy, w, h, text, face, fs=8.3, bold_first=True):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=face, ec="#8a8985", lw=0.9))
    lines = text.split("\n")
    if bold_first:
        ax.text(x, y + h / 2 - 0.2, lines[0], ha="center", va="top", fontsize=fs + 0.7, fontweight="bold",
                color=INK)
        ax.text(x, y + h / 2 - 0.55, "\n".join(lines[1:]), ha="center", va="top", fontsize=fs, color=INK,
                linespacing=1.35)
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.35)


def _arrow(ax, p, q, text="", color=INK, ls="-", rad=0.0, tx=None, fs=8, lw=1.3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        x, y = tx if tx is not None else ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.15)
        ax.text(x, y, text, ha="center", va="bottom", fontsize=fs, color=color)


def dataflow(plt):
    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(-1.75, 6.0)
    ax.axis("off")
    online, target, loss, data = "#dbe9f8", "#ecebe7", "#fde4d6", "#d9f1e7"
    # replay buffer (left column)
    _box(ax, (1.1, 2.3), 1.9, 5.6, "replay buffer D\n\n(s, a, r, s', term)\n\nfilled by acting:\n"
         "DDPG/TD3:\nclip(mu_theta(s)\n + noise)\nSAC:\na ~ pi_theta(.|s)\n\nterm = 1 only on\ntermination,\n"
         "never on\ntruncation", data)
    # row 1: the target
    _box(ax, (4.4, 4.3), 2.9, 1.75, "target action a'\nDDPG: mu_thetabar(s')\nTD3: + clipped noise (12.10)\n"
         "SAC: a' ~ pi_theta(.|s'), current\npolicy, no target actor", target)
    _box(ax, (7.75, 4.3), 2.3, 1.75, "target critics\nQ_wbar1, Q_wbar2\n(DDPG: one critic)", target)
    _box(ax, (11.4, 4.3), 3.6, 1.75, "TD target y  (12.5, 12.9, 12.28)\nr + gamma (1 - term) [ min_j Q_wbarj(s', a')\n"
         "  - alpha log pi_theta(a'|s')  (SAC only) ]\nDDPG: no min; computed without gradient", loss)
    # row 2: the critic update
    _box(ax, (7.75, 1.9), 2.3, 1.45, "critics\nQ_w1, Q_w2\n(DDPG: Q_w)", online)
    _box(ax, (11.4, 1.9), 3.6, 1.45, "critic loss  (12.5, 12.9, 12.29)\nsum_j (Q_wj(s, a) - y)^2\n"
         "-> gradient step on w, every update", loss)
    # row 3: the actor update
    _box(ax, (4.4, -0.35), 2.9, 1.6, "actor\nDDPG/TD3: mu_theta(s)\nSAC: a~ = tanh(m_theta(s)\n"
         "  + sigma_theta(s) xi),  xi ~ N(0, I)", online)
    _box(ax, (11.4, -0.35), 3.6, 1.6, "actor loss  (12.6, 12.30)\nDDPG/TD3: -Q_w1(s, mu_theta(s))\n"
         "SAC: alpha log pi(a~|s) - min_j Q_wj(s, a~)\n-> step on theta only (TD3: every d = 2)", loss)
    _box(ax, (11.4, -1.45), 3.6, 0.55, "SAC only: temperature loss (12.33) -> step on log alpha",
         "#f3eefb", fs=7.8, bold_first=False)
    # data arrows
    _arrow(ax, (2.05, 4.3), (2.95, 4.3), "s'")
    _arrow(ax, (5.85, 4.3), (6.6, 4.3), "a'")
    _arrow(ax, (8.9, 4.3), (9.6, 4.3), "Q(s', a')")
    ax.plot([1.1, 1.1, 11.4], [5.1, 5.65, 5.65], color=INK, lw=1.3)
    _arrow(ax, (11.4, 5.65), (11.4, 5.2))
    ax.text(6.2, 5.7, "r, term", fontsize=8, ha="center", va="bottom")
    _arrow(ax, (2.05, 1.9), (6.6, 1.9), "s, a  (stored actions)", tx=(3.3, 2.0))
    _arrow(ax, (8.9, 1.9), (9.6, 1.9), "Q(s, a)", tx=(9.25, 2.0))
    _arrow(ax, (11.4, 3.42), (11.4, 2.63), "y", tx=(11.6, 2.9))
    _arrow(ax, (2.05, -0.35), (2.95, -0.35), "s")
    _arrow(ax, (5.85, 0.0), (6.9, 1.17), "mu_theta(s) or a~", tx=(5.95, 0.75))
    _arrow(ax, (8.6, 1.17), (9.6, 0.1), "Q(s, mu_theta(s))", tx=(9.75, 0.8))
    # gradient path back through the critic
    _arrow(ax, (9.6, -0.6), (5.85, -0.6), "", color=C[7], ls="--", lw=1.6)
    ax.text(7.72, -0.72, "gradient of the actor loss flows back through the critic:\n"
            "grad_theta mu(s) x grad_a Q(s, a)  (chain rule, 12.6 / 12.24);\ncritic weights are not updated by it",
            ha="center", va="top", fontsize=7.3, color=C[7])
    # Polyak copies
    _arrow(ax, (7.75, 2.63), (7.75, 3.42), "", color=GREY, ls=":", lw=1.6)
    ax.text(7.85, 3.0, "Polyak, tau (12.7)", fontsize=7.8, color=GREY, va="center")
    _arrow(ax, (4.4, 0.45), (4.4, 3.42), "", color=GREY, ls=":", lw=1.6)
    ax.text(4.5, 2.6, "Polyak, tau (12.7)\nDDPG/TD3 only", fontsize=7.8, color=GREY, va="center")
    ax.set_title("Off-policy actor-critic data flow: DDPG, TD3 and SAC", fontsize=12)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    print("Chapter 12 diagrams (deterministic, no seed needed)")
    plt = setup()
    figs = {"dpg_intuition.png": dpg_intuition(plt), "actor_critic_dataflow.png": dataflow(plt)}
    if args.quick:
        print("quick mode: figures drawn but not saved")
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    for name, fig in figs.items():
        path = os.path.join(FIG_DIR, name)
        fig.savefig(path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
