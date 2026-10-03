"""First-visit Monte Carlo prediction on Blackjack (Sec. 2.5; S&B Example 5.1, Figure 5.1).

Evaluates S&B's policy "stick only on 20 or 21" with first-visit MC (sample averages),
reproduces the value surfaces after 10,000 and 500,000 episodes, and -- unlike the book --
compares them with the EXACT values computed by blackjack.solve(), so we can watch the
RMS error fall like 1/sqrt(n). A constant-step-size variant (V += alpha (G - V)) is run
for comparison: it tracks recent returns and therefore stalls at an error floor.

    python code/ch04_monte_carlo/mc_prediction_blackjack.py [--quick]
"""
import argparse
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blackjack import (Blackjack, HIT, STICK, STATE_SHAPE, all_states, flat_index,
                       onpolicy_targets, stick_on_20_policy)

HERE = os.path.dirname(os.path.abspath(__file__))
GAMMA = 1.0


def policy_stick_20(obs):
    return HIT if obs[0] < 20 else STICK


def mc_prediction(env, policy, n_episodes, checkpoints=(), first_visit=True, alpha=None):
    """First-visit (or every-visit) MC prediction -- mirrors the pseudocode in Sec. 2.1.

    V[s] is a running average of the returns observed from s (Sec. 2.4), or, if `alpha`
    is given, an exponential recency-weighted average with constant step size alpha.
    Returns the final V (as a STATE_SHAPE array) and a dict {episode: V snapshot}."""
    V = [0.0] * 200
    N = [0] * 200
    snaps = {}
    checkpoints = set(checkpoints)
    for ep in range(1, n_episodes + 1):
        # 1. Generate an episode S0, A0, R1, ..., S_{T-1}, A_{T-1}, R_T following pi.
        states, rewards = [], []
        obs, done = env.reset(), False
        while not done:
            states.append(flat_index(obs))
            obs, r, done = env.step(policy(obs))
            rewards.append(r)
        # 2. Walk backwards accumulating the return G_t = R_{t+1} + gamma G_{t+1}.
        G = 0.0
        for t in range(len(states) - 1, -1, -1):
            G = GAMMA * G + rewards[t]
            s = states[t]
            if first_visit and s in states[:t]:      # only the first visit to s counts
                continue                             # (never happens in Blackjack, see text)
            N[s] += 1
            V[s] += ((1.0 / N[s]) if alpha is None else alpha) * (G - V[s])
        if ep in checkpoints:
            snaps[ep] = np.array(V).reshape(STATE_SHAPE)
    return np.array(V).reshape(STATE_SHAPE), snaps


def plot_surfaces(snaps, exact, path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
    dealer, player = np.meshgrid(np.arange(1, 11), np.arange(12, 22))
    cols = [(f"after {n:,} episodes", snaps[n]) for n in sorted(snaps)] + [("exact (solver)", exact)]
    fig = plt.figure(figsize=(13, 8))
    for row, (usable, label) in enumerate(((1, "usable ace"), (0, "no usable ace"))):
        for col, (title, V) in enumerate(cols):
            ax = fig.add_subplot(2, len(cols), row * len(cols) + col + 1, projection="3d")
            ax.plot_surface(dealer, player, V[:, :, usable], cmap="viridis", vmin=-1, vmax=1,
                            edgecolor="k", linewidth=0.2)
            ax.set_zlim(-1, 1)
            ax.set_xlabel("dealer showing", fontsize=8)
            ax.set_ylabel("player sum", fontsize=8)
            ax.set_xticks([1, 4, 7, 10])
            ax.set_xticklabels(["A", "4", "7", "10"], fontsize=7)
            ax.set_yticks([12, 15, 18, 21])
            ax.tick_params(labelsize=7)
            ax.view_init(elev=28, azim=-125)
            ax.set_title(f"{label}\n{title}", fontsize=9)
    fig.suptitle("Blackjack: state-value function of 'stick only on 20 or 21' (first-visit MC)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n_main = 20_000 if args.quick else 500_000
    n_seeds = 2 if args.quick else 5
    n_curve = 20_000 if args.quick else 500_000
    alpha = 0.01
    curve_seeds = [args.seed + 1000 + k for k in range(n_seeds)]
    print(f"MC prediction, Blackjack | seed={args.seed} episodes={n_main:,} gamma={GAMMA} "
          f"RMS curves: seeds {curve_seeds} x {n_curve:,} episodes, constant alpha={alpha}")
    t0 = time.time()

    target, visit = onpolicy_targets(stick_on_20_policy())
    rms = lambda V: float(np.sqrt(np.mean((V - target) ** 2)))

    # Figure 5.1 reproduction (one run, seed = args.seed).
    checkpoints = [10_000, n_main]
    _, snaps = mc_prediction(Blackjack(args.seed), policy_stick_20, n_main, checkpoints)
    for n in checkpoints:
        V = snaps[n]
        print(f"after {n:>7,} episodes: RMS error over 200 states = {rms(V):.4f}  "
              f"(usable ace {np.sqrt(np.mean((V - target)[..., 1] ** 2)):.4f}, "
              f"no usable ace {np.sqrt(np.mean((V - target)[..., 0] ** 2)):.4f})")
    p_usable = visit[..., 1].sum() / visit.sum()
    print(f"fraction of state visits with a usable ace: {p_usable:.3f}  "
          f"(least-visited state reached in {100 * visit.min():.3f}% of episodes)")

    # Every-visit gives identical estimates here: no state repeats within an episode.
    V_fv, _ = mc_prediction(Blackjack(args.seed + 1), policy_stick_20, 20_000, first_visit=True)
    V_ev, _ = mc_prediction(Blackjack(args.seed + 1), policy_stick_20, 20_000, first_visit=False)
    print(f"max |V_first-visit - V_every-visit| on the same 20,000 episodes: {np.abs(V_fv - V_ev).max():.1e}")

    # RMS error vs number of episodes, averaged over seeds.
    grid = np.unique(np.round(np.logspace(2, np.log10(n_curve), 15)).astype(int))
    curves = {"sample average (1/N)": [], f"constant alpha = {alpha}": []}
    for cs in curve_seeds:
        for name, a in (("sample average (1/N)", None), (f"constant alpha = {alpha}", alpha)):
            _, sn = mc_prediction(Blackjack(cs), policy_stick_20, n_curve, grid, alpha=a)
            curves[name].append([rms(sn[n]) for n in grid])
    print("\nRMS error vs episodes (mean over seeds):")
    print("  episodes | " + " | ".join(curves))
    for i, n in enumerate(grid):
        print(f"  {n:8,d} | " + " | ".join(f"{np.mean(c, axis=0)[i]:.4f}".rjust(len(name))
                                           for name, c in curves.items()))
    sa = np.mean(curves["sample average (1/N)"], axis=0)
    # Early on the error is dominated by rarely visited states that still hold V = 0, so we
    # fit the slope on two ranges: n >= 1,000 and n >= 10,000 (the clean 1/sqrt(n) regime).
    for lo in (1_000, 10_000):
        sel = grid >= lo
        if sel.sum() >= 2:
            slope = np.polyfit(np.log(grid[sel]), np.log(sa[sel]), 1)[0]
            print(f"log-log slope of sample-average RMS error (n >= {lo:,}): {slope:.3f}  (theory: -0.5)")
    print(f"elapsed {time.time() - t0:.1f}s")

    if args.quick:
        return
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    plot_surfaces(snaps, target, os.path.join(HERE, "figures", "blackjack_mc_prediction.png"))
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for name, c in curves.items():
        c = np.array(c)
        ax.loglog(grid, c.mean(0), "o-", ms=3, label=name)
        ax.fill_between(grid, c.min(0), c.max(0), alpha=0.2)
    # Reference slope anchored at the LAST point and drawn only where the 1/sqrt(n) law should
    # hold; anchoring at n = 100 (where unvisited states dominate the error) would mislead.
    sel = grid >= 1000
    ref = sa[-1] * np.sqrt(grid[-1] / grid[sel])
    ax.loglog(grid[sel], ref, "k:", label=r"$\propto 1/\sqrt{n}$ (anchored at the last point)")
    ax.set_xlabel("episodes n")
    ax.set_ylabel("RMS error vs exact values (200 states)")
    ax.set_title(f"First-visit MC prediction on Blackjack ({n_seeds} seeds, min-max band)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "blackjack_mc_prediction_rms.png"), dpi=110)
    print("saved figures/blackjack_mc_prediction.png, figures/blackjack_mc_prediction_rms.png")


if __name__ == "__main__":
    main()
