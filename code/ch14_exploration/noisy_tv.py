"""The noisy-TV problem: prediction-error curiosity vs RND vs counts.

Chapter 14, Section 7.2.

Reward-free four-rooms world (no goal) with a TV at (4, 4), six moves (three down, three
right) from the start (1, 1).  Standing on the TV cell the agent sees one of C = 10
channels; the extra action "press the remote" switches to a uniformly random channel.
That is the only stochastic transition.  Three agents use only intrinsic reward (same
two-stream tabular Q-learner as rnd_gridworld.py, extrinsic stream unused; there is no
termination, so both streams bootstrap through the 100-step time limit):
  forward-model curiosity  r_int = squared error of a tabular forward model predicting s'
                           from (s, a): an exponential moving average (step 0.1) of
                           one-hot(s').  At the TV even the best possible prediction has
                           error 1 - 1/C = 0.9 forever (aleatoric uncertainty); the EMA's
                           expected error is (1 - 1/C)(1 + 0.1/1.9) = 0.947.
  count bonus              r_int = 1/sqrt(n(s')).  The C screens are just C more states.
  RND                      r_int = RND error of the next observation (cell + channel).
We record the fraction of steps spent on the TV, the number of cells explored, and (for
the curiosity agent) the measured forward-model error of "press" transitions at the TV.

Run:  python code/ch14_exploration/noisy_tv.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from explore_lib import FOUR_ROOMS, RND, CountBonus, ForwardCuriosity, GridWorld, TwoStreamQ, train

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
torch.set_num_threads(1)

TV_ROOMS = [row.replace("G", ".") for row in FOUR_ROOMS]          # reward-free
TV_ROOMS[4] = TV_ROOMS[4][:4] + "T" + TV_ROOMS[4][5:]              # TV at (4, 4)


def run(kind, seed, episodes, K):
    rng = np.random.default_rng(seed)
    env = GridWorld(TV_ROOMS, horizon=100, n_channels=K, rng=rng)
    bonus = {"forward-model curiosity": lambda: ForwardCuriosity(env),
             "count bonus": lambda: CountBonus(env),
             "RND": lambda: RND(env, seed=seed)}[kind]()
    agent = TwoStreamQ(env.n_states, env.n_actions, beta=1.0)
    out = train(env, agent, bonus, episodes, rng)
    out["env"] = env
    out["tv_errors"] = getattr(bonus, "tv_errors", [])
    out["n_clipped"] = getattr(bonus, "n_clipped", 0)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--channels", type=int, default=10, help="number of TV channels C")
    args = parser.parse_args()
    seeds = [0] if args.quick else list(range(args.seeds))
    episodes = 30 if args.quick else args.episodes
    K = args.channels
    kinds = ["forward-model curiosity", "count bonus", "RND"]

    env0 = GridWorld(TV_ROOMS, n_channels=K)
    print("Noisy TV: forward-model curiosity vs count bonus vs RND (reward-free)")
    print(f"grid {env0.h}x{env0.w}, {env0.n_cells} cells, start {env0.start}, TV at "
          f"{env0.tv} with C={K} channels, horizon 100, episodes {episodes}, seeds {seeds}")
    print("agent: tabular Q-learning on intrinsic reward only, alpha 0.2, gamma_I 0.99, "
          "eps 0.05, intrinsic coefficient 1; forward model = exponential moving average "
          "of one-hot(s') (step 0.1)")
    print(f"irreducible forward-model error at the TV: 1 - 1/C = {1 - 1 / K:.3f}; expected "
          f"error of the EMA predictor (1 - 1/C)(1 + 0.1/1.9) = {(1 - 1 / K) * (1 + 0.1 / 1.9):.3f}")
    t0 = time.time()
    res = {k: [run(k, s, episodes, K) for s in seeds] for k in kinds}

    print("\nResults (mean over seeds; [min, max])")
    q = max(1, episodes // 4)
    for k in kinds:
        tv_last = np.array([r["tv_frac"][-q:].mean() for r in res[k]])
        tv_first = np.array([r["tv_frac"][:q].mean() for r in res[k]])
        cov = np.array([r["coverage"][-1] for r in res[k]])
        print(f"  {k:24s} TV share of steps: first {q} eps {tv_first.mean():.3f}, last {q} "
              f"eps {tv_last.mean():.3f} [{tv_last.min():.3f}, {tv_last.max():.3f}] | "
              f"cells visited {cov.mean():.1f} [{cov.min()}, {cov.max()}] of {env0.n_cells}")
    print(f"  RND rewards clipped at 10 (steps per seed): {[r['n_clipped'] for r in res['RND']]}")
    tv_err = [np.mean(r["tv_errors"][len(r["tv_errors"]) // 2:]) for r in
              res["forward-model curiosity"] if len(r["tv_errors"]) > 1]
    if tv_err:
        print(f"  forward-model error of 'press' at the TV (second half of each run's "
              f"presses): mean {np.mean(tv_err):.3f} [{min(tv_err):.3f}, {max(tv_err):.3f}]")
    print(f"Total time {time.time() - t0:.1f} s")

    if args.quick:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    os.makedirs(FIG_DIR, exist_ok=True)
    colors = {"forward-model curiosity": "tab:orange", "count bonus": "tab:blue",
              "RND": "tab:purple"}
    fig = plt.figure(figsize=(14, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.45)
    ax0, ax1 = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2])
    eps_axis = np.arange(1, episodes + 1)
    for k in kinds:
        tv = np.array([r["tv_frac"] for r in res[k]])
        ax0.plot(eps_axis, tv.mean(0), color=colors[k], label=k)
        ax0.fill_between(eps_axis, tv.min(0), tv.max(0), color=colors[k], alpha=0.15)
        cov = np.array([r["coverage"] for r in res[k]])
        ax1.plot(eps_axis, cov.mean(0), color=colors[k], label=k)
    ax0.set_xlabel("episode")
    ax0.set_ylabel("fraction of steps on the TV")
    ax0.set_title(f"Who gets hooked on the noisy TV? (C={K} channels, mean, band = min-max)")
    ax0.legend()
    ax0.grid(alpha=0.3)
    ax1.axhline(env0.n_cells, color="k", lw=0.8, ls=":")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("distinct cells visited")
    ax1.set_title("Coverage")
    ax1.grid(alpha=0.3)
    env = res["RND"][0]["env"]

    def visit_grid(r):
        # The TV cell's count is the sum over its K screen states.
        grid = np.full((env.h, env.w), np.nan)
        for i, (row, col) in enumerate(env.free):
            v = r["visits"][i] + (r["visits"][env.n_cells:].sum() if (row, col) == env.tv else 0)
            grid[row, col] = v if v > 0 else np.nan
        return grid

    grids = {k: visit_grid(res[k][0]) for k in kinds}
    vmax = max(np.nanmax(g) for g in grids.values())
    for j, k in enumerate(kinds):
        ax = fig.add_subplot(gs[1, j])
        walls = np.array([[1.0 if ch == "#" else np.nan for ch in row] for row in env.layout])
        ax.imshow(walls, cmap="Greys", vmin=0, vmax=1.2)
        im = ax.imshow(grids[k], cmap="viridis", norm=LogNorm(vmin=1, vmax=vmax))
        ax.add_patch(plt.Rectangle((env.tv[1] - 0.5, env.tv[0] - 0.5), 1, 1, fill=False,
                                   edgecolor="red", lw=2))
        tv_share = res[k][0]["visits"][env.n_cells:].sum() / res[k][0]["visits"].sum()
        ax.set_title(f"{k} (seed 0)\nred square = TV, {100 * tv_share:.0f}% of all visits")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=fig.axes[2:], shrink=0.8, label="visits (log; white = never)")
    path = os.path.join(FIG_DIR, "noisy_tv.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
