"""Random Network Distillation on a sparse-reward four-rooms gridworld: coverage maps.

Chapter 14, Sections 6-7 (count bonuses and RND; results in Section 7.3).

The 21 x 21 four-rooms world (328 free cells) has a single reward, +1 at the far corner of
the bottom-right room, 36 steps from the start.  Episodes are truncated after 100 steps.
All agents share the SAME tabular two-stream Q-learner (explore_lib.TwoStreamQ, the
Sec. 7.3 algorithm box) and differ only in the intrinsic reward:
  no bonus      beta = 0: until the goal is found every Q is 0, so with random
                tie-breaking the agent is a uniform random walk (eps-greedy w/o signal)
  count bonus   r_int = 1/sqrt(n(s')), exact tabular counts (Section 6.1)
  RND           r_int = ||f_hat(x') - f(x')||^2 / running sd, small MLPs (Section 7.3)
Variants (full mode):
  "... , intrinsic stops at goal"  the intrinsic return ends when the goal is reached
                (target r_i alone), instead of continuing from the start state.  The
                default non-episodic stream makes agents PROCRASTINATE next to the goal:
                entering G hands the intrinsic stream over to the over-visited start
                region, whose centred intrinsic value is negative.
  "count bonus, not centred"       the intrinsic reward is not centred by its running
                mean (the ablation behind Pitfall 3).
For each run we record which cells were visited, how fast coverage grows, when the goal
was first reached, how often the exploratory behaviour reaches it, and how often a GREEDY
evaluation episode on Q_E alone (beta = 0, eps = 0; every 5 episodes) reaches it.  We also
print the most-visited cell of every run.  At the end we compare the RND error of every
cell with its visit count, to check that RND behaves like a (pseudo-)count.

Run:  python code/ch14_exploration/rnd_gridworld.py [--quick] [--no-centre]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from scipy import stats

from explore_lib import FOUR_ROOMS, RND, CountBonus, GridWorld, TwoStreamQ, train

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
torch.set_num_threads(1)

# name -> (bonus kind, intrinsic stream at the goal, centre the intrinsic reward?)
AGENTS = {
    "no bonus": ("none", "continue", True),
    "count bonus": ("count", "continue", True),
    "RND": ("rnd", "continue", True),
    "count, stop at goal": ("count", "stop", True),
    "RND, stop at goal": ("rnd", "stop", True),
    "count, not centred": ("count", "continue", False),
}
MAIN = ["no bonus", "count bonus", "RND"]


def make_bonus(kind, env, seed):
    if kind == "none":
        return None
    if kind == "count":
        return CountBonus(env)
    return RND(env, seed=seed)


def run(name, seed, episodes, beta, centre_override=None):
    kind, at_goal, centre = AGENTS[name]
    if centre_override is not None:
        centre = centre_override
    rng = np.random.default_rng(seed)
    env = GridWorld(FOUR_ROOMS, horizon=100, rng=rng)
    bonus = make_bonus(kind, env, seed)
    agent = TwoStreamQ(env.n_states, env.n_actions, beta=0.0 if bonus is None else beta,
                       intrinsic_at_goal=at_goal)
    out = train(env, agent, bonus, episodes, rng, centre=centre,
                eval_env=GridWorld(FOUR_ROOMS, horizon=100), eval_every=5,
                eval_rng=np.random.default_rng(1_000 + seed))
    out["env"] = env
    out["agent"] = agent
    out["rnd_err"] = bonus.errors() if isinstance(bonus, RND) else None
    out["n_clipped"] = bonus.n_clipped if isinstance(bonus, RND) else 0
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--beta", type=float, default=0.1, help="intrinsic coefficient")
    parser.add_argument("--no-centre", action="store_true",
                        help="do not centre the intrinsic reward (all bonus agents)")
    args = parser.parse_args()
    seeds = [0] if args.quick else list(range(args.seeds))
    episodes = 40 if args.quick else args.episodes
    names = MAIN if args.quick else list(AGENTS)
    centre_override = False if args.no_centre else None

    env0 = GridWorld(FOUR_ROOMS)
    print("RND vs count bonus vs no bonus on a sparse-reward four-rooms gridworld")
    print(f"grid {env0.h}x{env0.w}, {env0.n_cells} free cells, horizon 100, "
          f"episodes {episodes}, seeds {seeds}")
    print(f"agent: two-stream tabular Q-learning (online + reverse replay of each episode), "
          f"alpha 0.2, gamma_E = gamma_I = 0.99, eps 0.05, intrinsic coefficient beta "
          f"{args.beta}, intrinsic reward centred by its running mean"
          + (" -- DISABLED (--no-centre)" if args.no_centre else ""))
    print("RND: target 1 hidden layer (64), predictor 2 hidden layers (128), out dim 32, "
          "Adam 1e-3, one minibatch (32) of the last 256 observations every 4 steps, "
          "reward = error / running sd of errors (warm-started), clipped to [0, 10]")
    print("greedy evaluation: one episode greedy on Q_E (beta = 0, eps = 0) every 5 episodes")
    t0 = time.time()
    res = {}
    for k in names:
        t_k = time.time()
        res[k] = [run(k, s, episodes, args.beta, centre_override) for s in seeds]
        print(f"  [{k}: {time.time() - t_k:.1f} s]")

    n_early = max(1, episodes // 6)
    print("\nResults (mean over seeds; [min, max])")
    for k in names:
        cov = np.array([r["coverage"][-1] for r in res[k]])
        first = [int(np.argmax(r["goals"])) + 1 if r["goals"].any() else None for r in res[k]]
        last = np.array([r["goals"][-50:].mean() for r in res[k]])
        n_goals = [int(r["goals"].sum()) for r in res[k]]
        cov_early = np.array([r["coverage"][n_early - 1] for r in res[k]])
        n_ev = max(1, 50 // res[k][0]["eval_every"])
        greedy_last = np.array([r["evals"][-n_ev:].mean() for r in res[k]])
        greedy_first = [int((np.argmax(r["evals"]) + 1) * r["eval_every"]) if r["evals"].any()
                        else None for r in res[k]]
        print(f"  {k:20s} cells visited: after {n_early} eps {cov_early.mean():6.1f}, "
              f"after {episodes} eps {cov.mean():6.1f} [{cov.min()}, {cov.max()}] of "
              f"{env0.n_cells}")
        print(f"  {'':20s} first goal episode {first} | goals per run {n_goals} | behaviour "
              f"goal rate in last 50 eps {last.mean():.2f} [{last.min():.2f}, {last.max():.2f}]")
        print(f"  {'':20s} greedy-Q_E eval: success rate over last 50 eps "
              f"{greedy_last.mean():.2f} [{greedy_last.min():.2f}, {greedy_last.max():.2f}] "
              f"| first greedy success after episode {greedy_first}")
        tops = []
        for r in res[k]:
            env = r["env"]
            i = int(np.argmax(r["visits"]))
            tops.append(f"{env.free[i]}:{int(r['visits'][i])} "
                        f"({100 * r['visits'][i] / r['visits'].sum():.0f}%)")
        print(f"  {'':20s} most-visited cell per seed (row, col):visits (share): "
              + ", ".join(tops))
        if any(r["n_clipped"] for r in res[k]):
            print(f"  {'':20s} RND rewards clipped at 10: "
                  f"{[r['n_clipped'] for r in res[k]]} steps")

    # Procrastination check: runs whose most-visited cell is next to the goal.  Show the
    # two value streams there: the action that ENTERS G has the top Q_E but a low
    # intrinsic value (the non-episodic stream continues from the over-visited start).
    print("\nRuns whose most-visited cell is adjacent to the goal (Q at that cell; actions "
          "up, down, left, right):")
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for k in names:
        for seed, r in zip(seeds, res[k]):
            env, ag = r["env"], r["agent"]
            i = int(np.argmax(r["visits"]))
            cell = env.free[i]
            if abs(cell[0] - env.goal[0]) + abs(cell[1] - env.goal[1]) != 1:
                continue
            a_in = moves.index((env.goal[0] - cell[0], env.goal[1] - cell[1]))
            s0 = env.cell_id[env.start]
            print(f"  {k:20s} seed {seed} cell {cell}: Q_E {np.round(ag.QE[i], 3)}, "
                  f"beta*Q_I {np.round(ag.beta * ag.QI[i], 3)}, enter-G action = "
                  f"{['up', 'down', 'left', 'right'][a_in]}; greedy on Q_E + beta*Q_I picks "
                  f"{['up', 'down', 'left', 'right'][int(np.argmax(ag.QE[i] + ag.beta * ag.QI[i]))]}"
                  f"; max_a beta*Q_I(start) = {ag.beta * ag.QI[s0].max():+.3f}")

    # Does RND's error track visit counts? Spearman rank correlation over all cells.
    print("\nRND error vs visit count at the end of training (all cells, per seed):")
    for r in res["RND"]:
        rho = stats.spearmanr(r["visits"], r["rnd_err"]).statistic
        unvisited = r["visits"] == 0
        e_unv = r["rnd_err"][unvisited].mean() if unvisited.any() else float("nan")
        e_top = r["rnd_err"][r["visits"] >= np.quantile(r["visits"], 0.9)].mean()
        print(f"  Spearman rho = {rho:+.3f}; mean error on unvisited cells {e_unv:.1e} "
              f"({unvisited.sum()} cells) vs 10% most-visited {e_top:.1e}")
    print(f"Total time {time.time() - t0:.1f} s")

    if args.quick:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    os.makedirs(FIG_DIR, exist_ok=True)
    env = res["RND"][0]["env"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    vmax = max(r["visits"].max() for k in MAIN for r in res[k][:1])
    for ax, k in zip(axes, MAIN):
        r = res[k][0]
        grid = np.full((env.h, env.w), np.nan)
        for i, (row, col) in enumerate(env.free):
            grid[row, col] = r["visits"][i] if r["visits"][i] > 0 else np.nan
        walls = np.array([[1.0 if ch == "#" else np.nan for ch in row] for row in env.layout])
        ax.imshow(walls, cmap="Greys", vmin=0, vmax=1.2)
        im = ax.imshow(grid, cmap="viridis", norm=LogNorm(vmin=1, vmax=vmax))
        ax.text(env.start[1], env.start[0], "S", color="w", ha="center", va="center",
                fontweight="bold")
        ax.text(env.goal[1], env.goal[0], "G", color="r", ha="center", va="center",
                fontweight="bold")
        ax.set_title(f"{k}: {int((r['visits'][:env.n_cells] > 0).sum())}/{env.n_cells} "
                     f"cells, {int(r['goals'].sum())} goals")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label="visits (log scale; white = never)")
    fig.suptitle(f"State visitation over {episodes} episodes x 100 steps (seed 0)")
    path = os.path.join(FIG_DIR, "rnd_coverage_maps.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")

    colors = {"no bonus": "tab:red", "count bonus": "tab:blue", "RND": "tab:purple",
              "count, stop at goal": "tab:blue", "RND, stop at goal": "tab:purple"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for k in MAIN:
        # Seeds take different numbers of steps (goal episodes are short), so put every
        # seed's coverage on a common step grid before averaging.
        grid_steps = np.linspace(0, min(r["steps"][-1] for r in res[k]), 400)
        cov = np.array([np.interp(grid_steps, np.r_[0, r["steps"]], np.r_[1, r["coverage"]])
                        for r in res[k]])
        axes[0].plot(grid_steps, cov.mean(0), color=colors[k], label=k)
        axes[0].fill_between(grid_steps, cov.min(0), cov.max(0), color=colors[k], alpha=0.15)
    for k in [n for n in names if n in colors]:
        g = np.array([np.cumsum(r["goals"]) for r in res[k]])
        ls = "--" if "stop" in k else "-"
        axes[1].plot(np.arange(1, episodes + 1), g.mean(0), color=colors[k], ls=ls, label=k)
        if ls == "-":
            axes[1].fill_between(np.arange(1, episodes + 1), g.min(0), g.max(0),
                                 color=colors[k], alpha=0.15)
    axes[0].axhline(env.n_cells, color="k", lw=0.8, ls=":")
    axes[0].set_xlabel("environment steps")
    axes[0].set_ylabel("distinct cells visited")
    axes[0].set_title("Coverage (mean, band = min-max)\n"
                      "each curve ends at its shortest seed's total steps", fontsize=10)
    axes[0].legend()
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("cumulative episodes reaching G (behaviour policy)")
    axes[1].set_title("Sparse extrinsic reward found\n"
                      "dashed: intrinsic return stops at the goal", fontsize=10)
    axes[1].legend(fontsize=8)
    r = res["RND"][0]
    v = np.maximum(r["visits"], 0.5)
    axes[2].scatter(v, r["rnd_err"], s=10, alpha=0.6, color="tab:purple")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    rho = stats.spearmanr(r["visits"], r["rnd_err"]).statistic
    axes[2].set_xlabel("visit count n(s) (0 plotted at 0.5)")
    axes[2].set_ylabel("RND prediction error at end")
    axes[2].set_title(f"RND error vs count, seed 0 (Spearman {rho:+.2f})")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "rnd_coverage_curves.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
