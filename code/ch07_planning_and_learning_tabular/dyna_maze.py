"""Dyna-Q on the Dyna maze (Sutton & Barto 2018, Example 8.1).

Chapter 07, Section 3. Reproduces the "steps per episode vs episode" learning
curves for n = 0 (plain one-step Q-learning), n = 5 and n = 50 planning steps
per real step, and the greedy policies found halfway through episode 2
(S&B Figure 8.3).

The agent is exactly the tabular Dyna-Q box of Section 2.1:
  (a)-(d) act eps-greedily, observe R, S', do a Q-learning update,
  (e)     store Model(S, A) <- (R, S', terminated)  [deterministic world],
  (f)     n times: pick a previously observed state uniformly, then an action
          previously taken in it uniformly, and do the same Q-learning update
          on the *simulated* transition.

Run:  python code/ch07_planning_and_learning_tabular/dyna_maze.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np

import maze_env
from maze_env import ARROWS, N_ACTIONS

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")


def greedy(q, rng):
    """Greedy action with RANDOM tie-breaking. With Q initialised to zero,
    always taking the first max would make the first episode a deterministic
    loop into a wall; random tie-breaking turns it into a random walk."""
    m = max(q)
    best = [a for a in range(N_ACTIONS) if q[a] == m]
    return best[0] if len(best) == 1 else rng.choice(best)


def eps_greedy(q, eps, rng):
    if rng.random() < eps:
        return rng.randrange(N_ACTIONS)
    return greedy(q, rng)


def dyna_q(maze, n_planning, n_episodes, alpha, gamma, eps, rng, snapshot_episode=None):
    """Tabular Dyna-Q (Section 2.1, box "Tabular Dyna-Q").

    Returns (steps_per_episode, snapshot) where snapshot is the list of Q-table
    copies taken after every real step of episode `snapshot_episode` (or None).
    """
    nS = maze.n_states
    table = maze.table
    Q = [[0.0] * N_ACTIONS for _ in range(nS)]
    model = [None] * (nS * N_ACTIONS)        # model[s*4+a] = (r, s', terminated)
    seen_states = []                         # states observed at least once
    seen_actions = [[] for _ in range(nS)]   # actions tried in each state
    steps_per_episode, snapshot = [], None
    for ep in range(n_episodes):
        s, steps = maze.start, 0
        record = (ep == snapshot_episode)
        if record:
            snapshot = []
        while True:
            # (b)-(c) act in the real environment
            a = eps_greedy(Q[s], eps, rng)
            s2, r, term = table[s][a]
            # (d) direct RL: one-step Q-learning (Chapter 05). No bootstrap at termination.
            target = r if term else r + gamma * max(Q[s2])
            Q[s][a] += alpha * (target - Q[s][a])
            # (e) model learning: deterministic world, so just remember the outcome
            key = s * N_ACTIONS + a
            if model[key] is None:
                if not seen_actions[s]:
                    seen_states.append(s)
                seen_actions[s].append(a)
            model[key] = (r, s2, term)
            # (f) planning: n simulated one-step Q-learning updates
            for _ in range(n_planning):
                ps = seen_states[rng.randrange(len(seen_states))]
                acts = seen_actions[ps]
                pa = acts[rng.randrange(len(acts))]
                pr, ps2, pterm = model[ps * N_ACTIONS + pa]
                ptarget = pr if pterm else pr + gamma * max(Q[ps2])
                Q[ps][pa] += alpha * (ptarget - Q[ps][pa])
            steps += 1
            if record:
                snapshot.append([row[:] for row in Q])
            if term:
                break
            s = s2
        steps_per_episode.append(steps)
    return steps_per_episode, snapshot


def greedy_arrows(maze, Q):
    """Arrow for every state whose greedy action has positive value (states
    still at all-zero value have no preferred action yet)."""
    arrows = {}
    for s in maze.free_states():
        if max(Q[s]) > 0:
            arrows[s] = ARROWS[int(np.argmax(Q[s]))]
    return arrows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    n_runs = 3 if args.quick else 30
    n_episodes = 50
    alpha, gamma, eps = 0.1, 0.95, 0.1
    planning_steps = [0, 5, 50]
    print(f"Dyna maze (S&B Ex. 8.1) | seed={args.seed} runs={n_runs} episodes={n_episodes} "
          f"alpha={alpha} gamma={gamma} eps={eps} n in {planning_steps}")

    maze = maze_env.dyna_maze()
    print(maze.render())
    print(f"shortest path from S to G: {maze.shortest_path_length()} steps\n")

    t0 = time.time()
    curves = {}
    for n in planning_steps:
        all_steps = []
        for run in range(n_runs):
            rng = random.Random(args.seed * 100_000 + run)  # same seeds for every n
            steps, _ = dyna_q(maze, n, n_episodes, alpha, gamma, eps, rng)
            all_steps.append(steps)
        curves[n] = np.array(all_steps, dtype=float)
        mean = curves[n].mean(axis=0)
        # first episode at which the mean curve gets within 10% of its final plateau
        final = mean[-10:].mean()
        reach = int(np.argmax(mean <= 1.1 * final)) + 1
        print(f"n={n:>2}: mean steps ep1={mean[0]:7.1f} ep2={mean[1]:6.1f} ep3={mean[2]:6.1f} "
              f"ep10={mean[9]:5.1f} | mean of eps 41-50={final:5.1f} "
              f"| first ep within 10% of that: {reach} | total real steps/run={curves[n].sum(1).mean():.0f}")
    print(f"(time {time.time() - t0:.1f}s)")

    # Sanity check. During episode 1 every reward is 0, so Q stays identically 0
    # and the eps-greedy agent (random tie-breaking) is a UNIFORM random walk.
    # Its expected hitting time h solves h(s) = 1 + (1/4) sum_a h(next(s,a)),
    # h(goal) = 0: a linear system we can solve exactly.
    free = maze.free_states()
    pos = {s: i for i, s in enumerate(free)}
    A, b = np.eye(len(free)), np.ones(len(free))
    for s in free:
        for a in range(N_ACTIONS):
            s2 = maze.table[s][a][0]
            if s2 != maze.goal:
                A[pos[s], pos[s2]] -= 1.0 / N_ACTIONS
    h = np.linalg.solve(A, b)
    pooled = np.concatenate([curves[n][:, 0] for n in planning_steps])
    print(f"episode 1 is a uniform random walk: exact expected length {h[pos[maze.start]]:.1f}; "
          f"empirical mean over all {len(pooled)} runs {pooled.mean():.1f} "
          f"(s.e. {pooled.std(ddof=1) / np.sqrt(len(pooled)):.1f})\n")

    # Policies halfway through the second episode (S&B Figure 8.3), one run each.
    snaps = {}
    for n in (0, 50):
        rng = random.Random(args.seed * 100_000 + 7)
        steps, snap = dyna_q(maze, n, 2, alpha, gamma, eps, rng, snapshot_episode=1)
        Q_half = snap[len(snap) // 2]
        arrows = greedy_arrows(maze, Q_half)
        snaps[n] = (arrows, steps)
        print(f"n={n}: greedy policy halfway through episode 2 (episode lengths {steps}); "
              f"{len(arrows)} of {len(maze.free_states())} states have a preferred action")
        print(maze.render(arrows), "\n")

    if args.quick:
        print("quick mode: no figures written")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    episodes = np.arange(1, n_episodes + 1)
    for n, color in zip(planning_steps, ["tab:blue", "tab:orange", "tab:green"]):
        mean = curves[n].mean(axis=0)
        se = curves[n].std(axis=0, ddof=1) / np.sqrt(n_runs)
        label = f"n = {n} planning steps" + (" (direct RL only)" if n == 0 else "")
        ax.plot(episodes[1:], mean[1:], color=color, label=label)
        ax.fill_between(episodes[1:], mean[1:] - se[1:], mean[1:] + se[1:], color=color, alpha=0.2)
    ax.axhline(maze.shortest_path_length(), color="gray", ls=":", label="shortest path (14)")
    ax.set_xlabel("episode")
    ax.set_ylabel("steps per episode")
    ax.set_ylim(0, 800)
    ax.set_title(f"Dyna-Q on the Dyna maze (mean of {n_runs} runs, episodes 2-50, ±1 s.e.)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "dyna_maze_learning_curves.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)

    # Figure 8.3-style policy picture
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, n in zip(axes, (0, 50)):
        arrows, steps = snaps[n]
        ax.set_xlim(-0.5, maze.n_cols - 0.5)
        ax.set_ylim(maze.n_rows - 0.5, -0.5)
        ax.set_xticks(np.arange(-0.5, maze.n_cols, 1), minor=False)
        ax.set_yticks(np.arange(-0.5, maze.n_rows, 1), minor=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, color="lightgray")
        ax.tick_params(length=0)
        for (r, c) in maze.walls:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="dimgray"))
        sr, sc = maze.start_cell
        gr, gc = maze.goal_cell
        ax.text(sc, sr, "S", ha="center", va="center", fontsize=14, weight="bold")
        ax.text(gc, gr, "G", ha="center", va="center", fontsize=14, weight="bold")
        for s, arrow in arrows.items():
            r, c = maze.cell(s)
            if s != maze.start:
                ax.text(c, r, arrow, ha="center", va="center", fontsize=15, color="tab:red")
            else:
                ax.text(c + 0.3, r + 0.3, arrow, ha="center", va="center", fontsize=10, color="tab:red")
        ax.set_title(f"n = {n}: greedy actions halfway through episode 2\n"
                     f"({len(arrows)} state{'s' if len(arrows) != 1 else ''} with a preferred action)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "dyna_maze_policies.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
