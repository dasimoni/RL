"""When the model is wrong: blocking and shortcut mazes (S&B Examples 8.2, 8.3).

Chapter 07, Section 4. Three agents are compared on two mazes that change
part-way through a run:

  * Dyna-Q      -- the plain algorithm of Section 2.1;
  * Dyna-Q+     -- planning rewards get an exploration bonus kappa*sqrt(tau),
                   where tau(s,a) is the number of real time steps since (s,a)
                   was last tried; actions never tried from a visited state are
                   allowed in planning with the model "stay put, reward 0"
                   (exactly as described by Sutton & Barto, Section 8.3);
  * Q+ (action bonus) -- the variant of S&B Exercise 8.4: the bonus is used
                   ONLY when selecting real actions, argmax_a Q + kappa*sqrt(tau),
                   and never enters the Q-values.

Blocking maze: the short route on the right is closed at t=1000 and a longer
route on the left opens.  Shortcut maze: at t=3000 a shorter route opens on the
right while the old route stays open.  Performance = cumulative reward (number
of times the goal was reached) vs time step.

Run:  python code/ch07_planning_and_learning_tabular/changing_mazes.py [--quick]
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time

import numpy as np

import maze_env
from maze_env import N_ACTIONS

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")


def argmax_random(values, rng):
    m = max(values)
    best = [a for a in range(len(values)) if values[a] == m]
    return best[0] if len(best) == 1 else rng.choice(best)


def run_agent(kind, maze, walls_before, walls_after, change_at, total_steps,
              n_planning, alpha, gamma, eps, kappa, rng):
    """Run one agent for `total_steps` real steps; the maze switches from
    `walls_before` to `walls_after` at step `change_at`.
    Returns the cumulative-reward curve (length total_steps)."""
    maze.set_walls(walls_before)
    nS = maze.n_states
    Q = [[0.0] * N_ACTIONS for _ in range(nS)]
    model = [None] * (nS * N_ACTIONS)        # (r, s', terminated)
    last_tried = [0] * (nS * N_ACTIONS)      # real time step of last try (for tau)
    seen_states, seen_actions = [], [[] for _ in range(nS)]
    seen = [False] * nS
    plus = (kind == "Dyna-Q+")
    action_bonus = (kind == "Q+ (action bonus)")
    cumulative = np.zeros(total_steps)
    total_reward = 0.0
    s = maze.start
    for t in range(1, total_steps + 1):
        if t == change_at + 1:
            maze.set_walls(walls_after)       # the world changes; the model does not know
            # If the agent happens to stand on a cell that has just become a wall
            # (e.g. the closed right gap of the blocking maze), it would otherwise
            # sit inside the wall and could still walk through the closed gap
            # once. Physically it cannot be there, so put it back at the start.
            # (The relocation is not a transition, so nothing enters the model.)
            if maze.cell(s) in maze.walls:
                s = maze.start
        # ---- act
        if rng.random() < eps:
            a = rng.randrange(N_ACTIONS)
        elif action_bonus:
            base = s * N_ACTIONS
            a = argmax_random([Q[s][b] + kappa * math.sqrt(t - last_tried[base + b])
                               for b in range(N_ACTIONS)], rng)
        else:
            a = argmax_random(Q[s], rng)
        s2, r, term = maze.table[s][a]
        total_reward += r
        cumulative[t - 1] = total_reward
        # ---- direct RL (no bonus on real experience)
        target = r if term else r + gamma * max(Q[s2])
        Q[s][a] += alpha * (target - Q[s][a])
        # ---- model learning
        if not seen[s]:
            seen[s] = True
            seen_states.append(s)
            if plus:
                # Dyna-Q+: untried actions may be planned with; initial model
                # "leads back to the same state with reward 0" (S&B Sec. 8.3).
                for b in range(N_ACTIONS):
                    model[s * N_ACTIONS + b] = (0.0, s, False)
                seen_actions[s] = list(range(N_ACTIONS))
        key = s * N_ACTIONS + a
        if not plus and model[key] is None:
            seen_actions[s].append(a)
        model[key] = (r, s2, term)
        last_tried[key] = t
        # ---- planning
        for _ in range(n_planning):
            ps = seen_states[rng.randrange(len(seen_states))]
            acts = seen_actions[ps]
            pa = acts[rng.randrange(len(acts))]
            pkey = ps * N_ACTIONS + pa
            pr, ps2, pterm = model[pkey]
            if plus:
                pr = pr + kappa * math.sqrt(t - last_tried[pkey])   # exploration bonus
            ptarget = pr if pterm else pr + gamma * max(Q[ps2])
            Q[ps][pa] += alpha * (ptarget - Q[ps][pa])
        s = maze.start if term else s2
    return cumulative


def eps_greedy_goal_rate(maze, gamma, eps, per=1000):
    """Reference rate: expected goals per `per` steps for an eps-greedy agent
    whose greedy policy is OPTIMAL (ties split evenly) in the current maze.
    Optimal Q by value iteration; expected episode length h(start) by solving
    h(s) = 1 + sum_a pi(a|s) h(next(s, a)), h(goal) = 0; then, by the renewal
    theorem, the long-run rate is per / h(start). Uses no random numbers."""
    free = maze.free_states()
    Q = np.zeros((maze.n_states, N_ACTIONS))
    for _ in range(2000):
        V = Q.max(axis=1)
        Qn = np.array([[r if term else r + gamma * V[s2] for (s2, r, term) in maze.table[s]]
                       for s in range(maze.n_states)])
        if np.max(np.abs(Qn - Q)) < 1e-12:
            break
        Q = Qn
    pos = {s: i for i, s in enumerate(free)}
    A = np.eye(len(free))
    for s in free:
        best = np.flatnonzero(Q[s] >= Q[s].max() - 1e-9)
        for a, (s2, _, term) in enumerate(maze.table[s]):
            p = eps / N_ACTIONS + ((1 - eps) / len(best) if a in best else 0.0)
            if not term:
                A[pos[s], pos[s2]] -= p
    h = np.linalg.solve(A, np.ones(len(free)))
    return per / h[pos[maze.start]], h[pos[maze.start]]


def experiment(name, make_maze, change_at, total_steps, n_runs, agents, hp, seed):
    maze, before, after = make_maze()
    maze.set_walls(before)
    print(f"--- {name}: change at t={change_at}, {total_steps} steps, {n_runs} runs")
    print("before:\n" + maze.render())
    print(f"  shortest path before: {maze.shortest_path_length()}", end="")
    rate_before, h_before = eps_greedy_goal_rate(maze, hp["gamma"], hp["eps"])
    maze.set_walls(after)
    print(f", after: {maze.shortest_path_length()}")
    print("after:\n" + maze.render())
    rate_after, h_after = eps_greedy_goal_rate(maze, hp["gamma"], hp["eps"])
    print(f"  reference (exact): an eps-greedy agent whose greedy policy is optimal needs {h_before:.1f} steps "
          f"per episode before the change and {h_after:.1f} after, i.e. {rate_before:.1f} and {rate_after:.1f} "
          f"goals per 1000 steps (eps = 0 would give {1000 / maze.shortest_path_length():.0f} after)")
    curves = {}
    for kind in agents:
        t0 = time.time()
        runs = []
        for run in range(n_runs):
            rng = random.Random(seed * 100_000 + run)
            runs.append(run_agent(kind, maze, before, after, change_at, total_steps, rng=rng, **hp))
        curves[kind] = np.array(runs)
        c = curves[kind]
        at_change = c[:, change_at - 1]
        gained_after = c[:, -1] - at_change
        print(f"  {kind:<18} reward by t={change_at}: {at_change.mean():6.1f} | "
              f"reward in t>{change_at}: {gained_after.mean():6.1f} ± {gained_after.std(ddof=1)/np.sqrt(n_runs):4.1f} | "
              f"total: {c[:, -1].mean():6.1f}  ({time.time() - t0:.1f}s)")
    return curves


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-planning", type=int, default=10, help="planning steps per real step")
    parser.add_argument("--alpha", type=float, default=0.5, help="step size")
    parser.add_argument("--no-figures", action="store_true", help="full-size run without writing figures")
    args = parser.parse_args()

    agents = ["Dyna-Q", "Dyna-Q+", "Q+ (action bonus)"]
    hp = dict(n_planning=args.n_planning, alpha=args.alpha, gamma=0.95, eps=0.1, kappa=1e-3)
    n_runs = 2 if args.quick else 30
    scale = 0.5 if args.quick else 1.0
    print(f"Changing mazes (S&B Ex. 8.2, 8.3) | seed={args.seed} runs={n_runs} {hp}")

    results = {}
    results["blocking"] = experiment("Blocking maze", maze_env.blocking_maze,
                                     change_at=int(1000 * scale), total_steps=int(3000 * scale),
                                     n_runs=n_runs, agents=agents, hp=hp, seed=args.seed)
    results["shortcut"] = experiment("Shortcut maze", maze_env.shortcut_maze,
                                     change_at=int(3000 * scale), total_steps=int(6000 * scale),
                                     n_runs=n_runs, agents=agents, hp=hp, seed=args.seed)

    # Blocking maze: how many runs are still stuck (no goal at all in the final
    # 1000 steps) -- the wide error bands come from these runs.
    c = results["blocking"]
    for kind in agents:
        last = c[kind][:, -1] - c[kind][:, -1 - int(1000 * scale)]
        print(f"  blocking maze, {kind:<18} runs with 0 goals in final {int(1000 * scale)} steps: "
              f"{int((last == 0).sum())}/{n_runs}")

    # How many runs discovered the shortcut? Measure the reward rate in the last
    # 1000 steps: the old path is 16 steps, the shortcut 10 (+ eps-greedy slack).
    c = results["shortcut"]
    for kind in agents:
        last = c[kind][:, -1] - c[kind][:, -1 - int(1000 * scale)]
        print(f"  shortcut maze, {kind:<18} goals in final {int(1000 * scale)} steps: "
              f"mean {last.mean():5.1f} (min {last.min():.0f}, max {last.max():.0f})")

    if args.quick or args.no_figures:
        print("no figures written")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"Dyna-Q": "tab:blue", "Dyna-Q+": "tab:red", "Q+ (action bonus)": "tab:green"}
    for ax, (key, change, title) in zip(axes, [("blocking", 1000, "Blocking maze (S&B Ex. 8.2)"),
                                               ("shortcut", 3000, "Shortcut maze (S&B Ex. 8.3)")]):
        for kind in agents:
            curve = results[key][kind]
            mean = curve.mean(axis=0)
            se = curve.std(axis=0, ddof=1) / np.sqrt(n_runs)
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, color=colors[kind], label=kind)
            ax.fill_between(x, mean - se, mean + se, color=colors[kind], alpha=0.2)
        ax.axvline(change, color="gray", ls="--", lw=1)
        ax.text(change, ax.get_ylim()[1] * 0.95, " maze changes", color="gray", va="top")
        ax.set_xlabel("time step")
        ax.set_ylabel("cumulative reward")
        ax.set_title(f"{title}: mean of {n_runs} runs ±1 s.e.", fontsize=10)
        ax.legend(loc="lower right")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "changing_mazes.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
