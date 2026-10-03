"""Cliff walking: SARSA vs Q-learning vs Expected SARSA (online reward and learned paths).

Chapter 05, Section 10, following Sutton & Barto (2018), Example 6.6.

Grid (4 rows x 12 columns), S = start, G = goal, C = cliff:

    row 0  . . . . . . . . . . . .
    row 1  . . . . . . . . . . . .
    row 2  . . . . . . . . . . . .
    row 3  S C C C C C C C C C C G

* actions: 0 = up, 1 = right, 2 = down, 3 = left (same numbering as gymnasium's CliffWalking-v1);
  a move off the grid leaves the agent where it is;
* reward -1 per step; stepping into the cliff gives -100 and teleports the agent back to S
  (the episode does NOT end);
* the episode terminates on reaching G.  gamma = 1.
* as a safety net an episode is abandoned after 10,000 steps (MAX_STEPS); with the settings
  of this script that never happens, but cliff_alpha_sweep.py reports when it does.

The environment is implemented here (rather than via gymnasium) so that every line of the
dynamics is visible; it is the same MDP as gymnasium's CliffWalking-v1 with is_slippery=False.

All three agents act epsilon-greedily with epsilon = 0.1 (held constant) and alpha = 0.5.

The script also
  * reads out the greedy policy argmax_a Q(s, a) after training.  This deterministic read-out
    breaks ties by LOWEST action index (unlike the agents, which break ties at random while
    learning); for every run whose read-out never reaches G it prints the loop it gets stuck in
    (a wall bump = a one-cell loop, or a two-cell cycle) with the Q-values involved;
  * solves EXACTLY the fixed point that SARSA and Expected SARSA share when epsilon is held
    fixed, Q(s,a) = r + sum_a' pi_eps(a'|s') Q(s',a') with pi_eps epsilon-greedy w.r.t. Q itself
    (epsilon-soft value iteration, Section 10.2), i.e. the action values of the best
    epsilon-greedy policy, and the exact epsilon-greedy returns of related policies;
  * re-trains SARSA and Expected SARSA with smaller step sizes and longer training to show
    that SARSA's preference for the top row at alpha = 0.5 is a step-size effect.

Outputs (full mode):
    figures/cliff_rewards.png   -- sum of rewards per episode (online, while exploring)
    figures/cliff_paths.png     -- greedy path implied by the final Q of each method

Run:  python code/ch05_temporal_difference/cliff_walking.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np

ROWS, COLS = 4, 12
START, GOAL = (3, 0), (3, 11)
N_STATES, N_ACTIONS = ROWS * COLS, 4
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]           # up, right, down, left
MAX_STEPS = 10_000
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def to_index(cell):
    return cell[0] * COLS + cell[1]


def is_cliff(cell):
    return cell[0] == 3 and 1 <= cell[1] <= 10


# Precompute the deterministic transition table: NEXT[s][a] = (s', r, terminated)
NEXT = [[None] * N_ACTIONS for _ in range(N_STATES)]
for r in range(ROWS):
    for c in range(COLS):
        for a, (dr, dc) in enumerate(MOVES):
            nr, nc = min(max(r + dr, 0), ROWS - 1), min(max(c + dc, 0), COLS - 1)
            if is_cliff((nr, nc)):
                NEXT[to_index((r, c))][a] = (to_index(START), -100.0, False)
            else:
                NEXT[to_index((r, c))][a] = (to_index((nr, nc)), -1.0, (nr, nc) == GOAL)


# ---------------------------------------------------------------------------------------
# Policies.  Q is a list of lists (fast scalar access in pure Python).
# ---------------------------------------------------------------------------------------
def greedy_actions(q_s):
    m = max(q_s)
    return [a for a, v in enumerate(q_s) if v == m]


def epsilon_greedy(q_s, eps, rng: random.Random):
    """Random tie-breaking among greedy actions matters early on when many Q's are equal."""
    if rng.random() < eps:
        return rng.randrange(N_ACTIONS)
    return rng.choice(greedy_actions(q_s))


def expected_value_eps_greedy(q_s, eps):
    """sum_a pi(a|s) Q(s,a) for the epsilon-greedy policy that breaks ties uniformly."""
    greedy = greedy_actions(q_s)
    p_greedy_extra = (1.0 - eps) / len(greedy)
    exp_v = eps / N_ACTIONS * sum(q_s)
    for a in greedy:
        exp_v += p_greedy_extra * q_s[a]
    return exp_v


# ---------------------------------------------------------------------------------------
# One episode of each control algorithm (the pseudocode boxes of Sections 7, 8, 9)
# ---------------------------------------------------------------------------------------
def sarsa_episode(Q, alpha, eps, gamma, rng, max_steps=MAX_STEPS):
    s = to_index(START)
    a = epsilon_greedy(Q[s], eps, rng)
    total = 0.0
    for _ in range(max_steps):
        s2, r, terminated = NEXT[s][a]
        total += r
        if terminated:
            Q[s][a] += alpha * (r - Q[s][a])          # Q(terminal, .) = 0
            return total, True
        a2 = epsilon_greedy(Q[s2], eps, rng)          # the action we WILL take
        Q[s][a] += alpha * (r + gamma * Q[s2][a2] - Q[s][a])
        s, a = s2, a2
    return total, False                               # hit the safety cap


def q_learning_episode(Q, alpha, eps, gamma, rng, max_steps=MAX_STEPS):
    s = to_index(START)
    total = 0.0
    for _ in range(max_steps):
        a = epsilon_greedy(Q[s], eps, rng)
        s2, r, terminated = NEXT[s][a]
        total += r
        target = r if terminated else r + gamma * max(Q[s2])   # greedy target policy
        Q[s][a] += alpha * (target - Q[s][a])
        if terminated:
            return total, True
        s = s2
    return total, False                               # hit the safety cap


def expected_sarsa_episode(Q, alpha, eps, gamma, rng, max_steps=MAX_STEPS):
    s = to_index(START)
    total = 0.0
    for _ in range(max_steps):
        a = epsilon_greedy(Q[s], eps, rng)
        s2, r, terminated = NEXT[s][a]
        total += r
        target = r if terminated else r + gamma * expected_value_eps_greedy(Q[s2], eps)
        Q[s][a] += alpha * (target - Q[s][a])
        if terminated:
            return total, True
        s = s2
    return total, False                               # hit the safety cap


ALGORITHMS = {
    "SARSA": sarsa_episode,
    "Q-learning": q_learning_episode,
    "Expected SARSA": expected_sarsa_episode,
}


def train(method, n_episodes, alpha, eps, seed, gamma=1.0):
    """Return the final Q, the per-episode sums of rewards, and how many episodes hit the cap."""
    rng = random.Random(seed)
    Q = [[0.0] * N_ACTIONS for _ in range(N_STATES)]
    run_episode = ALGORITHMS[method]
    rewards, n_capped = np.zeros(n_episodes), 0
    for ep in range(n_episodes):
        rewards[ep], reached = run_episode(Q, alpha, eps, gamma, rng)
        n_capped += not reached
    return Q, rewards, n_capped


def greedy_path(Q, max_len=100):
    """Follow argmax Q from S; return visited cells, the return, and whether G was reached.

    Ties are broken by the LOWEST action index (q.index(max(q))), so the read-out is
    deterministic.  Exact ties are rare in a trained table, except in never-visited cells
    (all four values still 0)."""
    s, path, ret = to_index(START), [START], 0.0
    for _ in range(max_len):
        q = Q[s]
        a = q.index(max(q))
        s, r, terminated = NEXT[s][a]
        ret += r
        path.append((s // COLS, s % COLS))
        if terminated:
            return path, ret, True
    return path, ret, False


def classify_path(path):
    """Which row does the greedy path travel along in the middle of the grid (column 5)?"""
    rows = [r for (r, c) in path if c == 5]
    return min(rows) if rows else None


def describe_loop(Q, path):
    """For a greedy read-out that never reaches G: find the cycle it is stuck in.

    Returns a short text: a one-cell loop is a wall bump (the greedy action leaves the agent in
    place); a two-cell loop is an oscillation between neighbours.  Q-values are printed in the
    order [up, right, down, left]."""
    first_seen = {}
    for i, cell in enumerate(path):
        if cell in first_seen:
            cycle = path[first_seen[cell]:i]
            break
        first_seen[cell] = i
    else:
        return "no cycle found"
    kind = "wall bump (1-cell loop)" if len(cycle) == 1 else f"{len(cycle)}-cell cycle"
    names = "URDL"
    parts = []
    for cell in cycle:
        q = Q[to_index(cell)]
        parts.append(f"{cell} greedy {names[q.index(max(q))]} Q={[round(v, 1) for v in q]}")
    return kind + ": " + "; ".join(parts)


# ---------------------------------------------------------------------------------------
# Exact epsilon-soft dynamic programming (uses the model; for analysis only, Section 10.2)
# ---------------------------------------------------------------------------------------
def _backup(V):
    """q(s, a) = r + V(s') for every pair (gamma = 1, V(terminal) = 0)."""
    return np.array([[NEXT[s][a][1] + (0.0 if NEXT[s][a][2] else V[NEXT[s][a][0]])
                      for a in range(N_ACTIONS)] for s in range(N_STATES)])


def eps_soft_value_iteration(eps, forced=None, tol=1e-10):
    """Solve Q(s,a) = r + sum_a' pi(a'|s') Q(s',a'), pi epsilon-greedy w.r.t. Q itself.

    This is the common fixed point of SARSA and Expected SARSA with a fixed epsilon: the
    action values of the best epsilon-greedy ("epsilon-soft optimal") policy.  Every
    epsilon-greedy policy reaches G with probability 1 here, so the iteration converges
    even with gamma = 1.  `forced` = {state: action} pins the greedy action in some cells,
    which gives the best epsilon-greedy policy whose greedy route is a prescribed one.
    Returns (Q, v) with v(s) = sum_a pi(a|s) Q(s, a)."""
    Q = np.zeros((N_STATES, N_ACTIONS))
    rows = np.arange(N_STATES)
    while True:
        greedy = Q.argmax(axis=1)
        for s_, a_ in (forced or {}).items():
            greedy[s_] = a_
        v = (1 - eps) * Q[rows, greedy] + eps / N_ACTIONS * Q.sum(axis=1)
        Q_new = _backup(v)
        if np.max(np.abs(Q_new - Q)) < tol:
            return Q_new, v
        Q = Q_new


def route_constraint(row):
    """Greedy actions that pin the greedy route to `row`: up column 0, right along `row`,
    down the last column."""
    forced = {to_index((r, 0)): 0 for r in range(row + 1, ROWS)}
    forced.update({to_index((row, c)): 1 for c in range(COLS - 1)})
    forced.update({to_index((r, COLS - 1)): 2 for r in range(row, ROWS - 1)})
    return forced


def eps_greedy_value(Q_fixed, eps):
    """Exact value from S of the epsilon-greedy policy w.r.t. a FIXED table (ties shared).

    Policy evaluation by solving the linear system v = r_pi + P_pi v (Chapter 03); P_pi
    leaves out transitions into the terminal goal, so I - P_pi is invertible because every
    epsilon-greedy policy reaches G with probability 1."""
    Q_fixed = np.asarray(Q_fixed, dtype=float)
    is_max = np.isclose(Q_fixed, Q_fixed.max(axis=1, keepdims=True), rtol=0, atol=1e-9)
    pi = eps / N_ACTIONS + (1 - eps) * is_max / is_max.sum(axis=1, keepdims=True)
    P, r = np.zeros((N_STATES, N_STATES)), np.zeros(N_STATES)
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            s2, rew, terminated = NEXT[s][a]
            r[s] += pi[s, a] * rew
            if not terminated:
                P[s, s2] += pi[s, a]
    v = np.linalg.solve(np.eye(N_STATES) - P, r)
    return float(v[to_index(START)])


def optimal_q(tol=1e-10):
    """q* by ordinary value iteration (gamma = 1)."""
    Q = np.zeros((N_STATES, N_ACTIONS))
    while True:
        Q_new = _backup(Q.max(axis=1))
        if np.max(np.abs(Q_new - Q)) < tol:
            return Q_new
        Q = Q_new


def smooth(x, k):
    return np.convolve(x, np.ones(k) / k, mode="valid")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    n_runs = 5 if args.quick else 100
    n_episodes = 200 if args.quick else 500
    alpha, eps, gamma = 0.5, 0.1, 1.0
    print(f"[cliff_walking] seed={args.seed} runs={n_runs} episodes={n_episodes} "
          f"alpha={alpha} epsilon={eps} gamma={gamma}")
    t0 = time.time()

    results, final_Q = {}, {}
    for method in ALGORITHMS:
        all_rewards, paths, capped, Qs = [], [], 0, []
        for run in range(n_runs):
            Q, rewards, n_capped = train(method, n_episodes, alpha, eps, seed=args.seed + run)
            all_rewards.append(rewards)
            paths.append(greedy_path(Q))
            Qs.append(Q)
            capped += n_capped
        results[method] = (np.array(all_rewards), paths)
        final_Q[method] = Qs
        if capped:
            print(f"  note: {capped} {method} episodes hit the {MAX_STEPS}-step safety cap")

    print("\nOnline performance (sum of rewards per episode while following eps-greedy, eps=0.1):")
    print("  method          | mean, episodes 1-100 | mean, last 100 episodes")
    for method, (R, _) in results.items():
        print(f"  {method:15s} | {R[:, :100].mean():8.1f}             | {R[:, -100:].mean():8.1f}")

    print("\nGreedy policy after training (follow argmax Q from S, no exploration):")
    print("  method          | reaches G | mean greedy return | row used at column 5 (count over runs)")
    for method, (_, paths) in results.items():
        reached = sum(p[2] for p in paths)
        rets = [p[1] for p in paths if p[2]]
        rows = [classify_path(p[0]) for p in paths if p[2]]
        hist = {row: rows.count(row) for row in sorted(set(rows))}
        print(f"  {method:15s} | {reached:3d}/{n_runs:<3d}   | {np.mean(rets):8.1f}           | {hist}")
    print("  (row 2 = edge of the cliff, the optimal 13-step path; row 0 = farthest from it)")

    # ---- Greedy read-outs that never reach G: which loop are they stuck in? ----------------
    for method, (_, paths) in results.items():
        stuck = [run for run, p in enumerate(paths) if not p[2]]
        if not stuck:
            continue
        kinds = [describe_loop(final_Q[method][run], paths[run][0]) for run in stuck]
        n_wall = sum(k.startswith("wall bump") for k in kinds)
        print(f"\n{method}: greedy read-out stuck in {len(stuck)} runs "
              f"({n_wall} wall bumps, {len(stuck) - n_wall} multi-cell cycles; ties -> lowest index):")
        for run, k in zip(stuck, kinds):
            print(f"  run {args.seed + run:3d}: {k}")

    # ---- The exact fixed point shared by SARSA and Expected SARSA (Section 10.2) -----------
    Q_soft, v_soft = eps_soft_value_iteration(eps)
    soft_path, _, _ = greedy_path([list(q) for q in Q_soft])
    print(f"\nExact epsilon-soft value iteration (epsilon={eps}): the common fixed point of SARSA and "
          f"Expected SARSA")
    print(f"  best epsilon-greedy policy: greedy route along row {classify_path(soft_path)} "
          f"({len(soft_path) - 1} steps), expected return from S under eps-greedy = "
          f"{v_soft[to_index(START)]:.2f}")
    for row in (0, 1, 2):
        _, v_row = eps_soft_value_iteration(eps, forced=route_constraint(row))
        print(f"  best epsilon-greedy policy whose greedy route is row {row}: {v_row[to_index(START)]:7.2f}")
    print(f"  epsilon-greedy w.r.t. q* (what Q-learning's behaviour converges to, ties shared): "
          f"{eps_greedy_value(optimal_q(), eps):7.2f}")
    for method in ALGORITHMS:
        vals = [eps_greedy_value(Q, eps) for Q in final_Q[method]]
        print(f"  {method:15s}: exact eps-greedy value of the FINAL Q tables (frozen): "
              f"median {np.median(vals):7.2f}  (online return, last 100 ep.: "
              f"{results[method][0][:, -100:].mean():.2f})")

    # ---- Is SARSA's top-row route a step-size effect? ---------------------------------------
    settings = [(0.5, 500), (0.1, 5000), (0.05, 10_000)]
    n_seeds_alpha = 2 if args.quick else 20
    if args.quick:
        settings = [(0.5, 200), (0.1, 500)]
    print(f"\nGreedy route vs step size ({n_seeds_alpha} seeds each, epsilon={eps}; "
          f"route = row used at column 5):")
    print("  method          alpha  episodes | routes                        | stuck | online return, last 100 ep.")
    for method in ("SARSA", "Expected SARSA"):
        for a_, n_ep in settings:
            routes, online = [], []
            for run in range(n_seeds_alpha):
                Q, rewards, _ = train(method, n_ep, a_, eps, seed=args.seed + run)
                path, _, ok = greedy_path(Q)
                routes.append(f"row {classify_path(path)}" if ok else "stuck")
                online.append(rewards[-100:].mean())
            hist = {k: routes.count(k) for k in sorted(set(routes))}
            print(f"  {method:15s} {a_:5.2f}  {n_ep:8d} | {str(hist):29s} | {routes.count('stuck'):5d} "
                  f"| {np.mean(online):7.2f}")

    if not args.quick:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIG_DIR, exist_ok=True)
        colors = {"SARSA": "tab:blue", "Q-learning": "tab:red", "Expected SARSA": "tab:green"}

        fig, ax = plt.subplots(figsize=(7, 4.4))
        k = 10
        for method, (R, _) in results.items():
            curve = smooth(R.mean(axis=0), k)
            ax.plot(np.arange(k, n_episodes + 1), curve, color=colors[method], label=method)
        ax.axhline(-13, color="gray", ls=":", lw=1)
        ax.text(n_episodes * 0.6, -11, "optimal greedy return = -13", color="gray", fontsize=8)
        ax.set_ylim(-110, 0)
        ax.set_xlabel("episode")
        ax.set_ylabel(f"sum of rewards during episode\n(mean of {n_runs} runs, {k}-episode moving avg)")
        ax.set_title(f"Cliff walking, ε-greedy with ε={eps}, α={alpha}")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "cliff_rewards.png"), dpi=110)
        plt.close(fig)

        # Paths of run 0 for each method, drawn on the grid
        fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.2))
        for ax, (method, (_, paths)) in zip(axes, results.items()):
            grid = np.zeros((ROWS, COLS))
            for c in range(1, 11):
                grid[3, c] = 1
            ax.imshow(grid, cmap="Greys", vmin=0, vmax=2.5)
            path = paths[0][0]
            ys = [p[0] for p in path]
            xs = [p[1] for p in path]
            ax.plot(xs, ys, "-o", color=colors[method], lw=2.5, ms=4)
            ax.text(0, 3, "S", ha="center", va="center", fontsize=12, weight="bold")
            ax.text(11, 3, "G", ha="center", va="center", fontsize=12, weight="bold")
            ax.text(5.5, 3, "the cliff  (-100)", ha="center", va="center", fontsize=9, color="white")
            ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
            ax.grid(which="minor", color="k", lw=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{method}: greedy path after {n_episodes} episodes (run 0), "
                         f"{len(path) - 1} steps", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "cliff_paths.png"), dpi=110)
        plt.close(fig)
        print(f"\nFigures written to {FIG_DIR}")
    print(f"\nRuntime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
