"""Reference solutions for the coding exercises of Chapter 05 (Exercises 5.12 and 5.13).

Exercise 5.12  Does decaying epsilon make SARSA find the optimal cliff-walking path?
               Three schedules (all with the SARSA box of Section 7.2):
                 (i)   constant epsilon = 0.1, constant alpha = 0.5;
                 (ii)  per-EPISODE decay epsilon_k = 0.1 * 100 / (100 + k), alpha = 0.5.
                       NOT covered by Theorem 5.3: alpha is constant (violates Robbins-Monro), and
                       decaying epsilon per episode is not guaranteed to be GLIE, because cells
                       reached only through exploration get sum(epsilon) < infinity over their
                       own visits (Section 7.3, Exercise 5.12);
                 (iii) per-STATE GLIE epsilon_t(s) = 1 / sqrt(n_t(s)) (n_t(s) = visits to s so
                       far, Exercise 5.11(d)) with Robbins-Monro steps alpha = 1 / n(s, a)^0.6.
               We also count visits to the cliff-edge cells (row 2, columns 2-9) and how often
               'right' is tried in cell (row 2, column 1), to see how much exploration they get.
Exercise 5.13  Step size alpha = 1 in a deterministic environment (Taxi-v4): which of
               Q-learning, Expected SARSA and SARSA tolerate it?

Run:  python code/ch05_temporal_difference/exercise_solutions.py [--quick]
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time

sys.dont_write_bytecode = True        # importing the sibling scripts must not leave __pycache__/

import gymnasium as gym                # noqa: E402
import numpy as np                     # noqa: E402

from cliff_walking import (MAX_STEPS, N_ACTIONS, N_STATES, NEXT, START,  # noqa: E402
                           describe_loop, epsilon_greedy, greedy_path, to_index)
from taxi_q_learning import epsilon_greedy as epsilon_greedy_np  # noqa: E402
from taxi_q_learning import evaluate, value_iteration  # noqa: E402

EDGE_CELLS = [to_index((2, c)) for c in range(2, 10)]   # cliff-edge cells (row 2, columns 2..9)
CELL_21, RIGHT = to_index((2, 1)), 1


# ------------------------------------------------------------------------------------------
# Exercise 5.12: SARSA on cliff walking with decaying epsilon
# ------------------------------------------------------------------------------------------
def sarsa_scheduled(eps_mode, alpha_mode, seed, n_episodes, gamma=1.0, checkpoints=()):
    """SARSA (Section 7.2 box) with an epsilon schedule and a step-size schedule.

    eps_mode   'const'   : epsilon = 0.1
               'episode' : epsilon_k = 0.1 * 100 / (100 + k) in episode k (k = 1, 2, ...)
               'state'   : epsilon_t(s) = 1 / sqrt(n_t(s)), n_t(s) = visits to s including this one
    alpha_mode 'const'   : alpha = 0.5
               'rm'      : alpha = 1 / n(s, a)^0.6 on the n-th update of (s, a)  (Robbins-Monro)
    Returns Q, the per-episode returns, and {checkpoint episode: (visits to the edge cells,
    times 'right' was taken in cell (2, 1), visits to each edge cell (2, 2) ... (2, 9),
    sum of epsilon over all action choices = expected number of random actions)}.
    """
    rng = random.Random(seed)
    Q = [[0.0] * N_ACTIONS for _ in range(N_STATES)]
    n_s = [0] * N_STATES
    n_sa = [[0] * N_ACTIONS for _ in range(N_STATES)]
    returns, snaps = np.zeros(n_episodes), {}
    edge_visits = right_21 = 0
    edge_by_cell = [0] * len(EDGE_CELLS)
    sum_eps = [0.0]                        # expected number of random (exploratory) actions so far

    def choose(s, k):                      # epsilon-greedy with the chosen schedule
        n_s[s] += 1
        if eps_mode == "const":
            eps = 0.1
        elif eps_mode == "episode":
            eps = 0.1 * 100 / (100 + k)
        else:
            eps = 1.0 / math.sqrt(n_s[s])
        sum_eps[0] += eps
        return epsilon_greedy(Q[s], eps, rng)

    def step_size(s, a):
        n_sa[s][a] += 1
        return 0.5 if alpha_mode == "const" else n_sa[s][a] ** -0.6

    for k in range(1, n_episodes + 1):
        s = to_index(START)
        a = choose(s, k)
        G = 0.0
        for _ in range(MAX_STEPS):
            if s in EDGE_CELLS:
                edge_visits += 1
                edge_by_cell[EDGE_CELLS.index(s)] += 1
            right_21 += (s == CELL_21 and a == RIGHT)
            s2, r, terminated = NEXT[s][a]
            G += r
            if terminated:
                Q[s][a] += step_size(s, a) * (r - Q[s][a])
                break
            a2 = choose(s2, k)                                  # the action we WILL take
            Q[s][a] += step_size(s, a) * (r + gamma * Q[s2][a2] - Q[s][a])
            s, a = s2, a2
        returns[k - 1] = G
        if k in checkpoints:
            snaps[k] = (edge_visits, right_21, tuple(edge_by_cell), sum_eps[0])
    return Q, returns, snaps


def greedy_summary(Qs):
    rets = [greedy_path(Q)[1] if greedy_path(Q)[2] else "stuck" for Q in Qs]
    return {g: rets.count(g) for g in sorted(set(rets), key=str)}


def exercise_glie(quick: bool):
    n_seeds, n_episodes = (5, 2000) if quick else (20, 5000)
    cp = (1000, n_episodes)
    configs = {
        "(i)   const eps=0.1, alpha=0.5": ("const", "const", 1.0),
        "(ii)  per-episode eps_k, alpha=0.5": ("episode", "const", 1.0),
        "(iii) per-state 1/sqrt(n(s)), 1/n^0.6": ("state", "rm", 1.0),
        "(iii) same, gamma=0.99": ("state", "rm", 0.99),
    }
    print(f"Exercise 5.12 -- SARSA on cliff walking, {n_seeds} seeds x {n_episodes} episodes")
    print("  (ii) is NOT GLIE per state and has constant alpha; (iii) is GLIE + Robbins-Monro")
    print(f"  schedule                               | online, last 100 | greedy returns           "
          f"| edge-cell visits ep {cp[0]}/{cp[1]} (median) | 'right' tried at (2,1), ep {cp[0]}/{cp[1]} (median)")
    for name, (em, am, g) in configs.items():
        res = [sarsa_scheduled(em, am, s, n_episodes, gamma=g, checkpoints=cp) for s in range(n_seeds)]
        online = np.mean([r[-100:].mean() for _, r, _ in res])
        ev = [np.median([sn[c][0] for *_, sn in res]) for c in cp]
        rt = [np.median([sn[c][1] for *_, sn in res]) for c in cp]
        print(f"  {name:38s} | {online:8.2f}         | {str(greedy_summary([Q for Q, _, _ in res])):24s} "
              f"| {ev[0]:6.0f} / {ev[1]:6.0f}                     | {rt[0]:5.0f} / {rt[1]:5.0f}")
        for seed, (Q, _, _) in enumerate(res):
            path, _, ok = greedy_path(Q)
            if not ok:
                print(f"      seed {seed}: greedy read-out stuck -- {describe_loop(Q, path)}")
    if quick:
        return
    n_long, n_long_seeds = 50_000, 10
    print(f"\n  Long runs: {n_long_seeds} seeds x {n_long} episodes "
          f"(q* for 'right' in cell (2,1) is -11; greedy returns: -13 optimal, -15 row 1, -17 row 0)")
    for name, (em, am, g) in list(configs.items())[1:3]:
        res = [sarsa_scheduled(em, am, s, n_long, gamma=g, checkpoints=(5000, n_long))
               for s in range(n_long_seeds)]
        q21 = [Q[CELL_21][RIGHT] for Q, _, _ in res]
        n_random = np.median([sn[n_long][3] - sn[5000][3] for *_, sn in res])
        print(f"  {name}: greedy returns {greedy_summary([Q for Q, _, _ in res])}; "
              f"Q((2,1), right): median {np.median(q21):.1f}, seed 0 {q21[0]:.1f}; "
              f"expected random actions in episodes 5,001-50,000 (all cells): median {n_random:.0f}")
        print("      per seed, edge-cell visits at episode 5,000 -> 50,000: "
              + ", ".join(f"{sn[5000][0]}->{sn[n_long][0]}" for *_, sn in res))
        print("      per seed, 'right' tried at (2,1) at episode 5,000 -> 50,000: "
              + ", ".join(f"{sn[5000][1]}->{sn[n_long][1]}" for *_, sn in res))
        for seed, (*_, sn) in enumerate(res):
            extra = [b - a for a, b in zip(sn[5000][2], sn[n_long][2])]
            if sum(extra) > 500:      # far more than the exploratory actions left (see chapter)
                cells = ", ".join(f"(2,{c + 2}): +{e}" for c, e in enumerate(extra) if e > 0)
                print(f"      seed {seed}: extra edge visits after episode 5,000 by cell: {cells}")


# ------------------------------------------------------------------------------------------
# Exercise 5.13: alpha = 1 on deterministic Taxi-v4
# ------------------------------------------------------------------------------------------
def train_taxi(method, alpha, seed, n_episodes, gamma=0.99, eps=0.1):
    env = gym.make("Taxi-v4")
    rng = np.random.default_rng(seed)
    n_a = env.action_space.n
    Q = np.zeros((env.observation_space.n, n_a))
    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed if ep == 0 else None)
        a = epsilon_greedy_np(Q[s], eps, rng)
        while True:
            s2, r, terminated, truncated, _ = env.step(a)
            if method == "SARSA":
                # SARSA box (Section 7.2): A' is chosen BEFORE the update, because the target
                # needs it, and A' is then the action actually executed.
                a2 = epsilon_greedy_np(Q[s2], eps, rng)
                target = r if terminated else r + gamma * Q[s2, a2]
            elif method == "Q-learning":
                target = r if terminated else r + gamma * Q[s2].max()
            else:                                          # Expected SARSA, eps-greedy target
                greedy = Q[s2] == Q[s2].max()
                pi = eps / n_a + (1 - eps) * greedy / greedy.sum()
                target = r if terminated else r + gamma * pi @ Q[s2]
            Q[s, a] += alpha * (target - Q[s, a])
            if terminated or truncated:
                break
            if method != "SARSA":
                # Q-learning / Expected SARSA boxes (Sections 8.1, 9): choose the next action at
                # the top of the loop, i.e. AFTER the update.  The order matters on
                # self-transitions (s2 == s: a wall bump or an illegal pickup/drop-off in Taxi).
                a2 = epsilon_greedy_np(Q[s2], eps, rng)
            s, a = s2, a2
    return Q


def exercise_taxi_alpha(quick: bool):
    n_seeds, n_episodes = (1, 300) if quick else (5, 2000)
    env = gym.make("Taxi-v4")
    P = env.unwrapped.P
    starts = np.flatnonzero(env.unwrapped.initial_state_distrib > 0)
    G_opt, _ = evaluate(P, value_iteration(P, 500, 6, 0.99), starts)
    print(f"\nExercise 5.13 -- Taxi-v4, eps=0.1, gamma=0.99, {n_seeds} seeds x {n_episodes} episodes "
          f"(optimal mean greedy return {G_opt.mean():.2f})")
    for alpha in (1.0, 0.1):
        for m in ("Q-learning", "Expected SARSA", "SARSA"):
            out = []
            for seed in range(n_seeds):
                G, _ = evaluate(P, train_taxi(m, alpha, seed, n_episodes), starts)
                out.append((G.mean(), np.isclose(G, G_opt).mean()))
            print(f"  alpha={alpha:3.1f} {m:15s} | greedy mean return {np.mean([o[0] for o in out]):8.2f} "
                  f"| start states with optimal return {np.mean([o[1] for o in out]):6.1%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test (no figures are ever written)")
    parser.add_argument("--seed", type=int, default=0, help="unused; seeds are 0..n_seeds-1")
    args = parser.parse_args()
    print(f"[exercise_solutions] quick={args.quick}  seeds 0..n-1 for every experiment")
    t0 = time.time()
    exercise_glie(args.quick)
    exercise_taxi_alpha(args.quick)
    print(f"\nRuntime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
