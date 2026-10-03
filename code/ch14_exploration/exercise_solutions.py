"""Numerical checks and reference code for the Chapter 14 exercises.

  Ex. 1   (and Sec. 1.3) eps-greedy Q on DeepSea: exact per-episode success probability of
          the learned eps-greedy behaviour policy vs the (eps/2)^N "fully learned" figure
  Ex. 2   combination lock: expected hitting time 2^(N+1) - 2 (linear solve + simulation)
  Ex. 3   pseudo-counts: empirical and Laplace density models give n(x) and n(x) + 1
  Ex. 5   pigeonhole: sum_t 1/sqrt(n_t(s,a)) <= 2 sqrt(SA T) - checked on random sequences
  Ex. 8   RND with one-hot inputs and a linear predictor: error decays like (1-eta)^(2n)
  Ex. 10  R-MAX "known" threshold from Hoeffding
  Ex. 11  Boltzmann (softmax) exploration on DeepSea fails like eps-greedy
  Ex. 12  SimHash counts: how many of the 328 four-rooms cells collide for k bits
  Ex. 13  Go-Explore (return by replay, then explore) on DeepSea

Run:  python code/ch14_exploration/exercise_solutions.py [--quick]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import deep_sea as ds
from explore_lib import FOUR_ROOMS, GridWorld


def ex1_eps_greedy_success(Ns, seeds, episodes, eps=0.1):
    """Train eps-greedy Q on DeepSea(N), then compute EXACTLY the probability that the
    eps-greedy behaviour policy reaches the treasure in one episode.

    The treasure needs "right" on every diagonal cell (h, h).  On a cell where Q prefers
    left the agent goes right with probability eps/2; on a tie (Q never updated for one
    of the actions) with probability 1/2 (random tie-breaking); where Q prefers right,
    with 1 - eps/2.  So P = (eps/2)^d (1/2)^u (1 - eps/2)^r, with d + u + r = N.
    """
    print(f"Ex. 1  eps-greedy Q (eps={eps}) on DeepSea after {episodes} training episodes: "
          f"exact per-episode success probability of the behaviour policy")
    for N in Ns:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            env = ds.DeepSea(N, np.random.default_rng(10_000 + seed))
            agent = ds.EpsGreedyQ(N, eps=eps)
            found = sum(ds.run_episode(env, agent, rng) for _ in range(episodes))
            d = u = r = 0
            prob = 1.0
            for h in range(N):
                q = agent.Q[h * N + h]
                right = env.right_action[h, h]
                if q[right] > q[1 - right]:
                    r += 1
                    prob *= 1 - eps / 2
                elif q[right] < q[1 - right]:
                    d += 1
                    prob *= eps / 2
                else:
                    u += 1
                    prob *= 0.5
            print(f"  N={N:2d} seed {seed}: diagonal cells learned left d={d}, "
                  f"ties u={u}, learned right r={r}; P(success) = {prob:.2e} "
                  f"(treasure found {found} times in training) vs (eps/2)^N = "
                  f"{(eps / 2) ** N:.1e}")


def ex2_combination_lock(Ns, n_sim, rng):
    """Uniform random policy on an N-state 'combination lock': right advances, wrong resets."""
    print("Ex. 2  combination lock: expected steps from state 0 to the end")
    for N in Ns:
        # E_i = 1 + 0.5 E_{i+1} + 0.5 E_0 for i < N, E_N = 0: solve the linear system.
        M = np.zeros((N, N))
        b = np.ones(N)
        for i in range(N):
            M[i, i] += 1.0
            M[i, 0] -= 0.5
            if i + 1 < N:
                M[i, i + 1] -= 0.5
        E = np.linalg.solve(M, b)
        # Simulate n_sim independent random walkers in parallel.
        s = np.zeros(n_sim, dtype=np.int64)
        steps = np.zeros(n_sim, dtype=np.int64)
        active = np.ones(n_sim, dtype=bool)
        while active.any():
            idx = np.flatnonzero(active)
            steps[idx] += 1
            advance = rng.random(len(idx)) < 0.5
            s[idx] = np.where(advance, s[idx] + 1, 0)
            active[idx[s[idx] >= N]] = False
        se = np.std(steps) / np.sqrt(n_sim)
        print(f"  N={N:2d}: formula 2^(N+1)-2 = {2 ** (N + 1) - 2:6d}, linear solve "
              f"{E[0]:8.1f}, simulation {np.mean(steps):8.1f} +- {se:.1f} ({n_sim} runs)")


def ex3_pseudocounts():
    print("\nEx. 3  pseudo-count N_hat = rho (1 - rho') / (rho' - rho)")
    n, M = 10, 4
    counts = np.array([2, 5, 0, 3])
    for name, rho_fn in [("empirical", lambda c, n: c / n),
                         ("Laplace (add-one)", lambda c, n: (c + 1) / (n + M))]:
        out = []
        for x in range(M):
            rho = rho_fn(counts[x], n)
            rho2 = rho_fn(counts[x] + 1, n + 1)            # after observing x once more
            out.append(rho * (1 - rho2) / (rho2 - rho))
        print(f"  {name:18s} counts {counts.tolist()} -> pseudo-counts "
              f"{np.round(out, 6).tolist()}")


def ex5_pigeonhole(rng):
    print("\nEx. 5  sum over steps of 1/sqrt(n_t) vs the bound 2 sqrt(SA T)")
    for SA, T in [(10, 1000), (100, 10_000), (100, 1000)]:
        worst = 0.0
        for policy in ["uniform", "skewed"]:
            p = np.ones(SA) / SA if policy == "uniform" else rng.dirichlet(0.2 * np.ones(SA))
            pairs = rng.choice(SA, size=T, p=p)
            n = np.zeros(SA)
            total = 0.0
            for i in pairs:
                n[i] += 1
                total += 1 / np.sqrt(n[i])
            worst = max(worst, total)
            print(f"  SA={SA:4d} T={T:6d} {policy:8s}: sum = {total:8.1f}   "
                  f"bound 2 sqrt(SA T) = {2 * np.sqrt(SA * T):8.1f}")


def ex8_rnd_onehot(eta=0.1, k=8, rng=None):
    print("\nEx. 8  linear predictor, one-hot input: RND error after n visits")
    W_target = rng.standard_normal((k, 5))
    W = np.zeros((k, 5))
    x = np.zeros(5)
    x[2] = 1.0
    e0 = np.sum((W @ x - W_target @ x) ** 2)
    for n in range(1, 6):
        g = 2 * np.outer(W @ x - W_target @ x, x)          # gradient of ||Wx - f(x)||^2
        W -= eta / 2 * g                                   # step eta/2 => factor (1 - eta)
        e = np.sum((W @ x - W_target @ x) ** 2)
        print(f"  n={n}: error/e0 = {e / e0:.6f}   (1-eta)^(2n) = {(1 - eta) ** (2 * n):.6f}")


def ex10_rmax_threshold():
    print("\nEx. 10 R-MAX known threshold m >= V_max^2 ln(2/delta) / (2 eps^2)")
    for v, eps, delta in [(1.0, 0.1, 0.05), (1.0, 0.05, 0.05), (10.0, 0.1, 0.01)]:
        m = v ** 2 * np.log(2 / delta) / (2 * eps ** 2)
        print(f"  V_max={v:5.1f} eps={eps:5.2f} delta={delta:5.2f}: m >= {m:9.1f} -> "
              f"{int(np.ceil(m))}")


class Boltzmann(ds.EpsGreedyQ):
    """Q-learning with softmax(Q/tau) behaviour (undirected, value-aware dithering)."""

    def __init__(self, N, tau):
        super().__init__(N, eps=0.0)
        self.tau = tau

    def act(self, s, rng):
        z = self.Q[s] / self.tau
        p = np.exp(z - z.max())
        return int(rng.random() < p[1] / p.sum())


def ex11_boltzmann(Ns, taus, seeds, cap):
    print(f"\nEx. 11 Boltzmann exploration on DeepSea (episodes to first treasure, cap {cap})")
    for tau in taus:
        row = []
        for N in Ns:
            runs = [ds.run_one(lambda N, t=tau: Boltzmann(N, t), N, s, cap) for s in seeds]
            med, cens = ds.summarize([r["t_first"] for r in runs])
            row.append(f"N={N}: {med:6.0f} ({cens}/{len(seeds)} cens.)")
        print(f"  tau={tau:6.3f}  " + " | ".join(row))


def ex12_simhash(ks, seeds):
    print("\nEx. 12 SimHash on the 328 four-rooms cells: phi(s) = sign(A g(s)), "
          "g = (row, col) scaled to [-1, 1]")
    env = GridWorld(FOUR_ROOMS)
    g = np.array([[2 * r / (env.h - 1) - 1, 2 * c / (env.w - 1) - 1] for r, c in env.free])
    for k in ks:
        n_codes = []
        for seed in seeds:
            A = np.random.default_rng(seed).standard_normal((k, 2))
            codes = (g @ A.T > 0)
            n_codes.append(len({tuple(c) for c in codes}))
        print(f"  k={k:3d} bits: distinct codes {np.mean(n_codes):6.1f} of {env.n_cells} cells "
              f"(min {min(n_codes)}, max {max(n_codes)})")
    g3 = np.c_[g, np.ones(len(g))]                         # add a bias feature
    for k in ks:
        n_codes = []
        for seed in seeds:
            A = np.random.default_rng(seed).standard_normal((k, 3))
            n_codes.append(len({tuple(c) for c in (g3 @ A.T > 0)}))
        print(f"  k={k:3d} bits, with bias feature: distinct codes {np.mean(n_codes):6.1f}")


def go_explore_deep_sea(N, seed, cap, explore_steps=None):
    """Go-Explore on DeepSea: archive of cells, return by replaying actions, then explore.

    One iteration = one episode: choose an archived cell with probability proportional to
    1/sqrt(1 + times chosen), replay the stored action sequence to reach it (the world is
    deterministic, like an emulator), then take uniformly random actions until the episode
    ends, adding every new cell to the archive.  Returns the episode of the first treasure.
    """
    rng = np.random.default_rng(seed)
    env = ds.DeepSea(N, np.random.default_rng(10_000 + seed))
    archive = {0: []}                                     # cell -> actions that reach it
    chosen = {0: 0}
    for ep in range(1, cap + 1):
        cells = list(archive)
        w = np.array([1 / np.sqrt(1 + chosen[c]) for c in cells])
        cell = cells[rng.choice(len(cells), p=w / w.sum())]
        chosen[cell] += 1
        env.reset()
        term = False
        actions = list(archive[cell])
        for a in actions:                                 # "go": return without exploring
            _, _, term = env.step(a)
        while not term:                                   # "explore" from the cell
            a = int(rng.integers(2))
            s, r, term = env.step(a)
            actions.append(a)
            if r > 0.5:
                return ep
            if not term and s not in archive:
                archive[s] = list(actions)
                chosen[s] = 0
    return None


def ex13_go_explore(Ns, seeds, cap):
    print(f"\nEx. 13 Go-Explore on DeepSea: episodes until the treasure is first found "
          f"(cap {cap})")
    for N in Ns:
        eps = [go_explore_deep_sea(N, s, cap) for s in seeds]
        med, cens = ds.summarize(eps)
        print(f"  N={N:3d}: median {med:7.0f}  runs {eps}  (N(N+1)/2 = {N * (N + 1) // 2} "
              f"reachable cells)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test")
    args = parser.parse_args()
    seed = 0
    rng = np.random.default_rng(seed)
    print(f"Chapter 14 exercise checks (seed {seed}, quick={args.quick})")
    t0 = time.time()
    if args.quick:
        ex1_eps_greedy_success([6], [0], 500)
    else:
        ex1_eps_greedy_success([6, 10], list(range(5)), 4000)
    print()
    ex2_combination_lock([3, 5, 8] if args.quick else [3, 5, 8, 10, 12], 2000 if args.quick
                         else 100_000, rng)
    ex3_pseudocounts()
    ex5_pigeonhole(rng)
    ex8_rnd_onehot(rng=rng)
    ex10_rmax_threshold()
    if args.quick:
        ex11_boltzmann([4, 6], [0.01, 1.0], [0, 1], 300)
        ex12_simhash([8, 32], [0, 1])
        ex13_go_explore([10, 20], [0, 1], 2000)
    else:
        ex11_boltzmann([4, 6, 8, 10], [0.001, 0.01, 0.1, 1.0], list(range(5)), 4000)
        ex12_simhash([4, 8, 16, 32, 64, 128], list(range(5)))
        ex13_go_explore([10, 20, 30], list(range(5)), 10000)
    print(f"Total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
