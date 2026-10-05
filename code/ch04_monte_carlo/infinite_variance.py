"""Infinite variance of ordinary importance sampling (Sec. 6.6; S&B Example 5.5, Figure 5.4).

One non-terminal state s, two actions:
  right -> terminate, reward 0
  left  -> with prob 0.9 back to s (reward 0); with prob 0.1 terminate with reward +1.
Target pi: always left, so v_pi(s) = 1 (gamma = 1). Behavior b: left/right with prob 1/2.

Under b, rho*G is nonzero only for episodes consisting of T lefts that end by the
left-termination; then rho = 2^T and G = 1. E_b[rho G] = 1 but E_b[(rho G)^2] = infinity.

The script simulates the MDP step by step (`run_episode`, used to validate) and, for the
10^8-episode runs, with an exactly equivalent vectorized sampler: each step ends the episode
with prob 0.5 + 0.5*0.1 = 0.55, so T ~ Geometric(0.55), and an ending is a left-termination
with prob 0.05/0.55 = 1/11 independently of T.

    python code/ch04_monte_carlo/infinite_variance.py [--quick]
"""
import argparse
import os
import random
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LEFT, RIGHT = 0, 1


def run_episode(rng: random.Random):
    """Literal simulation of one episode under b. Returns (rho, G, T)."""
    rho, T = 1.0, 0
    while True:
        a = LEFT if rng.random() < 0.5 else RIGHT
        T += 1
        rho *= (1.0 / 0.5) if a == LEFT else 0.0          # pi(left|s) = 1, pi(right|s) = 0
        if a == RIGHT:
            return rho, 0.0, T
        if rng.random() < 0.1:                              # left: terminate with +1
            return rho, 1.0, T
        # else: back to s with reward 0


def sample_rho_g(rng: np.random.Generator, n: int) -> np.ndarray:
    """n i.i.d. samples of rho*G under b (vectorized, distributionally identical)."""
    T = rng.geometric(0.55, size=n)
    left_end = rng.random(n) < 1 / 11
    return np.where(left_end, np.exp2(T.astype(float)), 0.0)


def ordinary_is_path(rng, n_total, checkpoints, chunk=5_000_000):
    """Running ordinary-IS estimate sum(rho G)/n at the checkpoints, plus the episode index
    of the first nonzero rho*G (when the weighted-IS estimate jumps from 0 to 1)."""
    out, total, done, first_nz = [], 0.0, 0, None
    cps = np.asarray(checkpoints)
    while done < n_total:
        m = min(chunk, n_total - done)
        x = sample_rho_g(rng, m)
        if first_nz is None and (x > 0).any():
            first_nz = done + int(np.argmax(x > 0)) + 1
        cs = total + np.cumsum(x)
        sel = cps[(cps > done) & (cps <= done + m)]
        out.extend(cs[sel - done - 1] / sel)
        total = cs[-1]
        done += m
    return np.array(out), first_nz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    runs = 3 if args.quick else 10
    n_total = 1_000_000 if args.quick else 100_000_000
    print(f"Infinite-variance example | seed={args.seed} runs={runs} episodes/run={n_total:,}")
    print("exact: E_b[rho G] = 0.1 * sum_k 0.9^k = 1;  E_b[(rho G)^2] = 0.2 * sum_k 1.8^k = infinity")

    # 1. Validate the vectorized sampler against the literal step-by-step simulation.
    t0 = time.time()
    n_val = 200_000
    prng = random.Random(args.seed)
    lit = np.array([run_episode(prng) for _ in range(n_val)])
    vec = sample_rho_g(np.random.default_rng(args.seed + 99), n_val)
    print(f"\nvalidation ({n_val:,} episodes each): P(rho G > 0) literal {np.mean(lit[:, 0] * lit[:, 1] > 0):.4f}"
          f" vs vectorized {np.mean(vec > 0):.4f} (exact {1 / 11:.4f}); "
          f"mean T literal {lit[:, 2].mean():.4f} (exact {1 / 0.55:.4f})")
    lit_x = lit[:, 0] * lit[:, 1]
    for k in (1, 2, 3, 4):   # P(rho G = 2^k) = P(k lefts, last one terminates) = 0.45^(k-1) 0.05
        print(f"  P(rho G = {2 ** k:2d}): literal {np.mean(lit_x == 2 ** k):.5f}  vectorized "
              f"{np.mean(vec == 2 ** k):.5f}  exact {0.45 ** (k - 1) * 0.05:.5f}")
    print("  (sample means of rho*G are NOT a useful check: with infinite variance they are "
          f"erratic -- here {lit_x.mean():.3f} and {vec.mean():.3f})")

    # 2. Ten long runs of ordinary IS (and weighted IS, which is trivially exact here).
    cps = np.unique(np.round(np.logspace(0, np.log10(n_total), 200)).astype(int))
    rng = np.random.default_rng(args.seed)
    paths, firsts = [], []
    for k in range(runs):
        p, f = ordinary_is_path(rng, n_total, cps)
        paths.append(p)
        firsts.append(f)
    paths = np.array(paths)
    print(f"\nordinary IS estimates (true value 1), {runs} runs, {time.time() - t0:.1f}s:")
    for n in (10, 1_000, 100_000, 10_000_000, 100_000_000):
        if n <= n_total:
            i = int(np.searchsorted(cps, n))
            print(f"  after {n:>11,} episodes: min {paths[:, i].min():.3f}  median "
                  f"{np.median(paths[:, i]):.3f}  max {paths[:, i].max():.3f}")
    print(f"weighted IS: 0 until the first episode consistent with pi (episode {min(firsts)}-"
          f"{max(firsts)} across runs), exactly 1 forever after")

    # 3. The sample variance of rho*G keeps growing instead of converging.
    x = sample_rho_g(np.random.default_rng(args.seed + 7), min(n_total, 10_000_000))
    print("\nsample second moment of rho*G over the first n episodes of one stream:")
    for n in (10**3, 10**4, 10**5, 10**6, 10**7):
        if n <= len(x):
            print(f"  n = {n:>10,}: mean {x[:n].mean():8.3f}   mean of squares {np.mean(x[:n] ** 2):14.1f}")
    print(f"total elapsed {time.time() - t0:.1f}s")

    if args.quick:
        return
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for k in range(runs):
        ax.semilogx(cps, paths[k], lw=0.9)
    ax.axhline(1.0, color="k", ls="--", lw=1, label=r"$v_\pi(s) = 1$ (= weighted IS after first consistent episode)")
    ax.set_ylim(0, 3)
    ax.set_xlabel("episodes (log scale)")
    ax.set_ylabel(r"ordinary IS estimate of $v_\pi(s)$")
    ax.set_title(f"Ordinary importance sampling with infinite variance ({runs} runs)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "infinite_variance.png"), dpi=110)
    print("saved figures/infinite_variance.png")


if __name__ == "__main__":
    main()
