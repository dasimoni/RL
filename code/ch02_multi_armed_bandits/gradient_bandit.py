"""Gradient bandit algorithm: with and without a baseline (S&B Figure 2.5 style) and a
numerical check of the derivation in Chapter 02, Section 8.

Part 1 (derivation check).  For a fixed problem q* and preferences H, the exact gradient
of the expected reward J(H) = sum_x pi(x) q*(x) is
        dJ/dH(a) = pi(a) (q*(a) - J)                                    (Eq. 8.7)
We draw A ~ pi, R ~ N(q*(A), 1) many times and check that the single-sample update
direction  g(a) = (R - b)(1[a = A] - pi(a))  (Eq. 8.6) has mean = exact gradient for any
baseline b that does not depend on A, and compare the variance for several b.  Two baselines
that DO depend on A (q*(A) and a noisy estimate Q(A)) show how the direction is lost.

Part 2 (learning curves).  10-armed testbed with q*(a) ~ N(+4, 1) (all rewards ~ +4),
alpha in {0.1, 0.4}, with and without the average-reward baseline.  A second panel repeats
the experiment with q*(a) ~ N(0, 1) to show that the no-baseline penalty comes from the
reward OFFSET.

Run:  python code/ch02_multi_armed_bandits/gradient_bandit.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from bandits import FIG_DIR, GaussianBandit, GradientBandit, run, setup_matplotlib, softmax, style


def gradient_check(n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    k = 10
    q = rng.normal(4.0, 1.0, size=k)                 # a +4 testbed problem
    H = rng.normal(0.0, 0.5, size=k)                 # some non-uniform preferences
    pi = softmax(H[None])[0]
    J = pi @ q
    exact = pi * (q - J)                             # Eq. (8.7)

    A = rng.choice(k, size=n_samples, p=pi)
    R = q[A] + rng.normal(size=n_samples)
    onehot = np.eye(k)[A]
    # variance-minimizing constant baseline: b* = E[R |e_A - pi|^2] / E[|e_A - pi|^2]
    sq = ((onehot - pi) ** 2).sum(axis=1)
    b_star = (R * sq).mean() / sq.mean()
    t = 5                                            # a time step for the "includes R_t" variant
    prev_mean = J                                    # pretend R_1..R_{t-1} averaged exactly J
    # Genuinely ACTION-dependent baselines (drawn after A and R, so the rows above are unchanged):
    #   b = q*(A_t):  expected update sum_x pi(x)(q*(x) - q*(x))(...) = 0  -> the gradient is destroyed;
    #   b = Q(A_t) with estimation errors: expected update pi(a)[e(a) - E_pi e], e = q* - Q (Exercise 2.8),
    #   which follows the ERRORS of the estimates, not q*(a) - J.
    Q_noisy = q + rng.normal(0.0, 0.5, size=k)
    baselines = {
        "b = 0 (no baseline)": np.zeros(n_samples),
        "b = 4 (rough guess)": np.full(n_samples, 4.0),
        "b = J = E[R_t]": np.full(n_samples, J),
        "b = b* (min-variance)": np.full(n_samples, b_star),
        f"b includes R_t (t={t})": ((t - 1) * prev_mean + R) / t,
        "b = q*(A_t)": q[A],
        "b = Q(A_t), Q = q* + N(0,.5²)": Q_noisy[A],
    }
    print(f"\nPart 1: gradient check on one +4 problem, {n_samples} samples of (A, R)")
    print(f"  exact gradient pi(a)(q*(a) - J): {np.array2string(exact, precision=3)}")
    print(f"  {'baseline':>30} | {'max |mean g - exact|':>20} | {'projection ratio':>16} | {'|mean g|/|grad|':>15} | {'cosine':>7} | "
          f"{'total variance':>14}")
    rows = []
    for name, b in baselines.items():
        g = (R - b)[:, None] * (onehot - pi)
        mean_g = g.mean(axis=0)
        ratio = (mean_g @ exact) / (exact @ exact)     # 1 = unbiased (on average along the gradient)
        cosine = (mean_g @ exact) / (np.linalg.norm(mean_g) * np.linalg.norm(exact))   # 1 = right direction
        var = g.var(axis=0).sum()
        se = np.sqrt(g.var(axis=0) / n_samples).max()
        norm_ratio = np.linalg.norm(mean_g) / np.linalg.norm(exact)
        rows.append((name, np.abs(mean_g - exact).max(), ratio, norm_ratio, cosine, var))
        print(f"  {name:>30} | {np.abs(mean_g - exact).max():20.4f} | {ratio:16.3f} | {norm_ratio:15.3f} | "
              f"{cosine:7.3f} | {var:14.3f}"
              f"   (MC s.e. ≈ {se:.4f})")
    print(f"  (J = {J:.3f}, b* = {b_star:.3f}; 'includes R_t' predicts a ratio of 1 - 1/t = {1 - 1 / t:.2f})")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    seed = 0
    runs, steps = (200, 1000) if args.quick else (2000, 1000)
    n_samples = 200_000 if args.quick else 2_000_000
    print(f"Seed {seed}; {runs} runs x {steps} steps; gradient check with {n_samples} samples")
    t0 = time.time()
    gradient_check(n_samples, seed)

    configs = [(0.1, True), (0.4, True), (0.1, False), (0.4, False)]
    results = {}
    for offset in (4.0, 0.0):
        results[offset] = []
        for i, (alpha, base) in enumerate(configs):
            env = GaussianBandit.testbed(runs, 10, np.random.default_rng(seed), mean_offset=offset)
            results[offset].append(run(env, GradientBandit(10, alpha, base), steps,
                                       np.random.default_rng(seed + 20 + i)))
    print(f"\nPart 2: learning curves ({runs} runs x {steps} steps)")
    print(f"  {'agent':>28} | {'% opt 901-1000, mean +4':>24} | {'% opt 901-1000, mean 0':>23}")
    for r4, r0 in zip(results[4.0], results[0.0]):
        print(f"  {r4.name:>28} | {100 * r4.pct_optimal[-100:].mean():23.1f}% | "
              f"{100 * r0.pct_optimal[-100:].mean():22.1f}%")

    if not args.quick:
        plt = setup_matplotlib()
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
        x = np.arange(1, steps + 1)
        for j, offset in enumerate((4.0, 0.0)):
            for i, r in enumerate(results[offset]):
                ax[j].plot(x, 100 * r.pct_optimal, label=r.name, **style(i), lw=1.2)
            ax[j].set(xlabel="Steps", ylim=(0, 100),
                      title=f"({'ab'[j]}) q*(a) ~ N({offset:g}, 1)")
            ax[j].legend(loc="lower right")
        ax[0].set_ylabel("% optimal action")
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "gradient_bandit.png")
        fig.savefig(out)
        print(f"\nsaved {out}")
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
