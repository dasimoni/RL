"""Nonstationary bandit: sample averages vs a constant step size (S&B Exercise 2.5).

Chapter 02, Section 4.

All q*(a) start equal (0) and take independent random walks: after every step each q*(a)
gets an N(0, 0.01^2) increment.  Rewards R ~ N(q*(A), 1).  10,000 steps.  All agents use
eps = 0.1 exploration unless stated:
  * sample averages (alpha_n = 1/n)                       -- weights all past rewards equally
  * constant alpha = 0.1                                  -- exponential recency weighting
  * 'unbiased constant step' beta_n = alpha / o_n (S&B Exercise 2.7), alpha = 0.1
  * optimistic greedy Q_1 = 5, alpha = 0.1, eps = 0       -- exploration only at the start
  * Gaussian Thompson sampling that assumes a STATIONARY world (posterior keeps shrinking)

Part 2 (what the unbiased step is for).  In Part 1 every q*(a) starts at 0 = Q_1, so there is
no initial bias to remove and the unbiased step changes nothing.  Here the drifting means start
at q*(a) ~ N(0, 1) and at q*(a) ~ N(4, 1) while Q_1 = 0, and we compare constant alpha = 0.1
with the unbiased step beta_n (Eq. 4.4) over 1000 steps.

Run:  python code/ch02_multi_armed_bandits/nonstationary.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from bandits import FIG_DIR, EpsilonGreedy, GaussianBandit, ThompsonGaussian, run, setup_matplotlib, style


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    seed = 0
    walk_std = 0.01
    runs, steps = (100, 2000) if args.quick else (2000, 10_000)
    print(f"Seed {seed}; {runs} runs x {steps} steps; q* random walk sd {walk_std} per step; k = 10")
    t0 = time.time()

    agents = [
        EpsilonGreedy(10, 0.1, alpha=None, name="sample average, ε=0.1"),
        EpsilonGreedy(10, 0.1, alpha=0.1, name="constant α=0.1, ε=0.1"),
        EpsilonGreedy(10, 0.1, alpha=0.1, unbiased=True, name="unbiased constant step α=0.1, ε=0.1"),
        EpsilonGreedy(10, 0.0, alpha=0.1, q1=5.0, name="optimistic greedy Q₁=5, α=0.1"),
        ThompsonGaussian(10, 0.0, 1.0, 1.0, name="Thompson (stationary Gaussian model)"),
    ]
    results = []
    for i, ag in enumerate(agents):
        env = GaussianBandit(np.zeros((runs, 10)), np.random.default_rng(seed), walk_std=walk_std)
        results.append(run(env, ag, steps, np.random.default_rng(seed + 1 + i)))
        print(f"  ran {ag.name:<40} in {results[-1].seconds:5.1f} s")
    # The best arm's value drifts upward: E[max_a q*_T(a)] ≈ 0.01 sqrt(T) * E[max of 10 N(0,1)] ≈ 1.54
    print(f"  mean over runs of max_a q*(a) at the final step: {env.q_star.max(axis=1).mean():.3f} "
          f"(theory ≈ {walk_std * np.sqrt(steps) * 1.539:.3f})")

    tail = slice(steps // 2, None)
    print(f"\n{'agent':>40} | {'avg R, 2nd half':>15} | {'% opt, 2nd half':>15} | {'% opt, last 1000':>16} | "
          f"{'regret/step, 2nd half':>21}")
    for r in results:
        reg_per_step = (r.regret_mean[-1] - r.regret_mean[steps // 2 - 1]) / (steps - steps // 2)
        print(f"{r.name:>40} | {r.avg_reward[tail].mean():15.3f} | {100 * r.pct_optimal[tail].mean():14.1f}% | "
              f"{100 * r.pct_optimal[-1000:].mean():15.1f}% | {reg_per_step:21.3f}")

    # ---------------- Part 2: initial bias ----------------
    runs2, steps2 = (200, 500) if args.quick else (2000, 1000)
    print(f"\nPart 2: initial bias. Drifting q* that START at N(offset, 1); Q_1 = 0; ε = 0.1, α = 0.1; "
          f"{runs2} runs x {steps2} steps")
    print(f"{'start q*(a) ~':>14} | {'step size':>22} | {'avg R, all steps':>16} | {'% opt, first 100':>16} | "
          f"{'% opt, last 100':>15}")
    for j, offset in enumerate((0.0, 4.0)):
        for unbiased in (False, True):
            env2 = GaussianBandit.testbed(runs2, 10, np.random.default_rng(seed + 5), mean_offset=offset,
                                          walk_std=walk_std)
            r2 = run(env2, EpsilonGreedy(10, 0.1, alpha=0.1, unbiased=unbiased), steps2,
                     np.random.default_rng(seed + 6))
            print(f"{f'N({offset:g}, 1)':>14} | {'unbiased β_n (4.4)' if unbiased else 'constant α = 0.1':>22} | "
                  f"{r2.avg_reward.mean():16.3f} | {100 * r2.pct_optimal[:100].mean():15.1f}% | "
                  f"{100 * r2.pct_optimal[-100:].mean():14.1f}%")

    if not args.quick:
        plt = setup_matplotlib()
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.0))
        x = np.arange(1, steps + 1)
        w = 100
        for i, r in enumerate(results):
            sm_r = np.convolve(r.avg_reward, np.ones(w) / w, mode="valid")
            sm_o = np.convolve(r.pct_optimal, np.ones(w) / w, mode="valid")
            ax[0].plot(x[w - 1:], sm_r, label=r.name, **style(i), lw=1.3)
            ax[1].plot(x[w - 1:], 100 * sm_o, label=r.name, **style(i), lw=1.3)
        ax[0].set(xlabel="Steps", ylabel="Average reward", title=f"(a) Average reward ({w}-step mov. avg.)")
        ax[1].set(xlabel="Steps", ylabel="% optimal action", ylim=(0, 100),
                  title=f"(b) % optimal action ({runs} runs, random-walk q*)")
        ax[0].legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "nonstationary.png")
        fig.savefig(out)
        print(f"\nsaved {out}")
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
