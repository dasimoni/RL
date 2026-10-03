"""Greedy vs epsilon-greedy on the 10-armed testbed (in the style of S&B Figure 2.2).

Chapter 02, Section 2.3.

Testbed: for each of `runs` independent problems, q*(a) ~ N(0, 1) for a = 1..10 and
R_t ~ N(q*(A_t), 1).  Every agent uses sample-average estimates with Q_1(a) = 0 and
breaks ties at random.  We plot (a) average reward and (b) % optimal action over the
first 1000 steps, and (c) % optimal action over a longer horizon, where the smaller
epsilon eventually wins.  All agents face the SAME 2000 problems (common random numbers).

Run:  python code/ch02_multi_armed_bandits/testbed_greedy_vs_epsilon.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from bandits import FIG_DIR, EpsilonGreedy, GaussianBandit, run, setup_matplotlib, style


def experiment(runs, steps, eps_values, seed):
    results = []
    for i, eps in enumerate(eps_values):
        env = GaussianBandit.testbed(runs, 10, np.random.default_rng(seed))   # same problems for all
        res = run(env, EpsilonGreedy(10, eps), steps, np.random.default_rng(seed + 1 + i))
        results.append(res)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    seed = 0
    eps_values = [0.0, 0.01, 0.1]
    runs, steps = (200, 1000) if args.quick else (2000, 1000)
    long_runs, long_steps = (50, 2000) if args.quick else (1000, 10000)
    print(f"Seed {seed}; eps in {eps_values}; sample averages, Q1 = 0")
    print(f"Short experiment: {runs} runs x {steps} steps; long experiment: {long_runs} runs x {long_steps} steps")
    t0 = time.time()

    short = experiment(runs, steps, eps_values, seed)
    long = experiment(long_runs, long_steps, eps_values, seed + 100)

    # Best possible: always pull the best arm -> E[max_a q*(a)] for 10 standard normals ~ 1.54
    best = GaussianBandit.testbed(runs, 10, np.random.default_rng(seed)).q_star.max(axis=1).mean()
    best_long = GaussianBandit.testbed(long_runs, 10, np.random.default_rng(seed + 100)).q_star.max(axis=1).mean()
    print(f"\nmean of max_a q*(a): {best:.3f} over the {runs} short-run problems; "
          f"{best_long:.3f} over the {long_runs} (different) long-run problems")
    print(f"{'agent':>16} | {'avg R 1-1000':>12} | {'avg R 901-1000':>14} | {'% opt 901-1000':>14} | "
          f"{'long run, last 10%: avg R':>25} | {'% opt':>6}")
    tail = slice(int(0.9 * long_steps), None)
    for s, l in zip(short, long):
        print(f"{s.name:>16} | {s.avg_reward.mean():12.3f} | {s.avg_reward[-100:].mean():14.3f} | "
              f"{100 * s.pct_optimal[-100:].mean():13.1f}% | {l.avg_reward[tail].mean():25.3f} | "
              f"{100 * l.pct_optimal[tail].mean():5.1f}%")
    # When does eps = 0.01 overtake eps = 0.1 in % optimal (500-step moving averages)?
    # (Panel (c) of the figure plots a 50-step moving average for readability.)
    w = 500
    ma = [np.convolve(l.pct_optimal, np.ones(w) / w, mode="valid") for l in long]
    ahead = np.nonzero(ma[1] > ma[2])[0]
    if len(ahead):
        print(f"long run: ε=0.01 first overtakes ε=0.1 in % optimal ({w}-step moving average) "
              f"around step {ahead[0] + w}")
    # Analytic ceilings for % optimal: 1 - eps + eps/k once Q is perfect
    print("asymptotic ceiling on % optimal (1 - eps + eps/k):",
          ", ".join(f"ε={e:g}: {100 * (1 - e + e / 10):.1f}%" for e in eps_values))
    # How often does greedy lock onto a suboptimal arm?
    g = short[0]
    stuck = np.mean(g.counts.argmax(axis=1) != GaussianBandit.testbed(runs, 10, np.random.default_rng(seed)).q_star.argmax(axis=1))
    print(f"greedy: most-pulled arm is NOT the best arm in {100 * stuck:.1f}% of runs")

    if not args.quick:
        plt = setup_matplotlib()
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.0))
        x = np.arange(1, steps + 1)
        for i, res in enumerate(short):
            ax[0].plot(x, res.avg_reward, label=res.name, **style(i), lw=1.2)
            ax[1].plot(x, 100 * res.pct_optimal, label=res.name, **style(i), lw=1.2)
        ax[0].axhline(best, color="#8a8984", lw=1, ls=":")
        ax[0].text(steps, best, " E[max q*]", va="center", ha="right", fontsize=8, color="#52514e",
                   bbox=dict(fc="white", ec="none", pad=1))
        ax[0].set(xlabel="Steps", ylabel="Average reward", title=f"(a) Average reward ({runs} runs)")
        ax[1].set(xlabel="Steps", ylabel="% optimal action", title="(b) % optimal action", ylim=(0, 100))
        xl = np.arange(1, long_steps + 1)
        w = 50   # moving average for readability in the long run (fewer runs -> noisier)
        for i, res in enumerate(long):
            sm = np.convolve(res.pct_optimal, np.ones(w) / w, mode="valid")
            ax[2].plot(xl[w - 1:], 100 * sm, label=res.name, **style(i), lw=1.2)
        ax[2].set(xlabel="Steps", ylabel="% optimal action", ylim=(0, 100),
                  title=f"(c) Longer horizon ({long_runs} runs, {w}-step mov. avg.)")
        for a in ax:
            a.legend(loc="lower right")
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "testbed_greedy_vs_epsilon.png")
        fig.savefig(out)
        print(f"\nsaved {out}")
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
