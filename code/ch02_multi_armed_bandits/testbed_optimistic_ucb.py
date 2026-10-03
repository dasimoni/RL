"""Optimistic initial values and UCB on the 10-armed testbed (S&B Figures 2.3 and 2.4 style).

Chapter 02, Sections 5 and 6.4.

(a) Optimistic greedy: Q_1 = 5, eps = 0, alpha = 0.1   vs   realistic eps-greedy: Q_1 = 0, eps = 0.1, alpha = 0.1
    (% optimal action).
(b) UCB with c = 2 (sample averages)   vs   eps-greedy eps = 0.1 (sample averages)  (average reward).

Both curves show a SPIKE at step 11.  The script measures it and explains it: after the
first 10 steps each arm has been pulled exactly once, and at step 11 both methods pick
the arm whose single reward was highest, which is often the best arm.

Run:  python code/ch02_multi_armed_bandits/testbed_optimistic_ucb.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from bandits import FIG_DIR, EpsilonGreedy, GaussianBandit, UCB, run, setup_matplotlib, style


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    seed = 0
    runs, steps = (200, 1000) if args.quick else (2000, 1000)
    print(f"Seed {seed}; {runs} runs x {steps} steps on the 10-armed testbed")
    t0 = time.time()

    def go(agent, i):
        env = GaussianBandit.testbed(runs, 10, np.random.default_rng(seed))   # common problems
        return run(env, agent, steps, np.random.default_rng(seed + 10 + i))

    opt = go(EpsilonGreedy(10, eps=0.0, alpha=0.1, q1=5.0, name="optimistic greedy Q₁=5, α=0.1"), 0)
    real = go(EpsilonGreedy(10, eps=0.1, alpha=0.1, q1=0.0, name="realistic ε=0.1, Q₁=0, α=0.1"), 1)
    ucb = go(UCB(10, c=2.0, name="UCB c=2"), 2)
    eg = go(EpsilonGreedy(10, eps=0.1, name="ε-greedy ε=0.1 (sample avg.)"), 3)

    print(f"\n{'agent':>32} | {'avg R 1-1000':>12} | {'avg R 901-1000':>14} | {'% opt 901-1000':>14}")
    for r in (opt, real, ucb, eg):
        print(f"{r.name:>32} | {r.avg_reward.mean():12.3f} | {r.avg_reward[-100:].mean():14.3f} | "
              f"{100 * r.pct_optimal[-100:].mean():13.1f}%")

    print("\nThe step-11 spike (steps are 1-indexed):")
    for r in (opt, ucb):
        print(f"  {r.name:>30}: % optimal at steps 10,11,12,13 = "
              + ", ".join(f"{100 * r.pct_optimal[s - 1]:.1f}%" for s in (10, 11, 12, 13))
              + "; avg reward = " + ", ".join(f"{r.avg_reward[s - 1]:.2f}" for s in (10, 11, 12, 13)))
    # Check the explanation directly: probability that the best arm also produced the highest
    # of 10 single rewards (one N(q*(a),1) sample per arm), on the same problems.
    q = GaussianBandit.testbed(runs, 10, np.random.default_rng(seed)).q_star
    one_sample = q + np.random.default_rng(99).normal(size=q.shape)
    p_best = np.mean(one_sample.argmax(axis=1) == q.argmax(axis=1))
    print(f"  P(arm with the highest single reward is the best arm) ≈ {100 * p_best:.1f}%  "
          f"(matches the spike height)")
    # Both agents pull every arm exactly once in steps 1-10: check it directly.
    for agent in (EpsilonGreedy(10, eps=0.0, alpha=0.1, q1=5.0), UCB(10, c=2.0)):
        first10 = run(GaussianBandit.testbed(runs, 10, np.random.default_rng(seed)), agent, 10,
                      np.random.default_rng(seed + 50))
        print(f"  {type(agent).__name__:>13}: fraction of runs whose first 10 pulls cover all 10 arms = "
              f"{np.mean((first10.counts == 1).all(axis=1)):.3f}")

    if not args.quick:
        plt = setup_matplotlib()
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
        x = np.arange(1, steps + 1)
        ax[0].plot(x, 100 * opt.pct_optimal, label=opt.name, **style(0), lw=1.2)
        ax[0].plot(x, 100 * real.pct_optimal, label=real.name, **style(1), lw=1.2)
        ax[0].set(xlabel="Steps", ylabel="% optimal action", ylim=(0, 100),
                  title="(a) Optimistic initial values (constant α)")
        ax[1].plot(x, ucb.avg_reward, label=ucb.name, **style(2), lw=1.2)
        ax[1].plot(x, eg.avg_reward, label=eg.name, **style(1), lw=1.2)
        ax[1].set(xlabel="Steps", ylabel="Average reward", title="(b) Upper confidence bound")
        for a in ax:
            a.legend(loc="lower right")
        ins = ax[1].inset_axes([0.42, 0.28, 0.3, 0.32])
        ins.plot(x[:30], ucb.avg_reward[:30], **style(2), lw=1.2, marker="o", ms=2.5)
        ins.set_title("first 30 steps", fontsize=8)
        ins.tick_params(labelsize=7)
        ins.axvline(11, color="#8a8984", lw=0.8, ls=":")
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "testbed_optimistic_ucb.png")
        fig.savefig(out)
        print(f"\nsaved {out}")
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
