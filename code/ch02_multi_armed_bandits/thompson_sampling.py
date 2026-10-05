"""Thompson sampling: Beta-Bernoulli posteriors in action, and every method of the chapter
on the 10-armed testbed.

Chapter 02, Sections 7.3 and 7.5.

Part 1.  A 3-armed Bernoulli bandit with q* = (0.45, 0.55, 0.60).  One run of Beta(1,1)
Thompson sampling; we snapshot the posteriors Beta(1 + S_a, 1 + F_a) after 0, 20, 200 and
2000 pulls.  The best arm's posterior concentrates; the others stay wide but are pushed far
enough left that they are rarely sampled above the best arm.  For each snapshot we also
print P(arm a is optimal | data), estimated by Monte Carlo; by construction this is exactly
the probability that TS pulls arm a next ("probability matching", Section 7.2).

Part 2.  10-armed Gaussian testbed, 2000 runs x 1000 steps: Gaussian Thompson sampling
(prior N(0,1), known noise sd 1, i.e. the TRUE generative prior of the testbed) against
eps-greedy, optimistic greedy, UCB and the gradient bandit at their usual settings.

Run:  python code/ch02_multi_armed_bandits/thompson_sampling.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from scipy import stats

from bandits import (FIG_DIR, BernoulliBandit, EpsilonGreedy, GaussianBandit, GradientBandit,
                     ThompsonBeta, ThompsonGaussian, UCB, run, setup_matplotlib, style)


def posterior_snapshots(probs, snap_times, seed):
    """One run of Beta-Bernoulli TS (written as a plain loop, mirroring Algorithm 7.1)."""
    rng = np.random.default_rng(seed)
    k = len(probs)
    S, F = np.zeros(k), np.zeros(k)
    snaps = {0: (S.copy(), F.copy())}
    for t in range(1, max(snap_times) + 1):
        theta = rng.beta(1 + S, 1 + F)          # one sample from each arm's posterior
        a = int(np.argmax(theta))
        r = float(rng.random() < probs[a])      # Bernoulli reward
        S[a] += r
        F[a] += 1 - r
        if t in snap_times:
            snaps[t] = (S.copy(), F.copy())
    return snaps


def probability_matching_check(S, F, rng, n=200_000):
    """P(arm a is optimal | posterior) by Monte Carlo (= TS's probability of pulling a next)."""
    draws = rng.beta(1 + S, 1 + F, size=(n, len(S)))
    return np.bincount(draws.argmax(axis=1), minlength=len(S)) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    seed = 0
    probs = np.array([0.45, 0.55, 0.60])
    snap_times = [20, 200, 2000]
    runs, steps = (200, 1000) if args.quick else (2000, 1000)
    print(f"Seed {seed}; Part 1: Bernoulli q* = {probs.tolist()}, snapshots at {snap_times}; "
          f"Part 2: {runs} runs x {steps} steps")
    t0 = time.time()

    # ---------------- Part 1 ----------------
    snaps = posterior_snapshots(probs, snap_times, seed)
    print("\nPart 1: one run of Beta(1,1) Thompson sampling")
    for t, (S, F) in snaps.items():
        n = S + F
        mean = (1 + S) / (2 + n)
        print(f"  t={t:5d}: pulls N = {n.astype(int).tolist()}, posterior means = "
              f"{np.round(mean, 3).tolist()}, P(arm optimal) = "
              f"{np.round(probability_matching_check(S, F, np.random.default_rng(1)), 3).tolist()}")
    # Probability matching, checked on the many-run simulation: frequency of each action at step t
    pm_runs = 400 if args.quick else 4000
    env = BernoulliBandit(probs, pm_runs, np.random.default_rng(seed + 1))
    res = run(env, ThompsonBeta(3), 200, np.random.default_rng(seed + 2))
    frac_best = res.counts[:, 2].mean() / 200
    print(f"  over {pm_runs} runs of 200 steps: fraction of pulls on the best arm = {frac_best:.3f}; "
          f"mean regret at T=200: {res.regret_mean[-1]:.2f}")

    # ---------------- Part 2 ----------------
    agents = [
        EpsilonGreedy(10, 0.1, name="ε-greedy ε=0.1"),
        EpsilonGreedy(10, 0.0, alpha=0.1, q1=5.0, name="optimistic greedy Q₁=5, α=0.1"),
        UCB(10, 2.0, name="UCB c=2"),
        GradientBandit(10, 0.1, name="gradient α=0.1"),
        ThompsonGaussian(10, 0.0, 1.0, 1.0, name="Thompson (Gaussian prior N(0,1))"),
    ]
    results = []
    for i, ag in enumerate(agents):
        env = GaussianBandit.testbed(runs, 10, np.random.default_rng(seed))
        results.append(run(env, ag, steps, np.random.default_rng(seed + 30 + i)))
    print(f"\nPart 2: 10-armed testbed, {runs} runs x {steps} steps")
    print(f"  {'agent':>34} | {'avg R 1-1000':>12} | {'avg R 901-1000':>14} | {'% opt 901-1000':>14} | {'time':>6}")
    for r in results:
        print(f"  {r.name:>34} | {r.avg_reward.mean():12.3f} | {r.avg_reward[-100:].mean():14.3f} | "
              f"{100 * r.pct_optimal[-100:].mean():13.1f}% | {r.seconds:5.1f}s")

    if not args.quick:
        plt = setup_matplotlib()
        # Figure 1: posterior evolution
        fig, ax = plt.subplots(1, len(snaps), figsize=(13, 3.2), sharey=False)
        xs = np.linspace(0, 1, 501)
        for j, (t, (S, F)) in enumerate(snaps.items()):
            for a in range(3):
                dens = stats.beta.pdf(xs, 1 + S[a], 1 + F[a])
                ax[j].plot(xs, dens, **style(a), lw=1.6,
                           label=f"arm {a + 1}: q*={probs[a]:.2f}, N={int(S[a] + F[a])}")
                ax[j].axvline(probs[a], color=style(a)["color"], lw=0.8, alpha=0.6)
            ax[j].set(title=f"after t = {t} pulls", xlabel="mean reward θ")
            ymax = max(1.0, max(stats.beta.pdf(xs, 1 + S[a], 1 + F[a]).max() for a in range(3)))
            ax[j].set_ylim(0, 1.5 * ymax)                  # headroom for the legend
            ax[j].ticklabel_format(axis="y", useOffset=False)
            ax[j].legend(loc="upper left", fontsize=7.5)
        ax[0].set_ylabel("posterior density")
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "thompson_posteriors.png")
        fig.savefig(out)
        print(f"\nsaved {out}")

        # Figure 2: all methods on the testbed
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.0))
        x = np.arange(1, steps + 1)
        w = 10
        for i, r in enumerate(results):
            sm = np.convolve(r.avg_reward, np.ones(w) / w, mode="valid")
            ax[0].plot(x[w - 1:], sm, label=r.name, **style(i), lw=1.3)
            ax[1].plot(x, 100 * r.pct_optimal, label=r.name, **style(i), lw=1.3)
        ax[0].set(xlabel="Steps", ylabel="Average reward", title=f"(a) Average reward ({w}-step mov. avg.)")
        ax[1].set(xlabel="Steps", ylabel="% optimal action", ylim=(0, 100), title="(b) % optimal action")
        ax[1].legend(loc="lower right")
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "testbed_all_methods.png")
        fig.savefig(out)
        print(f"saved {out}")
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
