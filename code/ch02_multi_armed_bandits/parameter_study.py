"""Parameter study on the 10-armed testbed (in the style of S&B Figure 2.6), extended with
Boltzmann exploration and Thompson sampling.

Chapter 02, Section 10.

Each point is the average reward over the first 1000 steps, averaged over 2000 runs, for
one algorithm at one value of its parameter.  The x-axis is log2-scaled and shared by
different parameters:
    eps-greedy (sample averages)            eps
    gradient bandit (with baseline)         alpha
    UCB (sample averages)                   c
    greedy with optimistic init, alpha=0.1  Q_1
    Boltzmann on sample averages            tau (temperature)
    Gaussian Thompson sampling              s0 (prior standard deviation; the truth is 1)
All algorithms see the same 2000 problems (common random numbers).

Run:  python code/ch02_multi_armed_bandits/parameter_study.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from bandits import (FIG_DIR, LINESTYLES, PALETTE, Boltzmann, EpsilonGreedy, GaussianBandit,
                     GradientBandit, ThompsonGaussian, UCB, run, setup_matplotlib)

FAMILIES = {
    # name: (parameter symbol, exponents of 2, agent factory)
    "ε-greedy": ("ε", range(-7, -1), lambda v: EpsilonGreedy(10, eps=v)),
    "gradient bandit": ("α", range(-5, 3), lambda v: GradientBandit(10, alpha=v)),
    "UCB": ("c", range(-4, 3), lambda v: UCB(10, c=v)),
    "greedy, optimistic init (α=0.1)": ("Q₁", range(-2, 3), lambda v: EpsilonGreedy(10, 0.0, alpha=0.1, q1=v)),
    "Boltzmann": ("τ", range(-6, 1), lambda v: Boltzmann(10, tau=v)),
    "Thompson (Gaussian)": ("s₀", range(-4, 3), lambda v: ThompsonGaussian(10, 0.0, s0=v, sigma=1.0)),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    seed = 0
    runs, steps = (100, 300) if args.quick else (2000, 1000)
    print(f"Seed {seed}; {runs} runs x {steps} steps per point; metric = mean reward over steps 1..{steps}")
    t0 = time.time()

    table = {}
    per_run = {}                # (family, parameter) -> per-run average reward [runs]  (for paired s.e.)
    for f, (name, (sym, exps, make)) in enumerate(FAMILIES.items()):
        vals, scores = [], []
        for i, e in enumerate(exps):
            v = 2.0 ** e
            env = GaussianBandit.testbed(runs, 10, np.random.default_rng(seed))
            res = run(env, make(v), steps, np.random.default_rng(seed + 1000 * f + i))
            vals.append(v)
            scores.append(res.avg_reward.mean())
            per_run[name, v] = res.run_avg_reward
        table[name] = (sym, np.array(vals), np.array(scores))
        best = int(np.argmax(scores))
        print(f"  {name:>32}: " + "  ".join(f"{sym}=2^{e}:{s:.3f}" for e, s in zip(exps, scores))
              + f"   -> best {sym}={vals[best]:g} ({scores[best]:.3f})")
    # Noise level.  A point's value varies a lot from problem set to problem set (max_a q*(a)
    # differs between problems), but every point uses the SAME problems, so differences between
    # points are much more precise than the points themselves: report both standard errors.
    def best_of(name):
        sym, vals, scores = table[name]
        return vals[int(np.argmax(scores))]

    def paired(n1, v1, n2, v2):
        d = per_run[n1, v1] - per_run[n2, v2]
        return d.mean(), d.std(ddof=1) / np.sqrt(runs)

    ucb, ts, opt, eps = "UCB", "Thompson (Gaussian)", "greedy, optimistic init (α=0.1)", "ε-greedy"
    se_point = per_run[ucb, 1.0].std(ddof=1) / np.sqrt(runs)
    print(f"\nnoise level: s.e. of one point (UCB c=1) ≈ {se_point:.4f}")
    print("paired differences on the common problems (difference ± paired s.e.):")
    for (n1, v1), (n2, v2) in [((ucb, 1.0), (ucb, 0.5)),
                               ((ucb, best_of(ucb)), (ts, best_of(ts))),
                               ((ts, best_of(ts)), (opt, best_of(opt))),
                               ((ucb, best_of(ucb)), (opt, best_of(opt))),
                               ((opt, best_of(opt)), (eps, best_of(eps)))]:
        m, se = paired(n1, v1, n2, v2)
        print(f"  {n1} [{v1:g}] - {n2} [{v2:g}] = {m:+.4f} ± {se:.4f}   ({abs(m) / se:.1f} s.e.)")
    print("ranking by best point:")
    for name, (sym, vals, scores) in sorted(table.items(), key=lambda kv: -kv[1][2].max()):
        print(f"  {scores.max():.3f}  {name} ({sym}={vals[np.argmax(scores)]:g})")

    if not args.quick:
        plt = setup_matplotlib()
        fig, ax = plt.subplots(figsize=(9, 5))
        markers = ["o", "s", "^", "D", "v", "P"]
        for i, (name, (sym, vals, scores)) in enumerate(table.items()):
            ax.plot(vals, scores, color=PALETTE[i], linestyle=LINESTYLES[i], marker=markers[i], ms=5,
                    lw=1.6, label=f"{name}  [{sym}]")
        ax.set_xscale("log", base=2)
        ticks = 2.0 ** np.arange(-7, 3)
        ax.set_xticks(ticks)
        ax.set_xticklabels(["1/128", "1/64", "1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "4"])
        ax.set(xlabel="parameter value (ε, α, c, Q₁, τ, s₀)  —  log scale",
               ylabel=f"Average reward over first {steps} steps",
               title=f"Parameter study on the 10-armed testbed ({runs} runs per point)")
        ax.legend(loc="lower center", ncol=2, fontsize=8.5)
        lo = min(s.min() for _, _, s in table.values())
        ax.set_ylim(max(lo - 0.05, 0.6), 1.55)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "parameter_study.png")
        fig.savefig(out)
        print(f"\nsaved {out}")
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
