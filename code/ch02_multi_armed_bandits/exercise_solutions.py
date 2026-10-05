"""Numerical solutions to the coding exercises of Chapter 02.

Exercise 2.9  (explore-then-commit): exact regret of ETC on two Bernoulli arms, computed from
              binomial distributions (no simulation), against the Hoeffding bound
              m Delta + (T - 2m) Delta exp(-m Delta^2 / 2) and the bound-optimal m.
Exercise 2.13 (nonstationary parameter study, S&B Exercise 2.11 shortened): random-walk bandit,
              average reward over the second half of 20,000 steps for eps-greedy / UCB /
              gradient bandit with constant step sizes, and eps-greedy with sample averages.
Exercise 2.14 (forgetting for Bayesian / UCB methods): discounted Gaussian Thompson sampling
              (statistics decayed by gamma each step) vs constant-alpha eps-greedy and plain
              (stationary) Thompson sampling on the same random-walk bandit.

Run:  python code/ch02_multi_armed_bandits/exercise_solutions.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from bandits import (FIG_DIR, Agent, EpsilonGreedy, GaussianBandit, GradientBandit, ThompsonGaussian, UCB,
                     etc_exact_regret_two_arms, run, setup_matplotlib, style)


# ------------------------------------------------------------------ Exercise 2.9
def etc_exact_regret(p1, p2, m, T):
    """Two Bernoulli arms, p1 > p2.  Pull each m times, commit to the higher empirical mean
    (ties at random).  Exact expected regret = m Delta + (T - 2m) Delta P(commit to arm 2).
    (The implementation lives in bandits.py, where regret_vs_gap.py uses it too.)"""
    return etc_exact_regret_two_arms(p1, p2, m, T)


def exercise_etc(quick):
    p1, p2, T = 0.6, 0.5, 10_000
    gap = p1 - p2
    ms = np.unique(np.geomspace(1, 2000, 40 if quick else 120).astype(int))
    exact = np.array([etc_exact_regret(p1, p2, m, T) for m in ms])
    bound = ms * gap + (T - 2 * ms) * gap * np.exp(-ms * gap ** 2 / 2)
    bound_fn = lambda m: m * gap + (T - 2 * m) * gap * np.exp(-m * gap ** 2 / 2)       # Eq. (11.9)
    # Closed-form APPROXIMATE minimizer of the bound (it replaces T - 2m by T; see the solution):
    m_bound = int(np.ceil(2 / gap ** 2 * np.log(T * gap ** 2 / 2)))
    m_grid = np.arange(1, T // 2)
    m_bound_exact = int(m_grid[np.argmin(bound_fn(m_grid))])                               # exact integer minimizer
    print(f"Exercise 2.9: ETC, Bernoulli({p1}) vs Bernoulli({p2}), T = {T}")
    m_all = np.arange(1, 2001)
    exact_all = np.array([etc_exact_regret(p1, p2, int(m), T) for m in m_all])
    j = int(np.argmin(exact_all))
    print(f"  exact-optimal m = {m_all[j]}, regret {exact_all[j]:.1f}")
    print(f"  closed-form bound-optimal m* ≈ {m_bound}: exact regret there {etc_exact_regret(p1, p2, m_bound, T):.1f}, "
          f"bound value {bound_fn(m_bound):.1f}")
    print(f"  exact minimizer of the bound (11.9): m = {m_bound_exact}, bound value {bound_fn(m_bound_exact):.2f}, "
          f"exact regret there {etc_exact_regret(p1, p2, m_bound_exact, T):.1f}")
    for m in (10, 50, 100, 300, 1000):
        print(f"  m = {m:5d}: exact regret {etc_exact_regret(p1, p2, m, T):7.1f}   Hoeffding bound "
              f"{m * gap + (T - 2 * m) * gap * np.exp(-m * gap ** 2 / 2):7.1f}")
    return ms, exact, bound, m_bound


# ------------------------------------------------------------------ Exercise 2.13
def exercise_nonstationary_study(quick):
    runs, steps = (50, 2000) if quick else (150, 20_000)
    families = {
        "ε-greedy, α=0.1": ("ε", range(-7, -1), lambda v: EpsilonGreedy(10, v, alpha=0.1)),
        "ε-greedy, sample avg.": ("ε", range(-7, -1), lambda v: EpsilonGreedy(10, v)),
        "UCB, α=0.1": ("c", range(-2, 5), lambda v: UCB(10, c=v, alpha=0.1)),
        "gradient bandit": ("α", range(-11, -1), lambda v: GradientBandit(10, alpha=v)),
    }
    print(f"\nExercise 2.13: nonstationary parameter study, {runs} runs x {steps} steps, "
          f"score = mean reward over the last {steps // 2} steps")
    table = {}
    for f, (name, (sym, exps, make)) in enumerate(families.items()):
        vals, scores = [], []
        for i, e in enumerate(exps):
            env = GaussianBandit(np.zeros((runs, 10)), np.random.default_rng(1), walk_std=0.01)
            res = run(env, make(2.0 ** e), steps, np.random.default_rng(100 * f + i))
            vals.append(2.0 ** e)
            scores.append(res.avg_reward[steps // 2:].mean())
        table[name] = (sym, np.array(vals), np.array(scores))
        b = int(np.argmax(scores))
        print(f"  {name:>22}: " + "  ".join(f"{sym}=2^{e}:{s:.3f}" for e, s in zip(exps, scores))
              + f"  -> best {sym}={vals[b]:g} ({scores[b]:.3f})")
    return table


# ------------------------------------------------------------------ Exercise 2.14
class DiscountedThompsonGaussian(Agent):
    """Gaussian TS whose sufficient statistics forget: every step N <- gamma N, S <- gamma S,
    then the new observation is added.  The posterior then reflects ~1/(1-gamma) recent pulls."""

    def __init__(self, k, gamma=0.99, s0=1.0, sigma=1.0, name=None):
        super().__init__(k)
        self.gamma, self.s0, self.sigma = gamma, s0, sigma      # gamma = forgetting factor (γ_f in the text)
        self.name = name or f"discounted Thompson γ_f={gamma}"

    def reset(self, runs, rng):
        super().reset(runs, rng)
        self.N = np.zeros((runs, self.k))
        self.S = np.zeros((runs, self.k))

    def act(self, t):
        prec = 1 / self.s0 ** 2 + self.N / self.sigma ** 2
        mean = (self.S / self.sigma ** 2) / prec
        return np.argmax(mean + self.rng.normal(size=mean.shape) / np.sqrt(prec), axis=1)

    def update(self, a, r):
        self.N *= self.gamma
        self.S *= self.gamma
        self.N[self.rows, a] += 1
        self.S[self.rows, a] += r


class SecondHalfRecorder:
    """Wraps an environment and accumulates each run's rewards over the second half of the steps
    (run() only returns per-run averages over ALL steps)."""

    def __init__(self, env, steps):
        self.env, self.steps, self.t = env, steps, 0
        self.total = np.zeros(env.runs)

    @property
    def q_star(self):
        return self.env.q_star

    def pull(self, a):
        r = self.env.pull(a)
        self.t += 1
        if self.t > self.steps // 2:
            self.total += r
        return r


def exercise_discounted_ts(quick):
    runs, steps = (50, 2000) if quick else (500, 10_000)
    agents = [EpsilonGreedy(10, 0.1, alpha=0.1, name="ε-greedy ε=0.1, α=0.1"),
              ThompsonGaussian(10, name="Thompson (stationary)")] + \
             [DiscountedThompsonGaussian(10, g) for g in (0.99, 0.995, 0.998, 0.999, 0.9995)] + \
             [EpsilonGreedy(10, e, alpha=0.1, name=f"ε-greedy ε=1/{round(1 / e)}, α=0.1") for e in (1 / 32, 1 / 16)]
    # (The last two rows give constant-α ε-greedy a tuned ε -- Exercise 2.13 finds small ε best on
    #  this drift -- so that the comparison with the tuned forgetting factor is fair.)
    print(f"\nExercise 2.14: discounted Thompson sampling on the random-walk bandit, {runs} runs x {steps} steps")
    print("  (every agent sees the same drift and reward noise: the environment's random stream does not")
    print("   depend on the actions, so paired differences between agents are precise)")
    results, second_half = [], []
    for i, ag in enumerate(agents):
        env = SecondHalfRecorder(GaussianBandit(np.zeros((runs, 10)), np.random.default_rng(2), walk_std=0.01), steps)
        r = run(env, ag, steps, np.random.default_rng(300 + i))
        results.append(r)
        second_half.append(env.total / (steps - steps // 2))
        print(f"  {r.name:>28}: mean reward, 2nd half {r.avg_reward[steps // 2:].mean():.3f} "
              f"± {second_half[-1].std(ddof=1) / np.sqrt(runs):.3f}; % optimal, 2nd half "
              f"{100 * r.pct_optimal[steps // 2:].mean():.1f}%")
    names = [r.name for r in results]
    i_ts = names.index("discounted Thompson γ_f=0.999")
    for j in (0, len(names) - 2, len(names) - 1):
        d = second_half[i_ts] - second_half[j]
        print(f"  paired difference [{names[i_ts]}] - [{names[j]}] = {d.mean():+.3f} ± {d.std(ddof=1) / np.sqrt(runs):.3f}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    print(f"Seeds fixed per experiment (see code); quick={args.quick}")
    t0 = time.time()
    ms, exact, bound, m_bound = exercise_etc(args.quick)
    table = exercise_nonstationary_study(args.quick)
    results = exercise_discounted_ts(args.quick)

    if not args.quick:
        plt = setup_matplotlib()
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        ax[0].plot(ms, exact, label="exact E[Reg(T)]", **style(0))
        ax[0].plot(ms, bound, label="Hoeffding bound mΔ + (T−2m)Δ exp(−mΔ²/2)", **style(1))
        ax[0].axvline(m_bound, color="#8a8984", lw=0.8, ls=":")
        ax[0].set_xscale("log")
        ax[0].set_ylim(0, 400)
        ax[0].set(xlabel="exploration pulls per arm m (log scale)", ylabel="regret at T = 10,000",
                  title="Ex. 2.9: explore-then-commit, Δ = 0.1")
        ax[0].legend(fontsize=8)
        markers = ["o", "s", "^", "D"]
        for i, (name, (sym, vals, scores)) in enumerate(table.items()):
            ax[1].plot(vals, scores, marker=markers[i], ms=4.5, label=f"{name} [{sym}]", **style(i))
        ax[1].set_xscale("log", base=2)
        ax[1].set(xlabel="parameter (log₂ scale)", ylabel="mean reward, second half",
                  title="Ex. 2.13: nonstationary parameter study")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "exercise_solutions.png")
        fig.savefig(out)
        print(f"\nsaved {out}")
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
