"""Regret on a Bernoulli bandit: linear vs logarithmic regret, and the Lai-Robbins bound.

Chapter 02, Section 11.

Instance: k = 5 Bernoulli arms with means q* = (0.7, 0.6, 0.5, 0.4, 0.3), horizon T = 20,000.
Algorithms: greedy, eps-greedy (eps = 0.1), decaying eps_t = min(1, 5k/t), explore-then-commit
(m = 100 pulls per arm, an 'A/B/n test'), UCB1, KL-UCB and Beta-Bernoulli Thompson sampling.

We report
  * the expected pseudo-regret E[Reg(t)] = sum_t (v* - q*(A_t)) on a log time axis, with the
    Lai-Robbins asymptotic lower bound  sum_a Delta_a / kl(q*(a), v*) * ln t  (Eq. 11.6) and the
    UCB1 upper bound of Auer et al. (2002) (Eq. 11.8);
  * the regret decomposition lemma E[T v* - sum_t R_t] = sum_a Delta_a E[N_T(a)] (Eq. 11.3),
    checked with the REALIZED rewards;
  * pull counts of each suboptimal arm against the Lai-Robbins prediction ln T / kl(q*(a), v*).

Run:  python code/ch02_multi_armed_bandits/regret_bernoulli.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from bandits import (FIG_DIR, BernoulliBandit, EpsilonDecreasing, EpsilonGreedy, ExploreThenCommit,
                     KLUCB, ThompsonBeta, UCB, bernoulli_kl, run, setup_matplotlib, style)

PROBS = np.array([0.7, 0.6, 0.5, 0.4, 0.3])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    seed = 0
    k = len(PROBS)
    runs, T = (100, 2000) if args.quick else (1000, 20_000)
    v_star = PROBS.max()
    gaps = v_star - PROBS
    sub = gaps > 0
    kl = bernoulli_kl(PROBS[sub], v_star)
    lr_const = np.sum(gaps[sub] / kl)
    ucb1_bound = lambda t: 8 * np.log(t) * np.sum(1 / gaps[sub]) + (1 + np.pi ** 2 / 3) * gaps.sum()
    print(f"Seed {seed}; q* = {PROBS.tolist()}; {runs} runs x T = {T}")
    print(f"gaps Δ = {np.round(gaps, 3).tolist()}; kl(q*(a), v*) = {np.round(kl, 4).tolist()}")
    print(f"Lai-Robbins constant Σ Δ/kl = {lr_const:.2f}  ->  lower bound ≈ {lr_const * np.log(T):.1f} at T={T}")
    print(f"UCB1 bound (Auer et al. 2002, Thm 1) at T={T}: {ucb1_bound(T):.0f}")
    t0 = time.time()

    agents = [
        EpsilonGreedy(k, 0.0, name="greedy"),
        EpsilonGreedy(k, 0.1, name="ε-greedy ε=0.1"),
        EpsilonDecreasing(k, c=5.0, name="ε_t-greedy ε_t=min(1,5k/t)"),
        ExploreThenCommit(k, m=100, name="explore-then-commit m=100"),
        UCB(k, c=np.sqrt(2), name="UCB1"),
        KLUCB(k),
        ThompsonBeta(k, name="Thompson (Beta(1,1))"),
    ]
    results = []
    for i, ag in enumerate(agents):
        env = BernoulliBandit(PROBS, runs, np.random.default_rng(seed + 100 + i))
        res = run(env, ag, T, np.random.default_rng(seed + i))
        results.append(res)
        # Regret decomposition (Eq. 11.3): realized regret T v* - sum R_t vs sum_a Delta_a E[N_T(a)]
        realized = T * v_star - res.run_avg_reward * T
        decomposition = res.counts.mean(axis=0) @ gaps
        print(f"  {res.name:>27}: E[Reg(T)] = {res.regret_mean[-1]:8.1f} ± {res.regret_se[-1]:5.1f}"
              f" | realized {realized.mean():8.1f} ± {realized.std(ddof=1) / np.sqrt(runs):5.1f}"
              f" | Σ Δ E[N] = {decomposition:8.1f} | Reg/ln T = {res.regret_mean[-1] / np.log(T):6.1f}"
              f" | {res.seconds:5.1f}s")

    # Explore-then-commit check.  Its simulated regret is very noisy (a run either commits to the
    # best arm or pays Delta per step for ~T steps), so we also compute its expectation almost
    # exactly: exploration costs m * sum_a Delta_a, and after exploration the expected cost per step
    # is sum_a P(A_hat = a) Delta_a, where P(A_hat = a) is estimated from 200,000 simulated
    # exploration phases (binomial counts, ties at random) -- cheap, because no bandit is run.
    etc_i = next(i for i, ag in enumerate(agents) if isinstance(ag, ExploreThenCommit))
    m_etc = agents[etc_i].m
    etc_res = results[etc_i]
    wrong = np.mean(etc_res.counts.argmax(axis=1) != PROBS.argmax())   # the committed arm is the most pulled
    n_phase = 200_000
    phase_rng = np.random.default_rng(seed + 999)
    S = phase_rng.binomial(m_etc, PROBS, size=(n_phase, k)).astype(float)
    a_hat = np.argmax(S + 1e-3 * phase_rng.random((n_phase, k)), axis=1)        # random tie-break
    p_commit = np.bincount(a_hat, minlength=k) / n_phase
    exact_etc = lambda t: (np.minimum(t, m_etc * k) / k) * gaps.sum() + np.maximum(t - m_etc * k, 0) * (p_commit @ gaps)
    print(f"\nExplore-then-commit (m={m_etc}): {100 * wrong:.1f}% of runs committed to a suboptimal arm; "
          f"P(commit to each arm) from {n_phase} exploration phases = {np.round(p_commit, 4).tolist()} "
          f"(wrong: {1 - p_commit[0]:.3f})")
    print(f"  expected regret m ΣΔ + (T - mk) Σ_a P(a) Δ_a = {exact_etc(T):.1f} at T (simulated "
          f"{etc_res.regret_mean[-1]:.1f} ± {etc_res.regret_se[-1]:.1f}); growth T/10 -> T: "
          f"×{exact_etc(T) / exact_etc(T // 10):.2f}")

    # Growth check: ratio of regret at T and at T/10 (log regret: ratio ~ ln T / ln(T/10); linear: ~10)
    print(f"\nGrowth from t = T/10 to t = T (linear regret -> ×10, logarithmic -> ×{np.log(T) / np.log(T / 10):.2f}):")
    for r in results:
        print(f"  {r.name:>27}: ×{r.regret_mean[-1] / r.regret_mean[T // 10 - 1]:.2f}")
    print("\nMean pulls of each suboptimal arm at T vs the Lai-Robbins rate ln T / kl:")
    print(f"  {'':>27}  " + "  ".join(f"q*={q:.1f}" for q in PROBS[sub]))
    print(f"  {'ln T / kl':>27}: " + "  ".join(f"{np.log(T) / d:6.0f}" for d in kl))
    for r in results[4:]:
        print(f"  {r.name:>27}: " + "  ".join(f"{n:6.0f}" for n in r.counts.mean(axis=0)[sub]))

    if not args.quick:
        plt = setup_matplotlib()
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.25, 1.25, 1]})
        t = np.arange(1, T + 1)
        for i, r in enumerate(results):
            for a_ in ax[:2]:
                a_.plot(t, r.regret_mean, label=r.name, **style(i), lw=1.6)
            ax[0].fill_between(t, r.regret_mean - 2 * r.regret_se, r.regret_mean + 2 * r.regret_se,
                               color=style(i)["color"], alpha=0.15, lw=0)
        for j, a_ in enumerate(ax[:2]):
            tt = t if j == 0 else t[t >= 10]          # ln t -> 0 at t = 1 looks odd on log-log axes
            a_.plot(tt, lr_const * np.log(tt), color="#0b0b0b", lw=1.2, ls=(0, (2, 2)),
                    label="Lai–Robbins (asymptotic) Σ Δ/kl · ln t")
            a_.set_xscale("log")
            a_.set_xlabel("t (log scale)")
        ax[1].plot(t, ucb1_bound(t), color="#8a8984", lw=1.2, ls=(0, (6, 3)), label="UCB1 upper bound (Auer et al. 2002)")
        ax[1].plot(t, 0.5 * t * gaps.mean(), color="#b5b4ae", lw=1.0, ls="-",
                   label="slope 1 (linear regret) reference")
        ax[0].set_ylim(0, 500)
        ax[0].set(ylabel="expected cumulative regret", title=f"(a) Regret, log time axis ({runs} runs, ±2 s.e.)")
        ax[1].set_yscale("log")
        ax[1].set_ylim(0.5, 2e4)
        ax[1].set(title="(b) Same curves on log-log axes (colours as in (a))")
        ax[0].legend(loc="upper left", fontsize=8)
        ax[1].legend(loc="upper left", fontsize=7.5, handles=ax[1].get_legend_handles_labels()[0][-3:])

        width = 0.2
        xs = np.arange(sub.sum())
        ax[2].bar(xs - 1.5 * width, np.log(T) / kl, width, color="#0b0b0b", label="Lai–Robbins ln T / kl")
        for j, r in enumerate(results[4:]):
            ax[2].bar(xs + (j - 0.5) * width, r.counts.mean(axis=0)[sub], width,
                      color=style(4 + j)["color"], label=r.name)
        ax[2].set_xticks(xs)
        ax[2].set_xticklabels([f"q*={q:.1f}\nΔ={g:.1f}" for q, g in zip(PROBS[sub], gaps[sub])])
        ax[2].set(ylabel=f"mean pulls N_T(a) at T={T}", title="(c) Pulls of each suboptimal arm")
        ax[2].legend(fontsize=8)
        ax[2].grid(axis="x", visible=False)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "regret_bernoulli.png")
        fig.savefig(out)
        print(f"\nsaved {out}")
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
