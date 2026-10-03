"""Best-arm identification (pure exploration) on a 10-armed Bernoulli bandit (Chapter 02, Section 15).

Goal: after a budget of n pulls, RECOMMEND one arm; we only care whether it is the best arm
(probability of error) or how much worse it is (simple regret), not about rewards collected
along the way.  Instance: q* = (0.6, 0.55, 0.5, 0.5, 0.45, 0.45, 0.4, 0.4, 0.35, 0.3).

Fixed-budget methods compared:
  * uniform allocation       n/k pulls per arm, recommend the empirical best
  * sequential halving       (Karnin, Koren & Somekh 2013) ceil(log2 k) rounds; each round splits
                             its share of the budget equally over the surviving arms and keeps
                             the better half (Algorithm 15.2)
  * UCB1, Thompson sampling  regret minimizers; recommend the most-pulled arm
Fixed-confidence method:
  * successive elimination   (Even-Dar, Mannor & Mansour 2006) sample all surviving arms once per
                             round; drop arm a when its upper confidence bound falls below the
                             best lower confidence bound; stop with one arm.  Radius from
                             Hoeffding + a union bound: sqrt(ln(4 k n^2 / delta) / (2 n)).

Run:  python code/ch02_multi_armed_bandits/best_arm_identification.py [--quick]
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

from bandits import (FIG_DIR, BernoulliBandit, ThompsonBeta, UCB, argmax_random_ties, run, setup_matplotlib,
                     style)

PROBS = np.array([0.6, 0.55, 0.5, 0.5, 0.45, 0.45, 0.4, 0.4, 0.35, 0.3])
GAPS = PROBS.max() - PROBS


def uniform(n, runs, rng, tie_rng):
    m = n // len(PROBS)
    means = rng.binomial(m, PROBS, size=(runs, len(PROBS))) / m
    # Ties between binomial counts are COMMON (e.g. 60 vs 60 successes), and np.argmax would
    # resolve them in favour of index 0 -- which is the best arm here -- biasing the error rate
    # downward.  Break ties at random (a separate stream, so the other methods' draws are unchanged).
    rec = argmax_random_ties(means, tie_rng)
    return rec, m * GAPS.sum()


def sequential_halving(n, runs, rng):
    k = len(PROBS)
    rounds = math.ceil(math.log2(k))
    alive = np.ones((runs, k), dtype=bool)
    cum_regret = 0.0
    for _ in range(rounds):
        n_alive = alive[0].sum()                     # same for every run
        t_r = n // (n_alive * rounds)
        means = rng.binomial(t_r, PROBS, size=(runs, k)) / max(t_r, 1)
        cum_regret += t_r * (alive * GAPS).sum(axis=1).mean()      # regret paid this round
        keep = math.ceil(n_alive / 2)
        score = np.where(alive, means + 1e-9 * rng.random((runs, k)), -np.inf)   # random tie-break
        order = np.argsort(-score, axis=1)[:, :keep]
        alive = np.zeros_like(alive)
        np.put_along_axis(alive, order, True, axis=1)
        if keep == 1:
            break
    return alive.argmax(axis=1), cum_regret


def regret_minimizer(agent, n, runs, seed):
    env = BernoulliBandit(PROBS, runs, np.random.default_rng(seed))
    res = run(env, agent, n, np.random.default_rng(seed + 1))
    return res.counts.argmax(axis=1), res.regret_mean[-1]


def successive_elimination(delta, runs, rng, max_rounds=200_000):
    k = len(PROBS)
    alive = np.ones((runs, k), dtype=bool)
    sums = np.zeros((runs, k))
    samples = np.zeros(runs)
    done_at = np.full(runs, -1)
    for n in range(1, max_rounds + 1):
        active_runs = done_at < 0
        if not active_runs.any():
            break
        pull = alive & active_runs[:, None]
        sums += pull * (rng.random((runs, k)) < PROBS)
        samples += pull.sum(axis=1)
        means = sums / n
        rad = math.sqrt(math.log(4 * k * n * n / delta) / (2 * n))
        best_lcb = np.where(alive, means, -np.inf).max(axis=1) - rad
        alive &= ~((means + rad < best_lcb[:, None]) & active_runs[:, None])
        finished = active_runs & (alive.sum(axis=1) == 1)
        done_at[finished] = n
    return alive.argmax(axis=1), samples, done_at


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    seed = 0
    budgets = [200, 1000] if args.quick else [200, 500, 1000, 2000, 5000, 10000, 20000]
    runs = 200 if args.quick else 1000
    print(f"Seed {seed}; q* = {PROBS.tolist()}; budgets {budgets}; {runs} runs")
    t0 = time.time()
    rng = np.random.default_rng(seed)
    tie_rng = np.random.default_rng(seed + 31)
    methods = ["uniform", "sequential halving", "UCB1 (most pulled)", "Thompson (most pulled)"]
    perr = {m: [] for m in methods}
    sreg = {m: [] for m in methods}
    creg = {m: [] for m in methods}
    for n in budgets:
        out = {
            "uniform": uniform(n, runs, rng, tie_rng),
            "sequential halving": sequential_halving(n, runs, rng),
            "UCB1 (most pulled)": regret_minimizer(UCB(len(PROBS), np.sqrt(2)), n, runs, seed + n),
            "Thompson (most pulled)": regret_minimizer(ThompsonBeta(len(PROBS)), n, runs, seed + 2 * n),
        }
        for m, (rec, cr) in out.items():
            perr[m].append(np.mean(rec != 0))
            sreg[m].append(GAPS[rec].mean())
            creg[m].append(cr)
    print(f"\n{'budget n':>9} | " + " | ".join(f"{m:>22}" for m in methods))
    print("P(recommended arm is not the best):")
    for j, n in enumerate(budgets):
        print(f"{n:9d} | " + " | ".join(f"{perr[m][j]:22.3f}" for m in methods))
    print("simple regret E[Δ of recommended arm]:")
    for j, n in enumerate(budgets):
        print(f"{n:9d} | " + " | ".join(f"{sreg[m][j]:22.4f}" for m in methods))
    print("cumulative regret paid DURING the budget:")
    for j, n in enumerate(budgets):
        print(f"{n:9d} | " + " | ".join(f"{creg[m][j]:22.1f}" for m in methods))

    delta = 0.05
    se_runs = 50 if args.quick else 200
    rec, samples, done_at = successive_elimination(delta, se_runs, np.random.default_rng(seed + 7),
                                                   max_rounds=2000 if args.quick else 200_000)
    finished = done_at > 0
    print(f"\nSuccessive elimination, δ = {delta}, {se_runs} runs: finished {finished.mean():.0%}", end="")
    if finished.any():
        print(f"; error rate {np.mean(rec[finished] != 0):.3f} (guarantee ≤ {delta}); "
              f"samples used: mean {samples[finished].mean():.0f}, median {np.median(samples[finished]):.0f}, "
              f"max {samples[finished].max():.0f}")
    else:
        print(" (round cap reached in --quick mode)")
    H = np.sum(1 / GAPS[1:] ** 2)
    print(f"problem complexity H_1 = Σ 1/Δ² = {H:.0f} (sample complexity scales like H_1 log(·/δ))")

    if not args.quick:
        plt = setup_matplotlib()
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        floor = 0.5 / runs                               # 0 errors in `runs` runs is drawn at this floor
        for i, m in enumerate(methods):
            ax[0].plot(budgets, np.maximum(perr[m], floor), marker="o", label=m, **style(i))
            ax[1].plot(creg[m], perr[m], marker="o", label=m, **style(i))
            for n, x, y in zip(budgets, creg[m], perr[m]):
                if m == "uniform":
                    ax[1].annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7,
                                   color="#52514e")
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].axhline(floor, color="#8a8984", lw=0.8, ls=":")
        ax[0].text(budgets[0], floor * 1.15, "0 errors in all runs", fontsize=7.5, color="#52514e")
        ax[0].set(xlabel="budget n (log scale)", ylabel="P(wrong arm recommended), log scale",
                  ylim=(floor * 0.7, 1), title=f"(a) Fixed-budget best-arm identification ({runs} runs)")
        ax[0].legend(fontsize=8, loc="upper right")
        ax[1].set_xscale("log")
        ax[1].set(xlabel="cumulative regret paid during the budget (log scale)", ylabel="P(wrong arm recommended)",
                  ylim=(0, 1), title="(b) Identification vs earning: the trade-off")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "best_arm_identification.png")
        fig.savefig(out)
        print(f"\nsaved {out}")
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
