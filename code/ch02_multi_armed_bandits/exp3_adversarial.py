"""Adversarial bandits: why deterministic algorithms fail, and EXP3 (Chapter 02, Section 12).

Part A: an oblivious adversary that defeats UCB1.
  UCB1 (with a fixed tie-breaking rule) is a DETERMINISTIC function of the reward history.
  An adversary who knows the algorithm can therefore simulate it in advance and write down a
  reward table x_t(a) in {0, 1} with x_t(A_t^UCB1) = 0 and x_t(a) = 1 for every other arm.
  The table is fixed before the game starts (an oblivious adversary), yet UCB1 earns 0 while
  the best fixed arm earns at least T (k-1)/k.  We then run randomized algorithms (EXP3,
  Thompson sampling, eps-greedy) on the SAME table: their regret against the best fixed arm
  in hindsight is what the EXP3 theory bounds by sqrt(2 T k ln k) (Eq. 12.4).

Part A2: a regime switch (also an oblivious, fixed table; k = 2).
  Arm 1 pays 1 for the first T/3 steps and 0 afterwards; arm 2 pays the opposite.  The best
  fixed arm (arm 2) earns 2T/3, but a policy that SWITCHES arms at t = T/3 earns T.  So regret
  against the best fixed arm can be negative: it is a weak benchmark when the world changes.
  EXP3 only promises not to fall far behind the best fixed arm; it switches only once the
  cumulative (estimated) losses cross, whereas UCB1 and Thompson sampling switch much sooner
  here.  (Tracking a changing best arm needs other tools: Section 12.3.)

Part B: the price of robustness on a stochastic problem.
  On the 5-armed Bernoulli instance of regret_bernoulli.py, EXP3 (eta = sqrt(2 ln k / (T k)), the
  theoretical value for each horizon T, which requires knowing T) pays
  ~sqrt(T) regret, much more than the log T of UCB1 / Thompson sampling.

Run:  python code/ch02_multi_armed_bandits/exp3_adversarial.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from bandits import (FIG_DIR, BernoulliBandit, EXP3, EpsilonGreedy, ThompsonBeta, UCB, run,
                     sample_categorical, setup_matplotlib, softmax, style)


def adversarial_table_against_ucb1(k, T):
    """Simulate deterministic UCB1 (ties -> lowest index) and give it reward 0 every step."""
    Q, N = np.zeros(k), np.zeros(k)
    table = np.ones((T, k))
    ucb_actions = np.zeros(T, dtype=int)
    for t in range(1, T + 1):
        if (N == 0).any():
            a = int(np.argmax(N == 0))
        else:
            a = int(np.argmax(Q + np.sqrt(2 * np.log(t) / N)))
        table[t - 1, a] = 0.0                  # the arm UCB1 is about to pull pays nothing
        r = table[t - 1, a]
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        ucb_actions[t - 1] = a
    return table, ucb_actions


def play_table(agent, table, runs, rng):
    """Run an agent (vectorized over runs) against a fixed reward table; return per-run
    regret curves against the best fixed arm in hindsight (Eq. 12.1)."""
    T, k = table.shape
    agent.reset(runs, rng)
    earned = np.zeros((runs, T))
    for t in range(1, T + 1):
        a = agent.act(t)
        r = table[t - 1, a]
        agent.update(a, r)
        earned[:, t - 1] = r
    best_fixed = np.cumsum(table, axis=0).max(axis=1)        # best fixed arm for each prefix
    return best_fixed[None, :] - np.cumsum(earned, axis=1)   # regret curve against it


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    seed = 0
    k, T = 3, (2000 if args.quick else 10_000)
    runs = 50 if args.quick else 300
    print(f"Seed {seed}; Part A: k={k}, T={T}, {runs} runs of each randomized algorithm")
    t0 = time.time()

    # ---------------- Part A ----------------
    table, ucb_actions = adversarial_table_against_ucb1(k, T)
    totals = table.sum(axis=0)
    print(f"  adversary's table: total reward of each fixed arm = {totals.astype(int).tolist()}; "
          f"UCB1's first 12 actions: {ucb_actions[:12].tolist()}")
    print(f"  UCB1 earns 0 -> regret vs best fixed arm = {totals.max():.0f} = {totals.max() / T:.3f} T (linear)")
    bound = np.sqrt(2 * T * k * np.log(k))
    algos = [
        EXP3(k, T, name="EXP3"),
        ThompsonBeta(k, name="Thompson (Beta)"),
        EpsilonGreedy(k, 0.1, name="ε-greedy ε=0.1"),
    ]
    curves = {}
    for i, ag in enumerate(algos):
        reg = play_table(ag, table, runs, np.random.default_rng(seed + i))
        curves[ag.name] = reg
        fin = reg[:, -1]
        print(f"  {ag.name:>16}: E[regret vs best fixed arm] at T = {fin.mean():8.1f} ± "
              f"{fin.std(ddof=1) / np.sqrt(runs):5.1f}   (max over runs {fin.max():.0f})")
    print(f"  EXP3 bound sqrt(2 T k ln k) = {bound:.1f}")
    # The UCB1 curve, for the plot (deterministic)
    best_fixed = np.cumsum(table, axis=0).max(axis=1)
    curves["UCB1 (deterministic)"] = best_fixed[None, :]

    # ---------------- Part A2 ----------------
    switch = T // 3
    table2 = np.zeros((T, 2))
    table2[:switch, 0] = 1.0
    table2[switch:, 1] = 1.0
    print(f"\nPart A2: regime switch at t = {switch} (k = 2); best fixed arm earns {table2.sum(axis=0).max():.0f}")
    curves2 = {}
    for i, ag in enumerate([UCB(2, np.sqrt(2), name="UCB1"), EXP3(2, T, name="EXP3"),
                            ThompsonBeta(2, name="Thompson (Beta)"), EpsilonGreedy(2, 0.1, name="ε-greedy ε=0.1")]):
        reg = play_table(ag, table2, runs, np.random.default_rng(seed + 20 + i))
        curves2[ag.name] = reg
        fin = reg[:, -1]
        print(f"  {ag.name:>16}: E[regret vs best fixed arm] at T = {fin.mean():8.1f} ± "
              f"{fin.std(ddof=1) / np.sqrt(runs):5.1f}  ({fin.mean() / T:.3f} T)")
    print(f"  EXP3 bound sqrt(2 T k ln k) = {np.sqrt(2 * T * 2 * np.log(2)):.1f}")

    # ---------------- Part B ----------------
    probs = np.array([0.7, 0.6, 0.5, 0.4, 0.3])
    kB = len(probs)
    horizons = [500, 2000] if args.quick else [2500, 10_000, 40_000]
    TB = horizons[-1]
    runsB = 50 if args.quick else 300
    print(f"\nPart B: stochastic Bernoulli q* = {probs.tolist()}, horizons {horizons}, {runsB} runs")
    resB = {}
    for i, ag in enumerate([UCB(kB, np.sqrt(2), name="UCB1"), ThompsonBeta(kB, name="Thompson (Beta)")]):
        env = BernoulliBandit(probs, runsB, np.random.default_rng(seed + 50 + i))
        resB[ag.name] = run(env, ag, TB, np.random.default_rng(seed + 60 + i))   # anytime algorithms
    exp3_final = {}
    for j, Th in enumerate(horizons):                     # EXP3's eta depends on the horizon
        env = BernoulliBandit(probs, runsB, np.random.default_rng(seed + 70 + j))
        r = run(env, EXP3(kB, Th, name=f"EXP3 (η set for T={Th})"), Th, np.random.default_rng(seed + 80 + j))
        exp3_final[Th] = (r.regret_mean[-1], r.regret_se[-1])
        if Th == TB:
            resB["EXP3"] = r
    print(f"  {'T':>6} | {'EXP3 (η set for T)':>18} | {'UCB1':>13} | {'Thompson':>13} | {'EXP3 bound √(2Tk ln k)':>22}")
    for Th in horizons:
        u, ts = resB["UCB1"], resB["Thompson (Beta)"]
        print(f"  {Th:6d} | {exp3_final[Th][0]:9.1f} ± {exp3_final[Th][1]:4.1f} | "
              f"{u.regret_mean[Th - 1]:6.1f} ± {u.regret_se[Th - 1]:4.1f} | "
              f"{ts.regret_mean[Th - 1]:6.1f} ± {ts.regret_se[Th - 1]:4.1f} | {np.sqrt(2 * Th * kB * np.log(kB)):22.1f}")
    for name, vals in [("EXP3", [exp3_final[Th][0] for Th in horizons]),
                       ("UCB1", [resB["UCB1"].regret_mean[Th - 1] for Th in horizons]),
                       ("Thompson", [resB["Thompson (Beta)"].regret_mean[Th - 1] for Th in horizons])]:
        print(f"  growth per ×4 in T, {name:>8}: " + ", ".join(f"×{b / a:.2f}" for a, b in zip(vals[:-1], vals[1:]))
              + f"   (√T -> ×2.00, ln T -> ×{np.log(horizons[1]) / np.log(horizons[0]):.2f})")

    if not args.quick:
        plt = setup_matplotlib()
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
        t = np.arange(1, T + 1)
        names = ["UCB1 (deterministic)", "EXP3", "Thompson (Beta)", "ε-greedy ε=0.1"]
        colors = {"UCB1 (deterministic)": 4, "UCB1": 4, "EXP3": 7, "Thompson (Beta)": 6, "ε-greedy ε=0.1": 1}
        for name in names:
            m = curves[name].mean(axis=0)
            ax[0].plot(t, m, label=name, **style(colors[name]), lw=1.6)
        ax[0].plot(t, np.sqrt(2 * t * k * np.log(k)), color="#8a8984", ls=(0, (6, 3)), lw=1.2,
                   label="EXP3 bound √(2tk ln k)")
        ax[0].set(xlabel="t", ylabel="regret vs best fixed arm",
                  title=f"(a) Oblivious adversary built against UCB1 (k={k})")
        ax[0].legend(loc="upper left", fontsize=8)
        # Zoomed inset: the randomized algorithms' regret (tens) is invisible on the 0..6667 scale.
        ins = ax[0].inset_axes([0.55, 0.07, 0.42, 0.36])
        for name in names[1:]:
            ins.plot(t, curves[name].mean(axis=0), **style(colors[name]), lw=1.3)
        ins.set_ylim(0, 1.25 * max(curves[n].mean(axis=0).max() for n in names[1:]))
        ins.set_title("zoom: randomized algorithms", fontsize=8)
        ins.tick_params(labelsize=7)
        for name, reg in curves2.items():
            ax[1].plot(t, reg.mean(axis=0), label=name, **style(colors[name]), lw=1.6)
        ax[1].axvline(switch, color="#8a8984", lw=0.8, ls=":")
        ax[1].axhline(0, color="#52514e", lw=0.8)
        ax[1].set(xlabel="t", ylabel="regret vs best fixed arm",
                  title=f"(b) Regime switch at t={switch} (k=2): regret can be < 0")
        ax[1].legend(loc="lower left", fontsize=8)
        ax = [ax[0], ax[2]]
        tB = np.arange(1, TB + 1)
        for name, c in (("EXP3", 7), ("UCB1", 4), ("Thompson (Beta)", 6)):
            r = resB[name]
            lab = f"EXP3 (η set for T={TB})" if name == "EXP3" else name
            ax[1].plot(tB, r.regret_mean, label=lab, **style(c), lw=1.6)
            ax[1].fill_between(tB, r.regret_mean - 2 * r.regret_se, r.regret_mean + 2 * r.regret_se,
                               color=style(c)["color"], alpha=0.15, lw=0)
        ax[1].plot(horizons, [exp3_final[Th][0] for Th in horizons], "o", color=style(7)["color"], ms=6,
                   label="EXP3 final regret, η = √(2 ln k/(Tk)) for each T")
        ax[1].set(xlabel="t", ylabel="expected pseudo-regret",
                  title=f"(c) Stochastic Bernoulli bandit (k={kB}, {runsB} runs)")
        ax[1].legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "exp3_adversarial.png")
        fig.savefig(out)
        print(f"\nsaved {out}")
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
