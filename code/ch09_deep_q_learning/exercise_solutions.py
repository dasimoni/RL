"""Numerical checks for the Chapter 09 exercises (the numbers quoted in the solutions).

  Ex 9.2   n-step targets with truncation vs termination, via dqn.NStepAccumulator
  Ex 9.4   human-normalized scores and their aggregates
  Ex 9.5   E[max of m iid U(-b, b)] = b (m-1)/(m+1), by simulation
  Ex 9.6   target refreshes needed to reach 90% / 99% of 1/(1-gamma), 1-step and 3-step
  Ex 9.7   mean lag of Polyak averaging vs hard copies
  Ex 9.8   fixed points of prioritized replay with partial IS correction
  Ex 9.10  quantiles of a discrete distribution, by minimising the quantile loss on a grid
  Ex 9.12  the inverse of R2D2's value rescaling h
  Ex 9.13  3-step DQN on CartPole: value growth vs the prediction (1 - gamma^(3k)) / (1 - gamma)
  Ex 9.14  Polyak averaging (tau = 0.008) vs hard target copies every 250 steps
Ex 9.13 and 9.14 train 5 seeds each with dqn_batched.py (full mode only; --quick skips them).

Run:  python code/ch09_deep_q_learning/exercise_solutions.py [--quick]
"""
from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np
from scipy.optimize import brentq

from dqn import Config, NStepAccumulator

SEED = 0


def ex_9_2():
    print("Ex 9.2  3-step targets, gamma = 0.9, reward 1 per step, episode S0..S4 ending at step 4")
    q_max = {3: 5.0, 4: 4.0}                         # max_a q(S_k, a; w^-) for the bootstrap states
    for ended_by in ("truncated", "terminated"):
        acc = NStepAccumulator(3, 0.9)
        out = []
        for t in range(4):
            last = t == 3
            out += acc.push(f"S{t}", 0, 1.0, f"S{t + 1}", last and ended_by == "terminated", last)
        ys = []
        for (s, a, g, s2, term, disc) in out:
            y = g + disc * (1 - term) * q_max[int(s2[1])]
            ys.append(f"{s}: G={g:.2f}, boot {s2} x {disc:.3f} x (1-term={1 - term:.0f}) -> y={y:.3f}")
        print(f"  {ended_by}:\n    " + "\n    ".join(ys))


def ex_9_4():
    games = {"A": (0, 100, 250), "B": (-20, 10, -5), "C": (100, 1100, 1300), "D": (10, 20, 9)}
    hns = np.array([(ag - r) / (h - r) for r, h, ag in games.values()])
    print(f"Ex 9.4  HNS per game {np.round(hns, 3).tolist()}; mean {hns.mean():.3f}; median {np.median(hns):.3f}")


def ex_9_5(rng):
    print("Ex 9.5  E[max of m iid U(-1,1)] vs (m-1)/(m+1):",
          ", ".join(f"m={m}: {rng.uniform(-1, 1, (200_000, m)).max(1).mean():.4f} vs {(m - 1) / (m + 1):.4f}"
                    for m in (2, 4, 18)))


def ex_9_6():
    g = 0.99
    for frac in (0.9, 0.99):
        k1 = np.log(1 - frac) / np.log(g)
        print(f"Ex 9.6  {100 * frac:.0f}% of 1/(1-gamma): 1-step needs k >= {k1:.1f} refreshes "
              f"({int(np.ceil(k1)) * 250} steps at C=250); 3-step needs k >= {k1 / 3:.1f} "
              f"({int(np.ceil(k1 / 3)) * 250} steps)")


def ex_9_7():
    for tau in (0.004, 0.008):
        k = np.arange(20_000)
        wts = tau * (1 - tau) ** k
        print(f"Ex 9.7  Polyak tau={tau}: weights sum {wts.sum():.4f}, mean lag {np.sum(k * wts) / wts.sum():.1f} "
              f"(formula (1-tau)/tau = {(1 - tau) / tau:.1f}); hard copy every C=250: mean lag {(250 - 1) / 2:.1f}")


def ex_9_8():
    r = np.r_[np.zeros(900), np.full(100, 10.0)]
    for a, b in ((1.0, 0.0), (1.0, 0.5), (0.6, 0.0), (0.6, 1.0)):
        e = a * (1 - b)
        f = (lambda q: np.sum(np.abs(r - q) ** e * (r - q))) if e > 0 else (lambda q: np.sum(r - q))
        print(f"Ex 9.8  alpha={a}, beta={b}: effective exponent {e:.2f}, fixed point Q = {brentq(f, 1e-9, 10 - 1e-9):.4f}")


def ex_9_10():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    grid = np.linspace(-0.5, 3.5, 4001)
    for tau in (0.375, 0.5, 0.875):
        u = y[None, :] - grid[:, None]
        loss = np.mean(u * (tau - (u < 0)), axis=1)
        best = grid[np.isclose(loss, loss.min(), atol=1e-9)]
        print(f"Ex 9.10 Y uniform on {{0,1,2,3}}, tau={tau}: minimisers of E[rho_tau(Y-theta)] span "
              f"[{best.min():.3f}, {best.max():.3f}]")


def h(x, eps=1e-3):
    return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + eps * x


def h_inv(y, eps=1e-3):
    return np.sign(y) * (((np.sqrt(1 + 4 * eps * (np.abs(y) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1)


def ex_9_12():
    x = np.linspace(-1e4, 1e4, 200_001)
    print(f"Ex 9.12 max |h^-1(h(x)) - x| on [-1e4, 1e4]: {np.abs(h_inv(h(x)) - x).max():.2e}; "
          f"h(1)={h(1.0):.4f}, h(100)={h(100.0):.4f}, h(1e4)={h(1e4):.4f}")


def ex_9_13_14():
    from dqn_batched import train_batched
    base = Config(seed=0, eval_every=5_000)
    K = 5
    runs = {"1-step, hard copy C=250 (reference)": dict(),
            "3-step, hard copy C=250 (Ex 9.13)": dict(n_step=3),
            "1-step, Polyak tau=0.008 (Ex 9.14)": dict(polyak=0.008)}
    syncs = lambda t: max(0, (t - base.learning_starts) // base.target_update)
    for name, kw in runs.items():
        hst = train_batched(dataclasses.replace(base, **kw), K, verbose=False)
        steps = hst["eval_step"]
        n = kw.get("n_step", 1)
        pred = [(1 - 0.99 ** (n * syncs(t))) / 0.01 for t in steps]
        print(f"Ex 9.13/14  {name}: {hst['runtime_s']:.0f}s")
        print(f"    steps          {steps.tolist()}")
        print(f"    mean Q         {np.round(hst['q'].mean(0), 1).tolist()}")
        if "Polyak" not in name:
            print(f"    Eq. (9.4) pred {np.round(pred, 1).tolist()}")
        print(f"    mean score     {np.round(hst['score'].mean(0)).astype(int).tolist()}  "
              f"(final, last 10k steps: {hst['score'][:, -2:].mean():.1f} +- {hst['score'][:, -2:].mean(1).std(ddof=1):.1f}, sample std over seeds)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    print(f"Seed {SEED}; quick={args.quick}")
    ex_9_2()
    ex_9_4()
    ex_9_5(rng)
    ex_9_6()
    ex_9_7()
    ex_9_8()
    ex_9_10()
    ex_9_12()
    if not args.quick:
        ex_9_13_14()
    print(f"Total runtime {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
