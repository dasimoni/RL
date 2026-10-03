"""Overestimation in Q-learning targets, and what Double DQN does about it.

Chapter 09, Section 4.

Part A (numpy, exact/statistical facts about the max operator)
  1. The tight lower bound of van Hasselt, Guez & Silver (2016), Theorem 9.1 / Eq. (9.7): if the errors
     e_a = Q(s,a) - V*(s) sum to zero and have mean square sigma^2, then max_a e_a >= sqrt(sigma^2/(m-1)).
     We check it by random search and with the extremal error vector of the proof.
  2. Bias of a single max-backup when the m errors are iid N(0,1): E[max_a e_a] grows with m,
     while the double estimator e'_{argmax_a e_a} (independent second estimate) is unbiased.
     (The bound of 1. is NOT a bound on this curve: iid errors do not sum to zero, so the figure
     shows only the bias; the bound is checked numerically in the printout.)

Part B (deep RL, 5 seeds per configuration, batched agents from dqn_batched.py)
  DQN and Double DQN on CartPole-v1 with the actions offered once (2 actions, the standard task)
  or four times (8 actions; action a means real action a mod 2 -- the true values are unchanged,
  but the max is now over 8 separately estimated outputs, as with Atari's 18 partly-redundant
  actions).  Every 2,500 steps: predicted max_a Q on greedy-policy states vs the discounted
  Monte Carlo return actually obtained from those states, and the fraction of states whose
  predicted value exceeds 1/(1-gamma) = 100, which no policy can achieve.

Output (full mode): figures/overestimation.png
Run:  python code/ch09_deep_q_learning/overestimation.py [--quick]
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import time

import numpy as np

from dqn import Config, FIG_DIR
from dqn_batched import train_batched

SEED = 0


def part_a(rng, n_search=200_000, n_rep=20_000):
    print("Part A.1  Tight lower bound max_a e_a >= sqrt(sigma^2/(m-1)) for zero-sum errors with mean square sigma^2=1")
    for m in (2, 3, 5, 10, 20):
        e = rng.standard_normal((n_search, m))
        e -= e.mean(1, keepdims=True)                       # "unbiased on the whole": sum_a e_a = 0
        e /= np.sqrt((e ** 2).mean(1, keepdims=True))       # mean square exactly sigma^2 = 1
        bound = np.sqrt(1.0 / (m - 1))
        tight = np.r_[np.full(m - 1, bound), -np.sqrt(m - 1.0)]   # extremal vector of the proof
        print(f"  m={m:2d}: bound {bound:.4f}; smallest max over {n_search} random vectors {e.max(1).min():.4f}; "
              f"extremal vector: sum {tight.sum():+.1e}, mean square {np.mean(tight ** 2):.4f}, max {tight.max():.4f}")
    print("Part A.2  Bias of one backup with iid N(0,1) errors (mean over", n_rep, "repetitions, +- 2 s.e.)")
    ms = 2 ** np.arange(1, 11)
    single, double, s_se, d_se = [], [], [], []
    for m in ms:
        e1 = rng.standard_normal((n_rep, m))
        e2 = rng.standard_normal((n_rep, m))                # independent second estimator
        mx = e1.max(1)
        dbl = e2[np.arange(n_rep), e1.argmax(1)]
        single.append(mx.mean()); s_se.append(mx.std() / np.sqrt(n_rep))
        double.append(dbl.mean()); d_se.append(dbl.std() / np.sqrt(n_rep))
        print(f"  m={m:4d}: max-estimator bias {single[-1]:+.3f} +- {2 * s_se[-1]:.3f}   "
              f"double-estimator bias {double[-1]:+.3f} +- {2 * d_se[-1]:.3f}")
    return dict(ms=ms, single=np.array(single), double=np.array(double), d_se=np.array(d_se))


CONFIGS = {
    ("DQN", 1): dict(),
    ("Double DQN", 1): dict(double=True),
    ("DQN", 4): dict(action_copies=4),
    ("Double DQN", 4): dict(double=True, action_copies=4),
}


def part_b(base, K):
    print(f"\nPart B  CartPole-v1, {K} seeds per configuration, {base.total_steps} steps")
    out = {}
    for (name, copies), kw in CONFIGS.items():
        cfg = dataclasses.replace(base, **kw)
        h = train_batched(cfg, K, verbose=False)
        out[(name, copies)] = h
        last = slice(-4, None)                               # last 4 evaluations = last 10k steps
        q, mc = h["q"][:, last].mean(1), h["mc"][:, last].mean(1)
        print(f"  {name:10s} {2 * copies} actions: {h['runtime_s']:.0f}s | final mean Q {q.mean():6.1f} "
              f"(per seed {np.round(q, 1)}) | MC return {mc.mean():5.1f} | Q - MC {np.mean(q - mc):+6.1f} | "
              f"states with Q > 100: {100 * h['frac_over'][:, last].mean():4.1f}% (last 10k), "
              f"{100 * h['frac_over'].mean():4.1f}% (whole run) | final score {h['score'][:, last].mean():5.1f}",
              flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    rng = np.random.default_rng(SEED)
    base = Config(seed=SEED)
    K = 5
    if args.quick:
        base = dataclasses.replace(base, total_steps=2_000, eval_every=1_000, learning_starts=500,
                                   eps_decay_steps=1_000)
        K = 2
    print(f"Seed {SEED}; Part B base config: {dataclasses.asdict(base)}")
    t0 = time.perf_counter()
    a = part_a(rng, n_search=20_000 if args.quick else 200_000, n_rep=2_000 if args.quick else 20_000)
    b = part_b(base, K)
    print(f"Total runtime {time.perf_counter() - t0:.0f}s")
    if args.quick:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import setup, C, GREY
    setup()
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8))
    ax0 = ax[0, 0]
    ax0.plot(a["ms"], a["single"], "o-", color=C[1], label=r"max estimator $\max_a(V^\ast+e_a)-V^\ast$")
    ax0.plot(a["ms"], a["double"], "s--", color=C[0], label=r"double estimator $e'_{\arg\max_a e_a}$")
    ax0.set_xscale("log", base=2)
    ax0.axhline(0, color="k", lw=0.6)
    ax0.set(xlabel="number of actions m", ylabel="bias of one backup",
            title="A: one backup, errors iid N(0,1) (20,000 draws per m)")
    ax0.legend(loc="upper left", fontsize=8.5)
    for j, copies in enumerate((1, 4)):
        axj = ax[0, 1] if j == 0 else ax[1, 0]
        for i, name in enumerate(("DQN", "Double DQN")):
            h = b[(name, copies)]
            x = h["eval_step"] / 1000
            axj.plot(x, h["q"].mean(0), "-", marker="o", ms=3.5, color=C[[1, 0][i]],
                     label=f"{name}: predicted $\\max_a Q$")
            axj.plot(x, h["mc"].mean(0), "--", marker="s", ms=3, color=C[[1, 0][i]], alpha=0.7,
                     label=f"{name}: Monte Carlo return")
        axj.axhline(100, color=GREY, ls=":", lw=1.2)
        axj.set(xlabel="environment steps (thousands)", ylabel="discounted value (mean over 5 seeds)",
                title=f"{'B' if j == 0 else 'C'}: CartPole, {2 * copies} actions"
                      + (" (each real action offered 4 times)" if copies > 1 else ""), ylim=(0, 125))
        axj.legend(loc="lower right", fontsize=8)
    ax3 = ax[1, 1]
    for i, ((name, copies), h) in enumerate(b.items()):
        ax3.plot(h["eval_step"] / 1000, 100 * h["frac_over"].mean(0), ["-", "--", "-", "--"][i],
                 marker=["o", "s", "^", "D"][i], ms=3.5, color=C[[1, 0, 3, 2][i]],
                 label=f"{name}, {2 * copies} actions")
    ax3.set(xlabel="environment steps (thousands)", ylabel="% of visited states",
            title=r"D: states with predicted $\max_a Q > 1/(1-\gamma)$ (impossible)")
    ax3.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, "overestimation.png")
    fig.savefig(path)
    print("saved", path)


if __name__ == "__main__":
    main()
