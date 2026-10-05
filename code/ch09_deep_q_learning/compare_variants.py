"""DQN vs Double DQN vs Dueling DQN vs Double+Dueling on CartPole-v1, 5 seeds each.

Chapter 09, Sections 4-5 and "In code".  Every variant uses the same hyperparameters (dqn.Config
defaults: Adam 5e-4, batch 64, replay 50k, target copy every 250 steps, epsilon 1 -> 0.05 over
10k steps, Huber loss, 64-64 ReLU MLP) and the same seeds; only the target (Eq. 9.2 vs 9.8) and
the head (plain vs dueling, Eq. 9.9) change.  The 5 seeds of a variant are trained in lock-step
by dqn_batched.py: equivalent in distribution to 5 independent dqn.py runs (same algorithm and
hyperparameters, independent weights and data), but not the same random streams as
`dqn.py --seed k`.  The dueling network has about twice the parameters of the plain one
(8.8k vs 4.6k), because each stream has its own hidden layer, as in Wang et al. (2016).

Measurements, every 2,500 environment steps: greedy score (5 episodes, capped at 500) and the
mean predicted max_a Q(s, a) on the states those episodes visit.

Output (full mode): figures/compare_variants.png
Run:  python code/ch09_deep_q_learning/compare_variants.py [--quick]
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import time

import numpy as np

from dqn import Config, FIG_DIR
from dqn_batched import train_batched

VARIANTS = {
    "DQN": dict(),
    "Double DQN": dict(double=True),
    "Dueling DQN": dict(dueling=True),
    "Double + Dueling": dict(double=True, dueling=True),
}


def summarise(name, h):
    """Per-seed numbers: final = mean of the last 4 evaluations (last 10k steps)."""
    final = h["score"][:, -4:].mean(1)
    auc = h["score"].mean(1)                          # mean score over the whole run
    solved = (h["score"] >= 500).mean(1)              # fraction of evaluations at the 500 cap
    q_last = h["q"][:, -4:].mean(1)
    return dict(name=name, final=final, auc=auc, solved=solved, q_last=q_last)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()
    base = Config(seed=0)
    K = args.seeds
    if args.quick:
        base = dataclasses.replace(base, total_steps=2_000, eval_every=1_000, learning_starts=500,
                                   eps_decay_steps=1_000)
        K = 2
    print(f"Seeds {base.seed}..{base.seed + K - 1} | {base.total_steps} env steps per run | "
          f"base config: {dataclasses.asdict(base)}")
    t0 = time.perf_counter()
    results, rows = {}, []
    for name, kw in VARIANTS.items():
        cfg = dataclasses.replace(base, **kw)
        print(f"\n=== {name} ({kw}) ===", flush=True)
        h = train_batched(cfg, K, verbose=False)
        results[name] = h
        r = summarise(name, h)
        rows.append(r)
        print(f"  {h['runtime_s']:.0f}s; final score per seed {np.round(r['final']).astype(int)}; "
              f"final Q per seed {np.round(r['q_last'], 1)}", flush=True)

    print("\nSummary (mean +- sample std over seeds, ddof=1; final = mean greedy score over the last 10k steps)")
    print(f"{'variant':18s} {'final score':>15s} {'median':>7s} {'mean over run':>14s} "
          f"{'evals at 500':>13s} {'final mean Q':>13s}")
    for r in rows:
        print(f"{r['name']:18s} {r['final'].mean():7.1f} +- {r['final'].std(ddof=1):5.1f} {np.median(r['final']):7.1f} "
              f"{r['auc'].mean():8.1f} +- {r['auc'].std(ddof=1):4.1f} {100 * r['solved'].mean():12.0f}% "
              f"{r['q_last'].mean():7.1f} +- {r['q_last'].std(ddof=1):4.1f}")
    print(f"Total runtime {time.perf_counter() - t0:.0f}s")
    if args.quick:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import setup, C, GREY
    setup()
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
    styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]
    for i, (name, h) in enumerate(results.items()):
        x = h["eval_step"] / 1000
        m, se = h["score"].mean(0), h["score"].std(0, ddof=1) / np.sqrt(K)
        ax[0].plot(x, m, styles[i], marker=markers[i], ms=3.5, color=C[i], label=name)
        ax[0].fill_between(x, m - se, m + se, color=C[i], alpha=0.12, lw=0)
        qm = h["q"].mean(0)
        ax[1].plot(x, qm, styles[i], marker=markers[i], ms=3.5, color=C[i], label=name)
        ax[1].fill_between(x, h["q"].min(0), h["q"].max(0), color=C[i], alpha=0.08, lw=0)
    ax[0].set(xlabel="environment steps (thousands)", ylabel="greedy score (max 500)",
              title=f"CartPole-v1: mean $\\pm$ s.e. over {K} seeds", ylim=(0, 520))
    ax[0].legend(loc="upper left")
    ax[1].axhline(100, color=GREY, ls=":", lw=1.2)
    ax[1].text(1, 102, r"$1/(1-\gamma)=100$: largest possible return", color=GREY, fontsize=8.5)
    ax[1].set(xlabel="environment steps (thousands)", ylabel=r"mean predicted $\max_a Q$",
              title="Value estimates (line: mean; band: min-max over seeds)")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, "compare_variants.png")
    fig.savefig(path)
    print("saved", path)


if __name__ == "__main__":
    main()
