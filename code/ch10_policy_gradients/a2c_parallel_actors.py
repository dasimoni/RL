"""What are A2C's parallel actors for?  Four ways of collecting the same-sized batch.

Chapter 10, Section 9.5.  Uses train() from a2c_gae_cartpole.py (A2C + GAE, lambda = 0.95, the
same hyperparameters as there).  Every update uses a batch of 128 transitions, collected as

    (N_env = 8,  n = 16)                 8 parallel environments, 16-step rollouts (the default)
    (N_env = 1,  n = 16, 8 segments)     the SAME estimator (GAE truncated every 16 steps), but the
                                         8 segments are consecutive pieces of ONE environment's
                                         trajectory -> isolates the decorrelation effect
    (N_env = 1,  n = 128)                one environment, 128-step rollouts (longer GAE horizon)
    (N_env = 32, n = 4)                  32 environments, 4-step rollouts (shorter GAE horizon)

Outputs (full mode): figures/a2c_parallel_actors.png
Run:  python code/ch10_policy_gradients/a2c_parallel_actors.py [--quick]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from a2c_gae_cartpole import FIG_DIR, plot_mean_sd, summarize, train

torch.set_num_threads(1)
CONFIGS = [(8, 16, 1), (1, 16, 8), (1, 128, 1), (32, 4, 1)]


def label(cfg):
    ne, ns, sg = cfg
    return f"N_env = {ne}, n = {ns}" + (f", {sg} consecutive segments" if sg > 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n_seeds, total = (1, 6_000) if args.quick else (5, 150_000)
    base = dict(gamma=0.99, lr=1e-3, c_v=0.5, c_ent=0.01, max_grad_norm=0.5)
    print(f"seed={args.seed}  seeds={n_seeds}  env steps/run={total}  lambda=0.95  {base}  "
          f"configs (N_env, n, segments) = {CONFIGS}")
    t0 = time.time()
    res = {}
    for cfg in CONFIGS:
        ne, ns, sg = cfg
        res[cfg] = [train(0.95, args.seed + i, total, n_env=ne, n_steps=ns, segments=sg, **base)
                    for i in range(n_seeds)]
        fin, auc = summarize(res[cfg], total)
        sd = fin.std(ddof=1) if n_seeds > 1 else 0.0
        print(f"  {label(cfg):40s}: final return {fin.mean():6.1f} +- {sd:5.1f} "
              f"(per seed {np.round(fin).astype(int).tolist()}),  mean over training {auc.mean():6.1f}"
              f"  [{time.time() - t0:.0f}s]")
    print(f"\nTotal time {time.time() - t0:.0f}s")

    if not args.quick:
        from matplotlib.ticker import FuncFormatter
        from plot_style import setup, C
        plt = setup()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        styles = ["-", "--", "-.", ":"]
        for j, (cfg, rs) in enumerate(res.items()):
            plot_mean_sd(ax, rs, total, C[j], styles[j], label(cfg), band_alpha=0.10)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
        ax.set_xlabel("environment steps")
        ax.set_ylabel("episode return (10k-step window)")
        ax.set_ylim(0, 520)
        ax.set_title(f"A2C, batches of 128 transitions ($\\lambda$=0.95, {n_seeds} seeds;\n"
                     "band: $\\pm$1 s.d. over seeds)", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "a2c_parallel_actors.png"))
        plt.close(fig)
        print(f"figure written to {FIG_DIR}/a2c_parallel_actors.png")


if __name__ == "__main__":
    main()
