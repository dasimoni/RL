"""Ablation: which PPO "implementation details" matter on CartPole?  (Chapter 11, Section 8.3)

Standard regime (CleanRL ppo.py defaults: 4 envs x 128 steps, K = 4 epochs, 4 minibatches,
lr 2.5e-4 annealed, eps = 0.2, value clipping, gradient-norm clip 0.5, orthogonal init) plus
reward scaling, which is our CartPole preset.  Each variant switches ONE detail off:

  full                 : the preset
  no adv. norm         : advantages are not standardised per minibatch
  no reward scaling    : raw rewards (+1 per step), so value targets reach ~100
  no grad-norm clip    : no global gradient-norm clipping
  no reward scaling,   : raw rewards AND no gradient clipping -- tests the explanation that the
    no grad-norm clip    critic's large gradient, through the *shared* norm clip, starves the actor
  no orthogonal init   : PyTorch's default (Kaiming-uniform) initialisation

With --extra, the script instead runs only "full" and

  no reward scaling,   : raw rewards AND no value clipping -- tests whether PPO2's value clip (which
    no value clipping    limits each critic prediction's change per batch to 0.2) is what stops the
                         critic from fitting targets of ~100

(kept out of the default run to stay within the time budget; same seeds and code).

For each variant we report the final return and the area under the learning curve (AUC) with
standard deviations over seeds (ddof = 1) and the standard error of the AUC difference to
"full"; the median over updates of the realised KL(pi_old || pi_new) per update (exact, after the
epochs); the median over updates of the per-update mean pre-clipping gradient norms of actor and
critic; and the median explained variance of the critic over the last quarter of training.

Run:  python code/ch11_trust_regions_and_ppo/ppo_details_ablation.py [--quick] [--extra]
"""
from __future__ import annotations

import os

# One thread for BLAS *before* numpy is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

from ppo import FIG_DIR, make_config, smoothed_curve, train  # noqa: E402

VARIANTS = {
    "full": dict(),
    "no adv. norm": dict(norm_adv=False),
    "no reward scaling": dict(norm_reward=False),
    "no grad-norm clip": dict(max_grad_norm=1e9),
    "no reward scaling, no grad-norm clip": dict(norm_reward=False, max_grad_norm=1e9),
    "no orthogonal init": dict(ortho_init=False),
}
EXTRA = {
    "full": dict(),
    "no reward scaling, no value clipping": dict(norm_reward=False, clip_vloss=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--extra", action="store_true", help="run only 'full' and the value-clipping row")
    args = ap.parse_args()
    t0 = time.time()
    seeds = [1] if args.quick else [1, 2, 3, 4, 5]
    total = 4_096 if args.quick else 70_000
    table = EXTRA if args.extra else VARIANTS
    variants = dict(list(table.items())[:2]) if args.quick else table
    print(f"CartPole-v1 preset (CleanRL defaults + reward scaling), seeds={seeds}, total_steps={total}, "
          f"variants={list(variants)}")
    grid = np.linspace(0, total, 160)
    rows = []
    curves_all = {}
    auc_full = None
    sd = (lambda x: np.std(x, ddof=1)) if len(seeds) > 1 else (lambda x: float("nan"))
    for name, ov in variants.items():
        runs = [train(make_config("CartPole-v1", seed=s, total_steps=total, **ov), verbose=False) for s in seeds]
        with warnings.catch_warnings():        # nanmean over grid points before the first episode
            warnings.simplefilter("ignore", RuntimeWarning)
            curves = np.array([smoothed_curve(r["episodes"], grid) for r in runs])
            final, auc = curves[:, -1], np.nanmean(curves, axis=1)
        curves_all[name] = curves
        if name == "full":
            auc_full = auc
        kl = np.median(np.concatenate([r["logs"]["kl_post"] for r in runs]))
        k3 = np.median(np.concatenate([r["logs"]["approx_kl_k3"] for r in runs]))
        ga = np.median(np.concatenate([r["logs"]["grad_norm_actor"] for r in runs]))
        gc = np.median(np.concatenate([r["logs"]["grad_norm_critic"] for r in runs]))
        q = max(1, len(runs[0]["logs"]["explained_var"]) // 4)
        ev = np.nanmedian(np.concatenate([r["logs"]["explained_var"][-q:] for r in runs]))
        se = sd(auc) / np.sqrt(len(seeds))
        rows.append((name, np.nanmean(final), sd(final), np.mean(auc), sd(auc), kl))
        print(f"  {name:38s} final {np.nanmean(final):6.1f} +- {sd(final):5.1f}   "
              f"AUC {np.mean(auc):6.1f} +- {sd(auc):5.1f} (SE {se:4.1f})   per seed final {np.round(final, 0)}",
              flush=True)
        if auc_full is not None and name != "full":
            se_diff = np.sqrt(sd(auc) ** 2 / len(seeds) + sd(auc_full) ** 2 / len(seeds))
            print(f"  {'':38s} AUC difference to full {np.mean(auc) - np.mean(auc_full):+6.1f} "
                  f"(SE of the difference {se_diff:.1f}, i.e. {(np.mean(auc) - np.mean(auc_full)) / se_diff:+.1f} SE)")
        print(f"  {'':38s} median realised KL per update {kl:.2e} (pre-step k3 estimate {k3:.2e});  "
              f"pre-clipping grad norm (median over updates of the per-update mean): actor {ga:.3f}, "
              f"critic {gc:.2f};  explained variance (last quarter) {ev:.2f}", flush=True)

    if not args.quick and not args.extra:
        from plot_style import C, kfmt, setup
        plt = setup()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
        styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for (name, curves), c, ls in zip(curves_all.items(), C, styles):
                ax[0].plot(grid, np.nanmean(curves, 0), color=c, ls=ls, label=name)
        ax[0].set_xlabel("environment steps")
        kfmt(ax[0])
        ax[0].set_ylabel("return (last 20 episodes), mean of 5 seeds")
        ax[0].set_ylim(0, 520)
        ax[0].set_title("CartPole-v1: switching off one detail at a time")
        ax[0].legend(fontsize=8, loc="upper left")
        names = [r[0] for r in rows]
        y = np.arange(len(rows))[::-1]
        ax[1].barh(y, [r[3] for r in rows], xerr=[r[4] for r in rows], color=[C[i] for i in range(len(rows))],
                   alpha=0.85, capsize=3)
        ax[1].set_yticks(y)
        ax[1].set_yticklabels(names, fontsize=8)
        ax[1].set_xlabel("mean return over training (AUC), +- 1 sd over seeds")
        ax[1].set_title("area under the learning curve")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "ppo_details_ablation.png"))
        plt.close(fig)
        print("wrote figures/ppo_details_ablation.png")
    print(f"total wall time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
