"""Ablation: what does the clip in PPO actually buy?  (Chapter 11, Sections 7.4 and 8.4)

CartPole-v1, 5 seeds, in an *aggressive sample-reuse* regime (K = 10 epochs per batch,
learning rate 1e-3), where the surrogate is optimised hard enough for the trust region to
matter.  Three variants share every other detail of ppo.py, including value clipping
(its range vclip_coef = 0.2 is separate from the policy clip):

  clip           : PPO-clip, epsilon = 0.2                                   (Eq. 11.30)
  no clip        : the plain importance-weighted surrogate rho_t(theta) A_t   (Eq. 11.29)
  no clip + KL   : the plain surrogate, but stop the epoch loop once the k3 estimate of
                   KL(pi_old || pi_theta) exceeds 0.02 (early stopping, Section 8.2)

For each update we log the realised KL(pi_old || pi_new) of the policy the update produced
(exact, on the whole batch, after the epochs -- the same quantity TRPO's line search controls
in Section 6.7) and, for that same policy, max_t |rho_t - 1| and the fraction of samples with
|rho_t - 1| > 0.2.

Run:  python code/ch11_trust_regions_and_ppo/ppo_clip_ablation.py [--quick]
"""
from __future__ import annotations

import os

# One thread for BLAS *before* numpy is imported (ppo.py sets the same, but too late for numpy
# if numpy were imported first).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

from ppo import FIG_DIR, first_reach, make_config, smoothed_curve, train  # noqa: E402

VARIANTS = {
    "clip (eps = 0.2)": dict(),
    "no clip": dict(clip_coef=math.inf),
    "no clip + KL early stop (0.02)": dict(clip_coef=math.inf, target_kl=0.02),
}
BASE = dict(norm_reward=True, update_epochs=10, lr=1e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    seeds = [1] if args.quick else [1, 2, 3, 4, 5]
    total = 8_192 if args.quick else 80_000
    cfg0 = make_config("CartPole-v1", total_steps=total, **BASE)
    print(f"CartPole-v1, seeds={seeds}, total_steps={total}, epochs K={cfg0.update_epochs}, lr={cfg0.lr} "
          f"(annealed), batch={cfg0.batch_size}, minibatch={cfg0.minibatch_size}, reward scaling on, "
          f"value clipping on (range {cfg0.vclip_coef})")
    grid = np.linspace(0, total, 160)
    results = {}
    for name, ov in VARIANTS.items():
        runs = []
        for s in seeds:
            res = train(make_config("CartPole-v1", seed=s, total_steps=total, **{**BASE, **ov}), verbose=False)
            runs.append(res)
        results[name] = runs
        with warnings.catch_warnings():         # nanmean over grid points before the first episode
            warnings.simplefilter("ignore", RuntimeWarning)
            curves = np.array([smoothed_curve(r["episodes"], grid) for r in runs])
            final = curves[:, -1]
            auc = np.nanmean(curves, axis=1)
        kl = np.concatenate([r["logs"]["kl_post"] for r in runs])
        k3pre = np.concatenate([r["logs"]["approx_kl_k3"] for r in runs])
        dev = np.concatenate([r["logs"]["max_ratio_dev_post"] for r in runs])
        cf = np.concatenate([r["logs"]["clipfrac_post"] for r in runs])
        ep = np.concatenate([r["logs"]["epochs_run"] for r in runs])
        reach = [first_reach(c, grid, 500.0) for c in curves]
        # a "collapse": the smoothed return falls by more than 200 below its running maximum
        drops = [(np.fmax.accumulate(np.nan_to_num(c)) - np.nan_to_num(c)).max() for c in curves]
        collapses = sum(int(d > 200) for d in drops)
        sd = np.std(auc, ddof=1) if len(auc) > 1 else float("nan")
        print(f"\n{name}")
        print(f"  final return per seed {np.round(final, 1)}  mean {np.nanmean(final):.1f}")
        print(f"  mean return over training (AUC) {np.mean(auc):.1f} +- {sd:.1f} (sd over seeds, ddof=1); "
              f"seeds with a collapse (drop > 200): {collapses}/{len(seeds)}; largest drop per seed "
              f"{np.round(drops, 0)}")
        print(f"  first step at which the smoothed return reaches 500, per seed: {np.round(reach, -2)}")
        print(f"  realised KL(pi_old || pi_new) per update (exact, after the epochs): median {np.median(kl):.2e}, "
              f"90th pct {np.quantile(kl, 0.9):.2e}, max {kl.max():.3g};  "
              f"pre-step k3 estimate: median {np.median(k3pre):.2e}")
        print(f"  after the update: max_t |rho_t-1| median {np.median(dev):.2f}, max {dev.max():.3g};  "
              f"fraction |rho-1| > 0.2: median {np.median(cf):.3f}, max {cf.max():.3f};  "
              f"epochs run: mean {ep.mean():.1f}", flush=True)

    if not args.quick:
        from plot_style import C, kfmt, setup
        plt = setup()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for (name, runs), c in zip(results.items(), [C[0], C[1], C[2]]):
                curves = np.array([smoothed_curve(r["episodes"], grid) for r in runs])
                for cv in curves:
                    ax[0].plot(grid, cv, color=c, lw=0.6, alpha=0.35)
                ax[0].plot(grid, np.nanmean(curves, 0), color=c, lw=2.2, label=name)
                steps = np.array(runs[0]["logs"]["step"])
                kl = np.array([r["logs"]["kl_post"] for r in runs])
                dev = np.array([r["logs"]["max_ratio_dev_post"] for r in runs])
                ax[1].semilogy(steps, np.median(kl, 0), color=c, label=name)
                ax[2].semilogy(steps, np.median(dev, 0), color=c, label=name)
        ax[0].set_title("return (thin: 5 seeds, thick: mean)")
        ax[0].set_xlabel("environment steps")
        ax[0].legend(fontsize=8, loc="lower right")
        ax[1].set_title("realised KL(pi_old || pi_new) per update (median)")
        ax[1].set_xlabel("environment steps")
        ax[2].set_title("max |rho_t - 1| after the update (median over seeds)")
        ax[2].axhline(0.2, color="#8a8985", lw=0.8)
        ax[2].set_xlabel("environment steps")
        for a in ax:
            kfmt(a)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "ppo_clip_ablation.png"))
        plt.close(fig)
        print("\nwrote figures/ppo_clip_ablation.png")
    print(f"total wall time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
