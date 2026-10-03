"""SAC with a fixed entropy temperature vs automatic temperature tuning (Chapter 12, Section 6.5).

For a fixed alpha, multiplying all rewards by c is the same as dividing alpha by c (the soft
objective is c * [sum r + (alpha / c) H]).  So a fixed alpha is really a statement about the
reward scale, and a value that works for one task can be far too large or too small for another.
Automatic tuning instead fixes the *entropy* (target H_bar = -dim(A)) and lets alpha adapt.

We run, on Pendulum-v1 with identical settings and seeds:
    fixed alpha in {0.01, 0.1, 1, 10}   and   auto-tuned alpha (initial value 1.0)
and record the deterministic evaluation return (action tanh(mean)), the return of the stochastic
training episodes, the policy entropy and alpha.

Run (from the repository root):
  python code/ch12_continuous_control_actor_critic/sac_temperature.py          # 5 configs x 2 seeds
  python code/ch12_continuous_control_actor_critic/sac_temperature.py --quick  # smoke test, no figure
"""
from __future__ import annotations

import argparse
import dataclasses
import os

import numpy as np

from common import FIG_DIR
from sac import SACConfig, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--total-steps", type=int, default=7_000)
    args = ap.parse_args()

    configs = {
        "fixed alpha = 0.01": dict(autotune=False, init_alpha=0.01),
        "fixed alpha = 0.1": dict(autotune=False, init_alpha=0.1),
        "fixed alpha = 1.0": dict(autotune=False, init_alpha=1.0),
        "fixed alpha = 10": dict(autotune=False, init_alpha=10.0),
        "auto alpha (H_bar = -1)": dict(autotune=True, init_alpha=1.0),
    }
    if args.quick:
        base = SACConfig(total_steps=1_200, learning_starts=400, eval_every=400, eval_episodes=1)
        seeds = [0]
        configs = {k: configs[k] for k in ["fixed alpha = 1.0", "auto alpha (H_bar = -1)"]}
    else:
        base = SACConfig(total_steps=args.total_steps, eval_every=1_000, eval_episodes=5)
        seeds = list(range(args.seeds))
    print(f"SAC temperature experiment | configs {list(configs)} | seeds {seeds}")
    print(f"shared settings: {dataclasses.asdict(base)}")

    res = {}
    for name, kw in configs.items():
        res[name] = [train(dataclasses.replace(base, seed=sd, **kw), label=name) for sd in seeds]

    steps = np.array(res[next(iter(configs))][0]["step"])
    print("\n=== Summary (mean over seeds) ===")
    print(f"{'config':26s} {'eval return (last 3)':>21s} {'stochastic train return (last 10 ep)':>37s} "
          f"{'entropy (last)':>15s} {'alpha (last)':>13s}")
    summary = {}
    for name in configs:
        ev = np.array([h["eval_return"] for h in res[name]])
        tr = np.array([np.mean([r for _, r in h["train_returns"][-10:]]) for h in res[name]])
        ent = np.array([h["entropy"] for h in res[name]])
        al = np.array([h["alpha"] for h in res[name]])
        summary[name] = dict(ev=ev, ent=ent, al=al, tr=[h["train_returns"] for h in res[name]])
        print(f"{name:26s} {ev[:, -3:].mean():21.1f} {tr.mean():37.1f} {ent[:, -1].mean():15.2f} "
              f"{al[:, -1].mean():13.4f}")
        print(f"{'':26s} per seed eval (last 3): {np.round(ev[:, -3:].mean(1), 1).tolist()}, "
              f"stochastic: {np.round(tr, 1).tolist()}")

    if not args.quick:
        from plot_style import C, kfmt, setup
        plt = setup()
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
        marks = "os^dv"
        for k, name in enumerate(configs):
            col, s = C[k], summary[name]
            axes[0].plot(steps, s["ev"].mean(0), color=col, marker=marks[k], ms=4, label=name)
            # stochastic training-episode returns, binned per 1000 steps and averaged over seeds
            binned = []
            for tr in s["tr"]:
                t = np.array([x for x, _ in tr]); r = np.array([y for _, y in tr])
                binned.append([r[(t > b - 1000) & (t <= b)].mean() if np.any((t > b - 1000) & (t <= b))
                               else np.nan for b in steps])
            axes[1].plot(steps, np.nanmean(np.array(binned), 0), color=col, marker=marks[k], ms=4, label=name)
            axes[2].plot(steps, s["ent"].mean(0), color=col, marker=marks[k], ms=4, label=name)
            axes[3].semilogy(steps, s["al"].mean(0), color=col, marker=marks[k], ms=4, label=name)
        axes[0].set_title("deterministic evaluation (tanh of the mean)")
        axes[0].set_ylabel("return")
        axes[1].set_title("stochastic training episodes")
        axes[1].set_ylabel("return")
        for ax in axes[:2]:
            ax.axhline(-200, color="#8a8985", ls="--", lw=1)
            ax.set_ylim(-1700, 0)
        axes[2].axhline(-1.0, color="#8a8985", ls=":", lw=1, label="target entropy -1")
        axes[2].set_title("policy entropy E[-log pi(a|s)]")
        axes[2].set_ylabel("nats (actions in [-1, 1])")
        axes[3].set_title("temperature alpha")
        axes[0].legend(fontsize=8, loc="lower right")
        axes[2].legend(fontsize=7)
        for ax in axes:
            ax.set_xlabel("environment steps")
            kfmt(ax)
        fig.tight_layout()
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, "sac_temperature.png")
        fig.savefig(path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
