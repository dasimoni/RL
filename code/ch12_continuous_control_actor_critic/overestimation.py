"""Q-value over-estimation: single critic (DDPG-style) vs clipped double-Q (Chapter 12, Section 4.6).

We train learners that share *everything* (networks, learning rates, exploration noise,
replay, seeds) except the ingredients named.  The default full run compares the first two, so the
only difference is the min over two critics; TD3 can be added with --configs.

  DDPG        one critic, target y = r + gamma Q_bar(s', mu_bar(s'))                 (Eq. 12.5)
  DDPG + CDQ  two critics, target y = r + gamma min_j Q_bar_j(s', mu_bar(s'))         (Eq. 12.9)
  TD3         CDQ + delayed policy updates + target policy smoothing                 (Alg. 12.2)

Every `eval_every` steps we draw a fixed number of states from the replay buffer and compare

  estimated value  Q_1(s, mu(s))                (what the actor climbs)
  true value       v_mu(s) = sum_k gamma^k R    (exact: Pendulum is deterministic, so one
                                                 1000-step rollout of mu from s gives it)

exactly as in Fujimoto et al. (2018, Fig. 1).  The gap Q - v is the estimation bias.

Run (from the repository root):
  python code/ch12_continuous_control_actor_critic/overestimation.py          # DDPG vs DDPG+CDQ, 3 seeds
  python code/ch12_continuous_control_actor_critic/overestimation.py --configs DDPG "DDPG + CDQ" TD3
  python code/ch12_continuous_control_actor_critic/overestimation.py --utd 3 --seeds 2 --total-steps 8000  # Ex. 13
  python code/ch12_continuous_control_actor_critic/overestimation.py --quick  # smoke test, no figures
"""
from __future__ import annotations

import argparse
import dataclasses
import os

import numpy as np
import torch

from common import FIG_DIR, check_pendulum_true_value, pendulum_true_value
from td3 import TD3Config, train

CONFIGS = {
    "DDPG": dict(twin_q=False, policy_delay=1, target_noise=0.0),
    "DDPG + CDQ": dict(twin_q=True, policy_delay=1, target_noise=0.0),
    "TD3": dict(twin_q=True, policy_delay=2, target_noise=0.2),
}


def make_measure(n_states: int, horizon: int, gamma: float, seed: int):
    rng = np.random.default_rng(seed)

    @torch.no_grad()
    def measure(agent, buf, step):
        idx = rng.integers(0, buf.size, size=n_states)
        s = torch.from_numpy(buf.s[idx])
        a = agent.actor(s)
        q1 = agent.q[0](s, a).squeeze(1).numpy()
        out = {"Q_est": float(q1.mean())}
        if len(agent.q) > 1:
            q2 = agent.q[1](s, a).squeeze(1).numpy()
            out["minQ_est"] = float(np.minimum(q1, q2).mean())

        def act_batch(o):
            return agent.actor(torch.from_numpy(o)).numpy()

        v_true = pendulum_true_value(buf.s[idx], act_batch, gamma, horizon=horizon)
        out["V_true"] = float(v_true.mean())
        out["bias"] = out["Q_est"] - out["V_true"]
        return out

    return measure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--total-steps", type=int, default=12_000)
    ap.add_argument("--configs", nargs="+", default=["DDPG", "DDPG + CDQ"], choices=list(CONFIGS))
    ap.add_argument("--late", type=int, default=6_000, help="'late phase' starts at this step")
    ap.add_argument("--utd", type=int, default=1, help="critic updates per env step (Exercise 13)")
    args = ap.parse_args()

    if args.quick:
        base = TD3Config(total_steps=1_200, learning_starts=400, eval_every=400, eval_episodes=1)
        seeds, n_states, horizon = [0], 32, 200
        names = ["DDPG", "TD3"]
    else:
        base = TD3Config(total_steps=args.total_steps, eval_every=500, eval_episodes=3, utd=args.utd)
        seeds, n_states, horizon = list(range(args.seeds)), 256, 1000
        names = args.configs
    print(f"Over-estimation experiment | configs {names} | seeds {seeds} | "
          f"{n_states} buffer states per measurement, rollout horizon {horizon}")
    switches = ("twin_q", "policy_delay", "target_noise")
    print(f"shared settings: { {k: v for k, v in dataclasses.asdict(base).items() if k not in switches} }")
    print(f"per-config switches: { {n: CONFIGS[n] for n in names} }")
    n_chk = 5 if args.quick else 20
    print(f"check of the batched NumPy Pendulum against Gymnasium: max |difference| in the discounted "
          f"200-step return over {n_chk} episodes = {check_pendulum_true_value(n_chk):.1e}")

    results = {}
    for name in names:
        results[name] = []
        for sd in seeds:
            cfg = dataclasses.replace(base, seed=sd, **CONFIGS[name])
            h = train(cfg, measure=make_measure(n_states, horizon, cfg.gamma, seed=999 + sd), label=name)
            results[name].append(h)

    steps = np.array(results[names[0]][0]["step"])
    print("\n=== Summary (mean over seeds) ===")
    print(f"{'config':12s} {'bias@first':>11s} {'max bias':>9s} {'mean bias':>10s} "
          f"{'bias@end':>9s} {'|bias|/|V| end':>14s} {'return last5':>13s}")
    summary = {}
    for name in names:
        bias = np.array([[m["bias"] for m in h["measure"]] for h in results[name]])
        vtrue = np.array([[m["V_true"] for m in h["measure"]] for h in results[name]])
        rets = np.array([h["eval_return"] for h in results[name]])
        mb = bias.mean(0)
        summary[name] = dict(bias=bias, vtrue=vtrue, rets=rets)
        print(f"{name:12s} {mb[0]:11.1f} {mb.max():9.1f} {mb.mean():10.1f} {mb[-1]:9.1f} "
              f"{np.abs(mb[-1]) / np.abs(vtrue.mean(0)[-1]):14.3f} {rets[:, -5:].mean():13.1f}")
        if "minQ_est" in results[name][0]["measure"][0]:
            mq = np.array([[m["minQ_est"] - m["V_true"] for m in h["measure"]] for h in results[name]])
            gap = np.array([[m["Q_est"] - m["minQ_est"] for m in h["measure"]] for h in results[name]])
            print(f"{'':12s} (bias of min(Q1,Q2) instead of Q1: mean {mq.mean():.1f}, end {mq.mean(0)[-1]:.1f}; "
                  f"largest gap mean Q1 - mean min(Q1,Q2) at any measurement: {gap.max():.2f})")
        late = steps >= args.late if not args.quick else steps >= steps[-1]
        rel = bias[:, late] / np.abs(vtrue[:, late])
        ratio_of_means = bias[:, late].mean(1) / np.abs(vtrue[:, late]).mean(1)
        print(f"{'':12s} late phase (steps >= {steps[late][0]}): mean bias per seed "
              f"{np.round(bias[:, late].mean(1), 1).tolist()}, mean relative bias (Q - v)/|v| per seed "
              f"{np.round(rel.mean(1), 3).tolist()} (ratio of means: {np.round(ratio_of_means, 3).tolist()})")
        if not args.quick:
            win = (steps >= args.late) & (steps <= args.late + 2_000)
            print(f"{'':12s} window {args.late}-{args.late + 2_000}: mean bias per seed "
                  f"{np.round(bias[:, win].mean(1), 1).tolist()}, mean relative bias per seed "
                  f"{np.round((bias[:, win] / np.abs(vtrue[:, win])).mean(1), 3).tolist()}")
        first = [int(steps[np.nonzero(rr >= -200)[0][0]]) if np.any(rr >= -200) else None for rr in rets]
        print(f"{'':12s} first evaluation with return >= -200 at steps {first}")

    if not args.quick and args.utd == 1:     # the chapter's figure is the UTD = 1 run
        os.makedirs(FIG_DIR, exist_ok=True)
        np.savez(os.path.join(FIG_DIR, "overestimation_data.npz"), steps=steps,
                 **{f"{n}|{k}": v for n in names for k, v in summary[n].items()})
        from plot_style import C, kfmt, setup
        plt = setup()
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
        axes = axes.ravel()
        styles = {"DDPG": (C[1], "-"), "DDPG + CDQ": (C[2], "--"), "TD3": (C[0], "-.")}
        lt = steps >= args.late
        for name in names:
            col, ls = styles[name]
            b, v, r = summary[name]["bias"], summary[name]["vtrue"], summary[name]["rets"]
            q_est = b + v
            axes[0].plot(steps, q_est.mean(0), color=col, ls=ls, label=f"{name}: estimate Q(s, mu(s))")
            axes[0].plot(steps, v.mean(0), color=col, ls=ls, lw=1.0, alpha=0.6,
                         marker="o", ms=3, label=f"{name}: true value v(s)")
            for i, bi in enumerate(b):        # late phase, one thin line per seed + the mean
                axes[1].plot(steps[lt], bi[lt], color=col, ls=ls, lw=0.8, alpha=0.5)
            axes[1].plot(steps[lt], b.mean(0)[lt], color=col, ls=ls, lw=2.4, label=f"{name} (mean of {len(b)})")
            rel = b / np.abs(v)
            for bi in rel:
                axes[2].plot(steps[lt], bi[lt], color=col, ls=ls, lw=0.8, alpha=0.5)
            axes[2].plot(steps[lt], rel.mean(0)[lt], color=col, ls=ls, lw=2.4, label=name)
            axes[3].plot(steps, r.mean(0), color=col, ls=ls, label=name)
            axes[3].fill_between(steps, r.min(0), r.max(0), color=col, alpha=0.12)
        axes[0].set_title("whole run: value estimate vs true value")
        axes[0].set_ylabel("average over 256 buffer states")
        axes[0].legend(fontsize=7, loc="lower right")
        axes[1].axhline(0, color="#8a8985", lw=1)
        axes[1].set_title(f"late phase (from {args.late // 1000}k steps): bias Q - v")
        axes[1].set_ylabel("bias (thin lines: seeds)")
        axes[1].legend()
        axes[2].axhline(0, color="#8a8985", lw=1)
        axes[2].set_title("late phase: relative bias (Q - v) / |v|")
        axes[2].set_ylabel("relative bias")
        axes[2].legend()
        axes[3].set_ylim(-1600, 0)
        axes[3].axhline(-200, color="#8a8985", ls="--", lw=1)
        axes[3].set_title("evaluation return (band: min-max over seeds)")
        axes[3].set_ylabel("return (3 episodes)")
        axes[3].legend(loc="lower right")
        for ax in axes:
            ax.set_xlabel("environment steps")
            kfmt(ax)
        fig.tight_layout()
        path = os.path.join(FIG_DIR, "overestimation.png")
        fig.savefig(path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
