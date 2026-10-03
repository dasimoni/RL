"""Ornstein-Uhlenbeck vs Gaussian exploration noise (Chapter 12, Section 3.4).

DDPG explored by adding temporally correlated Ornstein-Uhlenbeck (OU) noise to its deterministic
actions; TD3 and most later work use independent Gaussian noise.  Here we isolate the noise
process: the "policy" outputs 0 and the action is the noise alone (clipped to [-1, 1], i.e. to
the torque range [-2, 2]).  Every episode starts hanging straight down, almost at rest; we ask
how high each noise process swings the pendulum and how much of the state space it visits.

  OU (Eq. 12.8):  x_{t+1} = x_t - kappa x_t dt + sigma sqrt(dt) xi_t,   xi_t ~ N(0, 1)
     stationary std  sigma sqrt(dt / (1 - (1 - kappa dt)^2)),  lag-k autocorrelation (1 - kappa dt)^k
  (the code calls the mean-reversion rate `theta`, as the DDPG paper does; the chapter writes kappa)

Run (from the repository root):
  python code/ch12_continuous_control_actor_critic/exploration_noise.py          # 100 episodes, figure
  python code/ch12_continuous_control_actor_critic/exploration_noise.py --quick  # 10 episodes, no figure
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from common import FIG_DIR, make_env


def ou_std(theta, sigma, dt):
    return sigma * np.sqrt(dt / (1 - (1 - theta * dt) ** 2))


def make_noise(kind, rng, theta=0.15, sigma=0.2, dt=1.0, std=0.38):
    """Returns (reset_fn, sample_fn)."""
    if kind == "gaussian":
        return (lambda: None), (lambda: rng.normal(0.0, std))
    if kind == "uniform":
        return (lambda: None), (lambda: rng.uniform(-1.0, 1.0))
    state = {"x": 0.0}

    def reset():
        state["x"] = 0.0

    def sample():
        state["x"] = state["x"] - theta * state["x"] * dt + sigma * np.sqrt(dt) * rng.standard_normal()
        return state["x"]

    return reset, sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seed = 0
    n_ep = 10 if args.quick else 100
    s_ou = ou_std(0.15, 0.2, 1.0)
    configs = [
        ("Gaussian, std 0.1 (TD3 default)", dict(kind="gaussian", std=0.1)),
        (f"Gaussian, std {s_ou:.2f}", dict(kind="gaussian", std=s_ou)),
        ("OU kappa=0.15 sigma=0.2 dt=1 (DDPG paper)", dict(kind="ou", theta=0.15, sigma=0.2, dt=1.0)),
        ("OU kappa=0.15 sigma=0.2 dt=0.01", dict(kind="ou", theta=0.15, sigma=0.2, dt=0.01)),
        ("uniform on [-1, 1] (warm-up)", dict(kind="uniform")),
    ]
    print(f"Exploration noise on Pendulum-v1 with a zero policy | seed {seed} | {n_ep} episodes per config")
    print(f"OU stationary std: dt=1 -> {ou_std(0.15, 0.2, 1.0):.4f}, dt=0.01 -> {ou_std(0.15, 0.2, 0.01):.4f}; "
          f"lag-1 autocorrelation: dt=1 -> {1 - 0.15:.3f}, dt=0.01 -> {1 - 0.0015:.4f} "
          f"(correlation time {1 / 0.15:.1f} vs {1 / 0.0015:.0f} steps)")

    nb = 30
    results = []
    print(f"\n{'noise process':42s} {'emp. std':>8s} {'lag-1 corr':>10s} {'P(rise above horizontal)':>22s} "
          f"{'mean max height':>16s} {'state cells visited':>20s}")
    for name, kw in configs:
        rng = np.random.default_rng(seed)
        reset_noise, sample = make_noise(rng=rng, **kw)
        env = make_env("Pendulum-v1")
        visited = np.zeros((nb, nb), dtype=bool)
        max_heights, reached, noise_trace, all_noise = [], 0, [], []
        for ep in range(n_ep):
            env.reset(seed=seed + ep)
            # start hanging straight down, almost at rest (same start states for every noise process)
            srng = np.random.default_rng(1000 + ep)
            th0, thd0 = np.pi + srng.uniform(-0.1, 0.1), srng.uniform(-0.1, 0.1)
            env.unwrapped.state = np.array([th0, thd0])
            obs = np.array([np.cos(th0), np.sin(th0), thd0], dtype=np.float32)
            reset_noise()
            best = -1.0
            for t in range(200):
                n = sample()
                all_noise.append(n)
                if ep == 0:
                    noise_trace.append(n)
                obs, r, term, trunc, _ = env.step(np.array([np.clip(n, -1.0, 1.0)], dtype=np.float32))
                th = np.arctan2(obs[1], obs[0])
                i = min(int((th + np.pi) / (2 * np.pi) * nb), nb - 1)
                j = min(int((obs[2] + 8.0) / 16.0 * nb), nb - 1)
                visited[i, j] = True
                best = max(best, float(obs[0]))           # cos(theta): 1 = upright, -1 = hanging
            max_heights.append(best)
            reached += best > 0.0                     # rose above the horizontal
        env.close()
        x = np.array(all_noise)
        lag1 = np.corrcoef(x[:-1], x[1:])[0, 1]
        res = dict(name=name, trace=np.array(noise_trace), heights=np.array(max_heights),
                   frac=reached / n_ep, cover=visited.mean(), visited=visited, std=x.std(), lag1=lag1)
        results.append(res)
        print(f"{name:42s} {res['std']:8.3f} {lag1:10.3f} {res['frac']:22.2f} {np.mean(max_heights):16.3f} "
              f"{res['cover']:20.3f}")

    print("\n=== Summary ===")
    g, o = results[1], results[2]
    print(f"Equal marginal std ({s_ou:.2f}): OU (dt=1) rises above the horizontal in {o['frac']:.0%} of "
          f"episodes and visits {o['cover']:.1%} of state cells; i.i.d. Gaussian: {g['frac']:.0%} and {g['cover']:.1%}.")

    if not args.quick:
        from plot_style import C, setup
        plt = setup()
        # short legend labels (the chapter's kappa is the OU mean-reversion rate)
        short = ["Gauss 0.1", f"Gauss {s_ou:.2f}", "OU \u03ba=0.15, \u0394t=1", "OU \u0394t=0.01", "uniform"]
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.9), gridspec_kw={"width_ratios": [1.3, 1, 1]})
        ax = axes[0]
        for k, idx in enumerate([1, 2, 3]):
            ax.plot(results[idx]["trace"], color=C[idx], lw=1.2 if k == 0 else 1.8, label=short[idx])
        ax.set_xlabel("time step (one episode)")
        ax.set_ylabel("noise (normalised torque)")
        ax.set_title("noise sample paths")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.45 * (hi - lo))       # headroom so the legend does not cover the paths
        ax.legend(fontsize=8, loc="upper center", ncol=3)
        ax = axes[1]
        for k, res in enumerate(results):
            h = np.sort(res["heights"])
            ax.plot(h, np.arange(1, len(h) + 1) / len(h), color=C[k], marker="os^dv"[k], ms=3, markevery=10,
                    label=short[k])
        ax.set_xlabel("max height reached in the episode (cos angle; 1 = upright)")
        ax.set_ylabel("fraction of episodes <= x")
        ax.set_title("how high does pure noise swing the pendulum?")
        ax.legend(fontsize=8, loc="lower right")
        ax = axes[2]
        ax.bar(range(len(results)), [r["cover"] for r in results], color=[C[k] for k in range(len(results))])
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels([x.replace(", ", "\n") for x in short], fontsize=8)
        ax.set_ylabel(f"fraction of {nb}x{nb} (angle, velocity) cells visited")
        ax.set_title(f"state-space coverage over {n_ep} episodes")
        fig.tight_layout()
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, "exploration_noise.png")
        fig.savefig(path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
