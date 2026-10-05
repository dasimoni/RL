"""Visualize accumulating, replacing and dutch eligibility traces over one episode.

Chapter 06, Section 9.2. One episode of the 19-state random walk is generated
(seed 0, lambda = 0.9, gamma = 1). For each kind of trace we record z_t(s) for every
time step t and state s (the trace *after* the update at time t, i.e. the vector
that multiplies delta_t):

    accumulating:  z_t = gamma lambda z_{t-1} + e_{S_t}
    replacing:     z_t = gamma lambda z_{t-1}, then z_t(S_t) = 1
    dutch:         z_t = gamma lambda z_{t-1} + (1 - alpha gamma lambda z_{t-1}(S_t)) e_{S_t}   (alpha = 0.2)

Traces do not depend on the value estimates, only on the visited states (and, for
the dutch trace, on alpha), so no learning is needed to draw them. The last column,
z_{T-1}, is the credit that each state receives for the final, rewarded transition.

Outputs (full mode): figures/eligibility_traces.png

Run:  python code/ch06_n_step_and_eligibility_traces/trace_visualization.py [--quick]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import random_walk19 as rw

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def trace_history(states, lam, gamma=1.0, kind="accumulating", alpha=0.2):
    """Return Z with Z[t, s] = z_t(s) for t = 0..T-1 and s = 0..20."""
    T = len(states) - 1
    z = np.zeros(rw.N_TOTAL)
    Z = np.zeros((T, rw.N_TOTAL))
    for t in range(T):
        s = states[t]
        zs = z[s]
        z *= gamma * lam
        if kind == "accumulating":
            z[s] += 1.0
        elif kind == "replacing":
            z[s] = 1.0
        elif kind == "dutch":
            z[s] += 1.0 - alpha * gamma * lam * zs
        else:
            raise ValueError(kind)
        Z[t] = z
    return Z


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    lam, alpha, gamma = 0.9, 0.2, 1.0
    rng = np.random.default_rng(args.seed)
    states, rewards = rw.generate_episode(rng)
    T = len(rewards)
    print(f"Eligibility traces on one random-walk episode | seed={args.seed} lambda={lam} "
          f"gamma={gamma} dutch alpha={alpha}")
    print(f"episode length T={T}, terminated {'right (+1)' if rewards[-1] > 0 else 'left (-1)'}, "
          f"distinct states visited={len(set(states[:-1].tolist()))}")

    kinds = ["accumulating", "replacing", "dutch"]
    Z = {k: trace_history(states, lam, gamma, k, alpha) for k in kinds}
    counts = np.bincount(states[:-1], minlength=rw.N_TOTAL)
    s_star = int(np.argmax(counts))
    print(f"most visited state: {s_star} ({counts[s_star]} visits)")
    print("\n             max_t z_t(s)   bound          z at final step for s*")
    bounds = {"accumulating": 1 / (1 - gamma * lam), "replacing": 1.0,
              "dutch": 1 / (1 - (1 - alpha) * gamma * lam)}
    for k in kinds:
        print(f"  {k:12s}  {Z[k].max():8.3f}     {bounds[k]:8.3f}       {Z[k][-1, s_star]:.4f}")
        assert Z[k].max() <= bounds[k] + 1e-12
    final = {k: Z[k][-1, 1:-1] for k in kinds}
    print("\ncredit for the final reward, z_{T-1}(s), states 1..19:")
    for k in kinds:
        print(f"  {k:12s} " + " ".join(f"{v:4.2f}" for v in final[k]))
    print(f"sum of final traces: " + ", ".join(f"{k} {final[k].sum():.3f}" for k in kinds))

    if args.quick:
        return
    from matplotlib.colors import LinearSegmentedColormap
    from plot_style import setup, C, BLUES, MARKERS
    plt = setup()
    cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb"] + BLUES)
    vmax = max(Z[k].max() for k in kinds)
    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.6)
    for i, k in enumerate(kinds):
        ax = fig.add_subplot(gs[0, 2 * i: 2 * i + 2])
        im = ax.imshow(Z[k][:, 1:-1].T, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                       extent=(-0.5, T - 0.5, 0.5, rw.N_STATES + 0.5), interpolation="nearest")
        ax.plot(np.arange(T), states[:-1], color=C[1], lw=0.6, alpha=0.8)
        ax.grid(False)
        ax.set_yticks([1, 5, 10, 15, 19])
        ax.set_xlabel("time step t")
        ax.set_ylabel("state s")
        title = {"accumulating": "accumulating trace", "replacing": "replacing trace",
                 "dutch": rf"dutch trace ($\alpha$ = {alpha})"}[k]
        ax.set_title(f"{title}: max {Z[k].max():.2f}")
    cb = fig.colorbar(im, ax=fig.axes[:3], fraction=0.02, pad=0.01)
    cb.set_label(r"$z_t(s)$")

    ax = fig.add_subplot(gs[1, 0:3])
    for i, k in enumerate(kinds):
        ax.plot(np.arange(T), Z[k][:, s_star], color=C[i], lw=1.4, label=k)
    visits = np.flatnonzero(states[:-1] == s_star)
    ax.plot(visits, np.zeros_like(visits) - 0.08, "|", color="#52514e", ms=8, label=f"visits to s = {s_star}")
    ax.set_xlabel("time step t")
    ax.set_ylabel(rf"$z_t({s_star})$")
    ax.set_title(f"trace of the most visited state (s = {s_star}, {counts[s_star]} visits)")
    ax.legend(loc="upper right", ncol=2)

    ax = fig.add_subplot(gs[1, 3:6])
    x = np.arange(1, rw.N_STATES + 1)
    width = 0.27
    for i, k in enumerate(kinds):
        ax.bar(x + (i - 1) * width, final[k], width=width, color=C[i], label=k)
    ax.set_xticks(x)
    ax.set_xlabel("state s")
    ax.set_ylabel(r"$z_{T-1}(s)$")
    ax.set_title("credit each state receives for the final reward")
    ax.legend(loc="upper left")
    fig.suptitle(f"Eligibility traces over one 19-state random-walk episode (λ = {lam}, γ = 1, T = {T})",
                 y=0.98)
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, "eligibility_traces.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
