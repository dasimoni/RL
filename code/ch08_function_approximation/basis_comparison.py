"""Polynomial vs Fourier basis vs tile coding with gradient Monte Carlo on the
1000-state random walk (Chapter 08, Section 6; cf. Sutton & Barto Figs 9.5, 9.10).

Gradient MC (Section 3.1) is run with several fixed feature sets. Because the
policy is fixed, trajectories do not depend on the weights, so we generate every
run's stream of (S_t, G_t) pairs once and then update the weights of all runs in
lock-step with numpy (one row of W per run). Every feature set sees the same
trajectories (common random numbers), which makes the comparison sharper.

For each basis we also print the best achievable error sqrt(min_w VE(w)), i.e.
what an infinitely patient learner would reach, so you can separate "the basis
cannot represent v_pi" from "SGD is slow with this basis".

Run:  python code/ch08_function_approximation/basis_comparison.py [--quick]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread: the machine is shared
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import random
import time

import numpy as np

import rw1000 as rw

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def make_streams(n_runs: int, n_episodes: int, seed: int):
    """Concatenate each run's episodes into one stream of (state, return) pairs."""
    runs = []
    for r in range(n_runs):
        rng = random.Random(seed + r)
        S, G, end = [], [], []
        for _ in range(n_episodes):
            states, ret = rw.generate_episode(rng)
            S.extend(states)
            G.extend([ret] * len(states))          # gamma = 1: G_t = final reward
            end.extend([False] * (len(states) - 1) + [True])
        runs.append((S, G, end))
    L = max(len(s) for s, _, _ in runs)
    S = np.ones((n_runs, L), dtype=np.int16)
    G = np.zeros((n_runs, L), dtype=np.float32)
    M = np.zeros((n_runs, L), dtype=np.float32)    # 1 where the stream is still alive
    E = np.zeros((n_runs, L), dtype=bool)          # True at the last step of an episode
    for r, (s, g, e) in enumerate(runs):
        n = len(s)
        S[r, :n], G[r, :n], M[r, :n], E[r, :n] = s, g, 1.0, e
    return S, G, M, E


def gradient_mc_batched(X, alpha, S, G, M, E, n_episodes):
    """Gradient MC (Section 3.1 box), all runs at once.
    w <- w + alpha [G_t - x(S_t)^T w] x(S_t). Returns weights after every episode."""
    R, L = S.shape
    d = X.shape[1]
    W = np.zeros((R, d))
    hist = np.zeros((R, n_episodes, d))
    count = np.zeros(R, dtype=int)
    for j in range(L):
        Xj = X[S[:, j] - 1]                        # (R, d) feature vectors x(S_t)
        err = G[:, j] - np.einsum("rd,rd->r", Xj, W)
        W += (alpha * err * M[:, j])[:, None] * Xj
        done = np.flatnonzero(E[:, j])
        if done.size:
            hist[done, count[done]] = W[done]
            count[done] += 1
    return hist


def ve_curve(X, hist):
    """VE(w) = w^T C w - 2 w^T c + const, with C = X^T D X, c = X^T D v: O(d^2) per evaluation."""
    e = rw.exact()
    D, v = e["mu"], e["v"]
    C = X.T @ (D[:, None] * X)
    c = X.T @ (D * v)
    const = D @ v ** 2
    ve = np.einsum("rei,ij,rej->re", hist, C, hist) - 2 * hist @ c + const
    return np.maximum(ve, 0.0)                     # (runs, episodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figures")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n_runs, n_ep = (3, 300) if args.quick else (30, 5000)

    configs = []   # (name, X, alpha, group)
    for order in (5, 10, 20):
        configs.append((f"polynomial, order {order}", rw.polynomial_features(order), 1e-4, "poly"))
    for order in (5, 10, 20):
        configs.append((f"Fourier, order {order}", rw.fourier_features(order), 5e-5, "fourier"))
    configs.append(("tile coding, 1 tiling (200-state tiles)", rw.tile_features(1), 1e-4, "tile"))
    configs.append(("tile coding, 50 tilings (offset 4)", rw.tile_features(50), 1e-4 / 50, "tile"))
    if args.quick:
        configs = [configs[1], configs[4], configs[7]]

    print(f"seed={args.seed} quick={args.quick} runs={n_runs} episodes={n_ep}")
    t0 = time.time()
    S, G, M, E = make_streams(n_runs, n_ep, args.seed)
    print(f"generated streams: {S.shape[1]:,} steps per run (max) in {time.time() - t0:.1f}s\n")

    results = {}
    n_upd = float(M.sum(1).mean())                 # updates per run (= time steps)
    print(f"{'features':42s} {'d':>4s} {'alpha':>9s} {'cond(C)':>9s} {'sqrtVE@100':>11s} {'sqrtVE@end':>11s}"
          f" {'noise-free':>10s} {'sqrt(minVE)':>12s} {'time':>6s}")
    for name, X, alpha, group in configs:
        t0 = time.time()
        hist = gradient_mc_batched(X, alpha, S, G, M, E, n_ep)
        ve = ve_curve(X, hist)
        curve = np.sqrt(ve.mean(0))                # root of the run-averaged VE
        best = np.sqrt(rw.ve(X, rw.projection_solution(X)))
        results[name] = (curve, best, group)
        k = min(99, n_ep - 1)
        # condition number of C = E_mu[x x^T]: SGD's speed along the slowest direction scales with
        # alpha * lambda_min, while stability caps alpha at about 2 / lambda_max (Section 7)
        # (eigenvalues below 1e-12 * lambda_max are dropped: exactly redundant directions, e.g. the
        # tilings' indicators each sum to one, never change v_hat and do not slow learning down)
        ev, U = np.linalg.eigh(X.T @ (rw.exact()["mu"][:, None] * X))
        cond = ev.max() / ev[ev > 1e-12 * ev.max()].min()
        # "noise-free": sqrt VE of the EXPECTED gradient-MC iteration w <- w + alpha (c - C w) from
        # w = 0 after the same number of updates. Along eigenvector u_i of C the excess error decays
        # like (1 - alpha lambda_i)^t, so this isolates slow (ill-conditioned) learning from noise.
        w_star = rw.projection_solution(X)
        excess = np.sum(ev * (U.T @ w_star) ** 2 * (1 - alpha * ev) ** (2 * n_upd))
        nf = np.sqrt(best ** 2 + excess)
        print(f"{name:42s} {X.shape[1]:4d} {alpha:9.2e} {cond:9.1e} {curve[k]:11.4f} {curve[-1]:11.4f}"
              f" {nf:10.4f} {best:12.4f} {time.time() - t0:5.1f}s")

    if args.quick:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    ep = np.arange(1, n_ep + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    styles = {5: ":", 10: "--", 20: "-"}
    for name, (curve, best, group) in results.items():
        if group in ("poly", "fourier"):
            order = int(name.split()[-1])
            col = "tab:red" if group == "poly" else "tab:blue"
            axes[0].plot(ep, curve, styles[order], color=col, label=name)
    axes[0].set_xlabel("episodes")
    axes[0].set_ylabel(r"$\sqrt{\overline{VE}}$ (avg. over runs)")
    axes[0].set_title(f"Gradient MC: polynomial vs Fourier ({n_runs} runs)")
    axes[0].set_ylim(0, 0.45)
    axes[0].legend(fontsize=8)
    for name, (curve, best, group) in results.items():
        if group == "tile":
            line, = axes[1].plot(ep, curve, label=name)
            axes[1].axhline(best, ls=":", color=line.get_color(), lw=1)
    axes[1].set_xlabel("episodes")
    axes[1].set_ylabel(r"$\sqrt{\overline{VE}}$ (avg. over runs)")
    axes[1].set_title("Gradient MC: one tiling vs 50 tilings\n(dotted: best achievable)")
    axes[1].set_ylim(0, 0.45)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "basis_comparison.png"), dpi=110)
    print(f"figure saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
