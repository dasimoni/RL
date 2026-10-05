"""Gradient Monte Carlo and semi-gradient TD with state aggregation on the
1000-state random walk (Chapter 08, Sections 2-4; cf. Sutton & Barto Figs 9.1, 9.2).

What it shows
  (a) Gradient MC (Section 3.1) with 10 groups of 100 states converges to the
      VE-minimising staircase, which over-weights states with large mu(s).
  (b) Semi-gradient TD(0) (Section 3.2) converges to a *different* staircase:
      the TD fixed point w_TD = A^{-1} b (Section 4), which has larger VE.
  (c) n-step semi-gradient TD with 20 groups: early-learning error as a function
      of alpha and n -- intermediate n is best, just as in the tabular case.

Run:  python code/ch08_function_approximation/random_walk_aggregation.py [--quick]
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


# --------------------------------------------------------------------------- #
# Algorithms (one-hot features => v_hat(s,w) = w[group(s)], grad = unit vector)
# --------------------------------------------------------------------------- #
def gradient_mc_aggregation(n_groups, alpha, n_episodes, rng):
    """Gradient MC, Section 3.1 box. With aggregation the update
    w <- w + alpha [G_t - v_hat(S_t,w)] x(S_t) touches one component only."""
    size = rw.N_STATES // n_groups
    w = [0.0] * n_groups
    for _ in range(n_episodes):
        states, G = rw.generate_episode(rng)       # gamma = 1: G_t = final reward
        for s in states:
            g = (s - 1) // size
            w[g] += alpha * (G - w[g])
    return np.array(w)


def semi_gradient_td0_aggregation(n_groups, alpha, n_episodes, rng):
    """Semi-gradient TD(0), Section 3.2 box (v_hat(terminal) = 0)."""
    size = rw.N_STATES // n_groups
    w = [0.0] * n_groups
    for _ in range(n_episodes):
        states, R_T = rw.generate_episode(rng)
        T = len(states)
        for t in range(T):
            g = (states[t] - 1) // size
            if t + 1 < T:                              # non-terminal successor, reward 0
                target = w[(states[t + 1] - 1) // size]
            else:                                      # terminal: target is just R_T
                target = R_T
            w[g] += alpha * (target - w[g])
    return np.array(w)


def nstep_td_episode(w, states, R_T, n, alpha, size):
    """One episode of n-step semi-gradient TD (gamma = 1, reward only at the end).
    G_{tau:tau+n} = R_T if tau+n >= T else v_hat(S_{tau+n}). Updates are applied in
    the same order as the online algorithm (tau = 0, 1, ..., T-1)."""
    T = len(states)
    for tau in range(T):
        if tau + n >= T:
            G = R_T
        else:
            G = w[(states[tau + n] - 1) // size]
        g = (states[tau] - 1) // size
        w[g] += alpha * (G - w[g])


def nstep_sweep(n_values, alphas, n_runs, n_episodes, seed):
    """Average (over first n_episodes and runs) of the unweighted RMS error, 20 groups."""
    n_groups = 20
    size = rw.N_STATES // n_groups
    v = rw.exact()["v"]
    # per-group sufficient statistics so the RMS error over all 1000 states is O(groups)
    sum_v = v.reshape(n_groups, size).sum(1)
    sum_v2 = (v ** 2).reshape(n_groups, size).sum(1)
    err = np.zeros((len(n_values), len(alphas)))
    for run in range(n_runs):
        rng = random.Random(seed + run)
        episodes = [rw.generate_episode(rng) for _ in range(n_episodes)]  # common random numbers
        for i, n in enumerate(n_values):
            for j, a in enumerate(alphas):
                w = [0.0] * n_groups
                tot = 0.0
                for states, R_T in episodes:
                    nstep_td_episode(w, states, R_T, n, a, size)
                    wa = np.array(w)
                    mse = (sum_v2 - 2 * wa * sum_v + size * wa ** 2).sum() / rw.N_STATES
                    tot += np.sqrt(mse)
                err[i, j] += tot / n_episodes
    return err / n_runs


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figures")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n_ep = 5_000 if args.quick else 100_000
    nstep_runs = 5 if args.quick else 100
    alpha_mc, alpha_td = 2e-5, 1e-4
    n_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    alphas = np.round(np.concatenate([[0.0, 0.025, 0.05, 0.075], np.linspace(0.1, 1.0, 10)]), 3)
    if args.quick:
        n_values = [1, 4, 16]
        alphas = alphas[::3]
    print(f"seed={args.seed} quick={args.quick}")
    print(f"(a,b) 10 groups, {n_ep} episodes, alpha_MC={alpha_mc}, alpha_TD={alpha_td}")
    print(f"(c) 20 groups, n in {n_values}, {len(alphas)} alphas, {nstep_runs} runs x 10 episodes")

    e = rw.exact()
    X10 = rw.aggregation_features(10)
    w_proj = rw.projection_solution(X10)
    w_td = rw.td_fixed_point(X10)

    t0 = time.time()
    w_mc = gradient_mc_aggregation(10, alpha_mc, n_ep, random.Random(args.seed))
    t_mc = time.time() - t0
    t0 = time.time()
    w_tdl = semi_gradient_td0_aggregation(10, alpha_td, n_ep, random.Random(args.seed + 1))
    t_td = time.time() - t0

    print(f"\nGradient MC ({t_mc:.1f}s):     VE(learned) = {rw.ve(X10, w_mc):.5f}"
          f"   min_w VE = VE(projection) = {rw.ve(X10, w_proj):.5f}")
    print(f"Semi-grad TD(0) ({t_td:.1f}s): VE(learned) = {rw.ve(X10, w_tdl):.5f}"
          f"   VE(w_TD exact)          = {rw.ve(X10, w_td):.5f}")
    D, Pm = e["mu"], e["P"]
    for name, Xf in [("10 groups", X10), ("Fourier order 5", rw.fourier_features(5))]:
        A = Xf.T @ (D[:, None] * (Xf - Pm @ Xf))                 # A = X^T D (I - P) X, gamma = 1
        print(f"A for {name:16s}: smallest eigenvalue of (A + A^T)/2 = "
              f"{np.linalg.eigvalsh((A + A.T) / 2).min():.3e}  (> 0: positive definite)")
    # Where does the asymmetry of the TD staircase come from? Re-solve with the start state
    # split 50/50 between states 500 and 501, which makes mu symmetric about 500.5.
    h = np.zeros(rw.N_STATES)
    h[[rw.START - 1, rw.START]] = 0.5
    eta = np.linalg.solve((np.eye(rw.N_STATES) - Pm).T, h)
    mu_sym = eta / eta.sum()
    A_sym = X10.T @ (mu_sym[:, None] * (X10 - Pm @ X10))
    w_td_sym = np.linalg.solve(A_sym, X10.T @ (mu_sym * e["r"]))
    print("exact w_TD with the start split between states 500 and 501:", np.round(w_td_sym, 3).tolist())
    print("group :   w_MC  w_proj  (unweighted mean of v) |  w_TD(learned)  w_TD(exact)")
    for g in range(10):
        vbar = e["v"][100 * g:100 * g + 100].mean()
        print(f"  {g:2d}  : {w_mc[g]:+.3f}  {w_proj[g]:+.3f}  ({vbar:+.3f})              |"
              f"    {w_tdl[g]:+.3f}       {w_td[g]:+.3f}")

    t0 = time.time()
    err = nstep_sweep(n_values, alphas, nstep_runs, 10, args.seed + 1000)
    t_ns = time.time() - t0
    best = np.unravel_index(np.argmin(err), err.shape)
    print(f"\nn-step sweep ({t_ns:.1f}s). Best: n={n_values[best[0]]}, alpha={alphas[best[1]]},"
          f" avg RMS error={err[best]:.3f}")
    for i, n in enumerate(n_values):
        j = int(np.argmin(err[i]))
        print(f"  n={n:3d}: best alpha={alphas[j]:.3f}, error={err[i, j]:.3f}")

    if args.quick:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    s = np.arange(1, rw.N_STATES + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, w_l, w_x, name, lab in [
        (axes[0], w_mc, w_proj, "Gradient MC", "min-VE solution (projection)"),
        (axes[1], w_tdl, w_td, "Semi-gradient TD(0)", r"TD fixed point $A^{-1}b$")]:
        ax.plot(s, e["v"], "k", lw=1.5, label=r"true value $v_\pi$")
        ax.plot(s, X10 @ w_l, color="tab:blue", lw=2, label=f"{name}, learned")
        ax.plot(s, X10 @ w_x, "--", color="tab:orange", lw=1.5, label=lab)
        ax.set_xlabel("state")
        ax.set_ylabel("value")
        ax.set_ylim(-1, 1)
        ax.set_title(f"{name}, 10 groups, {n_ep:,} episodes\nVE = {rw.ve(X10, w_l):.4f}")
        ax2 = ax.twinx()
        ax2.fill_between(s, e["mu"], color="gray", alpha=0.25, step="mid")
        ax2.set_ylim(0, e["mu"].max() * 3)
        ax2.set_ylabel(r"on-policy distribution $\mu(s)$", color="gray")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "rw_aggregation_mc_vs_td.png"), dpi=110)

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    cmap = plt.get_cmap("viridis")
    for i, n in enumerate(n_values):
        ax.plot(alphas, err[i], ".-", ms=3, color=cmap(i / (len(n_values) - 1)), label=f"n={n}")
    ax.set_ylim(0.25, 0.55)
    ax.set_xlabel(r"step size $\alpha$")
    ax.set_ylabel("RMS error over 1000 states\n(avg. over first 10 episodes)")
    ax.set_title(f"n-step semi-gradient TD, 20 groups ({nstep_runs} runs)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "rw_nstep_sweep.png"), dpi=110)
    print(f"figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
