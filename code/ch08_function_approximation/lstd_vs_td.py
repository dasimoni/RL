"""LSTD vs semi-gradient TD(0) on the 1000-state random walk (Chapter 08, Sections 4, 9, 15).

Part 1 -- what do the different objectives converge to? For several feature sets we
          compute *exactly* (linear algebra on the known model):
            * argmin VE           (limit of gradient MC),
            * w_TD = A^{-1} b     (limit of semi-gradient TD(0) and of LSTD),
            * argmin TDE          (limit of the naive residual-gradient algorithm,
                                   which descends the mean squared TD error),
            * argmin BE           (limit of the residual-gradient algorithm with
                                   double sampling).
Part 1b -- the TD(lambda) fixed point (Section 4.5): solves X w = Pi T^lambda X w, i.e.
          A_lam w = b_lam with A_lam = X^T D (I - lam P)^{-1} (I - P) X and
          b_lam = X^T D (I - lam P)^{-1} r (gamma = 1). lam = 0 is TD(0), lam = 1 the MC solution.
Part 2 -- learning curves: LSTD (O(d^2) per step, Sherman-Morrison, no step size)
          vs TD(0) and linear semi-gradient TD(lambda) (accumulating traces, Section 3.5 box)
          with several step sizes, all runs in lock-step with numpy.
          Polynomial features are badly conditioned: TD(0) suffers, LSTD does not.
Part 3 -- measured wall-clock cost per update as a function of d: O(d) vs O(d^2).

Run:  python code/ch08_function_approximation/lstd_vs_td.py [--quick]
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
# Part 1: exact limits of different objectives (gamma = 1, episodic)
# --------------------------------------------------------------------------- #
def transitions():
    """All 200 equally likely outcomes from each state: (next index or -1 if terminal, reward)."""
    nxt = np.zeros((rw.N_STATES, 2 * rw.JUMP), dtype=int)
    rew = np.zeros((rw.N_STATES, 2 * rw.JUMP))
    for i in range(rw.N_STATES):
        s = i + 1
        for j, k in enumerate(list(range(-rw.JUMP, 0)) + list(range(1, rw.JUMP + 1))):
            s2 = s + k
            if s2 < 1:
                nxt[i, j], rew[i, j] = -1, -1.0
            elif s2 > rw.N_STATES:
                nxt[i, j], rew[i, j] = -1, 1.0
            else:
                nxt[i, j] = s2 - 1
    return nxt, rew


def exact_limits(X):
    e = rw.exact()
    mu, P, r = e["mu"], e["P"], e["r"]
    out = {"argmin VE (gradient MC)": rw.projection_solution(X),
           "w_TD (semi-gradient TD, LSTD)": rw.td_fixed_point(X)}
    # BE(w) = sum_s mu(s) (r(s) + sum_s' P(s,s') xhat(s') w - x(s) w)^2 : weighted least squares
    M = X - P @ X
    sw = np.sqrt(mu)
    out["argmin BE (residual gradient)"] = np.linalg.lstsq(sw[:, None] * M, sw * r, rcond=None)[0]
    # TDE(w) = sum_s mu(s) mean_outcomes (R + x(S')w - x(s)w)^2, x(terminal) = 0
    nxt, rew = transitions()
    Xp = np.vstack([X, np.zeros(X.shape[1])])               # row -1 -> terminal -> zeros
    rows = (X[:, None, :] - Xp[nxt]).reshape(-1, X.shape[1])  # (x(s) - x(s')) per outcome
    wts = np.repeat(np.sqrt(mu / (2 * rw.JUMP)), 2 * rw.JUMP)
    out["argmin TDE (naive residual gradient)"] = np.linalg.lstsq(
        wts[:, None] * rows, wts * rew.reshape(-1), rcond=None)[0]
    return out


def td_lambda_fixed_point(X, lam):
    """Exact TD(lambda) fixed point for the episodic random walk (gamma = 1, Section 4.5):
    T^lambda v = (I - lam P)^{-1} (r + (1 - lam) P v), so v - T^lambda v = (I - lam P)^{-1}((I - P) v - r)."""
    e = rw.exact()
    mu, P, r = e["mu"], e["P"], e["r"]
    I = np.eye(rw.N_STATES)
    L = np.linalg.solve(I - lam * P, np.column_stack([(I - P) @ X, r]))
    A = X.T @ (mu[:, None] * L[:, :-1])
    b = X.T @ (mu * L[:, -1])
    return np.linalg.solve(A, b)


# --------------------------------------------------------------------------- #
# Part 2: streams of transitions and batched learners
# --------------------------------------------------------------------------- #
def make_transition_streams(n_runs, n_episodes, seed):
    """Per run: S_t, R_{t+1}, S_{t+1} (0 = terminal), episode-end flags; padded to equal length."""
    runs = []
    for k in range(n_runs):
        rng = random.Random(seed + k)
        S, S2, R, E = [], [], [], []
        for _ in range(n_episodes):
            states, ret = rw.generate_episode(rng)
            S += states
            S2 += states[1:] + [0]
            R += [0.0] * (len(states) - 1) + [ret]
            E += [False] * (len(states) - 1) + [True]
        runs.append((S, S2, R, E))
    L = max(len(r[0]) for r in runs)
    S = np.ones((n_runs, L), dtype=np.int64)
    S2 = np.zeros((n_runs, L), dtype=np.int64)
    R = np.zeros((n_runs, L))
    M = np.zeros((n_runs, L))
    E = np.zeros((n_runs, L), dtype=bool)
    for k, (s, s2, r, e) in enumerate(runs):
        n = len(s)
        S[k, :n], S2[k, :n], R[k, :n], M[k, :n], E[k, :n] = s, s2, r, 1.0, e
    return S, S2, R, M, E


def run_batched(X, streams, n_episodes, method, alpha=None, eps=1.0, lam=0.0):
    """Semi-gradient TD(0), linear TD(lambda) or incremental LSTD (Section 9 box), all runs
    at once. Returns the weights at the end of every episode, shape (runs, episodes, d)."""
    S, S2, R, M, E = streams
    n_runs, L = S.shape
    d = X.shape[1]
    Xp = np.vstack([np.zeros(d), X])             # index 0 = terminal state -> x = 0
    W = np.zeros((n_runs, d))
    Z = np.zeros((n_runs, d))                    # eligibility traces (TD(lambda) only)
    Ainv = np.repeat(np.eye(d)[None] / eps, n_runs, axis=0)   # \hat A^{-1}, \hat A_0 = eps I
    bvec = np.zeros((n_runs, d))
    hist = np.zeros((n_runs, n_episodes, d))
    count = np.zeros(n_runs, dtype=int)
    for j in range(L):
        x = Xp[S[:, j]]
        x2 = Xp[S2[:, j]]
        m = M[:, j]
        if method == "td":
            delta = R[:, j] + np.einsum("rd,rd->r", x2, W) - np.einsum("rd,rd->r", x, W)
            W += (alpha * delta * m)[:, None] * x
        elif method == "tdlam":                  # Section 3.5 box, gamma = 1, accumulating trace
            Z = lam * Z + x * m[:, None]
            delta = R[:, j] + np.einsum("rd,rd->r", x2, W) - np.einsum("rd,rd->r", x, W)
            W += (alpha * delta * m)[:, None] * Z
        else:
            u = (x - x2) * m[:, None]                          # x_t - gamma x_{t+1}, gamma = 1
            Ax = np.einsum("rij,rj->ri", Ainv, x * m[:, None])  # \hat A^{-1} x
            v = np.einsum("rji,rj->ri", Ainv, u)               # \hat A^{-T} (x - gamma x')
            denom = 1.0 + np.einsum("ri,ri->r", v, x * m[:, None])
            Ainv -= Ax[:, :, None] * v[:, None, :] / denom[:, None, None]   # Sherman-Morrison
            bvec += (R[:, j] * m)[:, None] * x
            W = np.einsum("rij,rj->ri", Ainv, bvec)
        done = np.flatnonzero(E[:, j])
        if done.size:
            hist[done, count[done]] = W[done]
            count[done] += 1
            Z[done] = 0.0                        # traces start from zero in every episode
    return hist


def sqrt_ve_curve(X, hist):
    e = rw.exact()
    D, v = e["mu"], e["v"]
    C = X.T @ (D[:, None] * X)
    c = X.T @ (D * v)
    with np.errstate(over="ignore", invalid="ignore"):
        ve = np.einsum("rei,ij,rej->re", hist, C, hist) - 2 * hist @ c + D @ v ** 2
    return np.sqrt(np.maximum(ve, 0).mean(0))      # nan/inf if some run diverged


# --------------------------------------------------------------------------- #
# Part 3: cost per update
# --------------------------------------------------------------------------- #
def time_per_update(d, n_updates, rng):
    x = rng.normal(size=(n_updates + 1, d)) / np.sqrt(d)
    w = np.zeros(d)
    t0 = time.perf_counter()
    for t in range(n_updates):
        delta = 0.1 + x[t + 1] @ w - x[t] @ w
        w += 0.01 * delta * x[t]
    t_td = (time.perf_counter() - t0) / n_updates
    Ainv, b = np.eye(d), np.zeros(d)
    t0 = time.perf_counter()
    for t in range(n_updates):
        u = x[t] - 0.9 * x[t + 1]
        Ax = Ainv @ x[t]
        v = u @ Ainv
        Ainv -= np.outer(Ax, v) / (1.0 + v @ x[t])
        b += 0.1 * x[t]
        w = Ainv @ b
    t_lstd = (time.perf_counter() - t0) / n_updates
    return t_td, t_lstd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figures")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n_runs, n_ep = (3, 30) if args.quick else (30, 300)
    print(f"seed={args.seed} quick={args.quick}; runs={n_runs}, episodes={n_ep}, gamma=1\n")

    # ---------------- Part 1 ----------------
    feats = {"aggregation, 10 groups": rw.aggregation_features(10),
             "aggregation, 20 groups": rw.aggregation_features(20),
             "polynomial, order 5": rw.polynomial_features(5),
             "Fourier, order 5": rw.fourier_features(5),
             "Fourier, order 10": rw.fourier_features(10)}
    print("Part 1: sqrt(VE) at the exact limit of each objective")
    names = None
    for fname, X in (feats.items() if not args.quick else list(feats.items())[:2]):
        lim = exact_limits(X)
        if names is None:
            names = list(lim)
            print(f"{'features':24s}" + "".join(f"{n.split(' (')[0]:>14s}" for n in names))
        print(f"{fname:24s}" + "".join(f"{np.sqrt(rw.ve(X, w)):14.4f}" for w in lim.values()))
    print("  (columns: " + "; ".join(names) + ")")

    # ---------------- Part 1b: TD(lambda) fixed points ----------------
    lams = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
    lam_feats = {k: feats[k] for k in ("aggregation, 10 groups", "aggregation, 20 groups", "Fourier, order 5")}
    if args.quick:
        lams, lam_feats = [0.0, 0.9, 1.0], {"aggregation, 10 groups": feats["aggregation, 10 groups"]}
    print("\nPart 1b: sqrt(VE) at the exact TD(lambda) fixed point (lambda = 1 is the MC solution)")
    print(f"{'features':24s}" + "".join(f"{'lam=' + str(l):>10s}" for l in lams))
    lam_sweep = {}
    for fname, X in lam_feats.items():
        vals = [np.sqrt(rw.ve(X, td_lambda_fixed_point(X, l))) for l in lams]
        lam_sweep[fname] = vals
        print(f"{fname:24s}" + "".join(f"{x:10.4f}" for x in vals))

    # ---------------- Part 2 ----------------
    streams = make_transition_streams(n_runs, n_ep, args.seed)
    print(f"\nPart 2: learning curves (stream length {streams[0].shape[1]:,} steps)")
    setups = {"aggregation, 20 groups": (rw.aggregation_features(20), [0.01, 0.03, 0.1]),
              "polynomial, order 5": (rw.polynomial_features(5), [0.01, 0.03, 0.1])}
    curves = {}
    for fname, (X, alphas) in setups.items():
        floor = np.sqrt(rw.ve(X, rw.td_fixed_point(X)))
        curves[fname] = {"floor": floor}
        for eps in (0.01, 1.0):
            t0 = time.time()
            c = sqrt_ve_curve(X, run_batched(X, streams, n_ep, "lstd", eps=eps))
            curves[fname][f"LSTD, eps={eps}"] = c
            print(f"  {fname:24s} LSTD eps={eps:<5}   sqrtVE after 10 / 100 / {n_ep} episodes ="
                  f" {c[min(9, n_ep - 1)]:.4f} / {c[min(99, n_ep - 1)]:.4f} / {c[-1]:.4f}  ({time.time() - t0:.1f}s)")
        for a in alphas:
            t0 = time.time()
            c = sqrt_ve_curve(X, run_batched(X, streams, n_ep, "td", alpha=a))
            curves[fname][f"TD(0), alpha={a}"] = c
            print(f"  {fname:24s} TD(0) alpha={a:<5} sqrtVE after 10 / 100 / {n_ep} episodes ="
                  f" {c[min(9, n_ep - 1)]:.4f} / {c[min(99, n_ep - 1)]:.4f} / {c[-1]:.4f}  ({time.time() - t0:.1f}s)")
        for lam in (0.5, 0.9):
            best = None
            for a in [0.003] + alphas:                # traces enlarge the effective step: try smaller alpha too
                t0 = time.time()
                c = sqrt_ve_curve(X, run_batched(X, streams, n_ep, "tdlam", alpha=a, lam=lam))
                fin = c[-1] if np.isfinite(c[-1]) else float("inf")
                print(f"  {fname:24s} TD({lam}) alpha={a:<5} sqrtVE after 10 / 100 / {n_ep} episodes ="
                      f" {c[min(9, n_ep - 1)]:.4f} / {c[min(99, n_ep - 1)]:.4f} / {c[-1]:.4f}  ({time.time() - t0:.1f}s)")
                if best is None or fin < best[0]:
                    best = (fin, a, c)
            curves[fname][f"TD({lam}), alpha={best[1]} (best)"] = best[2]
            lim = np.sqrt(rw.ve(X, td_lambda_fixed_point(X, lam)))
            print(f"  {fname:24s} TD({lam}): best alpha={best[1]}, exact sqrt VE at the TD({lam}) fixed point = {lim:.4f}")
        print(f"  {fname:24s} sqrt VE(w_TD) = {floor:.4f}")

    # ---------------- Part 3 ----------------
    ds = [10, 50, 100, 200, 400] if args.quick else [10, 25, 50, 100, 200, 400, 800]
    n_upd = 200 if args.quick else 1000
    rng = np.random.default_rng(args.seed)
    timing = [time_per_update(d, n_upd, rng) for d in ds]
    print("\nPart 3: measured time per update (single thread numpy)")
    for d, (ttd, tls) in zip(ds, timing):
        print(f"  d={d:4d}: TD(0) {ttd * 1e6:8.1f} us   LSTD {tls * 1e6:9.1f} us   ratio {tls / ttd:6.1f}")

    if args.quick:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.6))
    axes = [axs[0, 0], axs[0, 1], axs[1, 1]]
    ep = np.arange(1, n_ep + 1)
    for ax, (fname, cs) in zip(axes[:2], curves.items()):
        for k, c in cs.items():
            if k == "floor":
                ax.axhline(c, color="k", ls=":", lw=1, label=r"$\sqrt{\overline{VE}(\mathbf{w}_{TD})}$")
            else:
                ls = "-" if k.startswith("LSTD") else ("-." if k.startswith("TD(0.") else "--")
                ax.plot(ep, c, ls=ls, label=k)
        ax.set_yscale("log")
        ax.set_xlabel("episodes")
        ax.set_ylabel(r"$\sqrt{\overline{VE}}$ (log scale)")
        ax.set_title(f"{fname} ({n_runs} runs)")
        ax.legend(fontsize=7)
    axl = axs[1, 0]
    for fname, vals in lam_sweep.items():
        axl.plot(lams, vals, "o-", label=fname)
    axl.set_yscale("log")
    axl.set_xlabel(r"$\lambda$")
    axl.set_ylabel(r"$\sqrt{\overline{VE}}$ at the TD($\lambda$) fixed point")
    axl.set_title(r"Exact TD($\lambda$) fixed points ($\lambda=1$: the MC solution)")
    axl.legend(fontsize=8)
    t = np.array(timing) * 1e6
    axes[2].loglog(ds, t[:, 0], "o-", label="semi-gradient TD(0)")
    axes[2].loglog(ds, t[:, 1], "s-", label="LSTD (Sherman-Morrison)")
    dd = np.array(ds, dtype=float)
    axes[2].loglog(dd, t[-1, 1] * (dd / dd[-1]) ** 2, "k:", lw=1, label=r"slope 2 ($O(d^2)$)")
    axes[2].set_xlabel("number of features d")
    axes[2].set_ylabel("microseconds per update")
    axes[2].set_title("Measured cost per update")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "lstd_vs_td.png"), dpi=110)
    print(f"figure saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
