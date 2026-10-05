"""Asynchronous value iteration (Section 8): does the ORDER of state backups matter?

On FrozenLake 8x8 (slippery and deterministic, gamma = 0.99) we compare, per single-state backup
(V(s) <- max_a [r(s,a) + gamma sum_s' p(s'|s,a) V(s')]):
  * synchronous (Jacobi) value iteration: every backup in a sweep reads the OLD array,
  * Gauss-Seidel sweeps in the natural order 0..63 (start -> goal),
  * Gauss-Seidel sweeps in reverse order 63..0 (goal -> start),
  * random asynchronous backups (each picks a uniformly random state),
  * "largest Bellman error first" (the idea behind prioritized sweeping, Chapter 07).
All converge to v_*, as the asynchronous convergence theorem says they must; the number of
backups they need differs a lot.

Run:  python code/ch03_dynamic_programming/async_dp.py [--quick]
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from dp import bellman_optimality, policy_iteration  # noqa: E402
from mdps import frozenlake_mdp  # noqa: E402

SEED = 0
GAMMA = 0.99
TARGET = 1e-6


def backup(mdp, V, s):
    q = mdp.R[s] + mdp.gamma * mdp.P[s] @ V
    return np.max(np.where(mdp.valid[s], q, -np.inf))


def run(mdp, v_star, method, budget, rng):
    """Returns (backups/|S|, error ||V - v*||) recorded after every |S| backups, and #backups to reach TARGET."""
    S = mdp.S
    V = np.zeros(S)
    errs = [np.max(np.abs(V - v_star))]
    ns = [0]
    hit = None
    n = 0
    sweep_orders = {"Gauss-Seidel 0..S-1": np.arange(S), "Gauss-Seidel S-1..0": np.arange(S)[::-1]}
    while n < budget:
        if method == "synchronous (Jacobi)":
            V = bellman_optimality(mdp, V)                 # S backups that all read the old V
            n += S
            errs.append(np.max(np.abs(V - v_star)))
            ns.append(n)
        else:
            for _ in range(S):
                if method in sweep_orders:
                    s = sweep_orders[method][n % S]
                elif method == "random state":
                    s = rng.integers(S)
                else:                                      # largest Bellman error first
                    s = int(np.argmax(np.abs(bellman_optimality(mdp, V) - V)))
                V[s] = backup(mdp, V, s)                   # in place: later backups see it at once
                n += 1
                if hit is None and np.max(np.abs(V - v_star)) < TARGET:
                    hit = n                                # exact count (diagnostic, not part of the algorithm)
                    break
            errs.append(np.max(np.abs(V - v_star)))
            ns.append(n)
        if hit is None and errs[-1] < TARGET:
            hit = n
        if hit is not None:
            break
    return np.array(ns) / S, np.array(errs), hit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    budget_sweeps = 60 if args.quick else 1500
    print(f"async_dp.py  seed={SEED}  quick={args.quick}  gamma={GAMMA}  target ||V - v*|| < {TARGET}  "
          f"budget = {budget_sweeps} x |S| backups")
    t0 = time.perf_counter()
    methods = ["synchronous (Jacobi)", "Gauss-Seidel 0..S-1", "Gauss-Seidel S-1..0", "random state",
               "largest Bellman error"]
    out = {}
    for slippery in (True, False):
        mdp, _ = frozenlake_mdp("8x8", GAMMA, is_slippery=slippery)
        v_star = policy_iteration(mdp, np.zeros(mdp.S, dtype=int))[1]
        label = "slippery" if slippery else "deterministic"
        print(f"\nFrozenLake 8x8, {label}: v*(start) = {v_star[0]:.6f}")
        print("   method                    backups to reach the target   (= sweeps of |S| = 64)")
        for mth in methods:
            rng = np.random.default_rng(SEED)
            x, errs, hit = run(mdp, v_star, mth, budget_sweeps * mdp.S, rng)
            out[(label, mth)] = (x, errs)
            msg = f"{hit:8d}   ({hit / mdp.S:7.1f})" if hit else f"not reached within budget (error {errs[-1]:.1e})"
            print(f"   {mth:25s} {msg}")
    if not args.quick:
        make_figure(out, methods)
    print(f"\nDone in {time.perf_counter() - t0:.1f} s")


def make_figure(out, methods):
    from plotting import C, plt, save

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, label in zip(axes, ("slippery", "deterministic")):
        for i, m in enumerate(methods):
            x, e = out[(label, m)]
            ax.semilogy(x, np.maximum(e, 1e-16), color=C[i], lw=1.6, label=m)
        ax.axhline(TARGET, color="0.5", lw=0.7, ls=":")
        ax.set_xlabel("single-state backups / |S|   (= sweeps)")
        ax.set_ylabel(r"$\|V - v_\ast\|_\infty$")
        ax.set_title(f"Asynchronous VI, FrozenLake 8x8 {label}, $\\gamma={GAMMA}$")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "async_dp.png")


if __name__ == "__main__":
    main()
