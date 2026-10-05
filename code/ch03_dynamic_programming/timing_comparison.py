"""Wall-clock comparison of policy iteration, value iteration and modified policy iteration (Sections 7 and 12).

Part A (discount sweep): for FrozenLake 8x8 and a random MDP, and gamma in {0.9, 0.99, 0.999}, run
  * value iteration (= modified PI with m = 1),
  * modified PI with m in {2, 5, 10, 20, 50, 100} evaluation sweeps per improvement,
  * policy iteration with exact evaluation (a linear solve),
all to the SAME accuracy guarantee ||V - v_*|| < tol (VI / MPI: stop when ||T*V - V||/(1-gamma) < tol,
eq. 3.11; PI: stop when the policy is stable, which gives V = v_* up to round-off).
Part B (state-space scaling): sparse random MDPs with |S| from 100 to 3200 at gamma = 0.99, stored as
scipy.sparse CSR matrices so that a sweep costs O(|S||A|b) instead of O(|S|^2 |A|). PI iteration
counts, and wall time of PI (dense LU solve, O(|S|^3) per evaluation), VI, MPI(m=20) and, up to
|S| = --lp-max (default 1600 in full mode), the primal LP (HiGHS, sparse constraint matrix).

All timings use ONE thread (OMP/OPENBLAS/MKL_NUM_THREADS=1) and report the minimum of `reps` runs
(the machine is shared, so the minimum is the most reproducible statistic; even so, timings can differ by
a factor of 2-3 between runs; iteration counts are exact).
Run:  python code/ch03_dynamic_programming/timing_comparison.py [--quick]
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
import scipy.sparse as sp  # noqa: E402
from scipy.optimize import linprog  # noqa: E402

from dp import evaluate_exact, modified_policy_iteration, policy_iteration  # noqa: E402
from mdps import frozenlake_mdp, random_mdp  # noqa: E402

SEED = 0
TOL = 1e-6


def timed(fn, reps):
    times, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return float(np.min(times)), out           # min: the least contaminated by other processes


def run_all(mdp, ms, reps):
    """Returns dict name -> (seconds, improvements, eval sweeps, ||V - v*||, greedy policy optimal?)."""
    pi_t, (acts, v_star, hist) = timed(lambda: policy_iteration(mdp, np.zeros(mdp.S, dtype=int)), reps)
    res = {"PI": (pi_t, len(hist), 0, 0.0, True)}
    for m in ms:
        t, (V, a, its, nev) = timed(lambda: modified_policy_iteration(mdp, m, tol=TOL), reps)
        opt = np.max(v_star - evaluate_exact(mdp, a)) < 1e-9
        res["VI" if m == 1 else f"MPI m={m}"] = (t, its, nev, float(np.max(np.abs(V - v_star))), bool(opt))
    return res


# ---------------------------------------------------------------------------------------------
# Part B engine: the same algorithms as dp.py, for a sparse model Pf[(s*A + a), s'] (CSR).
# ---------------------------------------------------------------------------------------------
def sparse_random_mdp(S, A, gamma, b, seed):
    """Same distribution as mdps.random_mdp, built directly in sparse form."""
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(S * A), b)
    cols = np.concatenate([rng.choice(S, size=b, replace=False) for _ in range(S * A)])
    vals = np.concatenate([rng.dirichlet(np.ones(b)) for _ in range(S * A)])
    Pf = sp.csr_matrix((vals, (rows, cols)), shape=(S * A, S))
    return Pf, rng.random((S, A)), gamma


def sp_greedy(Q, old):
    new = Q.argmax(axis=1)
    if old is not None:
        keep = Q[np.arange(len(old)), old] >= Q.max(axis=1) - 1e-9 * np.maximum(1, np.abs(Q.max(axis=1)))
        new = np.where(keep, old, new)
    return new


def sp_mpi(Pf, R, g, m, tol=TOL):
    """Algorithm 3.4 (m = 1: value iteration). Returns V, actions, improvements, extra sweeps."""
    S, A = R.shape
    V, acts, n_eval = np.zeros(S), None, 0
    for it in range(1, 10 ** 7):
        Q = R + g * (Pf @ V).reshape(S, A)
        TV = Q.max(axis=1)
        if np.max(np.abs(TV - V)) / (1 - g) < tol:
            return V, sp_greedy(Q, acts), it, n_eval
        acts = sp_greedy(Q, acts)
        V = TV
        if m > 1:
            idx = np.arange(S) * A + acts
            P_pi, r_pi = Pf[idx], R[np.arange(S), acts]          # CSR row selection: O(nnz)
            for _ in range(m - 1):
                V = r_pi + g * (P_pi @ V)
            n_eval += m - 1


def sp_pi(Pf, R, g):
    """Algorithm 3.2 with a dense LU solve of (I - g P_pi) v = r_pi."""
    S, A = R.shape
    acts = np.zeros(S, dtype=int)
    for it in range(1, 10 ** 4):
        idx = np.arange(S) * A + acts
        V = np.linalg.solve(np.eye(S) - g * Pf[idx].toarray(), R[np.arange(S), acts])
        Q = R + g * (Pf @ V).reshape(S, A)
        new = sp_greedy(Q, acts)
        if np.array_equal(new, acts):
            return V, acts, it
        acts = new


def sp_lp(Pf, R, g):
    S, A = R.shape
    E = sp.csr_matrix((np.ones(S * A), (np.arange(S * A), np.repeat(np.arange(S), A))), shape=(S * A, S))
    t0 = time.perf_counter()
    res = linprog(np.full(S, 1.0 / S), A_ub=(g * Pf - E).tocsr(), b_ub=-R.ravel(),
                  bounds=[(None, None)] * S, method="highs")
    return time.perf_counter() - t0, res.x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--lp-max", type=int, default=None,
                        help="largest |S| for which Part B also times the LP (default: 200 quick, 1600 full)")
    args = parser.parse_args()
    reps = 1 if args.quick else 5
    gammas = [0.9, 0.99] if args.quick else [0.9, 0.99, 0.999]
    ms = [1, 5, 20] if args.quick else [1, 2, 5, 10, 20, 50, 100]
    sizes = [100, 200] if args.quick else [100, 200, 400, 800, 1600, 3200]
    lp_max = args.lp_max if args.lp_max is not None else (200 if args.quick else 1600)   # LP is slow at 3200
    S_rand = 100 if args.quick else 200
    print(f"timing_comparison.py  seed={SEED}  quick={args.quick}  tol={TOL}  reps={reps} (min)  "
          f"gammas={gammas}  m in {ms}  random MDP: S={S_rand}, A=5, branching=10  threads=1  lp_max={lp_max}")
    t_start = time.perf_counter()

    partA = {}
    for name in ("FrozenLake 8x8", f"random S={S_rand}"):
        for g in gammas:
            mdp = frozenlake_mdp("8x8", g)[0] if name.startswith("Frozen") else \
                random_mdp(S_rand, 5, g, branching=10, seed=SEED)
            res = run_all(mdp, ms, reps)
            partA[(name, g)] = res
            print(f"\n--- {name}, gamma = {g} ---")
            print("   algorithm      time [ms]   improvements   extra eval sweeps   ||V - v*||   greedy optimal")
            for k, (t, its, nev, err, opt) in res.items():
                print(f"   {k:12s} {t * 1e3:10.2f}   {its:12d}   {nev:17d}   {err:10.1e}   {opt}")

    print("\n--- Part B: scaling with |S| (sparse random MDPs, A = 5, branching = 10, gamma = 0.99) ---")
    print("   |S|   PI iters   PI [ms]   VI sweeps   VI [ms]   MPI(m=20) [ms]    LP [ms]   max|V - v*| (VI, MPI, LP)")
    partB = []
    for S in sizes:
        Pf, R, g = sparse_random_mdp(S, 5, 0.99, 10, seed=SEED + S)
        r_b = 1 if S >= 1600 else 3
        t_pi, (v_star, _, n_pi) = timed(lambda: sp_pi(Pf, R, g), r_b)
        t_vi, (V1, _, n_vi, _) = timed(lambda: sp_mpi(Pf, R, g, 1), r_b)
        t_m, (V2, _, _, _) = timed(lambda: sp_mpi(Pf, R, g, 20), r_b)
        t_lp, e_lp = np.nan, np.nan
        if S <= lp_max:
            t_lp, v_lp = sp_lp(Pf, R, g)
            e_lp = float(np.max(np.abs(v_lp - v_star)))
        e1, e2 = np.max(np.abs(V1 - v_star)), np.max(np.abs(V2 - v_star))
        partB.append((S, n_pi, t_pi, n_vi, t_vi, t_m, t_lp))
        print(f"  {S:4d}   {n_pi:8d}   {t_pi * 1e3:7.1f}   {n_vi:9d}   {t_vi * 1e3:7.1f}   {t_m * 1e3:14.1f}"
              f"   {t_lp * 1e3:8.1f}   {e1:.1e}, {e2:.1e}, {e_lp:.1e}")

    if not args.quick:
        make_figures(partA, partB, ms, gammas, S_rand)
    print(f"\nDone in {time.perf_counter() - t_start:.1f} s")


def make_figures(partA, partB, ms, gammas, S_rand):
    from plotting import C, plt, save

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, name in zip(axes[:2], ("FrozenLake 8x8", f"random S={S_rand}")):
        for i, g in enumerate(gammas):
            res = partA[(name, g)]
            t = [res["VI" if m == 1 else f"MPI m={m}"][0] * 1e3 for m in ms]
            ax.loglog(ms, t, "o-", color=C[i], label=f"$\\gamma={g}$: VI (m=1) and MPI")
            ax.axhline(res["PI"][0] * 1e3, color=C[i], ls="--", lw=1,
                       label=f"$\\gamma={g}$: PI ({res['PI'][1]} iterations)")
        ax.set_xlabel("evaluation sweeps per improvement, m")
        ax.set_ylabel("wall time [ms] (1 thread)")
        ax.set_title(f"{name}: time to $\\|V-v_\\ast\\|<10^{{-6}}$")
        ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    ax = axes[2]
    B = np.array(partB)
    ax.loglog(B[:, 0], B[:, 2] * 1e3, "o-", color=C[1], label="PI (exact solve)")
    ax.loglog(B[:, 0], B[:, 4] * 1e3, "o-", color=C[0], label="VI")
    ax.loglog(B[:, 0], B[:, 5] * 1e3, "o-", color=C[2], label="MPI m=20")
    ok = ~np.isnan(B[:, 6])
    ax.loglog(B[ok, 0], B[ok, 6] * 1e3, "o-", color=C[3], label="LP (HiGHS, primal)")
    for S, its, t in zip(B[:, 0], B[:, 1], B[:, 2]):
        ax.annotate(f"{int(its)} it", (S, t * 1e3), textcoords="offset points", xytext=(-6, 6), fontsize=7,
                    color=C[1])
    ax.set_xlabel("|S|  (|A| = 5, 10 successors per (s,a), sparse storage)")
    ax.set_ylabel("wall time [ms] (1 thread)")
    ax.set_title(r"Scaling with $|\mathcal{S}|$ ($\gamma=0.99$)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "timing_pi_vi_mpi.png")


if __name__ == "__main__":
    main()
