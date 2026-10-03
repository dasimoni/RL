"""Iterative policy evaluation on Sutton & Barto's 4x4 gridworld (Example 4.1, Figure 4.1).

* Two-array (synchronous) evaluation of the equiprobable random policy, gamma = 1:
  prints v_k for k = 0, 1, 2, 3, 10 and the fixed point, and the greedy policy w.r.t. each v_k.
* Checks that the greedy policy w.r.t. v_k is already optimal from k = 3 on (Section 4).
* Compares two-array vs in-place (Gauss-Seidel) evaluation: sweeps needed for several
  thresholds theta, and the TRUE error ||V - v_pi||_inf when the stopping rule fires; explains the
  measured rates by spectral radii (two-array: rho(P_pi); in-place: the Gauss-Seidel matrix).

Run:  python code/ch03_dynamic_programming/gridworld_policy_evaluation.py [--quick]
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

from dp import (evaluate_exact, evaluate_iterative, greedy, greedy_actions_all,  # noqa: E402
                policy_model, uniform_policy, value_iteration)
from mdps import GRID_ARROWS, gridworld_4x4  # noqa: E402

SEED = 0  # everything here is deterministic; printed for uniformity with the other scripts
SHOW_K = [0, 1, 2, 3, 10]


def spectral_radius(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def gauss_seidel_matrix(P, order):
    """Error-propagation matrix of one in-place sweep in `order` (gamma = 1): (I - L)^{-1} U."""
    Po = P[np.ix_(order, order)]
    L = np.tril(Po, -1)
    return np.linalg.solve(np.eye(len(order)) - L, Po - L)


def fmt_grid(V):
    return "\n".join("    " + " ".join(f"{x:6.1f}" for x in row) for row in V.reshape(4, 4))


def fmt_policy(mdp, V):
    acts = greedy_actions_all(mdp, V, tol=1e-9)
    rows = []
    for r in range(4):
        cells = []
        for c in range(4):
            s = 4 * r + c
            cells.append("  T  " if mdp.terminal[s] else "".join(GRID_ARROWS[a] for a in acts[s]).ljust(5))
        rows.append("    " + " ".join(cells))
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(f"gridworld_policy_evaluation.py  seed={SEED}  quick={args.quick}  "
          "gamma=1, reward -1 per step, equiprobable random policy")
    t0 = time.perf_counter()
    mdp = gridworld_4x4()
    pi = uniform_policy(mdp)
    v_pi = evaluate_exact(mdp, pi)                      # the k = infinity column, solved directly

    # ---- Figure 4.1: two-array sweeps --------------------------------------------------
    V_inf, n_sweeps, hist = evaluate_iterative(mdp, pi, theta=1e-12, record=True)
    print(f"\n[1] Two-array iterative policy evaluation (Algorithm 3.1); {n_sweeps} sweeps to Delta < 1e-12; "
          f"max |V - v_pi(exact solve)| = {np.max(np.abs(V_inf - v_pi)):.1e}")
    for k in SHOW_K:
        print(f"  k = {k}:\n{fmt_grid(hist[k])}\n  greedy policy w.r.t. v_{k}:\n{fmt_policy(mdp, hist[k])}")
    print(f"  k = inf:\n{fmt_grid(v_pi)}\n  greedy policy w.r.t. v_pi:\n{fmt_policy(mdp, v_pi)}")

    # ---- Is the greedy policy optimal? -------------------------------------------------
    v_star = value_iteration(mdp, theta=1e-12)[0]
    print("\n[2] Value of the greedy policy w.r.t. v_k (ties broken towards the first action)")
    for k in range(0, 6):
        g = greedy(mdp, hist[k])
        try:
            vg = evaluate_exact(mdp, g)
            gap = np.max(v_star - vg)
            msg = f"max_s (v_*(s) - v_greedy(s)) = {gap:.2e}"
        except np.linalg.LinAlgError:
            msg = "greedy policy is IMPROPER (never terminates from some state): value -infinity"
        print(f"  k = {k}: {msg}")
    print(f"  v_* = \n{fmt_grid(v_star)}")

    # ---- In-place vs two-array ---------------------------------------------------------
    print("\n[3] Two-array vs in-place evaluation of the random policy")
    perm = np.random.default_rng(SEED).permutation(16)
    print(f"  (random fixed order used in the last column: {perm.tolist()})")
    print("   theta      two-array: sweeps  true error  |  in-place (0..15): sweeps  true error"
          "  |  in-place (random order): sweeps  true error")
    thetas = [1e-2, 1e-4] if args.quick else [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]
    results = {}
    for th in thetas:
        row = []
        for inplace, order in ((False, None), (True, None), (True, perm)):
            V, k, _ = evaluate_iterative(mdp, pi, theta=th, inplace=inplace, order=order)
            row.append((k, np.max(np.abs(V - v_pi))))
        results[th] = row
        print(f"  {th:7.0e}       {row[0][0]:5d}       {row[0][1]:9.2e}  |"
              f"       {row[1][0]:5d}       {row[1][1]:9.2e}  |       {row[2][0]:5d}        {row[2][1]:9.2e}")
    print("  (gamma = 1, so the a-posteriori bound gamma*Delta/(1-gamma) of eq. (3.16) does not apply:"
          " the true error at stopping is much larger than theta)")

    # Why these rates? Two-array: V_{k+1} - v_pi = P_pi (V_k - v_pi), so the error shrinks asymptotically
    # by rho(P_pi) per sweep. In-place in a fixed order: split P_pi (rows/columns in sweep order) into its
    # strictly lower part L (already updated) and the rest U; then e_{k+1} = (I - L)^{-1} U e_k.
    _, P_pi = policy_model(mdp, pi)
    rho_j = spectral_radius(P_pi)
    rho_gs = [spectral_radius(gauss_seidel_matrix(P_pi, o)) for o in (np.arange(16), perm)]
    print(f"  asymptotic error ratio per sweep: two-array rho(P_pi) = {rho_j:.4f}; in-place rho((I-L)^-1 U) = "
          f"{rho_gs[0]:.4f} (order 0..15), {rho_gs[1]:.4f} (random order)")
    print(f"  error/theta at stopping predicted by rho/(1-rho): two-array {rho_j / (1 - rho_j):.1f}, "
          f"in-place {rho_gs[0] / (1 - rho_gs[0]):.1f}")

    if not args.quick:
        make_figures(mdp, hist, v_pi, pi)
    print(f"\nDone in {time.perf_counter() - t0:.1f} s")


def make_figures(mdp, hist, v_pi, pi):
    from plotting import C, grid_values, plt, save

    cols = SHOW_K + ["inf"]
    fig, axes = plt.subplots(2, len(cols), figsize=(2.15 * len(cols), 4.9))
    for j, k in enumerate(cols):
        V = v_pi if k == "inf" else hist[k]
        grid_values(axes[0, j], V, (4, 4), title=f"$v_{{{k if k != 'inf' else chr(92) + 'infty'}}}$",
                    fmt="{:.1f}", cmap="viridis", vmin=-22, vmax=0, fontsize=8)
        acts = greedy_actions_all(mdp, V, tol=1e-9)
        arrows = [None if mdp.terminal[s] else acts[s] for s in range(16)]
        ax = axes[1, j]
        ax.imshow(np.zeros((4, 4)), cmap="Greys", vmin=0, vmax=1)
        for s in range(16):
            r, c = divmod(s, 4)
            if mdp.terminal[s]:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="0.6"))
                continue
            for a in arrows[s]:
                dr, dc = [(-1, 0), (1, 0), (0, 1), (0, -1)][a]
                ax.arrow(c, r, 0.28 * dc, 0.28 * dr, head_width=0.12, head_length=0.1, color=C[0])
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True); ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        ax.grid(which="minor", color="0.7"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"greedy w.r.t. $v_{{{k if k != 'inf' else chr(92) + 'infty'}}}$", fontsize=9)
    fig.suptitle("Iterative policy evaluation of the random policy (two-array sweeps), "
                 "4x4 gridworld, $\\gamma=1$", y=1.0)
    save(fig, "gridworld_fig4_1.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    perm = np.random.default_rng(SEED).permutation(16)
    labels = ["two-array (synchronous)", "in-place, order 0..15", "in-place, random fixed order"]
    for i, (inplace, order) in enumerate(((False, None), (True, None), (True, perm))):
        _, k, h = evaluate_iterative(mdp, pi, theta=1e-10, inplace=inplace, order=order, record=True)
        err = [np.max(np.abs(V - v_pi)) for V in h]
        ax.semilogy(err, color=C[i], label=f"{labels[i]} ({k} sweeps)")
    ax.set_xlabel("sweep k"); ax.set_ylabel(r"$\|V_k - v_\pi\|_\infty$")
    ax.set_title("Two-array vs in-place policy evaluation (stop at $\\Delta<10^{-10}$)")
    ax.legend()
    save(fig, "gridworld_inplace_vs_twoarray.png")


if __name__ == "__main__":
    main()
