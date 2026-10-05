"""Chapter 03 worked example: DP by hand on the two-state maintenance MDP, plus checks of the theory.

Reproduces every number of the hand calculations in the chapter (Sections 2.1, 5.3 and 6.4):
  * two value-iteration sweeps from two different starts: the gap shrinks by at most gamma = 0.8,
  * value iteration V_0 .. V_6 from V_0 = 0, with the a-priori and a-posteriori error bounds,
  * policy iteration from "always fix" -> "always run" -> "run when Good, fix when Worn",
  * the policy improvement theorem on every pair of deterministic policies.
Then it checks the theorems of Sections 2 and 6 numerically on random MDPs:
  * T^pi and T* are gamma-contractions in the max norm (largest observed ratio <= gamma),
  * both are monotone,
  * the greedy-policy loss bound 2 gamma Delta / (1 - gamma) of eq. (3.21) and MacQueen's bounds (3.23).
Full mode also draws the iterates in the (v(G), v(W)) plane and the LP feasible region.

Run:  python code/ch03_dynamic_programming/maintenance_by_hand.py [--quick]
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse  # noqa: E402
import itertools  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from dp import (bellman_expectation, bellman_optimality, evaluate_exact, greedy,  # noqa: E402
                policy_iteration, q_from_v, value_iteration, modified_policy_iteration)
from mdps import maintenance_mdp, random_mdp  # noqa: E402

SEED = 0


def two_starts(mdp, v_star, n=3):
    print("\n[0] Contraction by hand: the optimality backup from two different starts   (Section 2.1)")
    print("  k   U_k = (G, W)        W_k = (G, W)        ||U_k - W_k||   ratio to previous   "
          "||U_k - v*||   ||W_k - v*||")
    U, W = np.zeros(2), np.array([12.0, 1.0])
    prev = None
    for k in range(n + 1):
        gap = np.max(np.abs(U - W))
        ratio = "" if prev is None else f"{gap / prev:.4f}"
        print(f"  {k}   ({U[0]:6.3f}, {U[1]:6.3f})   ({W[0]:6.3f}, {W[1]:6.3f})   {gap:10.4f}      {ratio:10s}"
              f"       {np.max(np.abs(U - v_star)):8.4f}      {np.max(np.abs(W - v_star)):8.4f}")
        prev = gap
        U, W = bellman_optimality(mdp, U), bellman_optimality(mdp, W)


def hand_value_iteration(mdp, v_star, n=6):
    print("\n[1] Value iteration by hand, V_0 = (0, 0), gamma = 0.8   (Section 6.4)")
    print("  k   V_k(G)    V_k(W)   greedy(G,W)   ||V_k - v*||   gamma^k*||V_0-v*||   "
          "a-posteriori gamma/(1-gamma)*||V_k - V_{k-1}||")
    g = mdp.gamma
    V = np.zeros(2)
    e0 = np.max(np.abs(V - v_star))
    prev = None
    for k in range(n + 1):
        err = np.max(np.abs(V - v_star))
        post = "" if prev is None else f"{g / (1 - g) * np.max(np.abs(V - prev)):.4f}"
        act = greedy(mdp, V)
        names = ",".join(mdp.action_names[a] for a in act)
        print(f"  {k}  {V[0]:7.4f}  {V[1]:7.4f}   {names:10s}   {err:8.4f}       {g ** k * e0:8.4f}"
              f"            {post}")
        prev, V = V, bellman_optimality(mdp, V)


def hand_policy_iteration(mdp):
    print("\n[2] Policy iteration by hand from pi_0 = (fix, fix)   (Section 5.3)")
    _, _, hist = policy_iteration(mdp, np.array([1, 1]))
    for i, h in enumerate(hist):
        acts = ",".join(mdp.action_names[a] for a in h["actions"])
        Q = q_from_v(mdp, h["V"])
        print(f"  pi_{i} = ({acts}):  v = ({h['V'][0]:.4f}, {h['V'][1]:.4f})   "
              f"q = [[G: run {Q[0, 0]:.3f}, fix {Q[0, 1]:.3f}], [W: run {Q[1, 0]:.3f}, fix {Q[1, 1]:.3f}]]")
    print(f"  stable after {len(hist)} evaluations; final policy = "
          f"({','.join(mdp.action_names[a] for a in hist[-1]['actions'])})")
    return hist


def improvement_theorem_check(mdp):
    print("\n[3] Policy improvement theorem on all ordered pairs (pi, pi') of deterministic policies")
    pols = [np.array(p) for p in itertools.product(range(2), repeat=2)]
    n_hyp, n_ok = 0, 0
    for p, p2 in itertools.product(pols, pols):
        v = evaluate_exact(mdp, p)
        lhs = bellman_expectation(mdp, v, p2)          # (T^{pi'} v_pi)(s) = q_pi(s, pi'(s))
        if np.all(lhs >= v - 1e-12):                   # hypothesis of Theorem 3.3
            n_hyp += 1
            n_ok += np.all(evaluate_exact(mdp, p2) >= v - 1e-12)
    print(f"  pairs satisfying the hypothesis q_pi(s, pi'(s)) >= v_pi(s) for all s: {n_hyp}/16; "
          f"conclusion v_pi' >= v_pi holds in {n_ok}/{n_hyp}")


def contraction_checks(rng, n_pairs, quick):
    print(f"\n[4] Contraction and monotonicity of the Bellman operators ({n_pairs} random pairs per MDP)")
    for gamma in (0.5, 0.9, 0.99):
        mdp = random_mdp(30, 4, gamma, branching=5, seed=int(rng.integers(1 << 30)))
        pi = rng.dirichlet(np.ones(4), size=30)             # a random stochastic policy
        r_pi = np.max(np.abs(mdp.R)) / (1 - gamma)
        worst = {"T^pi": 0.0, "T*": 0.0}
        mono_viol = 0
        for _ in range(n_pairs):
            u = rng.normal(0, r_pi, 30)
            w = u + rng.normal(0, rng.choice([1e-3, 1.0, 10.0]), 30)
            d = np.max(np.abs(u - w))
            worst["T^pi"] = max(worst["T^pi"], np.max(np.abs(bellman_expectation(mdp, u, pi)
                                                             - bellman_expectation(mdp, w, pi))) / d)
            worst["T*"] = max(worst["T*"], np.max(np.abs(bellman_optimality(mdp, u)
                                                         - bellman_optimality(mdp, w))) / d)
            hi = np.maximum(u, w)                            # hi >= u componentwise
            mono_viol += np.any(bellman_optimality(mdp, hi) < bellman_optimality(mdp, u) - 1e-12)
            mono_viol += np.any(bellman_expectation(mdp, hi, pi) < bellman_expectation(mdp, u, pi) - 1e-12)
        # The worst case is a constant shift: T(u + c 1) = T u + gamma c 1 when rows of P sum to 1.
        u = rng.normal(0, r_pi, 30)
        tight = np.max(np.abs(bellman_optimality(mdp, u + 3.0) - bellman_optimality(mdp, u))) / 3.0
        print(f"  gamma = {gamma:4.2f}: max ratio ||T u - T w|| / ||u - w|| over random pairs:  "
              f"T^pi {worst['T^pi']:.4f}, T* {worst['T*']:.4f};  for w = u + 3*1: {tight:.6f};  "
              f"monotonicity violations: {mono_viol}")


def bound_checks(rng, n_mdps):
    print(f"\n[5] Value-iteration stopping bounds on {n_mdps} random MDPs (S=40, A=4, gamma=0.9)")
    worst_ratio, worst_post, macq_viol, n_checks = 0.0, 0.0, 0, 0
    first_opt, total = [], []
    for i in range(n_mdps):
        mdp = random_mdp(40, 4, 0.9, branching=3, seed=int(rng.integers(1 << 30)))
        g = mdp.gamma
        v_star = policy_iteration(mdp, np.zeros(40, dtype=int))[1]
        V = rng.normal(0, 5, 40)                             # arbitrary start
        found = None
        for k in range(1, 400):
            V_new = bellman_optimality(mdp, V)
            delta = np.max(np.abs(V_new - V))
            if delta < 1e-12:
                break
            pi = greedy(mdp, V_new)
            loss = np.max(v_star - evaluate_exact(mdp, pi))
            worst_ratio = max(worst_ratio, loss / (2 * g * delta / (1 - g)))          # eq. (3.21)
            worst_post = max(worst_post, np.max(np.abs(V_new - v_star)) / (g * delta / (1 - g)))
            dmin, dmax = np.min(V_new - V), np.max(V_new - V)                          # eq. (3.23)
            lo, hi = V_new + g / (1 - g) * dmin, V_new + g / (1 - g) * dmax
            macq_viol += np.any(v_star < lo - 1e-9) or np.any(v_star > hi + 1e-9)
            n_checks += 1
            if found is None and loss < 1e-10:
                found = k
            V = V_new
        first_opt.append(found)
        total.append(k)
    print(f"  max over all iterations of  (true loss of greedy policy) / (2 gamma Delta/(1-gamma)) = "
          f"{worst_ratio:.4f}  (bound holds if <= 1)")
    print(f"  max of ||V_k+1 - v*|| / (gamma Delta/(1-gamma)) = {worst_post:.6f}  (<= 1)")
    print(f"  MacQueen bounds violated in {macq_viol} of {n_checks} checks")
    print(f"  greedy policy first optimal after a median of {int(np.median(first_opt))} sweeps; "
          f"Delta < 1e-12 needs a median of {int(np.median(total))} sweeps")


def make_figure(mdp, v_star, pi_hist):
    from plotting import C, plt, save

    g = mdp.gamma
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    # all four deterministic policies
    for p in itertools.product(range(2), repeat=2):
        v = evaluate_exact(mdp, np.array(p))
        ax.plot(*v, "ks", ms=5)
        ax.annotate(r"$\pi_{" + "".join("rf"[a] for a in p) + "}$", v, textcoords="offset points",
                    xytext=(5, -12), fontsize=9)
    starts = [np.zeros(2), np.array([12.0, 1.0]), np.array([1.0, 9.0]), np.array([-3.0, -4.0])]
    for i, V0 in enumerate(starts):
        _, _, _, hist = value_iteration(mdp, theta=1e-3, V0=V0, record=True)
        H = np.array(hist)
        ax.plot(H[:, 0], H[:, 1], ".-", color=C[0], ms=3, lw=0.8, alpha=0.8,
                label="value iteration (4 starts)" if i == 0 else None)
        ax.plot(*V0, "o", color=C[0], mfc="white", ms=6)
    P = np.array([h["V"] for h in pi_hist])
    ax.plot(P[:, 0], P[:, 1], "-o", color=C[1], lw=2, ms=6, label="policy iteration from $\\pi_{ff}$")
    ax.plot(*v_star, "*", color=C[4], ms=16, mec="k", label=r"$v_\ast=(7.5,\,5)$", zorder=5)
    ax.set_xlabel("V(G)"); ax.set_ylabel("V(W)")
    ax.set_title(r"DP iterates in the value plane ($\gamma=0.8$)")
    ax.legend(loc="lower right")

    # Right: the LP feasible region {v : v >= T* v}
    ax = axes[1]
    xs = np.linspace(-2, 14, 400)
    X, Y = np.meshgrid(xs, np.linspace(-4, 12, 400))
    feas = np.ones_like(X, dtype=bool)
    lines = []
    for s in range(2):
        for a in range(2):
            # constraint v(s) >= R[s,a] + g * (P[s,a,0] v(G) + P[s,a,1] v(W))
            lhs = (X if s == 0 else Y) - g * (mdp.P[s, a, 0] * X + mdp.P[s, a, 1] * Y)
            feas &= lhs >= mdp.R[s, a] - 1e-12
            lines.append((s, a, lhs - mdp.R[s, a]))
    ax.contourf(X, Y, feas.astype(float), levels=[0.5, 1.5], colors=[C[5]], alpha=0.35)
    for k, (s, a, F) in enumerate(lines):
        ax.contour(X, Y, F, levels=[0.0], colors=[C[k % 4 + 0]], linewidths=1.3)
        ax.plot([], [], color=C[k % 4], label=f"constraint ({mdp.state_names[s]}, {mdp.action_names[a]})")
    ax.plot(*v_star, "*", color=C[4], ms=16, mec="k", zorder=5)
    ax.annotate(r"$v_\ast$: smallest feasible point", v_star, xytext=(8, 0.5),
                arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.set_xlim(-2, 14); ax.set_ylim(-4, 12)
    ax.set_xlabel("v(G)"); ax.set_ylabel("v(W)")
    ax.set_title(r"LP feasible set $\{v: v \geq \mathcal{T}^\ast v\}$ (shaded)")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "maintenance_value_plane.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    n_pairs = 300 if args.quick else 5000
    n_mdps = 5 if args.quick else 50
    print(f"maintenance_by_hand.py  seed={SEED}  quick={args.quick}  gamma=0.8 for the hand example; "
          f"{n_pairs} operator pairs; {n_mdps} random MDPs for the bound check")
    t0 = time.perf_counter()

    mdp = maintenance_mdp(0.8)
    v_star = value_iteration(mdp, theta=1e-14)[0]
    print(f"v* = ({v_star[0]:.6f}, {v_star[1]:.6f})")
    two_starts(mdp, v_star)
    hand_value_iteration(mdp, v_star)
    pi_hist = hand_policy_iteration(mdp)
    improvement_theorem_check(mdp)
    for m in (1, 2, 5):
        V, acts, its, nev = modified_policy_iteration(mdp, m, tol=1e-6)
        print(f"  modified PI m={m}: {its} improvement steps, {nev} extra evaluation sweeps, "
              f"policy=({','.join(mdp.action_names[a] for a in acts)})")
    contraction_checks(rng, n_pairs, args.quick)
    bound_checks(rng, n_mdps)
    if not args.quick:
        make_figure(mdp, v_star, pi_hist)
    print(f"\nDone in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
