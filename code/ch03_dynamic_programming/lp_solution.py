"""Solving MDPs by linear programming (Section 10): primal (values) and dual (occupancy measures).

Primal (eq. 3.25):  minimise  sum_s mu(s) v(s)
                    s.t.      v(s) >= r(s,a) + gamma * sum_s' p(s'|s,a) v(s')   for all (s, a)
Dual   (eq. 3.26):  maximise  sum_{s,a} x(s,a) r(s,a)
                    s.t.      sum_a x(s',a) - gamma * sum_{s,a} p(s'|s,a) x(s,a) = mu(s')   for all s'
                              x >= 0
Both are solved with scipy.optimize.linprog (HiGHS) and compared with value/policy iteration on the
maintenance MDP (by hand in the chapter), FrozenLake 8x8 (gamma=0.99), the gambler's problem
(gamma = 1) and a random MDP. Checks: strong duality, x is the discounted occupancy measure of
the optimal policy, one positive x(s, .) per state (a deterministic policy), complementary
slackness, the primal's dual multipliers equal x, and what goes wrong if mu is not strictly positive.

Run:  python code/ch03_dynamic_programming/lp_solution.py [--quick]
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
from scipy.optimize import linprog  # noqa: E402

from dp import evaluate_exact, policy_iteration, policy_model, value_iteration  # noqa: E402
from mdps import frozenlake_mdp, gamblers_problem, maintenance_mdp, random_mdp  # noqa: E402

SEED = 0


def pairs(mdp):
    return [(s, a) for s in range(mdp.S) for a in range(mdp.A) if mdp.valid[s, a]]


def solve_primal(mdp, mu):
    """Variables v(s), free. One inequality row per valid (s, a): gamma P[s,a] v - v(s) <= -r(s,a)."""
    sa = pairs(mdp)
    A_ub = np.array([mdp.gamma * mdp.P[s, a] - np.eye(mdp.S)[s] for s, a in sa])
    b_ub = np.array([-mdp.R[s, a] for s, a in sa])
    t0 = time.perf_counter()
    res = linprog(c=mu, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * mdp.S, method="highs")
    dt = time.perf_counter() - t0
    assert res.status == 0, res.message
    return res, sa, dt


def solve_dual(mdp, mu):
    """Variables x(s,a) >= 0 for valid pairs. Equality per s': sum_a x(s',a) - gamma sum p(s'|s,a) x(s,a) = mu(s')."""
    sa = pairs(mdp)
    A_eq = np.zeros((mdp.S, len(sa)))
    for j, (s, a) in enumerate(sa):
        A_eq[s, j] += 1.0
        A_eq[:, j] -= mdp.gamma * mdp.P[s, a]
    c = -np.array([mdp.R[s, a] for s, a in sa])        # linprog minimises
    t0 = time.perf_counter()
    res = linprog(c=c, A_eq=A_eq, b_eq=mu, bounds=[(0, None)] * len(sa), method="highs")
    dt = time.perf_counter() - t0
    assert res.status == 0, res.message
    X = np.zeros((mdp.S, mdp.A))
    for j, (s, a) in enumerate(sa):
        X[s, a] = res.x[j]
    return res, X, dt


def analyse(name, mdp, v_ref, t_ref, mu=None):
    S = mdp.S
    mu = np.full(S, 1.0 / S) if mu is None else mu
    pres, sa, t_p = solve_primal(mdp, mu)
    dres, X, t_d = solve_dual(mdp, mu)
    v_lp = pres.x
    primal_obj, dual_obj = pres.fun, -dres.fun
    xs = X.sum(axis=1)
    n_pos = (X > 1e-9 * max(1.0, X.max())).sum(axis=1)
    pol = X.argmax(axis=1)                               # pi(s) = the action with x(s, a) > 0
    v_pol = evaluate_exact(mdp, pol)
    # occupancy of that policy, computed independently: x^T = mu^T (I - gamma P_pi)^{-1}
    _, P_pi = policy_model(mdp, pol)
    occ = np.linalg.solve((np.eye(S) - mdp.gamma * P_pi).T, mu)
    # complementary slackness: x(s,a) > 0  =>  constraint (s,a) is tight
    slack = np.array([v_lp[s] - mdp.R[s, a] - mdp.gamma * mdp.P[s, a] @ v_lp for s, a in sa])
    xv = np.array([X[s, a] for s, a in sa])
    cs = np.max(np.abs(xv * slack))
    marg = -pres.ineqlin.marginals                       # HiGHS multipliers of the <= rows
    print(f"\n=== {name}: |S| = {S}, valid (s,a) pairs = {len(sa)}, gamma = {mdp.gamma} ===")
    print(f"  primal LP: {t_p * 1e3:7.1f} ms   max|v_LP - v_*(PI/VI)| = {np.max(np.abs(v_lp - v_ref)):.2e}"
          f"   (reference solver: {t_ref * 1e3:.1f} ms)")
    print(f"  dual LP:   {t_d * 1e3:7.1f} ms   primal objective = {primal_obj:.10f}, dual objective = "
          f"{dual_obj:.10f} (difference {abs(primal_obj - dual_obj):.1e})")
    print(f"  states with exactly one positive x(s,.): {int(np.sum(n_pos == 1))} of {S}   "
          f"(min_s sum_a x(s,a) = {xs.min():.4f} >= min mu = {mu.min():.4f})")
    print(f"  policy read from x is optimal: max|v_pi - v_*| = {np.max(np.abs(v_pol - v_ref)):.2e};  "
          f"x equals its occupancy mu^T (I - gamma P_pi)^-1: max diff = {np.max(np.abs(xs - occ)):.2e}")
    if mdp.gamma < 1:
        print(f"  sum_(s,a) x(s,a) = {X.sum():.6f}   (= 1/(1-gamma) = {1 / (1 - mdp.gamma):.6f} when no episode "
              f"terminates; smaller if they do)")
    print(f"  complementary slackness max |x * slack| = {cs:.1e};  primal multipliers vs x: "
          f"max diff = {np.max(np.abs(marg - xv)):.1e}")
    if np.max(np.abs(marg - xv)) > 1e-8:
        # With ties (several optimal policies) the dual optimum is not unique. Check that the primal's
        # multipliers are ANOTHER optimal dual solution: feasible, nonnegative, same objective.
        A_eq = np.zeros((S, len(sa)))
        for j, (s, a) in enumerate(sa):
            A_eq[s, j] += 1.0
            A_eq[:, j] -= mdp.gamma * mdp.P[s, a]
        feas = np.max(np.abs(A_eq @ marg - mu))
        obj = sum(marg[j] * mdp.R[s, a] for j, (s, a) in enumerate(sa))
        print(f"    -> not the same vertex: the multipliers are another optimal dual solution "
              f"(constraint violation {feas:.1e}, min entry {marg.min():.1e}, objective {obj:.10f}); "
              f"the dual optimum is not unique because optimal actions tie")
    return v_lp, X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(f"lp_solution.py  seed={SEED}  quick={args.quick}  solver=scipy.optimize.linprog(method='highs'); "
          f"mu uniform over all states unless stated")
    t0 = time.perf_counter()

    # 1. maintenance MDP: the hand example of Section 10.3
    m = maintenance_mdp(0.8)
    v_ref = np.array([7.5, 5.0])
    _, X = analyse("maintenance (gamma = 0.8), mu = (0.5, 0.5)", m, v_ref, 0.0, mu=np.array([0.5, 0.5]))
    print(f"  x = {np.round(X, 6).tolist()}  (rows G, W; columns run, fix)")

    # 2. FrozenLake 8x8
    fl, desc = frozenlake_mdp("8x8", 0.99)
    t = time.perf_counter(); _, v_fl, _ = policy_iteration(fl, np.zeros(fl.S, dtype=int)); t_pi = time.perf_counter() - t
    analyse("FrozenLake 8x8", fl, v_fl, t_pi)
    t = time.perf_counter(); value_iteration(fl, theta=1e-10); t_vi = time.perf_counter() - t
    print(f"  (for comparison, value iteration to Delta < 1e-10: {t_vi * 1e3:.1f} ms)")

    # mu concentrated on the start state: what the LP does NOT determine
    mu0 = np.zeros(fl.S); mu0[0] = 1.0
    pres, _, _ = solve_primal(fl, mu0)
    dres, X0, _ = solve_dual(fl, mu0)
    U = (X0.sum(axis=1) <= 1e-12) & ~fl.terminal           # never visited from the start under pi_*
    print(f"  with mu = delta(start): v_LP(start) - v_*(start) = {pres.x[0] - v_fl[0]:.1e}; x(s,.) = 0 in "
          f"{int(U.sum())} non-terminal states {np.flatnonzero(U).tolist()}, so the dual says nothing about "
          f"the action there")
    # ... and the primal does not pin down v there: v_* + c 1_U is feasible for small c > 0 with the same
    # objective (mu is zero on U). Largest such c from the slack of the constraints that see U:
    sa = pairs(fl)
    slack = np.array([v_fl[s] - fl.R[s, a] - fl.gamma * fl.P[s, a] @ v_fl for s, a in sa])
    into_U = np.array([fl.gamma * fl.P[s, a, U].sum() for s, a in sa])
    from_U = np.array([U[s] for s, a in sa])
    ok = (~from_U) & (into_U > 0)
    c = np.min(slack[ok] / into_U[ok]) if ok.any() else np.inf
    v_alt = v_fl + 0.5 * c * U
    viol = max(fl.R[s, a] + fl.gamma * fl.P[s, a] @ v_alt - v_alt[s] for s, a in sa)
    print(f"  v_* + c 1_U with c = {0.5 * c:.4f} is also an optimal primal solution: max constraint violation "
          f"{max(viol, 0):.1e}, objective {v_alt[0]:.10f} = v_*(start) {v_fl[0]:.10f}")

    # 3. gambler's problem (gamma = 1; every policy with stakes >= 1 terminates)
    gm = gamblers_problem(0.4)
    t = time.perf_counter(); v_g = value_iteration(gm, theta=1e-14)[0]; t_g = time.perf_counter() - t
    analyse("gambler's problem ph = 0.4 (gamma = 1)", gm, v_g, t_g)

    # 4. a random MDP
    if not args.quick:
        rm = random_mdp(200, 5, 0.95, branching=10, seed=SEED)
        t = time.perf_counter(); _, v_r, _ = policy_iteration(rm, np.zeros(rm.S, dtype=int)); t_r = time.perf_counter() - t
        analyse("random MDP (S=200, A=5, b=10)", rm, v_r, t_r)
        make_figure(fl, desc, v_fl, X0)
    print(f"\nDone in {time.perf_counter() - t0:.1f} s")


def make_figure(fl, desc, v_fl, X0):
    from plotting import grid_values, plt, save

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    grid_values(axes[0], v_fl, fl.shape, desc=desc, fmt="{:.2f}", fontsize=7)
    axes[0].set_title(r"Primal LP solution $v_\ast$ (FrozenLake 8x8, $\gamma=0.99$)")
    occ = X0.sum(axis=1)
    im = grid_values(axes[1], occ, fl.shape, desc=desc, fmt="{:.2f}", fontsize=7, cmap="magma")
    axes[1].set_title(r"Dual LP solution: $\sum_a x(s,a)$ with $\mu=\delta_{\mathrm{start}}$" "\n"
                      "expected discounted visits under the optimal policy", fontsize=10)
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    save(fig, "lp_frozenlake.png")


if __name__ == "__main__":
    main()
