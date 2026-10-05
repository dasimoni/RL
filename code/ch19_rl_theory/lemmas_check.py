"""Numerical checks of the workhorse lemmas of RL theory. Chapter 19, Sections 2 and 4.

Every quantity is computed exactly (linear solves / backward induction), so each identity should
hold to floating-point round-off and each inequality should never be violated.

  (1) Performance difference lemma, discounted (Eq. 19.15):
        J(pi') - J(pi) = 1/(1-gamma) E_{s ~ d^{pi'}} E_{a ~ pi'} [A_pi(s, a)]
  (2) Performance difference lemma, finite horizon (Eq. 19.14):
        V^{pi'}_0(s0) - V^{pi}_0(s0) = sum_h E_{pi'}[A^pi_h(S_h, A_h)]
  (3) Simulation lemma, finite horizon, both forms (Eqs. 19.11-19.12):
        Vhat^pi_0 - V^pi_0 = sum_h E_{Phat, pi}[(rhat_h - r_h) + (Phat_h - P_h) V^pi_{h+1}]
                           = sum_h E_{P,    pi}[(rhat_h - r_h) + (Phat_h - P_h) Vhat^pi_{h+1}]
      and the bound (19.13) |Vhat^pi_0 - V^pi_0| <= H eps_r + sum_h (H-h-1) eps_P / 2.
  (4) Greedy-policy loss (Theorem 19.1, Singh & Yee 1994, Eq. 19.3): ||Q - q*||_inf <= eps implies
        v*(s) - v^{pi_Q}(s) <= 2 eps / (1 - gamma), and the two-action example that makes it tight.
  (5) Approximate value iteration (Eq. 19.4): if ||Q_{k+1} - T* Q_k||_inf <= eps for all k then
        limsup ||Q_k - q*||_inf <= eps / (1 - gamma); a constant +eps error attains it.
  (6) Discounted simulation lemma bound (Chapter 13, Eq. 13.4), as a cross-check.

Run from the repository root:
  python code/ch19_rl_theory/lemmas_check.py           # full run (a few seconds; no figure needed)
  python code/ch19_rl_theory/lemmas_check.py --quick   # fewer random instances
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from theory_lib import (fh_eval, fh_state_dists, greedy, occupancy, policy_eval, random_mdp,
                        random_policy, value_iteration)


def random_fh_mdp(rng, H, S, A):
    P = rng.dirichlet(np.ones(S), size=(H, S, A))
    r = rng.uniform(0, 1, size=(H, S, A))
    return P, r


def random_fh_policy(rng, H, S, A):
    return rng.dirichlet(np.ones(A), size=(H, S))


def check_pdl_discounted(rng, n_pairs, S=6, A=3, gamma=0.9):
    P, r = random_mdp(rng, S, A)
    d0 = rng.dirichlet(np.ones(S))
    worst = 0.0
    for _ in range(n_pairs):
        pi, pi2 = random_policy(rng, S, A, 0.5), random_policy(rng, S, A, 0.5)
        v, q = policy_eval(P, r, gamma, pi)
        v2, _ = policy_eval(P, r, gamma, pi2)
        adv = q - v[:, None]                                       # A_pi(s, a)
        d2 = occupancy(P, gamma, pi2, d0)                          # d^{pi'}_{d0}
        rhs = (d2 @ np.einsum("sa,sa->s", pi2, adv)) / (1 - gamma)
        worst = max(worst, abs((d0 @ v2 - d0 @ v) - rhs))
    print(f"(1) PDL, discounted (S={S}, A={A}, gamma={gamma}, {n_pairs} random policy pairs): "
          f"max |lhs - rhs| = {worst:.1e}")


def check_pdl_finite_horizon(rng, n_pairs, H=8, S=5, A=3):
    P, r = random_fh_mdp(rng, H, S, A)
    d0 = rng.dirichlet(np.ones(S))
    worst = 0.0
    for _ in range(n_pairs):
        pi, pi2 = random_fh_policy(rng, H, S, A), random_fh_policy(rng, H, S, A)
        Q, V = fh_eval(P, r, pi)
        _, V2 = fh_eval(P, r, pi2)
        adv = Q - V[:H, :, None]                                   # A^pi_h(s, a)
        d = fh_state_dists(P, pi2, d0)                             # Pr_{pi'}(S_h = s)
        rhs = sum(d[h] @ np.einsum("sa,sa->s", pi2[h], adv[h]) for h in range(H))
        worst = max(worst, abs((d0 @ V2[0] - d0 @ V[0]) - rhs))
    print(f"(2) PDL, finite horizon (H={H}, S={S}, A={A}, {n_pairs} pairs): max |lhs - rhs| = {worst:.1e}")


def check_simulation_lemma_fh(rng, n_models, H=8, S=5, A=3):
    P, r = random_fh_mdp(rng, H, S, A)
    d0 = rng.dirichlet(np.ones(S))
    worst1 = worst2 = 0.0
    worst_ratio = 0.0
    for _ in range(n_models):
        lam = rng.uniform(0.01, 0.5)
        P_hat = (1 - lam) * P + lam * rng.dirichlet(np.ones(S), size=(H, S, A))
        r_hat = np.clip(r + rng.normal(0, 0.05, size=r.shape), 0, 1)
        pi = random_fh_policy(rng, H, S, A)
        _, V = fh_eval(P, r, pi)
        _, V_hat = fh_eval(P_hat, r_hat, pi)
        lhs = d0 @ (V_hat[0] - V[0])
        d_model = fh_state_dists(P_hat, pi, d0)                   # trajectories in the MODEL
        d_true = fh_state_dists(P, pi, d0)                        # trajectories in the TRUE MDP
        form1 = form2 = 0.0
        for h in range(H):
            err1 = (r_hat[h] - r[h]) + (P_hat[h] - P[h]) @ V[h + 1]          # true values
            err2 = (r_hat[h] - r[h]) + (P_hat[h] - P[h]) @ V_hat[h + 1]      # model values
            form1 += d_model[h] @ np.einsum("sa,sa->s", pi[h], err1)
            form2 += d_true[h] @ np.einsum("sa,sa->s", pi[h], err2)
        worst1, worst2 = max(worst1, abs(lhs - form1)), max(worst2, abs(lhs - form2))
        # Bound: |(Phat-P) V_{h+1}| <= ||Phat-P||_1 * span(V_{h+1}) / 2 and span(V_{h+1}) <= H-h-1.
        eps_r = np.abs(r_hat - r).max()
        eps_P = np.abs(P_hat - P).sum(-1).max()
        bound = H * eps_r + sum((H - h - 1) * eps_P / 2 for h in range(H))
        worst_ratio = max(worst_ratio, np.abs(V_hat[0] - V[0]).max() / bound)
    print(f"(3) simulation lemma, finite horizon ({n_models} perturbed models): "
          f"max residual, model-trajectory form = {worst1:.1e}, true-trajectory form = {worst2:.1e}; "
          f"max (actual / bound) = {worst_ratio:.3f}")


def check_greedy_loss(rng, n_trials, S=10, A=4, gamma=0.9):
    P, r = random_mdp(rng, S, A)
    q_star, v_star = value_iteration(P, r, gamma)
    worst = 0.0
    for _ in range(n_trials):
        eps = 10 ** rng.uniform(-3, 0)
        Q = q_star + rng.uniform(-eps, eps, size=q_star.shape)    # ||Q - q*||_inf <= eps
        v_pi, _ = policy_eval(P, r, gamma, greedy(Q))
        worst = max(worst, (v_star - v_pi).max() / (2 * eps / (1 - gamma)))
    print(f"(4) greedy loss on a random MDP ({n_trials} perturbations of q*): "
          f"max loss / (2 eps/(1-gamma)) = {worst:.3f}  (must be <= 1)")
    # The tight two-action example (Section 2.1, worked example): one state, two self-loops with
    # rewards 1 and 1 - Delta (code: delta). Perturb q* by +eps on the worse action and -eps on the better one.
    for delta_over_eps in (1.0, 1.9, 1.99):
        eps, g = 0.05, gamma
        delta = delta_over_eps * eps
        q = np.array([1.0 + g / (1 - g), 1.0 - delta + g / (1 - g)])     # q*(x, L), q*(x, R)
        Q = q + np.array([-eps, +eps]) * 0.999999                        # strictly inside the ball
        loss = (1.0 - (1.0 - delta)) / (1 - g) if Q.argmax() == 1 else 0.0
        print(f"    tight example: gap Delta = {delta_over_eps:.2f} eps, eps = {eps}, gamma = {g}: "
              f"loss = {loss:.4f}, bound 2eps/(1-gamma) = {2 * eps / (1 - g):.4f}")


def check_approximate_vi(rng, S=10, A=4, gamma=0.9, eps=0.01, iters=400):
    P, r = random_mdp(rng, S, A)
    q_star, _ = value_iteration(P, r, gamma)
    results = {}
    for kind in ("random sign", "constant +eps"):
        Q = np.zeros((S, A))
        for _ in range(iters):
            noise = rng.uniform(-eps, eps, size=Q.shape) if kind == "random sign" else np.full(Q.shape, eps)
            Q = r + gamma * P @ Q.max(axis=1) + noise          # ||Q_{k+1} - T*Q_k||_inf <= eps
        results[kind] = np.abs(Q - q_star).max()
    print(f"(5) approximate VI (eps = {eps}, gamma = {gamma}): final ||Q_k - q*||_inf = "
          f"{results['random sign']:.4f} (random errors), {results['constant +eps']:.4f} (constant +eps); "
          f"bound eps/(1-gamma) = {eps / (1 - gamma):.4f}")


def check_simulation_lemma_discounted(rng, n_models, S=10, A=3, gamma=0.9):
    P, r = random_mdp(rng, S, A)
    worst = 0.0
    for _ in range(n_models):
        lam = rng.uniform(0.01, 0.3)
        P_hat = (1 - lam) * P + lam * rng.dirichlet(np.ones(S), size=(S, A))
        pi = random_policy(rng, S, A)
        v, _ = policy_eval(P, r, gamma, pi)
        v_hat, _ = policy_eval(P_hat, r, gamma, pi)
        eps_P = np.abs(P_hat - P).sum(-1).max()
        bound = gamma * eps_P / (2 * (1 - gamma) ** 2)
        worst = max(worst, np.abs(v_hat - v).max() / bound)
    print(f"(6) simulation lemma, discounted bound gamma eps_P/(2(1-gamma)^2) ({n_models} models): "
          f"max actual/bound = {worst:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer random instances")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n = 100 if args.quick else 2000
    print(f"seed={args.seed}  random instances per check={n}")
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    check_pdl_discounted(rng, n)
    check_pdl_finite_horizon(rng, n)
    check_simulation_lemma_fh(rng, n)
    check_greedy_loss(rng, n)
    check_approximate_vi(rng)
    check_simulation_lemma_discounted(rng, n)
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
