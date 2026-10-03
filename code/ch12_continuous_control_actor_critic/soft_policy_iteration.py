"""Soft (maximum-entropy) policy iteration on a small random MDP, done exactly (Chapter 12, Section 5).

We check, with exact linear algebra (no sampling):

  1. Soft policy evaluation: the soft Bellman operator T^pi is a gamma-contraction (Eq. 12.16),
     and its fixed point solves v = (I - gamma P_pi)^{-1} (r_pi + alpha H_pi).
  2. Soft policy improvement (Lemma 12.5): pi_new(.|s) ∝ exp(q_old(s,.)/alpha) satisfies
     q_new(s,a) >= q_old(s,a) for EVERY (s,a), at every iteration.
  3. Soft policy iteration (Theorem 12.6) converges to the soft-optimal q* found independently by
     soft value iteration  v(s) <- alpha log sum_a exp(q(s,a)/alpha)  (Eq. 12.20).
  4. The temperature limit: v* <= v*_soft <= v* + alpha log|A| / (1 - gamma), and the soft-optimal
     policy is at most alpha log|A| / (1 - gamma) worse than optimal in the ordinary MDP.

Run (from the repository root):
  python code/ch12_continuous_control_actor_critic/soft_policy_iteration.py          # full, figure
  python code/ch12_continuous_control_actor_critic/soft_policy_iteration.py --quick  # no figure
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.special import logsumexp, softmax

from common import FIG_DIR


def random_mdp(n_s: int, n_a: int, rng):
    P = rng.dirichlet(0.3 * np.ones(n_s), size=(n_s, n_a))     # P[s, a, s']
    R = rng.uniform(-1.0, 1.0, size=(n_s, n_a))                 # r(s, a)
    return P, R


def soft_evaluate(pi, P, R, gamma, alpha):
    """Exact soft policy evaluation: returns (q, v) with
    v(s) = sum_a pi(a|s) [q(s,a) - alpha log pi(a|s)],  q = r + gamma P v     (Eqs. 12.12-12.13)."""
    n_s = R.shape[0]
    logpi = np.log(np.clip(pi, 1e-300, None))
    r_pi = (pi * (R - alpha * logpi)).sum(1)                    # r_pi(s) + alpha H(pi(.|s))
    P_pi = np.einsum("sa,sat->st", pi, P)
    v = np.linalg.solve(np.eye(n_s) - gamma * P_pi, r_pi)
    return R + gamma * P @ v, v


def soft_bellman(q, pi, P, R, gamma, alpha):
    """T^pi q (Eq. 12.15)."""
    logpi = np.log(np.clip(pi, 1e-300, None))
    v = (pi * (q - alpha * logpi)).sum(1)
    return R + gamma * P @ v


def soft_value_iteration(P, R, gamma, alpha, tol=1e-13, max_iter=100_000, keep_iterates=False):
    """Iterate v <- alpha log sum_a exp((r + gamma P v)/alpha) until ||v_{k+1} - v_k|| < tol.
    Returns (q, v, errs) and, with keep_iterates, also the list of q-iterates q_k = r + gamma P v_k."""
    v = np.zeros(R.shape[0])
    errs, qs = [], []
    for _ in range(max_iter):
        q = R + gamma * P @ v
        if keep_iterates:
            qs.append(q)
        v_new = alpha * logsumexp(q / alpha, axis=1)
        errs.append(np.max(np.abs(v_new - v)))
        v = v_new
        if errs[-1] < tol:
            break
    q = R + gamma * P @ v
    if keep_iterates:
        return q, v, errs, qs
    return q, v, errs


def hard_value_iteration(P, R, gamma, tol=1e-13):
    v = np.zeros(R.shape[0])
    while True:
        q = R + gamma * P @ v
        v_new = q.max(1)
        if np.max(np.abs(v_new - v)) < tol:
            return R + gamma * P @ v_new, v_new
        v = v_new


def evaluate_hard(pi, P, R, gamma):
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = (pi * R).sum(1)
    return np.linalg.solve(np.eye(R.shape[0]) - gamma * P_pi, r_pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seed, n_s, n_a, gamma = 0, 10, 4, 0.9
    rng = np.random.default_rng(seed)
    P, R = random_mdp(n_s, n_a, rng)
    print(f"Soft policy iteration | seed {seed} | random MDP with |S|={n_s}, |A|={n_a}, gamma={gamma}")

    # ---- 1. contraction of the soft Bellman operator ----
    pi = softmax(rng.normal(size=(n_s, n_a)), axis=1)
    worst = 0.0
    for _ in range(200 if args.quick else 2000):
        q1, q2 = rng.normal(size=(2, n_s, n_a)) * 5
        ratio = np.max(np.abs(soft_bellman(q1, pi, P, R, gamma, 1.0) - soft_bellman(q2, pi, P, R, gamma, 1.0))) \
            / np.max(np.abs(q1 - q2))
        worst = max(worst, ratio)
    q1 = rng.normal(size=(n_s, n_a))                       # a constant shift attains the bound exactly
    tight = np.max(np.abs(soft_bellman(q1 + 1.0, pi, P, R, gamma, 1.0) - soft_bellman(q1, pi, P, R, gamma, 1.0)))
    q_fix, _ = soft_evaluate(pi, P, R, gamma, 1.0)
    resid = np.max(np.abs(soft_bellman(q_fix, pi, P, R, gamma, 1.0) - q_fix))
    print(f"\n1) soft Bellman operator: max ||T q1 - T q2|| / ||q1 - q2|| over random pairs = {worst:.4f} "
          f"(<= gamma = {gamma}); for q2 = q1 + 1 the ratio is {tight:.4f};  fixed-point residual of the linear solve = {resid:.1e}")

    # ---- 2-3. soft policy iteration vs soft value iteration ----
    alphas = [1.0, 0.3, 0.1]
    spi_curves, all_gains = {}, []
    print("\n2-3) soft policy iteration from the uniform policy:")
    for alpha in alphas:
        q_star, v_star, vi_errs, vi_qs = soft_value_iteration(P, R, gamma, alpha, keep_iterates=True)
        vi_q_errs = [np.max(np.abs(qk - q_star)) for qk in vi_qs]     # same metric as for SPI
        pi = np.full((n_s, n_a), 1.0 / n_a)
        q_old, _ = soft_evaluate(pi, P, R, gamma, alpha)
        errs, min_gain = [np.max(np.abs(q_old - q_star))], np.inf
        for k in range(50):
            pi = softmax(q_old / alpha, axis=1)                     # soft policy improvement (Eq. 12.19)
            q_new, _ = soft_evaluate(pi, P, R, gamma, alpha)        # soft policy evaluation
            min_gain = min(min_gain, np.min(q_new - q_old))         # Lemma 12.5: should be >= 0
            errs.append(np.max(np.abs(q_new - q_star)))
            q_old = q_new
            if errs[-1] < 1e-12:
                break
        spi_curves[alpha] = (errs, vi_errs, vi_q_errs)
        all_gains.append(min_gain)
        pi_star = softmax(q_star / alpha, axis=1)
        print(f"   alpha={alpha:4.1f}: SPI reached ||q_k - q*|| < 1e-12 after {len(errs) - 1} iterations "
              f"(soft VI needed {len(vi_errs)} sweeps);  min over k,s,a of q_(k+1) - q_k = {min_gain:.2e};  "
              f"final error {errs[-1]:.1e};  mean policy entropy at optimum "
              f"{-(pi_star * np.log(pi_star)).sum(1).mean():.3f} nats (max log|A| = {np.log(n_a):.3f})")

    # ---- 4. temperature limit ----
    q_hard, v_hard = hard_value_iteration(P, R, gamma)
    sweep = np.logspace(-3, 1, 25)
    gaps, subopt, ents = [], [], []
    for alpha in sweep:
        q_s, v_s, _ = soft_value_iteration(P, R, gamma, alpha)
        pi_a = softmax(q_s / alpha, axis=1)
        gaps.append(np.max(v_s - v_hard))
        lower_ok = np.min(v_s - v_hard) >= -1e-9
        subopt.append(np.max(v_hard - evaluate_hard(pi_a, P, R, gamma)))
        ents.append(-(pi_a * np.log(np.clip(pi_a, 1e-300, None))).sum(1).mean())
        assert lower_ok
    bound = sweep * np.log(n_a) / (1 - gamma)
    print("\n4) temperature limit (max over states):")
    for i in [0, 6, 12, 18, 24]:
        print(f"   alpha={sweep[i]:7.4f}: v*_soft - v* = {gaps[i]:8.4f}   v* - v^(pi_alpha) = {subopt[i]:8.4f}   "
              f"bound alpha log|A|/(1-gamma) = {bound[i]:8.4f}   mean entropy {ents[i]:.3f}")
    ok = np.all(np.array(gaps) <= bound + 1e-9) and np.all(np.array(subopt) <= bound + 1e-9)
    print(f"   both gaps within the bound for all {len(sweep)} temperatures: {ok}")

    # worked example of Section 5.5: one state, two actions, q = (1, 0)
    for alpha in [1.0, 0.1]:
        q = np.array([1.0, 0.0])
        v = alpha * logsumexp(q / alpha)
        p = softmax(q / alpha)
        print(f"   worked example q=(1,0), alpha={alpha}: soft value {v:.6f}, Boltzmann policy {np.round(p, 6)}")

    print("\n=== Summary ===")
    mono = all(g >= -1e-10 for g in all_gains)
    print(f"T^pi contraction factor <= {worst:.3f} on random pairs ({tight:.3f} for a constant shift); "
          f"q_(k+1) >= q_k (up to 1e-10) at every iteration for all alphas: {mono}; "
          f"SPI converged in {[len(spi_curves[a][0]) - 1 for a in alphas]} iterations for alpha={alphas}; "
          f"temperature bounds hold: {ok}")

    if not args.quick:
        from plot_style import C, setup
        plt = setup()
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
        ax = axes[0]
        for i, alpha in enumerate(alphas):
            errs, _, vi_q_errs = spi_curves[alpha]
            ax.semilogy(range(len(errs)), np.maximum(errs, 1e-16), "o-", color=C[i], ms=4,
                        label=f"soft policy iteration, alpha={alpha}")
            # q* is VI's own last iterate (accurate to ~1e-12), so errors below 1e-11 are not
            # meaningful and are not drawn
            k_ok = [k for k, e in enumerate(vi_q_errs) if e >= 1e-11]
            ax.semilogy(k_ok, [vi_q_errs[k] for k in k_ok], color=C[i], lw=1, ls="--",
                        label=f"soft value iteration, alpha={alpha}")
        ax.set_xlim(-5, 300)
        ax.set_xlabel("iteration (SPI) / sweep (VI)  k")
        ax.set_ylabel("||q_k - q*||  (max norm)")
        ax.set_title("soft PI converges in a handful of iterations")
        ax.legend(fontsize=7)
        ax = axes[1]
        ax.loglog(sweep, np.maximum(gaps, 1e-13), "o-", color=C[0], ms=4, label="v*_soft - v*  (max over s)")
        ax.loglog(sweep, np.maximum(subopt, 1e-12), "s-", color=C[1], ms=4,
                  label="v* - v of soft-optimal policy")
        ax.loglog(sweep, bound, color="#8a8985", ls="--", label="alpha log|A| / (1 - gamma)")
        ax.set_xlabel("temperature alpha")
        ax.set_ylim(1e-13, 1e3)
        ax.set_title("alpha -> 0 recovers the ordinary optimum")
        ax.legend(fontsize=8)
        ax = axes[2]
        s_show = 0
        probs = np.array([softmax(soft_value_iteration(P, R, gamma, a)[0][s_show] / a) for a in sweep])
        for j in range(n_a):
            ax.semilogx(sweep, probs[:, j], color=C[j], marker="osd^"[j], ms=3, label=f"action {j}")
        ax.set_xlabel("temperature alpha")
        ax.set_ylabel(f"soft-optimal pi*(a | s={s_show})")
        ax.set_title("soft-optimal (Boltzmann) policy at one state")
        ax.legend(fontsize=8)
        fig.tight_layout()
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, "soft_policy_iteration.png")
        fig.savefig(path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
