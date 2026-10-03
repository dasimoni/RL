"""Global convergence of exact policy gradients: softmax PG vs natural PG. Chapter 19, Section 8.

Tabular softmax policies pi_theta(a|s) proportional to exp(theta[s,a]) on a random MDP, with EXACT
gradients (no sampling), so every effect below is about optimisation geometry, not noise.

  * Softmax PG:  theta <- theta + eta * dV(mu)/dtheta,
        dV(mu)/dtheta[s,a] = 1/(1-gamma) d^pi_mu(s) pi(a|s) A^pi(s,a)                     (Eq. 19.25)
    with eta = (1-gamma)^3/8 (the step that the 8/(1-gamma)^3 smoothness guarantees) and larger eta.
  * Natural PG (NPG): theta <- theta + eta/(1-gamma) A^pi(s,a), i.e.
        pi_{t+1}(a|s) proportional to pi_t(a|s) exp(eta A^{pi_t}(s,a)/(1-gamma))             (Eq. 19.28)
    and the bound (19.29) of Agarwal, Kakade, Lee & Mahajan (2021):
        V*(rho) - V^{(T)}(rho) <= log|A|/(eta T) + 1/((1-gamma)^2 T).
Checks printed:
  (1) gradient domination for the direct parameterisation (Lemma 19.12) on random policies;
  (2) the non-uniform Lojasiewicz inequality for softmax (Eq. 19.26) along the PG trajectory;
  (3) t * (V* - V^{pi_t}) for softmax PG (the O(1/t) rate) and the minimum optimal-action
      probability c_t = min_s pi_t(a*(s)|s) that enters the constant;
  (4) the NPG bound at every iteration;
  (5) the effect of the distribution mismatch coefficient ||d^{pi*}_rho / mu||_inf: softmax PG
      run on V(mu) for increasingly concentrated mu, and NPG (whose update does not involve mu).

Run from the repository root:
  python code/ch19_rl_theory/pg_convergence.py           # full run (~1 min), writes the figure
  python code/ch19_rl_theory/pg_convergence.py --quick   # smoke test, no figure
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from theory_lib import (COLORS, loglog_slope, occupancy, policy_eval, random_mdp, random_policy,
                        setup_style, value_iteration)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")


def softmax(theta):
    z = theta - theta.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def run(P, r, gamma, mu, rho, method, eta, iters, checkpoints, a_star=None, first_below=1e-3):
    """Exact PG or NPG from theta = 0 (uniform policy). Iteration t evaluates pi_t, the policy AFTER t
    updates (t = 0 is the uniform policy), records its gap on rho if t is a checkpoint, then applies the
    (t+1)-th update. Returns the gaps of pi_t at the checkpoints, the running minimum over t of
    c_t = min_s pi_t(a*(s)|s), the worst Lojasiewicz ratio, and the first t with gap < first_below."""
    S, A = r.shape
    _, v_star = value_iteration(P, r, gamma)
    J_star = rho @ v_star
    theta = np.zeros((S, A))
    gaps = np.zeros(len(checkpoints))
    c_min = 1.0
    loj_min = np.inf
    t_first = None
    d_star_rho = None
    if a_star is not None:
        pi_star = np.zeros((S, A))
        pi_star[np.arange(S), a_star] = 1.0
        d_star_rho = occupancy(P, gamma, pi_star, rho)
    ci = 0
    for t in range(0, iters + 1):
        pi = softmax(theta)                                                        # pi_t
        v, q = policy_eval(P, r, gamma, pi)
        gap = J_star - rho @ v
        if t_first is None and gap < first_below:
            t_first = t
        if a_star is not None:
            c_min = min(c_min, pi[np.arange(S), a_star].min())
        if ci < len(checkpoints) and t == checkpoints[ci]:
            gaps[ci] = gap                       # gap of pi_t, the policy after t updates
            ci += 1
        if t == iters:
            break
        adv = q - v[:, None]
        if method == "pg":
            d_mu = occupancy(P, gamma, pi, mu)
            grad = d_mu[:, None] * pi * adv / (1 - gamma)                     # Eq. (19.25)
            if a_star is not None and t % 10 == 0:
                # Non-uniform Lojasiewicz (Eq. 19.26), with rho = mu: ||grad||_2 >= c_t/(sqrt(S) D_t) (V*-V)
                c_t = pi[np.arange(S), a_star].min()
                D_t = np.max(d_star_rho / d_mu)
                gap_mu = rho @ (v_star - v)
                if gap_mu > 1e-12:
                    loj_min = min(loj_min, np.linalg.norm(grad) / (c_t / (np.sqrt(S) * D_t) * gap_mu))
            theta = theta + eta * grad
        elif method == "npg":
            theta = theta + eta * adv / (1 - gamma)                               # Eq. (19.28)
    return gaps, c_min, loj_min, t_first


def gradient_domination_check(rng, P, r, gamma, n_policies=500):
    S, A = r.shape
    q_star, v_star = value_iteration(P, r, gamma)
    pi_star = np.zeros((S, A))
    pi_star[np.arange(S), q_star.argmax(1)] = 1.0
    rho = np.full(S, 1.0 / S)
    worst_tight, worst_loose = 0.0, 0.0
    for _ in range(n_policies):
        mu = rng.dirichlet(np.ones(S))
        pi = random_policy(rng, S, A, conc=0.3)
        v, q = policy_eval(P, r, gamma, pi)
        adv = q - v[:, None]
        d_mu = occupancy(P, gamma, pi, mu)
        d_star = occupancy(P, gamma, pi_star, rho)
        max_dir = (d_mu @ adv.max(axis=1)) / (1 - gamma)        # max_pibar (pibar - pi)^T grad_pi V^pi(mu)
        lhs = rho @ (v_star - v)
        rhs_tight = np.max(d_star / d_mu) * max_dir
        rhs_loose = np.max(d_star / mu) / (1 - gamma) * max_dir
        worst_tight = max(worst_tight, lhs / rhs_tight)
        worst_loose = max(worst_loose, lhs / rhs_loose)
    print(f"(1) gradient domination (Lemma 19.12) on {n_policies} random (pi, mu): max LHS/RHS = "
          f"{worst_tight:.3f} with ||d*/d^pi_mu||, {worst_loose:.3f} with ||d*/mu||/(1-gamma)  (must be <= 1)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figure")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    S, A, gamma = 10, 4, 0.9
    iters = 3_000 if args.quick else 100_000
    print(f"seed={args.seed}  S={S} A={A} gamma={gamma}  iterations={iters}  (exact gradients)")
    rng = np.random.default_rng(args.seed)
    P, r = random_mdp(rng, S, A, conc=0.5, branching=3)
    q_star, v_star = value_iteration(P, r, gamma)
    a_star = q_star.argmax(axis=1)
    rho = np.full(S, 1.0 / S)
    t0 = time.time()
    gradient_domination_check(rng, P, r, gamma, n_policies=100 if args.quick else 1000)

    # Log-spaced checkpoints that contain every power of 10 exactly (the reported columns).
    powers = 10 ** np.arange(0, int(np.floor(np.log10(iters) + 1e-9)) + 1)
    checkpoints = np.unique(np.concatenate([np.round(np.logspace(0, np.log10(iters), 60)).astype(int),
                                            powers, [iters]]))
    eta_theory = (1 - gamma) ** 3 / 8
    configs = [
        ("softmax PG, η = (1-γ)³/8", "pg", eta_theory),
        ("softmax PG, η = 1", "pg", 1.0),
        ("softmax PG, η = 10", "pg", 10.0),
        ("NPG, η = 0.1", "npg", 0.1),
        ("NPG, η = 1", "npg", 1.0),
    ]
    curves = {}
    print(f"\n(2)-(4) mu = rho = uniform; J* = {rho @ v_star:.4f}")
    for name, method, eta in configs:
        gaps, c_min, loj, _ = run(P, r, gamma, rho, rho, method, eta, iters, checkpoints, a_star)
        curves[name] = gaps
        report = [n for n in (10, 100, 1000, 10_000, 100_000) if n <= iters]
        idx = {n: int(np.searchsorted(checkpoints, n)) for n in report}
        assert all(checkpoints[idx[n]] == n for n in report)        # exact t, not a nearby grid point
        vals = {n: gaps[idx[n]] for n in report}
        line = "  ".join(f"t={n:>6d}: {vals[n]:.2e}" for n in report)
        print(f"  {name:<26s} gap  {line}")
        if method == "pg":
            tg = "  ".join(f"{n * vals[n]:.3g}" for n in report)
            print(f"  {'':<26s} t*gap: {tg};  min_t c_t = {c_min:.4f};  min Lojasiewicz ratio = {loj:.3f} (>= 1)")
        else:
            bound = np.log(A) / (eta * checkpoints) + 1 / ((1 - gamma) ** 2 * checkpoints)
            ok = np.all(gaps <= bound + 1e-12)
            first = checkpoints[np.argmax(gaps < 1e-10)] if np.any(gaps < 1e-10) else None
            print(f"  {'':<26s} NPG bound (19.29) holds at every checkpoint: {ok}; "
                  f"bound at T=100: {bound[idx[100]]:.3f}; gap < 1e-10 from checkpoint t = {first}")

    # (5) distribution mismatch: optimise V(mu_beta), measure the gap on rho = uniform.
    pi_star = np.zeros((S, A))
    pi_star[np.arange(S), a_star] = 1.0
    d_star = occupancy(P, gamma, pi_star, rho)
    worst_state = int(np.argmin(d_star))
    mism = {}
    print(f"\n(5) distribution mismatch: mu_beta = (1-beta) * e_{worst_state} + beta * uniform; gap measured on rho = uniform")
    for beta in (1.0, 0.1, 0.01):
        mu = np.full(S, beta / S)
        mu[worst_state] += 1 - beta
        D = np.max(d_star / mu)
        gaps, _, _, hit = run(P, r, gamma, mu, rho, "pg", 1.0, iters, checkpoints)
        mism[beta] = (D, gaps)
        report = [n for n in (1000, 10_000, 100_000) if n <= iters]
        at = "  ".join(f"t={n}: {gaps[np.searchsorted(checkpoints, n)]:.2e}" for n in report if n in checkpoints)
        print(f"  beta = {beta:5.2f}: ||d*_rho/mu||_inf = {D:7.2f}; softmax PG (eta=1) first t with gap < 1e-3: {hit}; "
              f"gap at {at}")
    print(f"  NPG's tabular update does not depend on mu at all (Eq. 19.28), so its curve is the same for every beta.")
    print(f"\ntotal time {time.time() - t0:.1f} s")
    if args.quick:
        return

    setup_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    styles = ["-", "--", "-.", "-", "--"]
    for i, (name, method, eta) in enumerate(configs):
        g = np.maximum(curves[name], 1e-16)
        ax.loglog(checkpoints, g, styles[i], color=COLORS[i], label=name)
    ref = checkpoints[checkpoints >= 100].astype(float)
    g1 = curves["softmax PG, η = 1"][-1]
    ax.loglog(ref, g1 * checkpoints[-1] / ref, color="#444444", lw=0.9, ls=(0, (1, 1.5)), label="slope −1 (O(1/t))")
    ax.loglog(checkpoints, np.log(A) / (1.0 * checkpoints) + 1 / ((1 - gamma) ** 2 * checkpoints), "-",
              color="#999999", lw=1.0, label="NPG bound (19.29), η = 1 (η = 0.1 is almost the same)")
    ax.set_xlabel("iteration t")
    ax.set_ylabel(r"$V^\ast(\rho) - V^{\pi_t}(\rho)$")
    ax.set_title(f"Exact gradients, random MDP (S={S}, A={A}, γ={gamma})")
    ax.set_ylim(1e-12, 30)
    ax.legend(fontsize=7.3, loc="lower left")
    ax = axes[1]
    for i, beta in enumerate((1.0, 0.1, 0.01)):
        D, g = mism[beta]
        ax.loglog(checkpoints, np.maximum(g, 1e-16), ["-", "--", "-."][i], color=COLORS[i],
                  label=f"softmax PG on V(μ), β = {beta}  (D∞ = {D:.0f})")
    ax.loglog(checkpoints, np.maximum(curves["NPG, η = 1"], 1e-16), "-", color=COLORS[4], label="NPG, η = 1 (any μ)")
    ax.set_xlabel("iteration t")
    ax.set_ylabel(r"$V^\ast(\rho) - V^{\pi_t}(\rho)$, ρ uniform")
    ax.set_title("Distribution mismatch delays softmax PG's plateau escape;\nNPG is unaffected", fontsize=10)
    ax.set_ylim(1e-12, 30)
    ax.legend(fontsize=7.6, loc="lower left")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "pg_convergence.png")
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
