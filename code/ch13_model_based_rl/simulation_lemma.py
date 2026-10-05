"""The simulation lemma, checked numerically on tabular MDPs.

Chapter 13, Sections 2.2-2.4 (Eqs. 13.2-13.5).

For an MDP M = (P, r) and a model M_hat = (P_hat, r_hat) with the same S, A and gamma, and any
stationary policy pi:

  exact identity (Eq. 13.2):
      V_hat - V = (I - gamma P_hat_pi)^{-1} [ (r_hat_pi - r_pi) + gamma (P_hat_pi - P_pi) V ]
  occupancy form (Eq. 13.3), d0 an initial distribution, d_hat the normalised discounted state
  occupancy of pi in the MODEL:
      V_hat(d0) - V(d0) = 1/(1-gamma) E_{s ~ d_hat}[ (r_hat_pi - r_pi)(s) + gamma ((P_hat_pi - P_pi) V)(s) ]
  bound (Eq. 13.4), eps_r = max |r_hat - r|, eps_P = max_{s,a} ||P_hat(.|s,a) - P(.|s,a)||_1,
  rewards in [0, R_max]:
      || V_hat - V ||_inf <= eps_r / (1-gamma) + gamma eps_P R_max / (2 (1-gamma)^2)
  (the factor 1/2 uses span(V) <= R_max/(1-gamma); the commonly quoted version omits it).

The script (1) verifies the identity to machine precision, (2) works the two-state example of
Section 2.3, which attains the bound up to a factor that tends to 1 as eps -> 0, and (3) measures how
loose the bound is on random MDPs and how much an agent loses by acting optimally in the model.

Run (from the repository root):
  python code/ch13_model_based_rl/simulation_lemma.py           # full (figure)
  python code/ch13_model_based_rl/simulation_lemma.py --quick   # smoke test, no figure
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True          # importing plot_style must not leave __pycache__/
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")


def policy_eval(P, r, pi, gamma):
    """Exact V^pi for P (S, A, S), r (S, A), deterministic pi (S,)."""
    S = P.shape[0]
    P_pi = P[np.arange(S), pi]           # (S, S)
    r_pi = r[np.arange(S), pi]           # (S,)
    return np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi), P_pi, r_pi


def optimal_policy(P, r, gamma, max_iter=1000):
    """Exact optimal deterministic policy by policy iteration (Chapter 03): evaluate exactly, improve
    greedily, stop when no action improves by more than a relative 1e-12 (so ties cannot cycle).
    Policy iteration terminates in finitely many steps, unlike a value-iteration loop whose stopping
    threshold could fall below float64 resolution when |V| ~ 1/(1-gamma) is large."""
    S = P.shape[0]
    pi = r.argmax(1)
    for _ in range(max_iter):
        V, _, _ = policy_eval(P, r, pi, gamma)
        Q = r + gamma * P @ V
        cur = Q[np.arange(S), pi]
        better = Q.max(1) > cur + 1e-12 * max(1.0, np.abs(V).max())
        if not better.any():
            return pi, V
        pi = np.where(better, Q.argmax(1), pi)
    raise RuntimeError("policy iteration did not converge")


def random_mdp(rng, S, A, conc=1.0):
    P = rng.dirichlet(np.full(S, conc), size=(S, A))
    r = rng.uniform(0, 1, size=(S, A))
    return P, r


def check_identity(rng, gamma=0.9):
    """Verify Eqs. 13.2-13.3 on a random MDP with a perturbed transition AND reward model."""
    S, A = 15, 3
    P, r = random_mdp(rng, S, A)
    U, _ = random_mdp(rng, S, A)
    P_hat = 0.8 * P + 0.2 * U
    r_hat = np.clip(r + 0.05 * rng.standard_normal(r.shape), 0, 1)
    pi = rng.integers(0, A, size=S)
    V, P_pi, r_pi = policy_eval(P, r, pi, gamma)
    V_hat, Ph_pi, rh_pi = policy_eval(P_hat, r_hat, pi, gamma)
    rhs = np.linalg.solve(np.eye(S) - gamma * Ph_pi, (rh_pi - r_pi) + gamma * (Ph_pi - P_pi) @ V)
    d0 = np.full(S, 1.0 / S)
    d_hat = (1 - gamma) * np.linalg.solve((np.eye(S) - gamma * Ph_pi).T, d0)   # normalised occupancy
    occ = (d_hat @ ((rh_pi - r_pi) + gamma * (Ph_pi - P_pi) @ V)) / (1 - gamma)
    print(f"(1) identity (13.2): max |lhs - rhs| = {np.max(np.abs((V_hat - V) - rhs)):.2e};  "
          f"occupancy form (13.3): |lhs - rhs| = {abs(d0 @ (V_hat - V) - occ):.2e};  sum d_hat = {d_hat.sum():.6f}")


def two_state_example(gamma, eps):
    """State A: reward 1, absorbing in M.  The model leaks probability eps to an absorbing state B with
    reward 0.  V(A) = 1/(1-gamma),  V_hat(A) = 1/(1-gamma(1-eps)),  eps_P = 2 eps."""
    V = 1 / (1 - gamma)
    V_hat = 1 / (1 - gamma * (1 - eps))
    bound = gamma * (2 * eps) / (2 * (1 - gamma) ** 2)
    return V, V_hat, V - V_hat, bound


def random_mdp_study(rng, gammas, eps_targets, n_mdps, n_pols, S=20, A=4):
    """For each gamma and perturbation size: the worst observed value error over a set of policies
    (what the bound 13.4 addresses), the bound, the value error over the two policies that matter
    for the corollary (13.5), and the loss of acting optimally in the model."""
    out = {g: {"eps": [], "err": [], "err_two": [], "bound": [], "loss": []} for g in gammas}
    for _ in range(n_mdps):
        P, r = random_mdp(rng, S, A, conc=0.3)          # fairly sparse transitions
        U, _ = random_mdp(rng, S, A, conc=0.3)   # random kernel the model leaks towards
        for lam in eps_targets:
            P_hat = (1 - lam) * P + lam * U
            eps_P = np.abs(P_hat - P).sum(-1).max()
            for g in gammas:
                pi_hat, _ = optimal_policy(P_hat, r, g)
                pi_star, V_star = optimal_policy(P, r, g)
                pols = [pi_hat, pi_star] + [rng.integers(0, A, size=S) for _ in range(n_pols)]
                errs = []
                for pi in pols:
                    V, _, _ = policy_eval(P, r, pi, g)
                    V_h, _, _ = policy_eval(P_hat, r, pi, g)
                    errs.append(np.max(np.abs(V_h - V)))
                V_pihat, _, _ = policy_eval(P, r, pi_hat, g)
                out[g]["eps"].append(eps_P)
                out[g]["err"].append(max(errs))            # max over all 2 + n_pols policies (Eq. 13.4)
                out[g]["err_two"].append(max(errs[:2]))    # epsilon_V of Eq. 13.5: pi_hat and pi_star only
                out[g]["bound"].append(g * eps_P / (2 * (1 - g) ** 2))
                out[g]["loss"].append(max(0.0, np.max(V_star - V_pihat)))   # >= 0 up to round-off
    return {g: {k: np.array(v) for k, v in d.items()} for g, d in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    print(f"seed {args.seed}")
    check_identity(rng)

    print("\n(2) two-state example (Section 2.3): V(A) - V_hat(A) vs the bound gamma*eps_P*R_max/(2(1-gamma)^2)")
    print("   gamma     eps    V(A)   V_hat(A)   error    bound   error/bound")
    for g in (0.9, 0.99):
        for eps in (0.01, 0.001):
            V, Vh, e, b = two_state_example(g, eps)
            print(f"   {g:5.2f}  {eps:6.3f}  {V:7.2f}  {Vh:8.3f}  {e:7.3f}  {b:7.3f}   {e / b:6.3f}")

    gammas = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99]
    lams = [0.01, 0.03, 0.1] if not args.quick else [0.03]
    n_mdps, n_pols = (20, 30) if not args.quick else (2, 5)
    res = random_mdp_study(rng, gammas, lams, n_mdps, n_pols)
    print(f"\n(3) random MDPs (S=20, A=4, {n_mdps} MDPs x {len(lams)} perturbation sizes); "
          "P_hat = (1-lam) P + lam U, rewards exact")
    print("   err = max over pi_hat, pi_star and random policies of ||V_hat - V||_inf (what 13.4 bounds);")
    print("   eps_V = the same max over pi_hat and pi_star only (what 13.5 needs)")
    print("   gamma  1/(1-g)  mean eps_P  mean err  mean bound  mean err/bound  max err/bound  "
          "mean loss  max loss/(2 eps_V)  frac. pi_hat suboptimal")
    for g in gammas:
        d = res[g]
        assert np.all(d["err"] <= d["bound"] + 1e-12), "simulation lemma violated?!"
        assert np.all(d["loss"] <= 2 * d["err_two"] + 1e-12), "corollary (13.5) violated?!"
        ratio = d["loss"] / np.maximum(2 * d["err_two"], 1e-12)
        print(f"   {g:5.2f}  {1 / (1 - g):7.0f}  {d['eps'].mean():10.3f}  {d['err'].mean():8.4f}  "
              f"{d['bound'].mean():10.2f}  {np.mean(d['err'] / d['bound']):14.4f}  "
              f"{np.max(d['err'] / d['bound']):13.4f}  {d['loss'].mean():9.4f}  {ratio.max():18.3f}"
              f"  {np.mean(d['loss'] > 1e-9):23.2f}")
    print(f"   bound (13.4) held in all {sum(len(res[g]['err']) for g in gammas)} cases; "
          f"corollary (13.5) held in all of them")
    print(f"Total time {time.time() - t0:.1f}s")
    if args.quick:
        return

    sys.path.insert(0, HERE)
    from plot_style import C, GREY, setup
    plt = setup()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axes[0]
    H = np.array([1 / (1 - g) for g in gammas])
    for k, lam in enumerate(lams):
        errs = [np.mean(res[g]["err"][k::len(lams)]) for g in gammas]
        bnds = [np.mean(res[g]["bound"][k::len(lams)]) for g in gammas]
        ax.plot(H, errs, color=C[k], marker="o", label=f"random MDPs, observed ($\\lambda$={lam})")
        ax.plot(H, bnds, color=C[k], ls="--", lw=1.2, label=f"bound (13.4), $\\lambda$={lam}")
    ex = [two_state_example(g, 0.01)[2] for g in gammas]
    ax.plot(H, ex, color=GREY, marker="s", ls=":", label="two-state example, $\\epsilon$=0.01")
    ax.plot(H, [g * 0.02 / (2 * (1 - g) ** 2) for g in gammas], color=GREY, ls="--", lw=1,
            label="bound (13.4), two-state example")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"effective horizon $1/(1-\gamma)$"); ax.set_ylabel(r"$\max_\pi\Vert\hat V^\pi - V^\pi\Vert_\infty$")
    ax.set_title("Value error vs horizon (dashed: bound)")
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]
    for k, g in enumerate([0.5, 0.9, 0.99]):
        d = res[g]
        ax.scatter(2 * d["err_two"], d["loss"], s=12, color=C[k], alpha=0.7, marker="os^"[k],
                   label=f"$\\gamma$={g}")
    lo = min(2 * res[0.5]["err_two"].min(), 1e-3)
    lim = max(2 * res[0.99]["err_two"].max(), 1e-3)
    ax.plot([lo, lim], [lo, lim], color=GREY, ls="--", lw=1, label="loss = $2\\epsilon_V$ (bound 13.5)")
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$2\epsilon_V = 2\max_{\pi\in\{\pi_\ast,\hat\pi\}}\Vert\hat V^\pi - V^\pi\Vert_\infty$")
    ax.set_ylabel(r"$\max_s [V^{\pi_\ast}(s) - V^{\hat\pi}(s)]$")
    ax.set_title("Loss from acting optimally in the model")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "simulation_lemma.png")
    fig.savefig(out); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
