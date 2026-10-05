"""Exact checks on a small random MDP: the performance difference lemma, the surrogate
objective, the TRPO lower bound, conservative policy iteration, and what the theory's
penalty coefficient would mean in practice.

Chapter 11, Sections 2-3 (Lemma 11.1, Eqs. 11.6-11.15, Theorem 11.3, Corollary 11.4, Proposition 11.5).
Everything here is computed exactly with linear algebra -- no sampling.

  1. Performance difference lemma (Eq. 11.5): J(pi') - J(pi) = 1/(1-gamma) E_{s~d^pi'} E_{a~pi'}[A_pi]
     checked on random policy pairs, to machine precision.
  2. The surrogate L_pi (Eq. 11.6) matches J to first order at pi (value and gradient).
  3. The bound J(pi') >= L_pi(pi') - 4 eps gamma (D_TV^max)^2/(1-gamma)^2 (Theorem 11.3) and its KL
     and average-TV versions hold on thousands of random pairs; how loose they are.
  4. A log-linear policy with 3 SHARED features pi(a|s) ~ exp(theta . x(s,a)), stepped along its
     natural-gradient direction: the surrogate is accurate for small KL and badly optimistic
     for large KL (the true return eventually falls below the start while L still promises a
     gain); the lower bounds certify only microscopic steps.
  5. CPI: tabular mixture policies (1-kappa) pi + kappa pi_greedy, the CPI bound and its optimal kappa*.
  6. Iterating (tabular softmax, exponentiated-advantage steps): maximising the theoretical
     lower bound (monotone but painfully slow) vs. a KL trust region (delta = 0.01, 0.1) vs.
     policy iteration.

Run:  python code/ch11_trust_regions_and_ppo/surrogate_bound.py [--quick]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


class MDP:
    """Finite discounted MDP with P[s, a, s'], r[s, a], initial distribution d0."""

    def __init__(self, n_s, n_a, gamma, rng):
        self.n_s, self.n_a, self.gamma = n_s, n_a, gamma
        self.P = rng.dirichlet(np.ones(n_s) * 0.5, size=(n_s, n_a))
        self.r = rng.uniform(0, 1, size=(n_s, n_a))
        self.d0 = np.ones(n_s) / n_s

    def evaluate(self, pi):
        """v_pi, q_pi, A_pi by solving the Bellman equation exactly."""
        P_pi = np.einsum("sa,sat->st", pi, self.P)
        r_pi = (pi * self.r).sum(1)
        v = np.linalg.solve(np.eye(self.n_s) - self.gamma * P_pi, r_pi)
        q = self.r + self.gamma * self.P @ v
        return v, q, q - v[:, None]

    def visitation(self, pi):
        """Normalised discounted state distribution d^pi = (1-gamma) d0^T (I - gamma P_pi)^-1."""
        P_pi = np.einsum("sa,sat->st", pi, self.P)
        return (1 - self.gamma) * np.linalg.solve((np.eye(self.n_s) - self.gamma * P_pi).T, self.d0)

    def J(self, pi):
        return self.d0 @ self.evaluate(pi)[0]


def surrogate(mdp, pi, pi_new):
    """L_pi(pi_new) = J(pi) + 1/(1-gamma) sum_s d^pi(s) sum_a pi_new(a|s) A_pi(s,a)."""
    v, q, A = mdp.evaluate(pi)
    d = mdp.visitation(pi)
    return mdp.d0 @ v + (d * (pi_new * A).sum(1)).sum() / (1 - mdp.gamma)


def kl_rows(p, q):
    """KL(p(.|s) || q(.|s)) for every s, with the convention 0 log 0 = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * (np.log(p) - np.log(q)), 0.0)
    return terms.sum(1)


def tv_rows(p, q):
    return 0.5 * np.abs(p - q).sum(1)


def softmax(x):
    z = np.exp(x - x.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def random_policy(rng, n_s, n_a, scale=1.0):
    return softmax(rng.normal(0, scale, size=(n_s, n_a)))


def bounds(mdp, pi, pi_new):
    """Return J(pi_new), L_pi(pi_new) and the three lower bounds of Section 3."""
    g = mdp.gamma
    v, q, A = mdp.evaluate(pi)
    d = mdp.visitation(pi)
    L = surrogate(mdp, pi, pi_new)
    eps = np.abs(A).max()                                   # max_{s,a} |A_pi(s,a)|
    a_tv = tv_rows(pi, pi_new).max()                        # D_TV^max = max_s D_TV
    kl_max = kl_rows(pi, pi_new).max()                      # max_s KL(pi || pi_new)
    abar = (pi_new * A).sum(1)                              # E_{a~pi_new} A_pi(s,a)
    eps_new = np.abs(abar).max()
    tv_avg = d @ tv_rows(pi, pi_new)
    return dict(
        J=mdp.J(pi_new), L=L,
        b_tv=L - 4 * eps * g * a_tv**2 / (1 - g) ** 2,          # Theorem 11.3, Eq. 11.11
        b_kl=L - 4 * eps * g * kl_max / (1 - g) ** 2,           # Eq. 11.12 (D_TV^2 <= KL)
        b_avg=L - 2 * g * eps_new * tv_avg / (1 - g) ** 2,      # Corollary 11.4, Eq. 11.13 (Achiam et al.)
        kl_avg=d @ kl_rows(pi, pi_new), kl_max=kl_max)


def exp_path(pi, A, eta):
    """pi_eta(a|s) proportional to pi(a|s) exp(eta A(s,a)) -- the tabular natural-gradient step."""
    with np.errstate(divide="ignore"):
        return softmax(np.log(pi) + eta * A)


def solve_opt(mdp, iters=3000):
    """q_* by value iteration (to machine precision for gamma = 0.9)."""
    q = np.zeros((mdp.n_s, mdp.n_a))
    for _ in range(iters):
        q = mdp.r + mdp.gamma * mdp.P @ q.max(1)
    return q


def greedy_pi(mdp, q):
    p = np.zeros_like(q)
    p[np.arange(mdp.n_s), q.argmax(1)] = 1.0
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    seed, n_s, n_a, gamma, n_feat = 146, 6, 3, 0.9, 3
    rng = np.random.default_rng(seed)
    mdp = MDP(n_s, n_a, gamma, rng)
    X = rng.normal(size=(n_s, n_a, n_feat))          # shared features x(s,a) for experiment 4
    theta0 = 0.5 * rng.normal(size=n_feat)
    rng = np.random.default_rng(0)                   # separate stream for the random pairs
    J_star = mdp.J(greedy_pi(mdp, solve_opt(mdp)))
    print(f"seed={seed}  random MDP: |S|={n_s}, |A|={n_a}, gamma={gamma}, rewards U[0,1], "
          f"P(.|s,a) ~ Dirichlet(0.5), d0 uniform;  J* = {J_star:.4f}")

    # ---- 1. performance difference lemma -----------------------------------------------
    n_pairs = 200 if args.quick else 2000
    err = 0.0
    for _ in range(n_pairs):
        pi, pi2 = random_policy(rng, n_s, n_a, 2.0), random_policy(rng, n_s, n_a, 2.0)
        _, _, A = mdp.evaluate(pi)
        rhs = mdp.visitation(pi2) @ (pi2 * A).sum(1) / (1 - gamma)
        err = max(err, abs(mdp.J(pi2) - mdp.J(pi) - rhs))
    print(f"\n1. Performance difference lemma on {n_pairs} random pairs: max |LHS - RHS| = {err:.1e}")

    # ---- 2. first-order match of L and J (tabular softmax parameters, central differences) ---
    theta = rng.normal(size=(n_s, n_a))
    pi0 = softmax(theta)
    h = 1e-5
    gJ, gL = np.zeros_like(theta), np.zeros_like(theta)
    for i in range(n_s):
        for j in range(n_a):
            e = np.zeros_like(theta)
            e[i, j] = h
            gJ[i, j] = (mdp.J(softmax(theta + e)) - mdp.J(softmax(theta - e))) / (2 * h)
            gL[i, j] = (surrogate(mdp, pi0, softmax(theta + e)) - surrogate(mdp, pi0, softmax(theta - e))) / (2 * h)
    print(f"2. L_pi(pi) - J(pi) = {surrogate(mdp, pi0, pi0) - mdp.J(pi0):.1e};  "
          f"max |grad L - grad J| = {np.abs(gJ - gL).max():.1e} (max |grad J| = {np.abs(gJ).max():.3f})")

    # ---- 3. validity and looseness of the bounds on random nearby pairs ---------------------
    viol = {"b_tv": 0, "b_kl": 0, "b_avg": 0}
    n_b = 300 if args.quick else 3000
    ratios = []
    for _ in range(n_b):
        th = rng.normal(0, 2.0, size=(n_s, n_a))
        pi = softmax(th)
        pi2 = softmax(th + rng.normal(0, rng.choice([0.01, 0.1, 1.0]), size=(n_s, n_a)))
        b = bounds(mdp, pi, pi2)
        for k in viol:
            viol[k] += b[k] > b["J"] + 1e-12
        if abs(b["J"] - b["L"]) > 1e-10:
            ratios.append((b["L"] - b["b_avg"]) / abs(b["J"] - b["L"]))
    print(f"3. Bounds on {n_b} random pairs: violations TV={viol['b_tv']}, KL={viol['b_kl']}, "
          f"avg-TV={viol['b_avg']};  avg-TV penalty / actual |J - L|: median {np.median(ratios):.0f}x")

    # ---- 4. shared-feature log-linear policy along its natural-gradient direction -----------
    pol = lambda th: softmax(X @ th)  # noqa: E731
    pi = pol(theta0)
    _, _, A = mdp.evaluate(pi)
    dpi = mdp.visitation(pi)
    psi = X - (pi[:, :, None] * X).sum(1, keepdims=True)             # grad log pi(a|s)
    g = np.einsum("s,sa,sa,sai->i", dpi, pi, A, psi) / (1 - gamma)   # policy gradient
    F = np.einsum("s,sa,sai,saj->ij", dpi, pi, psi, psi)            # Fisher (Eq. 11.19)
    u = np.linalg.solve(F, g)
    u /= np.sqrt(u @ F @ u)                     # unit Fisher norm: mean KL ~ t^2 / 2 at step t
    J0 = mdp.J(pi)
    ts = np.concatenate([[0.0], np.logspace(-4, np.log10(4.0), 60 if args.quick else 400)])
    path = [bounds(mdp, pi, pol(theta0 + t * u)) for t in ts]
    get = lambda k: np.array([p[k] for p in path])  # noqa: E731
    imp, Limp = get("J") - J0, get("L") - J0
    bkl, bavg, klavg = get("b_kl") - J0, get("b_avg") - J0, get("kl_avg")
    print(f"\n4. Log-linear policy, {n_feat} shared features, along the natural gradient (J = {J0:.4f}):")
    print(f"   C = 4 eps gamma/(1-gamma)^2 = {4 * np.abs(A).max() * gamma / (1 - gamma) ** 2:.1f}")
    for name, b in [("max-KL bound (Eq. 11.12)", bkl), ("avg-TV bound (Cor. 11.4)", bavg)]:
        i = int(np.argmax(b))
        print(f"   {name}: maximised at mean KL {klavg[i]:.1e}, certifies {b[i]:+.5f} (true gain {imp[i]:+.5f})")
    for target in [0.001, 0.01, 0.1, 0.3, 1.0, 2.0, 5.0]:
        i = int(np.argmin(np.abs(klavg - target)))
        print(f"   mean KL {klavg[i]:6.3f}: true gain {imp[i]:+.4f}   surrogate gain {Limp[i]:+.4f}")
    i_best = int(np.argmax(imp))
    print(f"   best true gain {imp[i_best]:+.4f} at mean KL {klavg[i_best]:.3f}; surrogate keeps rising to "
          f"{Limp.max():+.4f} at mean KL {klavg[int(np.argmax(Limp))]:.2f}")

    # ---- 5. conservative policy iteration (tabular) -------------------------------------------
    pi = random_policy(np.random.default_rng(7), n_s, n_a, 1.0)
    J0t = mdp.J(pi)
    _, q, A = mdp.evaluate(pi)
    pi_g = greedy_pi(mdp, q)
    d = mdp.visitation(pi)
    adv_pol = d @ (pi_g * A).sum(1)                          # policy advantage  A_pi(pi')
    eps_c = np.abs((pi_g * A).sum(1)).max()
    kappas = np.linspace(0, 1, 101)                          # mixture weights kappa (Section 3.4)
    cpi_true = np.array([mdp.J((1 - a) * pi + a * pi_g) for a in kappas]) - J0t
    cpi_L = kappas * adv_pol / (1 - gamma)
    cpi_bound = cpi_L - 2 * gamma * eps_c * kappas**2 / (1 - gamma) ** 2
    a_star = min(1.0, adv_pol * (1 - gamma) / (4 * gamma * eps_c))
    kz = np.linspace(0, 0.06, 121)                           # fine grid for the zoomed inset
    cpi_true_z = np.array([mdp.J((1 - a) * pi + a * pi_g) for a in kz]) - J0t
    cpi_L_z = kz * adv_pol / (1 - gamma)
    cpi_bound_z = cpi_L_z - 2 * gamma * eps_c * kz**2 / (1 - gamma) ** 2
    print(f"\n5. CPI from a random tabular policy (J = {J0t:.4f}): policy advantage A_pi(pi_greedy) = {adv_pol:.4f}, "
          f"eps = {eps_c:.4f};\n   kappa* = {a_star:.4f} certifies gain {adv_pol**2 / (8 * gamma * eps_c):.5f} "
          f"(true gain at kappa*: {mdp.J((1 - a_star) * pi + a_star * pi_g) - J0t:.5f}; "
          f"kappa = 1, i.e. policy iteration: {cpi_true[-1]:.5f})")

    # ---- 6. iterate: MM on the theoretical bound vs trust region vs policy iteration ---------
    n_iter = 30 if args.quick else 400
    curves = {}
    grid = np.concatenate([[0.0], np.logspace(-4, 2, 200)])
    for name in ["MM on the Eq. 11.12 bound", "trust region, mean KL <= 0.01",
                 "trust region, mean KL <= 0.1", "policy iteration"]:
        p = pi.copy()
        Js = [mdp.J(p)]
        for _ in range(n_iter):
            _, q, A = mdp.evaluate(p)
            if name == "policy iteration":
                p = greedy_pi(mdp, q)
            else:
                cands = [exp_path(p, A, e) for e in grid]
                if name.startswith("MM"):
                    vals = [bounds(mdp, p, c)["b_kl"] for c in cands]
                    p = cands[int(np.argmax(vals))]
                else:
                    delta = 0.01 if "0.01" in name else 0.1
                    dist = mdp.visitation(p)
                    ok = [k for k, c in enumerate(cands) if dist @ kl_rows(p, c) <= delta]
                    p = cands[ok[-1]]
            Js.append(mdp.J(p))
        curves[name] = np.array(Js)
        diffs = np.diff(curves[name])
        print(f"6. {name:30s}: J after 1/10/{n_iter} iters = {Js[1]:.4f}/{Js[min(10, n_iter)]:.4f}/{Js[-1]:.4f}; "
              f"monotone: {bool((diffs >= -1e-12).all())}")
    print(f"   optimum J* = {J_star:.4f}")

    if not args.quick:
        from plot_style import C, GREY, setup
        plt = setup()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.4))
        a = ax[0, 0]
        a.plot(klavg[1:], imp[1:], color=C[0], label="true gain J(pi') - J(pi)")
        a.plot(klavg[1:], Limp[1:], color=C[1], ls="--", label="surrogate gain L_pi(pi') - J(pi)")
        a.set_xscale("log")
        a.set_xlim(1e-3, klavg.max())
        a.axhline(0, color=GREY, lw=0.8)
        a.axvline(0.01, color=GREY, lw=0.8, ls=":")
        a.set_xlabel("mean KL(pi || pi') under d^pi (log scale)")
        a.set_title("(a) shared features: the surrogate is optimistic far away")
        a.legend(fontsize=8, loc="upper left")

        a = ax[0, 1]
        a.plot(klavg[1:], imp[1:], color=C[0], label="true gain")
        a.plot(klavg[1:], Limp[1:], color=C[1], ls="--", label="surrogate gain")
        a.plot(klavg[1:], bavg[1:], color=C[2], ls="-.", label="lower bound, avg TV (Cor. 11.4)")
        a.plot(klavg[1:], bkl[1:], color=C[3], ls=":", label="lower bound, max KL (Eq. 11.12)")
        for b, c in [(bavg, C[2]), (bkl, C[3])]:
            i = int(np.argmax(b))
            a.plot([klavg[i]], [b[i]], "o", color=c, ms=5)
        a.set_xscale("log")
        a.set_xlim(1e-8, 1e-2)
        a.set_ylim(-0.02, 0.06)
        a.axhline(0, color=GREY, lw=0.8)
        a.set_xlabel("mean KL(pi || pi') under d^pi (log scale)")
        a.set_title("(b) zoom: what the lower bounds certify (dots = maxima)")
        a.legend(fontsize=8, loc="upper left")

        a = ax[1, 0]
        a.plot(kappas, cpi_true, color=C[0], label="true gain")
        a.plot(kappas, cpi_L, color=C[1], ls="--", label="surrogate (linear in kappa)")
        a.plot(kappas, cpi_bound, color=C[2], ls="-.", label="CPI lower bound")
        a.axvline(a_star, color=GREY, lw=0.8, ls=":")
        a.set_ylim(-0.1, max(cpi_L.max(), cpi_true.max()) * 1.1)
        a.axhline(0, color=GREY, lw=0.8)
        a.set_xlabel("mixing weight kappa")
        a.set_title(f"(c) CPI (tabular): (1-kappa) pi + kappa greedy(q_pi); kappa* = {a_star:.3f}")
        a.legend(fontsize=8, loc="upper left")
        # zoomed inset: the CPI bound is positive only for tiny kappa
        ins = a.inset_axes([0.56, 0.08, 0.41, 0.42])
        ins.plot(kz, cpi_true_z, color=C[0])
        ins.plot(kz, cpi_L_z, color=C[1], ls="--")
        ins.plot(kz, cpi_bound_z, color=C[2], ls="-.", lw=1.8)
        ins.axvline(a_star, color=GREY, lw=0.8, ls=":")
        ins.plot([a_star], [adv_pol**2 / (8 * gamma * eps_c)], "o", color=C[2], ms=4)
        ins.axhline(0, color=GREY, lw=0.8)
        ins.set_xlim(0, 0.06)
        ins.set_ylim(-0.03, 0.16)
        ins.set_title("zoom: kappa in [0, 0.06]", fontsize=8)
        ins.tick_params(labelsize=7)

        a = ax[1, 1]
        styles = ["-.", "--", "-", ":"]
        cols = [C[3], C[2], C[0], C[1]]
        for (name, Js), ls, c in zip(curves.items(), styles, cols):
            gap = np.maximum(J_star - Js, 1e-16)
            a.plot(np.arange(1, len(Js) + 1), gap, ls=ls, color=c, label=name)
        a.set_yscale("log")
        a.set_xscale("log")
        a.set_xlabel("iteration k + 1 (log scale)")
        a.set_ylabel("J* - J(pi_k)")
        a.set_title("(d) iterating each rule (tabular softmax)")
        a.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "surrogate_bound.png"))
        plt.close(fig)
        print("wrote figures/surrogate_bound.png")
    print(f"total wall time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
