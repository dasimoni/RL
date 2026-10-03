"""Why parameter-space step sizes are the wrong knob, and what the natural gradient fixes.

Chapter 11, Sections 1 and 5 (exact computations, numpy only).

  (a) A Bernoulli policy p = sigmoid(theta): the same parameter step Delta = 0.5 changes the
      action distribution by wildly different amounts depending on theta; the KL is
      1/2 F(theta) Delta^2 to second order, with Fisher information F = p(1-p)  (Eq. 11.17).
  (b) Reparameterisation: a 3-armed softmax bandit trained by exact gradient ascent with
      logits theta, or with theta = c * phi (c = (1, 4, 0.25)).  Vanilla gradient ascent follows
      different paths through the simplex; natural gradient ascent follows the same path
      (Section 5.3).  Also prints the worked example of Section 5.4.
  (c) A 6-state chain MDP started from a policy that prefers the small nearby reward: exact
      vanilla policy gradient sits on a long plateau because its update at state s is scaled
      by d^pi(s) pi(a|s); the natural policy gradient (Eq. 11.22, tabular form
      theta(s,a) += eta A(s,a)/(1-gamma)) is not.  Computing the vanilla gradient under a uniform
      start distribution (full support) removes the trap but can leave a plateau.

Run:  python code/ch11_trust_regions_and_ppo/natural_gradient_geometry.py [--quick]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def kl_bern(p, q):
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def softmax(x):
    z = np.exp(x - x.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


# ---------------------------------------------------------------------------------------------
# (b) softmax bandit: exact gradient and Fisher in logit space
# ---------------------------------------------------------------------------------------------
def bandit_grad_fisher(theta, r):
    pi = softmax(theta)
    J = pi @ r
    g = pi * (r - J)                       # dJ/dtheta_a = pi_a (r_a - J)
    F = np.diag(pi) - np.outer(pi, pi)     # E[psi psi^T] with psi = e_a - pi
    return pi, J, g, F


def bandit_path(r, c, natural, eta, steps, theta0):
    """Gradient ascent on phi where theta = c * phi; returns the sequence of policies."""
    phi = theta0 / c
    out = []
    for _ in range(steps):
        theta = c * phi
        pi, J, g, F = bandit_grad_fisher(theta, r)
        out.append(pi)
        g_phi = c * g                      # chain rule: dJ/dphi = C^T dJ/dtheta
        if natural:
            F_phi = np.diag(c) @ F @ np.diag(c)
            step = np.linalg.lstsq(F_phi, g_phi, rcond=None)[0]   # min-norm F^+ g (F is singular)
        else:
            step = g_phi
        phi = phi + eta * step
    return np.array(out)


# ---------------------------------------------------------------------------------------------
# (c) chain MDP, exact vanilla PG vs exact NPG with tabular softmax
# ---------------------------------------------------------------------------------------------
def chain_mdp(n=6, r_small=0.1, r_big=1.0):
    """States 0..n-1, start in 0.  LEFT moves to max(s-1, 0); RIGHT to min(s+1, n-1).
    LEFT in state 0 pays r_small; RIGHT in state n-1 pays r_big.  Continuing, discounted."""
    P = np.zeros((n, 2, n))
    r = np.zeros((n, 2))
    for s in range(n):
        P[s, 0, max(s - 1, 0)] = 1.0
        P[s, 1, min(s + 1, n - 1)] = 1.0
    r[0, 0], r[n - 1, 1] = r_small, r_big
    d0 = np.zeros(n)
    d0[0] = 1.0
    return P, r, d0


def evaluate(P, r, d0, pi, gamma):
    n = P.shape[0]
    P_pi = np.einsum("sa,sat->st", pi, P)
    v = np.linalg.solve(np.eye(n) - gamma * P_pi, (pi * r).sum(1))
    q = r + gamma * P @ v
    d = (1 - gamma) * np.linalg.solve((np.eye(n) - gamma * P_pi).T, d0)
    return d0 @ v, q - v[:, None], d


def chain_run(natural, eta, iters, gamma=0.9, n=6, grad_start="d0"):
    """Exact updates.  J is always measured from the true start state 0 (d0).  For vanilla PG,
    grad_start="uniform" computes the gradient of J under a uniform start distribution mu
    instead (as in the global-convergence analyses that require full support)."""
    P, r, d0 = chain_mdp(n)
    mu = d0 if grad_start == "d0" else np.ones(n) / n
    theta = np.zeros((n, 2))
    theta[:, 0] = np.log(4.0)          # pi(LEFT|s) = 0.8 everywhere at the start
    Js = []
    for _ in range(iters):
        pi = softmax(theta)
        J, A, _ = evaluate(P, r, d0, pi, gamma)
        Js.append(J)
        if natural:
            theta = theta + eta * A / (1 - gamma)                         # Eq. 11.22 (tabular)
        else:
            d = evaluate(P, r, mu, pi, gamma)[2]                          # d^pi under the start dist.
            theta = theta + eta * d[:, None] * pi * A / (1 - gamma)       # Eq. 11.23 (policy gradient theorem)
    return np.array(Js), (gamma ** (n - 1)) * 1.0 / (1 - gamma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    print("Deterministic, exact computations (no random numbers).")

    # ---- (a) Bernoulli --------------------------------------------------------------------
    delta = 0.5
    print("\n(a) Bernoulli policy p = sigmoid(theta), step Delta = 0.5:")
    for th in [0.0, 2.0, 4.0]:
        p, q = sigmoid(th), sigmoid(th + delta)
        F = p * (1 - p)
        print(f"    theta = {th:.0f}: p = {p:.5f} -> {q:.5f};  KL = {kl_bern(p, q):.5f};  "
              f"1/2 F Delta^2 = {0.5 * F * delta**2:.5f};  step with KL = 0.01: {np.sqrt(2 * 0.01 / F):.3f}")

    # ---- (b) bandit worked example and reparameterisation ---------------------------------
    r = np.array([1.0, 0.8, 0.0])
    pi_ex = np.array([0.5, 0.3, 0.2])
    _, J, g, F = bandit_grad_fisher(np.log(pi_ex), r)
    ng = np.linalg.lstsq(F, g, rcond=None)[0]
    print(f"\n(b) Worked example (Section 5.4): r = {r}, pi = {pi_ex}")
    print(f"    J = {J:.3f};  vanilla gradient g = {np.round(g, 4)};  natural gradient F^+ g = {np.round(ng, 4)}"
          f"  (= r - mean(r) = {np.round(r - r.mean(), 4)})")
    p1 = softmax(np.log(pi_ex) + 1.0 * ng)
    print(f"    one natural step with eta = 1: pi' = pi * exp(r) / Z = {np.round(p1, 4)};  "
          f"KL(pi || pi') = {np.sum(pi_ex * np.log(pi_ex / p1)):.4f}")
    c = np.array([1.0, 4.0, 0.25])
    theta0 = np.zeros(3)
    steps = 40 if args.quick else 400
    paths = {}
    for nat, eta in [(False, 0.5), (True, 0.02)]:
        for cc, nm in [(np.ones(3), "theta"), (c, "phi")]:
            paths[(nat, nm)] = bandit_path(r, cc, nat, eta, steps, theta0)
    dv = np.abs(paths[(False, "theta")] - paths[(False, "phi")]).max()
    dn = np.abs(paths[(True, "theta")] - paths[(True, "phi")]).max()
    print(f"    reparameterisation theta = c * phi, c = {c}:")
    print(f"    max |pi_theta-path - pi_phi-path| over {steps} steps: vanilla {dv:.3f}, natural {dn:.1e}")
    for (nat, nm), path in paths.items():
        print(f"    {'natural' if nat else 'vanilla'} ascent in {nm:5s}: final pi = {np.round(path[-1], 3)}")
    if not args.quick:
        # How long does vanilla ascent in phi stay near the second-best arm?  The gradient on
        # phi_1 is c_1 pi_1 (r_1 - J) > 0 at every step (J < 1), so it is a plateau, not a
        # spurious maximum -- but a very long one.
        long_steps = 100_000
        far = bandit_path(r, c, False, 0.5, long_steps, theta0)[-1]
        g_phi1 = c[0] * far[0] * (r[0] - far @ r)
        print(f"    vanilla ascent in phi continued to {long_steps} steps: pi = {np.round(far, 5)}, "
              f"dJ/dphi_1 = {g_phi1:.1e} (> 0, but tiny)")

    # ---- (c) chain plateau ------------------------------------------------------------------
    iters = 300 if args.quick else 20000
    print(f"\n(c) 6-state chain, gamma = 0.9, start pi(LEFT) = 0.8; exact updates, {iters} iterations")
    chain = {}
    for nat, eta, gs in [(False, 1.0, "d0"), (False, 10.0, "d0"), (False, 1.0, "uniform"),
                         (False, 30.0, "uniform"), (True, 0.1, "d0"), (True, 1.0, "d0")]:
        Js, Jstar = chain_run(nat, eta, iters, grad_start=gs)
        name = f"{'NPG' if nat else 'vanilla PG'}, eta = {eta:g}" + (", mu uniform" if gs == "uniform" else "")
        chain[name] = Js
        hit = np.argmax(Js > 0.9 * Jstar) if (Js > 0.9 * Jstar).any() else None
        print(f"    {name:34s}: J(0) = {Js[0]:.3f}, J(final) = {Js[-1]:.3f} (J* = {Jstar:.3f});  "
              f"iterations to reach 90% of J*: {hit if hit is not None else '> ' + str(iters)}")

    if not args.quick:
        from plot_style import C, GREY, setup
        plt = setup()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.1))
        th = np.linspace(-6, 6, 400)
        p, q = sigmoid(th), sigmoid(th + delta)
        ax[0].semilogy(th, kl_bern(p, q), color=C[0], label="KL(p_theta || p_theta+0.5)")
        ax[0].semilogy(th, 0.5 * p * (1 - p) * delta**2, color=C[1], ls="--", label="1/2 F(theta) Delta^2")
        ax[0].set_xlabel("theta (logit)")
        ax[0].set_ylabel("nats")
        ax[0].set_title("(a) same parameter step, different KL")
        ax[0].legend(fontsize=8)

        # simplex coordinates: x = pi_2 - pi_3 tilt, use (pi_1, pi_2) directly
        for (nat, nm), path in paths.items():
            col = C[1] if nat else C[0]
            if nm == "theta":
                ax[1].plot(path[:, 0], path[:, 1], color=col, lw=2.2,
                           label=f"{'natural (eta 0.02)' if nat else 'vanilla (eta 0.5)'}, logits theta")
            else:
                ax[1].plot(path[:, 0], path[:, 1], color=col, ls="--", lw=1.0, marker="o", ms=3.5, mfc="none",
                           markevery=[0, 1, 2, 3, 5, 8, 12, 20] + list(range(30, len(path), 15)),
                           label=f"{'natural (eta 0.02)' if nat else 'vanilla (eta 0.5)'}, theta = c * phi")
        ax[1].plot([1, 0], [0, 1], color=GREY, lw=0.8)
        ax[1].set_xlabel("pi(arm 1)  (r = 1)")
        ax[1].set_ylabel("pi(arm 2)  (r = 0.8)")
        ax[1].set_xlim(0, 1.02)
        ax[1].set_ylim(0, 1.02)
        ax[1].set_title("(b) bandit: paths under two parameterisations")
        ax[1].legend(fontsize=7.5, loc="upper right")

        styles = [":", "--", "-.", (0, (5, 1, 1, 1)), "--", "-"]
        cols = [C[0], C[0], C[2], C[2], C[1], C[1]]
        _, Jstar = chain_run(True, 1.0, 1)
        def short(name):    # compact legend labels
            return name.replace("vanilla PG", "vanilla").replace(" = ", " ").replace(", mu uniform", ", uniform mu")
        for (name, Js), ls, col in zip(chain.items(), styles, cols):
            it = np.arange(1, len(Js) + 1)
            if name == "vanilla PG, eta = 10":
                # identical to the eta = 1 curve after a few iterations (both stuck at J = 1):
                # draw it with sparse markers so it can be seen on top of the other one
                mk = np.unique(np.logspace(0, np.log10(len(Js)), 12).astype(int)) - 1
                ax[2].plot(it, Js, ls="none", color=col, marker="s", ms=4, mfc="none", markevery=list(mk),
                           label=short(name) + " (markers)")
            else:
                ax[2].plot(it, Js, ls=ls, color=col, label=short(name))
        ax[2].axhline(Jstar, color=GREY, lw=0.8)
        ax[2].set_xscale("log")
        ax[2].set_xlabel("iteration (log scale)")
        ax[2].set_ylabel("J(pi)")
        ax[2].set_title("(c) 6-state chain: exact vanilla PG vs NPG")
        ax[2].legend(fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.45))
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "natural_gradient_geometry.png"))
        plt.close(fig)
        print("wrote figures/natural_gradient_geometry.png")
    print(f"total wall time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
