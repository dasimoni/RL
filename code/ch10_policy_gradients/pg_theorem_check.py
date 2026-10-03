"""Numerical verification of the policy gradient theorem and of the REINFORCE family of estimators.

Chapter 10, Sections 4-6, 12 and 13.  Everything is checked against EXACT quantities computed
by linear algebra on small MDPs (see tabular_pg.py):

Part A  The hand-worked example of Section 4.4: the short corridor (S&B Example 13.1) at
        pi(right) = 1/2.  Expected visit counts, q-values and the theorem's gradient
        (should be 6, 4, 2 visits and dJ/dtheta = 2).
Part B  A random 5-state, 3-action episodic MDP (gamma = 0.9, tabular softmax policy):
        theorem gradient vs finite differences, then 200,000 sampled episodes to compare
        per-episode gradient estimators -- total return, reward-to-go, + baseline v(s),
        + "optimal" baseline b*(s), true advantage, TD error with the true v -- for bias and
        variance; and the biased "drop gamma^t" estimator against its exact expectation.
Part C  Compatible function approximation (Sutton et al., 2000) with a 4-feature linear
        softmax policy: the compatible critic gives the exact gradient, a non-compatible
        critic with as many features does not, and w = natural gradient / sum(eta).
Part D  Tabular softmax: the compatible critic recovers the advantage function exactly.

Outputs (full mode): figures/pg_estimator_variance.png, figures/pg_mc_convergence.png

Run:  python code/ch10_policy_gradients/pg_theorem_check.py [--quick]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from tabular_pg import (LinearSoftmax, corridor_policy, exact_gradient, fd_gradient,
                        random_mdp, short_corridor)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# --------------------------------------------------------------------------------------
# Part A: the hand-worked example
# --------------------------------------------------------------------------------------
def part_a():
    print("\n=== Part A: short corridor at pi(right) = 1/2 (hand-worked example, Section 4.4) ===")
    mdp, pol = short_corridor(gamma=1.0), corridor_policy()
    theta = np.array([0.0])                              # sigmoid(0) = 1/2
    pi = pol.probs(theta)
    v, q = mdp.evaluate(pi)
    eta = mdp.visits(pi)
    grad_pi = pi[..., None] * pol.score(theta)
    per_state = np.einsum("sa,sad->sd", q, grad_pi)[:, 0]
    print("state | eta(s) | v(s)   | q(s,R)  q(s,L) | sum_a q grad pi")
    for s in range(3):
        print(f"  {s}   | {eta[s]:6.3f} | {v[s]:6.2f} | {q[s,0]:6.2f}  {q[s,1]:6.2f} | {per_state[s]:+.3f}")
    g_thm = exact_gradient(mdp, pol, theta)
    g_fd = fd_gradient(mdp, pol, theta)
    print(f"J = {mdp.J(pi):.4f};  sum eta = {eta.sum():.4f} (= expected episode length)")
    print(f"dJ/dtheta: theorem {g_thm[0]:.6f}  finite differences {g_fd[0]:.6f}")
    return g_thm[0]


# --------------------------------------------------------------------------------------
# Part B: estimators on a random MDP
# --------------------------------------------------------------------------------------
def per_episode_estimates(mdp, pol, theta, M, rng, chunk=25_000):
    """Return a dict name -> (M, d) array of single-episode gradient estimates.

    Every estimator has the form  g = sum_t w_t * psi(S_t, A_t)  (Sections 5-7) with weights
        total     : w_t = G_0                               (trajectory likelihood, Eq. 10.11)
        rtg       : w_t = gamma^t G_t                       (reward-to-go, Eq. 10.13)
        rtg-v     : w_t = gamma^t (G_t - v(S_t))            (baseline = true v)
        rtg-b*    : w_t = gamma^t (G_t - b*(S_t))           (variance-optimal state baseline, Sec. 6.3)
        adv       : w_t = gamma^t A(S_t, A_t)               (true advantage)
        td        : w_t = gamma^t delta_t,  delta_t = R + gamma v(S') - v(S)   (Sec. 7.2)
        nogamma   : w_t = G_t - v(S_t)                      (the usual "drop gamma^t", Sec. 12)
    """
    pi = pol.probs(theta)
    v, q = mdp.evaluate(pi)
    A = q - v[:, None]
    psi = pol.score(theta)                               # (S, A, d)
    sq = (psi ** 2).sum(axis=2)                          # ||psi(s,a)||^2
    b_star = (pi * sq * q).sum(1) / (pi * sq).sum(1)     # Eq. (10.18)
    g = mdp.gamma
    names = ["total", "rtg", "rtg-v", "rtg-b*", "adv", "td", "nogamma"]
    out = {k: [] for k in names}
    done = 0
    while done < M:
        m = min(chunk, M - done)
        S, Act, R, alive = mdp.sample_episodes(pi, m, rng)
        T = S.shape[0]
        disc = g ** np.arange(T)[:, None]                # gamma^t
        # reward-to-go G_t = R_{t+1} + gamma G_{t+1}, computed backwards (zeros after the end)
        G = np.zeros_like(R)
        run = np.zeros(m)
        for t in range(T - 1, -1, -1):
            run = R[t] + g * run
            G[t] = run
        v_next = np.zeros_like(R)
        v_next[:-1] = np.where(alive[1:], v[S[1:]], 0.0)   # v(terminal) = 0
        delta = R + g * v_next - v[S]
        weights = {
            "total": np.broadcast_to(G[0], R.shape),
            "rtg": disc * G,
            "rtg-v": disc * (G - v[S]),
            "rtg-b*": disc * (G - b_star[S]),
            "adv": disc * A[S, Act],
            "td": disc * delta,
            "nogamma": G - v[S],
        }
        Psi = psi[S, Act] * alive[..., None]              # (T, m, d), zero after termination
        for k in names:
            out[k].append(np.einsum("tm,tmd->md", weights[k], Psi))
        done += m
    return {k: np.concatenate(vs) for k, vs in out.items()}, b_star, v


def part_b(n_episodes, rng, quick):
    print("\n=== Part B: random MDP (5 states, 3 actions, gamma = 0.9), tabular softmax ===")
    mdp = random_mdp(5, 3, gamma=0.9, rng=rng)
    pol = LinearSoftmax.tabular(5, 3)
    theta = 0.5 * rng.standard_normal(pol.d)
    pi = pol.probs(theta)
    g_true = exact_gradient(mdp, pol, theta)
    g_fd = fd_gradient(mdp, pol, theta)
    eta = mdp.visits(pi)
    print(f"J(theta) = {mdp.J(pi):.4f};  sum_s eta_gamma(s) = {eta.sum():.4f}; "
          f"expected episode length = {mdp.visits(pi, gamma=1.0).sum():.3f}")
    print(f"theorem vs finite differences: max |diff| = {np.abs(g_true - g_fd).max():.2e}  "
          f"(||grad J|| = {np.linalg.norm(g_true):.4f})")

    g_nogamma = exact_gradient(mdp, pol, theta, visit_gamma=1.0)
    cos = g_true @ g_nogamma / (np.linalg.norm(g_true) * np.linalg.norm(g_nogamma))
    print(f"exact E[drop-gamma^t estimator]: ||.|| = {np.linalg.norm(g_nogamma):.4f}, "
          f"cosine with grad J = {cos:.4f}, ||diff|| = {np.linalg.norm(g_nogamma - g_true):.4f}")

    t0 = time.time()
    est, b_star, v = per_episode_estimates(mdp, pol, theta, n_episodes, rng)
    print(f"simulated {n_episodes} episodes in {time.time() - t0:.1f}s")
    print("state baselines: v(s) =", np.round(v, 3), "  b*(s) =", np.round(b_star, 3))

    targets = {k: g_true for k in est}
    targets["nogamma"] = g_nogamma
    print(f"\n{'estimator':>9} | {'||mean - grad J||':>17} | {'max |z|':>7} | {'tr Cov':>9} | "
          f"{'tr Cov / ||grad J||^2':>21}")
    stats = {}
    for k, gs in est.items():
        mean = gs.mean(0)
        se = gs.std(0, ddof=1) / np.sqrt(len(gs))
        z = np.abs(mean - targets[k]) / se               # |z| < ~3.5 everywhere => no detectable bias
        trcov = gs.var(0, ddof=1).sum()
        bias_vs_true = np.linalg.norm(mean - g_true)
        stats[k] = dict(trcov=trcov, bias=bias_vs_true, zmax=z.max())
        print(f"{k:>9} | {bias_vs_true:17.4f} | {z.max():7.2f} | {trcov:9.3f} | "
              f"{trcov / (g_true @ g_true):21.1f}")
    print("(max |z| is measured against the estimator's own exact expectation: grad J for all "
          "but 'nogamma',\n which is compared with its exact (biased) expectation)")
    return mdp, pol, theta, g_true, est, stats


# --------------------------------------------------------------------------------------
# Part C/D: compatible function approximation
# --------------------------------------------------------------------------------------
def compatible_check(mdp, pol, theta, label):
    pi = pol.probs(theta)
    v, q = mdp.evaluate(pi)
    eta = mdp.visits(pi)
    d = eta / eta.sum()
    psi = pol.score(theta)
    weight = d[:, None] * pi                              # d(s) pi(a|s): the regression weights
    # Fisher matrix F = E_{s~d, a~pi}[psi psi^T];  w* solves F w = E[psi q]   (Eq. 10.33)
    F = np.einsum("sa,sai,saj->ij", weight, psi, psi)
    b = np.einsum("sa,sai,sa->i", weight, psi, q)
    w = np.linalg.lstsq(F, b, rcond=None)[0]
    f_w = psi @ w                                        # compatible approximator f_w(s,a)
    g_true = exact_gradient(mdp, pol, theta)
    g_compat = eta.sum() * np.einsum("sa,sad,sa->d", weight, psi, f_w)
    print(f"[{label}] ||g_compatible - grad J|| = {np.linalg.norm(g_compat - g_true):.2e}  "
          f"(||grad J|| = {np.linalg.norm(g_true):.4f})")
    print(f"[{label}] max_s |sum_a pi f_w| = {np.abs((pi * f_w).sum(1)).max():.2e}  "
          "(f_w has zero mean under pi: it approximates the advantage, not q)")
    nat = np.linalg.lstsq(F, g_true, rcond=None)[0]      # natural gradient F^{-1} grad J
    print(f"[{label}] ||w - F^+ grad J / sum(eta)|| = {np.linalg.norm(w - nat / eta.sum()):.2e}")
    return w, f_w, q - v[:, None], weight, g_true, eta


def part_c(rng):
    print("\n=== Part C: compatible function approximation, linear softmax with 4 features ===")
    mdp = random_mdp(5, 3, gamma=0.9, rng=rng)
    X = rng.standard_normal((5, 3, 4))
    pol = LinearSoftmax(X)
    theta = 0.5 * rng.standard_normal(4)
    w, f_w, A, weight, g_true, eta = compatible_check(mdp, pol, theta, "compatible")
    err_A = np.sqrt((weight * (f_w - A) ** 2).sum())
    print(f"[compatible] weighted RMS(f_w - A) = {err_A:.3f}  "
          f"(weighted RMS of A itself = {np.sqrt((weight * A ** 2).sum()):.3f}): "
          "a poor advantage estimate, yet an exact gradient")
    # Non-compatible linear critics q_hat = u . phi(s,a) with k random features, fitted by the
    # same weighted least squares (first feature = constant).  With k = 15 = |S||A| the fit is
    # exact, hence so is the gradient.
    pi = pol.probs(theta)
    _, q = mdp.evaluate(pi)
    psi = pol.score(theta)
    rms_q = np.sqrt((weight * (q - (weight * q).sum()) ** 2).sum())
    cos_out = {}
    for k in (4, 8, 12, 15):
        phi = rng.standard_normal((5, 3, k))
        phi[..., 0] = 1.0                                # a bias feature, so the mean of q is representable
        G = np.einsum("sa,sai,saj->ij", weight, phi, phi)
        u = np.linalg.solve(G, np.einsum("sa,sai,sa->i", weight, phi, q))
        q_hat = phi @ u
        fit = np.sqrt((weight * (q_hat - q) ** 2).sum())
        g_bad = eta.sum() * np.einsum("sa,sad,sa->d", weight, psi, q_hat)
        cos = g_bad @ g_true / (np.linalg.norm(g_bad) * np.linalg.norm(g_true))
        cos_out[k] = cos
        print(f"[non-compatible, {k:2d} random features] weighted RMS(q_hat - q) = {fit:.3f} "
              f"(spread of q: {rms_q:.3f});  ||g - grad J||/||grad J|| = "
              f"{np.linalg.norm(g_bad - g_true) / np.linalg.norm(g_true):.3f}, "
              f"cosine = {cos:+.3f}")
    return cos_out[4]


def part_d(rng):
    print("\n=== Part D: tabular softmax -- the compatible critic IS the advantage ===")
    mdp = random_mdp(5, 3, gamma=0.9, rng=rng)
    pol = LinearSoftmax.tabular(5, 3)
    theta = rng.standard_normal(pol.d)
    w, f_w, A, *_ = compatible_check(mdp, pol, theta, "tabular")
    print(f"[tabular] max |f_w(s,a) - A_pi(s,a)| = {np.abs(f_w - A).max():.2e}")


# --------------------------------------------------------------------------------------
def make_figures(est, stats, g_true):
    from plot_style import setup, C, GREY
    plt = setup()
    os.makedirs(FIG_DIR, exist_ok=True)
    labels = {"total": "total return $G_0$", "rtg": "reward-to-go $\\gamma^tG_t$",
              "rtg-v": "$\\gamma^t(G_t-v_\\pi(S_t))$", "rtg-b*": "$\\gamma^t(G_t-b^{*}(S_t))$",
              "td": "$\\gamma^t\\delta_t$ (true $v_\\pi$)", "adv": "$\\gamma^tA_\\pi(S_t,A_t)$",
              "nogamma": "$G_t-v_\\pi(S_t)$ (no $\\gamma^t$)"}
    order = ["total", "rtg", "rtg-v", "rtg-b*", "td", "adv", "nogamma"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    vals = [stats[k]["trcov"] for k in order]
    cols = [C[0]] * 6 + [C[7]]
    ax.barh(range(len(order)), vals, color=cols)
    ax.set_yticks(range(len(order)), [labels[k] for k in order])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(1, 300)
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("total variance of a one-episode estimate, tr Cov")
    ax.set_title("Variance of single-episode gradient estimators")
    for i, x in enumerate(vals):
        ax.text(x * 1.05, i, f"{x:.2f}", va="center", fontsize=8)
    ax = axes[1]
    vals = [stats[k]["bias"] for k in order]
    ax.barh(range(len(order)), vals, color=cols)
    ax.set_yticks(range(len(order)), ["" for _ in order])
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Vert$mean of 200k estimates $-\nabla J\Vert$")
    ax.set_title("Bias (only the no-$\\gamma^t$ estimator is biased)")
    for i, x in enumerate(vals):
        ax.text(x + 0.002, i, f"{x:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pg_estimator_variance.png"))
    plt.close(fig)

    # Mean error of an N-episode average vs N (RMS over disjoint batches)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    Ns = np.unique(np.logspace(0, 4, 13).astype(int))
    marks = ["o", "s", "^", "D", "v"]
    for j, k in enumerate(["total", "rtg", "rtg-v", "adv", "nogamma"]):
        gs = est[k]
        errs = []
        for N in Ns:
            nb = len(gs) // N
            means = gs[: nb * N].reshape(nb, N, -1).mean(1)
            errs.append(np.sqrt(((means - g_true) ** 2).sum(1).mean()))
        ax.loglog(Ns, errs, marker=marks[j], ms=4, label=labels[k],
                  color=C[7] if k == "nogamma" else C[j], ls="--" if k == "nogamma" else "-")
    ref = 0.5 * np.sqrt(stats["adv"]["trcov"] / Ns)          # slope reference, drawn below the data
    ax.loglog(Ns, ref, color=GREY, lw=1, ls=":", label=r"slope $1/\sqrt{N}$")
    ax.set_xlabel("episodes averaged, N")
    ax.set_ylabel(r"RMS $\Vert\hat g_N-\nabla J\Vert$")
    ax.set_title("Monte Carlo error of the gradient estimate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pg_mc_convergence.png"))
    plt.close(fig)
    print(f"figures written to {FIG_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figures")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n_episodes = 20_000 if args.quick else 200_000
    print(f"seed={args.seed}  episodes for Part B={n_episodes}  quick={args.quick}")
    t0 = time.time()
    part_a()
    # independent generators per part, so Parts C and D do not depend on n_episodes
    mdp, pol, theta, g_true, est, stats = part_b(n_episodes, np.random.default_rng([args.seed, 1]),
                                                 args.quick)
    cos_bad = part_c(np.random.default_rng([args.seed, 2]))
    part_d(np.random.default_rng([args.seed, 3]))
    if not args.quick:
        make_figures(est, stats, g_true)
    print("\n=== Summary ===")
    print(f"variance reduction, total return -> reward-to-go -> + v baseline -> true advantage: "
          f"{stats['total']['trcov']:.2f} -> {stats['rtg']['trcov']:.2f} -> "
          f"{stats['rtg-v']['trcov']:.2f} -> {stats['adv']['trcov']:.2f}")
    print(f"non-compatible critic gradient cosine: {cos_bad:.3f}")
    print(f"total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
