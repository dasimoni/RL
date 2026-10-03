"""The KL-regularized objective on a small discrete bandit, solved four ways.

Chapter 18, Sections 3, 5, 6 and 8.  One prompt, K possible "responses" (arms), a fixed
reference policy pi_ref and a true reward r.  Everything here is exact or nearly so, which
makes it the right place to check the chapter's central formulas numerically:

  1. Closed form (eq. 18.9):  pi*_beta(y) = pi_ref(y) exp(r(y)/beta) / Z_beta,
     with optimal value  J*_beta = beta * log Z_beta  (eq. 18.11).
     Checked against (a) a generic constrained optimizer (SLSQP on the simplex),
     (b) exact gradient ascent on softmax logits, and (c) *sampled* REINFORCE with the KL
     folded into the reward and a leave-one-out baseline (Section 4.3, Algorithm 18.3).
  2. DPO (Section 5): with infinitely many Bradley-Terry preferences, the DPO minimizer is
     the same pi*_beta.  With *deterministic* preferences the DPO loss has no finite
     minimizer and the policy collapses onto the best arm whatever beta is; IPO keeps a
     beta-dependent (tau-dependent) solution pi_ref * exp(p(y > pi_ref)/tau) (Sections 5.4, 6).
  3. Best-of-n (Section 8): the exact distribution of the best of n samples, its exact KL
     to pi_ref versus the folklore formula log n - (n-1)/n (an upper bound), and its
     reward-KL trade-off versus the optimal frontier traced by pi*_beta.

Run from the repository root:
    python code/ch18_rl_for_language_models/kl_bandit.py           # full (figures), ~15 s
    python code/ch18_rl_for_language_models/kl_bandit.py --quick   # smoke test, no figures
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import time  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit, log_softmax, logsumexp, softmax  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
SEED = 0


# --------------------------------------------------------------------------------------
# The bandit and the closed form
# --------------------------------------------------------------------------------------
def make_bandit(K: int, rng: np.random.Generator):
    """A reference policy that is *not* uniform and a reward that disagrees with it a bit."""
    pi_ref = softmax(rng.normal(0.0, 1.0, K))
    r = rng.normal(0.0, 1.0, K)
    return pi_ref, r


def closed_form(pi_ref, r, beta):
    """pi*_beta = pi_ref exp(r/beta) / Z (eq. 18.9), computed in log space for stability."""
    logits = np.log(pi_ref) + r / beta
    log_Z = logsumexp(logits) - 0.0  # sum_y pi_ref(y) exp(r(y)/beta), in log space
    pi = np.exp(logits - log_Z)
    return pi, beta * log_Z  # policy and optimal value J* = beta log Z (eq. 18.11)


def kl(p, q):
    m = p > 0
    return float(np.sum(p[m] * (np.log(p[m]) - np.log(q[m]))))


def objective(pi, pi_ref, r, beta):
    """J_beta(pi) = E_pi[r] - beta KL(pi || pi_ref)   (eq. 18.4 for a single prompt)."""
    return float(pi @ r) - beta * kl(pi, pi_ref)


# --------------------------------------------------------------------------------------
# Three independent solvers
# --------------------------------------------------------------------------------------
def solve_slsqp(pi_ref, r, beta):
    """Generic constrained optimizer on the simplex (no knowledge of the closed form)."""
    K = len(r)
    eps = 1e-12

    def negJ(p):
        p = np.clip(p, eps, 1.0)
        return -(p @ r - beta * np.sum(p * (np.log(p) - np.log(pi_ref))))

    cons = [{"type": "eq", "fun": lambda p: np.sum(p) - 1.0}]
    res = minimize(negJ, np.ones(K) / K, method="SLSQP", bounds=[(eps, 1.0)] * K,
                   constraints=cons, options={"ftol": 1e-14, "maxiter": 1000})
    p = np.clip(res.x, 0, None)
    return p / p.sum()


def solve_exact_gradient(pi_ref, r, beta, steps=20000, lr=0.5):
    """Gradient ascent on softmax logits theta with the *exact* gradient of J_beta.

    dJ/dtheta_a = pi_a * (g_a - sum_b pi_b g_b),  g = r - beta (log pi - log pi_ref) - beta,
    and the constant -beta drops out of the centred expression.  This is the bandit form of
    the policy gradient with the KL folded into the reward (Section 4.1, eq. 18.13).
    """
    theta = np.log(pi_ref).copy()  # start at the reference policy, as RLHF does
    for _ in range(steps):
        pi = softmax(theta)
        g = r - beta * (np.log(pi) - np.log(pi_ref))
        theta += lr * pi * (g - pi @ g)
    return softmax(theta)


def solve_reinforce(pi_ref, r, beta, rng, iters=60000, k=4, lr0=0.5):
    """Sampled REINFORCE with KL-in-reward and a leave-one-out (RLOO) baseline.

    Each iteration draws k responses y_1..y_k ~ pi_theta, forms the KL-shaped reward
    R_i = r(y_i) - beta * log(pi_theta(y_i)/pi_ref(y_i)) (no gradient through R_i),
    the baseline b_i = mean_{j != i} R_j, and ascends (1/k) sum_i (R_i - b_i) grad log pi(y_i).
    Step size decays as lr0 / (1 + t/2000) (Robbins-Monro), so the iterate settles.
    """
    K = len(r)
    theta = np.log(pi_ref).copy()
    for t in range(iters):
        pi = softmax(theta)
        ys = rng.choice(K, size=k, p=pi)
        R = r[ys] - beta * (np.log(pi[ys]) - np.log(pi_ref[ys]))
        b = (R.sum() - R) / (k - 1)  # leave-one-out baseline: unbiased (Section 4.3)
        grad = np.zeros(K)
        for y, adv in zip(ys, R - b):
            score = -pi.copy()
            score[y] += 1.0  # grad_theta log softmax(theta)_y = e_y - pi
            grad += adv * score
        theta += lr0 / (1.0 + t / 2000.0) * grad / k
    return softmax(theta)


# --------------------------------------------------------------------------------------
# DPO and IPO with population (infinite-data) preferences
# --------------------------------------------------------------------------------------
def pref_matrix(r, deterministic=False):
    """P[a, b] = Pr{a preferred to b}.  Bradley-Terry sigma(r_a - r_b), or 1[r_a > r_b]."""
    D = r[:, None] - r[None, :]
    if deterministic:
        return (D > 0).astype(float) + 0.5 * (D == 0)
    return expit(D)


def _dpo_loss_grad(theta, pi_ref, W, beta):
    """Population DPO loss and its gradient w.r.t. softmax logits theta.

    Pairs (a, b) ~ pi_ref x pi_ref; the label says a wins with probability P[a, b];
    W[a, b] = pi_ref(a) pi_ref(b) P[a, b] is the weight of the event "a beats b".
    L(theta) = - sum_{a,b} W[a,b] log sigma(beta (h_a - h_b)),  h = log pi_theta - log pi_ref
    (the expectation of the DPO loss, eq. 18.19).  Because h_a - h_b = theta_a - theta_b + const,
    L is a *convex* function of the logits, so L-BFGS finds its minimizer reliably.
    """
    logpi = log_softmax(theta)
    h = logpi - np.log(pi_ref)
    z = beta * (h[:, None] - h[None, :])
    loss = -np.sum(W * np.log(expit(z) + 1e-300))
    g_ab = W * (1.0 - expit(z))
    dL_dh = -beta * (g_ab.sum(1) - g_ab.sum(0))
    pi = np.exp(logpi)
    # chain rule through h_a = theta_a - logsumexp(theta) - log pi_ref(a):
    # dL/dtheta_c = dL/dh_c - pi_c * sum_a dL/dh_a   (the sum is zero here: L sees only h_a - h_b)
    return loss, dL_dh - pi * dL_dh.sum()


def population_dpo(pi_ref, P, beta, steps=20000, lr=1.0, track=None, lbfgs=False):
    """Minimize the population DPO loss, by plain gradient descent (to watch the trajectory)
    or by L-BFGS (to get the minimizer to high precision)."""
    W = pi_ref[:, None] * pi_ref[None, :] * P
    theta = np.log(pi_ref).copy()  # start at the reference policy
    if lbfgs:
        res = minimize(_dpo_loss_grad, theta, args=(pi_ref, W, beta), jac=True, method="L-BFGS-B",
                       options={"maxiter": steps, "gtol": 1e-12, "ftol": 1e-15})
        return softmax(res.x), []
    hist = []
    for s in range(steps):
        _, g = _dpo_loss_grad(theta, pi_ref, W, beta)
        theta -= lr * g
        if track is not None and (s % track == 0 or s == steps - 1):
            hist.append((s + 1, kl(softmax(theta), pi_ref)))
    return softmax(theta), hist


def _ipo_loss_grad(theta, pi_ref, W, tau):
    logpi = log_softmax(theta)
    h = logpi - np.log(pi_ref)
    res = h[:, None] - h[None, :] - 1.0 / (2.0 * tau)
    loss = np.sum(W * res**2)
    g = 2 * W * res
    dL_dh = g.sum(1) - g.sum(0)
    pi = np.exp(logpi)
    return loss, dL_dh - pi * dL_dh.sum()


def population_ipo(pi_ref, P, tau, steps=5000):
    """Minimize the population IPO loss sum_ab W[a,b] (h_a - h_b - 1/(2 tau))^2 by L-BFGS."""
    W = pi_ref[:, None] * pi_ref[None, :] * P
    res = minimize(_ipo_loss_grad, np.log(pi_ref), args=(pi_ref, W, tau), jac=True,
                   method="L-BFGS-B", options={"maxiter": steps, "gtol": 1e-13, "ftol": 1e-16})
    return softmax(res.x)


def ipo_target(pi_ref, P, tau):
    """Azar et al.'s IPO optimum: pi(y) propto pi_ref(y) exp(p(y > pi_ref) / tau)."""
    win = P @ pi_ref  # p(y beats a random draw from pi_ref)
    return softmax(np.log(pi_ref) + win / tau)


# --------------------------------------------------------------------------------------
# Best-of-n
# --------------------------------------------------------------------------------------
def best_of_n_dist(pi_ref, r, n):
    """Exact distribution of argmax_{i<=n} r(Y_i), Y_i ~ pi_ref iid (rewards distinct).

    Sort responses by reward; with CDF F(j) = Pr{r(Y) <= r(y_(j))}, the best of n equals
    y_(j) with probability F(j)^n - F(j-1)^n.
    """
    order = np.argsort(r)
    F = np.cumsum(pi_ref[order])
    F_prev = np.concatenate([[0.0], F[:-1]])
    p_sorted = F**n - F_prev**n
    p = np.empty_like(p_sorted)
    p[order] = p_sorted
    return p


def bon_kl_bound(n):
    return np.log(n) - (n - 1) / n


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    quick = args.quick
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    K = 8
    pi_ref, r = make_bandit(K, rng)
    betas = [0.1, 0.5, 2.0]
    print(f"seed={SEED}  K={K}  betas={betas}  quick={quick}")
    np.set_printoptions(precision=3, suppress=True)
    print("pi_ref =", pi_ref)
    print("r      =", r)

    # ---- 1. closed form vs three solvers ------------------------------------------------
    print("\n[1] Closed form pi*_beta vs independent solvers (max |difference| in probability)")
    iters = 6000 if quick else 60000
    solutions = {}
    for beta in betas:
        pi_star, J_star = closed_form(pi_ref, r, beta)
        p_slsqp = solve_slsqp(pi_ref, r, beta)
        p_grad = solve_exact_gradient(pi_ref, r, beta, steps=4000 if quick else 20000,
                                      lr=min(2.0, 0.5 / beta))
        p_rf = solve_reinforce(pi_ref, r, beta, np.random.default_rng(SEED + 1), iters=iters)
        solutions[beta] = (pi_star, p_slsqp, p_grad, p_rf)
        print(f"  beta={beta:<4}  J*=beta log Z={J_star:.5f}  J(pi*)={objective(pi_star, pi_ref, r, beta):.5f}"
              f"  |SLSQP|={np.abs(p_slsqp - pi_star).max():.1e}  |exact-grad|={np.abs(p_grad - pi_star).max():.1e}"
              f"  |REINFORCE|={np.abs(p_rf - pi_star).max():.1e}  KL(pi*||ref)={kl(pi_star, pi_ref):.3f}"
              f"  E[r]={pi_star @ r:.3f}")
    if not quick:
        # Why is exact gradient ascent least accurate at beta = 0.1?  In logit coordinates the
        # curvature of J along arm a is roughly beta * pi_a, and pi*_0.1 gives some arms
        # probabilities as small as 1e-11, so the problem is badly conditioned and plain gradient
        # ascent converges slowly.  Five times more steps shrink the error, but not to zero.
        pi_star, _ = closed_form(pi_ref, r, 0.1)
        p_long = solve_exact_gradient(pi_ref, r, 0.1, steps=100000, lr=2.0)
        print(f"  beta=0.1, exact gradient with 100000 steps instead of 20000: "
              f"|exact-grad|={np.abs(p_long - pi_star).max():.1e}  (smallest pi*_0.1(y) = {pi_star.min():.1e})")
    # any other policy does worse: random perturbations of pi*
    beta = 0.5
    pi_star, J_star = closed_form(pi_ref, r, beta)
    worse = 0
    for _ in range(1000):
        q = softmax(np.log(pi_star) + 0.3 * rng.normal(size=K))
        worse += objective(q, pi_ref, r, beta) < J_star
    print(f"  beta=0.5: {worse}/1000 random perturbations of pi* have a lower objective")
    # The identity J(pi) = J* - beta KL(pi || pi*) (eq. 18.10) on a random policy
    q = softmax(rng.normal(size=K))
    print(f"  identity check: J(q) = {objective(q, pi_ref, r, beta):.6f},  "
          f"J* - beta KL(q||pi*) = {J_star - beta * kl(q, pi_star):.6f}")

    # ---- 2. DPO and IPO ----------------------------------------------------------------
    print("\n[2] Population DPO with Bradley-Terry preferences recovers pi*_beta")
    P_bt = pref_matrix(r)
    dpo_steps = 3000 if quick else 20000
    dpo_bt = {}
    for beta in betas:
        p_dpo, _ = population_dpo(pi_ref, P_bt, beta, steps=dpo_steps, lbfgs=True)
        pi_star, _ = closed_form(pi_ref, r, beta)
        dpo_bt[beta] = p_dpo
        print(f"  beta={beta:<4}  max|pi_DPO - pi*| = {np.abs(p_dpo - pi_star).max():.1e}")

    print("\n[2b] Deterministic preferences (annotator always picks the higher true reward)")
    P_det = pref_matrix(r, deterministic=True)
    det_betas = [0.1, 0.5, 2.0]
    taus = [0.1, 0.5, 2.0]
    track = 10
    dpo_det_curves = {}
    a_star = int(np.argmax(r))
    for beta in det_betas:
        # gradient descent with step size 1/beta^2, so that beta*(h_a - h_b), the quantity the
        # loss sees, moves at a beta-independent speed and the curves are comparable
        p_det, hist = population_dpo(pi_ref, P_det, beta, steps=dpo_steps, lr=1.0 / beta**2,
                                     track=track)
        dpo_det_curves[beta] = hist
        p_lb, _ = population_dpo(pi_ref, P_det, beta, steps=dpo_steps, lbfgs=True)
        print(f"  DPO beta={beta:<4}  after {dpo_steps} GD steps: pi(best arm)={p_det[a_star]:.4f}  "
              f"KL={kl(p_det, pi_ref):.3f};  L-BFGS run to its stopping rule: pi(best arm)="
              f"{p_lb[a_star]:.4f}  KL={kl(p_lb, pi_ref):.3f}")
    print(f"  (KL of the point mass on the best arm = {-np.log(pi_ref[a_star]):.3f})")
    ipo_res = {}
    for tau in taus:
        p_ipo = population_ipo(pi_ref, P_det, tau)
        target = ipo_target(pi_ref, P_det, tau)
        ipo_res[tau] = p_ipo
        print(f"  IPO tau={tau:<4}  pi(best arm)={p_ipo[a_star]:.4f}  KL={kl(p_ipo, pi_ref):.3f}  "
              f"max|pi_IPO - pi_ref*exp(p(y>ref)/tau)/Z| = {np.abs(p_ipo - target).max():.1e}")

    # ---- 3. Best-of-n ------------------------------------------------------------------
    print("\n[3] Best-of-n: exact KL vs log n - (n-1)/n, and the reward-KL frontier")
    # a log-spaced grid plus the values of n quoted in the text and on the figure, so that every
    # printed or annotated n is a value that was actually computed
    ns = np.unique(np.concatenate([np.round(np.logspace(0, 3, 25)),
                                   [2, 4, 8, 16, 64, 256, 512]]).astype(int))
    big_K = 10000
    pi_u = np.ones(big_K) / big_K
    r_u = np.random.default_rng(SEED + 2).normal(size=big_K)
    rows = []
    for n in ns:
        p8 = best_of_n_dist(pi_ref, r, n)
        pu = best_of_n_dist(pi_u, r_u, n)
        rows.append((n, kl(p8, pi_ref), kl(pu, pi_u), bon_kl_bound(n), p8 @ r, pu @ r_u))
    rows = np.array(rows)
    for n in [1, 2, 4, 16, 64, 256, 1000]:
        i = int(np.flatnonzero(rows[:, 0] == n)[0])  # exact match: n is on the grid
        print(f"  n={int(rows[i, 0]):<5} bound={rows[i, 3]:.3f}  exact KL (K=8)={rows[i, 1]:.3f}  "
              f"exact KL (K=10^4 uniform)={rows[i, 2]:.3f}  E[r] BoN (K=8)={rows[i, 4]:.3f}")
    # optimal frontier on the K=10^4 problem vs best-of-n at equal KL
    beta_grid = np.logspace(-2.5, 1.5, 200)
    front = np.array([(kl(closed_form(pi_u, r_u, b)[0], pi_u), closed_form(pi_u, r_u, b)[0] @ r_u)
                      for b in beta_grid])
    gap_rows = []
    for n in [4, 16, 64, 256]:
        i = int(np.flatnonzero(rows[:, 0] == n)[0])
        kl_n, r_n = rows[i, 2], rows[i, 5]
        r_opt = np.interp(kl_n, front[::-1, 0], front[::-1, 1])
        gap_rows.append((n, kl_n, r_n, r_opt))
        print(f"  K=10^4: n={int(rows[i, 0]):<4} KL={kl_n:.3f}  E[r] best-of-n={r_n:.3f}  "
              f"E[r] of pi*_beta at the same KL={r_opt:.3f}  (ratio {r_n / r_opt:.3f})")

    print(f"\nTotal time {time.time() - t0:.1f}s")
    if quick:
        return

    # ---- figures -----------------------------------------------------------------------
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    x = np.arange(K)
    w = 0.2
    ax.bar(x - 1.5 * w, pi_ref, w, color="0.6", label=r"$\pi_{\mathrm{ref}}$")
    cols = ["tab:blue", "tab:orange", "tab:green"]
    for j, beta in enumerate(betas):
        pi_star, p_slsqp, p_grad, p_rf = solutions[beta]
        ax.bar(x + (j - 0.5) * w, pi_star, w, color=cols[j], label=rf"$\pi^\ast_\beta$, $\beta={beta}$")
        ax.plot(x + (j - 0.5) * w, p_rf, "kx", ms=5)
        ax.plot(x + (j - 0.5) * w, dpo_bt[beta], "o", mfc="none", mec="k", ms=6)
    ax.plot([], [], "kx", label="sampled REINFORCE (RLOO)")
    ax.plot([], [], "o", mfc="none", mec="k", label="population DPO")
    ax.set_xticks(x, [f"y={i}\nr={r[i]:+.2f}" for i in range(K)], fontsize=7)
    ax.set_xlabel("response y and its true reward r(y)")
    ax.set_ylabel("probability")
    ax.set_title("Closed form vs solvers (K=8)")
    ax.set_ylim(0, 1.25)  # head-room so the legend does not cover the bars
    ax.legend(fontsize=7, loc="upper left", ncol=2)

    ax = axes[1]
    for beta in det_betas:
        h = np.array(dpo_det_curves[beta])
        ax.plot(h[:, 0], h[:, 1], label=rf"DPO, $\beta={beta}$")
    for tau, ls in zip(taus, [":", "-.", "--"]):
        ax.axhline(kl(ipo_res[tau], pi_ref), color="k", ls=ls, lw=1, label=rf"IPO optimum, $\tau={tau}$")
    ax.axhline(-np.log(pi_ref[a_star]), color="r", lw=0.8, label="point mass on best arm")
    ax.set_xscale("log")
    ax.set_xlabel("gradient step")
    ax.set_ylabel(r"KL$(\pi_\theta\,\Vert\,\pi_{\mathrm{ref}})$")
    ax.set_title("Deterministic preferences: DPO ignores $\\beta$")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(front[:, 0], front[:, 1], "k-", label=r"optimal frontier $\pi^\ast_\beta$")
    ax.plot(rows[:, 2], rows[:, 5], "o", color="tab:purple", ms=4, label="best-of-n (exact KL)")
    for n in [2, 8, 64, 512]:
        i = int(np.flatnonzero(rows[:, 0] == n)[0])
        ax.annotate(f"n={int(rows[i, 0])}", (rows[i, 2], rows[i, 5]), fontsize=7, xytext=(4, -10),
                    textcoords="offset points")
    ax.set_xlim(0, 7.5)
    ax.set_xlabel(r"KL$(\pi\,\Vert\,\pi_{\mathrm{ref}})$ (nats)")
    ax.set_ylabel("expected true reward")
    ax.set_title(r"Reward-KL trade-off ($K=10^4$, uniform $\pi_{\mathrm{ref}}$)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "kl_bandit.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(rows[:, 0], rows[:, 3], "k-", label=r"$\log n-(n-1)/n$")
    ax.plot(rows[:, 0], rows[:, 2], "o", ms=4, label=r"exact, $K=10^4$ uniform $\pi_{\mathrm{ref}}$")
    ax.plot(rows[:, 0], rows[:, 1], "s", ms=4, label=r"exact, the $K=8$ bandit")
    ax.axhline(-np.log(pi_ref[a_star]), color="tab:orange", lw=0.8, ls="--",
               label=r"$-\log\pi_{\mathrm{ref}}(y^\ast)$ ($K=8$ ceiling)")
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel(r"KL$(\pi_{\mathrm{BoN}}\,\Vert\,\pi_{\mathrm{ref}})$")
    ax.set_title("KL of best-of-n: the formula is an upper bound")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "bon_kl.png"), dpi=110)
    plt.close(fig)
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
