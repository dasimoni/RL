"""Conservative Q-learning in a table: checking the lower-bound theorems numerically
(Chapter 16, Section 8).

CQL policy evaluation (Kumar, Zhou, Tucker & Levine 2020) for a fixed target policy pi:

  Q_{k+1} = argmin_Q  alpha * ( E_{s~D, a~mu}[Q(s,a)] - E_{s~D, a~pi_beta_hat}[Q(s,a)] )
                      + 1/2 E_{(s,a,s')~D}[ (Q(s,a) - B_hat^pi Q_k(s,a))^2 ]

In a table this has the closed form (Eq. 16.28)

  Q_{k+1}(s,a) = B_hat^pi Q_k(s,a) - alpha * (mu(a|s) - pi_beta_hat(a|s)) / pi_beta_hat(a|s),

and without the second ("push-up") term, Q_{k+1} = B_hat^pi Q_k - alpha * mu / pi_beta_hat.

Part 1 (exact, no sampling error: true model, true behaviour policy)
  * push-down only (Theorem 16.2):  Q_hat <= Q^pi at EVERY (s, a);
  * with push-up and mu = pi (Theorem 16.3):  V_hat(s) = V^pi(s) - alpha [(I - gamma P_pi)^-1 D_CQL](s)
    with D_CQL(s) = sum_a pi(a|s) (pi(a|s) / pi_beta(a|s) - 1) >= 0 (Eq. 16.30): the state values
    are lower bounds although individual Q_hat(s, a) need not be.
Part 2 (finite datasets): the empirical Bellman operator has sampling error, so plain
  evaluation (alpha = 0) over- or under-estimates at random; we measure, over many datasets, how
  often V_hat <= V^pi holds at all states as alpha grows -- and how pessimistic it becomes.

Run:  python code/ch16_offline_rl_and_imitation/cql_tabular.py [--quick]
Output (full mode): figures/cql_tabular.png
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
NS, NA, GAMMA = 6, 3, 0.9
REWARD_NOISE = 0.3


def make_problem(seed=0):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(0.5 * np.ones(NS), size=(NS, NA))            # P[s, a, s']
    R = rng.uniform(0, 1, size=(NS, NA))
    logits = 0.8 * rng.standard_normal((NS, NA))
    pi_beta = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)  # behaviour policy
    return P, R, pi_beta


def evaluate(P, R, pi):
    """Exact Q^pi and V^pi by solving the linear Bellman equations (Chapter 03)."""
    P_pi = np.einsum("sap,pb->sapb", P, pi).reshape(NS * NA, NS * NA)   # (s,a) -> (s',a')
    q = np.linalg.solve(np.eye(NS * NA) - GAMMA * P_pi, R.ravel()).reshape(NS, NA)
    return q, (pi * q).sum(1)


def cql_evaluate(P, R, pi, pi_beta_hat, alpha, mu=None, push_up=True, iters=None):
    """Fixed point of the tabular CQL evaluation iteration.  With a fixed per-(s, a) penalty c,
    Q = B^pi Q - alpha c is ordinary policy evaluation with reward R - alpha c."""
    mu = pi if mu is None else mu
    c = (mu - pi_beta_hat) / pi_beta_hat if push_up else mu / pi_beta_hat
    if iters is None:
        return evaluate(P, R - alpha * c, pi)
    Q = np.zeros((NS, NA))                                    # literal iteration, for checking
    for _ in range(iters):
        V = (pi * Q).sum(1)
        Q = R + GAMMA * P @ V - alpha * c
    return Q, (pi * Q).sum(1)


def sample_dataset(P, R, pi_beta, n_per_state, rng):
    """n_per_state transitions from every state (d_D uniform), actions from pi_beta.  Returns
    empirical P_hat, R_hat, pi_beta_hat; re-draws until every (s, a) has been seen once (so the
    tabular CQL penalty is finite), and reports how many draws that took."""
    tries = 0
    while True:
        tries += 1
        cnt = np.zeros((NS, NA)); cnt_sp = np.zeros((NS, NA, NS)); rsum = np.zeros((NS, NA))
        for s in range(NS):
            a = rng.choice(NA, size=n_per_state, p=pi_beta[s])
            for ai in range(NA):
                k = int((a == ai).sum())
                if k == 0:
                    continue
                cnt[s, ai] = k
                cnt_sp[s, ai] = rng.multinomial(k, P[s, ai])
                rsum[s, ai] = k * R[s, ai] + REWARD_NOISE * np.sqrt(k) * rng.standard_normal()
        if (cnt > 0).all():
            break
    P_hat = cnt_sp / cnt[:, :, None]
    R_hat = rsum / cnt
    pi_beta_hat = cnt / cnt.sum(1, keepdims=True)
    return P_hat, R_hat, pi_beta_hat, tries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    quick = args.quick
    seed = 0
    rng = np.random.default_rng(seed)
    P, R, pi_beta = make_problem(seed)
    q_opt = np.zeros((NS, NA))
    for _ in range(500):                                      # value iteration for q_*
        q_opt = R + GAMMA * P @ q_opt.max(1)
    greedy = np.eye(NA)[q_opt.argmax(1)]
    pi = 0.8 * greedy + 0.1 * (1 - greedy)                     # target policy to evaluate
    q_true, v_true = evaluate(P, R, pi)
    D_cql = (pi * (pi / pi_beta - 1)).sum(1)
    print(f"cql_tabular: seed={seed} quick={quick} |S|={NS} |A|={NA} gamma={GAMMA} "
          f"reward noise sd={REWARD_NOISE}")
    print("behaviour pi_beta(a|s) min/max:", f"{pi_beta.min():.3f}/{pi_beta.max():.3f}",
          "  D_CQL(s) =", np.round(D_cql, 3).tolist())
    print("true V^pi(s) =", np.round(v_true, 3).tolist())
    t0 = time.time()

    # ---------------- Part 1: exact ----------------
    print("\nPart 1 (exact model and behaviour policy)")
    alphas_exact = [0.0, 0.05, 0.2]
    P_s = np.einsum("sap,sa->sp", P, pi)                       # state transition matrix under pi
    exact = {}
    for al in alphas_exact:
        q1, v1 = cql_evaluate(P, R, pi, pi_beta, al, push_up=False)
        q2, v2 = cql_evaluate(P, R, pi, pi_beta, al, push_up=True)
        q2_it, _ = cql_evaluate(P, R, pi, pi_beta, al, push_up=True, iters=400)
        v2_formula = v_true - al * np.linalg.solve(np.eye(NS) - GAMMA * P_s, D_cql)
        exact[al] = (v1, v2)
        print(f"  alpha={al:4.2f}  push-down only: max_(s,a) [Q_hat - Q^pi] = {np.max(q1 - q_true):+.4f}"
              f" | with push-up: #(s,a) with Q_hat > Q^pi = {int(np.sum(q2 > q_true + 1e-12))}"
              f" of {NS * NA}, max_s [V_hat - V^pi] = {np.max(v2 - v_true):+.4f},"
              f" |V_hat - formula| = {np.max(np.abs(v2 - v2_formula)):.1e},"
              f" |closed form - iteration| = {np.max(np.abs(q2 - q2_it)):.1e}")

    # ---------------- Part 2: finite datasets ----------------
    n_sets = 40 if quick else 1000
    sizes = [20, 200]
    alphas = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3]
    print(f"\nPart 2 (finite datasets, {n_sets} datasets per size)")
    res = {}
    for n in sizes:
        frac, gap, tries_tot = np.zeros(len(alphas)), np.zeros(len(alphas)), 0
        for _ in range(n_sets):
            P_hat, R_hat, pib_hat, tries = sample_dataset(P, R, pi_beta, n, rng)
            tries_tot += tries
            for j, al in enumerate(alphas):
                _, v_hat = cql_evaluate(P_hat, R_hat, pi, pib_hat, al, push_up=True)
                frac[j] += np.all(v_hat <= v_true) / n_sets
                gap[j] += np.mean(v_hat - v_true) / n_sets
        res[n] = (frac, gap)
        print(f"  {n} transitions per state ({tries_tot - n_sets} re-draws for unseen (s,a): "
              f"kept {n_sets} of {tries_tot} draws, {n_sets / tries_tot:.0%}; results are "
              f"conditional on every (s,a) being seen):")
        for j, al in enumerate(alphas):
            print(f"    alpha={al:5.3f}: P(V_hat <= V^pi at all states) = {frac[j]:.3f}, "
                  f"mean_s E[V_hat - V^pi] = {gap[j]:+.3f}")
    print(f"total time {time.time() - t0:.1f}s")

    if quick:
        return
    import plot_style
    plt = plot_style.setup()
    C = plot_style.C
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    ax = axes[0]
    xs = np.arange(NS)
    ax.plot(xs, v_true, "k-", marker="o", label=r"true $V^\pi$")
    for k, al in enumerate(alphas_exact[1:]):
        ax.plot(xs, exact[al][1], marker="s", color=C[k], ls="--",
                label=fr"CQL, $\alpha$={al} (push-down + push-up)")
        ax.plot(xs, exact[al][0], marker="^", color=C[k], ls=":",
                label=fr"CQL, $\alpha$={al} (push-down only)")
    ax.set_xlabel("state $s$")
    ax.set_ylabel("state value")
    ax.set_title("Exact case: CQL values sit below $V^\\pi$")
    ax.legend(fontsize=7.5)
    ax = axes[1]
    for k, n in enumerate(sizes):
        frac, gap = res[n]
        ax.semilogx(np.array(alphas[1:]), frac[1:], marker="o", color=C[k],
                    label=f"P(lower bound at all states), {n}/state")
        ax.axhline(frac[0], color=C[k], ls=":", lw=1, label=fr"same with $\alpha=0$, {n}/state")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("fraction of datasets")
    ax.set_ylim(0, 1.03)
    ax2 = ax.twinx()
    for k, n in enumerate(sizes):
        ax2.semilogx(np.array(alphas[1:]), res[n][1][1:], ls="--", color=C[k], marker="x",
                     label=f"mean $\\hat V - V^\\pi$, {n}/state")
    ax2.set_ylabel(r"mean $\hat V(s) - V^\pi(s)$ (dashed)")
    ax2.grid(False)
    lines = ax.get_lines() + ax2.get_lines()
    ax.set_title("Finite data: larger $\\alpha$ buys a reliable bound with pessimism")
    fig.tight_layout()
    # legend below the axes, so it covers neither the curves nor the alpha = 0 reference lines
    fig.subplots_adjust(bottom=0.32)
    ax.legend(lines, [l.get_label() for l in lines], fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.savefig(os.path.join(FIG_DIR, "cql_tabular.png"))
    plt.close(fig)
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
