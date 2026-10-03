"""Off-policy evaluation on a small tabular MDP: IS, WIS, PDIS, WPDIS, FQE, DR and WDR
(Chapter 16, Section 12).

Problem.  A random finite-horizon MDP (8 states, 3 actions, horizon H = 10, gamma = 0.95,
Gaussian reward noise).  A logging (behaviour) policy b generated n episodes; we must estimate
the value v(pi) = E_pi[sum_{t<H} gamma^t R_{t+1}] of a different target policy pi from those
logs alone.  The true v(pi) is computed exactly by backward induction on the true model.

Estimators (equation numbers refer to Section 12)
  IS      trajectory-wise importance sampling                       (Eq. 16.39, Algorithm 16.15)
  WIS     weighted (self-normalised) trajectory-wise IS             (Eq. 16.39).  When every
          trajectory weight is 0 (no logged trajectory matches a deterministic pi), WIS is 0/0,
          undefined; the code's guard max(sum w, 1e-300) then reports 0.
  PDIS    per-decision IS                                           (Eq. 16.40)
  WPDIS   weighted per-decision IS (per-time-step normalisation)    (Eq. 16.40)
  FQE     fitted Q evaluation, tabular: regress r + gamma V_{t+1}(s') on (s, a), backwards in t
          (Algorithm 16.16)
  FQE-agg FQE with a MISSPECIFIED model class: states 0-3 share one value, 4-7 another
  DR      per-decision doubly robust estimator with the FQE model   (Eq. 16.42, Algorithm 16.17)
  DR-agg  doubly robust with the misspecified FQE-agg model
  WDR     weighted DR (Thomas & Brunskill 2016), FQE model          (Algorithm 16.17)

Experiments
  1. RMSE, bias and standard deviation vs the number of logged episodes n.
  2. RMSE vs the horizon H at fixed n (the "curse of horizon" for trajectory-wise IS).
  3. RMSE vs how far pi is from b (mixtures between b and a deterministic greedy policy).

Run:  python code/ch16_offline_rl_and_imitation/ope_tabular.py [--quick]
Output (full mode): figures/ope_vs_n.png, figures/ope_horizon_mismatch.png
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

NS, NA = 8, 3
GAMMA = 0.95
REWARD_NOISE = 0.5


def make_mdp(seed=0):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(0.3 * np.ones(NS), size=(NS, NA))       # P[s, a, s']
    R = rng.uniform(0, 1, size=(NS, NA))                        # mean reward r(s, a)
    d0 = np.ones(NS) / NS
    return P, R, d0


def true_value(P, R, d0, pi, H):
    """v(pi) by backward induction; also returns q_t for t = 0..H-1."""
    q = np.zeros((H, NS, NA))
    v_next = np.zeros(NS)
    for t in reversed(range(H)):
        q[t] = R + GAMMA * P @ v_next
        v_next = (pi * q[t]).sum(1)
    return float(d0 @ v_next), q


def optimal_q(P, R, H):
    q = np.zeros((H, NS, NA)); v = np.zeros(NS)
    for t in reversed(range(H)):
        q[t] = R + GAMMA * P @ v
        v = q[t].max(1)
    return q


def generate(P, R, d0, b, n, H, rng):
    """n episodes of length H under b: arrays S [n, H], A [n, H], Rw [n, H] (Rw[:, t] = R_{t+1})."""
    S = np.zeros((n, H), int); A = np.zeros((n, H), int); Rw = np.zeros((n, H))
    s = rng.choice(NS, size=n, p=d0)
    cdf_P = P.cumsum(2)
    cdf_b = b.cumsum(1)
    for t in range(H):
        a = (rng.random(n)[:, None] > cdf_b[s]).sum(1)
        a = np.minimum(a, NA - 1)
        S[:, t], A[:, t] = s, a
        Rw[:, t] = R[s, a] + REWARD_NOISE * rng.standard_normal(n)
        s = np.minimum((rng.random(n)[:, None] > cdf_P[s, a]).sum(1), NS - 1)
    return S, A, Rw


# ---------------------------------------------------------------------- importance sampling
def ratios(S, A, pi, b):
    """rho_{0:t} for every episode and t (cumulative products), shape [n, H]."""
    return np.cumprod(pi[S, A] / b[S, A], axis=1)


def est_is(S, A, Rw, pi, b):
    disc = GAMMA ** np.arange(S.shape[1])
    G = (Rw * disc).sum(1)
    w = ratios(S, A, pi, b)[:, -1]
    return float(np.mean(w * G)), float(np.sum(w * G) / max(np.sum(w), 1e-300))


def est_pdis(S, A, Rw, pi, b):
    disc = GAMMA ** np.arange(S.shape[1])
    rho = ratios(S, A, pi, b)
    pdis = float(np.mean((rho * Rw * disc).sum(1)))
    w = rho / np.maximum(rho.sum(0, keepdims=True), 1e-300)          # per-time-step weights
    wpdis = float(((w * Rw).sum(0) * disc).sum())
    return pdis, wpdis


# ---------------------------------------------------------------------- fitted Q evaluation
def fqe(S, A, Rw, pi, H, groups=None):
    """Tabular FQE (Algorithm 16.16), returns Q_hat[t, s, a] for t = 0..H-1.

    For t = H-1, ..., 0 regress the targets y = R_{t+1} + gamma * V_hat_{t+1}(S_{t+1}) on (S, A)
    using ALL logged transitions (the dynamics are time-homogeneous).  With a lookup table the
    least-squares fit is the per-(s, a) average of the targets.  `groups` maps each state to a
    group: states in the same group share a Q value (a misspecified function class).
    (s, a) pairs never seen get Q_hat = 0."""
    g = np.arange(NS) if groups is None else np.asarray(groups)
    s_now, a_now = S[:, :-1].ravel(), A[:, :-1].ravel()
    s_next, r_now = S[:, 1:].ravel(), Rw[:, :-1].ravel()
    s_last, a_last, r_last = S[:, -1], A[:, -1], Rw[:, -1]
    # every transition (s, a, r, s'), with s' unavailable after the last step (target = r there
    # only for t = H-1; for t < H-1 every transition has a successor)
    s_all = np.concatenate([s_now, s_last]); a_all = np.concatenate([a_now, a_last])
    r_all = np.concatenate([r_now, r_last])
    cnt = np.zeros((NS, NA))
    np.add.at(cnt, (g[s_all], a_all), 1)
    cnt_now = np.zeros((NS, NA))
    np.add.at(cnt_now, (g[s_now], a_now), 1)
    Q = np.zeros((H, NS, NA))
    # last step: target is the reward alone (episode ends at H)
    tot = np.zeros((NS, NA)); np.add.at(tot, (g[s_all], a_all), r_all)
    Qg = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
    Q[H - 1] = Qg[g]
    for t in reversed(range(H - 1)):
        v_next = (pi * Q[t + 1]).sum(1)
        y = r_now + GAMMA * v_next[s_next]
        tot = np.zeros((NS, NA)); np.add.at(tot, (g[s_now], a_now), y)
        Qg = np.where(cnt_now > 0, tot / np.maximum(cnt_now, 1), 0.0)
        Q[t] = Qg[g]
    return Q


def est_fqe(S, Q, pi):
    """Average V_hat_0 over the logged initial states (the empirical d0)."""
    return float((pi * Q[0]).sum(1)[S[:, 0]].mean())


def est_dr(S, A, Rw, pi, b, Q):
    """Per-decision doubly robust (Jiang & Li 2016), Eq. (16.42), and its weighted version WDR
    (Thomas & Brunskill 2016) in which rho_{0:t} / n is replaced by rho_{0:t} / sum_i rho_{0:t}."""
    n, H = S.shape
    disc = GAMMA ** np.arange(H)
    rho = ratios(S, A, pi, b)                                        # rho_{0:t}
    rho_prev = np.concatenate([np.ones((n, 1)), rho[:, :-1]], 1)     # rho_{0:t-1}
    t_idx = np.arange(H)[None, :]
    q_sa = Q[t_idx, S, A]                                            # Q_hat_t(S_t, A_t)
    v_s = (pi[S] * Q[t_idx, S]).sum(2)                               # V_hat_t(S_t)
    dr = float(np.mean((disc * (rho * (Rw - q_sa) + rho_prev * v_s)).sum(1)))
    w = rho / np.maximum(rho.sum(0, keepdims=True), 1e-300)
    w_prev = rho_prev / np.maximum(rho_prev.sum(0, keepdims=True), 1e-300)
    wdr = float((disc * (w * (Rw - q_sa) + w_prev * v_s)).sum())
    return dr, wdr


AGG = [0, 0, 0, 0, 1, 1, 1, 1]          # misspecified class: two groups of 4 states share values
NAMES = ["IS", "WIS", "PDIS", "WPDIS", "FQE", "FQE-agg", "DR", "DR-agg", "WDR"]


def all_estimates(S, A, Rw, pi, b, H):
    is_, wis = est_is(S, A, Rw, pi, b)
    pdis, wpdis = est_pdis(S, A, Rw, pi, b)
    Q = fqe(S, A, Rw, pi, H)
    Qa = fqe(S, A, Rw, pi, H, groups=AGG)
    dr, wdr = est_dr(S, A, Rw, pi, b, Q)
    dra, _ = est_dr(S, A, Rw, pi, b, Qa)
    return [is_, wis, pdis, wpdis, est_fqe(S, Q, pi), est_fqe(S, Qa, pi), dr, dra, wdr]


def run_reps(P, R, d0, b, pi, n, H, reps, rng):
    est = np.array([all_estimates(*generate(P, R, d0, b, n, H, rng), pi, b, H) for _ in range(reps)])
    v, _ = true_value(P, R, d0, pi, H)
    err = est - v
    return v, np.sqrt((err ** 2).mean(0)), err.mean(0), est.std(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    quick = args.quick
    seed = 0
    rng = np.random.default_rng(seed)
    reps = 30 if quick else 400
    H = 10
    P, R, d0 = make_mdp(seed)
    # behaviour: a fixed random softmax policy; target: 0.7 on the optimal action, 0.15 on others
    b = np.exp(0.7 * np.random.default_rng(1).standard_normal((NS, NA)))
    b /= b.sum(1, keepdims=True)
    qstar = optimal_q(P, R, H)[0]
    greedy = np.eye(NA)[qstar.argmax(1)]
    pi = 0.7 * greedy + 0.15 * (1 - greedy)
    v_pi, _ = true_value(P, R, d0, pi, H)
    v_b, _ = true_value(P, R, d0, b, H)
    per_step_var = (pi ** 2 / b).sum(1)
    print(f"ope_tabular: seed={seed} quick={quick} |S|={NS} |A|={NA} H={H} gamma={GAMMA} "
          f"reward noise sd={REWARD_NOISE} reps={reps}")
    print(f"true v(pi) = {v_pi:.4f}   v(b) = {v_b:.4f}   min b(a|s) = {b.min():.3f}   "
          f"E_b[rho^2] per step ranges {per_step_var.min():.2f}..{per_step_var.max():.2f}")
    t0 = time.time()

    # ---- Experiment 1: error vs number of episodes
    ns = [10, 30, 100] if quick else [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    rmse_n = []
    print(f"\nExperiment 1: RMSE vs n (H = {H}); columns: " + " ".join(NAMES))
    for n in ns:
        v, rmse, bias, sd = run_reps(P, R, d0, b, pi, n, H, reps, rng)
        rmse_n.append((rmse, bias, sd))
        print(f"  n={n:5d} RMSE " + " ".join(f"{x:7.3f}" for x in rmse))
    k100 = ns.index(100)
    print("  at n=100:  bias " + " ".join(f"{x:+7.3f}" for x in rmse_n[k100][1]))
    print("             sd   " + " ".join(f"{x:7.3f}" for x in rmse_n[k100][2]))

    # ---- Experiment 2: error vs horizon at n = 200
    Hs = [2, 5, 10] if quick else [2, 5, 10, 15, 20, 30, 40]
    n_fix = 200
    rmse_H = []
    print(f"\nExperiment 2: RMSE vs horizon H (n = {n_fix}); columns: " + " ".join(NAMES))
    for Hh in Hs:
        qs = optimal_q(P, R, Hh)[0]
        g_ = np.eye(NA)[qs.argmax(1)]
        pi_h = 0.7 * g_ + 0.15 * (1 - g_)
        v, rmse, bias, sd = run_reps(P, R, d0, b, pi_h, n_fix, Hh, reps, rng)
        rmse_H.append(rmse / max(abs(v), 1e-9))
        print(f"  H={Hh:3d} v(pi)={v:6.3f} relative RMSE " + " ".join(f"{x:7.3f}" for x in rmse_H[-1]))

    # ---- Experiment 3: error vs policy mismatch at n = 200, H = 10
    kappas = [0.0, 0.5, 1.0] if quick else [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    rmse_k = []
    print(f"\nExperiment 3: RMSE vs mismatch, pi_kappa = (1-kappa) b + kappa greedy (n = {n_fix}, H = {H})")
    for kap in kappas:
        pi_k = (1 - kap) * b + kap * greedy
        v, rmse, bias, sd = run_reps(P, R, d0, b, pi_k, n_fix, H, reps, rng)
        rmse_k.append(rmse)
        print(f"  kappa={kap:.1f} v={v:6.3f} max_s E_b[rho^2]={((pi_k ** 2) / b).sum(1).max():5.2f} RMSE "
              + " ".join(f"{x:7.3f}" for x in rmse))
    print(f"total time {time.time() - t0:.1f}s")

    if quick:
        return
    import plot_style
    plt = plot_style.setup()
    C = plot_style.C
    style = {"IS": (C[7], "o", "-"), "WIS": (C[3], "o", "--"), "PDIS": (C[4], "^", "-"),
             "WPDIS": (C[6], "^", "--"), "FQE": (C[2], "s", "-"), "FQE-agg": (C[2], "s", ":"),
             "DR": (C[0], "D", "-"), "DR-agg": (C[0], "D", ":"), "WDR": (C[1], "v", "--")}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for j, nm in enumerate(NAMES):
        col, mk, ls = style[nm]
        ax.loglog(ns, [r[0][j] for r in rmse_n], marker=mk, ls=ls, color=col, ms=4, label=nm)
    ax.loglog(ns, rmse_n[0][0][6] * np.sqrt(ns[0] / np.array(ns)), color=plot_style.GREY, lw=1,
              ls="-.", label=r"$\propto 1/\sqrt{n}$")
    ax.set_xlabel("number of logged episodes $n$")
    ax.set_ylabel(r"RMSE of $\hat v(\pi)$")
    ax.set_title(f"Error vs data size (H = {H}, {reps} repetitions)")
    ax.legend(fontsize=7.5, ncol=2)
    ax = axes[1]
    x = np.arange(len(NAMES))
    bias = np.abs(rmse_n[k100][1]); sd = rmse_n[k100][2]
    ax.bar(x - 0.2, bias, width=0.4, color=C[1], label="|bias|")
    ax.bar(x + 0.2, sd, width=0.4, color=C[0], label="standard deviation")
    ax.set_xticks(x); ax.set_xticklabels(NAMES, rotation=35)
    ax.set_yscale("log")
    ax.set_ylabel("error component")
    ax.set_title("Bias and spread at n = 100")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ope_vs_n.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for j, nm in enumerate(NAMES):
        if nm in ("FQE-agg", "DR-agg"):
            continue
        col, mk, ls = style[nm]
        ax.semilogy(Hs, [r[j] for r in rmse_H], marker=mk, ls=ls, color=col, ms=4, label=nm)
    ax.set_xlabel("horizon $H$")
    ax.set_ylabel(r"RMSE / $v(\pi)$")
    ax.set_title(f"Curse of horizon (n = {n_fix})")
    ax.legend(fontsize=7.5, ncol=2)
    ax = axes[1]
    for j, nm in enumerate(NAMES):
        col, mk, ls = style[nm]
        ax.semilogy(kappas, [r[j] for r in rmse_k], marker=mk, ls=ls, color=col, ms=4, label=nm)
    ax.set_xlabel(r"mismatch $\kappa$: $\pi_\kappa = (1-\kappa)\,b + \kappa\,\pi_{greedy}$")
    ax.set_ylabel(r"RMSE of $\hat v(\pi_\kappa)$")
    ax.set_title(f"Policy mismatch (n = {n_fix}, H = {H})")
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ope_horizon_mismatch.png"))
    plt.close(fig)
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
