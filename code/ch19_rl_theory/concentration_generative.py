"""Concentration inequalities and sample complexity with a generative model. Chapter 19, Section 5.

Part A  The concentration toolkit (Section 5.1). For Bernoulli(p) samples we compare the
        Hoeffding radius sqrt(L/(2n)) and the Bernstein radius sqrt(2 p(1-p) L/n) + 2bL/(3n)
        (Eq. 19.17), L = ln(2/delta), with b = max(p, 1-p), the smallest valid bound on |X_i - p|,
        with the empirical failure frequency, and show why a union bound is needed when many
        intervals must hold at once.

Part B  The toy theorem of Section 5.2 (Theorem 19.8) (plug-in planning with a generative model, finite horizon,
        fresh samples for every step h). With N next-state samples per (s, a, h), with probability
        at least 1 - delta,
            max_{s,a} |Qhat_0(s,a) - Q*_0(s,a)| <= H(H-1)/2 * sqrt(ln(2SAH/delta) / (2N))     (19.18)
            max_s (V*_0(s) - V^{pihat}_0(s))   <= H(H-1)   * sqrt(ln(4SAH/delta) / (2N))     (19.19)
        We measure the actual errors on a random MDP, count bound violations, and look at how the
        error scales with N and with H.

Part C  Why the minimax rate has (1-gamma)^-3 and not (1-gamma)^-4 (Section 5.3). The plug-in
        estimate on the hard instance of Azar, Munos & Kappen (2013) (self-loop probability
        p = (4 gamma - 1)/(3 gamma), reward 1, otherwise absorbed with reward 0) has error
        ~ (1-gamma)^-1.5 / sqrt(N), whereas a Hoeffding-only argument predicts (1-gamma)^-2 / sqrt(N).
        We also measure ||qhat* - q*||_inf of the plug-in estimate on a random dense MDP.

Run from the repository root:
  python code/ch19_rl_theory/concentration_generative.py           # full run (about 1 min), figure
  python code/ch19_rl_theory/concentration_generative.py --quick   # smoke test, no figure
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from theory_lib import COLORS, fh_eval, fh_optimal, loglog_slope, random_mdp, setup_style, value_iteration

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")


# ------------------------------------------------------------------------------------------------
# Part A
# ------------------------------------------------------------------------------------------------
def part_a(rng, trials):
    n, delta = 100, 0.05
    L = np.log(2 / delta)
    r_hoeff = np.sqrt(L / (2 * n))
    print(f"\nPart A: n = {n} Bernoulli samples, delta = {delta}, {trials:,d} trials")
    print(f"{'p':>6s} | {'Hoeffding radius':>16s} {'fail freq':>9s} | {'Bernstein radius':>16s} {'fail freq':>9s}"
          f"   (Bernstein with b = max(p, 1-p))")
    for p in (0.5, 0.1, 0.01):
        p_hat = rng.binomial(n, p, size=trials) / n
        b = max(p, 1 - p)                                   # |X_i - p| <= b for X_i in {0, 1}
        r_bern = np.sqrt(2 * p * (1 - p) * L / n) + 2 * b * L / (3 * n)
        fh = np.mean(np.abs(p_hat - p) >= r_hoeff)
        fb = np.mean(np.abs(p_hat - p) >= r_bern)
        print(f"{p:6.2f} | {r_hoeff:16.4f} {fh:9.4f} | {r_bern:16.4f} {fb:9.4f}")
    # Union bound: m arms, each needs its own interval to hold.
    m = 50
    p_hat = rng.binomial(n, 0.5, size=(trials // 10, m)) / n
    no_union = np.mean(np.all(np.abs(p_hat - 0.5) < r_hoeff, axis=1))
    r_union = np.sqrt(np.log(2 * m / delta) / (2 * n))
    with_union = np.mean(np.all(np.abs(p_hat - 0.5) < r_union, axis=1))
    print(f"union bound, m = {m} arms (p = 0.5): all {m} Hoeffding intervals hold simultaneously in "
          f"{no_union:.3f} of trials at per-arm delta = {delta} (radius {r_hoeff:.3f}); in {with_union:.4f} "
          f"with per-arm delta/m (radius {r_union:.3f})")


# ------------------------------------------------------------------------------------------------
# Part B
# ------------------------------------------------------------------------------------------------
def plug_in_fh(rng, P, r, N):
    """Generative-model plug-in: N fresh next-state samples for every (h, s, a); exact rewards."""
    H, S, A, _ = P.shape
    counts = np.zeros_like(P)
    for h in range(H):
        for s in range(S):
            for a in range(A):
                counts[h, s, a] = rng.multinomial(N, P[h, s, a])
    P_hat = counts / N
    Q_hat, _ = fh_optimal(P_hat, r)
    pi_hat = np.zeros_like(Q_hat)
    for h in range(H):
        pi_hat[h, np.arange(S), Q_hat[h].argmax(axis=1)] = 1.0
    return Q_hat, pi_hat


def part_b(rng, reps, Ns, Hs, delta=0.1):
    S, A, H0 = 5, 2, 10
    print(f"\nPart B: finite-horizon plug-in with a generative model, S = {S}, A = {A}, delta = {delta}, "
          f"{reps} repetitions per setting")
    out = {"N": Ns, "H0": H0}

    def run(H, N, P, r):
        Q_star, V_star = fh_optimal(P, r)
        q_err, loss = [], []
        for _ in range(reps):
            Q_hat, pi_hat = plug_in_fh(rng, P, r, N)
            _, V_pi = fh_eval(P, r, pi_hat)
            q_err.append(np.abs(Q_hat[0] - Q_star[0]).max())
            loss.append((V_star[0] - V_pi[0]).max())
        bq = H * (H - 1) / 2 * np.sqrt(np.log(2 * S * A * H / delta) / (2 * N))
        bl = H * (H - 1) * np.sqrt(np.log(4 * S * A * H / delta) / (2 * N))
        return np.array(q_err), np.array(loss), bq, bl

    P = rng.dirichlet(np.ones(S), size=(H0, S, A))
    r = rng.uniform(0, 1, size=(H0, S, A))
    print(f"(i) H = {H0}, varying N:")
    print(f"{'N':>7s} | {'mean max|Qhat-Q*|':>17s} {'95th pct':>9s} {'bound (19.18)':>13s} {'viol':>5s} | "
          f"{'mean loss':>9s} {'max loss':>9s} {'bound (19.19)':>13s} {'viol':>5s}")
    rows = []
    for N in Ns:
        q_err, loss, bq, bl = run(H0, N, P, r)
        rows.append((N, q_err.mean(), np.percentile(q_err, 95), bq, loss.mean(), loss.max(), bl))
        print(f"{N:7d} | {q_err.mean():17.4f} {np.percentile(q_err, 95):9.4f} {bq:13.3f} "
              f"{int(np.sum(q_err > bq)):5d} | {loss.mean():9.4f} {loss.max():9.4f} {bl:13.3f} {int(np.sum(loss > bl)):5d}")
    rows = np.array(rows)
    out["rows"] = rows
    print("    looseness: bound (19.18) / mean error = " + ", ".join(f"{b / m:.0f}" for m, b in rows[:, [1, 3]])
          + ";  bound (19.19) / max loss = " + ", ".join(f"{b / m:.0f}" for m, b in rows[:, [5, 6]]))
    print(f"    slope of mean max|Qhat-Q*| vs N (log-log): {loglog_slope(rows[:, 0], rows[:, 1]):+.3f}  (theory -1/2)")

    # (ii) H-scaling on the finite-horizon analogue of the hard instance (one action, so this is
    # policy evaluation): state x pays 1 per step and stays with probability p = 1 - 1/H, else the
    # episode moves to an absorbing state z that pays 0. Fresh samples for every step h.
    N_fixed = 1000
    print(f"(ii) N = {N_fixed} samples per (x, h), finite-horizon hard instance (p = 1 - 1/H), "
          f"{20 * reps:,d} repetitions:")
    hrows = []
    for H in Hs:
        p = 1.0 - 1.0 / H
        V = np.zeros(H + 1)                      # exact V_h(x)
        for h in range(H - 1, -1, -1):
            V[h] = 1.0 + p * V[h + 1]
        p_hat = rng.binomial(N_fixed, p, size=(20 * reps, H)) / N_fixed
        V_hat = np.zeros(20 * reps)
        for h in range(H - 1, -1, -1):           # backward induction in the estimated model
            V_hat = 1.0 + p_hat[:, h] * V_hat
        rmse = np.sqrt(np.mean((V_hat - V[0]) ** 2))
        # Exact sd of the first-order error sum_h Pr(S_h = x) (p_hat_h - p) V_{h+1}(x)  (cf. Eq. 19.20)
        sd_lin = np.sqrt(sum((p ** h) ** 2 * p * (1 - p) / N_fixed * V[h + 1] ** 2 for h in range(H)))
        bound = H * (H - 1) / 2 * np.sqrt(np.log(2 * H / delta) / (2 * N_fixed))
        hrows.append((H, rmse, bound, sd_lin))
        print(f"    H = {H:3d}: V_0(x) = {V[0]:6.2f}, RMSE of Vhat_0(x) = {rmse:.4f}, first-order prediction "
              f"= {sd_lin:.4f}, Hoeffding bound = {bound:.3f}, ratio bound/RMSE = {bound / rmse:.0f}")
    hrows = np.array(hrows)
    out["hrows"] = hrows
    print(f"    slope of RMSE vs H (log-log): {loglog_slope(hrows[:, 0], hrows[:, 1]):+.2f} (variance argument: +1); "
          f"Hoeffding bound slope {loglog_slope(hrows[:, 0], hrows[:, 2]):+.2f}")
    return out


# ------------------------------------------------------------------------------------------------
# Part C
# ------------------------------------------------------------------------------------------------
def part_c(rng, gammas, reps_hard, reps_rand, N_hard, N_rand):
    print(f"\nPart C: plug-in error vs effective horizon 1/(1-gamma)")
    print(f"(i) hard instance, N = {N_hard:,d} samples of the self-loop, {reps_hard:,d} repetitions")
    hard = []
    for g in gammas:
        p = (4 * g - 1) / (3 * g)
        v = 1.0 / (1.0 - g * p)
        p_hat = rng.binomial(N_hard, p, size=reps_hard) / N_hard
        v_hat = 1.0 / (1.0 - g * p_hat)
        rmse = np.sqrt(np.mean((v_hat - v) ** 2))
        pred = g * np.sqrt(p * (1 - p) / N_hard) / (1 - g * p) ** 2          # delta method (Eq. 19.20)
        hard.append((1 / (1 - g), rmse, pred))
        print(f"    gamma = {g:6.4f}: v(x) = {v:8.2f}, RMSE of vhat = {rmse:.4f}, delta-method prediction = {pred:.4f}")
    hard = np.array(hard)
    print(f"    fitted exponent of RMSE in 1/(1-gamma): {loglog_slope(hard[:, 0], hard[:, 1]):.3f} (theory 1.5)")

    S, A = 10, 2
    P, r = random_mdp(np.random.default_rng(1234), S, A, conc=1.0)
    print(f"(ii) random dense MDP (S = {S}, A = {A}), N = {N_rand} samples per (s,a), {reps_rand} repetitions")
    rand = []
    for g in gammas:
        q_star, v_star = value_iteration(P, r, g, tol=1e-10)
        errs = []
        for _ in range(reps_rand):
            counts = np.array([[rng.multinomial(N_rand, P[s, a]) for a in range(A)] for s in range(S)])
            q_hat, _ = value_iteration(counts / N_rand, r, g, tol=1e-10)
            errs.append(np.abs(q_hat - q_star).max())
        rmse = np.sqrt(np.mean(np.square(errs)))
        span = v_star.max() - v_star.min()
        rand.append((1 / (1 - g), rmse, span))
        print(f"    gamma = {g:6.4f}: ||q*||_inf = {np.abs(q_star).max():8.2f}, span(v*) = {span:6.3f}, "
              f"RMS of ||qhat* - q*||_inf = {rmse:.4f}")
    rand = np.array(rand)
    print(f"    fitted exponent of error in 1/(1-gamma): {loglog_slope(rand[:, 0], rand[:, 1]):.3f}")
    return hard, rand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figure")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    quick = args.quick
    trials = 20_000 if quick else 200_000
    reps_b = 5 if quick else 100
    Ns = [10, 100, 1000] if quick else [10, 30, 100, 300, 1000, 3000, 10000]
    Hs = [5, 10] if quick else [5, 10, 20, 40]
    gammas = [0.8, 0.9, 0.95] if quick else [0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
    print(f"seed={args.seed}  quick={quick}  Part A trials={trials}  Part B reps={reps_b} N={Ns} H={Hs}  "
          f"Part C gammas={gammas}")
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    part_a(rng, trials)
    out_b = part_b(rng, reps_b, Ns, Hs)
    hard, rand = part_c(rng, gammas, reps_hard=20_000 if quick else 200_000, reps_rand=3 if quick else 200,
                        N_hard=1_000_000, N_rand=1000)
    print(f"\ntotal time {time.time() - t0:.1f} s")
    if quick:
        return

    setup_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
    ax = axes[0]
    rows = out_b["rows"]
    ax.loglog(rows[:, 0], rows[:, 1], "o-", color=COLORS[0], label=r"mean $\max|\hat Q_0 - Q^\ast_0|$")
    ax.loglog(rows[:, 0], rows[:, 3], "--", color=COLORS[0], label="Hoeffding bound (19.18)")
    ax.loglog(rows[:, 0], np.maximum(rows[:, 5], 1e-6), "s-", color=COLORS[1],
              label=r"max over runs of $\max_s(V^\ast_0 - V^{\hat\pi}_0)$")
    ax.loglog(rows[:, 0], rows[:, 6], ":", color=COLORS[1], label="Hoeffding bound (19.19)")
    ax.set_xlabel("N = samples per (s, a, h)")
    ax.set_ylabel("error")
    ax.set_title(f"Toy theorem: plug-in planning, H = {out_b['H0']}, S = 5, A = 2")
    ax.legend(fontsize=7.8, loc="lower left")

    ax = axes[1]
    hrows = out_b["hrows"]
    ax.loglog(hrows[:, 0], hrows[:, 1], "o-", color=COLORS[3], label=r"RMSE of $\hat V_0(x)$ (measured)")
    ax.loglog(hrows[:, 0], hrows[:, 3], "k:", lw=1, label="first-order (variance) prediction")
    ax.loglog(hrows[:, 0], hrows[:, 2], "--", color=COLORS[3], label=r"Hoeffding bound $\propto H^2$")
    ax.set_xlabel("horizon H")
    ax.set_xticks(hrows[:, 0])
    ax.set_xticklabels([str(int(h)) for h in hrows[:, 0]])
    ax.minorticks_off()
    ax.set_ylabel("error")
    ax.set_title("Hard instance, N = 1000 per (x, h): error ∝ H, bound ∝ H²")
    ax.legend(fontsize=7.8, loc="upper left")

    ax = axes[2]
    x = hard[:, 0]
    sqN_h, sqN_r = np.sqrt(1_000_000), np.sqrt(1000)
    ax.loglog(x, hard[:, 1] * sqN_h, "o-", color=COLORS[2], label="hard instance (Azar et al.)")
    ax.loglog(x, hard[:, 2] * sqN_h, "k:", lw=1, label="delta-method prediction")
    ax.loglog(rand[:, 0], rand[:, 1] * sqN_r, "s-", color=COLORS[4], label="random dense MDP, S=10, A=2")
    ref = hard[0, 1] * sqN_h
    ax.loglog(x, ref * (x / x[0]) ** 1.5, "--", color=GREY_LINE, lw=1, label=r"slope 1.5: $(1-\gamma)^{-3/2}$")
    ax.loglog(x, ref * (x / x[0]) ** 2, "-.", color=GREY_LINE, lw=1, label=r"slope 2: $(1-\gamma)^{-2}$ (Hoeffding)")
    ax.set_xlabel(r"effective horizon $1/(1-\gamma)$")
    ax.set_ylabel(r"$\sqrt{N}\,\times$ RMS error")
    ax.set_title("Hard instance: error ∝ (1−γ)$^{-3/2}$; random MDP: much flatter")
    ax.legend(fontsize=7.8, loc="upper left")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "generative_model.png")
    fig.savefig(out, dpi=110)
    print("saved", out)


GREY_LINE = "#555555"

if __name__ == "__main__":
    main()
