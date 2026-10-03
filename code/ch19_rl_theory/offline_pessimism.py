"""Offline RL theory in miniature: coverage, concentrability and pessimism. Chapter 19, Section 10.

We learn from a fixed dataset of N i.i.d. transitions (s, a, r, s') drawn as
    s ~ Uniform(S),  a ~ pi_b(.|s) = (1 - lam) * pi*(.|s) + lam * Uniform(A),  r ~ Bernoulli(r(s,a)),
    s' ~ p(.|s,a)
on a random discounted MDP, and compare three tabular learners:
  * plug-in (certainty equivalence): value iteration in the empirical model, restricted to the
    state-action pairs that appear in the data;
  * VI-LCB (pessimism, Algorithm 19.6): the same, minus a penalty
        b(s,a) = c * (1/(1-gamma)) * sqrt(ln(S A N / delta) / n(s,a)),   c = 0.1 (plus a sweep over c),
    with V clipped below at 0 (Rashidinejad et al. 2021; Jin, Yang & Wang 2021 for the episodic PEVI);
  * behaviour cloning (BC): the most frequent action in each state.
Two data regimes: lam = 1 (uniform behaviour: every policy is covered) and lam = 0.1 (mostly
expert: only pi* is well covered). We print the single-policy concentrability
    C* = max_{s,a} d^{pi*}_rho(s,a) / nu_D(s,a)
and the all-policy concentrability C_all = max_{s,a} max_pi d^pi_rho(s,a) / nu_D(s,a), and the mean
suboptimality V*(rho) - V^{pihat}(rho) (rho uniform) of each learner versus N.

Run from the repository root:
  python code/ch19_rl_theory/offline_pessimism.py           # full run (< 1 min), writes the figure
  python code/ch19_rl_theory/offline_pessimism.py --quick   # smoke test, no figure
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from theory_lib import COLORS, greedy, occupancy, policy_eval, random_mdp, setup_style, value_iteration

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")


def sample_dataset(rng, P, r, pi_b, N):
    S, A, _ = P.shape
    s = rng.integers(0, S, size=N)
    cum_b = np.cumsum(pi_b, axis=1)
    a = (rng.random(N)[:, None] > cum_b[s]).sum(axis=1)
    a = np.minimum(a, A - 1)
    rew = (rng.random(N) < r[s, a]).astype(float)
    cumP = np.cumsum(P, axis=-1)
    s2 = (rng.random(N)[:, None] > cumP[s, a]).sum(axis=1)
    s2 = np.minimum(s2, S - 1)
    return s, a, rew, s2


def empirical_model(S, A, data):
    s, a, rew, s2 = data
    n = np.zeros((S, A))
    np.add.at(n, (s, a), 1)
    rsum = np.zeros((S, A))
    np.add.at(rsum, (s, a), rew)
    cnt = np.zeros((S, A, S))
    np.add.at(cnt, (s, a, s2), 1)
    nn = np.maximum(n, 1)
    return n, rsum / nn, cnt / nn[..., None]


def lcb_vi(n, r_hat, P_hat, gamma, penalty, iters=5000):
    """Pessimistic value iteration (Algorithm 19.6):
        Q(s,a) = r_hat(s,a) + gamma * P_hat(.|s,a) V - penalty(s,a)   for pairs seen in the data,
        V(s)   = max(0, max_{a seen} Q(s,a))   (true values are >= 0, so clipping at 0 keeps V a lower bound).
    Pairs never seen get Q = -inf (never chosen unless a state has no data, where V = 0).
    With penalty = 0 this is the plug-in (certainty-equivalence) solution restricted to seen pairs."""
    S, A = n.shape
    V = np.zeros(S)
    for _ in range(iters):
        Q = np.where(n > 0, r_hat + gamma * P_hat @ V - penalty, -np.inf)
        V_new = np.maximum(0.0, Q.max(axis=1))
        if np.max(np.abs(V_new - V)) < 1e-10:
            V = V_new
            break
        V = V_new
    return np.where(n > 0, r_hat + gamma * P_hat @ V - penalty, -np.inf)


def max_occupancy(P, gamma, rho):
    """max over policies of d^pi_rho(s, a), for every (s, a): an MDP with reward 1[(s,a)]."""
    S, A, _ = P.shape
    out = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            rew = np.zeros((S, A))
            rew[s, a] = 1.0
            _, v = value_iteration(P, rew, gamma, tol=1e-10)
            out[s, a] = (1 - gamma) * rho @ v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figure")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    S, A, gamma, delta, c = 10, 4, 0.9, 0.1, 0.1
    Ns = [100, 300, 1000] if args.quick else [100, 200, 500, 1000, 2000, 5000, 10000, 30000]
    reps = 5 if args.quick else 100
    lams = [1.0, 0.1]
    print(f"seed={args.seed}  S={S} A={A} gamma={gamma}  LCB penalty constant c={c}, delta={delta}  "
          f"N={Ns}  datasets per point={reps}")
    rng = np.random.default_rng(args.seed)
    P, r = random_mdp(rng, S, A, conc=0.5, branching=3)
    q_star, v_star = value_iteration(P, r, gamma)
    pi_star = greedy(q_star)
    rho = np.full(S, 1.0 / S)
    J_star = rho @ v_star
    unif = np.full((S, A), 1.0 / A)
    J_unif = rho @ policy_eval(P, r, gamma, unif)[0]
    d_star_sa = occupancy(P, gamma, pi_star, rho)[:, None] * pi_star
    d_max_sa = max_occupancy(P, gamma, rho)
    print(f"J* = {J_star:.4f}, J(uniform policy) = {J_unif:.4f}")
    t0 = time.time()
    results = {}
    for lam in lams:
        pi_b = (1 - lam) * pi_star + lam * unif
        nu_D = pi_b / S                                         # data distribution over (s, a)
        C_star = np.max(d_star_sa / nu_D)
        C_all = np.max(d_max_sa / nu_D)
        J_b = rho @ policy_eval(P, r, gamma, pi_b)[0]
        print(f"\n=== behaviour lam = {lam}: J(pi_b) = {J_b:.4f}, single-policy C* = {C_star:.2f}, "
              f"all-policy C_all = {C_all:.2f}")
        print(f"{'N':>7s} | {'plug-in':>16s} | {'VI-LCB':>16s} | {'BC':>16s}    (mean suboptimality ± s.e.)")
        rows = []
        for N in Ns:
            subs = {"plug-in": [], "VI-LCB": [], "BC": []}
            for _ in range(reps):
                data = sample_dataset(rng, P, r, pi_b, N)
                n, r_hat, P_hat = empirical_model(S, A, data)
                iota = np.log(S * A * N / delta)
                Q_plug = lcb_vi(n, r_hat, P_hat, gamma, 0.0)
                pen = c / (1 - gamma) * np.sqrt(iota / np.maximum(n, 1))
                Q_lcb = lcb_vi(n, r_hat, P_hat, gamma, pen)
                # BC: most frequent action per state (random tie-break; unseen state -> uniform).
                noise = rng.random((S, A)) * 1e-6
                pi_bc = greedy(n + noise)
                for name, pi in (("plug-in", greedy(Q_plug + noise)), ("VI-LCB", greedy(Q_lcb + noise)), ("BC", pi_bc)):
                    subs[name].append(max(0.0, J_star - rho @ policy_eval(P, r, gamma, pi)[0]))
            row = [N]
            txt = []
            for name in ("plug-in", "VI-LCB", "BC"):
                arr = np.array(subs[name])
                row += [arr.mean(), arr.std(ddof=1) / np.sqrt(len(arr))]
                txt.append(f"{arr.mean():7.4f} ± {arr.std(ddof=1) / np.sqrt(len(arr)):6.4f}")
            rows.append(row)
            print(f"{N:7d} | " + " | ".join(txt))
        results[lam] = (np.array(rows), C_star, C_all)
        # Sensitivity to the penalty constant c at one dataset size.
        N_sweep = 1000
        sweep = []
        for c_sw in (0.0, 0.03, 0.1, 0.3, 1.0):
            subs = []
            for _ in range(reps):
                n, r_hat, P_hat = empirical_model(S, A, sample_dataset(rng, P, r, pi_b, N_sweep))
                pen = c_sw / (1 - gamma) * np.sqrt(np.log(S * A * N_sweep / delta) / np.maximum(n, 1))
                pi = greedy(lcb_vi(n, r_hat, P_hat, gamma, pen) + rng.random((S, A)) * 1e-6)
                subs.append(max(0.0, J_star - rho @ policy_eval(P, r, gamma, pi)[0]))
            sweep.append(f"c={c_sw}: {np.mean(subs):.4f}")
        print(f"    penalty sweep at N = {N_sweep} (c = 0 is plug-in): " + ", ".join(sweep))
    print(f"\ntotal time {time.time() - t0:.1f} s")
    if args.quick:
        return

    setup_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, lam in zip(axes, lams):
        rows, C_star, C_all = results[lam]
        for i, (name, mk) in enumerate((("plug-in", "o-"), ("VI-LCB", "s-"), ("BC", "^--"))):
            m, se = rows[:, 1 + 2 * i], rows[:, 2 + 2 * i]
            ax.loglog(rows[:, 0], np.maximum(m, 1e-4), mk, color=COLORS[i], label=name)
            ax.fill_between(rows[:, 0], np.maximum(m - 2 * se, 1e-4), m + 2 * se, color=COLORS[i], alpha=0.15, lw=0)
        if np.all(rows[:, 3] == 0) and np.all(rows[:, 5] == 0):
            ax.text(0.5, 0.06, "VI-LCB and BC: suboptimality exactly 0 in every dataset\n(plotted at the floor 1e-4)",
                    transform=ax.transAxes, ha="center", fontsize=8.5, color="#444444")
        ax.set_xlabel("dataset size N (transitions)")
        title = "uniform behaviour (λ = 1)" if lam == 1.0 else f"mostly-expert behaviour (λ = {lam})"
        ax.set_title(f"{title}\nC* = {C_star:.1f}, C_all = {C_all:.0f}")
    axes[0].set_ylabel(r"suboptimality $V^\ast(\rho) - V^{\hat\pi}(\rho)$")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "offline_pessimism.png")
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
