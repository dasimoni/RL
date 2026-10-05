"""Regret of exploration strategies in small episodic MDPs. Chapter 19, Sections 6-7 (experiment: Section 7.6).

Agents (all tabular, time-inhomogeneous tables Q_h(s,a), h = 0..H-1, as in Jin et al. 2018):
  * eps-greedy Q-learning, constant eps = 0.1      (no optimism; Q_0 = 0)
  * eps-greedy Q-learning, decaying eps_k = 1/sqrt(k) (GLIE schedule)
  * UCB-Hoeffding Q-learning (Algorithm 19.3): greedy w.r.t. an optimistic Q, Q_0 = H - h,
        alpha_t = (H+1)/(H+t),  bonus b_t = c * sqrt(H^3 * iota / t),  iota = ln(S A T / p)
    with c = 0.1 and c = 1 (Jin et al. prove the bound for "some absolute constant c")
  * UCBVI-Hoeffding (model-based, Algorithm 19.2): optimistic backward induction in the empirical
    model with bonus c * (H - h) * sqrt(iota / n), c = 0.1, recomputed before every episode.
All Q-learning agents use the step size (H+1)/(H+t); only the exploration mechanism differs.

Regret is computed EXACTLY: before episode k each agent's policy pi_k is fixed (Q_h is only updated
after step h has been taken, and step h is visited once per episode), so we evaluate pi_k by
backward induction in the true MDP and add E_{s0~d0}[V*_0(s0) - V^{pi_k}_0(s0)] to the regret.

Environments:
  * "random": S = 5, A = 3, H = 5, transitions Dirichlet(0.5), Bernoulli rewards with means U[0,1].
  * "lock": a combination lock, S = 7, A = 2, H = 9. In lock states 0..5 one (random) action
    advances (reward 0), the other returns to state 0 with a Bernoulli "distractor" reward of mean
    0.05. State 6 is absorbing and pays 1 per step. Optimal value 3.0. All rewards are Bernoulli
    with the stated means.

Lock diagnostics (printed for every agent): in how many episodes each run reached the goal, the
first such episode, the exact probability that the episode policy pi_k reaches the goal (forward
dynamic programming in the true MDP) at k = 100, 1000 and K, and, for the Q-learners, the fraction
of reachable lock entries (h, s), h >= s, whose two actions are still exactly tied at the end
(ties are broken at random afresh in every episode).

Run from the repository root:
  python code/ch19_rl_theory/regret_experiment.py           # full run (~3 min), writes the figure
  python code/ch19_rl_theory/regret_experiment.py --quick   # smoke test, no figure
  python code/ch19_rl_theory/regret_experiment.py --lock-seed-check   # only the two eps-greedy agents
        on the lock, for seeds 0, 1, 2 (different locks and random streams), ~2 min, no figure
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from theory_lib import COLORS, fh_optimal, loglog_slope, setup_style

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")


# ------------------------------------------------------------------------------------------------
# Environments: (P (H,S,A,S), mean reward (H,S,A), d0 (S,), H)
# ------------------------------------------------------------------------------------------------
def make_random_env(seed, S=5, A=3, H=5):
    rng = np.random.default_rng(seed)
    P1 = rng.dirichlet(np.full(S, 0.5), size=(S, A))
    r1 = rng.uniform(0, 1, size=(S, A))
    P = np.repeat(P1[None], H, axis=0)
    r = np.repeat(r1[None], H, axis=0)
    d0 = np.full(S, 1.0 / S)
    return P, r, d0, H


def make_lock_env(seed, L=6, H=9, distractor=0.05):
    rng = np.random.default_rng(seed)
    S, A = L + 1, 2
    correct = rng.integers(0, A, size=L)
    P1 = np.zeros((S, A, S))
    r1 = np.zeros((S, A))
    for s in range(L):
        for a in range(A):
            if a == correct[s]:
                P1[s, a, s + 1] = 1.0
            else:
                P1[s, a, 0] = 1.0
                r1[s, a] = distractor
    P1[L, :, L] = 1.0
    r1[L, :] = 1.0
    P = np.repeat(P1[None], H, axis=0)
    r = np.repeat(r1[None], H, axis=0)
    d0 = np.zeros(S)
    d0[0] = 1.0
    return P, r, d0, H


# ------------------------------------------------------------------------------------------------
# Exact evaluation of the episode policy for every run at once
# ------------------------------------------------------------------------------------------------
def eval_policies(P, r, d0, greedy_a, eps):
    """greedy_a (R, H, S) greedy actions; eps (R,) exploration probability. Returns E_{d0} V^pi_0 (R,)."""
    H, S, A, _ = P.shape
    R = greedy_a.shape[0]
    V = np.zeros((R, S))
    rr, ss = np.arange(R)[:, None], np.arange(S)[None, :]
    w_greedy, w_unif = (1 - eps)[:, None], (eps / A)[:, None]
    for h in range(H - 1, -1, -1):
        Qh = r[h][None] + np.einsum("sat,rt->rsa", P[h], V)            # (R, S, A)
        # V^pi_h(s) = (1 - eps) Q_h(s, greedy) + eps * mean_a Q_h(s, a)
        V = w_greedy * Qh[rr, ss, greedy_a[:, h]] + w_unif * Qh.sum(axis=2)
    return V @ d0


def argmax_random_ties(x, rng):
    """argmax over the last axis, breaking ties uniformly at random."""
    noise = rng.random(x.shape) * 1e-9
    return np.argmax(x + noise, axis=-1)


def reach_prob(P, d0, greedy_a, eps, target):
    """Exact probability that the episode policy (greedy actions greedy_a (R, H, S), exploration
    eps (R,)) is in state `target` after H steps: forward dynamic programming in the true MDP.
    For an absorbing goal this is the probability that the episode reaches the goal."""
    H, S, A, _ = P.shape
    R = greedy_a.shape[0]
    dist = np.broadcast_to(d0, (R, S)).copy()
    rr, ss = np.arange(R)[:, None], np.arange(S)[None, :]
    for h in range(H):
        pi = np.broadcast_to((eps / A)[:, None, None], (R, S, A)).copy()
        pi[rr, ss, greedy_a[:, h]] += 1.0 - eps[:, None]
        dist = np.einsum("rs,rsa,sat->rt", dist, pi, P[h])
    return dist[:, target]


def new_diag(R, goal, K):
    probe = sorted({k for k in (100, 1000, K) if k <= K})
    return {"goal": goal, "hits": np.zeros(R, dtype=np.int64), "first_hit": np.full(R, -1),
            "probe_k": probe, "p_goal": {}, "both_tried": np.full(R, -1)}


def record_both_tried(diag, k, counts_start):
    """counts_start (R, A): visit counts of the start state at step h = 0. Records the first episode
    by the end of which every action had been tried there."""
    if diag["goal"] is None:
        return
    newly = (diag["both_tried"] < 0) & np.all(counts_start > 0, axis=1)
    diag["both_tried"][newly] = k


def record_episode(diag, k, reached, P, d0, greedy_a, eps):
    """Update the lock diagnostics after episode k (no effect on the random stream)."""
    if diag["goal"] is None:
        return
    diag["hits"] += reached
    diag["first_hit"][(diag["first_hit"] < 0) & reached] = k
    if k in diag["probe_k"]:
        diag["p_goal"][k] = reach_prob(P, d0, greedy_a, eps, diag["goal"])


# ------------------------------------------------------------------------------------------------
# Agents
# ------------------------------------------------------------------------------------------------
def run_q_learning(P, r, d0, H, K, R, rng, mode, c=0.0, p_fail=0.05, goal=None):
    """mode: 'eps-const', 'eps-decay' or 'ucb-h'. Returns per-episode expected regret (R, K) and a dict
    of lock diagnostics (empty if goal is None).

    The eps-greedy agents are Algorithm 19.3 with c = 0 (no bonus), Q_0 = 0 instead of H - h, and
    eps-greedy action selection around the greedy action."""
    _, S, A, _ = P.shape
    _, V_star = fh_optimal(P, r)
    v_star = V_star[0] @ d0
    T = K * H
    iota = np.log(S * A * T / p_fail)
    remaining = (H - np.arange(H)).astype(float)                      # H - h
    if mode == "ucb-h":
        Q = np.broadcast_to(remaining[None, :, None, None], (R, H, S, A)).copy()   # optimistic init
    else:
        Q = np.zeros((R, H, S, A))
    N = np.zeros((R, H, S, A), dtype=np.int64)
    cumP = np.cumsum(P, axis=-1)
    cumP[..., -1] = 1.0
    cumd0 = np.cumsum(d0)
    cumd0[-1] = 1.0
    runs = np.arange(R)
    regret = np.zeros((R, K))
    diag = new_diag(R, goal, K)
    for k in range(1, K + 1):
        if mode == "eps-const":
            eps = np.full(R, 0.1)
        elif mode == "eps-decay":
            eps = np.full(R, 1.0 / np.sqrt(k))
        else:
            eps = np.zeros(R)
        greedy_a = argmax_random_ties(Q, rng)                          # (R, H, S): policy pi_k
        regret[:, k - 1] = v_star - eval_policies(P, r, d0, greedy_a, eps)
        s = (rng.random(R)[:, None] > cumd0[None]).sum(axis=1)
        reached = np.zeros(R, dtype=bool)
        for h in range(H):
            explore = rng.random(R) < eps
            a = np.where(explore, rng.integers(0, A, size=R), greedy_a[runs, h, s])
            rew = (rng.random(R) < r[h, s, a]).astype(float)
            s_next = (rng.random(R)[:, None] > cumP[h, s, a]).sum(axis=1)
            N[runs, h, s, a] += 1
            t = N[runs, h, s, a]
            alpha = (H + 1.0) / (H + t)
            if h + 1 < H:
                v_next = np.minimum(remaining[h + 1], Q[runs, h + 1, s_next].max(axis=1))
            else:
                v_next = np.zeros(R)
            bonus = c * np.sqrt(H ** 3 * iota / t) if mode == "ucb-h" else 0.0
            Q[runs, h, s, a] = (1 - alpha) * Q[runs, h, s, a] + alpha * (rew + v_next + bonus)
            if goal is not None:
                reached |= s_next == goal
            s = s_next
        record_episode(diag, k, reached, P, d0, greedy_a, eps)
        record_both_tried(diag, k, N[:, 0, int(np.argmax(d0))])
    if goal is not None:
        # Fraction of reachable lock entries (h, s), s < goal and h >= s, whose actions are exactly tied.
        mask = np.array([[h >= st for st in range(goal)] for h in range(H)])          # (H, goal)
        Ql = Q[:, :, :goal, :]
        tied = (Ql == Ql.max(axis=-1, keepdims=True)).sum(axis=-1) > 1                 # (R, H, goal)
        diag["tied_frac"] = tied[:, mask].mean(axis=1)
    return regret, diag


def run_ucbvi(P, r, d0, H, K, R, rng, c=1.0, p_fail=0.05, goal=None):
    """Model-based optimism: plan in the empirical model with a Hoeffding bonus every episode."""
    _, S, A, _ = P.shape
    _, V_star = fh_optimal(P, r)
    v_star = V_star[0] @ d0
    iota = np.log(S * A * K * H / p_fail)
    n = np.zeros((R, H, S, A))
    n_next = np.zeros((R, H, S, A, S))
    rsum = np.zeros((R, H, S, A))
    cumP = np.cumsum(P, axis=-1)
    cumP[..., -1] = 1.0
    cumd0 = np.cumsum(d0)
    cumd0[-1] = 1.0
    runs = np.arange(R)
    regret = np.zeros((R, K))
    diag = new_diag(R, goal, K)
    for k in range(1, K + 1):
        # Optimistic backward induction (Algorithm 19.2).
        Qt = np.zeros((R, H, S, A))
        V = np.zeros((R, S))
        for h in range(H - 1, -1, -1):
            nh = np.maximum(n[:, h], 1)
            P_hat = n_next[:, h] / nh[..., None]
            r_hat = rsum[:, h] / nh
            bonus = c * (H - h) * np.sqrt(iota / nh)
            q = r_hat + np.einsum("rsat,rt->rsa", P_hat, V) + bonus
            q = np.where(n[:, h] > 0, np.minimum(q, H - h), H - h)
            Qt[:, h] = q
            V = q.max(axis=2)
        greedy_a = argmax_random_ties(Qt, rng)
        regret[:, k - 1] = v_star - eval_policies(P, r, d0, greedy_a, np.zeros(R))
        s = (rng.random(R)[:, None] > cumd0[None]).sum(axis=1)
        reached = np.zeros(R, dtype=bool)
        for h in range(H):
            a = greedy_a[runs, h, s]
            rew = (rng.random(R) < r[h, s, a]).astype(float)
            s_next = (rng.random(R)[:, None] > cumP[h, s, a]).sum(axis=1)
            n[runs, h, s, a] += 1
            n_next[runs, h, s, a, s_next] += 1
            rsum[runs, h, s, a] += rew
            if goal is not None:
                reached |= s_next == goal
            s = s_next
        record_episode(diag, k, reached, P, d0, greedy_a, np.zeros(R))
        record_both_tried(diag, k, n[:, 0, int(np.argmax(d0))])
    return regret, diag


AGENTS = [
    ("ε-greedy Q, ε = 0.1", lambda P, r, d0, H, K, R, rng, g: run_q_learning(P, r, d0, H, K, R, rng, "eps-const", goal=g)),
    ("ε-greedy Q, ε = 1/√k", lambda P, r, d0, H, K, R, rng, g: run_q_learning(P, r, d0, H, K, R, rng, "eps-decay", goal=g)),
    ("UCB-H Q-learning, c = 1", lambda P, r, d0, H, K, R, rng, g: run_q_learning(P, r, d0, H, K, R, rng, "ucb-h", c=1.0, goal=g)),
    ("UCB-H Q-learning, c = 0.1", lambda P, r, d0, H, K, R, rng, g: run_q_learning(P, r, d0, H, K, R, rng, "ucb-h", c=0.1, goal=g)),
    ("UCBVI-Hoeffding, c = 0.1", lambda P, r, d0, H, K, R, rng, g: run_ucbvi(P, r, d0, H, K, R, rng, c=0.1, goal=g)),
]


def print_lock_diag(diag, reg, v0):
    """One block of lock diagnostics for one agent."""
    hits, first = diag["hits"], diag["first_hit"]
    K = reg.shape[1]
    late = reg[:, int(0.9 * K):].mean(axis=1)
    print(f"{'':>28s}goal reached in >= 1 episode in {int(np.sum(hits > 0))}/{len(hits)} runs; "
          f"goal episodes per run {hits.tolist()}; first goal episode per run {first.tolist()}")
    pg = "  ".join(f"k={k}: {diag['p_goal'][k].mean():.2e}" for k in diag["probe_k"])
    print(f"{'':>28s}exact P(pi_k reaches goal), mean over runs: {pg}; "
          f"runs with late regret/episode < V*/2: {int(np.sum(late < v0 / 2))}/{len(late)}")
    print(f"{'':>28s}first episode by which every action had been tried in the start state at h = 0: "
          f"{diag['both_tried'].tolist()}")
    if "tied_frac" in diag:
        tf = diag["tied_frac"]
        print(f"{'':>28s}tied reachable lock entries (h >= s) at the end: mean {tf.mean():.2f} "
              f"(range {tf.min():.2f}-{tf.max():.2f})")


def lock_seed_check(K, R, seeds=(0, 1, 2)):
    """The two eps-greedy agents on the lock, for several seeds (lock layout and random stream)."""
    print(f"lock seed check: eps-greedy agents only, K = {K:,d}, {R} runs per seed, seeds {list(seeds)}")
    n_learned, n_total = 0, 0
    for seed in seeds:
        P, r, d0, H = make_lock_env(seed + 3)
        goal = P.shape[1] - 1
        v0 = fh_optimal(P, r)[1][0] @ d0
        for i, (name, fn) in enumerate(AGENTS[:2]):
            t0 = time.time()
            reg, diag = fn(P, r, d0, H, K, R, np.random.default_rng(1000 * seed + 17 * i + 1), goal)
            late = reg[:, int(0.9 * K):].mean(axis=1)
            learned = int(np.sum(late < v0 / 2))
            n_learned += learned
            n_total += R
            print(f"  seed {seed}, {name:<20s}: Reg(K) = {reg.sum(axis=1).mean():9.1f}; runs reaching the goal "
                  f"{int(np.sum(diag['hits'] > 0))}/{R}, goal episodes per run {diag['hits'].tolist()}; "
                  f"runs that learned the lock (late regret/episode < V*/2): {learned}/{R}  ({time.time() - t0:.1f} s)")
    print(f"  in total {n_learned} of {n_total} eps-greedy runs learned the lock")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--lock-seed-check", action="store_true",
                    help="run only the eps-greedy agents on the lock for seeds 0, 1, 2 (no figure)")
    args = ap.parse_args()
    K = args.episodes or (300 if args.quick else 30_000)
    R = args.runs or (2 if args.quick else 10)
    if args.lock_seed_check:
        t0 = time.time()
        lock_seed_check(K, R)
        print(f"total time {time.time() - t0:.1f} s")
        return
    envs = {"random": make_random_env(args.seed + 7), "lock": make_lock_env(args.seed + 3)}
    print(f"seed={args.seed}  episodes K={K:,d}  runs={R}  (regret = exact expected per-episode regret, summed)")
    t_all = time.time()
    results = {}
    for env_name, (P, r, d0, H) in envs.items():
        _, V_star = fh_optimal(P, r)
        print(f"\n=== env '{env_name}': H = {H}, S = {P.shape[1]}, A = {P.shape[2]}, "
              f"E V*_0 = {V_star[0] @ d0:.4f}, T = KH = {K * H:,d} steps")
        print("    Reg(K)/Reg(K/2) is 2 for linear regret, 1.41 for sqrt(T) regret, about 1 for log(T) regret")
        results[env_name] = {}
        goal = P.shape[1] - 1 if env_name == "lock" else None
        for i, (name, fn) in enumerate(AGENTS):
            t0 = time.time()
            reg, diag = fn(P, r, d0, H, K, R, np.random.default_rng(1000 * args.seed + 17 * i + 1), goal)
            cum = np.cumsum(reg, axis=1)
            results[env_name][name] = reg
            m = cum.mean(axis=0)
            ratio = m[-1] / m[K // 2 - 1]
            ks = np.arange(1, K + 1)
            last = ks >= K / 10
            slope = loglog_slope(ks[last] * H, m[last])
            per_ep = [reg[:, a:b].mean() for a, b in ((K // 100, K // 50), (K // 10, K // 5), (int(0.9 * K), K))]
            v0 = V_star[0] @ d0
            below = np.flatnonzero(reg.mean(axis=0) < 0.5 * v0)
            first_half = int(below[0]) + 1 if below.size else None
            print(f"{name:>26s}: Reg(K) = {m[-1]:9.1f} ± {cum[:, -1].std(ddof=1) / np.sqrt(R):6.1f}"
                  f" | Reg(K)/Reg(K/2) = {ratio:4.2f} | log-log slope, last decade {slope:+.2f}"
                  f" | regret/episode at k~K/100, K/10, K: {per_ep[0]:.4f}, {per_ep[1]:.4f}, {per_ep[2]:.4f}"
                  f" | first k with mean regret < V*/2: {first_half}"
                  f"  ({time.time() - t0:.1f} s)")
            if goal is not None:
                print_lock_diag(diag, reg, v0)
    print(f"\ntotal time {time.time() - t_all:.1f} s")
    if args.quick:
        return

    setup_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    styles = ["-", "--", "-", "-.", ":"]
    for col, (env_name, (P, r, d0, H)) in enumerate(envs.items()):
        ks = np.arange(1, K + 1)
        T = ks * H
        ax = axes[0, col]
        for i, (name, _) in enumerate(AGENTS):
            cum = np.cumsum(results[env_name][name], axis=1)
            lo, hi = np.percentile(cum, [10, 90], axis=0)
            ax.loglog(T, cum.mean(axis=0), styles[i], color=COLORS[i], label=name)
            ax.fill_between(T, np.maximum(lo, 1e-3), hi, color=COLORS[i], alpha=0.12, lw=0)
        ref_T = T[T >= 100]
        top = max(np.cumsum(results[env_name][n], axis=1).mean(axis=0)[-1] for n, _ in AGENTS)
        ax.loglog(ref_T, top * ref_T / ref_T[-1], color="#444444", lw=0.9, ls=(0, (4, 2)), label="slope 1 (linear)")
        ax.loglog(ref_T, 0.05 * top * np.sqrt(ref_T / ref_T[-1]), color="#444444", lw=0.9, ls=(0, (1, 1.5)),
                  label="slope 1/2 (√T)")
        ax.set_xlabel("T = total steps (K episodes × H)")
        ax.set_ylabel("cumulative regret")
        ax.set_title(f"'{env_name}' MDP (S={P.shape[1]}, A={P.shape[2]}, H={H}), {R} runs")
        ax.set_ylim(bottom=1e-1)
        # Bottom row: per-episode regret averaged over log-spaced blocks of episodes.
        ax = axes[1, col]
        edges = np.unique(np.round(np.logspace(0, np.log10(K), 45)).astype(int))
        centers = np.sqrt(edges[:-1] * np.maximum(edges[1:] - 1, edges[:-1]))
        for i, (name, _) in enumerate(AGENTS):
            reg = results[env_name][name].mean(axis=0)
            vals = np.array([reg[a - 1:b - 1].mean() for a, b in zip(edges[:-1], edges[1:])])
            ax.loglog(centers, np.maximum(vals, 1e-5), styles[i], color=COLORS[i], label=name)
        kk = centers[centers >= 30]
        ax.loglog(kk, 2.0 * np.sqrt(30 / kk), color="#444444", lw=0.9, ls=(0, (1, 1.5)), label=r"slope $-1/2$")
        ax.set_xlabel("episode k")
        ax.set_ylabel(r"expected regret in episode k, $V^\ast_0 - V^{\pi_k}_0$")
        ax.set_title("per-episode regret: flat = linear, slope −1/2 = √T")
        ax.set_ylim(1e-3, 5)
    axes[0, 0].legend(fontsize=7.6, loc="upper left")
    axes[1, 0].legend(fontsize=7.6, loc="lower left")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "regret_loglog.png")
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
