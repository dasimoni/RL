"""RiverSwim: why a stochastic world needs a decaying bonus, not just optimistic init.

Chapter 14, Sections 3-4 (results reported in Section 5.3).

A RiverSwim-style chain (after Strehl & Littman, 2008): 6 states in a river.  "Left"
(downstream) always works and pays 0.005 in the leftmost state s0.  "Right" (upstream)
succeeds with probability 0.35 (0.4 from s0), leaves the agent in place with probability 0.6,
and pushes it back with probability 0.05 (0.4 from s5).  Taking "right" in the rightmost
state s5 pays 1.  Episodes start in s0 and last H = 20 steps; we use time-indexed values
Q_h(s, a) as in the finite-horizon analysis of Jin et al. (2018).

Agents (finite-horizon tabular methods, updated at the end of each episode):
  eps-greedy Q          model-free Q-learning, Q0 = 0, eps = 0.1           (dithering)
  UCB-Q                 model-free Q-learning, Q0 = H, bonus beta/sqrt(n),
                        alpha_n = (H+1)/(H+n) (Jin et al. 2018 shape)       (Sec. 3.4)
  optimistic CE         model-based: plan in the empirical MDP; a pair never tried is
                        worth H - h (R-MAX with m = 1).  Optimism only for the UNTRIED.
  UCBVI-style           model-based: plan in the empirical MDP with bonus beta/sqrt(n),
                        values clipped at H - h (UCBVI / MBIE-EB shape)    (Sec. 3.3)
                        It assumes TIME-HOMOGENEOUS dynamics and pools every transition
                        of (s, a) into one model, whatever the step h.
  UCBVI-style, time-indexed   the same planner with separate counts n_h(s, a, s') for
                        each step h, as in the Sec. 3.3 analysis.  Comparing it with the
                        pooled version and with UCB-Q (which also keeps separate Q_h and
                        n_h) separates "model vs no model" from "pooled vs per-step data".
  PSRL                  posterior sampling: Dirichlet transitions, Gaussian rewards
                                                                           (Sec. 4.1)
The model-based agents know nothing but the state and action sets and the reward range
[0, 1]; they share one planner (backward induction), so they differ only in how they
treat uncertainty (and, for the time-indexed variant, in how they pool data).

Pitfall demo (full mode): UCB-Q with an INTEGER Q-table (np.full(shape, H) with an int H),
which silently truncates every update -- the bug of Pitfall 7.

We report the cumulative (pseudo-)regret: sum over episodes of V*_0(s0) - V^{pi_k}_0(s0),
where pi_k is the policy the agent actually followed in episode k, evaluated exactly.

Run:  python code/ch14_exploration/riverswim.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
S, A, H = 6, 2, 20
LEFT, RIGHT = 0, 1


def riverswim():
    """Transition tensor P[s, a, s'] and expected rewards R[s, a]."""
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    for s in range(S):
        P[s, LEFT, max(s - 1, 0)] = 1.0
    R[0, LEFT] = 0.005
    P[0, RIGHT, 1], P[0, RIGHT, 0] = 0.4, 0.6
    for s in range(1, S - 1):
        P[s, RIGHT, s + 1], P[s, RIGHT, s], P[s, RIGHT, s - 1] = 0.35, 0.6, 0.05
    P[S - 1, RIGHT, S - 1], P[S - 1, RIGHT, S - 2] = 0.6, 0.4
    R[S - 1, RIGHT] = 1.0
    return P, R


def backward_induction(P, R):
    """Optimal finite-horizon Q_h for h = 0..H-1 (Q[H] = 0)."""
    Q = np.zeros((H + 1, S, A))
    for h in range(H - 1, -1, -1):
        Q[h] = R + P @ Q[h + 1].max(axis=1)
    return Q


def policy_value(P, R, pi):
    """Exact V^pi_0(s0) for a (possibly stochastic) time-indexed policy pi[h, s, a]."""
    V = np.zeros(S)
    for h in range(H - 1, -1, -1):
        Qh = R + P @ V
        V = (pi[h] * Qh).sum(axis=1)
    return V[0]


def greedy_policy(Q, rng, eps=0.0):
    """pi[h, s, a] greedy w.r.t. Q[h] (ties split evenly), mixed with eps-uniform."""
    best = Q[:H] == Q[:H].max(axis=2, keepdims=True)
    pi = best / best.sum(axis=2, keepdims=True)
    return (1 - eps) * pi + eps / A


class ModelBased:
    """Optimistic planning in the empirical MDP (Sections 3.2-3.3).

    beta = 0: certainty equivalence with optimism only for untried pairs (value H - h).
    beta > 0: add the bonus beta/sqrt(n(s, a)) to every tried pair as well (UCBVI-style).
    time_indexed=False pools all transitions of (s, a) over the steps h (a time-homogeneous
    model, counts n(s, a, s')); time_indexed=True keeps one model per step h (counts
    n_h(s, a, s'), as in the analysis of Sec. 3.3), so each entry sees ~1/H of the data.
    """

    def __init__(self, beta=0.0, name="", time_indexed=False):
        self.time_indexed = time_indexed
        shape = (H,) if time_indexed else (1,)
        self.counts = np.zeros(shape + (S, A, S))
        self.r_sum = np.zeros(shape + (S, A))
        self.beta, self.name = beta, name

    def policy(self, rng):
        n = self.counts.sum(axis=-1)                              # (1 or H, S, A)
        tried = n > 0
        P_hat = self.counts / np.maximum(n, 1)[..., None]
        R_hat = self.r_sum / np.maximum(n, 1)
        bonus = self.beta / np.sqrt(np.maximum(n, 1))
        Q = np.zeros((H + 1, S, A))
        for h in range(H - 1, -1, -1):
            j = h if self.time_indexed else 0
            q = R_hat[j] + bonus[j] + P_hat[j] @ Q[h + 1].max(axis=1)
            Q[h] = np.where(tried[j], np.minimum(q, H - h), H - h)   # unknown -> max value
        return greedy_policy(Q, rng)

    def update(self, traj):
        for h, s, a, r, s2 in traj:
            j = h if self.time_indexed else 0
            self.counts[j, s, a, s2] += 1
            self.r_sum[j, s, a] += r


class QLearner:
    """Finite-horizon Q-learning with separate tables Q_h(s, a) and counts n_h(s, a).

    Updates are applied at the end of the episode in REVERSE order, so a reward found late
    in the episode reaches the start in one episode (each sample is still used for exactly
    one stochastic-approximation update).  int_table=True reproduces Pitfall 7: the table
    is created as np.full(shape, H) with an integer H, so every update is truncated.
    """

    def __init__(self, q0=0.0, beta=0.0, eps=0.0, name="", int_table=False):
        if int_table:
            self.Q = np.full((H + 1, S, A), int(q0))     # BUG on purpose: integer dtype
        else:
            self.Q = np.full((H + 1, S, A), float(q0))   # float! (an int array would truncate)
        self.Q[H] = 0
        self.n = np.zeros((H, S, A))
        self.beta, self.eps, self.name = beta, eps, name

    def policy(self, rng):
        return greedy_policy(self.Q, rng, self.eps)

    def update(self, traj):
        for h, s, a, r, s2 in reversed(traj):
            self.n[h, s, a] += 1
            t = self.n[h, s, a]
            alpha = (H + 1) / (H + t)
            bonus = self.beta / np.sqrt(t)
            v_next = min(H - h - 1, self.Q[h + 1, s2].max())   # V_{h+1} clipped at H-h-1
            self.Q[h, s, a] += alpha * (r + bonus + v_next - self.Q[h, s, a])


class PSRLAgent:
    """Dirichlet(1/S) prior on each P(.|s,a); N(0,1) prior on mean rewards, noise sd 0.1."""

    name = "PSRL"

    def __init__(self, sigma_r=0.1):
        self.counts = np.zeros((S, A, S))
        self.n = np.zeros((S, A))
        self.r_sum = np.zeros((S, A))
        self.sigma_r = sigma_r

    def policy(self, rng):
        G = rng.gamma(1.0 / S + self.counts)
        P = G / G.sum(axis=2, keepdims=True)
        prec = 1.0 + self.n / self.sigma_r ** 2
        R = (self.r_sum / self.sigma_r ** 2) / prec + rng.standard_normal((S, A)) / np.sqrt(prec)
        return greedy_policy(backward_induction(P, R), rng)

    def update(self, traj):
        for h, s, a, r, s2 in traj:
            self.counts[s, a, s2] += 1
            self.n[s, a] += 1
            self.r_sum[s, a] += r


def make_agents(beta, beta_q):
    return {
        "eps-greedy Q": lambda: QLearner(q0=0.0, eps=0.1, name="eps-greedy Q"),
        f"UCB-Q, model-free (beta={beta_q})": lambda: QLearner(q0=H, beta=beta_q),
        "optimistic CE (no bonus)": lambda: ModelBased(beta=0.0),
        f"UCBVI-style (beta={beta})": lambda: ModelBased(beta=beta),
        f"UCBVI-style, time-indexed (beta={beta})":
            lambda: ModelBased(beta=beta, time_indexed=True),
        "PSRL": lambda: PSRLAgent(),
    }


def run(make, episodes, seed, P, R, v_star):
    rng = np.random.default_rng(seed)
    agent = make()
    Pc = P.cumsum(axis=2)
    regret = np.zeros(episodes)
    for k in range(episodes):
        pi = agent.policy(rng)
        regret[k] = v_star - policy_value(P, R, pi)       # exact expected regret of pi_k
        s, traj = 0, []
        u = rng.random((H, 2))
        for h in range(H):
            a = 0 if u[h, 0] < pi[h, s, 0] else 1          # sample a ~ pi_k(.|h, s)
            s2 = min(int(np.searchsorted(Pc[s, a], u[h, 1], side="right")), S - 1)
            traj.append((h, s, a, R[s, a], s2))
            s = s2
        agent.update(traj)
    return np.cumsum(regret)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--beta", type=float, default=0.3, help="UCBVI-style bonus scale")
    parser.add_argument("--beta-q", type=float, default=0.3, help="UCB-Q bonus scale")
    args = parser.parse_args()
    seeds = list(range(3 if args.quick else args.seeds))
    episodes = 200 if args.quick else args.episodes
    P, R = riverswim()
    Qs = backward_induction(P, R)
    v_star = Qs[0, 0].max()
    v_left = policy_value(P, R, np.tile(np.array([1.0, 0.0]), (H, S, 1)))
    print("RiverSwim (6 states, H = 20, start s0)")
    print(f"seeds {seeds[0]}..{seeds[-1]}, episodes {episodes}; V*_0(s0) = {v_star:.4f}; "
          f"'always left' policy value = {v_left:.4f}")
    print(f"agents: eps-greedy Q (eps 0.1, Q0 0) | UCB-Q (Q0 = H, beta {args.beta_q}; "
          f"tables Q_h, n_h per step) | optimistic CE (untried = H - h) | UCBVI-style "
          f"(beta {args.beta}; pooled time-homogeneous model, and a time-indexed variant "
          f"with n_h(s,a,s')) | PSRL (Dirichlet 1/S, reward prior N(0,1), noise sd 0.1)")
    t0 = time.time()
    agents = make_agents(args.beta, args.beta_q)
    res = {name: np.array([run(make, episodes, s, P, R, v_star) for s in seeds])
           for name, make in agents.items()}

    print(f"\nCumulative regret after {episodes} episodes (mean over seeds [min, max]); "
          f"'stuck' = regret in the last 10% of episodes > 50% of the maximum possible")
    last = max(1, episodes // 10)
    for name, cr in res.items():
        late = (cr[:, -1] - cr[:, -last - 1]) / last          # average per-episode regret
        stuck = int(np.sum(late > 0.5 * v_star))
        print(f"  {name:40s} {cr[:, -1].mean():8.1f} [{cr[:, -1].min():7.1f}, "
              f"{cr[:, -1].max():7.1f}]   late per-episode regret {late.mean():.3f}   "
              f"stuck {stuck}/{len(seeds)}")

    sweep_seeds = seeds[:10]
    print(f"\nBonus-scale sweeps (seeds {sweep_seeds[0]}..{sweep_seeds[-1]}): final "
          f"cumulative regret, mean [min, max]")
    sweep = {"UCBVI-style (model-based)": {}, "UCB-Q (model-free)": {}}
    makers = {"UCBVI-style (model-based)": lambda b: ModelBased(beta=b),
              "UCB-Q (model-free)": lambda b: QLearner(q0=H, beta=b)}
    for fam, mk in makers.items():
        for b in ([0.1, 1.0] if args.quick else [0.03, 0.1, 0.3, 1.0, 3.0]):
            cr = np.array([run(lambda b=b: mk(b), episodes, s, P, R, v_star)
                           for s in sweep_seeds])
            sweep[fam][b] = cr
            print(f"  {fam:26s} beta={b:5.2f}  {cr[:, -1].mean():8.1f} "
                  f"[{cr[:, -1].min():7.1f}, {cr[:, -1].max():7.1f}]")
    finals = {name: cr[:, -1].mean() for name, cr in res.items()}
    k_q = f"UCB-Q, model-free (beta={args.beta_q})"
    k_ti = f"UCBVI-style, time-indexed (beta={args.beta})"
    k_po = f"UCBVI-style (beta={args.beta})"
    if args.beta_q == args.beta:
        print(f"\nDecomposition of the model-free / model-based gap (same beta = {args.beta}):"
              f" UCB-Q / pooled model-based = {finals[k_q] / finals[k_po]:.1f}x"
              f" = (UCB-Q / time-indexed model-based {finals[k_q] / finals[k_ti]:.1f}x)"
              f" x (time-indexed / pooled {finals[k_ti] / finals[k_po]:.1f}x)")

    demo_seeds = seeds[:1] if args.quick else seeds[:5]
    cr_int = np.array([run(lambda: QLearner(q0=H, beta=args.beta_q, int_table=True),
                           episodes, s, P, R, v_star) for s in demo_seeds])
    cr_flt = res[k_q][:len(demo_seeds)]
    print(f"\nPitfall 7 demo: UCB-Q with an INTEGER table (np.full(shape, H), int H), "
          f"seeds {demo_seeds[0]}..{demo_seeds[-1]}: regret {cr_int[:, -1].mean():.1f} "
          f"[{cr_int[:, -1].min():.1f}, {cr_int[:, -1].max():.1f}] vs float table "
          f"{cr_flt[:, -1].mean():.1f} on the same seeds")
    print(f"Total time {time.time() - t0:.1f} s")

    if args.quick:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    colors = ["tab:red", "tab:orange", "tab:gray", "tab:blue", "tab:cyan", "tab:green"]
    ks = np.arange(1, episodes + 1)
    for (name, cr), col in zip(res.items(), colors):
        axes[0].plot(ks, cr.mean(0), color=col, label=name)
        axes[0].fill_between(ks, np.percentile(cr, 10, 0), np.percentile(cr, 90, 0),
                             color=col, alpha=0.15)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("episode k")
    axes[0].set_ylabel("cumulative regret (log scale)")
    axes[0].set_title(f"RiverSwim, H={H}: mean of {len(seeds)} seeds (band = 10-90%)")
    axes[0].legend(fontsize=8, loc="center right")
    axes[0].grid(alpha=0.3)
    bs = list(sweep["UCBVI-style (model-based)"])
    for fam, mk, ls in [("UCBVI-style (model-based)", "o", "-"),
                        ("UCB-Q (model-free)", "s", "--")]:
        final = [sweep[fam][b][:, -1] for b in bs]
        axes[1].errorbar(bs, [f.mean() for f in final],
                         yerr=[[f.mean() - f.min() for f in final],
                               [f.max() - f.mean() for f in final]],
                         marker=mk, ls=ls, capsize=3, label=fam)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"bonus scale $\beta$ in $\beta/\sqrt{n}$")
    axes[1].set_ylabel(f"cumulative regret after {episodes} episodes")
    axes[1].set_title(f"Bonus scale (mean, bars = min-max, {len(sweep_seeds)} seeds)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "riverswim_regret.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
