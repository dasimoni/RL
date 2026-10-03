"""Compounding errors: behaviour cloning vs DAgger on an unstable "tightrope" (Chapter 16, Section 2).

The environment (Section 2.6)
  Positions x in {-N, ..., N}, start at x = 0.  Cost c(x) = 1 when |x| >= 2 ("off balance"),
  else 0.  Actions a in {-2, ..., 2} push the walker sideways.  Off balance, gravity adds a
  drift of sign(x) per step, and random wind w in {-2, ..., 2} blows every step:
        x' = clip(x + a + drift(x) + w, -N, N),   P(w = 0, +-1, +-2) = (0.54, 0.20, 0.03).
  The expert plays a* = clip(-(x + drift(x)), -2, 2): it cancels its lean when balanced and
  pushes back at full strength when it is not, so it recovers from every position.

The learner (Section 2.6)
  The expert sees the true position (think of a human demonstrator, or a controller with
  privileged state).  The learner sees a noisy sensor: o = clip(x + xi), xi = -1, 0, +1 with
  probabilities 0.05, 0.9, 0.05, shown as a one-hot "image".  It is a classifier from o to the
  expert's label: a per-observation majority vote over its labelled data, falling back on the
  overall majority label (a = 0, "stand still") for observations it has never seen.  The
  sensor noise makes it err now and then even in familiar states (that is Ross & Bagnell's
  epsilon); the one-hot input means it knows nothing about positions absent from its data.

Everything is tabular, so expected costs are computed EXACTLY by propagating state
distributions through the Markov chain of each policy (no Monte Carlo noise in the
evaluation).  The only randomness is in which trajectories the learner happened to be trained
on; we average over many independent draws of those.

  BC      Algorithm 16.1 : fit the classifier to states visited by the EXPERT.
  DAgger  Algorithm 16.2 : roll out the current learner, have the expert label every visited
                           state, aggregate, refit (beta_1 = 1, beta_i = 0 for i > 1).

Run:  python code/ch16_offline_rl_and_imitation/dagger_vs_bc.py [--quick]
Output (full mode): figures/dagger_horizon.png, figures/dagger_budget.png
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

N = 5                                    # positions -N..N
XS = np.arange(-N, N + 1)
NS = len(XS)
ACTIONS = np.arange(-2, 3)               # 5 actions
WIND = np.arange(-2, 3)
P_WIND = np.array([0.03, 0.20, 0.54, 0.20, 0.03])
SENSOR = np.arange(-1, 2)                # sensor error xi
P_SENSOR = np.array([0.05, 0.9, 0.05])
COST = (np.abs(XS) >= 2).astype(float)
X0 = N                                   # index of position 0
DEFAULT_ACTION = 0                       # majority label in expert data (x = 0 dominates)


def drift(x):
    return np.where(np.abs(x) >= 2, np.sign(x), 0)


EXPERT = np.clip(-(XS + drift(XS)), -2, 2)      # expert action at every true position


def obs_index(i, xi):
    return int(np.clip(i + xi, 0, NS - 1))


def action_probs(policy_o, i):
    """Learner's action distribution at TRUE state index i (it acts on the noisy observation)."""
    p = np.zeros(len(ACTIONS))
    for xi, pxi in zip(SENSOR, P_SENSOR):
        p[policy_o[obs_index(i, xi)] + 2] += pxi
    return p


def transition_matrix(policy_o=None):
    """P[i, j] = Pr(x' = XS[j] | x = XS[i]); policy_o=None means the expert (true state)."""
    P = np.zeros((NS, NS))
    for i, x in enumerate(XS):
        if policy_o is None:
            pa = np.zeros(len(ACTIONS)); pa[EXPERT[i] + 2] = 1.0
        else:
            pa = action_probs(policy_o, i)
        for k, a in enumerate(ACTIONS):
            if pa[k] == 0:
                continue
            for w, pw in zip(WIND, P_WIND):
                P[i, int(np.clip(x + a + drift(x) + w, -N, N)) + N] += pa[k] * pw
    return P


def state_distributions(policy_o, T):
    """d_t for t = 0..T-1 (rows), starting from x = 0."""
    P = transition_matrix(policy_o)
    d = np.zeros((T, NS))
    d[0, X0] = 1.0
    for t in range(1, T):
        d[t] = d[t - 1] @ P
    return d


def expected_costs(policy_o, horizons):
    """J_T = sum_{t<T} E[c(x_t)] for every T in `horizons` -- exact."""
    per_step = state_distributions(policy_o, max(horizons)) @ COST
    cum = np.cumsum(per_step)
    return np.array([cum[T - 1] for T in horizons])


def rollout(policy_o, T, rng):
    """One trajectory of T steps.  policy_o=None: the expert drives.  Returns (obs index,
    expert label) for every visited state -- the label the expert would give there."""
    xis = rng.choice(SENSOR, size=T, p=P_SENSOR)     # pre-sample the noise for speed
    winds = rng.choice(WIND, size=T, p=P_WIND)
    pairs = []
    x = 0
    for t in range(T):
        i = x + N
        o = obs_index(i, xis[t])
        pairs.append((o, EXPERT[i]))
        a = EXPERT[i] if policy_o is None else policy_o[o]
        x = int(np.clip(x + a + drift(x) + winds[t], -N, N))
    return pairs


def fit_classifier(counts):
    """Per-observation majority vote; never-seen observations get DEFAULT_ACTION."""
    policy = np.full(NS, DEFAULT_ACTION)
    seen = counts.sum(1) > 0
    policy[seen] = ACTIONS[counts[seen].argmax(1)]
    return policy


def add_labels(counts, pairs):
    for o, a in pairs:
        counts[o, a + 2] += 1


def behaviour_cloning(n_traj, T_demo, rng):
    counts = np.zeros((NS, len(ACTIONS)), dtype=int)
    for _ in range(n_traj):
        add_labels(counts, rollout(None, T_demo, rng))
    return fit_classifier(counts)


def dagger(n_iters, T_demo, rng, traj_schedule=None):
    """Algorithm 16.2 with beta_1 = 1 (iteration 1: expert drives) and beta_i = 0 afterwards.
    traj_schedule[i] = number of trajectories collected in iteration i (default 1 each)."""
    schedule = traj_schedule or [1] * n_iters
    counts = np.zeros((NS, len(ACTIONS)), dtype=int)
    policy = None                                   # pi_1 = expert
    for m in schedule:
        for _ in range(m):
            add_labels(counts, rollout(policy, T_demo, rng))   # states of the CURRENT policy
        policy = fit_classifier(counts)                         # pi_{i+1} fit to aggregate D
    return policy


def epsilon_under_expert(policy_o, T=1000):
    """Ross & Bagnell's epsilon: probability that the learner's action differs from the
    expert's, with the state drawn from the expert's (time-averaged) distribution."""
    d = state_distributions(None, T).mean(0)
    err = np.array([1.0 - action_probs(policy_o, i)[EXPERT[i] + 2] for i in range(NS)])
    return float(d @ err)


def epsilon_by_horizon(policy_o, horizons, d_expert):
    """epsilon_T = (1/T) sum_{t<T} eps_t of Eq. (16.3): the error rate under the expert's state
    distribution d_t, averaged over the FIRST T steps only -- the epsilon that Theorem 16.1
    uses for horizon T.  d_expert = state_distributions(None, max(horizons))."""
    err = np.array([1.0 - action_probs(policy_o, i)[EXPERT[i] + 2] for i in range(NS)])
    cum = np.cumsum(d_expert @ err)
    return np.array([cum[T - 1] / T for T in horizons])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    quick = args.quick
    seed = 0
    rng = np.random.default_rng(seed)
    n_draws = 20 if quick else 300
    T_demo = 100
    n_traj = 10                              # label budget 10 x 100 = 1000 for both methods
    horizons = np.unique(np.round(np.geomspace(10, 3000, 12 if quick else 25)).astype(int))
    print(f"dagger_vs_bc: seed={seed} quick={quick} N={N} P(wind)={P_WIND.tolist()} "
          f"P(sensor error)={1 - P_SENSOR[1]:.1f} trajectory length={T_demo} "
          f"budget={n_traj * T_demo} labels (BC: {n_traj} expert trajectories; DAgger: "
          f"{n_traj} iterations x 1 trajectory) draws={n_draws}")
    t0 = time.time()

    # ---- Experiment 1: excess cost as a function of the horizon T at a fixed label budget ----
    J_exp = expected_costs(None, horizons)
    J_bc = np.zeros((n_draws, len(horizons)))
    J_dg = np.zeros((n_draws, len(horizons)))
    eps_bc, eps_dg = np.zeros(n_draws), np.zeros(n_draws)
    eps_bc_T = np.zeros((n_draws, len(horizons)))         # epsilon_T for each horizon (Eq. 16.3)
    d_expert = state_distributions(None, max(horizons))
    vis = {"expert": state_distributions(None, 1000).mean(0), "BC": np.zeros(NS),
           "DAgger": np.zeros(NS)}                 # time-averaged state distributions, T = 1000
    for k in range(n_draws):
        pi_bc = behaviour_cloning(n_traj, T_demo, rng)
        pi_dg = dagger(n_traj, T_demo, rng)
        J_bc[k] = expected_costs(pi_bc, horizons)
        J_dg[k] = expected_costs(pi_dg, horizons)
        eps_bc[k] = epsilon_under_expert(pi_bc)
        eps_dg[k] = epsilon_under_expert(pi_dg)
        eps_bc_T[k] = epsilon_by_horizon(pi_bc, horizons, d_expert)
        vis["BC"] += state_distributions(pi_bc, 1000).mean(0) / n_draws
        vis["DAgger"] += state_distributions(pi_dg, 1000).mean(0) / n_draws
    ex_bc = J_bc.mean(0) - J_exp
    ex_dg = J_dg.mean(0) - J_exp
    print(f"\nexpert per-step cost at T = 1000: {J_exp[np.searchsorted(horizons, 1000)] / horizons[np.searchsorted(horizons, 1000)]:.4f}"
          if 1000 in horizons else "")
    print(f"BC:     epsilon (0-1 error under the expert's state distribution) = {eps_bc.mean():.4f}")
    print(f"DAgger: epsilon under the expert's state distribution            = {eps_dg.mean():.4f}")
    print("(the table's eps_T is BC's error under the expert's distribution averaged over the first "
          "T steps only,\n as Theorem 16.1 uses it; eps_T * T^2 is the bound (16.4) on BC's excess cost)")
    print(f"\n{'T':>6s} {'J(expert)':>10s} {'excess BC':>10s} {'excess DAgger':>14s} "
          f"{'BC/T':>8s} {'DAgger/T':>9s} {'eps_T (BC)':>11s} {'eps_T*T^2':>10s}")
    eps_T = eps_bc_T.mean(0)
    for i, T in enumerate(horizons):
        print(f"{T:6d} {J_exp[i]:10.2f} {ex_bc[i]:10.2f} {ex_dg[i]:14.2f} {ex_bc[i] / T:8.4f} "
              f"{ex_dg[i] / T:9.4f} {eps_T[i]:11.4f} {eps_T[i] * T * T:10.1f}")
    for lo, hi in [(10, 100), (100, 1000)]:
        sel = (horizons >= lo) & (horizons <= hi)
        sb = np.polyfit(np.log(horizons[sel]), np.log(ex_bc[sel]), 1)[0]
        sd = np.polyfit(np.log(horizons[sel]), np.log(ex_dg[sel]), 1)[0]
        print(f"log-log slope of the excess cost for {lo} <= T <= {hi}: BC {sb:.2f}, DAgger {sd:.2f}")

    # ---- Experiment 2: excess cost at T = 1000 as a function of the label budget ----
    T_eval = 1000
    budgets = [2, 5, 10, 20] if quick else [2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100]
    n_b = n_draws // 3
    J0 = expected_costs(None, [T_eval])[0]
    res_b = []
    for nb in budgets:
        exb = [expected_costs(behaviour_cloning(nb, T_demo, rng), [T_eval])[0] - J0 for _ in range(n_b)]
        exd = [expected_costs(dagger(nb, T_demo, rng), [T_eval])[0] - J0 for _ in range(n_b)]
        res_b.append((nb * T_demo, np.mean(exb), np.std(exb) / np.sqrt(n_b),
                      np.mean(exd), np.std(exd) / np.sqrt(n_b)))
    print(f"\nexcess cost at T = {T_eval} vs number of expert labels (mean +- s.e. over {n_b} draws)")
    print(f"{'labels':>7s} {'BC':>15s} {'DAgger':>15s}")
    for lab, mb, sb, md, sd in res_b:
        print(f"{lab:7d} {mb:8.1f} +-{sb:5.1f} {md:8.1f} +-{sd:5.1f}")

    # ---- one example draw, and the state distributions averaged over all draws ----
    pi_bc = behaviour_cloning(n_traj, T_demo, rng)
    pi_dg = dagger(n_traj, T_demo, rng)
    print("\none example draw -- action chosen for each observation o = -N..N:")
    print("  expert (true x)", EXPERT.tolist())
    print("  BC             ", pi_bc.tolist())
    print("  DAgger         ", pi_dg.tolist())
    print("time-averaged state distribution at T = 1000, averaged over draws (x = -N..N):")
    for name in vis:
        print(f"  {name:7s}", " ".join(f"{p:.1e}" for p in vis[name]),
              f"| per-step cost {vis[name] @ COST:.3f}")
    print(f"total time {time.time() - t0:.1f}s")

    if quick:
        return
    import plot_style
    plt = plot_style.setup()
    C = plot_style.C

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    ax = axes[0]
    h = horizons.astype(float)
    ax.loglog(h, ex_bc, "o-", color=C[1], ms=4, label=f"BC ({n_traj * T_demo} expert labels)")
    ax.loglog(h, ex_dg, "s-", color=C[0], ms=4, label=f"DAgger (same {n_traj * T_demo} labels)")
    ax.loglog(h, J_exp, "-", color=C[2], lw=1.2, label=r"$J(\pi^\ast)$ itself (for scale)")
    # guide lines through the first BC point: slope 1 (linear) and slope 2 (quadratic)
    ax.loglog(h, ex_bc[0] * (h / h[0]), "--", color=plot_style.GREY, lw=1.1, label=r"$\propto T$")
    ax.loglog(h, ex_bc[0] * (h / h[0]) ** 2, ":", color=plot_style.INK, lw=1.1, label=r"$\propto T^2$")
    ax.set_xlabel("horizon $T$")
    ax.set_ylabel(r"excess cost $J(\hat\pi) - J(\pi^\ast)$")
    ax.set_title("Compounding errors: excess cost vs horizon")
    ax.set_ylim(0.05, 5e3)
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    w = 0.27
    for k, (name, col) in enumerate([("expert", C[2]), ("BC", C[1]), ("DAgger", C[0])]):
        ax.bar(XS + (k - 1) * w, np.maximum(vis[name], 1e-9), width=w, color=col, label=name)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1)
    ax.axvspan(-1.5, 1.5, color=C[2], alpha=0.08, lw=0)
    ax.set_xlabel("position $x$ (cost 0 only inside the shaded band)")
    ax.set_ylabel("time-averaged state distribution, $T = 1000$")
    ax.set_title(f"Where each policy spends its time (average over {n_draws} draws)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dagger_horizon.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    labs = np.array([r[0] for r in res_b])
    for (j, name, col, mk) in [(1, "BC", C[1], "o"), (3, "DAgger", C[0], "s")]:
        m = np.array([r[j] for r in res_b]); se = np.array([r[j + 1] for r in res_b])
        ax.errorbar(labs, m, yerr=se, marker=mk, color=col, label=name, capsize=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of expert-labelled states")
    ax.set_ylabel(f"excess cost at $T = {T_eval}$")
    ax.set_title("Label efficiency (mean $\\pm$ s.e.)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dagger_budget.png"))
    plt.close(fig)
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
