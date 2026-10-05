"""Maximization bias: Q-learning vs Double Q-learning (Sutton & Barto 2018, Example 6.7).

Chapter 05, Section 11.

The MDP (gamma = 1), drawn as in S&B's Figure 6.5:

    terminal <--any of N_B actions-- [B] <--left (r=0)-- [A] --right (r=0)--> terminal
             reward ~ Normal(-0.1, 1)

Going left is a mistake *in expectation*: q*(A, left) = -0.1 < q*(A, right) = 0.  But B has
many actions whose sample means are noisy, and max_b Q(B, b) is an upward-biased estimate
of max_b q(B, b) = -0.1, so Q-learning is lured to the left early on.

Settings as in S&B: epsilon-greedy with epsilon = 0.1, alpha = 0.1, Q initialised to 0,
300 episodes.  S&B only say that B has "many" actions; we use N_B = 10 (a common choice in
reproductions) and also vary it.  With epsilon = 0.1 and 2 actions in A, the best
achievable fraction of 'left' is epsilon / 2 = 5%.

For speed, all independent runs are simulated *in parallel* with numpy: row r of every
array belongs to run r.  The per-run logic is exactly the sequential algorithm.

Diagnostics (Section 11.4): how many times Q(A, left) and each Q(B, b) have been updated by
episode 300 and the mean bootstrap target from B; then long runs (20,000 episodes) of
Q-learning and Double Q-learning under four *behaviour* policies (eps-greedy as in S&B, and
uniformly random choices in A and/or B) to see where their estimates of Q(A, left) end up and
why.  Every action in B has the same true value, so the double estimator (5.17) is exactly
unbiased on this MDP; the long-run undershoot comes from eps-greedy behaviour combined with a
constant step size (estimates that happen to be low are rarely revisited and stay low).

Run:  python code/ch05_temporal_difference/maximization_bias.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

LEFT, RIGHT = 0, 1
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def argmax_random_ties(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row-wise argmax of X (shape [runs, k]) with ties broken uniformly at random."""
    is_max = X == X.max(axis=1, keepdims=True)
    return np.argmax(np.where(is_max, rng.random(X.shape), -1.0), axis=1)


def eps_greedy(X: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    greedy = argmax_random_ties(X, rng)
    explore = rng.random(X.shape[0]) < eps
    return np.where(explore, rng.integers(0, X.shape[1], X.shape[0]), greedy)


def eps_greedy_expectation(X: np.ndarray, eps: float) -> np.ndarray:
    """sum_a pi(a|s) X[a] for the eps-greedy policy (ties share the greedy mass)."""
    k = X.shape[1]
    is_max = X == X.max(axis=1, keepdims=True)
    probs = eps / k + (1 - eps) * is_max / is_max.sum(axis=1, keepdims=True)
    return (probs * X).sum(axis=1)


# Behaviour policies for the long diagnostic runs of Section 11.5: the exploration rate used to
# *choose* actions in A and in B (1.0 = uniformly random).  The learning targets do not change.
BEHAVIOURS = {
    "eps-greedy": lambda eps: (eps, eps),          # the standard setting (S&B Example 6.7)
    "uniform in A and B": lambda eps: (1.0, 1.0),  # visits independent of the estimates
    "uniform in B only": lambda eps: (eps, 1.0),   # eps-greedy in A, uniform in B
    "uniform in A only": lambda eps: (1.0, eps),   # uniform in A, eps-greedy in B
}


def run(method: str, n_runs: int, n_episodes: int, n_b: int, alpha: float, eps: float,
        seed: int, reward_mean: float = -0.1, behaviour: str = "eps-greedy"):
    """Simulate n_runs independent learners.

    `behaviour` (a key of BEHAVIOURS) sets how actions are *chosen* in A and B; the update
    rules (and Expected SARSA's eps-greedy target policy) are the same for every behaviour.

    Returns (frac_left[episode], mean Q(A,left)[episode], diagnostics).  The diagnostics are
    measured after the last episode and averaged over runs:
      updates_Q_A_left_per_table, updates_per_B_action_per_table : update counts
      init_weight_B   : mean over the Q(B,b) entries of (1-alpha)^n(b), the weight still left
                        on the initial value 0 after n(b) updates
      mean_target_B   : the method's (expected) bootstrap target from B for Q(A,left):
                        Q-learning max_b Q(B,b); SARSA and Expected SARSA the expectation of
                        Q(B,A') under the behaviour / eps-greedy target policy in B;
                        Double Q the average of Q2(B, argmax Q1(B,.)) and Q1(B, argmax Q2(B,.))
      mean_Q_B        : mean of all Q(B,b) entries (all tables)
      q_left_se       : standard error over runs of the final estimate of Q(A,left)
      target_sum, target_cnt : per-episode sum and number of the bootstrap targets actually
                        used to update Q(A,left) (for averaging them over episode windows)
    Double Q-learning only (the "selected by Q1, evaluated by Q2" case):
      agree_frac      : fraction of runs where argmax_b Q1(B,b) is also the greedy action of
                        the behaviour, argmax_b [Q1+Q2](B,b)
      target_agree, target_disagree : mean of Q2(B, argmax_b Q1(B,b)) over those runs and
                        over the remaining runs"""
    rng = np.random.default_rng(seed)
    eps_A, eps_B = BEHAVIOURS[behaviour](eps)
    idx = np.arange(n_runs)
    double = method == "Double Q-learning"
    # Q1/Q2 for Double Q-learning; single tables use QA1/QB1 only.
    QA1, QB1 = np.zeros((n_runs, 2)), np.zeros((n_runs, n_b))
    QA2, QB2 = np.zeros((n_runs, 2)), np.zeros((n_runs, n_b))
    frac_left = np.zeros(n_episodes)
    q_left = np.zeros(n_episodes)
    target_sum = np.zeros(n_episodes)              # sum / count of the bootstrap targets actually
    target_cnt = np.zeros(n_episodes)              # used to update Q(A,left) in each episode
    n_upd_left = np.zeros((n_runs, 2))            # updates of Q(A,left) in table 1 / table 2
    n_upd_B = np.zeros((n_runs, 2, n_b))          # updates of each Q(B, b) in table 1 / table 2

    for ep in range(n_episodes):
        # ---- step 1: in A ------------------------------------------------------------
        a = eps_greedy(QA1 + QA2 if double else QA1, eps_A, rng)  # behaviour: eps-greedy on Q1+Q2
        left = a == LEFT
        frac_left[ep] = left.mean()
        L, Rt = idx[left], idx[~left]
        # SARSA needs the next action in B before updating Q(A, left): choose it now for everyone
        b = eps_greedy(QB1 + QB2 if double else QB1, eps_B, rng)

        # 'right' -> terminal with reward 0: target = 0 for every method
        if double:
            coin = rng.random(n_runs) < 0.5               # which table to update at this step
            r1, r2 = Rt[coin[Rt]], Rt[~coin[Rt]]
            QA1[r1, RIGHT] += alpha * (0.0 - QA1[r1, RIGHT])
            QA2[r2, RIGHT] += alpha * (0.0 - QA2[r2, RIGHT])
        else:
            QA1[Rt, RIGHT] += alpha * (0.0 - QA1[Rt, RIGHT])

        # 'left' -> B with reward 0: the target bootstraps from Q(B, .)
        if not double:
            n_upd_left[L, 0] += 1
        if method == "Q-learning":
            target = QB1[L].max(axis=1)
            QA1[L, LEFT] += alpha * (target - QA1[L, LEFT])
        elif method == "SARSA":
            target = QB1[L, b[L]]
            QA1[L, LEFT] += alpha * (target - QA1[L, LEFT])
        elif method == "Expected SARSA":
            target = eps_greedy_expectation(QB1[L], eps)
            QA1[L, LEFT] += alpha * (target - QA1[L, LEFT])
        elif double:
            # Update Q1 with probability 1/2: select with Q1, evaluate with Q2 (and vice versa)
            l1, l2 = L[coin[L]], L[~coin[L]]
            n_upd_left[l1, 0] += 1
            n_upd_left[l2, 1] += 1
            a1 = argmax_random_ties(QB1[l1], rng)
            QA1[l1, LEFT] += alpha * (QB2[l1, a1] - QA1[l1, LEFT])
            a2 = argmax_random_ties(QB2[l2], rng)
            QA2[l2, LEFT] += alpha * (QB1[l2, a2] - QA2[l2, LEFT])
            target = np.concatenate([QB2[l1, a1], QB1[l2, a2]])
        else:
            raise ValueError(method)
        target_sum[ep], target_cnt[ep] = target.sum(), len(target)

        # ---- step 2: in B (only runs that went left) -----------------------------------
        r = rng.normal(reward_mean, 1.0, size=len(L))
        bL = b[L]
        if double:
            coin2 = rng.random(len(L)) < 0.5
            l1, l2 = L[coin2], L[~coin2]
            QB1[l1, bL[coin2]] += alpha * (r[coin2] - QB1[l1, bL[coin2]])
            QB2[l2, bL[~coin2]] += alpha * (r[~coin2] - QB2[l2, bL[~coin2]])
            n_upd_B[l1, 0, bL[coin2]] += 1
            n_upd_B[l2, 1, bL[~coin2]] += 1
        else:
            QB1[L, bL] += alpha * (r - QB1[L, bL])        # terminal next state: target = R
            n_upd_B[L, 0, bL] += 1

        q_left[ep] = ((QA1[:, LEFT] + QA2[:, LEFT]) / 2 if double else QA1[:, LEFT]).mean()
    # ---- diagnostics after the last episode (they use rng only after learning has ended) ----
    n_tables = 2 if double else 1
    diag = {
        "updates_Q_A_left_per_table": n_upd_left.sum(axis=1).mean() / n_tables,
        "updates_per_B_action_per_table": n_upd_B[:, :n_tables].sum(axis=(1, 2)).mean() / (n_tables * n_b),
        "init_weight_A": ((1 - alpha) ** n_upd_left[:, :n_tables]).mean(),
        "init_weight_B": ((1 - alpha) ** n_upd_B[:, :n_tables]).mean(),
        "mean_Q_B": (np.concatenate([QB1, QB2], axis=1) if double else QB1).mean(),
    }
    diag["target_sum"], diag["target_cnt"] = target_sum, target_cnt
    final_left = (QA1[:, LEFT] + QA2[:, LEFT]) / 2 if double else QA1[:, LEFT]
    diag["q_left_se"] = final_left.std(ddof=1) / np.sqrt(n_runs)
    if method == "Q-learning":
        diag["mean_target_B"] = QB1.max(axis=1).mean()
    elif method == "SARSA":            # E[Q(B, A')] with A' drawn from the behaviour in B
        diag["mean_target_B"] = eps_greedy_expectation(QB1, eps_B).mean()
    elif method == "Expected SARSA":   # its target policy is eps-greedy whatever the behaviour
        diag["mean_target_B"] = eps_greedy_expectation(QB1, eps).mean()
    else:
        sel1 = argmax_random_ties(QB1, rng)          # selected by Q1 ...
        sel2 = argmax_random_ties(QB2, rng)
        t1, t2 = QB2[idx, sel1], QB1[idx, sel2]      # ... evaluated by Q2 (and vice versa)
        diag["mean_target_B"] = ((t1 + t2) / 2).mean()
        agree = sel1 == argmax_random_ties(QB1 + QB2, rng)
        diag["agree_frac"] = agree.mean()
        diag["target_agree"] = t1[agree].mean()
        diag["target_disagree"] = t1[~agree].mean() if (~agree).any() else np.nan
    return frac_left, q_left, diag


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    n_runs = 1000 if args.quick else 10_000
    n_episodes, n_b, alpha, eps = 300, 10, 0.1, 0.1
    methods = ["Q-learning", "Double Q-learning", "SARSA", "Expected SARSA"]
    print(f"[maximization_bias] seed={args.seed} runs={n_runs} episodes={n_episodes} "
          f"actions in B={n_b} alpha={alpha} epsilon={eps}")
    t0 = time.time()

    res = {m: run(m, n_runs, n_episodes, n_b, alpha, eps, args.seed) for m in methods}
    print("\n% of episodes in which 'left' was taken from A (optimal under eps-greedy: 5%)")
    print("  method              | ep 1   ep 10  ep 25  ep 50  ep 100 ep 300 | peak (episode)")
    for m, (fl, _, _) in res.items():
        print(f"  {m:19s} | " + "  ".join(f"{100 * fl[e - 1]:5.1f}" for e in (1, 10, 25, 50, 100, 300))
              + f" | {100 * fl.max():5.1f} ({int(fl.argmax()) + 1})")
    print("\nMean estimate of Q(A, left) (true q*(A,left) = -0.1)")
    print("  method              | ep 10   ep 25   ep 50   ep 100  ep 300 | peak (episode)")
    for m, (_, ql, _) in res.items():
        print(f"  {m:19s} | " + "  ".join(f"{ql[e - 1]:+.3f}" for e in (10, 25, 50, 100, 300))
              + f" | {ql.max():+.4f} ({int(ql.argmax()) + 1})")

    # Why is every estimate still above -0.1 at episode 300?  Count the updates (Section 11.4).
    print(f"\nAfter {n_episodes} episodes (mean over runs and tables; alpha={alpha}, so after n updates a "
          f"weight (1-alpha)^n is left on the initial value 0):")
    print("  method              | updates of Q(A,left) | its init weight | updates per Q(B,b) "
          "| init weight of Q(B,b) | mean bootstrap target from B")
    for m, (_, _, d) in res.items():
        print(f"  {m:19s} | {d['updates_Q_A_left_per_table']:8.1f}             | {d['init_weight_A']:9.1e}       "
              f"| {d['updates_per_B_action_per_table']:6.1f}             | {d['init_weight_B']:6.3f}                "
              f"| {d['mean_target_B']:+.3f}")
    print("  (bootstrap target: Q-learning max_b Q(B,b); SARSA/Expected SARSA E[Q(B,A')] under eps-greedy;"
          "\n   Double Q-learning Q_j(B, argmax_b Q_i(B,b)), averaged over i != j)")
    # Q(A,left) is a weighted average of its initial value 0 (weight w) and of the targets it was
    # updated towards (weight 1 - w).  Had every target been the true -0.1, its mean would be
    # -(1 - w) * 0.1; the rest of the gap comes from the targets themselves.
    windows = [(1, 50), (51, 100), (101, 200), (201, 300)]
    print("\nMean bootstrap target actually used to update Q(A,left), by episode window, and the value")
    print("Q(A,left) would have at episode 300 if every target had been the true -0.1:")
    print("  method              | " + "  ".join(f"ep {lo}-{hi:<4d}" for lo, hi in windows)
          + "| -(1 - w) * 0.1 | actual Q(A,left)")
    for m, (_, ql, d) in res.items():
        tw = [d["target_sum"][lo - 1:hi].sum() / max(d["target_cnt"][lo - 1:hi].sum(), 1) for lo, hi in windows]
        print(f"  {m:19s} | " + "  ".join(f"{t:+.3f}     " for t in tw)
              + f"| {-(1 - d['init_weight_A']) * 0.1:+.3f}         | {ql[-1]:+.3f}")
    print("  (targets pooled over all updates in the window, so runs that go left often weigh more;\n"
          "   for the single-table methods these are runs whose target is currently high)")

    # Long runs: where do the estimates of Q(A, left) end up, and why?  The behaviour policy
    # decides which estimates get revisited; we compare eps-greedy with uniformly random choices.
    n_runs_long, n_ep_long = (100, 600) if args.quick else (1000, 20_000)
    checkpoints = [e for e in (300, 1000, 3000, 10_000, 20_000) if e <= n_ep_long]
    print(f"\nLong runs ({n_runs_long} runs x {n_ep_long} episodes, seed {args.seed + 7}): mean estimate "
          f"of Q(A, left) (true -0.1)")
    print("  behaviour           method              | " + "  ".join(f"ep {e:<6d}" for e in checkpoints)
          + f"| ±SE (last) | % left, last {min(1000, n_ep_long):,} ep. | mean target from B | mean Q(B,b)")
    long_res = {}
    for beh in BEHAVIOURS:
        for m in ("Q-learning", "Double Q-learning"):
            fl, ql, d = run(m, n_runs_long, n_ep_long, n_b, alpha, eps, args.seed + 7, behaviour=beh)
            long_res[(beh, m)] = (fl, ql, d)
            print(f"  {beh:19s} {m:19s} | " + "  ".join(f"{ql[e - 1]:+.3f}   " for e in checkpoints)
                  + f"| {d['q_left_se']:.3f}      | {100 * fl[-1000:].mean():5.1f}"
                  + f"                  | {d['mean_target_B']:+.3f}             | {d['mean_Q_B']:+.3f}")
    print("\nDouble Q-learning at the end of the long runs: is the action selected by Q1 in B also the")
    print("behaviour's greedy action argmax(Q1+Q2)?  Mean of its evaluation Q2(B, argmax Q1) in each case:")
    print("  behaviour           | agree (fraction of runs) | Q2(sel.) if agree | Q2(sel.) if not")
    for beh in BEHAVIOURS:
        d = long_res[(beh, "Double Q-learning")][2]
        print(f"  {beh:19s} | {d['agree_frac']:5.2f}                    | {d['target_agree']:+.3f}            "
              f"| {d['target_disagree']:+.3f}")

    # How the bias grows with the number of noisy actions in B
    nb_list = [1, 2, 10, 50] if not args.quick else [1, 10]
    print("\nEffect of the number of actions in B: % 'left' averaged over episodes 1-100")
    print("  N_B  | Q-learning  Double Q-learning")
    for nb in nb_list:
        fq = run("Q-learning", n_runs, 100, nb, alpha, eps, args.seed + 1)[0].mean()
        fd = run("Double Q-learning", n_runs, 100, nb, alpha, eps, args.seed + 1)[0].mean()
        print(f"  {nb:4d} | {100 * fq:8.1f}%   {100 * fd:8.1f}%")

    if not args.quick:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIG_DIR, exist_ok=True)
        colors = {"Q-learning": "tab:red", "Double Q-learning": "tab:green",
                  "SARSA": "tab:blue", "Expected SARSA": "tab:purple"}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        x = np.arange(1, n_episodes + 1)
        for m, (fl, ql, _) in res.items():
            lw = 2.2 if m in ("Q-learning", "Double Q-learning") else 1.0
            axes[0].plot(x, 100 * fl, color=colors[m], lw=lw, label=m)
            axes[1].plot(x, ql, color=colors[m], lw=lw, label=m)
        axes[0].axhline(5, color="k", ls="--", lw=1, label="optimal (ε/2 = 5%)")
        axes[0].set_ylim(0, 100)
        axes[0].set_xlabel("episode")
        axes[0].set_ylabel("% left actions from A")
        axes[0].set_title(f"Maximization bias ({n_runs:,} runs, {n_b} actions in B)")
        axes[0].legend(fontsize=8)
        axes[1].axhline(-0.1, color="k", ls="--", lw=1, label="q*(A, left) = -0.1")
        axes[1].axhline(0.0, color="gray", ls=":", lw=1, label="q*(A, right) = 0 (= its estimate)")
        axes[1].set_xlabel("episode")
        axes[1].set_ylabel("mean estimate of Q(A, left)")
        axes[1].set_title("Value estimate of the bad action")
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "maximization_bias.png"), dpi=110)
        plt.close(fig)
        print(f"\nFigure written to {FIG_DIR}/maximization_bias.png")

        # Long runs: mean estimate of Q(A, left) under each behaviour, on a log episode axis.
        # Episodes are averaged in log-spaced bins so the late part of the curve is readable.
        edges = np.unique(np.geomspace(1, n_ep_long + 1, 120).astype(int))
        centres = np.sqrt(edges[:-1] * (edges[1:] - 1).clip(min=edges[:-1]))
        styles = {"eps-greedy": ("k", "-"), "uniform in A and B": ("tab:blue", "--"),
                  "uniform in B only": ("tab:orange", "-."), "uniform in A only": ("tab:purple", ":")}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, m in zip(axes, ("Q-learning", "Double Q-learning")):
            for beh, (c, ls) in styles.items():
                ql = long_res[(beh, m)][1]
                binned = [ql[lo - 1:hi - 1].mean() for lo, hi in zip(edges[:-1], edges[1:])]
                ax.plot(centres, binned, color=c, ls=ls, lw=1.8,
                        label=beh + (" (S&B setting)" if beh == "eps-greedy" else ""))
            ax.axhline(-0.1, color="gray", ls="--", lw=1, label="true q*(A, left) = -0.1")
            ax.set_xscale("log")
            ax.set_xlabel("episode (log scale)")
            ax.set_title(f"{m}: behaviour policy vs long-run estimate")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(f"mean estimate of Q(A, left) ({n_runs_long:,} runs)")
        axes[1].legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "maximization_bias_long.png"), dpi=110)
        plt.close(fig)
        print(f"Figure written to {FIG_DIR}/maximization_bias_long.png")
    print(f"\nRuntime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
