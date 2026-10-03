"""Return-conditioned supervised learning (the idea behind Decision Transformer) vs dynamic
programming on two tiny offline problems (Chapter 16, Section 11).

RCSL ("upside-down RL", Decision Transformer, RvS) learns pi(a | s, g) by supervised learning,
where g is the return-to-go that FOLLOWED action a in the data, and at test time conditions
on a high desired return g_0, decreasing it by every reward received.  Here pi(a | s, g) is
tabular: at state s it plays the logged action whose return-to-go is closest to the commanded
g (ties broken by frequency) -- the most faithful reading of "imitate what was done when that
return was obtained".  Offline DP is value iteration on the empirical MDP restricted to the
logged actions (in-sample Q-learning: no out-of-data action is ever queried).

Problem 1 -- stitching.  Deterministic.  From the start A, the data only contain
    A -up-> U -> M -left-> (end, return 0)        and   A -down-> D -> (end, return 3);
from a different start B they contain  B -> M -right-> (end, return 10).
The optimal plan from A, A -up-> U -> M -right->, appears in no trajectory: it must be stitched
from two.  DP does that; RCSL cannot, whatever return it is told to achieve.

Problem 2 -- luck.  One decision.  "safe" pays 5; "gamble" pays 10 with probability 0.3 and 0
otherwise (mean 3).  Logged with a uniform policy.  Conditioned on g_0 = 10, RCSL imitates the
trajectories that achieved 10 -- all of which gambled and got lucky.

Run:  python code/ch16_offline_rl_and_imitation/sequence_vs_dp.py [--quick]
(No figures: the results are small tables.)
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np


# ------------------------------------------------------------------ generic tabular machinery
def returns_to_go(traj):
    """traj = [(s, a, r), ...] -> list of (s, a, g) with g = sum of rewards from that step on
    (no discounting in these toy problems)."""
    g, out = 0.0, []
    for s, a, r in reversed(traj):
        g += r
        out.append((s, a, g))
    return out[::-1]


def fit_rcsl(dataset):
    """pi(a | s, g): for each state, the list of (action, return-to-go) pairs seen in the data."""
    table = defaultdict(list)
    for traj in dataset:
        for s, a, g in returns_to_go(traj):
            table[s].append((a, g))
    return table


def rcsl_act(table, s, g):
    pairs = table[s]
    best = min(abs(gg - g) for _, gg in pairs)
    close = [a for a, gg in pairs if abs(gg - g) == best]
    return Counter(close).most_common(1)[0][0]


def fit_bc(dataset):
    """Deterministic BC: the majority action in each state (ties: first occurrence)."""
    counts = defaultdict(Counter)
    for traj in dataset:
        for s, a, _ in traj:
            counts[s][a] += 1
    return {s: c.most_common(1)[0][0] for s, c in counts.items()}


def fit_bc_stochastic(dataset):
    """Stochastic BC: sample from the empirical action frequencies in each state."""
    counts = defaultdict(Counter)
    for traj in dataset:
        for s, a, _ in traj:
            counts[s][a] += 1
    return {s: {a: k / sum(c.values()) for a, k in c.items()} for s, c in counts.items()}


def stitch_expected_return(probs, s="A"):
    """Exact expected return of a stochastic policy in the (deterministic) stitching MDP."""
    if s == "end":
        return 0.0
    return sum(p * (STITCH[s][a][0] + stitch_expected_return(probs, STITCH[s][a][1]))
               for a, p in probs[s].items())


def offline_dp(dataset, terminal="end"):
    """In-sample value iteration on the empirical MDP: Q(s,a) = mean over logged outcomes of
    r + max_{a' logged at s'} Q(s', a').  Only (s, a) pairs present in the data get a value."""
    outcomes = defaultdict(list)                      # (s, a) -> list of (r, s')
    for traj in dataset:
        for i, (s, a, r) in enumerate(traj):
            s2 = traj[i + 1][0] if i + 1 < len(traj) else terminal
            outcomes[(s, a)].append((r, s2))
    Q = {sa: 0.0 for sa in outcomes}
    for _ in range(100):
        V = defaultdict(float)
        for (s, a), q in Q.items():
            V[s] = max(V.get(s, -np.inf), q)
        V[terminal] = 0.0
        Q = {sa: float(np.mean([r + V[s2] for r, s2 in outs])) for sa, outs in outcomes.items()}
    policy = {}
    for (s, a), q in Q.items():
        if s not in policy or q > Q[(s, policy[s])]:
            policy[s] = a
    return Q, policy


# ------------------------------------------------------------------ problem 1: stitching
STITCH = {  # state -> action -> (reward, next state)
    "A": {"up": (0.0, "U"), "down": (0.0, "D")},
    "B": {"go": (0.0, "M")},
    "U": {"go": (0.0, "M")},
    "D": {"go": (3.0, "end")},
    "M": {"left": (0.0, "end"), "right": (10.0, "end")},
}


def stitch_dataset(n_each):
    up_left = [("A", "up", 0.0), ("U", "go", 0.0), ("M", "left", 0.0)]
    down = [("A", "down", 0.0), ("D", "go", 3.0)]
    b_right = [("B", "go", 0.0), ("M", "right", 10.0)]
    return [up_left] * n_each + [down] * n_each + [b_right] * n_each


def stitch_rollout(choose, s="A"):
    """choose(s, g) -> action; g is the remaining commanded return (ignored by non-RCSL)."""
    total, path = 0.0, [s]
    while s != "end":
        a = choose(s)
        r, s = STITCH[path[-1]][a]
        total += r
        path += [a, s]
    return total, " ".join(path)


# ------------------------------------------------------------------ problem 2: luck
def luck_dataset(n, rng):
    data = []
    for _ in range(n):
        if rng.random() < 0.5:
            data.append([("S", "safe", 5.0)])
        else:
            data.append([("S", "gamble", 10.0 if rng.random() < 0.3 else 0.0)])
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seed = 0
    rng = np.random.default_rng(seed)
    n_luck = 200 if args.quick else 2000
    print(f"sequence_vs_dp: seed={seed} quick={args.quick} (stitching: 5 copies of each "
          f"trajectory type; luck: {n_luck} logged one-step episodes)")

    # ---- problem 1
    data = stitch_dataset(5)
    bc = fit_bc(data)
    table = fit_rcsl(data)
    Q, dp_pol = offline_dp(data)
    print("\nProblem 1 (stitching).  Optimal return from A is 10 (A up U go M right).")
    print("  offline DP Q-values:", {f"{s},{a}": round(q, 2) for (s, a), q in sorted(Q.items())})
    ret, path = stitch_rollout(lambda s: dp_pol[s])
    print(f"  offline DP (in-sample value iteration): return {ret:5.1f}   path {path}")
    ret, path = stitch_rollout(lambda s: bc[s])
    print(f"  behaviour cloning (majority action):    return {ret:5.1f}   path {path}"
          "   (ties at A and M broken by first occurrence)")
    ev_bc = stitch_expected_return(fit_bc_stochastic(data))
    print(f"  behaviour cloning (sampling the empirical action frequencies): expected return {ev_bc:.2f}")
    for g0 in [0, 3, 5, 10, 20]:
        g = [float(g0)]

        def choose(s, g=g):
            a = rcsl_act(table, s, g[0])
            g[0] -= STITCH[s][a][0]           # decrement the command by the reward received
            return a
        ret, path = stitch_rollout(choose)
        print(f"  RCSL conditioned on g0 = {g0:2d}:          return {ret:5.1f}   path {path}")

    # ---- problem 2
    data = luck_dataset(n_luck, rng)
    table = fit_rcsl(data)
    Q, dp_pol = offline_dp(data)
    ev = {"safe": 5.0, "gamble": 0.3 * 10.0}
    print("\nProblem 2 (luck).  E[return]: safe 5.0, gamble 3.0.")
    print(f"  empirical Q: safe {Q[('S', 'safe')]:.2f}, gamble {Q[('S', 'gamble')]:.2f}  ->  "
          f"offline DP plays '{dp_pol['S']}' (expected return {ev[dp_pol['S']]:.1f})")
    for g0 in [0, 5, 10]:
        a = rcsl_act(table, "S", g0)
        print(f"  RCSL conditioned on g0 = {g0:2d}: plays '{a}' (expected return {ev[a]:.1f})")


if __name__ == "__main__":
    main()
