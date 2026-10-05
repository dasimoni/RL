"""Finite-horizon DP (backward induction, Section 11) on FrozenLake with Gymnasium's 100-step TimeLimit.

The real task in Gymnasium is "reach G within 100 steps": a FINITE-horizon problem (gamma = 1,
reward 1 at the goal). Backward induction (Algorithm 3.6) solves it exactly in H = 100 backward
sweeps and yields a NON-STATIONARY optimal policy pi_t(s). We compare it with the best stationary
policies of Sections 5 and 6 (optimal for gamma = 0.99, and for gamma = 1 with no time limit), check the
success probabilities by simulation in the real environment, and verify that the value-iteration
iterate V_k = (T*)^k 0 equals the optimal k-step value.

Run:  python code/ch03_dynamic_programming/finite_horizon.py [--quick]
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

from dp import (backward_induction, bellman_optimality, evaluate_finite_horizon, policy_iteration,  # noqa: E402
                policy_model, q_from_v, uniform_policy, value_iteration)
from mdps import frozenlake_mdp  # noqa: E402

SEED = 0
H = 100   # gym.spec("FrozenLake-v1").max_episode_steps


def simulate(map_name, act_fn, n_episodes):
    """Run in the real environment (default TimeLimit). act_fn(t, obs) -> action."""
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=True)
    assert env.spec.max_episode_steps == H
    obs, _ = env.reset(seed=SEED)
    wins = 0
    for ep in range(n_episodes):
        if ep > 0:
            obs, _ = env.reset()
        t = 0
        while True:
            obs, r, terminated, truncated, _ = env.step(int(act_fn(t, obs)))
            t += 1
            if terminated or truncated:
                wins += r > 0
                break
    env.close()
    p = wins / n_episodes
    return p, np.sqrt(p * (1 - p) / n_episodes)


def stationary_optimal(map_name, gamma):
    """Optimal stationary policy for the infinite-horizon problem with discount gamma (no time limit)."""
    mdp, _ = frozenlake_mdp(map_name, gamma)
    if gamma < 1:
        return policy_iteration(mdp, uniform_policy(mdp))[0]
    # gamma = 1: v_* = max probability of ever reaching G. Greedy w.r.t. v_* can be improper
    # (Chapter 01, Sec. 12.4), so we keep, among the actions greedy w.r.t. v_*, one that is also
    # greedy for gamma = 0.99999 (which breaks the ties in favour of reaching G sooner).
    v1 = value_iteration(mdp, theta=1e-13)[0]
    Q1 = q_from_v(mdp, v1)
    near, _ = frozenlake_mdp(map_name, 0.99999)
    Q2 = q_from_v(near, policy_iteration(near, uniform_policy(near))[1])
    Q2 = np.where(Q1 >= Q1.max(axis=1, keepdims=True) - 1e-10, Q2, -np.inf)
    return Q2.argmax(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n_eps = 300 if args.quick else 10000
    maps = ["4x4"] if args.quick else ["4x4", "8x8"]
    print(f"finite_horizon.py  seed={SEED}  quick={args.quick}  horizon H={H} (Gymnasium TimeLimit), gamma=1, "
          f"reward 1 at G  MC episodes={n_eps}")
    t0 = time.perf_counter()
    res = {}
    for map_name in maps:
        mdp, desc = frozenlake_mdp(map_name, 1.0)
        tb = time.perf_counter()
        V, pi = backward_induction(mdp, H)          # V[t] = optimal value with H - t steps to go
        tb = time.perf_counter() - tb
        print(f"\n=== FrozenLake {map_name} ===")
        print(f"  backward induction: {H} backward sweeps in {tb * 1e3:.1f} ms; "
              f"max P(reach G within {H} steps) from start = {V[0, 0]:.4f}")
        stat = {}
        for g in (0.99, 1.0):
            a = stationary_optimal(map_name, g)
            stat[g] = a
            Vs = evaluate_finite_horizon(mdp, lambda t: a, H)
            print(f"  stationary policy optimal for gamma={g} (no limit): P(reach G within {H}) = {Vs[0]:.4f}")
        # Where does the time-dependent policy differ in a way that matters?
        a_stat = stat[0.99]
        strict = np.zeros((H, mdp.S), dtype=bool)
        for t in range(H):
            Q = q_from_v(mdp, V[t + 1])
            strict[t] = Q[np.arange(mdp.S), a_stat] < Q.max(axis=1) - 1e-12
        strict[:, mdp.terminal] = False
        togo = H - np.arange(H)
        rows = np.flatnonzero(strict.any(axis=1))
        print(f"  (t, s) pairs where the gamma=0.99 stationary action is strictly worse: {int(strict.sum())} of "
              f"{H * int((~mdp.terminal).sum())}; they occur with steps-to-go in "
              f"[{togo[rows].min() if len(rows) else '-'}, {togo[rows].max() if len(rows) else '-'}]")
        p_fh, se_fh = simulate(map_name, lambda t, s: pi[t, s], n_eps)
        p_st, se_st = simulate(map_name, lambda t, s: a_stat[s], n_eps)
        print(f"  simulated in Gymnasium ({n_eps} episodes each): non-stationary optimal {p_fh:.4f} +- {se_fh:.4f};"
              f" stationary (gamma=0.99) {p_st:.4f} +- {se_st:.4f}")
        curve_fh = V[::-1, 0]                                   # index k = steps to go

        def curve(a):                                           # P(reach G within k steps), k = 0..H
            Vk, out = np.zeros(mdp.S), [0.0]
            for _ in range(H):
                Vk = mdp.R[np.arange(mdp.S), a] + mdp.P[np.arange(mdp.S), a] @ Vk
                out.append(Vk[0])
            return np.array(out)
        res[map_name] = dict(mdp=mdp, desc=desc, pi=pi, V=V, strict=strict, curve_fh=curve_fh,
                             curve_st=curve(a_stat), curve_st1=curve(stat[1.0]), a_stat=a_stat)

    # Value iteration's k-th iterate is the optimal k-step value (here with gamma = 0.99).
    # (a) (T*)^k 0 vs backward induction's V with k steps to go. In code these are literally the same
    #     recursion (both call q_from_v on the same arrays), so agreement is by construction.
    # (b) The informative check is ATTAINMENT (Theorem 3.9): evaluate the non-stationary greedy policy
    #     (pi_{H-k}, ..., pi_{H-1}) over k steps by POLICY EVALUATION on its induced Markov chain
    #     r_pi + gamma P_pi v (a different code path, no max)
    #     and compare with (T*)^k 0, the upper bound of Step 1 of Theorem 3.2.
    mdp, _ = frozenlake_mdp("8x8" if not args.quick else "4x4", 0.99)
    Vbi, pibi = backward_induction(mdp, H)
    Vk = np.zeros(mdp.S); worst = 0.0; worst_att = 0.0
    for k in range(1, H + 1):
        Vk = bellman_optimality(mdp, Vk)
        worst = max(worst, np.max(np.abs(Vk - Vbi[H - k])))
        v_pol = np.zeros(mdp.S)                             # k-step value of (pi_{H-k}, ..., pi_{H-1})
        for t in range(H - 1, H - k - 1, -1):               # evaluate backwards: r_pi + gamma P_pi v
            r_pi, P_pi = policy_model(mdp, pibi[t])
            v_pol = r_pi + mdp.gamma * (P_pi @ v_pol)
        worst_att = max(worst_att, np.max(np.abs(Vk - v_pol)))
    print(f"\n(a) max_k max_s |(T*)^k 0 - V^(k steps to go)| for k = 1..{H} (gamma = 0.99): {worst:.1e}"
          f"  (same recursion in code, so 0 by construction)")
    print(f"(b) max_k max_s |(T*)^k 0 - value of the greedy non-stationary policy over k steps| = {worst_att:.1e}"
          f"  (attainment, Theorem 3.9)")
    if not args.quick:
        make_figure(res)
    print(f"\nDone in {time.perf_counter() - t0:.1f} s")


def make_figure(res):
    from plotting import C, plt, save
    from mdps import FL_ARROWS

    fig = plt.figure(figsize=(13, 4.4))
    ax = fig.add_subplot(1, 3, 1)
    for i, name in enumerate(res):
        R = res[name]
        k = np.arange(len(R["curve_fh"]))
        ax.plot(k, R["curve_fh"], color=C[i], lw=2, label=f"{name}: optimal non-stationary $\\pi_t$")
        ax.plot(k, R["curve_st"], color=C[i], lw=1.2, ls="--", label=f"{name}: stationary ($\\gamma$=0.99-optimal)")
        if name == "8x8":
            ax.plot(k, R["curve_st1"], color=C[i], lw=1.2, ls=":",
                    label=f"{name}: stationary ($\\gamma$=1-optimal, no limit)")
    ax.set_xlabel("steps allowed, k"); ax.set_ylabel("P(reach G within k steps)")
    ax.set_title("Success probability vs horizon")
    ax.legend(fontsize=7.5)

    R = res["8x8"]
    mdp, desc, pi, strict = R["mdp"], R["desc"], R["pi"], R["strict"]
    ax = fig.add_subplot(1, 3, 2)
    togo = H - np.arange(H)
    ax.plot(togo, strict.sum(axis=1), color=C[1])
    ax.set_xlabel("steps to go"); ax.set_ylabel("# states")
    ax.set_title("8x8: states where the stationary action\nis strictly suboptimal", fontsize=10)
    ax.set_xlim(H, 0)

    ax = fig.add_subplot(1, 3, 3)
    t = H - 10                                           # 10 steps to go
    ax.imshow(np.zeros(mdp.shape), cmap="Greys", vmin=0, vmax=1)
    for s in range(mdp.S):
        r, c = divmod(s, mdp.shape[1])
        if desc[r, c] in "HG":
            ax.text(c, r, desc[r, c], ha="center", va="center", fontsize=9, color="0.4")
            continue
        if R["V"][t, s] < 1e-12:                         # G unreachable in 10 steps: every action ties
            ax.text(c, r, "\u00b7", ha="center", va="center", fontsize=14, color="0.5")
            continue
        diff = strict[t, s]
        ax.text(c, r, FL_ARROWS[pi[t, s]], ha="center", va="center", fontsize=12,
                color=C[1] if diff else "k", fontweight="bold" if diff else "normal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("8x8: optimal action with 10 steps to go (orange: stationary\naction strictly worse; dot: G unreachable, all actions tie)", fontsize=9)
    fig.tight_layout()
    save(fig, "finite_horizon.png")


if __name__ == "__main__":
    main()
