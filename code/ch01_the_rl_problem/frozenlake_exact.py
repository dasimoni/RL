"""An episodic MDP from Gymnasium: FrozenLake 4x4 (slippery), solved exactly with gamma = 1.

Chapter 01, Sections 4.4, 6, 10 and 12.

* Gymnasium's toy-text environments publish their model: ``env.unwrapped.P[s][a]`` is a
  list of (probability, next_state, reward, terminated) tuples - exactly the
  four-argument dynamics p(s', r | s, a), plus a flag marking terminal states.
  Terminal states (holes, goal) loop to themselves with reward 0: the
  absorbing-state convention of Section 4.4.
* With gamma = 1 the value of a state is the probability of eventually reaching the
  goal.  The linear system restricted to non-terminal states, (I - P~_pi) v = r~_pi
  (P~_pi = P_pi restricted to non-terminal states), is solvable because every policy
  evaluated this way terminates with probability 1 (Section 10.3); (I - P~_pi)^{-1} 1
  gives the expected episode length.
* We compare exact values with episodes simulated through the real Gymnasium API,
  and show that the default 100-step TimeLimit (a *truncation*, not part of the MDP)
  lowers the success rate of the policy that is optimal without a limit.  A policy that
  also sees the number of steps left (finite-horizon backward induction) does slightly
  better under the limit (Section 4.5).

Usage:  python code/ch01_the_rl_problem/frozenlake_exact.py [--quick]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from mdp import (FiniteMDP, bellman_optimality_backup, deterministic_policy, evaluate_policy_exact,
                 greedy_actions, greedy_policy, policy_matrices, q_from_v, uniform_policy,
                 value_iteration)

ACTIONS = ["left", "down", "right", "up"]          # FrozenLake's action order
ARROW = ["←", "↓", "→", "↑"]


def mdp_from_gymnasium(env) -> FiniteMDP:
    """Read the published model of a toy-text environment into a FiniteMDP."""
    P = env.unwrapped.P
    n_s, n_a = env.observation_space.n, env.action_space.n
    desc = env.unwrapped.desc.ravel()
    terminal = np.array([c in (b"H", b"G") for c in desc])
    outcomes = [[[(p, s2, float(r)) for p, s2, r, _ in P[s][a]] for a in range(n_a)] for s in range(n_s)]
    return FiniteMDP(n_s, n_a, outcomes, terminal=terminal,
                     state_names=[f"{s}:{desc[s].decode()}" for s in range(n_s)],
                     action_names=list(ACTIONS))


def run_episodes(env, policy_fn, n_episodes: int, seed: int):
    """Play episodes with the standard Gymnasium loop; return success, length, truncated flags."""
    success = np.zeros(n_episodes, dtype=bool)
    lengths = np.zeros(n_episodes, dtype=int)
    truncs = np.zeros(n_episodes, dtype=bool)
    for i in range(n_episodes):
        # seed only the first reset; later resets continue the same random stream
        obs, info = env.reset(seed=seed if i == 0 else None)
        done, t, ret = False, 0, 0.0
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ret += reward
            t += 1
            done = terminated or truncated
        success[i], lengths[i], truncs[i] = ret > 0, t, truncated and not terminated
    return success, lengths, truncs


def success_within(P_pi: np.ndarray, start: int, goal: int, horizons) -> np.ndarray:
    """Pr{goal reached within H steps} under a stationary policy (terminal states absorb)."""
    dist = np.zeros(P_pi.shape[0])
    dist[start] = 1.0
    out, H_max = {}, max(horizons)
    for t in range(1, H_max + 1):
        dist = dist @ P_pi
        if t in horizons:
            out[t] = dist[goal]
    return np.array([out[h] for h in horizons])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smoke test: fewer episodes, no figures")
    args = parser.parse_args()
    t0 = time.perf_counter()
    seed, gamma = 0, 1.0
    n_rand, n_opt = (2_000, 1_000) if args.quick else (20_000, 10_000)
    print(f"frozenlake_exact.py | seed={seed} gamma={gamma} episodes: random={n_rand} optimal={n_opt} "
          f"quick={args.quick}")
    rng = np.random.default_rng(seed)

    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    print(f"time limit of the registered environment: {env.spec.max_episode_steps} steps")
    mdp = mdp_from_gymnasium(env)
    desc = env.unwrapped.desc
    print("map:", " / ".join(row.tobytes().decode() for row in desc))
    print("env.unwrapped.P[14][2] (state 14, action 'right') =", env.unwrapped.P[14][2])
    print("env.unwrapped.P[5][0]  (a hole: absorbing, reward 0) =", env.unwrapped.P[5][0])
    start, goal = 0, int(np.flatnonzero(desc.ravel() == b"G")[0])
    nt = ~mdp.terminal

    # ---- equiprobable random policy
    pi_rand = uniform_policy(mdp.n_states, mdp.n_actions)
    v_rand = evaluate_policy_exact(mdp, pi_rand, gamma)        # solves on non-terminal states
    P_pi, _ = policy_matrices(mdp, pi_rand)
    P_nt = P_pi[np.ix_(nt, nt)]                                # P~_pi: non-terminal block (substochastic)
    exp_len = np.zeros(mdp.n_states)
    exp_len[nt] = np.linalg.solve(np.eye(nt.sum()) - P_nt, np.ones(nt.sum()))
    print(f"\n[random policy] spectral radius of P~_pi (P_pi restricted to non-terminal states) = "
          f"{np.max(np.abs(np.linalg.eigvals(P_nt))):.4f} < 1 -> I - P~_pi invertible")
    print(f"  exact: Pr(success from start) = v_pi(0) = {v_rand[start]:.4f}; expected episode length "
          f"= {exp_len[start]:.3f}")
    succ, lens, tr = run_episodes(env, lambda s: int(rng.integers(4)), n_rand, seed)
    se = succ.std(ddof=1) / np.sqrt(n_rand)
    print(f"  simulated through gymnasium ({n_rand} episodes): success {succ.mean():.4f} +- {1.96 * se:.4f} "
          f"(95% CI half-width; one standard error = {se:.4f}), mean length {lens.mean():.3f}, truncated by the time limit: {tr.sum()}")

    # ---- optimal policy (gamma = 1: v_* = maximal probability of reaching the goal)
    v_star, deltas = value_iteration(mdp, gamma, tol=1e-13)
    q_star = q_from_v(mdp, v_star, gamma)
    pi_star = greedy_policy(q_star, tol=1e-9)
    v_greedy = evaluate_policy_exact(mdp, deterministic_policy(pi_star, 4), gamma)
    P_star, _ = policy_matrices(mdp, deterministic_policy(pi_star, 4))
    P_nt_star = P_star[np.ix_(nt, nt)]
    len_star = np.zeros(mdp.n_states)
    len_star[nt] = np.linalg.solve(np.eye(nt.sum()) - P_nt_star, np.ones(nt.sum()))
    print(f"\n[optimal policy] value iteration with gamma=1: {len(deltas)} sweeps")
    print("  v_* (probability of eventually reaching G):")
    for r in range(4):
        print("   ", "  ".join(f"{v_star[4 * r + c]:.3f}" for c in range(4)))
    acts = greedy_actions(q_star, tol=1e-9)
    print("  greedy actions:", " / ".join(" ".join("".join(ARROW[a] for a in acts[4 * r + c])
                                                  if nt[4 * r + c] else desc[r, c].decode()
                                                  for c in range(4)) for r in range(4)))
    print(f"  exact evaluation of the greedy policy: max |v_pi - v_*| = {np.abs(v_greedy - v_star).max():.1e}; "
          f"expected episode length from start = {len_star[start]:.2f}")
    # ---- pitfall: with gamma = 1, "greedy w.r.t. v_*" does not guarantee optimality (Section 12.4)
    pi_bad = pi_star.copy()
    pi_bad[start] = 3                                   # 'up' is also in the greedy set at the start
    is_greedy = all(int(pi_bad[s]) in acts[s] for s in range(mdp.n_states) if nt[s])
    P_bad, _ = policy_matrices(mdp, deterministic_policy(pi_bad, 4))
    rho_bad = np.max(np.abs(np.linalg.eigvals(P_bad[np.ix_(nt, nt)])))
    print(f"  pitfall: the policy with 'up' at the start (and 'up' in states 1-3) is greedy w.r.t. v_*: "
          f"{is_greedy}; spectral radius of its P~_pi = {rho_bad:.4f} (I - P~_pi singular); "
          f"Pr(goal within 1000 steps) = {success_within(P_bad, start, goal, [1000])[0]:.4f}")
    v99, _ = value_iteration(mdp, 0.99, tol=1e-13)
    acts99 = greedy_actions(q_from_v(mdp, v99, 0.99), tol=1e-9)
    pi99 = greedy_policy(q_from_v(mdp, v99, 0.99), tol=1e-9)
    succ99 = evaluate_policy_exact(mdp, deterministic_policy(pi99, 4), 1.0)[start]
    ties99 = {s: [ACTIONS[a] for a in acts99[s]] for s in range(mdp.n_states) if nt[s] and len(acts99[s]) > 1}
    print(f"  with gamma = 0.99 instead: states with tied greedy actions = {len(ties99)} {ties99}; "
          f"greedy policy reaches the goal "
          f"with probability {succ99:.4f}; same actions as the gamma=1 greedy policy: {bool(np.all(pi99 == pi_star))}")

    horizons = [25, 50, 100, 200, 500, 1000]
    within = success_within(P_star, start, goal, horizons)
    print("  Pr(goal within H steps) under pi_*: "
          + ", ".join(f"H={h}: {w:.4f}" for h, w in zip(horizons, within)))
    # The best a policy can do under a 100-step limit if it may also look at the number of steps
    # left: backward induction over (state, steps left), i.e. value iteration with gamma = 1 started
    # from V_0 = 0.  Its H-th iterate is the maximal Pr(goal within H steps) - a preview of Chapter 03.
    H_max = 1000
    V_h, best_within = np.zeros(mdp.n_states), np.zeros(H_max)
    for h in range(H_max):
        V_h = bellman_optimality_backup(mdp, V_h, 1.0)
        V_h[~nt] = 0.0
        best_within[h] = V_h[start]
    print(f"  best time-aware policy (sees the steps left): Pr(goal within 100 steps) = {best_within[99]:.4f} "
          f"vs {within[2]:.4f} for the stationary pi_*")
    succ_o, lens_o, tr_o = run_episodes(env, lambda s: int(pi_star[s]), n_opt, seed + 1)
    se_o = succ_o.std(ddof=1) / np.sqrt(n_opt)
    print(f"  simulated with the default 100-step TimeLimit ({n_opt} episodes): success "
          f"{succ_o.mean():.4f} +- {1.96 * se_o:.4f} (95% CI half-width; one standard error = {se_o:.4f}; "
          f"exact within 100: {within[2]:.4f}; "
          f"without a limit: {v_star[start]:.4f}); truncated episodes: {tr_o.mean():.3f}")

    if not args.quick:
        import matplotlib.pyplot as plt
        from plotting import AQUA, BLUE, INK, INK2, ORANGE, grid_heatmap, setup_style
        setup_style()
        figdir = Path(__file__).parent / "figures"
        figdir.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), gridspec_kw={"width_ratios": [1, 1.3]})
        ax = axes[0]
        grid_heatmap(ax, np.zeros((4, 4)), r"FrozenLake 4x4: $v_\ast$ and $\pi_\ast$ ($\gamma$ = 1)",
                     fmt="", vmin=0, vmax=1)
        from matplotlib.patches import Rectangle
        deltas_rc = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        for s in range(16):
            r, c = divmod(s, 4)
            if not nt[s]:
                is_goal = desc[r, c] == b"G"
                ax.add_patch(Rectangle((c - 0.48, r - 0.48), 0.96, 0.96, color=AQUA if is_goal else "#c3c2b7"))
                ax.text(c, r, "G (goal)" if is_goal else "H (hole)", ha="center", va="center", fontsize=10,
                        color=INK, fontweight="bold")
                continue
            for a in acts[s]:
                dr, dc = deltas_rc[a]
                ax.annotate("", xy=(c + 0.3 * dc, r - 0.08 + 0.3 * dr), xytext=(c, r - 0.08),
                            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, mutation_scale=11))
            ax.text(c, r + 0.33, f"{v_star[s]:.3f}", ha="center", va="center", fontsize=8.5, color=INK2)
        ax = axes[1]
        H = np.arange(1, 1001)
        curve = success_within(P_star, start, goal, list(H))
        ax.plot(H, curve, color=BLUE, label=r"exact Pr(goal within $H$ steps) under $\pi_\ast$")
        ax.plot(H, best_within[:len(H)], color=AQUA, ls="-.", lw=1.3,
                label="best time-aware policy (sees steps left)")
        ax.axhline(v_star[start], color=ORANGE, ls="--", label=rf"$v_\ast$(start) = {v_star[start]:.3f} (no limit)")
        ax.errorbar([100], [succ_o.mean()], yerr=[1.96 * se_o], fmt="o", color=INK, ms=5, capsize=3,
                    label=f"simulated with TimeLimit(100): {succ_o.mean():.3f} (bar: 95% CI)")
        ax.axvline(100, color=INK2, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("step limit H (log scale)")
        ax.set_ylabel("probability of success")
        ax.set_title("A time limit is not part of the MDP")
        ax.legend(loc="lower right", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(figdir / "frozenlake_values.png")
        plt.close(fig)
        print(f"\nsaved figure to {figdir / 'frozenlake_values.png'}")

    print(f"\nSUMMARY: random policy success exact {v_rand[start]:.4f} vs simulated {succ.mean():.4f}; "
          f"optimal success exact {v_star[start]:.4f} (no limit), {within[2]:.4f} within 100 steps, "
          f"simulated {succ_o.mean():.4f}; time-aware optimum within 100 steps {best_within[99]:.4f}")
    print(f"done in {time.perf_counter() - t0:.2f} s")
    env.close()


if __name__ == "__main__":
    main()
