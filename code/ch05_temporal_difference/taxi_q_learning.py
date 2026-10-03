"""Tabular Q-learning on Gymnasium's Taxi-v4: a real environment loop done right.

Chapter 05, Section 13.

What this script shows
  1. The canonical Gymnasium interaction loop with the 5-tuple
         obs, reward, terminated, truncated, info = env.step(action)
     and the rule:  bootstrap through truncation, never through termination.
         target = R                          if terminated (true end of the task)
         target = R + gamma * max_a Q(S', a) otherwise, *including* truncated=True
  2. That tabular Q-learning finds an (essentially) optimal policy: we compute the true optimal
     action values by value iteration on the environment's own transition table env.unwrapped.P
     (Chapter 03) and compare the learned greedy policy against them from every start state.
  3. What happens if you instead treat a time-out as termination (a common bug):
       (a) on Taxi with the default 200-step limit (time-outs are rare once learning gets going);
       (b) on a tiny custom Gymnasium env, "stay-or-quit", wrapped in a TimeLimit, where the bug
           flips the learned policy.

Taxi-v4 (default settings) is deterministic: 500 states, 6 actions, reward -1 per step,
+20 for a correct drop-off (terminates the episode), -10 for an illegal pickup/drop-off.
gym.make wraps it in a TimeLimit of 200 steps, which sets truncated=True.

Run:  python code/ch05_temporal_difference/taxi_q_learning.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def epsilon_greedy(q_s: np.ndarray, eps: float, rng: np.random.Generator) -> int:
    if rng.random() < eps:
        return int(rng.integers(len(q_s)))
    best = np.flatnonzero(q_s == q_s.max())          # random tie-breaking
    return int(rng.choice(best))


def q_learning(env: gym.Env, n_episodes: int, alpha: float, gamma: float, eps: float, seed: int,
               truncation_as_termination: bool = False, eval_fn=None, eval_every: int = 100):
    """Tabular Q-learning (Section 8 pseudocode) with correct terminated/truncated handling.

    Returns the Q table, the undiscounted return of every training episode, and a boolean
    array marking the episodes that ended by truncation (time limit) rather than termination.
    If eval_fn is given, eval_fn(Q) is also called after every `eval_every` episodes and the
    list of its results is returned as a fourth value (used to plot the greedy policy's return
    next to the epsilon-greedy behaviour's return; it does not touch the env or the RNG).
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    returns = np.zeros(n_episodes)
    was_truncated = np.zeros(n_episodes, dtype=bool)
    evals = []
    for ep in range(n_episodes):
        # Seed the env's RNG once; later resets continue that RNG stream (reproducible).
        obs, info = env.reset(seed=seed if ep == 0 else None)
        G = 0.0
        while True:
            a = epsilon_greedy(Q[obs], eps, rng)
            next_obs, r, terminated, truncated, info = env.step(a)
            G += r
            if terminated or (truncation_as_termination and truncated):
                target = r                                   # no future after a true terminal state
            else:
                target = r + gamma * Q[next_obs].max()       # bootstrap -- also when truncated!
            Q[obs, a] += alpha * (target - Q[obs, a])
            if terminated or truncated:
                was_truncated[ep] = truncated and not terminated
                break
            obs = next_obs
        returns[ep] = G
        if eval_fn is not None and (ep + 1) % eval_every == 0:
            evals.append(eval_fn(Q))
    if eval_fn is not None:
        return Q, returns, was_truncated, evals
    return Q, returns, was_truncated


# ------------------------------------------------------------------------------------------
# Exact evaluation using the environment's transition table (model used ONLY for evaluation)
# ------------------------------------------------------------------------------------------
def value_iteration(P, n_s, n_a, gamma, tol=1e-10):
    """q* by value iteration on the deterministic/stochastic table P[s][a] = [(p, s', r, done)]."""
    Q = np.zeros((n_s, n_a))
    while True:
        V = Q.max(axis=1)
        Q_new = np.zeros_like(Q)
        for s in range(n_s):
            for a in range(n_a):
                Q_new[s, a] = sum(p * (r + (0.0 if done else gamma * V[s2])) for p, s2, r, done in P[s][a])
        if np.max(np.abs(Q_new - Q)) < tol:
            return Q_new
        Q = Q_new


def greedy_rollout_return(P, Q, s, max_steps=200):
    """Undiscounted return of the greedy policy from s (Taxi is deterministic), with the time limit.

    The read-out is deterministic: np.argmax breaks ties by the lowest action index (the
    learning agent itself breaks ties at random).  Exact ties only remain in rarely visited
    states, where they can make the read-out loop until the time limit."""
    G = 0.0
    for _ in range(max_steps):
        a = int(np.argmax(Q[s]))
        (p, s2, r, done), = P[s][a]
        G += r
        if done:
            return G, True
        s = s2
    return G, False


def evaluate(P, Q, start_states):
    res = [greedy_rollout_return(P, Q, s) for s in start_states]
    G = np.array([g for g, _ in res])
    success = np.array([ok for _, ok in res])
    return G, success


# ------------------------------------------------------------------------------------------
# A toy continuing task cut into episodes by a TimeLimit wrapper
# ------------------------------------------------------------------------------------------
class StayOrQuit(gym.Env):
    """One state.  Action 0 = 'stay': reward +1 and the task continues forever.
    Action 1 = 'quit': reward +quit_reward and the task terminates.

    With gamma = 0.9: q*(stay) = 1 / (1 - 0.9) = 10 and q*(quit) = 6, so staying is optimal.
    The task itself never ends when you stay; only the TimeLimit wrapper cuts episodes.
    """

    def __init__(self, quit_reward: float = 6.0):
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(2)
        self.quit_reward = quit_reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return 0, {}

    def step(self, action):
        if action == 1:
            return 0, self.quit_reward, True, False, {}      # terminated: a real end
        return 0, 1.0, False, False, {}                       # never terminates by itself


def toy_experiment(truncation_as_termination: bool, horizon: int, n_episodes: int, seed: int,
                   alpha=0.05, gamma=0.9, eps=0.1):
    env = TimeLimit(StayOrQuit(), max_episode_steps=horizon)  # sets truncated=True at step `horizon`
    Q, _, _ = q_learning(env, n_episodes, alpha, gamma, eps, seed,
                         truncation_as_termination=truncation_as_termination)
    return Q[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    n_seeds = 2 if args.quick else 5
    n_episodes = 600 if args.quick else 5000
    alpha, gamma, eps = 0.1, 0.99, 0.1
    print(f"[taxi_q_learning] seed={args.seed} seeds={n_seeds} episodes={n_episodes} "
          f"alpha={alpha} gamma={gamma} epsilon={eps} (Taxi-v4, default 200-step TimeLimit)")
    t0 = time.time()

    env = gym.make("Taxi-v4")
    P = env.unwrapped.P
    n_s, n_a = env.observation_space.n, env.action_space.n
    start_states = np.flatnonzero(env.unwrapped.initial_state_distrib > 0)   # 300 start states

    Q_star = value_iteration(P, n_s, n_a, gamma)
    G_opt, ok_opt = evaluate(P, Q_star, start_states)
    print(f"Optimal policy (value iteration on env.unwrapped.P, gamma={gamma}): mean undiscounted return "
          f"over the {len(start_states)} start states = {G_opt.mean():.3f} (success {ok_opt.mean():.0%})")

    variants = {
        "correct (bootstrap on truncation)": dict(truncation_as_termination=False),
        "bug: truncation treated as termination": dict(truncation_as_termination=True),
    }
    curves, greedy_curves = {}, {}
    eval_every = 100
    print("\n--- Taxi-v4, Q-learning, 200-step TimeLimit ---")
    print("  variant                                  | train return  | greedy policy from all 300 start states:"
          " mean return, success, optimal | truncated episodes")
    print("                                           | last 100 ep.  |")
    for vname, kw in variants.items():
        all_ret, evals, succ, match, n_tr, g_curves = [], [], [], [], [], []
        for k in range(n_seeds):
            if args.quick:
                Q, rets, ntr = q_learning(env, n_episodes, alpha, gamma, eps, seed=args.seed + k, **kw)
            else:   # also track the greedy policy's mean return every `eval_every` episodes
                Q, rets, ntr, g_curve = q_learning(
                    env, n_episodes, alpha, gamma, eps, seed=args.seed + k,
                    eval_fn=lambda Q_: evaluate(P, Q_, start_states)[0].mean(), eval_every=eval_every, **kw)
                g_curves.append(g_curve)
            G, ok = evaluate(P, Q, start_states)
            all_ret.append(rets); evals.append(G.mean()); succ.append(ok.mean())
            match.append(np.mean(np.isclose(G, G_opt))); n_tr.append(ntr)   # ntr: bool per episode
        all_ret = np.array(all_ret)
        curves[vname] = all_ret.mean(axis=0)
        if g_curves:
            greedy_curves[vname] = np.mean(g_curves, axis=0)
        print(f"  {vname:40s} | {all_ret[:, -100:].mean():8.2f}      | {np.mean(evals):7.3f}, "
              f"{np.mean(succ):6.1%}, {np.mean(match):6.1%}                     | "
              f"{np.mean([t.sum() for t in n_tr]):.1f} of {n_episodes}")
        print(f"      per-seed greedy mean return: " + ", ".join(f"{e:.2f}" for e in evals))
        print(f"      per-seed truncated episodes (all / after episode 500): "
              + ", ".join(f"{t.sum()}/{t[500:].sum()}" for t in n_tr))
        if g_curves:
            gc = greedy_curves[vname]
            print("      greedy-policy mean return during training (all 300 starts, mean of seeds) at episodes "
                  + ", ".join(f"{e}: {gc[e // eval_every - 1]:.2f}" for e in (500, 1000, 2000, 3000, 5000)
                              if e <= n_episodes))

    # ---- toy: stay-or-quit --------------------------------------------------------------
    horizon, toy_eps = 10, 3000
    print(f"\n--- Toy 'stay-or-quit' (gamma=0.9, TimeLimit={horizon}, alpha=0.05, eps=0.1, "
          f"{toy_eps} episodes): q*(stay)=10, q*(quit)=6 ---")
    toy = {}
    for vname, kw in variants.items():
        qs = np.array([toy_experiment(kw["truncation_as_termination"], horizon, toy_eps, seed=args.seed + k)
                       for k in range(n_seeds)])
        toy[vname] = qs
        greedy = ["stay" if q[0] > q[1] else "quit" for q in qs]
        print(f"  {vname:40s} | Q(stay) = {qs[:, 0].mean():6.3f}  Q(quit) = {qs[:, 1].mean():6.3f} "
              f"| greedy action per seed: {greedy}")

    if not args.quick:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        k = 50
        styles = {"correct (bootstrap on truncation)": ("tab:blue", "-"),
                  "bug: truncation treated as termination": ("tab:orange", "--")}
        x_eval = np.arange(1, n_episodes // eval_every + 1) * eval_every
        for ax, zoom in zip(axes, (False, True)):
            for vname, c in curves.items():
                sm = np.convolve(c, np.ones(k) / k, mode="valid")
                col, ls = styles[vname]
                short = "correct" if vname.startswith("correct") else "bug"
                ax.plot(np.arange(k, n_episodes + 1), sm, color=col, ls=ls,
                        label=f"ε-greedy behaviour ({short})")
                if zoom:
                    ax.plot(x_eval, greedy_curves[vname], color=col, ls=ls, marker="o", ms=3, lw=1,
                            alpha=0.8, label=f"greedy policy, all 300 starts ({short})")
            ax.axhline(G_opt.mean(), color="k", ls=":", lw=1,
                       label=f"optimal policy, greedy ({G_opt.mean():.2f})")
            ax.set_xlabel("training episode")
            ax.grid(alpha=0.3)
        axes[0].set_ylim(-220, 15)
        axes[0].set_ylabel(f"undiscounted return\n(mean of {n_seeds} seeds; behaviour: {k}-episode moving avg)")
        axes[0].set_title("Q-learning on Taxi-v4 (α=0.1, γ=0.99, ε=0.1)")
        axes[0].legend(fontsize=8, loc="lower right")
        axes[1].set_xlim(500, n_episodes)
        axes[1].set_ylim(-5, 10)
        axes[1].set_title(f"Zoom: behaviour vs greedy policy (evaluated every {eval_every} episodes)")
        axes[1].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "taxi_learning_curve.png"), dpi=110)
        plt.close(fig)
        print(f"\nFigure written to {FIG_DIR}/taxi_learning_curve.png")
    print(f"\nRuntime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
