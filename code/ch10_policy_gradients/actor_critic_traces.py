"""One-step actor-critic and actor-critic with eligibility traces (tabular), on FrozenLake.

Chapter 10, Section 7 (Algorithms 10.3 and 10.4; Sutton & Barto 2018, Section 13.5).

Environment: gymnasium's FrozenLake-v1, 4x4, is_slippery=False -- a deterministic grid with
holes (episode ends, reward 0) and a single goal (episode ends, reward +1).  The reward is
sparse and delayed (the shortest path has 6 steps), so credit assignment matters.  We step the
MDP through its transition table env.unwrapped.P (identical dynamics, no wrapper overhead) and
apply the 100-step time limit of FrozenLake-v1 as a truncation: the last TD error still
bootstraps from v_hat(S') because the state is not terminal.

Agent: tabular softmax actor theta(s, a), tabular critic w(s); gamma = 0.99; uniform initial
policy.  lambda = 0 is the one-step actor-critic (Algorithm 10.3); lambda > 0 uses accumulating
traces for both actor and critic (Algorithm 10.4, lambda^theta = lambda^w = lambda).

Experiment: a parameter study in the style of Sutton & Barto -- for each lambda and step size
alpha (= alpha^theta = alpha^w), the success rate averaged over the first N_EPISODES episodes
and over n_seeds independent runs.

Outputs (full mode): figures/actor_critic_traces.png
Run:  python code/ch10_policy_gradients/actor_critic_traces.py [--quick]
"""
from __future__ import annotations

import argparse
import os
import time

import gymnasium as gym
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def load_tables():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    P = env.unwrapped.P                      # P[s][a] = [(prob, s', r, terminated)]
    nS, nA = env.observation_space.n, env.action_space.n
    nxt = np.zeros((nS, nA), dtype=int)
    rew = np.zeros((nS, nA))
    term = np.zeros((nS, nA), dtype=bool)
    for s in range(nS):
        for a in range(nA):
            (pr, s2, r, d), = P[s][a]        # deterministic: exactly one outcome
            nxt[s, a], rew[s, a], term[s, a] = s2, r, d
    max_steps = env.spec.max_episode_steps   # 100
    env.close()
    return nxt, rew, term, nS, nA, max_steps


def actor_critic(tables, lam, alpha, gamma, n_episodes, seed):
    """Algorithm 10.4 (lam = 0 gives Algorithm 10.3). Returns the per-episode return (0 or 1)."""
    nxt, rew, term, nS, nA, max_steps = tables
    rng = np.random.default_rng(seed)
    theta = np.zeros((nS, nA))
    w = np.zeros(nS)
    eye = np.eye(nA)
    returns = np.zeros(n_episodes)
    for ep in range(n_episodes):
        s = 0
        z_theta = np.zeros((nS, nA))
        z_w = np.zeros(nS)
        I = 1.0                                       # gamma^t, the discount of the update
        for t in range(max_steps):
            h = theta[s] - theta[s].max()
            pi = np.exp(h)
            pi /= pi.sum()
            a = int(rng.choice(nA, p=pi))
            s2, r, done = nxt[s, a], rew[s, a], term[s, a]
            returns[ep] += r
            v_next = 0.0 if done else w[s2]           # bootstrap unless TERMINATED
            delta = r + gamma * v_next - w[s]         # TD error: estimates the advantage
            z_w *= gamma * lam
            z_w[s] += 1.0                             # grad of v_hat(s,w) = w[s] is the indicator of s
            z_theta *= gamma * lam
            z_theta[s] += I * (eye[a] - pi)           # grad log pi(a|s) for the tabular softmax
            w += alpha * delta * z_w
            theta += alpha * delta * z_theta
            I *= gamma
            if done:
                break
            s = s2
    return returns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.quick:
        lams, alphas, n_seeds, n_eps = [0.0, 0.8], [0.2, 0.8], 3, 100
    else:
        lams, alphas, n_seeds, n_eps = [0.0, 0.5, 0.8, 0.9, 0.95], [0.05, 0.1, 0.2, 0.4, 0.8, 1.6], 50, 200
    gamma = 0.99
    print(f"seed={args.seed}  FrozenLake-v1 4x4 deterministic  gamma={gamma}  lambdas={lams}  "
          f"alphas={alphas}  seeds={n_seeds}  episodes={n_eps}")
    t0 = time.time()
    tables = load_tables()
    perf = np.zeros((len(lams), len(alphas)))
    curves = {}
    for i, lam in enumerate(lams):
        for j, alpha in enumerate(alphas):
            R = np.array([actor_critic(tables, lam, alpha, gamma, n_eps, args.seed * 10_000 + k)
                          for k in range(n_seeds)])
            perf[i, j] = R.mean()
            curves[(lam, alpha)] = R.mean(0)
        print(f"  lambda={lam:4.2f}: success rate over first {n_eps} episodes, by alpha: "
              + "  ".join(f"{a}:{p:.3f}" for a, p in zip(alphas, perf[i])) + f"  [{time.time() - t0:.0f}s]")
    print("\nBest alpha per lambda:")
    for i, lam in enumerate(lams):
        j = int(np.argmax(perf[i]))
        c = curves[(lam, alphas[j])]
        print(f"  lambda={lam:4.2f}: alpha={alphas[j]:4.2f}  mean success {perf[i, j]:.3f}  "
              f"(first 50 eps {c[:50].mean():.3f}, last 50 eps {c[-50:].mean():.3f})")
    print(f"Time {time.time() - t0:.0f}s")

    if not args.quick:
        from plot_style import setup, C
        plt = setup()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        marks = ["o", "s", "^", "D", "v"]
        styles = ["-", "--", "-.", ":", "-"]
        ax = axes[0]
        for i, lam in enumerate(lams):
            ax.plot(alphas, perf[i], marker=marks[i], color=C[i], ls=styles[i],
                    label=f"$\\lambda$ = {lam}" + (" (one-step AC)" if lam == 0 else ""))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("step size $\\alpha=\\alpha^{\\theta}=\\alpha^{w}$")
        ax.set_ylabel(f"success rate, first {n_eps} episodes")
        ax.set_title(f"Actor-critic($\\lambda$) on FrozenLake 4x4 ({n_seeds} runs)")
        ax.legend(fontsize=8)
        ax = axes[1]
        for i, lam in enumerate(lams):
            j = int(np.argmax(perf[i]))
            c = np.convolve(curves[(lam, alphas[j])], np.ones(10) / 10, mode="valid")
            ax.plot(np.arange(10, n_eps + 1), c, color=C[i], ls=styles[i],
                    label=f"$\\lambda$ = {lam}, $\\alpha$ = {alphas[j]}")
        ax.set_xlabel("episode")
        ax.set_ylabel("success rate (10-episode moving average)")
        ax.set_title("Learning curves at each $\\lambda$'s best $\\alpha$")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "actor_critic_traces.png"))
        plt.close(fig)
        print(f"figure written to {FIG_DIR}/actor_critic_traces.png")


if __name__ == "__main__":
    main()
