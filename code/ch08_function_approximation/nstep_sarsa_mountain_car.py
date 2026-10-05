"""Exercise 12 of Chapter 08: n-step semi-gradient SARSA on Gymnasium's MountainCar-v0
(cf. Sutton & Barto Sec. 10.2, Figs 10.3-10.4).

Same agent as mountain_car_sarsa.py (8 tilings of 8x8 tiles, one weight vector per
action, epsilon = 0, gamma = 1, w0 = 0), but the target of the update at time tau is
the n-step return

    G_{tau:tau+n} = R_{tau+1} + ... + R_{min(tau+n, T)}  (+ q_hat(S_{tau+n}, A_{tau+n}, w)  if tau+n < T).

Episode ends (Section 11.2 of the chapter):
  * terminated: S_T is terminal, q_hat(S_T, .) = 0, so the remaining tau's are flushed
    with no bootstrap term;
  * truncated (and not terminated): S_T is an ordinary state. We draw A_T from the
    policy ONCE, and every remaining tau bootstraps from q_hat(S_T, A_T, w) evaluated
    with the CURRENT weights at the time of its update (the flush keeps changing w).
  Gymnasium can set terminated and truncated together (goal reached on the last allowed
  step); then the state is terminal and we must not bootstrap -- hence `trunc and not term`.

Experiment: n in {1, 4, 8}, a sweep over alpha, Gymnasium's default 200-step limit,
mean steps per episode over the first 100 episodes.

Run:  python code/ch08_function_approximation/nstep_sarsa_mountain_car.py [--quick]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread: the machine is shared
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import time

import gymnasium as gym
import numpy as np

from mountain_car_sarsa import SarsaAgent

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
INF = float("inf")


def nstep_episode(env, agent, n, seed=None):
    """One episode of n-step semi-gradient SARSA (gamma = 1). Returns its length T.
    Whole lists are kept for clarity instead of circular buffers of size n+1."""
    obs, _ = env.reset(seed=seed)
    idx = [agent.features(obs)]                 # idx[t]  = active tiles of S_t
    acts = [agent.act(idx[0])[0]]               # acts[t] = A_t
    rews = [0.0]                                # rews[t] = R_t (rews[0] unused)
    T, t = INF, 0
    boot = None                                 # (tiles of S_T, A_T) if truncated, not terminated
    while True:
        if t < T:
            obs, r, term, trunc, _ = env.step(acts[t])
            rews.append(r)
            if term or trunc:
                T = t + 1
                if trunc and not term:
                    last_idx = agent.features(obs)
                    boot = (last_idx, agent.act(last_idx)[0])   # A_T drawn once from the policy
            else:
                idx.append(agent.features(obs))
                acts.append(agent.act(idx[-1])[0])
        tau = t - n + 1                         # the time whose estimate is updated
        if tau >= 0:
            G = sum(rews[tau + 1:int(min(tau + n, T)) + 1])
            if tau + n < T:
                G += agent.q(idx[tau + n])[acts[tau + n]]
            elif boot is not None:              # truncated: bootstrap from S_T with the current w
                G += agent.q(boot[0])[boot[1]]
            agent.update(idx[tau], acts[tau], G)
        if tau == T - 1:
            return int(T)
        t += 1


def run(n, alpha, n_runs, n_episodes, seed, max_steps=200):
    steps = np.zeros((n_runs, n_episodes))
    for k in range(n_runs):
        env = gym.make("MountainCar-v0", max_episode_steps=max_steps)
        agent = SarsaAgent(alpha=alpha, seed=seed + k)
        for ep in range(n_episodes):
            steps[k, ep] = nstep_episode(env, agent, n, seed=seed + 10_000 * k if ep == 0 else None)
        env.close()
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figures")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ns = [1, 4, 8]
    alphas = [0.1, 0.2, 0.4, 0.7, 1.0]
    n_runs, n_episodes = (10, 100)
    if args.quick:
        alphas, n_runs, n_episodes = [0.4], 1, 10
    print(f"seed={args.seed} quick={args.quick}; n-step semi-gradient SARSA, 8 tilings of 8x8, epsilon=0, gamma=1,"
          f" default 200-step limit; n in {ns}, alpha (x1/8) in {alphas}; {n_runs} runs x {n_episodes} episodes")
    res = {}
    for n in ns:
        for a in alphas:
            t0 = time.time()
            s = run(n, a, n_runs, n_episodes, args.seed)
            res[(n, a)] = s
            print(f"  n={n} alpha={a}/8: mean steps/episode over episodes 1-{n_episodes} = {s.mean():6.1f}"
                  f" (first 10: {s[:, :10].mean():6.1f}, last 20: {s[:, -20:].mean():6.1f});"
                  f" truncated episodes = {(s >= 200).mean():.2f}  [{time.time() - t0:.1f}s]")
    print(f"\nSummary: best alpha per n (mean steps/episode over episodes 1-{n_episodes},"
          f" +- standard error over {n_runs} runs):")
    for n in ns:
        best = min(alphas, key=lambda a: res[(n, a)].mean())
        per_run = res[(n, best)].mean(1)
        se = per_run.std(ddof=1) / np.sqrt(n_runs) if n_runs > 1 else float("nan")
        print(f"  n={n}: best alpha = {best}/8, mean = {per_run.mean():.1f} +- {se:.1f}")

    if args.quick:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for n in ns:
        m = [res[(n, a)].mean() for a in alphas]
        se = [res[(n, a)].mean(1).std(ddof=1) / np.sqrt(n_runs) for a in alphas]
        ax.errorbar(alphas, m, yerr=se, marker="o", capsize=3, label=f"n = {n}")
    ax.set_xlabel(r"$\alpha \times$ number of tilings (8)")
    ax.set_ylabel(f"steps per episode, mean of first {n_episodes}")
    ax.set_title(f"n-step semi-gradient SARSA, MountainCar-v0 ({n_runs} runs, ±1 s.e.)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "nstep_sarsa_mountain_car.png"), dpi=110)
    print(f"figure saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
