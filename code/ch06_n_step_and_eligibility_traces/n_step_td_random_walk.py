"""n-step TD prediction on the 19-state random walk (S&B 2018, Figure 7.2).

Chapter 06, Section 2. For n in {1, 2, 4, ..., 512} and a grid of step sizes
alpha, run n-step TD (the boxed algorithm of Section 2.2) for 10 episodes from
V = 0, and measure the RMS error over the 19 states, averaged over the first 10
episodes and over independent runs.

    G_{t:t+n} = R_{t+1} + gamma R_{t+2} + ... + gamma^{n-1} R_{t+n} + gamma^n V(S_{t+n})   (Eq. 6.1)
    V(S_t)   <- V(S_t) + alpha [G_{t:t+n} - V(S_t)]                                       (Eq. 6.2)

n = 1 is TD(0); n >= the episode length is constant-alpha (every-visit) Monte Carlo.
Implementation note: all step sizes are run in parallel (V has shape (n_alpha, 21)),
and every (n, alpha) pair sees the same episodes.

Outputs (full mode): figures/n_step_td_random_walk.png

Run:  python code/ch06_n_step_and_eligibility_traces/n_step_td_random_walk.py [--quick]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

import random_walk19 as rw

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def n_step_td_episode(V: np.ndarray, states: np.ndarray, rewards: np.ndarray,
                      n: int, alphas: np.ndarray, gamma: float = 1.0) -> None:
    """One episode of n-step TD, updating V (shape (n_alpha, 21)) in place.

    Mirrors the pseudocode: at time t we have just observed R_{t+1}, S_{t+1};
    the state visited at time tau = t - n + 1 is now n steps in the past, so its
    n-step return is complete and V(S_tau) is updated. After the terminal step
    (t >= T) no new data arrive, and we flush the remaining n - 1 updates.
    """
    T = len(rewards)
    discounts = gamma ** np.arange(n)
    for t in range(T + n - 1):
        tau = t - n + 1
        if tau < 0:
            continue
        end = min(tau + n, T)
        G = discounts[: end - tau] @ rewards[tau:end]          # sum_{i=tau+1}^{end} gamma^{i-tau-1} R_i
        if tau + n < T:                                        # bootstrap only if S_{tau+n} is not terminal
            G = G + gamma ** n * V[:, states[tau + n]]
        s = states[tau]
        V[:, s] += alphas * (G - V[:, s])


def run(ns, alphas, n_runs, n_episodes, seed):
    episodes = rw.generate_runs(n_runs, n_episodes, seed)
    err = np.zeros((len(ns), len(alphas)))          # mean RMS over episodes and runs
    for i, n in enumerate(ns):
        for run_eps in episodes:
            V = np.zeros((len(alphas), rw.N_TOTAL))
            for states, rewards in run_eps:
                n_step_td_episode(V, states, rewards, n, alphas)
                err[i] += rw.rms_error(V)
        err[i] /= n_runs * n_episodes
    return err


def plot(ns, alphas, err, path):
    from plot_style import setup, ramp, MARKERS
    plt = setup()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    cols = ramp(len(ns))
    for i, n in enumerate(ns):
        ax.plot(alphas, err[i], color=cols[i], marker=MARKERS[i % len(MARKERS)], markersize=3.5,
                label=f"n = {n}")
    ax.set_ylim(0.25, 0.55)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(r"step size $\alpha$")
    ax.set_ylabel("RMS error over 19 states,\naveraged over first 10 episodes")
    ax.set_title("n-step TD on the 19-state random walk (100 runs)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    if args.quick:
        alphas = np.round(np.linspace(0.0, 1.0, 6), 3)
        n_runs = 10
    else:
        # a uniform grid plus a few small values, so the minima of the large-n curves are resolved
        alphas = np.unique(np.round(np.concatenate([[0.01, 0.02, 0.03, 0.04], np.linspace(0.0, 1.0, 21)]), 3))
        n_runs = 100
    n_episodes = 10
    print(f"n-step TD on the 19-state random walk | seed={args.seed} runs={n_runs} "
          f"episodes={n_episodes} gamma=1 V0=0")
    print(f"n in {ns}")
    print(f"alpha in {alphas.tolist()}")

    t0 = time.time()
    err = run(ns, alphas, n_runs, n_episodes, args.seed)
    print(f"\nRMS error at alpha=0 (no learning): {err[0, 0]:.4f}  "
          f"(= sqrt(mean v^2) = {np.sqrt(np.mean(rw.TRUE_V ** 2)):.4f})")
    print("\nBest step size for each n (RMS error averaged over first 10 episodes):")
    print("   n   best alpha   RMS error")
    for i, n in enumerate(ns):
        j = int(np.argmin(err[i]))
        print(f"{n:4d}   {alphas[j]:9.2f}   {err[i, j]:.4f}")
    i_best, j_best = np.unravel_index(np.argmin(err), err.shape)
    print(f"\nOverall best: n={ns[i_best]}, alpha={alphas[j_best]:.2f}, RMS={err[i_best, j_best]:.4f}")
    print(f"TD(0) (n=1) best {err[0].min():.4f};  n=512 (MC) best {err[-1].min():.4f}")
    print(f"elapsed {time.time() - t0:.1f} s")

    if not args.quick:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, "n_step_td_random_walk.png")
        plot(ns, alphas, err, path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
