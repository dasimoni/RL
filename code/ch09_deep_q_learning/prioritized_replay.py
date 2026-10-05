"""Prioritized experience replay: why it speeds learning up, and why it needs IS weights.

Chapter 09, Section 6.  Pure numpy + the SumTree of dqn.py; tabular, so it is fast.

Part A  Blind Cliffwalk (after Schaul et al., 2016).  A chain of n states with two actions.
        'wrong' ends the episode with reward 0; 'right' moves one state on, and in the last state
        it ends the episode with reward 1.  As in the paper, the replay memory holds the
        transitions of all 2^n action sequences executed until termination: state i appears with
        2^(n-i-1) copies of each action, 2^(n+1) - 2 transitions in all, and exactly ONE of them
        carries the reward.  Tabular Q-learning (step size 1/4, gamma = 1 - 1/n, Q initialised to
        0; these three choices are ours) replays one transition per update.  We count updates until
        the mean squared error of Q over all 2n state-action pairs falls below 1% of its initial
        value, for uniform replay and proportional prioritization (alpha = 0.6 and 1).  Part A
        uses no importance-sampling weights: it measures only how fast prioritization spreads
        the reward information (the true values here are deterministic, so there is no bias for
        IS weights to correct).
Part B  Prioritization changes the data distribution, hence the fixed point.  One state-action
        pair with a terminal transition and a skewed random reward (0 with prob. 0.9, 10 with prob.
        0.1, true value 1) stored as 1,000 transitions, replayed in minibatches of 32 through the
        ReplayBuffer / PrioritizedReplay classes of dqn.py.  With priorities |delta|^alpha and IS
        exponent beta, the stationary point solves  sum_i |r_i - Q|^{alpha(1-beta)} (r_i - Q) = 0
        (Section 6.3); we compare it with simulation.

Output (full mode): figures/prioritized_replay.png
Run:  python code/ch09_deep_q_learning/prioritized_replay.py [--quick]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from scipy.optimize import brentq

from dqn import FIG_DIR, PrioritizedReplay, ReplayBuffer, SumTree

SEED = 0


# --------------------------------------------------------------------------------------------
# Part A: Blind Cliffwalk
# --------------------------------------------------------------------------------------------
def cliffwalk_memory(n):
    """All transitions produced by the 2^n action sequences (Schaul et al.'s memory)."""
    S, A, R, S2, D = [], [], [], [], []
    for i in range(n):
        c = 2 ** (n - i - 1)                     # sequences that reach state i, times 1/2 per action
        S += [i] * c; A += [0] * c; R += [0.0] * c; S2 += [i] * c; D += [1.0] * c          # wrong
        last = i == n - 1
        S += [i] * c; A += [1] * c; R += [1.0 if last else 0.0] * c                      # right
        S2 += [i if last else i + 1] * c; D += [1.0 if last else 0.0] * c
    return tuple(np.array(x) for x in (S, A, R, S2, D))


def cliffwalk_updates(n, method, rng, alpha=0.6, eta=0.25, eps=1e-3, max_updates=3_000_000):
    """Number of replay updates until MSE(Q, q) < 1% of the initial MSE (checked every 10)."""
    gamma = 1.0 - 1.0 / n
    S, A, R, S2, D = (x.tolist() for x in cliffwalk_memory(n))
    M = len(S)
    q_true = np.zeros((n, 2))
    q_true[:, 1] = gamma ** (n - 1 - np.arange(n))      # right: reward 1 after n-1-i more steps
    Q = np.zeros((n, 2))
    mse0 = np.mean((Q - q_true) ** 2)
    if method != "uniform":
        tree = SumTree(M)
        tree.update(np.arange(M), np.ones(M))            # new transitions: maximal priority
    u = 0
    while u < max_updates:
        draws = rng.random(10_000)                       # pre-drawn uniforms, used one per update
        for x in draws:
            u += 1
            if method == "uniform":
                j = min(int(x * M), M - 1)
            else:
                j = min(tree.find_one(x * tree.total), M - 1)    # P(j) = p_j / sum_k p_k
            s, a, s2 = S[j], A[j], S2[j]
            delta = R[j] + gamma * (1.0 - D[j]) * max(Q[s2, 0], Q[s2, 1]) - Q[s, a]
            Q[s, a] += eta * delta
            if method != "uniform":
                tree.update_one(j, (abs(delta) + eps) ** alpha)
            if u % 10 == 0 and np.mean((Q - q_true) ** 2) < 0.01 * mse0:
                return u, M
    return max_updates, M


# --------------------------------------------------------------------------------------------
# Part B: the fixed point of prioritized replay with and without importance sampling
# --------------------------------------------------------------------------------------------
def fixed_point(rewards, alpha, beta, eps=1e-3):
    """Root of  sum_i P_i w_i (r_i - Q) = 0  with P_i ~ (|r_i - Q| + eps)^alpha, w_i ~ P_i^-beta."""
    f = lambda q: np.sum((np.abs(rewards - q) + eps) ** (alpha * (1 - beta)) * (rewards - q))
    return brentq(f, rewards.min() + 1e-9, rewards.max() - 1e-9)


def simulate_per(rewards, alpha, beta, rng, n_updates, eta=0.01, batch=32):
    """Replay terminal one-step transitions through dqn.py's own replay classes.

    Q <- Q + eta * mean_b( w_b * delta_b ),  delta_b = r_b - Q  (terminal, so no bootstrap),
    with w_b the IS weights of PrioritizedReplay.sample (normalised by the minibatch maximum) and
    priorities refreshed for the sampled transitions only, exactly as in the DQN agent.
    """
    N = len(rewards)
    buf = ReplayBuffer(N, 1, rng) if alpha == 0 else PrioritizedReplay(N, 1, rng, alpha=alpha, eps=1e-3)
    for r in rewards:
        buf.add(np.zeros(1), 0, r, np.zeros(1), 1.0, 0.0)
    Q, traj = 0.0, np.zeros(n_updates)
    for u in range(n_updates):
        b = buf.sample(batch, beta)
        delta = b["ret"].numpy() - Q
        Q += eta * float(np.mean(b["weights"].numpy() * delta))
        buf.update_priorities(b["idx"], np.abs(delta))
        traj[u] = Q
    return traj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    ns = [4, 6, 8] if args.quick else [4, 6, 8, 10, 12, 14]
    n_seeds = 3 if args.quick else 10
    methods = {"uniform": dict(method="uniform"), "PER alpha=0.6": dict(method="per", alpha=0.6),
               "PER alpha=1": dict(method="per", alpha=1.0)}
    print(f"Seed {SEED}.  Part A: Blind Cliffwalk, n in {ns}, {n_seeds} seeds, "
          "eta=1/4, gamma=1-1/n, eps=1e-3; median [min, max] updates to reach 1% of the initial MSE")
    A = {name: [] for name in methods}
    sizes = []
    for n in ns:
        line = f"  n={n:2d} (memory {2 ** (n + 1) - 2:6d}):"
        for name, kw in methods.items():
            res = [cliffwalk_updates(n, rng=rng, **kw)[0] for _ in range(n_seeds)]
            A[name].append(res)
            line += f"  {name} {int(np.median(res)):8d} [{min(res)}, {max(res)}]"
        sizes.append(2 ** (n + 1) - 2)
        print(line, flush=True)
    for name in methods:
        ratio = np.median(A[name][-1]) / sizes[-1]
        print(f"  {name}: median updates / memory size at n={ns[-1]}: {ratio:.2f}")

    print("\nPart B: rewards 0 (p=0.9) / 10 (p=0.1), true Q = 1.0; step size 0.01, minibatch 32, "
          "IS weights normalised by the minibatch max; mean Q over the second half of training")
    rewards = np.r_[np.zeros(900), np.full(100, 10.0)]
    n_upd = 5_000 if args.quick else 40_000
    B = {}
    for alpha, beta in [(0, 0), (1.0, 0.0), (1.0, 1.0), (0.6, 0.0), (0.6, 0.4), (0.6, 1.0)]:
        traj = simulate_per(rewards, alpha, beta, rng, n_upd)
        fp = 1.0 if alpha == 0 else fixed_point(rewards, alpha, beta)
        B[(alpha, beta)] = (traj, fp)
        label = "uniform" if alpha == 0 else f"alpha={alpha}, beta={beta}"
        print(f"  {label:22s} simulated Q {traj[n_upd // 2:].mean():.3f}   predicted fixed point {fp:.3f}")
    print(f"Total runtime {time.perf_counter() - t0:.0f}s")
    if args.quick:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import setup, C, GREY
    setup()
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
    for i, name in enumerate(methods):
        res = np.array(A[name])
        med = np.median(res, 1)
        ax[0].plot(sizes, med, ["-o", "--s", "-.^"][i], ms=4, color=C[i], label=name)
        ax[0].fill_between(sizes, res.min(1), res.max(1), color=C[i], alpha=0.12, lw=0)
    ax[0].plot(sizes, sizes, ":", color=GREY, lw=1, label="= memory size")
    ax[0].set(xscale="log", yscale="log", xlabel="replay memory size $2^{n+1}-2$",
              ylabel="updates to reach 1% of initial MSE",
              title=f"A: Blind Cliffwalk (median, min-max over {n_seeds} seeds)")
    ax[0].legend(loc="upper left")
    styles = {(0, 0): ("uniform", C[0], "-"), (1.0, 0.0): (r"$\alpha=1$, $\beta=0$", C[1], "-"),
              (1.0, 1.0): (r"$\alpha=1$, $\beta=1$", C[1], "--"), (0.6, 0.0): (r"$\alpha=0.6$, $\beta=0$", C[2], "-"),
              (0.6, 0.4): (r"$\alpha=0.6$, $\beta=0.4$", C[3], "-."), (0.6, 1.0): (r"$\alpha=0.6$, $\beta=1$", C[2], "--")}
    k = 2000                                                         # moving average window
    for key, (traj, fp) in B.items():
        lab, col, ls = styles[key]
        sm = np.convolve(traj, np.ones(k) / k, mode="valid")
        ax[1].plot(np.arange(len(sm)) + k, sm, ls, color=col, lw=1.6, label=f"{lab} (fixed point {fp:.2f})")
    ax[1].axhline(1.0, color="k", lw=0.8, ls=":")
    ax[1].set(xlabel="replay updates", ylabel=f"Q (moving average over {k})", ylim=(0, 3.6),
              title="B: PER without IS weights converges to the wrong value")
    ax[1].legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, "prioritized_replay.png")
    fig.savefig(path)
    print("saved", path)


if __name__ == "__main__":
    main()
