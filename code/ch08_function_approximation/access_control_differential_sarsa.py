"""Differential semi-gradient SARSA on the access-control queuing task
(Chapter 08, Section 12; cf. Sutton & Barto Example 10.2, Fig 10.5).

Task (continuing, average reward): 10 servers; customers of priority 1, 2, 4, 8
arrive at the head of a queue (each priority equally likely). Each step the agent
accepts (reward = priority, one server becomes busy) or rejects (reward 0) the
head customer; with no free server the customer is always rejected. Each busy
server becomes free with probability 0.06 per step.

Learner: differential semi-gradient one-step SARSA (Section 12.2 box) with
one-hot features over (free servers, priority, action) -- i.e. the tabular
special case of the linear method -- alpha = 0.01, beta = 0.01, epsilon = 0.1.

Check: because the model is tiny we also solve the average-reward MDP exactly by
relative value iteration and evaluate the learned greedy policy exactly, so we can
say how close the learner got to the optimal average reward r*.

Run:  python code/ch08_function_approximation/access_control_differential_sarsa.py [--quick]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread: the machine is shared
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import time

import numpy as np
from scipy.stats import binom

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
N_SERVERS = 10
PRIORITIES = [1, 2, 4, 8]
P_FREE = 0.06
REJECT, ACCEPT = 0, 1


def feature_index(free, pri, a):
    """One-hot feature: x(s,a) has a single 1 at this index, so q_hat(s,a,w) = w[index]."""
    return (free * len(PRIORITIES) + pri) * 2 + a


def differential_sarsa(n_steps, alpha=0.01, beta=0.01, epsilon=0.1, seed=0, block=100_000):
    rng = np.random.default_rng(seed)
    w = [0.0] * ((N_SERVERS + 1) * len(PRIORITIES) * 2)
    r_bar = 0.0
    r_bar_hist = []
    visits = [0] * (N_SERVERS + 1)                               # how often each free-server count occurs

    def choose(free, pri, u_explore, u_action):
        if free == 0:
            return REJECT                                        # only legal action
        if u_explore < epsilon:
            return ACCEPT if u_action < 0.5 else REJECT
        qa, qr = w[feature_index(free, pri, ACCEPT)], w[feature_index(free, pri, REJECT)]
        if qa == qr:
            return ACCEPT if u_action < 0.5 else REJECT          # random tie-breaking
        return ACCEPT if qa > qr else REJECT

    free, pri = N_SERVERS, int(rng.integers(4))
    a = ACCEPT
    t = 0
    while t < n_steps:
        m = min(block, n_steps - t)
        # pre-draw randomness in blocks (pure-Python loop below stays fast)
        freed_tab = rng.binomial(np.arange(N_SERVERS + 1)[None, :], P_FREE, size=(m, N_SERVERS + 1)).tolist()
        new_pri = rng.integers(4, size=m).tolist()
        u1 = rng.random(m).tolist()
        u2 = rng.random(m).tolist()
        for k in range(m):
            # --- environment step ---
            if a == ACCEPT:
                reward = PRIORITIES[pri]
                free2 = free - 1
            else:
                reward = 0.0
                free2 = free
            free2 += freed_tab[k][N_SERVERS - free2]                 # busy servers become free
            pri2 = new_pri[k]
            # --- differential semi-gradient SARSA update ---
            a2 = choose(free2, pri2, u1[k], u2[k])
            i, i2 = feature_index(free, pri, a), feature_index(free2, pri2, a2)
            delta = reward - r_bar + w[i2] - w[i]                    # differential TD error
            r_bar += beta * delta                                    # average-reward estimate
            w[i] += alpha * delta                                    # grad q_hat = one-hot x(S,A)
            free, pri, a = free2, pri2, a2
            visits[free] += 1
            if (t + k) % 1000 == 0:
                r_bar_hist.append(r_bar)
        t += m
    return np.array(w), r_bar, np.array(r_bar_hist), np.array(visits)


# --------------------------------------------------------------------------- #
# Exact model, relative value iteration and exact policy evaluation
# --------------------------------------------------------------------------- #
def exact_model():
    """P[a][s, s'] and r[a][s] for s = free * 4 + pri."""
    nS = (N_SERVERS + 1) * 4
    P = np.zeros((2, nS, nS))
    r = np.zeros((2, nS))
    for free in range(N_SERVERS + 1):
        for pri in range(4):
            s = free * 4 + pri
            for a in (REJECT, ACCEPT):
                f = free - 1 if (a == ACCEPT and free > 0) else free
                r[a, s] = PRIORITIES[pri] if (a == ACCEPT and free > 0) else 0.0
                busy = N_SERVERS - f
                for k in range(busy + 1):
                    for p2 in range(4):
                        P[a, s, (f + k) * 4 + p2] += binom.pmf(k, busy, P_FREE) / 4
    return P, r


def relative_value_iteration(P, r, iters=20_000, tol=1e-12):
    h = np.zeros(P.shape[1])
    for _ in range(iters):
        q = r + P @ h                       # shape (2, nS)
        h_new = q.max(0)
        gain = h_new[0]                     # reference state (0 free, priority 1)
        h_new = h_new - gain
        if np.abs(h_new - h).max() < tol:
            break
        h = h_new
    return gain, h, (r + P @ h).argmax(0)


def average_reward_of(policy_probs, P, r):
    """Exact average reward of a stochastic policy pi(a|s) (array (nS, 2))."""
    Ppi = np.einsum("sa,ast->st", policy_probs, P)
    rpi = np.einsum("sa,as->s", policy_probs, r)
    vals, vecs = np.linalg.eig(Ppi.T)
    mu = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    mu /= mu.sum()
    return float(mu @ rpi)


def simulate_policy(policy, n_steps, seed):
    """Average reward of a deterministic policy (array over s = free*4 + pri) by simulation,
    used only to cross-check the exact model."""
    rng = np.random.default_rng(seed)
    free, pri, total = N_SERVERS, 0, 0.0
    freed_tab = rng.binomial(np.arange(N_SERVERS + 1)[None, :], P_FREE, size=(n_steps, N_SERVERS + 1)).tolist()
    pris = rng.integers(4, size=n_steps).tolist()
    for t in range(n_steps):
        if free > 0 and policy[free * 4 + pri] == ACCEPT:
            total += PRIORITIES[pri]
            free -= 1
        free += freed_tab[t][N_SERVERS - free]
        pri = pris[t]
    return total / n_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figures")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n_steps = 200_000 if args.quick else 2_000_000
    print(f"seed={args.seed} quick={args.quick}; {n_steps:,} steps, alpha=0.01, beta=0.01, epsilon=0.1")

    t0 = time.time()
    w, r_bar, hist, visits = differential_sarsa(n_steps, seed=args.seed)
    print(f"learning took {time.time() - t0:.1f}s; final R_bar = {r_bar:.3f}; "
          f"R_bar averaged over the second half of training = {hist[len(hist) // 2:].mean():.3f}")
    print("fraction of steps with k free servers, k = 0..10:", np.round(visits / visits.sum(), 4).tolist())

    P, r = exact_model()
    nS = P.shape[1]
    r_star, h, pi_star = relative_value_iteration(P, r)
    greedy = np.zeros((nS, 2))
    eps_greedy = np.zeros((nS, 2))
    for free in range(N_SERVERS + 1):
        for pri in range(4):
            s = free * 4 + pri
            if free == 0:
                greedy[s, REJECT] = eps_greedy[s, REJECT] = 1.0
                continue
            best = ACCEPT if w[feature_index(free, pri, ACCEPT)] > w[feature_index(free, pri, REJECT)] else REJECT
            greedy[s, best] = 1.0
            eps_greedy[s] = 0.05
            eps_greedy[s, best] += 0.9
    opt = np.zeros((nS, 2)); opt[np.arange(nS), pi_star] = 1.0
    print(f"exact optimal average reward r* (relative value iteration) = {r_star:.4f}")
    n_sim = 100_000 if args.quick else 1_000_000
    print(f"  cross-check: simulating the optimal policy for {n_sim:,} steps gives "
          f"{simulate_policy(pi_star, n_sim, args.seed + 1):.4f}")
    print(f"exact average reward of the learned greedy policy        = {average_reward_of(greedy, P, r):.4f}")
    print(f"exact average reward of the learned eps-greedy policy    = {average_reward_of(eps_greedy, P, r):.4f}"
          f"  (what R_bar estimates)")
    print(f"exact average reward of 'always accept if possible'      = "
          f"{average_reward_of(np.tile([0.0, 1.0], (nS, 1)), P, r):.4f}")

    print("\nlearned greedy policy (1 = accept), rows = priority, columns = free servers 1..10")
    for pri in range(4):
        row = [int(w[feature_index(f, pri, ACCEPT)] > w[feature_index(f, pri, REJECT)]) for f in range(1, N_SERVERS + 1)]
        row_opt = [int(pi_star[f * 4 + pri]) for f in range(1, N_SERVERS + 1)]
        print(f"  priority {PRIORITIES[pri]}: learned {row}   optimal {row_opt}")

    if args.quick:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    pol = np.array([[int(w[feature_index(f, p, ACCEPT)] > w[feature_index(f, p, REJECT)])
                     for f in range(1, N_SERVERS + 1)] for p in range(4)])
    axes[0].imshow(pol, cmap="RdYlGn", origin="lower", aspect="auto", vmin=0, vmax=1)
    for p in range(4):
        for f in range(N_SERVERS):
            axes[0].text(f, p, "A" if pol[p, f] else "R", ha="center", va="center", fontsize=9)
    axes[0].set_xticks(range(N_SERVERS), range(1, N_SERVERS + 1))
    axes[0].set_yticks(range(4), PRIORITIES)
    axes[0].set_xlabel("number of free servers")
    axes[0].set_ylabel("priority")
    axes[0].set_title("Learned greedy policy (A = accept, R = reject)")
    for p in range(4):
        vals = [max(w[feature_index(f, p, ACCEPT)], w[feature_index(f, p, REJECT)]) if f > 0
                else w[feature_index(f, p, REJECT)] for f in range(N_SERVERS + 1)]
        axes[1].plot(range(N_SERVERS + 1), vals, "o-", ms=3, label=f"priority {PRIORITIES[p]}")
    axes[1].set_xlabel("number of free servers")
    axes[1].set_ylabel(r"differential value of best action $\max_a \hat q(s,a,\mathbf{w})$")
    axes[1].set_title("Learned differential action values")
    axes[1].legend(fontsize=8)
    steps = np.arange(len(hist)) * 1000
    axes[2].plot(steps, hist, label=r"$\bar R_t$ (learned)")
    axes[2].axhline(r_star, color="k", ls="--", label=f"optimal r* = {r_star:.3f}")
    axes[2].axhline(average_reward_of(eps_greedy, P, r), color="tab:red", ls=":",
                    label="exact r of learned ε-greedy policy")
    axes[2].set_xlabel("steps")
    axes[2].set_ylabel("average reward")
    axes[2].set_ylim(0, 4)
    axes[2].set_title(r"Average-reward estimate $\bar R$")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "access_control.png"), dpi=110)
    print(f"figure saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
