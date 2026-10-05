"""Batch TD(0) vs batch Monte Carlo: certainty equivalence vs minimum training MSE.

Chapter 05, Section 4.  Three experiments:

  Part A  "You are the predictor" (Sutton & Barto 2018, Example 6.4).  Eight observed
          episodes:  A,0,B,0 | B,1 (x6) | B,0.  Batch MC and batch TD(0) disagree on V(A).
  Part B  Certainty-equivalence check: the batch-TD(0) fixed point coincides with the
          exact value function of the maximum-likelihood (count-based) model of the data;
          the batch-MC fixed point coincides with per-state sample means of the returns.
  Part C  Batch training on the 5-state random walk (S&B Figure 6.2): after each new
          episode, sweep the *whole* batch repeatedly until V stops changing, then
          record the RMS error.  Averaged over many runs.

Batch updating (Section 4.1).  For every transition in the batch compute the usual
increment, but change V only once per sweep, by the sum of the increments:

    TD(0):  V(s) <- V(s) + alpha_s * sum_{t: S_t = s} [R_{t+1} + gamma V(S_{t+1}) - V(S_t)]
    MC:     V(s) <- V(s) + alpha_s * sum_{t: S_t = s} [G_t - V(S_t)]

We use a per-state constant alpha_s = eta / n(s), where n(s) = number of visits to s
in the batch.  The fixed point (sum of increments = 0 for every s) does not depend on
the step size, so this gives exactly the same answer as a tiny global constant alpha,
only in far fewer sweeps.  Part B verifies this numerically.

Run:  python code/ch05_temporal_difference/batch_td_vs_mc.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# --------------------------------------------------------------------------------------
# Transition-batch representation.  Each episode is a list of (s, r, s_next, done).
# We flatten the batch into numpy arrays so that one "sweep" is a few vector operations.
# --------------------------------------------------------------------------------------
def flatten(episodes, gamma: float = 1.0):
    S, R, S2, D, G = [], [], [], [], []
    for ep in episodes:
        ret = 0.0
        rets = []
        for (_, r, _, _) in reversed(ep):            # G_t = R_{t+1} + gamma G_{t+1}
            ret = r + gamma * ret
            rets.append(ret)
        rets.reverse()
        for (s, r, s2, d), g in zip(ep, rets):
            S.append(s); R.append(r); S2.append(s2); D.append(d); G.append(g)
    return (np.array(S), np.array(R, float), np.array(S2), np.array(D, bool), np.array(G, float))


def batch_td0(batch, n_states: int, V0: np.ndarray, gamma=1.0, eta=0.5, tol=1e-10, max_sweeps=100_000):
    """Batch TD(0): repeat sweeps over the fixed batch until the largest change < tol."""
    S, R, S2, D, _ = batch
    n = np.bincount(S, minlength=n_states).astype(float)
    alpha_s = np.where(n > 0, eta / np.maximum(n, 1), 0.0)  # per-state constant step size
    V = V0.copy()
    for sweep in range(1, max_sweeps + 1):
        # delta_t for every transition, all computed with the SAME V (batch semantics)
        delta = R + gamma * np.where(D, 0.0, V[S2]) - V[S]
        increment = alpha_s * np.bincount(S, weights=delta, minlength=n_states)
        V += increment
        if np.max(np.abs(increment)) < tol:
            return V, sweep
    raise RuntimeError("batch TD did not converge")


def batch_mc(batch, n_states: int, V0: np.ndarray, eta=0.5, tol=1e-10, max_sweeps=100_000):
    """Batch every-visit constant-alpha MC, iterated to its fixed point."""
    S, _, _, _, G = batch
    n = np.bincount(S, minlength=n_states).astype(float)
    alpha_s = np.where(n > 0, eta / np.maximum(n, 1), 0.0)
    V = V0.copy()
    for sweep in range(1, max_sweeps + 1):
        increment = alpha_s * np.bincount(S, weights=G - V[S], minlength=n_states)
        V += increment
        if np.max(np.abs(increment)) < tol:
            return V, sweep
    raise RuntimeError("batch MC did not converge")


def certainty_equivalence_values(batch, n_states: int, gamma=1.0):
    """Closed form: build the maximum-likelihood model from counts and solve V = r_hat + gamma P_hat V.

    r_hat(s)     = average reward observed on transitions out of s
    P_hat(s, s') = n(s -> s') / n(s)   (terminal next states contribute 0 value)
    Only visited states are part of the model.
    """
    S, R, S2, D, _ = batch
    visited = np.unique(S)
    idx = {s: i for i, s in enumerate(visited)}
    m = len(visited)
    P = np.zeros((m, m))
    r = np.zeros(m)
    n = np.zeros(m)
    for s, rew, s2, d in zip(S, R, S2, D):
        i = idx[s]
        n[i] += 1
        r[i] += rew
        if not d:
            P[i, idx[s2]] += 1
    P /= n[:, None]
    r /= n
    v = np.linalg.solve(np.eye(m) - gamma * P, r)
    out = np.full(n_states, np.nan)
    out[visited] = v
    return out


def mc_sample_means(batch, n_states: int):
    S, _, _, _, G = batch
    n = np.bincount(S, minlength=n_states)
    sums = np.bincount(S, weights=G, minlength=n_states)
    return np.where(n > 0, sums / np.maximum(n, 1), np.nan)


# --------------------------------------------------------------------------------------
# Part A: "You are the predictor"
# --------------------------------------------------------------------------------------
def part_a():
    A, B, TERM = 0, 1, 2
    episodes = [[(A, 0.0, B, False), (B, 0.0, TERM, True)]]
    episodes += [[(B, 1.0, TERM, True)] for _ in range(6)]
    episodes += [[(B, 0.0, TERM, True)]]
    batch = flatten(episodes)
    V0 = np.zeros(3)
    v_td, sweeps_td = batch_td0(batch, 3, V0)
    v_mc, sweeps_mc = batch_mc(batch, 3, V0)
    print("Part A -- 'You are the predictor' (8 episodes: A,0,B,0 | B,1 x6 | B,0)")
    print(f"  batch TD(0): V(A) = {v_td[A]:.4f}, V(B) = {v_td[B]:.4f}   ({sweeps_td} sweeps)")
    print(f"  batch MC   : V(A) = {v_mc[A]:.4f}, V(B) = {v_mc[B]:.4f}   ({sweeps_mc} sweeps)")
    # Training-set mean-squared error of each answer: sum over all (S_t, G_t) pairs.
    S, _, _, _, G = batch
    for name, v in (("TD", v_td), ("MC", v_mc)):
        mse = np.mean((G - v[S]) ** 2)
        print(f"  training-set MSE of the {name} answer w.r.t. observed returns: {mse:.4f}")
    return v_td, v_mc


# --------------------------------------------------------------------------------------
# Random walk (same MRP as random_walk_td_vs_mc.py)
# --------------------------------------------------------------------------------------
N = 5
TRUE_V = np.arange(1, N + 1) / (N + 1)


def random_walk_episode(rng):
    s, ep = 3, []
    while True:
        s2 = s + (1 if rng.random() < 0.5 else -1)
        done = s2 in (0, N + 1)
        ep.append((s, 1.0 if s2 == N + 1 else 0.0, s2, done))
        if done:
            return ep
        s = s2


def rms(V):
    return float(np.sqrt(np.mean((V[1:N + 1] - TRUE_V) ** 2)))


def part_b(seed):
    """Check batch fixed points against closed forms on a few random batches."""
    rng = np.random.default_rng(seed)
    worst_td = worst_mc = 0.0
    for trial in range(20):
        eps = [random_walk_episode(rng) for _ in range(int(rng.integers(1, 30)))]
        batch = flatten(eps)
        V0 = np.full(N + 2, 0.5); V0[[0, N + 1]] = 0.0
        v_td, _ = batch_td0(batch, N + 2, V0)
        v_mc, _ = batch_mc(batch, N + 2, V0)
        ce = certainty_equivalence_values(batch, N + 2)
        sm = mc_sample_means(batch, N + 2)
        mask = ~np.isnan(ce)
        worst_td = max(worst_td, np.max(np.abs(v_td[mask] - ce[mask])))
        worst_mc = max(worst_mc, np.max(np.abs(v_mc[mask] - sm[mask])))
    print("\nPart B -- fixed points vs closed forms (20 random batches of random-walk episodes)")
    print(f"  max |batch TD(0) - certainty-equivalence solution| = {worst_td:.2e}")
    print(f"  max |batch MC    - per-state sample mean of G|      = {worst_mc:.2e}")


def part_c(n_runs, n_episodes, seed):
    err_td = np.zeros((n_runs, n_episodes))
    err_mc = np.zeros((n_runs, n_episodes))
    total_sweeps = 0
    for run in range(n_runs):
        rng = np.random.default_rng(seed + run)
        episodes = []
        V_td = np.full(N + 2, 0.5); V_td[[0, N + 1]] = 0.0
        V_mc = V_td.copy()
        for k in range(n_episodes):
            episodes.append(random_walk_episode(rng))
            batch = flatten(episodes)
            # warm start from the previous fixed point (does not change the fixed point)
            V_td, sw = batch_td0(batch, N + 2, V_td, tol=1e-8)
            V_mc, _ = batch_mc(batch, N + 2, V_mc, tol=1e-8)
            total_sweeps += sw
            err_td[run, k] = rms(V_td)
            err_mc[run, k] = rms(V_mc)
    return err_td.mean(0), err_mc.mean(0), total_sweeps / (n_runs * n_episodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    n_runs = 10 if args.quick else 200
    n_episodes = 30 if args.quick else 100
    print(f"[batch_td_vs_mc] seed={args.seed} runs={n_runs} episodes={n_episodes} gamma=1 "
          f"per-state step eta/n(s), eta=0.5, convergence tol=1e-8 (Part C)")
    t0 = time.time()

    part_a()
    part_b(args.seed)
    td_curve, mc_curve, mean_sweeps = part_c(n_runs, n_episodes, args.seed + 1000)

    print(f"\nPart C -- batch training on the random walk ({n_runs} runs)")
    print("  episodes |  batch TD RMS  batch MC RMS")
    for k in (1, 5, 10, 25, 50, 100):
        if k <= n_episodes:
            print(f"  {k:8d} |  {td_curve[k - 1]:.4f}        {mc_curve[k - 1]:.4f}")
    frac = np.mean(td_curve[1:] < mc_curve[1:])
    print(f"  batch TD has lower RMS error than batch MC at {100 * frac:.0f}% of episode counts 2..{n_episodes}")
    print(f"  mean sweeps to convergence (TD, warm-started): {mean_sweeps:.1f}")

    if not args.quick:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        x = np.arange(1, n_episodes + 1)
        ax.plot(x, td_curve, color="tab:blue", label="batch TD(0)")
        ax.plot(x, mc_curve, color="tab:red", ls="--", label="batch MC")
        ax.set_xlabel("episodes in the batch")
        ax.set_ylabel("RMS error, averaged over states")
        ax.set_title(f"Batch training on the random walk ({n_runs} runs)")
        ax.set_ylim(0, 0.42)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "batch_td_vs_mc.png"), dpi=110)
        plt.close(fig)
        print(f"\nFigure written to {FIG_DIR}/batch_td_vs_mc.png")
    print(f"\nRuntime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
