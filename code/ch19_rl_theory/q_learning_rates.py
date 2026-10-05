"""Stochastic approximation in action: how the step-size schedule decides whether (and how fast)
Q-learning converges. Chapter 19, Sections 3.2-3.5.

Setting. A random 8-state, 3-action MDP with Bernoulli rewards (means uniform in [0, 1]) and
dense Dirichlet(1) transitions. We run *synchronous* Q-learning with a generative model: at
iteration n every pair (s, a) receives one fresh sample (R, S') and is updated with

    Q_{n}(s,a) = Q_{n-1}(s,a) + alpha_n [ R + gamma * max_a' Q_{n-1}(S', a') - Q_{n-1}(s,a) ]  (Algorithm 19.1)

so all pairs share the same visit count n and the only thing that differs between runs is the
step-size schedule alpha_n. (Asynchronous, single-trajectory Q-learning is the same iteration with
pair-dependent counts; see Exercise 13, exercise_solutions.py.) Schedules compared:

    1/n                    Robbins-Monro conditions hold, but the bias decays like n^-(1-gamma)
    1/n^0.8                Robbins-Monro conditions hold (polynomial rate, Even-Dar & Mansour 2003)
    1/(1+(1-gamma) n)      "rescaled linear": Robbins-Monro, and it matches the contraction factor
    0.1 and 0.01 constant  sum alpha^2 = infinity: no convergence, a noise floor that shrinks with alpha

q* is computed exactly by value iteration. We report ||Q_n - q*||_inf, averaged over independent
runs that share their random samples across schedules (common random numbers), for gamma = 0.9
and gamma = 0.99, at checkpoints that include every power of 10 exactly. The script also evaluates
the one-state calculation of Section 3.5 (Eq. 19.9): with alpha_n = 1/n the deterministic error
contracts by prod_k (1 - (1-gamma)/k) ~ n^-(1-gamma).

Part 0 checks the worked example of Section 3.2 first: mean estimation with alpha_n = c/n, started
AT the mean, so the whole error is noise. Its RMSE decays like n^-1/2 when c > 1/2 but only like
n^-c when c < 1/2 (early noise is forgotten as slowly as an initial error would be).

Run from the repository root:
  python code/ch19_rl_theory/q_learning_rates.py           # full run (~1-2 min), writes the figure
  python code/ch19_rl_theory/q_learning_rates.py --quick   # smoke test, no figure
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from theory_lib import COLORS, loglog_slope, random_mdp, setup_style, value_iteration

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")

SCHEDULES = [
    ("1/n", lambda n, g: 1.0 / n),
    ("1/n^0.8", lambda n, g: n ** -0.8),
    ("1/(1+(1-γ)n)", lambda n, g: 1.0 / (1.0 + (1.0 - g) * n)),
    ("constant 0.1", lambda n, g: 0.1),
    ("constant 0.01", lambda n, g: 0.01),
]


def checkpoint_grid(N, n_log=70):
    """Log-spaced iteration counts in [1, N] that contain every power of 10 up to N exactly, so the
    reported columns n = 10^k are the iterates at n = 10^k (not the next grid point after it)."""
    grid = np.round(np.logspace(0, np.log10(N), n_log)).astype(int)
    powers = 10 ** np.arange(0, int(np.floor(np.log10(N) + 1e-9)) + 1)
    return np.unique(np.concatenate([grid, powers, [N]]))


def mean_estimation_rates(rng, cs, n_max, runs):
    """Section 3.2 worked example: theta_n = theta_{n-1} + (c/n)(X_n - theta_{n-1}), X_n ~ N(0, 1),
    started at the mean theta_0 = 0, so the error is pure noise. Returns RMSE at powers of 10 and the
    log-log slope over n = 10^2 .. n_max."""
    ns = 10 ** np.arange(1, int(np.log10(n_max)) + 1)
    out = {}
    theta = np.zeros((len(cs), runs))
    c = np.array(cs)[:, None]
    rmse = np.zeros((len(cs), len(ns)))
    j = 0
    for n in range(1, n_max + 1):
        x = rng.standard_normal(runs)                      # the same samples for every c (common random numbers)
        theta += np.minimum(c / n, 1.0) * (x[None] - theta)
        if n == ns[j]:
            rmse[:, j] = np.sqrt(np.mean(theta ** 2, axis=1))
            j += 1
    keep = ns >= 100
    for i, cc in enumerate(cs):
        out[cc] = (ns, rmse[i], loglog_slope(ns[keep], rmse[i, keep]))
    return out


def run_sync_q_learning(P, r, gamma, n_iter, n_runs, rng, checkpoints):
    """Synchronous Q-learning for every schedule at once (common random numbers).

    Returns errors[k, j, i] = ||Q - q*||_inf for schedule k, run j, checkpoint i."""
    S, A, _ = P.shape
    K = len(SCHEDULES)
    q_star, _ = value_iteration(P, r, gamma)
    cumP = np.cumsum(P, axis=-1)
    cumP[..., -1] = 1.0                                   # guard against round-off in the CDF
    Q = np.zeros((K, n_runs, S, A))                       # Q_0 = 0 for every schedule and run
    ridx = np.arange(n_runs)[:, None, None]
    errors = np.zeros((K, n_runs, len(checkpoints)))
    ci = 0
    for n in range(1, n_iter + 1):
        # One generative-model sample per (run, s, a): S' ~ P(.|s,a), R ~ Bernoulli(r(s,a)).
        u = rng.random((n_runs, S, A))
        s_next = (u[..., None] > cumP[None]).sum(axis=-1)            # inverse-CDF sampling
        rew = (rng.random((n_runs, S, A)) < r[None]).astype(float)
        v_next = Q.max(axis=-1)[:, ridx, s_next]                       # (K, runs, S, A)
        target = rew[None] + gamma * v_next
        alpha = np.array([f(n, gamma) for _, f in SCHEDULES])[:, None, None, None]
        Q += alpha * (target - Q)                                      # Algorithm 19.1, Eq. (19.5) with h(Q) = T*Q - Q
        if ci < len(checkpoints) and n == checkpoints[ci]:
            errors[:, :, ci] = np.abs(Q - q_star[None, None]).max(axis=(2, 3))
            ci += 1
    return errors, q_star


def one_state_bias(gamma, n):
    """Deterministic error factor prod_{k=1}^n (1 - (1-gamma)/k) for alpha_k = 1/k (Section 3.5, Eq. 19.9)."""
    k = np.arange(1, n + 1)
    return float(np.exp(np.sum(np.log1p(-(1.0 - gamma) / k))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=None)
    args = ap.parse_args()

    S, A = 8, 3
    gammas = [0.9, 0.99]
    n_iters = {0.9: 2_000, 0.99: 5_000} if args.quick else {0.9: 100_000, 0.99: 1_000_000}
    n_runs = args.runs or (2 if args.quick else 10)
    print(f"seed={args.seed}  S={S} A={A}  gammas={gammas}  iterations={n_iters}  runs={n_runs}")
    print("schedules:", ", ".join(name for name, _ in SCHEDULES))

    t0 = time.time()
    # Part 0: the step-size constant (Section 3.2 worked example).
    cs = (0.1, 0.3, 0.5, 1.0)
    n_max, me_runs = (10_000, 1_000) if args.quick else (100_000, 4_000)
    me = mean_estimation_rates(np.random.default_rng(args.seed + 2), cs, n_max, me_runs)
    print(f"\nPart 0: mean estimation with alpha_n = c/n started at the mean (pure noise), {me_runs} runs")
    for cc in cs:
        ns, rm, sl = me[cc]
        theory = -min(cc, 0.5)
        print(f"   c = {cc:3.1f}: RMSE at n = " + ", ".join(f"{n:,d}: {v:.4f}" for n, v in zip(ns, rm))
              + f" | slope over n >= 100: {sl:+.3f} (theory {theory:+.2f}"
              + (", with a log factor)" if cc == 0.5 else ")"))

    rng = np.random.default_rng(args.seed)
    P, r = random_mdp(rng, S, A, conc=1.0)
    results = {}
    for g in gammas:
        N = n_iters[g]
        checkpoints = checkpoint_grid(N)
        errors, q_star = run_sync_q_learning(P, r, g, N, n_runs, np.random.default_rng(args.seed + 1), checkpoints)
        results[g] = (checkpoints, errors, q_star)
        print(f"\n=== gamma = {g}: ||q*||_inf = {np.abs(q_star).max():.3f}  ({time.time() - t0:.1f} s elapsed)")
        report_at = [n for n in (10, 100, 1_000, 10_000, 100_000, 1_000_000) if n <= N]
        print(f"{'schedule':>16s} | " + " | ".join(f"n={n:>9,d}" for n in report_at)
              + " | slope over last decade")
        for k, (name, _) in enumerate(SCHEDULES):
            mean_err = errors[k].mean(axis=0)
            idx = [int(np.searchsorted(checkpoints, n)) for n in report_at]
            assert all(checkpoints[i] == n for i, n in zip(idx, report_at))   # exact n, not a nearby grid point
            vals = [mean_err[i] for i in idx]
            last = checkpoints >= N / 10
            slope = loglog_slope(checkpoints[last], mean_err[last])
            print(f"{name:>16s} | " + " | ".join(f"{v:11.4f}" for v in vals) + f" | {slope:+.3f}")
        bias = one_state_bias(g, N)
        print(f"one-state calculation, alpha=1/n: prod_k(1-(1-g)/k) at n={N:,d} is {bias:.4f} "
              f"(n^-(1-g) = {N ** -(1 - g):.4f})")
    print(f"\ntotal time {time.time() - t0:.1f} s")

    if args.quick:
        return
    setup_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    styles = ["-", "--", "-", ":", "-."]
    for ax, g in zip(axes, gammas):
        checkpoints, errors, q_star = results[g]
        for k, (name, _) in enumerate(SCHEDULES):
            m = errors[k].mean(axis=0)
            lo, hi = np.percentile(errors[k], [10, 90], axis=0)
            ax.loglog(checkpoints, m, styles[k], color=COLORS[k], label=name)
            ax.fill_between(checkpoints, lo, hi, color=COLORS[k], alpha=0.12, lw=0)
        # Reference: the one-state 1/n bias curve and a slope -1/2 line.
        e0 = np.abs(q_star).max()
        ns = checkpoints.astype(float)
        ax.loglog(ns, e0 * ns ** -(1 - g), color="k", lw=0.9, ls=(0, (1, 2)),
                  label=r"$\|q_\ast\|_\infty\, n^{-(1-\gamma)}$ (1/n bias, Eq. 19.9)")
        anchor = errors[2].mean(axis=0)[-1] * np.sqrt(ns[-1])
        ax.loglog(ns[ns >= 100], anchor / np.sqrt(ns[ns >= 100]), color="#666666", lw=0.9, ls="--",
                  label=r"slope $-1/2$")
        ax.set_title(f"Synchronous Q-learning, γ = {g}  ({errors.shape[1]} runs)")
        ax.set_xlabel("iteration n (samples per state–action pair)")
        ax.set_ylabel(r"$\|Q_n - q_\ast\|_\infty$")
        ax.set_ylim(bottom=max(1e-3, ax.get_ylim()[0]))
    axes[0].legend(loc="lower left", fontsize=7.8)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "q_learning_rates.png")
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
