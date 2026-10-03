"""Q-learning convergence and the Robbins-Monro step-size conditions, measured against q*.

Chapter 05, Section 8.4 (the tabular Q-learning convergence theorem).

Theorem (Watkins & Dayan 1992; Jaakkola, Jordan & Singh 1994; Tsitsiklis 1994): in a finite
MDP with bounded rewards and gamma < 1, Q-learning converges to q* with probability 1 if every
(s, a) is updated infinitely often and the step sizes alpha_n(s, a) used on the n-th update of
(s, a) satisfy   sum_n alpha_n = infinity   and   sum_n alpha_n^2 < infinity.

We check this on a small random MDP whose q* we compute exactly by value iteration:
  * 6 states, 2 actions, gamma = 0.9, transition probabilities drawn from a Dirichlet(1);
  * reward = r(s, a) + Normal(0, 1) noise, so the targets are genuinely noisy;
  * behaviour policy: uniformly random (so all pairs are visited infinitely often);
  * one long continuing trajectory per run; many runs simulated in parallel with numpy.

Step-size schedules compared (n = number of updates of that (s, a) so far, starting at 1):
  constant 0.1, constant 0.02        -- violate sum alpha^2 < inf: error plateaus at a noise floor
  1 / n                              -- satisfies Robbins-Monro, but can be very slow
  1 / n^0.6, 1 / n^0.8               -- "polynomial" rates (Even-Dar & Mansour 2003)

Run:  python code/ch05_temporal_difference/q_learning_step_sizes.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
N_S, N_A, GAMMA, NOISE_STD = 6, 2, 0.9, 1.0


def make_mdp(rng):
    P = rng.dirichlet(np.ones(N_S), size=(N_S, N_A))        # P[s, a, s']
    r = rng.uniform(-1, 1, size=(N_S, N_A))                  # mean rewards r(s, a)
    return P, r


def q_star(P, r, tol=1e-12):
    Q = np.zeros((N_S, N_A))
    while True:
        Q_new = r + GAMMA * P @ Q.max(axis=1)                # Bellman optimality operator
        if np.max(np.abs(Q_new - Q)) < tol:
            return Q_new
        Q = Q_new


def run_schedule(schedule, P, r, Qs, n_runs, n_steps, seed, log_every):
    """Run n_runs independent Q-learners in parallel; return max-norm error every log_every steps."""
    rng = np.random.default_rng(seed)
    cumP = P.cumsum(axis=2)
    Q = np.zeros((n_runs, N_S, N_A))
    counts = np.zeros((n_runs, N_S, N_A))
    s = rng.integers(N_S, size=n_runs)
    runs = np.arange(n_runs)
    errs = []
    for t in range(n_steps):
        a = rng.integers(N_A, size=n_runs)                   # uniform behaviour policy
        u = rng.random(n_runs)[:, None]
        s2 = (u > cumP[s, a]).sum(axis=1)                    # sample s' ~ P(.|s, a)
        s2 = np.minimum(s2, N_S - 1)                         # guard against round-off
        reward = r[s, a] + NOISE_STD * rng.standard_normal(n_runs)
        counts[runs, s, a] += 1
        alpha = schedule(counts[runs, s, a])
        target = reward + GAMMA * Q[runs, s2].max(axis=1)    # continuing task: always bootstrap
        Q[runs, s, a] += alpha * (target - Q[runs, s, a])
        s = s2
        if (t + 1) % log_every == 0:
            errs.append(np.abs(Q - Qs).max(axis=(1, 2)))     # ||Q_t - q*||_inf per run
    return np.array(errs)                                    # shape [n_logs, n_runs]


SCHEDULES = {
    "constant α = 0.1": lambda n: 0.1 * np.ones_like(n),
    "constant α = 0.02": lambda n: 0.02 * np.ones_like(n),
    "α = 1/n": lambda n: 1.0 / n,
    "α = 1/n^0.8": lambda n: 1.0 / n ** 0.8,
    "α = 1/n^0.6": lambda n: 1.0 / n ** 0.6,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    n_runs = 10 if args.quick else 20
    n_steps = 20_000 if args.quick else 400_000
    log_every = 1000
    print(f"[q_learning_step_sizes] seed={args.seed} runs={n_runs} steps={n_steps} "
          f"|S|={N_S} |A|={N_A} gamma={GAMMA} reward noise std={NOISE_STD}")
    t0 = time.time()
    P, r = make_mdp(np.random.default_rng(args.seed))
    Qs = q_star(P, r)
    print(f"q* range: [{Qs.min():.3f}, {Qs.max():.3f}]")

    results = {}
    for name, sched in SCHEDULES.items():
        results[name] = run_schedule(sched, P, r, Qs, n_runs, n_steps, args.seed + 1, log_every)

    checkpoints = [c for c in (10_000, 50_000, 100_000, 200_000, 400_000) if c <= n_steps]
    print("\nmean over runs of ||Q_t - q*||_inf")
    print("  schedule            | " + "  ".join(f"t={c:>7,d}" for c in checkpoints))
    for name, e in results.items():
        print(f"  {name:19s} | " + "  ".join(f"{e[c // log_every - 1].mean():9.3f}" for c in checkpoints))

    if not args.quick:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        x = np.arange(1, n_steps // log_every + 1) * log_every
        for name, e in results.items():
            ls = "--" if name.startswith("constant") else "-"
            ax.loglog(x, e.mean(axis=1), ls=ls, label=name)
        ax.set_xlabel("time step t")
        ax.set_ylabel(r"mean $\|Q_t - q_\ast\|_\infty$")
        ax.set_title(f"Q-learning on a random {N_S}-state MDP (γ={GAMMA}, {n_runs} runs)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "q_learning_step_sizes.png"), dpi=110)
        plt.close(fig)
        print(f"\nFigure written to {FIG_DIR}/q_learning_step_sizes.png")
    print(f"\nRuntime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
