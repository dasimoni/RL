"""Expected vs sample updates: error as a function of computation.

Chapter 07, Section 6 (S&B Section 8.5, Figure 8.7).

One state-action pair (s, a) has b equally likely successor states whose
values v_1..v_b are treated as exact (the reward is folded into them). The
correct value is their mean mu. Starting from an estimate with error 1:

  * an EXPECTED update reads all b successors (b "max_a' Q(s', a')"
    computations) and then has error exactly 0;
  * SAMPLE updates with step size 1/t read one random successor each; after t
    of them the estimate is the mean of t draws with replacement, and
        E[(Q_t - mu)^2] = sigma_b^2 / t,   sigma_b^2 = (1/b) sum_j (v_j - mu)^2.
    If the v_j are iid with variance 1, E[sigma_b^2] = (b-1)/b, giving the
    RMS error sqrt((b-1) / (b t))   (derived in Section 6.1 of the chapter).

We check the formula by simulation and plot both kinds of update against
computation measured in units of b.

Run:  python code/ch07_planning_and_learning_tabular/expected_vs_sample.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")


def simulate_rms_error(b, n_trials, t_max, rng, chunk=50):
    """Monte Carlo estimate of the RMS error after t = 0..t_max sample updates.

    Each trial draws fresh successor values v ~ N(0, 1)^b (so mu varies across
    trials), starts from the estimate mu + 1 (error 1) and applies
    Q <- Q + (1/t) (v_J - Q) with J uniform on {1..b}."""
    sq_err = np.zeros(t_max + 1)
    sq_err[0] = 1.0                                   # initial error is 1 by construction
    done = 0
    while done < n_trials:
        m = min(chunk, n_trials - done)
        v = rng.standard_normal((m, b))
        mu = v.mean(axis=1, keepdims=True)
        J = rng.integers(0, b, size=(m, t_max))
        draws = np.take_along_axis(v, J, axis=1)
        # With step size 1/t the first update overwrites the initial estimate,
        # so Q_t is just the running mean of the draws.
        Q = np.cumsum(draws, axis=1) / np.arange(1, t_max + 1)
        sq_err[1:] += ((Q - mu) ** 2).sum(axis=0)
        done += m
    sq_err[1:] /= n_trials
    return np.sqrt(sq_err)


def theory(b, t):
    t = np.asarray(t, dtype=float)
    return np.sqrt((b - 1) / (b * t))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    branching = [2, 10, 100, 1000] if args.quick else [2, 10, 100, 1000, 10000]
    n_trials = 200 if args.quick else 2000
    rng = np.random.default_rng(args.seed)
    print(f"Expected vs sample updates | seed={args.seed} trials={n_trials} b in {branching}")

    t0 = time.time()
    curves = {}
    print(f"{'b':>6} | {'RMS error after t sample updates (simulated / theory sqrt((b-1)/(bt)))':^70}")
    print(f"{'':>6} | {'t = 1':>15} {'t = b/10':>17} {'t = b':>17} {'t = 2b':>17}")
    for b in branching:
        err = simulate_rms_error(b, n_trials, 2 * b, rng)
        curves[b] = err
        cols = []
        for t in (1, max(1, b // 10), b, 2 * b):
            cols.append(f"{err[t]:.3f}/{theory(b, t):.3f}")
        print(f"{b:>6} | {cols[0]:>15} {cols[1]:>17} {cols[2]:>17} {cols[3]:>17}")
    print(f"(time {time.time() - t0:.1f}s)")
    print("Expected update: error 1.000 until all b successors are read, then exactly 0.")

    if args.quick:
        print("quick mode: no figures written")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    cmap = plt.get_cmap("viridis")
    for i, b in enumerate(branching):
        color = cmap(i / max(1, len(branching) - 1))
        x = np.arange(0, 2 * b + 1) / b
        ax.plot(x, curves[b], color=color, lw=2, label=f"sample updates, b = {b}")
        xt = np.arange(1, 2 * b + 1) / b
        ax.plot(xt, theory(b, np.arange(1, 2 * b + 1)), color="k", lw=0.8, ls=":")
    ax.plot([0, 1, 1, 2], [1, 1, 0, 0], color="tab:red", lw=2.5, label="expected update (any b)")
    ax.plot([], [], color="k", lw=0.8, ls=":", label=r"theory $\sqrt{(b-1)/(bt)}$")
    ax.set_xlabel("computation in units of one expected update\n(number of successor values read, divided by b)")
    ax.set_ylabel("RMS error in the value estimate")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Expected vs sample updates ({n_trials} trials per b)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "expected_vs_sample.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
