"""Incremental estimation and step-size schedules.

Chapter 00, Sections 2.3, 2.4 and 3.4. All estimators here are instances of

    Q_{n+1} = Q_n + alpha_n (R_n - Q_n)                                  (Eq. 2.6)

applied to a stream of noisy samples R_1, R_2, ... . We compare
  (a) bias / variance / MSE of the sample mean (alpha_n = 1/n) and constant-alpha
      exponential recency-weighted averages on a STATIONARY target, against the
      closed-form formulas derived in Section 2.4;
  (b) tracking a NON-STATIONARY (random-walk) target, where constant alpha wins;
  (c) step-size schedules alpha_n = n^{-p} that do / do not satisfy the
      Robbins-Monro conditions  sum alpha = inf, sum alpha^2 < inf  (Eq. 3.9).

Run:  python code/ch00_math_toolkit/estimation_step_sizes.py [--quick]
"""
import argparse
import os
import time

import numpy as np

from plot_style import setup, C, GREY

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def run_estimator(samples, step_size_fn, q1):
    """Apply Eq. 2.6 to every row of `samples` (runs x steps) in parallel.
    Returns the estimate Q_{n+1} after each sample n = 1..N (shape runs x N)."""
    runs, N = samples.shape
    Q = np.full(runs, q1, dtype=float)
    out = np.empty((runs, N))
    for n in range(1, N + 1):
        alpha = step_size_fn(n)
        Q += alpha * (samples[:, n - 1] - Q)  # NewEstimate <- Old + StepSize [Target - Old]
        out[:, n - 1] = Q
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    runs = 300 if args.quick else 2000
    N = 1000 if args.quick else 10_000
    print(f"[estimation_step_sizes] seed={args.seed} runs={runs} steps={N} quick={args.quick}")
    t0 = time.time()

    # ------------------------------------------------------------------ (a)
    mu, sigma, q1 = 1.0, 1.0, 0.0  # true mean, noise std, (deliberately wrong) initial estimate
    R = mu + sigma * rng.standard_normal((runs, N))
    schedules_a = {
        "sample mean (α=1/n)": lambda n: 1.0 / n,
        "constant α=0.1": lambda n: 0.1,
        "constant α=0.01": lambda n: 0.01,
    }
    n_axis = np.arange(1, N + 1)
    res_a = {}
    print("\n(a) Stationary target  R ~ N(1, 1),  Q_1 = 0")
    print(f"{'estimator':22s} {'n':>6s} {'bias':>9s} {'(theory)':>9s} {'var':>9s} {'(theory)':>9s} {'MSE':>9s}")
    for name, fn in schedules_a.items():
        Q = run_estimator(R, fn, q1)
        bias = Q.mean(axis=0) - mu
        var = Q.var(axis=0)
        mse = ((Q - mu) ** 2).mean(axis=0)
        if "sample" in name:
            th_bias = np.zeros(N); th_var = sigma ** 2 / n_axis
        else:
            a = fn(1)
            th_bias = (1 - a) ** n_axis * (q1 - mu)                                    # Eq. 2.9
            th_var = a * sigma ** 2 / (2 - a) * (1 - (1 - a) ** (2 * n_axis))           # Eq. 2.10
        res_a[name] = dict(mse=mse, th_mse=th_bias ** 2 + th_var)
        for n in (10, 100, N):
            i = n - 1
            print(f"{name:22s} {n:6d} {bias[i]:9.4f} {th_bias[i]:9.4f} {var[i]:9.5f} {th_var[i]:9.5f} {mse[i]:9.5f}")

    # ------------------------------------------------------------------ (b)
    # Non-stationary: the true mean performs a random walk (cf. S&B Exercise 2.5).
    walk_std = 0.01
    mu_t = 1.0 + np.cumsum(walk_std * rng.standard_normal((runs, N)), axis=1)
    Rb = mu_t + sigma * rng.standard_normal((runs, N))
    res_b = {}
    print(f"\n(b) Non-stationary target (random-walk mean, step std {walk_std}): mean sq. tracking error")
    for name, fn in schedules_a.items():
        Q = run_estimator(Rb, fn, q1)
        # Q[:, n-1] is the estimate after seeing R_n; compare it with the mean at the NEXT step
        err = ((Q[:, :-1] - mu_t[:, 1:]) ** 2).mean(axis=0)
        res_b[name] = err
        print(f"  {name:22s} avg over last 20% of steps: {err[int(0.8 * N):].mean():.4f}")

    # ------------------------------------------------------------------ (c)
    # Robbins-Monro: estimate mu = 5 from Q_1 = 0 with different schedules.
    mu_c = 5.0
    Rc = mu_c + sigma * rng.standard_normal((runs, N))
    schedules_c = {
        "α=1/n   (RM ✓, sample mean)": lambda n: 1.0 / n,
        "α=1/n^0.6 (RM ✓)": lambda n: n ** -0.6,
        "α=1/n^0.3 (Σα² = ∞)": lambda n: n ** -0.3,
        "α=0.05 constant (Σα² = ∞)": lambda n: 0.05,
        "α=1/(n+1)² (Σα < ∞)": lambda n: 1.0 / (n + 1) ** 2,
    }
    res_c = {}
    print("\n(c) Robbins-Monro step-size schedules: mu = 5, Q_1 = 0, sigma = 1")
    print(f"  {'schedule':30s} {'mean Q_N':>9s} {'RMSE@100':>9s} {'RMSE@N':>9s}")
    for name, fn in schedules_c.items():
        Q = run_estimator(Rc, fn, 0.0)
        rmse = np.sqrt(((Q - mu_c) ** 2).mean(axis=0))
        res_c[name] = rmse
        print(f"  {name:30s} {Q[:, -1].mean():9.4f} {rmse[99]:9.4f} {rmse[-1]:9.4f}")
    print("  (for alpha=1/(n+1)^2 the theory says E[Q_inf] = mu * (1 - prod_j(1-1/j^2)) = 5 * (1 - 1/2) = 2.5)")

    if not args.quick:
        plt = setup()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        ax = axes[0]
        styles = ["-", "--", ":"]
        for (name, r), c, ls in zip(res_a.items(), C, styles):
            ax.loglog(n_axis, r["mse"], ls, color=c, label=name)
            ax.loglog(n_axis, r["th_mse"], color=GREY, lw=0.8)
        ax.plot([], [], color=GREY, lw=0.8, label="closed-form MSE (Eqs. 2.9–2.10)")
        ax.set_title("(a) Stationary target: MSE = bias² + variance")
        ax.set_xlabel("number of samples n"); ax.set_ylabel(r"MSE of $Q_{n+1}$"); ax.legend()

        ax = axes[1]
        for (name, err), c, ls in zip(res_b.items(), C, styles):
            ax.loglog(n_axis[:-1], err, ls, color=c, label=name)
        ax.set_title("(b) Non-stationary (random-walk) target")
        ax.set_xlabel("number of samples n"); ax.set_ylabel("mean squared tracking error"); ax.legend()

        ax = axes[2]
        styles_c = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]
        for (name, r), c, ls in zip(res_c.items(), C, styles_c):
            ax.loglog(n_axis, r, linestyle=ls, color=c, label=name)
        ax.set_title("(c) Robbins–Monro step-size conditions")
        ax.set_xlabel("number of samples n"); ax.set_ylabel(r"RMSE of $Q_{n+1}$ (target μ=5)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(FIG_DIR, "step_sizes.png")
        fig.savefig(path)
        print(f"\nsaved {path}")

    print(f"\nSummary: constant alpha=0.1 plateaus at MSE ~ alpha*sigma^2/(2-alpha) = {0.1/1.9:.4f}; "
          f"1/n keeps improving as sigma^2/n; 1/(n+1)^2 (sum alpha < inf) stalls half-way; "
          f"1/n^0.3 still converges (slowly): RM conditions are sufficient, not necessary. "
          f"runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
