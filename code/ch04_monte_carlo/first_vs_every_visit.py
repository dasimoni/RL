"""First-visit vs every-visit Monte Carlo prediction: bias, variance and consistency (Sec. 2.3).

The task (S&B Exercise 5.5 / Singh & Sutton 1996): a single non-terminal state s. Every
step gives reward +1; with probability p the process returns to s, otherwise it terminates.
gamma = 1, so an episode that lasts L steps visits s at times 0..L-1 with returns
L, L-1, ..., 1 and the true value is v(s) = E[L] = 1/(1-p).

  first-visit estimate after n episodes :  mean_i L_i                          (unbiased)
  every-visit estimate after n episodes :  sum_i L_i(L_i+1)/2 / sum_i L_i       (biased, consistent)

We measure bias and MSE of both over many independent replications and compare them with
the exact n = 1 values and the large-n (delta-method) asymptotics derived in the chapter.

    python code/ch04_monte_carlo/first_vs_every_visit.py [--quick]
"""
import argparse
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def first_visit_returns(L):
    """Return observed at the first visit of an episode of length L (one per episode)."""
    return L.astype(float)


def every_visit_sums(L):
    """(sum of all returns observed in the episode, number of visits) = (L(L+1)/2, L)."""
    return L * (L + 1) / 2.0, L.astype(float)


def exact_moments(p, kmax=20_000):
    """Exact moments of L ~ Geometric(1-p) on {1, 2, ...} by direct summation."""
    l = np.arange(1, kmax + 1, dtype=float)
    w = (1 - p) * p ** (l - 1)
    X, Y = l * (l + 1) / 2, l
    EX, EY = (w * X).sum(), (w * Y).sum()
    v = EX / EY                                  # = 1/(1-p): every-visit is consistent
    var_y = (w * (Y - EY) ** 2).sum()
    cov_xy = (w * (X - EX) * (Y - EY)).sum()
    var_lin = (w * (X - v * Y) ** 2).sum()       # Var(X - vY), since E[X - vY] = 0
    return dict(EY=EY, v_every_limit=v, var_L=var_y,
                fv_nmse=var_y,                                   # n * MSE(first-visit)
                ev_nmse_asym=var_lin / EY ** 2,                  # n * MSE(every-visit), n -> inf
                ev_nbias_asym=(v * var_y - cov_xy) / EY ** 2)    # n * bias(every-visit), n -> inf


def simulate(p, n_grid, reps, rng, chunk=2000):
    """Bias and MSE of both estimators at each n in n_grid, over `reps` replications."""
    v = 1 / (1 - p)
    n_max = int(max(n_grid))
    idx = np.asarray(n_grid, dtype=int) - 1
    fv_est, ev_est = [], []
    for start in range(0, reps, chunk):
        r = min(chunk, reps - start)
        L = rng.geometric(1 - p, size=(r, n_max))
        fv = np.cumsum(first_visit_returns(L), axis=1)[:, idx] / (idx + 1)
        num, den = every_visit_sums(L)
        ev = np.cumsum(num, axis=1)[:, idx] / np.cumsum(den, axis=1)[:, idx]
        fv_est.append(fv)
        ev_est.append(ev)
    fv_est, ev_est = np.vstack(fv_est), np.vstack(ev_est)
    out = {}
    for name, est in (("first", fv_est), ("every", ev_est)):
        bias = est.mean(0) - v
        mse = ((est - v) ** 2).mean(0)
        # standard error of the MSE estimate (for honest reporting)
        mse_se = ((est - v) ** 2).std(0, ddof=1) / np.sqrt(est.shape[0])
        out[name] = dict(bias=bias, mse=mse, mse_se=mse_se)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=float, default=0.9)
    args = ap.parse_args()
    reps = 4_000 if args.quick else 40_000
    n_max = 100 if args.quick else 1000
    n_grid = np.unique(np.round(np.logspace(0, np.log10(n_max), 25)).astype(int))
    print(f"First- vs every-visit MC | seed={args.seed} p={args.p} reps={reps} n_max={n_max}")
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    p, v = args.p, 1 / (1 - args.p)
    ex = exact_moments(p)
    print(f"true v(s) = 1/(1-p) = {v:.4f};  every-visit limit E[L(L+1)/2]/E[L] = {ex['v_every_limit']:.4f}")
    print(f"exact n=1: first-visit MSE = Var(L) = {ex['var_L']:.3f};  every-visit (L+1)/2 has "
          f"bias {(v + 1) / 2 - v:+.3f}, MSE = {((v + 1) / 2 - v) ** 2 + ex['var_L'] / 4:.3f}")
    print(f"asymptotic n*MSE: first-visit = {ex['fv_nmse']:.2f}, every-visit = {ex['ev_nmse_asym']:.2f};"
          f"  asymptotic n*bias(every-visit) = {ex['ev_nbias_asym']:+.3f}")

    res = simulate(p, n_grid, reps, rng)
    print("\n   n | bias FV  bias EV |   MSE FV   MSE EV  (+- s.e.) | ratio EV/FV")
    for i, n in enumerate(n_grid):
        if n <= 10 or n in (20, 50, 100, 200, 500, 1000) or i == len(n_grid) - 1:
            f, e = res["first"], res["every"]
            print(f"{n:4d} | {f['bias'][i]:+7.3f} {e['bias'][i]:+7.3f} | {f['mse'][i]:8.3f} {e['mse'][i]:8.3f}"
                  f"  (+-{f['mse_se'][i]:.3f}, +-{e['mse_se'][i]:.3f}) | {e['mse'][i] / f['mse'][i]:.3f}")
    # where does every-visit stop being better?
    ratio = res["every"]["mse"] / res["first"]["mse"]
    cross = n_grid[np.argmax(ratio > 1)] if (ratio > 1).any() else None
    print(f"\nFirst n at which every-visit MSE exceeds first-visit MSE: {cross}" if cross else
          "\nEvery-visit had lower MSE for all n tested")
    print(f"elapsed {time.time() - t0:.1f}s")

    if args.quick:
        return
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogx(n_grid, res["first"]["bias"], "o-", ms=4, label="first-visit")
    ax[0].semilogx(n_grid, res["every"]["bias"], "s-", ms=4, label="every-visit")
    big = n_grid >= 3
    ax[0].semilogx(n_grid[big], ex["ev_nbias_asym"] / n_grid[big], "k:",
                   label=f"every-visit, asymptotic {ex['ev_nbias_asym']:.0f}/n")
    ax[0].axhline(0, color="gray", lw=0.8)
    ax[0].set_xlabel("episodes n")
    ax[0].set_ylabel(r"bias  $\mathbb{E}[V_n] - v(s)$")
    ax[0].set_title("Bias: every-visit is biased but consistent")
    ax[0].legend()
    ax[1].loglog(n_grid, res["first"]["mse"], "o-", ms=4, label="first-visit (simulated)")
    ax[1].loglog(n_grid, res["every"]["mse"], "s-", ms=4, label="every-visit (simulated)")
    ax[1].loglog(n_grid, ex["fv_nmse"] / n_grid, "C0:", label=f"first-visit exact: {ex['fv_nmse']:.0f}/n")
    ax[1].loglog(n_grid, ex["ev_nmse_asym"] / n_grid, "C1:",
                 label=f"every-visit asymptotic: {ex['ev_nmse_asym']:.1f}/n")
    ax[1].set_xlabel("episodes n")
    ax[1].set_ylabel("mean squared error")
    ax[1].set_title(f"MSE (one-state loop, p={p}, v={v:.0f}; {reps} runs)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(HERE, "figures", "first_vs_every_visit.png")
    fig.savefig(path, dpi=110)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
