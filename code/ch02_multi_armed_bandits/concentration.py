"""Hoeffding's inequality and the UCB confidence radius, checked exactly (Chapter 02, Section 6).

For n i.i.d. Bernoulli(p) rewards with sample mean Xbar_n we compare, for deviations u > 0
(the chapter writes the deviation as u, to keep it apart from the exploration rate epsilon),
  * the EXACT tail  P(Xbar_n - p >= u)  (binomial survival function, no simulation);
  * Hoeffding's bound            exp(-2 n u^2)                       (Eq. 6.1)
  * the Chernoff-KL bound        exp(-n kl(p + u, p))                (Section 6.2, tighter)
Hoeffding only uses that rewards lie in [0, 1]; it is nearly tight for p = 0.5 (variance 1/4,
the worst case) and very loose for p = 0.05, where the variance is small.  This looseness is
what KL-UCB removes (Eq. 11.7, Section 11.4).

We also check the coverage of the one-sided confidence bound Xbar_n + sqrt(ln(1/delta) / (2n))
(Eq. 6.3): the probability that it falls BELOW p must be <= delta.

Run:  python code/ch02_multi_armed_bandits/concentration.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from scipy import stats

from bandits import FIG_DIR, bernoulli_kl, setup_matplotlib, style


def exact_upper_tail(n, p, u):
    """P(Xbar_n - p >= u) = P(S >= ceil(n (p + u))) for S ~ Binomial(n, p)."""
    m = np.ceil(n * (p + u) - 1e-9)
    return stats.binom.sf(m - 1, n, p)


def exact_lower_tail(n, p, u):
    """P(p - Xbar_n >= u) = P(S <= floor(n (p - u)))."""
    m = np.floor(n * (p - u) + 1e-9)
    return np.where(m < 0, 0.0, stats.binom.cdf(m, n, p))


def main():
    p_ = argparse.ArgumentParser()
    p_.add_argument("--quick", action="store_true")
    args = p_.parse_args()
    print("Exact computation (no randomness).  n = 50 for the tail comparison.")
    t0 = time.time()
    n = 50
    for p in (0.5, 0.05):
        print(f"\np = {p}, n = {n}")
        print(f"  {'u':>5} | {'exact P(X̄-p ≥ u)':>20} | {'Hoeffding':>10} | {'Chernoff-KL':>11} | {'Hoeffding/exact':>15}")
        for u in (0.05, 0.1, 0.2, 0.3):
            if p + u > 1:
                continue
            ex = exact_upper_tail(n, p, u)
            ho = np.exp(-2 * n * u ** 2)
            kl = np.exp(-n * bernoulli_kl(p + u, p))
            print(f"  {u:5.2f} | {ex:20.3e} | {ho:10.3e} | {kl:11.3e} | {ho / max(ex, 1e-300):15.1f}")
            assert ex <= ho + 1e-15 and ex <= kl + 1e-15, "a bound is violated?!"

    print("\nCoverage of the upper confidence bound X̄_n + sqrt(ln(1/δ)/(2n)):  P(UCB < p) must be ≤ δ")
    print(f"  {'p':>5} {'n':>5} {'δ':>6} | {'radius':>7} | {'P(UCB < p) exact':>16}")
    worst = 0.0
    for p in (0.5, 0.2, 0.05):
        for nn in (10, 100, 1000):
            for delta in (0.1, 0.01):
                radius = np.sqrt(np.log(1 / delta) / (2 * nn))
                fail = float(exact_lower_tail(nn, p, radius + 1e-12))  # UCB < p  <=>  p - X̄ > radius
                worst = max(worst, fail / delta)
                print(f"  {p:5.2f} {nn:5d} {delta:6.2f} | {radius:7.4f} | {fail:16.2e}")
    print(f"  worst ratio P(UCB < p)/δ over the table: {worst:.3f}  (≤ 1, as Hoeffding guarantees)")
    print("\nUCB1 radius sqrt(2 ln t / n) is the case δ = t^-4:  ln(1/δ)/(2n) = 4 ln t/(2n) = 2 ln t / n")

    if not args.quick:
        plt = setup_matplotlib()
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
        for j, p in enumerate((0.5, 0.05)):
            u = np.linspace(0.005, min(0.45, 1 - p - 0.005), 300)
            ax[j].plot(u, exact_upper_tail(n, p, u), label="exact P(X̄ₙ − p ≥ u)", **style(0), lw=1.6)
            ax[j].plot(u, np.exp(-2 * n * u ** 2), label="Hoeffding exp(−2nu²)", **style(1), lw=1.6)
            ax[j].plot(u, np.exp(-n * bernoulli_kl(p + u, p)), label="Chernoff–KL exp(−n kl(p+u, p))",
                       **style(2), lw=1.6)
            ax[j].set_yscale("log")
            ax[j].set_ylim(1e-12, 2)
            ax[j].set(xlabel="deviation u", title=f"Bernoulli(p = {p}), n = {n}")
            ax[j].legend(loc="lower left", fontsize=8)
        ax[0].set_ylabel("tail probability (log scale)")
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "concentration.png")
        fig.savefig(out)
        print(f"\nsaved {out}")
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
