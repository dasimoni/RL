"""Check the tanh-squashing log-probability correction against Monte Carlo (Chapter 12, Section 6.3).

If u ~ N(mu, sigma^2) and a = tanh(u), the change-of-variables formula gives

    p_A(a) = p_U(atanh a) / |d tanh(u)/du| = N(atanh a; mu, sigma^2) / (1 - a^2)        (Eq. 12.25)
    log p_A(a) = log N(u; mu, sigma^2) - log(1 - tanh(u)^2)                              (Eq. 12.26)
    log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u))                                   (Eq. 12.27)

This script
  1. draws millions of squashed samples and compares a fine histogram (a Monte Carlo density
     estimate) with the corrected density and with the *uncorrected* one, N(atanh a; mu, sigma^2);
  2. checks that sac.py's log-prob agrees with torch.distributions' TanhTransform and with the
     histogram, and that E[-log p] (the entropy estimate SAC uses) matches a histogram estimate;
  3. shows why the naive log(1 - tanh(u)^2 + 1e-6) is inaccurate for large |u|;
  4. finds the most entropic tanh-Gaussian (mu = 0) and translates SAC's target entropy -1 (and the
     entropy -3.27 reached with alpha = 0.01 in sac_temperature.py) into pre-squash sigmas and action
     spreads, for centred and off-centre means.

Run (from the repository root):
  python code/ch12_continuous_control_actor_critic/tanh_squash_check.py          # full, figure
  python code/ch12_continuous_control_actor_critic/tanh_squash_check.py --quick  # fewer samples
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from common import FIG_DIR
from sac import SquashedGaussianActor, tanh_log_det_jacobian

torch.set_num_threads(1)


def density_tanh_gauss(a, mu, sigma):
    """Corrected density of a = tanh(u), u ~ N(mu, sigma^2), on (-1, 1)."""
    u = np.arctanh(a)
    return norm.pdf(u, mu, sigma) / (1.0 - a ** 2)


def entropy_tanh_gauss(mu, sigma, n=400_001):
    """H(a) = H(u) + E[log(1 - tanh(u)^2)], the expectation by quadrature on a fine u-grid."""
    u = np.linspace(mu - 12 * sigma, mu + 12 * sigma, n)
    w = norm.pdf(u, mu, sigma)
    logdet = 2.0 * (np.log(2.0) - u - np.logaddexp(0.0, -2.0 * u))
    e_logdet = np.trapezoid(w * logdet, u)
    return 0.5 * np.log(2 * np.pi * np.e * sigma ** 2) + e_logdet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seed = 0
    n = 200_000 if args.quick else 4_000_000
    n_bins = 100 if args.quick else 400
    rng = np.random.default_rng(seed)
    settings = [(0.0, 0.5), (1.0, 0.8), (0.0, 1.5)]
    print(f"tanh-squashing check | seed {seed} | {n:,} samples per setting | {n_bins} bins on (-1, 1)")

    # ---- 1. histogram vs corrected and uncorrected densities ----
    # We integrate each *formula* over every bin with adaptive quadrature (the corrected density
    # can spike very close to +-1 when sigma is large) and compare the expected counts
    # n * P(bin) with the observed counts by a chi-square statistic.  If the formula is right,
    # chi2 / dof is about 1; a wrong density gives an enormous value.
    edges = np.linspace(-1, 1, n_bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    width = edges[1] - edges[0]
    results = []
    print("\n1) Monte Carlo histogram vs bin probabilities obtained by integrating each formula")
    print(f"   {'mu':>4s} {'sigma':>5s} | {'chi2/dof corrected':>18s} | {'max |z| corrected':>17s} | "
          f"{'total mass corr.':>16s} | {'total mass uncorr.':>18s} | {'chi2/dof uncorrected':>20s}")
    for mu, sigma in settings:
        a = np.tanh(rng.normal(mu, sigma, n))
        counts, _ = np.histogram(a, bins=edges)
        stats = {}
        for name, dens in [("corr", lambda x: density_tanh_gauss(x, mu, sigma) if abs(x) < 1 else 0.0),
                           ("unc", lambda x: norm.pdf(np.arctanh(x), mu, sigma) if abs(x) < 1 else 0.0)]:
            pbin = np.array([quad(dens, lo, hi, limit=200, epsabs=1e-13)[0]
                             for lo, hi in zip(edges[:-1], edges[1:])])
            mass = pbin.sum()
            expected = n * pbin
            m = expected >= 5
            chi2 = np.sum((counts[m] - expected[m]) ** 2 / expected[m]) / (m.sum() - 1)
            zmax = np.max(np.abs(counts[m] - expected[m]) / np.sqrt(expected[m]))
            stats[name] = (chi2, zmax, mass)
        print(f"   {mu:4.1f} {sigma:5.1f} | {stats['corr'][0]:18.3f} | {stats['corr'][1]:17.2f} | "
              f"{stats['corr'][2]:16.5f} | {stats['unc'][2]:18.4f} | {stats['unc'][0]:20.1f}")
        results.append(dict(mu=mu, sigma=sigma, p_mc=counts / (n * width),
                            p_corr=density_tanh_gauss(centers, mu, sigma),
                            p_unc=norm.pdf(np.arctanh(centers), mu, sigma)))

    # ---- 2. sac.py's log-prob vs torch TanhTransform vs histogram; entropy estimate ----
    print("\n2) sac.py log-prob vs torch.distributions TanhTransform, and entropy checks:")
    torch.manual_seed(seed)
    actor = SquashedGaussianActor(obs_dim=1, act_dim=1, hidden=8)
    for mu, sigma in settings:
        with torch.no_grad():   # force the actor's output layer to produce (mu, log sigma)
            last = actor.net[-1]
            last.weight.zero_()
            last.bias.copy_(torch.tensor([mu, math.log(sigma)]))
            s = torch.zeros(n if args.quick else 1_000_000, 1)
            a, logp = actor(s)
            ref = TransformedDistribution(Normal(torch.tensor(mu), torch.tensor(sigma)), [TanhTransform()])
            inner = a.clamp(-1 + 1e-6, 1 - 1e-6)   # torch's inverse needs |a| < 1
            logp_ref = ref.log_prob(inner)          # shape (N, 1) like logp
            ok = (a.abs() < 0.999).squeeze(1)       # compare away from float32 saturation
            diff = (logp[ok] - logp_ref[ok]).abs().max().item()
        H_mc = -logp.mean().item()
        H_quad = entropy_tanh_gauss(mu, sigma)
        print(f"   mu={mu:3.1f} sigma={sigma:3.1f}: max |logp_sac - logp_torch| = {diff:.2e} (|a|<0.999);"
              f"  entropy E[-log p]: MC {H_mc:7.4f}  quadrature {H_quad:7.4f}  "
              f"(Gaussian before squashing {0.5 * math.log(2 * math.pi * math.e * sigma ** 2):7.4f})")

    # ---- 3. numerical stability of the log-det term ----
    print("\n3) log(1 - tanh(u)^2): exact (float64) vs stable formula (float32) vs naive (float32, +1e-6):")
    us = np.array([0.5, 2.0, 5.0, 8.0, 10.0, 15.0])
    exact = np.log(4.0) - 2.0 * np.abs(us) - 2.0 * np.log1p(np.exp(-2.0 * np.abs(us)))
    ut = torch.tensor(us, dtype=torch.float32).unsqueeze(1)
    stable = tanh_log_det_jacobian(ut).squeeze(1).numpy()
    naive = torch.log(1 - torch.tanh(ut) ** 2 + 1e-6).squeeze(1).numpy()
    naive0 = torch.log(1 - torch.tanh(ut) ** 2).squeeze(1).numpy()
    for u, e, s_, nv, n0 in zip(us, exact, stable, naive, naive0):
        print(f"   u={u:5.1f}: exact {e:9.4f}  stable {s_:9.4f}  naive+1e-6 {nv:9.4f}  naive (no eps) {n0:9.4f}")

    # ---- 4. most entropic tanh-Gaussian and the meaning of target entropy -1 ----
    res = minimize_scalar(lambda ls: -entropy_tanh_gauss(0.0, math.exp(ls)), bounds=(-3, 2), method="bounded")
    s_star = math.exp(res.x)
    print(f"\n4) max over sigma of H(tanh(N(0, sigma^2))) = {-res.fun:.4f} nats at sigma = {s_star:.3f} "
          f"(uniform on [-1,1]: log 2 = {math.log(2):.4f})")
    sig_target = math.exp(-1.0) / math.sqrt(2 * math.pi * math.e)
    print(f"   a Gaussian with entropy -1 nat has std {sig_target:.4f}; "
          f"tanh-Gaussian (mu=0) with entropy -1 has sigma = "
          f"{math.exp(minimize_scalar(lambda ls: (entropy_tanh_gauss(0.0, math.exp(ls)) + 1) ** 2, bounds=(-6, 0), method='bounded').x):.4f}")
    # Off-centre, tanh compresses intervals near the bounds, so the same entropy allows a larger
    # pre-squash sigma; the spread in action space (std of a) changes much less.
    for h_target, means in [(-1.0, [0.0, 0.5, 1.0, 1.5]), (-3.27, [0.0, 1.0, 2.0])]:
        parts = []
        for m in means:
            ls = minimize_scalar(lambda ls: (entropy_tanh_gauss(m, math.exp(ls)) - h_target) ** 2,
                                 bounds=(-9, math.log(s_star)), method="bounded").x
            sg = math.exp(ls)
            assert abs(entropy_tanh_gauss(m, sg) - h_target) < 1e-3, "target entropy not attained"
            u = np.linspace(m - 12 * sg, m + 12 * sg, 200_001)
            w = norm.pdf(u, m, sg)
            ea, ea2 = np.trapezoid(w * np.tanh(u), u), np.trapezoid(w * np.tanh(u) ** 2, u)
            parts.append(f"m={m:g}: sigma {sg:.4f}, std(a) {math.sqrt(max(ea2 - ea ** 2, 0)):.4f}")
        print(f"   entropy {h_target:g} nats:  " + ";  ".join(parts))

    if not args.quick:
        from plot_style import C, setup
        plt = setup()
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
        for ax, r in zip(axes[:3], results):
            ax.bar(centers, r["p_mc"], width=width, color="#c9c8c3", label="Monte Carlo histogram")
            ax.plot(centers, r["p_corr"], color=C[0], lw=1.8, label="corrected: N(atanh a)/(1-a^2)")
            ax.plot(centers, r["p_unc"], color=C[1], lw=1.6, ls="--", label="uncorrected: N(atanh a)")
            ax.set_title(f"u ~ N({r['mu']:g}, {r['sigma']:g}^2), a = tanh(u)")
            ax.set_xlabel("action a")
            ax.set_ylim(0, min(1.15 * r["p_mc"].max(), 6))
        axes[0].set_ylabel("density")
        axes[0].legend(fontsize=8, loc="upper left")
        u = np.linspace(0, 15, 400)
        ex = np.log(4.0) - 2 * u - 2 * np.log1p(np.exp(-2 * u))
        ut = torch.tensor(u, dtype=torch.float32).unsqueeze(1)
        axes[3].plot(u, ex, color="#0b0b0b", lw=2.5, label="exact")
        axes[3].plot(u, tanh_log_det_jacobian(ut).squeeze(1).numpy(), color=C[0], ls="--", lw=1.8,
                     label="2(log2 - u - softplus(-2u)), float32")
        axes[3].plot(u, torch.log(1 - torch.tanh(ut) ** 2 + 1e-6).squeeze(1).numpy(), color=C[1], lw=1.8,
                     label="log(1 - tanh(u)^2 + 1e-6), float32")
        axes[3].set_xlabel("u (pre-squash)")
        axes[3].set_ylabel("log(1 - tanh(u)^2)")
        axes[3].set_title("the log-det term for large |u|")
        axes[3].legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, "tanh_squash_check.png")
        fig.savefig(path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
