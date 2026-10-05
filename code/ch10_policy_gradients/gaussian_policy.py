"""Gaussian policies for continuous actions: score formulas, tanh squashing, learning, and DPG preview.

Chapter 10, Sections 2.2-2.3 and 11.

Part 1  The Gaussian score function (Eq. 10.4): closed forms for d/d mu and d/d log sigma of
        log N(a; mu, sigma^2) (diagonal, 3-D action) checked against PyTorch autograd.
Part 2  The tanh-squashed Gaussian (Eq. 10.5): the change-of-variables log-density checked
        against a histogram of 2,000,000 samples; the error made by forgetting the Jacobian term.
Part 3  REINFORCE on a continuous contextual bandit: context s ~ U(-1, 1), action a in R,
        reward r = -(a - a*(s))^2 + noise with a*(s) = sin(pi s).  Policy: mean mu_theta(s) =
        theta . phi(s) with 9 radial-basis features, state-independent log sigma.  Compares no
        baseline, a learned baseline, a learned baseline + entropy bonus, and (as a control)
        a learned baseline with NOISELESS rewards.  The optimal policy here is deterministic,
        so REINFORCE keeps shrinking sigma.  The reward noise eps then enters the estimate as
        eps (A - mu)/sigma^2, with variance Var(eps)/sigma^2 per unit feature, which no state
        baseline can cancel (Section 11.1); the noiseless control isolates that effect.
Part 4  Deterministic vs stochastic policy gradient (Section 11.3): at a fixed mean function,
        the variance of the one-sample score-function estimate of grad_theta J as sigma -> 0.
        Columns 1-3 use the NOISE-FREE weight q(s, A) - b(s): without a baseline, or with a
        baseline that is off by a constant, the variance grows like 1/sigma^2; with the exact
        baseline v_pi it stays bounded.  Column 4 uses the exact baseline but a NOISY sampled
        reward r = q(s, A) + eps (s.d. NOISE_SD), as REINFORCE would see it: the 1/sigma^2
        growth comes back.  The deterministic estimate phi(s) * dq/da(s, mu(s)) (which needs
        the action-gradient of q) is lowest.

Outputs (full mode): figures/gaussian_policy.png
Run:  python code/ch10_policy_gradients/gaussian_policy.py [--quick]
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch

torch.set_num_threads(1)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
CENTERS = torch.linspace(-1, 1, 9)
NOISE_SD = 0.1


def a_star(s):
    return torch.sin(math.pi * s)


def features(s):
    """Normalized radial-basis features phi(s) of a scalar context (shape (B, 9))."""
    f = torch.exp(-((s[:, None] - CENTERS[None, :]) ** 2) / (2 * 0.15 ** 2))
    return f / f.sum(1, keepdim=True)


# ----------------------------------------------------------------------------- Part 1
def part1(gen):
    print("=== Part 1: Gaussian score function vs autograd (3-D diagonal Gaussian) ===")
    mu = torch.randn(3, generator=gen, dtype=torch.float64, requires_grad=True)
    log_sd = (0.3 * torch.randn(3, generator=gen, dtype=torch.float64)).requires_grad_(True)
    a = (mu + log_sd.exp() * torch.randn(3, generator=gen, dtype=torch.float64)).detach()
    logp = torch.distributions.Normal(mu, log_sd.exp()).log_prob(a).sum()
    g_mu, g_ls = torch.autograd.grad(logp, [mu, log_sd])
    sd = log_sd.exp().detach()
    closed_mu = (a - mu.detach()) / sd ** 2                       # Eq. (10.4), mean
    closed_ls = (a - mu.detach()) ** 2 / sd ** 2 - 1.0            # Eq. (10.4), log-std
    e1 = (g_mu - closed_mu).abs().max().item()
    e2 = (g_ls - closed_ls).abs().max().item()
    print(f"max |autograd - closed form|: d/d mu {e1:.1e}, d/d log sigma {e2:.1e}")
    return max(e1, e2)


# ----------------------------------------------------------------------------- Part 2
def part2(gen, n):
    print("\n=== Part 2: tanh-squashed Gaussian, a = tanh(u), u ~ N(0.8, 0.7^2) ===")
    mu, sd = 0.8, 0.7
    u = mu + sd * torch.randn(n, generator=gen, dtype=torch.float64)
    a = torch.tanh(u)
    edges = torch.linspace(-0.95, 0.95, 39, dtype=torch.float64)
    hist = torch.histc(a, bins=38, min=-0.95, max=0.95) / (n * (edges[1] - edges[0]))
    mid = 0.5 * (edges[1:] + edges[:-1])
    u_mid = torch.atanh(mid)
    base = torch.distributions.Normal(mu, sd).log_prob(u_mid)
    correct = (base - torch.log1p(-mid ** 2)).exp()               # Eq. (10.5)
    forgot = base.exp()                                            # Jacobian term forgotten
    rel_ok = ((correct - hist).abs() / hist).max().item()
    rel_bad = ((forgot - hist).abs() / hist).max().item()
    print(f"max relative error vs histogram over 38 bins in [-0.95, 0.95]: with Jacobian "
          f"{rel_ok:.3f}, without {rel_bad:.3f}")
    return rel_ok, rel_bad


# ----------------------------------------------------------------------------- Part 3
def train_bandit(seed, iters, batch, use_baseline, c_ent, noise_sd=NOISE_SD, lr=0.05):
    gen = torch.Generator().manual_seed(seed)
    theta = torch.zeros(9, requires_grad=True)                    # mean weights
    log_sd = torch.tensor(0.0, requires_grad=True)                # sigma_0 = 1
    w = torch.zeros(9)                                            # linear baseline weights
    opt = torch.optim.Adam([theta, log_sd], lr=lr)
    hist = dict(J=[], sd=[], gnorm=[])
    for it in range(iters):
        s = 2 * torch.rand(batch, generator=gen) - 1
        phi = features(s)
        mu = phi @ theta
        sd = log_sd.exp()
        a = (mu + sd * torch.randn(batch, generator=gen)).detach()   # A ~ pi(.|s), no reparameterization
        r = -(a - a_star(s)) ** 2 + noise_sd * torch.randn(batch, generator=gen)
        b = phi @ w if use_baseline else torch.zeros(batch)
        logp = torch.distributions.Normal(mu, sd).log_prob(a)
        ent = torch.log(sd) + 0.5 * math.log(2 * math.pi * math.e)   # entropy of N(mu, sd^2)
        loss = -((r - b) * logp).mean() - c_ent * ent
        opt.zero_grad()
        loss.backward()
        hist["gnorm"].append(theta.grad.norm().item())
        opt.step()
        if use_baseline:                                          # least-mean-squares step on the baseline
            w += 0.5 * ((r - phi @ w)[:, None] * phi).mean(0) * 9
        with torch.no_grad():                                     # exact J by a fine grid over s
            sg = torch.linspace(-1, 1, 401)
            mg = features(sg) @ theta
            J = -(((mg - a_star(sg)) ** 2).mean() + log_sd.exp() ** 2)
        hist["J"].append(J.item())
        hist["sd"].append(log_sd.exp().item())
    return {k: np.array(v) for k, v in hist.items()}


# ----------------------------------------------------------------------------- Part 4
def part4(gen, n):
    """Per-sample variance of estimators of grad_theta J at a fixed (imperfect) mean function.

    score function:  (q(s,a) - b(s)) (a - mu)/sigma^2 phi(s)  with three baselines b:
        none (b = 0), a slightly wrong baseline (b = v_pi + 0.1) and the exact b = v_pi;
    score function with a noisy reward:  (q(s,a) + eps - v_pi(s)) (a - mu)/sigma^2 phi(s),
        eps ~ N(0, NOISE_SD^2)  -- what REINFORCE with a perfect baseline actually computes;
    deterministic:   phi(s) dq/da(s, mu(s))  (needs the action-gradient of q).
    """
    theta = 0.5 * torch.randn(9, generator=gen)
    noise_gen = torch.Generator().manual_seed(12345)              # separate stream for reward noise
    sds = np.logspace(-2, 0, 9)
    out = []
    for sd in sds:
        s = 2 * torch.rand(n, generator=gen) - 1
        phi = features(s)
        mu = phi @ theta
        a = mu + sd * torch.randn(n, generator=gen)
        q = -(a - a_star(s)) ** 2                                 # noise-free q(s, a)
        v = -(mu - a_star(s)) ** 2 - sd ** 2                      # exact state value
        score = ((a - mu) / sd ** 2)[:, None] * phi
        row = [sd]
        for b in (torch.zeros(n), v + 0.1, v):
            row.append((q - b)[:, None].mul(score).var(0).sum().item())
        r_noisy = q + NOISE_SD * torch.randn(n, generator=noise_gen)  # a sampled reward, as REINFORCE sees it
        row.append((r_noisy - v)[:, None].mul(score).var(0).sum().item())
        g_dpg = (-2 * (mu - a_star(s)))[:, None] * phi            # phi(s) * dq/da at a = mu(s)
        row.append(g_dpg.var(0).sum().item())
        out.append(row)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n_hist, iters, n_seeds, n4 = (200_000, 150, 2, 20_000) if args.quick else (2_000_000, 1500, 5, 200_000)
    batch = 64
    print(f"seed={args.seed}  histogram samples={n_hist}  bandit: iters={iters} batch={batch} "
          f"seeds={n_seeds} lr=0.05 c_H in {{0, 0.01}}  reward noise s.d.={NOISE_SD} (and 0 as a control)")
    t0 = time.time()
    gen = torch.Generator().manual_seed(args.seed)
    part1(gen)
    part2(gen, n_hist)

    print("\n=== Part 3: REINFORCE on the continuous bandit (optimum J = 0 with sigma -> 0) ===")
    configs = {"no baseline": (False, 0.0), "learned baseline": (True, 0.0),
               "baseline + entropy 0.01": (True, 0.01),
               "baseline, noiseless reward": (True, 0.0, 0.0)}
    runs = {k: [train_bandit(args.seed * 100 + i, iters, batch, *cfg) for i in range(n_seeds)]
            for k, cfg in configs.items()}
    for k, rs in runs.items():
        J = np.array([r["J"] for r in rs])
        sd = np.array([r["sd"] for r in rs])
        gn = np.array([r["gnorm"] for r in rs])
        q = max(1, iters // 10)
        print(f"  {k:24s} final J {J[:, -q:].mean():8.4f}   final sigma {sd[:, -1].mean():.4f}   "
              f"||grad theta||: first 10% {gn[:, :q].mean():.3f}, last 10% {gn[:, -q:].mean():.3f}")

    print("\n=== Part 4: score-function vs deterministic gradient estimator, per-sample variance ===")
    res = part4(gen, n4)
    print("  (columns 1-3: noise-free weight q(s,A) - b(s); column 4: noisy reward r - v(s), noise s.d. "
          f"{NOISE_SD})")
    print("  sigma  |  tr Var: SF, no baseline | SF, b = v + 0.1 | SF, b = v (exact) | SF, noisy r - v | "
          "deterministic")
    for sd, v0, v1, v2, v3, vd in res:
        print(f"  {sd:6.3f} | {v0:24.4g} | {v1:15.4g} | {v2:17.4g} | {v3:15.4g} | {vd:13.4g}")
    print(f"Time {time.time() - t0:.1f}s")

    if not args.quick:
        from plot_style import setup, C, GREY
        plt = setup()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
        styles = ["-", "--", "-.", ":"]
        x = np.arange(1, iters + 1)
        for j, (k, rs) in enumerate(runs.items()):
            J = np.array([r["J"] for r in rs])
            sd = np.array([r["sd"] for r in rs])
            axes[0].plot(x, -J.mean(0), color=C[j], ls=styles[j], label=k)
            axes[1].plot(x, sd.mean(0), color=C[j], ls=styles[j], label=k)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("iteration (64 contexts each)")
        axes[0].set_ylabel("$-J$ = E[(mean - a*)$^2$] + $\\sigma^2$")
        axes[0].set_title(f"REINFORCE, Gaussian policy ({n_seeds} seeds)")
        axes[0].legend(fontsize=8)
        axes[1].set_yscale("log")
        axes[1].set_xlabel("iteration")
        axes[1].set_ylabel("policy standard deviation $\\sigma$")
        axes[1].set_title("REINFORCE shrinks $\\sigma$ when the optimum is deterministic")
        ax = axes[2]
        ax.loglog(res[:, 0], res[:, 1], "o-", color=C[0], label="score function, no baseline")
        ax.loglog(res[:, 0], res[:, 2], "^-.", color=C[2], label="score function, $b=v_\\pi+0.1$")
        ax.loglog(res[:, 0], res[:, 3], "D-", color=C[3], label="score function, $b=v_\\pi$ (exact)")
        ax.loglog(res[:, 0], res[:, 4], "v:", color=C[4], label="score function, noisy $r - v_\\pi$")
        ax.loglog(res[:, 0], res[:, 5], "s--", color=C[1], label="deterministic: $\\mathbf{x}(s)\\,\\partial_a q$")
        ax.loglog(res[:, 0], res[0, 2] * (res[:, 0] / res[0, 0]) ** -2, ":", color=GREY,
                  lw=1, label="$\\propto 1/\\sigma^2$")
        ax.set_xlabel("policy standard deviation $\\sigma$")
        ax.set_ylabel("tr Var of a one-sample gradient estimate")
        ax.set_title("Stochastic vs deterministic policy gradient", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "gaussian_policy.png"))
        plt.close(fig)
        print(f"figure written to {FIG_DIR}/gaussian_policy.png")


if __name__ == "__main__":
    main()
