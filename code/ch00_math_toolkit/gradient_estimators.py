"""Score-function (REINFORCE) vs reparameterization gradient estimators.

Chapter 00, Section 6. We estimate  g = d/dmu E_{x ~ N(mu, sigma^2)}[f(x)]  with single-sample
estimators (x = mu + sigma*eps, eps ~ N(0, 1)):

  score function   g_SF = f(x) * d/dmu log N(x; mu, sigma^2) = f(x) (x - mu) / sigma^2     (Eq. 6.3)
  with baseline    g_SFb = (f(x) - b) (x - mu) / sigma^2                                      (Eq. 6.6)
  reparameterized  g_RP = f'(x) * dx/dmu = f'(x)                                               (Eq. 6.9)
  (The noise is called xi in the chapter, because epsilon is reserved for epsilon-greedy.)

  (a) f(x) = x^2: compare measured variances with the closed forms of Section 6.5,
      and sweep sigma (as sigma -> 0, SF without a baseline blows up, SF with the
      baseline E[f] stays bounded, reparameterization vanishes).
  (b) dimension D: f(x) = ||x||^2, x in R^D. SF variance per coordinate grows with D.
  (c) a counterexample: f(x) = cos(w x). For large w the reparameterization estimator
      is WORSE than the score function.
  (d) discrete actions (softmax policy over 5 actions): reparameterization is unavailable,
      the score function works, and a baseline matters when rewards have a large offset.
  (e) the same two estimators written in PyTorch (surrogate loss vs rsample).

Run:  python code/ch00_math_toolkit/gradient_estimators.py [--quick]
"""
import argparse
import os
import time

import numpy as np
import torch

from plot_style import setup, C, GREY

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def estimators_square(mu, sigma, eps):
    """Single-sample estimators for f(x) = x^2 (true gradient 2 mu)."""
    x = mu + sigma * eps
    f = x ** 2
    score = (x - mu) / sigma ** 2          # d/dmu log N(x; mu, sigma^2)
    b = mu ** 2 + sigma ** 2               # baseline = E[f(x)] (a constant w.r.t. x)
    return dict(sf=f * score, sfb=(f - b) * score, rp=2 * x)


def var_theory_square(mu, sigma):
    """Closed forms derived in Section 6.5 (worked example)."""
    return dict(sf=mu ** 4 / sigma ** 2 + 14 * mu ** 2 + 15 * sigma ** 2,
                sfb=8 * mu ** 2 + 10 * sigma ** 2,
                rp=4 * sigma ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    M = 200_000 if args.quick else 2_000_000
    print(f"[gradient_estimators] seed={args.seed} samples per estimate={M} quick={args.quick}")
    t0 = time.time()

    # ------------------------------------------------------------------ (a)
    mu, sigma = 1.0, 1.0
    eps = rng.standard_normal(M)
    est = estimators_square(mu, sigma, eps)
    th = var_theory_square(mu, sigma)
    print(f"\n(a) f(x)=x^2, mu={mu}, sigma={sigma}; true gradient = {2 * mu}")
    print(f"  {'estimator':16s} {'mean':>8s} {'± s.e.':>8s} {'variance':>9s} {'theory':>7s}")
    for k, name in [("sf", "score function"), ("sfb", "SF + baseline"), ("rp", "reparam")]:
        g = est[k]
        print(f"  {name:16s} {g.mean():8.4f} {g.std() / np.sqrt(M):8.4f} {g.var():9.3f} {th[k]:7.1f}")

    sigmas = np.geomspace(0.1, 3.0, 9 if args.quick else 25)
    sweep = {k: [] for k in ("sf", "sfb", "rp")}
    eps_s = rng.standard_normal(M // 4)
    for s in sigmas:
        e = estimators_square(mu, s, eps_s)
        for k in sweep:
            sweep[k].append(e[k].var())
    th_small = var_theory_square(mu, sigmas[0])
    print(f"  sigma sweep, smallest sigma={sigmas[0]:.2f}: SF {sweep['sf'][0]:.1f} (theory {th_small['sf']:.1f}), "
          f"SF+baseline {sweep['sfb'][0]:.2f} (theory {th_small['sfb']:.2f} -> 8 mu^2 = {8 * mu ** 2:.0f}), "
          f"reparam {sweep['rp'][0]:.3f} (theory {th_small['rp']:.3f})")

    # ------------------------------------------------------------------ (b)
    Ds = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    Mb = 20_000 if args.quick else 100_000
    dim = {k: [] for k in ("sf", "sfb", "rp")}
    print(f"\n(b) f(x)=||x||^2 in D dims, mu=1-vector, sigma=1: variance of gradient coordinate 1 ({Mb} samples)")
    for D in Ds:
        E = rng.standard_normal((Mb, D))
        X = 1.0 + E
        f = (X ** 2).sum(axis=1)
        b = D * (1.0 + 1.0)                    # E||x||^2 = D (mu^2 + sigma^2)
        dim["sf"].append((f * E[:, 0]).var())
        dim["sfb"].append(((f - b) * E[:, 0]).var())
        dim["rp"].append((2 * X[:, 0]).var())
        print(f"  D={D:4d}: SF {dim['sf'][-1]:10.1f}   SF+baseline {dim['sfb'][-1]:8.1f}   reparam {dim['rp'][-1]:6.2f}")

    # ------------------------------------------------------------------ (c)
    mu_c, sigma_c = 0.3, 1.0
    omegas = np.geomspace(0.1, 30, 9 if args.quick else 30)
    cos_res = {"sf": [], "rp": [], "true": []}
    eps_c = rng.standard_normal(M // 4)
    x = mu_c + sigma_c * eps_c
    for w in omegas:
        f = np.cos(w * x)
        cos_res["sf"].append((f * eps_c / sigma_c).var())
        cos_res["rp"].append((-w * np.sin(w * x)).var())
        cos_res["true"].append(-w * np.sin(w * mu_c) * np.exp(-0.5 * w ** 2 * sigma_c ** 2))
    cross = omegas[np.argmax(np.array(cos_res["rp"]) > np.array(cos_res["sf"]))]
    print(f"\n(c) f(x)=cos(w x), mu={mu_c}, sigma={sigma_c}: reparam variance exceeds SF variance from w ≈ {cross:.2f}")
    for w, vs, vr in list(zip(omegas, cos_res["sf"], cos_res["rp"]))[:: max(1, len(omegas) // 5)]:
        print(f"  w={w:7.3f}: Var SF {vs:8.3f}   Var reparam {vr:8.3f}")
    # closed form of Var[-w sin(w x)], x ~ N(mu, sigma^2), using E cos(c x) = cos(c mu) exp(-c^2 sigma^2 / 2):
    #   w^2 (1 - cos(2 w mu) e^{-2 w^2 sigma^2}) / 2  -  (w sin(w mu) e^{-w^2 sigma^2 / 2})^2   ~  w^2 / 2
    w_last = omegas[-1]
    rp_closed = (w_last ** 2 * (1 - np.cos(2 * w_last * mu_c) * np.exp(-2 * w_last ** 2 * sigma_c ** 2)) / 2
                 - (w_last * np.sin(w_last * mu_c) * np.exp(-0.5 * w_last ** 2 * sigma_c ** 2)) ** 2)
    print(f"  w={w_last:7.3f}: Var SF {cos_res['sf'][-1]:8.3f}   Var reparam {cos_res['rp'][-1]:8.1f} "
          f"(closed form {rp_closed:.1f}, ~ w^2/2 = {w_last ** 2 / 2:.0f})")

    # ------------------------------------------------------------------ (d)
    # softmax policy over 5 actions, objective J(theta) = sum_a pi(a) r(a)
    theta = np.array([0.5, -0.2, 0.1, 0.0, 0.3])
    r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pi = np.exp(theta - theta.max()); pi /= pi.sum()
    true_grad = pi * (r - pi @ r)            # d/dtheta_k sum_a pi(a) r(a) = pi_k (r_k - J)
    Md = M // 4
    print(f"\n(d) softmax policy over 5 actions, pi={np.round(pi, 3)}; exact gradient {np.round(true_grad, 4)}")
    for offset in (0.0, 100.0):
        rr = r + offset
        a = rng.choice(5, size=Md, p=pi)
        score = -np.tile(pi, (Md, 1)); score[np.arange(Md), a] += 1.0   # grad log softmax = e_a - pi
        g_nob = rr[a, None] * score
        g_b = (rr[a, None] - pi @ rr) * score  # baseline = J(theta)
        print(f"  reward offset {offset:5.0f}: mean SF grad {np.round(g_nob.mean(0), 3)}, "
              f"total variance no-baseline {g_nob.var(0).sum():10.2f}, with baseline {g_b.var(0).sum():6.3f}")

    # ------------------------------------------------------------------ (e)
    # The same estimators the way you would write them in PyTorch.
    mu_t = torch.tensor(1.0, requires_grad=True)
    dist = torch.distributions.Normal(mu_t, torch.tensor(1.0))
    n_t = 200_000
    # score function: differentiate the SURROGATE  mean( f(x) * log p(x) ), with f(x) held constant
    x_sf = dist.sample((n_t,))                       # .sample() is not differentiable
    surrogate = (x_sf ** 2 * dist.log_prob(x_sf)).mean()
    g_sf_torch, = torch.autograd.grad(surrogate, mu_t)
    # reparameterization: x = mu + sigma*xi is a differentiable function of mu
    x_rp = dist.rsample((n_t,))
    g_rp_torch, = torch.autograd.grad((x_rp ** 2).mean(), mu_t)
    print(f"\n(e) PyTorch, {n_t} samples: score-function surrogate grad {g_sf_torch.item():.4f}, "
          f"rsample grad {g_rp_torch.item():.4f}  (true 2.0)")

    if not args.quick:
        plt = setup()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        ax = axes[0]
        labels = {"sf": "score function", "sfb": "SF + baseline E[f]", "rp": "reparameterization"}
        marks = {"sf": "o", "sfb": "s", "rp": "^"}
        for i, k in enumerate(("sf", "sfb", "rp")):
            ax.loglog(sigmas, sweep[k], marks[k], color=C[[1, 3, 0][i]], ms=5, label=labels[k] + " (measured)")
            ax.loglog(sigmas, var_theory_square(mu, sigmas)[k], "-", color=C[[1, 3, 0][i]], lw=1.0)
        ax.plot([], [], color=GREY, lw=1, label="closed form (Sec. 6.5)")
        ax.set_title(r"(a) $f(x)=x^2$, $\mu=1$: variance vs $\sigma$")
        ax.set_xlabel("σ"); ax.set_ylabel("variance of single-sample estimator"); ax.legend(fontsize=8)

        ax = axes[1]
        for i, k in enumerate(("sf", "sfb", "rp")):
            ax.loglog(Ds, dim[k], marks[k] + "-", color=C[[1, 3, 0][i]], ms=5, label=labels[k])
        ax.set_title(r"(b) $f(\mathbf{x})=\|\mathbf{x}\|^2$: variance of one coordinate vs D")
        ax.set_xlabel("dimension D"); ax.set_ylabel("variance"); ax.legend()

        ax = axes[2]
        ax.loglog(omegas, cos_res["sf"], "o-", color=C[1], ms=4, label="score function")
        ax.loglog(omegas, cos_res["rp"], "^-", color=C[0], ms=4, label="reparameterization")
        ax.axvline(cross, color=GREY, lw=0.8)
        ax.set_title(r"(c) $f(x)=\cos(\omega x)$: reparam can lose")
        ax.set_xlabel("frequency ω"); ax.set_ylabel("variance"); ax.legend()
        fig.tight_layout()
        path = os.path.join(FIG_DIR, "gradient_estimators.png")
        fig.savefig(path)
        print(f"\nsaved {path}")

    print(f"\nSummary: at mu=sigma=1 the variances are SF {est['sf'].var():.1f}, SF+baseline {est['sfb'].var():.1f}, "
          f"reparam {est['rp'].var():.2f} (theory 30/18/4); at D=256 SF is {dim['sf'][-1] / dim['rp'][-1]:.0f}x "
          f"reparam. runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
