"""Forward vs reverse KL: mode-covering vs mode-seeking.

Chapter 00, Section 5.4. We fit a single Gaussian q = N(m, s^2) to a bimodal mixture p by
  * forward KL   min_q D_KL(p || q) = E_p[log p - log q]   -> moment matching (Eq. 5.9),
  * reverse KL   min_q D_KL(q || p) = E_q[log q - log p]   -> numerical minimisation;
    it is non-convex (here: one local minimum per mode plus a wide one between them),
    so we start from several initialisations. Locking onto a mode of weight w costs
    about -log w (the 'weak penalty' for ignoring the other mode).
Also prints the discrete asymmetry example worked by hand in Section 5.3.

All integrals are computed by quadrature on a fine grid (1-D, so this is exact enough).

Run:  python code/ch00_math_toolkit/kl_divergence.py [--quick]
"""
import argparse
import os
import time

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm

from plot_style import setup, C, GREY

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# target: a two-component Gaussian mixture
W = np.array([0.6, 0.4]); MU = np.array([-2.0, 2.5]); SD = np.array([0.6, 0.8])


def p_pdf(x):
    return sum(w * norm.pdf(x, m, s) for w, m, s in zip(W, MU, SD))


def kl_grid(f, g, x):
    """D_KL(f || g) = int f log(f/g) dx on grid x, with the 0 log 0 = 0 convention."""
    dx = x[1] - x[0]
    mask = f > 1e-300
    return float(np.sum(f[mask] * (np.log(f[mask]) - np.log(np.maximum(g[mask], 1e-300)))) * dx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    np.random.default_rng(args.seed)  # (deterministic script; seed printed for uniformity)
    print(f"[kl_divergence] seed={args.seed} quick={args.quick}")
    t0 = time.time()

    # ---------------- discrete asymmetry example (Section 5.3) -------------
    p = np.array([0.5, 0.5]); q = np.array([0.9, 0.1])
    kl_pq = float(np.sum(p * np.log(p / q))); kl_qp = float(np.sum(q * np.log(q / p)))
    H_p = -float(np.sum(p * np.log(p))); H_pq = -float(np.sum(p * np.log(q)))
    print(f"\nDiscrete: p={p}, q={q}")
    print(f"  KL(p||q) = {kl_pq:.4f} nats, KL(q||p) = {kl_qp:.4f} nats  (asymmetric)")
    print(f"  H(p) = {H_p:.4f}, cross-entropy H(p,q) = {H_pq:.4f} = H(p) + KL(p||q) = {H_p + kl_pq:.4f}")

    # ---------------- continuous: Gaussian fits to a mixture ---------------
    x = np.linspace(-9, 9, 6001 if args.quick else 36001)
    px = p_pdf(x)
    dx = x[1] - x[0]
    print(f"\nMixture p: weights {W}, means {MU}, std devs {SD}; integral on grid = {px.sum() * dx:.6f}")

    # forward KL: moment matching (derived in Section 5.4)
    m_f = float(np.sum(x * px) * dx)
    s_f = float(np.sqrt(np.sum((x - m_f) ** 2 * px) * dx))
    q_fwd = norm.pdf(x, m_f, s_f)

    def reverse_kl(params):
        m, log_s = params
        qx = norm.pdf(x, m, np.exp(log_s))
        return kl_grid(qx, px, x)

    def forward_kl(params):
        m, log_s = params
        return kl_grid(px, norm.pdf(x, m, np.exp(log_s)), x)

    # sanity check: numerically minimising forward KL reproduces moment matching
    res_fwd = minimize(forward_kl, x0=[0.0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-10))
    print(f"\nForward KL fit (moment matching): m = {m_f:.4f}, s = {s_f:.4f}"
          f"   [numerical minimiser: m = {res_fwd.x[0]:.4f}, s = {np.exp(res_fwd.x[1]):.4f}]")

    rev_fits = []
    for m0 in (-3.0, -1.0, 0.0, 1.0, 3.0):
        res = minimize(reverse_kl, x0=[m0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-10))
        rev_fits.append((m0, res.x[0], float(np.exp(res.x[1])), res.fun))
    print("Reverse KL fits from different initial means (local optima):")
    for m0, m, s, f in rev_fits:
        print(f"  init m0={m0:5.1f} -> m = {m:7.4f}, s = {s:.4f}, KL(q||p) = {f:.4f}")

    # distinct local optima, best first
    uniq = []
    for _, m, sd, f in sorted(rev_fits, key=lambda t: t[3]):
        if all(abs(m - u[0]) > 0.1 for u in uniq):
            uniq.append((m, sd, f))
    print(f"  -> {len(uniq)} distinct local optima of the reverse KL")
    best_rev = uniq[0]
    q_rev = norm.pdf(x, best_rev[0], best_rev[1])
    print("\n                  KL(p||q)   KL(q||p)")
    print(f"  {'forward fit':16s} {kl_grid(px, q_fwd, x):9.4f} {kl_grid(q_fwd, px, x):9.4f}")
    for i, (m, sd, _) in enumerate(uniq):
        qx = norm.pdf(x, m, sd)
        print(f"  {'reverse opt. ' + 'ABC'[i]:16s} {kl_grid(px, qx, x):9.4f} {kl_grid(qx, px, x):9.4f}")
    print(f"  matching one mode of weight w costs about -log w: -log {W[0]} = {-np.log(W[0]):.4f}, "
          f"-log {W[1]} = {-np.log(W[1]):.4f}")
    mass_between = float(np.sum(q_fwd[(x > -0.5) & (x < 0.5)]) * dx)
    p_between = float(np.sum(px[(x > -0.5) & (x < 0.5)]) * dx)
    print(f"  forward fit puts {mass_between:.3f} of its mass in (-0.5, 0.5), where p has only {p_between:.4f}")

    # profile over the mean: min over s of each divergence, for every m
    ms = np.linspace(-4, 4.5, 41 if args.quick else 171)
    prof_f, prof_r = [], []
    for m in ms:
        s2 = np.sum((x - m) ** 2 * px) * dx       # optimal s for forward KL at fixed m
        prof_f.append(forward_kl([m, 0.5 * np.log(s2)]))
        r = minimize_scalar(lambda ls: reverse_kl([m, ls]), bounds=(-3, 2), method="bounded")
        prof_r.append(r.fun)

    if not args.quick:
        plt = setup()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        ax = axes[0]
        ax.fill_between(x, px, color=GREY, alpha=0.25, lw=0)
        ax.plot(x, px, color=INK_COLOR, lw=1.5, label="target p (mixture)")
        ax.plot(x, q_fwd, "--", color=C[1], label=f"argmin KL(p‖q): N({m_f:.2f}, {s_f:.2f}²)")
        for i, (m, sd, f) in enumerate(uniq):
            ax.plot(x, norm.pdf(x, m, sd), ["-", ":", "-."][i], color=[C[0], C[2], C[3]][i],
                    lw=2.0 if i < 2 else 1.4,
                    label=f"KL(q‖p) local opt. {'ABC'[i]}: N({m:.2f}, {sd:.2f}²), KL={f:.2f}")
        ax.set_xlim(-6, 6.5); ax.set_ylim(0, 0.75)
        ax.set_title("(a) Forward KL covers modes, reverse KL seeks one")
        ax.set_xlabel("x"); ax.set_ylabel("density"); ax.legend(fontsize=8, loc="upper left")

        ax = axes[1]
        ax.plot(ms, prof_f, "--", color=C[1], label="min over s of KL(p‖q)")
        ax.plot(ms, prof_r, "-", color=C[0], label="min over s of KL(q‖p)")
        for mu in MU:
            ax.axvline(mu, color=GREY, lw=0.8)
        ax.set_ylim(0, 4)
        ax.set_title("(b) Divergence vs the mean m of q (s optimised)")
        ax.set_xlabel("mean m of q"); ax.set_ylabel("KL (nats)"); ax.legend()
        fig.tight_layout()
        path = os.path.join(FIG_DIR, "kl_forward_reverse.png")
        fig.savefig(path)
        print(f"\nsaved {path}")

    print(f"\nSummary: forward-KL fit N({m_f:.2f},{s_f:.2f}^2) straddles both modes; reverse-KL fits lock onto "
          f"one mode (best: m={best_rev[0]:.2f}, s={best_rev[1]:.2f}). runtime {time.time() - t0:.1f}s")


INK_COLOR = "#0b0b0b"

if __name__ == "__main__":
    main()
