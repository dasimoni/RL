"""Feature construction for linear methods: a visual gallery and two small
experiments (Chapter 08, Sections 6 and 7).

1. Gallery of 1-D bases on [0,1] (polynomial, Fourier cosine, RBF), 2-D Fourier
   features, and the generalisation pattern of tile coding after ONE update at a
   single point -- uniform offsets give diagonal artefacts, asymmetric offsets
   (1, 3) do not (cf. S&B Fig 9.11).
2. Coarse coding feature width (cf. S&B Fig 9.8): learn a square pulse with linear
   SGD using intervals that are narrow, medium or broad. Width shapes early
   generalisation; the final acuity is set by the number/density of features.
3. Step-size rule (Section 7): a single SGD update changes v_hat(s) by
   alpha * delta * x(s)^T x(s); with alpha = 1 / (tau * x^T x) one update moves the
   estimate 1/tau of the way to the target.

Run:  python code/ch08_function_approximation/feature_gallery.py [--quick]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread: the machine is shared
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse

import numpy as np

from tiles import TileCoder

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def coarse_coding_run(width, n_features, n_samples_list, alpha_total=0.2, seed=0):
    """Linear SGD on intervals of a given width. Returns the learned function on a grid
    after each sample count in n_samples_list."""
    rng = np.random.default_rng(seed)
    centers = np.linspace(-width / 2, 1 + width / 2, n_features)
    grid = np.linspace(0, 1, 500)
    target = lambda x: ((x >= 0.4) & (x < 0.6)).astype(float)       # a square pulse
    feats = lambda x: (np.abs(np.atleast_1d(x)[:, None] - centers[None, :]) <= width / 2).astype(float)
    Xg = feats(grid)
    w = np.zeros(n_features)
    out = {}
    xs = rng.random(max(n_samples_list))
    for t, x in enumerate(xs, start=1):
        f = feats(x)[0]
        n_active = max(f.sum(), 1.0)
        w += (alpha_total / n_active) * (target(np.array([x]))[0] - f @ w) * f   # alpha = 0.2 / n
        if t in n_samples_list:
            out[t] = Xg @ w
    return grid, target(grid), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figures")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"seed={args.seed} quick={args.quick}")

    # ---------------- 3. step-size rule ----------------
    tc = TileCoder([0, 0], [1, 1], n_tilings=8, tiles_per_dim=10)
    s = np.array([0.37, 0.61])
    idx = tc.active(s)
    print("Step-size rule with tile coding (x^T x = number of tilings = 8):")
    for tau in (1, 10):
        w = np.zeros(tc.n_features)
        alpha = 1.0 / (tau * len(idx))
        target = 1.0
        w[idx] += alpha * (target - w[idx].sum())          # one SGD step, binary features
        print(f"  alpha = 1/({tau} * 8): after one update v_hat(s) = {w[idx].sum():.4f} (target 1.0)")

    # ---------------- 2. coarse coding widths ----------------
    counts = [10, 40, 160, 640, 2560, 10240]
    if args.quick:
        counts = [10, 160]
    widths = {"narrow (0.04)": 0.04, "medium (0.12)": 0.12, "broad (0.36)": 0.36}
    results = {}
    for name, wd in widths.items():
        grid, tgt, out = coarse_coding_run(wd, 100, counts, seed=args.seed)
        results[name] = out
        errs = [np.sqrt(np.mean((out[c] - tgt) ** 2)) for c in counts]
        print(f"  coarse coding {name:14s}: RMS error after {counts} samples = {np.round(errs, 3).tolist()}")

    if args.quick:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    # ---------------- 1. gallery ----------------
    u = np.linspace(0, 1, 400)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i in range(5):
        axes[0, 0].plot(u, u ** i, label=f"$s^{i}$")
        axes[0, 1].plot(u, np.cos(i * np.pi * u), label=rf"$\cos({i}\pi s)$")
    for c in np.linspace(0, 1, 5):
        axes[0, 2].plot(u, np.exp(-(u - c) ** 2 / (2 * 0.1 ** 2)), label=f"c={c:.2f}")
    for ax, t in zip(axes[0], ["Polynomial basis", "Fourier cosine basis", r"Radial basis functions ($\sigma$=0.1)"]):
        ax.set_title(t)
        ax.set_xlabel("state s (normalised)")
        ax.legend(fontsize=7)
    # 2-D Fourier features
    g = np.linspace(0, 1, 120)
    S1, S2 = np.meshgrid(g, g)
    cs = [(1, 0), (0, 1), (1, 1), (2, 5)]
    img = np.concatenate([np.concatenate([np.cos(np.pi * (c[0] * S1 + c[1] * S2)) for c in cs[:2]], axis=1),
                          np.concatenate([np.cos(np.pi * (c[0] * S1 + c[1] * S2)) for c in cs[2:]], axis=1)], axis=0)
    axes[1, 0].imshow(img, origin="lower", cmap="RdBu", extent=[0, 2, 0, 2])
    for k, c in enumerate(cs):
        axes[1, 0].text((k % 2) + 0.05, (k // 2) + 0.05, f"c={c}", fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.7, lw=0))
    axes[1, 0].axhline(1, color="k", lw=1)
    axes[1, 0].axvline(1, color="k", lw=1)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    axes[1, 0].set_title(r"2-D Fourier features $\cos(\pi\, \mathbf{c}^\top \mathbf{s})$")
    # generalisation of one update: uniform vs asymmetric offsets
    for ax, mode in [(axes[1, 1], "uniform"), (axes[1, 2], "asymmetric")]:
        tcm = TileCoder([0, 0], [1, 1], n_tilings=8, tiles_per_dim=4, offsets=mode)
        w = np.zeros(tcm.n_features)
        idx = tcm.active([0.5, 0.5])
        w[idx] += (1.0 / 8) * (1.0 - w[idx].sum())
        V = np.array([[w[tcm.active([a, b])].sum() for a in g] for b in g])
        im = ax.imshow(V, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", vmin=0, vmax=1)
        ax.plot(0.5, 0.5, "r+", ms=10)
        ax.set_title(f"Tile coding, 8 tilings, {mode} offsets:\nv_hat after one update at (0.5, 0.5)")
        ax.set_xlabel("$s_1$")
        ax.set_ylabel("$s_2$")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "feature_gallery.png"), dpi=100)

    fig, axes = plt.subplots(len(widths), len(counts), figsize=(16, 6.5), sharex=True, sharey=True)
    for i, (name, out) in enumerate(results.items()):
        for j, c in enumerate(counts):
            ax = axes[i, j]
            ax.plot(grid, tgt, color="gray", lw=1)
            ax.plot(grid, out[c], color="tab:blue")
            if i == 0:
                ax.set_title(f"{c} samples")
            if j == 0:
                ax.set_ylabel(name)
            ax.set_ylim(-0.4, 1.5)
    fig.suptitle("Coarse coding: feature width affects early generalisation, not final acuity")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "coarse_coding_width.png"), dpi=100)
    print(f"figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
