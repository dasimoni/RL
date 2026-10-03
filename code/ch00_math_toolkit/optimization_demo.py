"""Gradient descent, momentum, Adam, and why noisy (stochastic) gradients still work.

Chapter 00, Sections 3.2-3.5. Everything is implemented from scratch in numpy so that each
line can be matched with an equation in the text.
  (a) An ill-conditioned 2-D quadratic f(w) = 1/2 w^T A w (condition number 25):
      plain GD (Eq. 3.3) with alpha = 1/L crawls along the flat valley floor (rate 1 - 1/kappa,
      Eq. 3.5); with alpha near 2/L it zig-zags across the valley; heavy-ball momentum
      (Eq. 3.10) and Adam (Eq. 3.11) for comparison.
  (b) Least-squares regression with minibatch SGD (Eq. 3.6): a constant step size
      reaches a noise floor whose height scales with alpha; a Robbins-Monro decaying
      step size keeps improving.

Run:  python code/ch00_math_toolkit/optimization_demo.py [--quick]
"""
import argparse
import os
import time

import numpy as np

from plot_style import setup, C, GREY

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ------------------------------------------------------------------ optimisers
def gd(grad, w0, alpha, K):
    w = w0.copy(); path = [w.copy()]
    for _ in range(K):
        w = w - alpha * grad(w)                       # Eq. 3.3
        path.append(w.copy())
    return np.array(path)


def heavy_ball(grad, w0, alpha, beta, K):
    w = w0.copy(); m = np.zeros_like(w); path = [w.copy()]
    for _ in range(K):
        m = beta * m + grad(w)                        # velocity: decaying sum of past gradients
        w = w - alpha * m                             # Eq. 3.10
        path.append(w.copy())
    return np.array(path)


def adam(grad, w0, alpha, K, beta1=0.9, beta2=0.999, eps=1e-8):
    w = w0.copy(); m = np.zeros_like(w); v = np.zeros_like(w); path = [w.copy()]
    for k in range(1, K + 1):
        g = grad(w)
        m = beta1 * m + (1 - beta1) * g               # first-moment EMA
        v = beta2 * v + (1 - beta2) * g * g           # second-moment EMA (per coordinate)
        m_hat = m / (1 - beta1 ** k)                  # bias correction (Eq. 3.12)
        v_hat = v / (1 - beta2 ** k)
        w = w - alpha * m_hat / (np.sqrt(v_hat) + eps)  # Eq. 3.11
        path.append(w.copy())
    return np.array(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    print(f"[optimization_demo] seed={args.seed} quick={args.quick}")
    t0 = time.time()

    # ------------------------------------------------------------------ (a)
    th = np.deg2rad(30)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A = R @ np.diag([1.0, 25.0]) @ R.T            # eigenvalues mu=1, L=25
    f = lambda w: 0.5 * w @ A @ w
    grad = lambda w: A @ w
    w0 = np.array([-4.0, -0.8])  # far along the shallow valley, a little up the steep wall
    K = 100
    L, mu_ = 25.0, 1.0
    runs = {
        "GD α=1/L": gd(grad, w0, 1 / L, K),
        "GD α=0.075 (near 2/L)": gd(grad, w0, 0.075, K),
        "heavy-ball momentum α=0.02, β=0.8": heavy_ball(grad, w0, 0.02, 0.8, K),
        "Adam α=0.3": adam(grad, w0, 0.3, K),
    }
    print(f"\n(a) quadratic with condition number L/mu = {L / mu_:.0f}, start {w0}, f(w0) = {f(w0):.2f}")
    loss_a = {}
    for name, path in runs.items():
        losses = np.array([f(w) for w in path])
        loss_a[name] = losses
        print(f"  {name:36s} f after 10/50/100 steps: {losses[10]:.2e} {losses[50]:.2e} {losses[100]:.2e}")
    print(f"  theory: GD with alpha=1/L contracts the error by (1 - mu/L) = {1 - mu_ / L:.2f} per step "
          f"(loss by {(1 - mu_ / L) ** 2:.4f})")
    gd_loss = loss_a["GD α=1/L"]
    rate = (gd_loss[100] / gd_loss[50]) ** (1 / 50)   # geometric-mean loss ratio per step, steps 50-100
    print(f"  measured: GD alpha=1/L loss ratio per step (steps 50-100) = {rate:.4f}")

    # ------------------------------------------------------------------ (b)
    N, d, sigma_noise, batch = 1000, 10, 0.5, 8
    X = rng.normal(size=(N, d)) * np.linspace(0.3, 2.0, d)   # unequal feature scales
    w_true = rng.normal(size=d)
    y = X @ w_true + sigma_noise * rng.normal(size=N)
    w_star = np.linalg.lstsq(X, y, rcond=None)[0]
    loss = lambda w: 0.5 * np.mean((X @ w - y) ** 2)
    f_star = loss(w_star)
    Kb = 3000 if args.quick else 20_000
    eval_every = 50
    Lb = np.linalg.eigvalsh(X.T @ X / N).max()
    schedules = {
        "full-batch GD, α=0.1": ("full", lambda k: 0.1),
        "SGD (batch 8), α=0.1": ("sgd", lambda k: 0.1),
        "SGD (batch 8), α=0.01": ("sgd", lambda k: 0.01),
        "SGD (batch 8), α_k=0.1/(1+k/500)": ("sgd", lambda k: 0.1 / (1 + k / 500)),
    }
    print(f"\n(b) least squares N={N}, d={d}, minibatch {batch}, {Kb} steps; L = {Lb:.2f} (so GD needs alpha < {2 / Lb:.3f})")
    curves = {}
    for name, (kind, sched) in schedules.items():
        w = np.zeros(d); xs, ys = [], []
        rng_b = np.random.default_rng(args.seed + 1)       # same minibatch sequence for every schedule
        for k in range(Kb):
            if kind == "full":
                g = X.T @ (X @ w - y) / N
            else:
                idx = rng_b.integers(0, N, batch)           # unbiased estimate of the full gradient
                g = X[idx].T @ (X[idx] @ w - y[idx]) / batch
            w -= sched(k) * g
            if k % eval_every == 0 or k == Kb - 1:
                xs.append(k + 1); ys.append(loss(w) - f_star)
        curves[name] = (np.array(xs), np.array(ys))
        tail = np.array(ys)[-20:].mean()
        print(f"  {name:36s} excess loss: after 1000 steps {ys[1000 // eval_every]:.2e}, final (avg of last 20 evals) {tail:.2e}")

    if not args.quick:
        plt = setup()
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
        ax = axes[0]
        g1, g2 = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-4, 4, 200))
        Z = 0.5 * (A[0, 0] * g1 ** 2 + 2 * A[0, 1] * g1 * g2 + A[1, 1] * g2 ** 2)
        ax.contour(g1, g2, Z, levels=np.geomspace(0.05, 200, 14), colors=GREY, linewidths=0.6)
        styles = ["-", "--", "-.", ":"]
        for (name, path), c, ls in zip(runs.items(), C, styles):
            ax.plot(path[:40, 0], path[:40, 1], ls, color=c, lw=1.4, marker="o", ms=2.5, label=name)
        ax.plot([0], [0], "*", color="#0b0b0b", ms=10)
        ax.set_aspect("equal"); ax.set_xlim(-5, 5); ax.set_ylim(-4, 4)
        ax.set_title("(a) First 40 steps on an ill-conditioned quadratic")
        ax.set_xlabel("w₁"); ax.set_ylabel("w₂"); ax.legend(fontsize=7.5, loc="upper left")

        ax = axes[1]
        for (name, losses), c, ls in zip(loss_a.items(), C, styles):
            ax.semilogy(losses, ls, color=c, label=name)
        ax.set_title("(b) Loss vs iteration (quadratic)")
        ax.set_xlabel("iteration k"); ax.set_ylabel("f(w_k)"); ax.set_ylim(1e-12, 1e3); ax.legend(fontsize=8)

        ax = axes[2]
        for (name, (xs, ys)), c, ls in zip(curves.items(), C, styles):
            ax.loglog(xs, np.maximum(ys, 1e-16), ls, color=c, label=name)  # GD hits float precision
        ax.set_title("(c) Least squares: SGD noise floor vs decaying step")
        ax.set_ylim(1e-16, 1e2)
        ax.set_xlabel("iteration k"); ax.set_ylabel(r"excess loss $f(w_k)-f(w_\ast)$"); ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(FIG_DIR, "optimizers.png")
        fig.savefig(path)
        print(f"\nsaved {path}")

    print(f"\nSummary: constant-step SGD stalls at a noise floor that shrinks with alpha; the decaying "
          f"schedule keeps going. runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
