"""Contraction mappings and the Banach fixed-point theorem, numerically.

Chapter 00, Sections 4.2-4.4.
  (a) The affine map  T(v) = r + gamma P v  (P row-stochastic) is a gamma-contraction in
      the max norm (Eq. 4.8). Iterating it from any start converges to the unique fixed
      point v* = (I - gamma P)^{-1} r (Eq. 4.4), with  ||v_k - v*||_inf <= gamma^k ||v_0 - v*||_inf
      (Eq. 4.6) and the a-posteriori stopping bound
      ||v_k - v*||_inf <= gamma/(1-gamma) ||v_k - v_{k-1}||_inf  (Eq. 4.7).
      (This map is the Bellman expectation operator of Chapters 01 and 03 in disguise.)
  (b) The SAME map need not be a contraction in the Euclidean norm, but it IS a
      gamma-contraction in the 2-norm weighted by the stationary distribution of P.
  (c) Cobweb plots: cos(x) is a contraction on [0, 1]; the logistic map with r = 3.3 is
      not (|f'(x*)| > 1) and its iterates never settle on the fixed point.
  (d) As gamma -> 1 the contraction weakens and the linear system becomes ill-conditioned.

Run:  python code/ch00_math_toolkit/contraction_fixed_point.py [--quick]
"""
import argparse
import os
import time

import numpy as np

from plot_style import setup, C, GREY

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def random_stochastic_matrix(n, rng, sparsity=0.5, hub_prob=0.5):
    """Random row-stochastic matrix in which every state also jumps to a 'hub' state 0
    with probability hub_prob. Hubs make ||P||_2 > 1 (a column with large entries),
    which is what lets part (b) show that T is NOT a contraction in the 2-norm."""
    P = rng.random((n, n)) * (rng.random((n, n)) > sparsity)
    P[np.arange(n), rng.integers(0, n, n)] += 0.1  # make sure every row has some mass
    P = P / P.sum(axis=1, keepdims=True)
    P = (1 - hub_prob) * P
    P[:, 0] += hub_prob
    return P


def fixed_point_iteration(T, v0, num_iters):
    vs = [v0]
    for _ in range(num_iters):
        vs.append(T(vs[-1]))
    return np.array(vs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    print(f"[contraction_fixed_point] seed={args.seed} quick={args.quick}")
    t0 = time.time()

    # ------------------------------------------------------------- (a)
    n, gamma = 8, 0.9
    P = random_stochastic_matrix(n, rng)
    r = rng.normal(size=n)
    T = lambda v: r + gamma * P @ v
    v_star = np.linalg.solve(np.eye(n) - gamma * P, r)   # Eq. (4.4), never form the inverse
    print(f"\n(a) T(v) = r + gamma P v, n={n}, gamma={gamma}")
    print("  v* (linear solve)       :", np.round(v_star, 4))
    print("  residual ||T(v*) - v*|| :", f"{np.abs(T(v_star) - v_star).max():.2e}")
    K = 250
    starts = {"v₀ = 0": np.zeros(n), "v₀ = 100·1": 100 * np.ones(n),
              "v₀ ~ N(0, 50²)": rng.normal(scale=50, size=n)}
    traj_err, bounds = {}, {}
    for name, v0 in starts.items():
        vs = fixed_point_iteration(T, v0, K)
        err = np.abs(vs - v_star).max(axis=1)
        traj_err[name] = err
        bounds[name] = gamma ** np.arange(K + 1) * err[0]
        assert np.all(err <= bounds[name] * (1 + 1e-9) + 1e-12), "a-priori bound (4.6) violated?!"
    vs = fixed_point_iteration(T, np.zeros(n), K)
    diffs = np.abs(np.diff(vs, axis=0)).max(axis=1)
    post = gamma / (1 - gamma) * diffs              # a-posteriori bound (4.7) for v_1..v_K
    true_err = np.abs(vs[1:] - v_star).max(axis=1)
    assert (post < 1e-6).any(), "increase K"
    k_stop = int(np.argmax(post < 1e-6)) + 1
    print(f"  a-priori bound holds for all starts and all k <= {K}: True (asserted)")
    print(f"  a-posteriori rule 'stop when gamma/(1-gamma)*||v_k - v_k-1|| < 1e-6' stops at k={k_stop};"
          f" true error there {true_err[k_stop - 1]:.2e}")
    for name, err in traj_err.items():
        print(f"  {name:16s}: error at k=0: {err[0]:8.3f}, k=50: {err[50]:.3e}, k=100: {err[100]:.3e}")

    # ------------------------------------------------------------- (b)
    pairs = 20_000 if not args.quick else 2000
    U = rng.normal(size=(pairs, n)); W = rng.normal(size=(pairs, n))
    D = U - W
    TD = (gamma * (P @ D.T)).T   # T(u) - T(w) = gamma P (u - w): r cancels
    ratio_inf = np.abs(TD).max(axis=1) / np.abs(D).max(axis=1)
    ratio_2 = np.linalg.norm(TD, axis=1) / np.linalg.norm(D, axis=1)
    op2 = gamma * np.linalg.norm(P, 2)
    # the worst direction for the 2-norm is the top right-singular vector of P
    v_worst = np.linalg.svd(P)[2][0]
    worst2 = np.linalg.norm(gamma * P @ v_worst) / np.linalg.norm(v_worst)
    worst_inf = np.abs(gamma * P @ v_worst).max() / np.abs(v_worst).max()
    print(f"\n(b) Lipschitz ratios ||T(u)-T(w)|| / ||u-w|| over {pairs} random pairs")
    print(f"  max-norm : max ratio {ratio_inf.max():.4f}  (theory: <= gamma = {gamma})")
    print(f"  2-norm   : max ratio {ratio_2.max():.4f}  (gamma*||P||_2 = {op2:.4f}); "
          f"fraction of pairs with ratio > 1: {(ratio_2 > 1).mean():.3f}")
    print(f"  along the top singular vector: 2-norm ratio {worst2:.4f}, max-norm ratio {worst_inf:.4f}")
    print(f"  spectral radius of gamma P = {np.abs(np.linalg.eigvals(gamma * P)).max():.4f} (< 1, so iteration still converges)")
    # A 2-norm weighted by the STATIONARY distribution d of P (d^T P = d^T) is different:
    # ||P v||_d^2 = sum_i d_i (sum_j P_ij v_j)^2 <= sum_i d_i sum_j P_ij v_j^2 = ||v||_d^2  (Jensen),
    # so T is a gamma-contraction in ||.||_d. This is why on-policy linear TD converges (Ch. 08).
    A_st = P.T - np.eye(n); A_st[-1, :] = 1.0
    d_st = np.linalg.solve(A_st, np.eye(n)[-1])            # stationary distribution (Sec. 1.6, method iii)
    ratio_d = np.sqrt((d_st * TD ** 2).sum(axis=1) / (d_st * D ** 2).sum(axis=1))
    sq = np.sqrt(d_st)
    op_d = gamma * np.linalg.norm(sq[:, None] * P / sq[None, :], 2)   # exact induced d-norm of gamma P
    print(f"  d-weighted 2-norm (d = stationary distribution of P): max ratio over pairs {ratio_d.max():.4f}, "
          f"exact operator norm {op_d:.4f} (theory: <= gamma = {gamma})")

    # ------------------------------------------------------------- (c)
    cos_traj = fixed_point_iteration(np.cos, 1.0, 40)
    dottie = cos_traj[-1]
    r_log = 3.3
    logistic = lambda x: r_log * x * (1 - x)
    log_traj = fixed_point_iteration(logistic, 0.2, 200)
    x_fp = 1 - 1 / r_log
    print(f"\n(c) cos(x): x_40 = {dottie:.6f}, |cos(x_40) - x_40| = {abs(np.cos(dottie) - dottie):.1e}, "
          f"Lipschitz const on [0,1] = sin(1) = {np.sin(1):.3f}")
    print(f"    logistic r={r_log}: fixed point {x_fp:.4f}, |f'(x*)| = |2 - r| = {abs(2 - r_log):.1f} > 1;"
          f" last iterates {np.round(log_traj[-4:], 4)} (a 2-cycle, not the fixed point)")

    # ------------------------------------------------------------- (d)
    print("\n(d) gamma -> 1: weaker contraction, worse-conditioned linear system")
    print(f"  {'gamma':>6s} {'cond_inf(I-gP)':>15s} {'(1+g)/(1-g)':>14s} {'iters to 1e-8':>14s} "
          f"{'predicted':>10s} {'solve residual':>15s}")
    gamma_rows = []
    for g in [0.5, 0.9, 0.99, 0.999]:
        A = np.eye(n) - g * P
        cond = np.linalg.cond(A, p=np.inf)
        vs_ = np.linalg.solve(A, r)
        v = np.zeros(n); k = 0
        while np.abs(v - vs_).max() > 1e-8 and k < 100_000:
            v = r + g * P @ v; k += 1
        pred = int(np.ceil(np.log(1e-8 / np.abs(vs_).max()) / np.log(g)))
        res = np.abs(A @ vs_ - r).max()
        gamma_rows.append((g, cond, k))
        print(f"  {g:6.3f} {cond:15.1f} {(1 + g) / (1 - g):14.1f} {k:14d} {pred:10d} {res:15.1e}")

    if not args.quick:
        plt = setup()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        ax = axes[0]
        styles = ["-", "--", ":"]
        for (name, err), c, ls in zip(traj_err.items(), C, styles):
            ax.semilogy(err, ls, color=c, label=f"{name}: error")
            ax.semilogy(bounds[name], color=c, lw=0.8, alpha=0.7)
        ax.semilogy(np.arange(1, K + 1), post, "-.", color=C[3], lw=1.5,
                    label=r"a-posteriori bound $\frac{\gamma}{1-\gamma}\|v_k - v_{k-1}\|_\infty$ (v₀=0)")
        ax.plot([], [], color=GREY, lw=0.8, label=r"thin lines: a-priori bound $\gamma^k\|v_0-v_\ast\|_\infty$")
        ax.set_title(r"(a) Iterating $T(v)=r+\gamma Pv$, $\gamma$=0.9")
        ax.set_xlabel("iteration k"); ax.set_ylabel(r"$\|v_k - v_\ast\|_\infty$"); ax.legend(fontsize=8)

        def cobweb(ax, f, traj, xs, title, color, n_show):
            ax.plot(xs, f(xs), color=color, label="y = f(x)")
            ax.plot(xs, xs, color=GREY, lw=1, label="y = x")
            px, py = [traj[0]], [0]
            for x0, x1 in zip(traj[:n_show], traj[1:n_show + 1]):
                px += [x0, x1]; py += [x1, x1]
            ax.plot(px, py, color=C[1], lw=0.9, alpha=0.85, label="iterates (cobweb)")
            ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.legend(loc="upper left")

        xs = np.linspace(0, 1.05, 200)
        cobweb(axes[1], np.cos, cos_traj, xs, "(b) f(x)=cos x: a contraction on [0,1]", C[0], 25)
        axes[1].plot([dottie], [dottie], "o", color=C[2], ms=8, label=f"fixed point {dottie:.4f}")
        axes[1].legend(loc="lower left")
        cobweb(axes[2], logistic, log_traj, xs, f"(c) f(x)={r_log}x(1−x): not a contraction", C[0], 60)
        axes[2].plot([x_fp], [x_fp], "o", color=C[2], ms=8, label=f"repelling fixed point {x_fp:.3f}")
        axes[2].legend(loc="lower center")
        fig.tight_layout()
        path = os.path.join(FIG_DIR, "contraction.png")
        fig.savefig(path)
        print(f"\nsaved {path}")

    print(f"\nSummary: max-norm Lipschitz ratio {ratio_inf.max():.3f} <= gamma; 2-norm ratio up to "
          f"{worst2:.3f}; iterations to 1e-8 grow from {gamma_rows[0][2]} (gamma=0.5) to "
          f"{gamma_rows[-1][2]} (gamma=0.999). runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
