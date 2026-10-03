"""Exact linear-algebra checks of the theory in Chapter 08 (Sections 3.3, 4, 5, 12.3, 15.2).

Everything here is computed exactly from small random Markov reward processes --
no sampling noise except in check 5, which simulates trajectories on purpose.

1. TD fixed point and its error bound (Section 5.3). For random ergodic MRPs with
   linear features and the *on-policy* (stationary) weighting mu:
       VE(w_TD) <= 1/(1-gamma^2) min_w VE(w) <= 1/(1-gamma) min_w VE(w).
   We verify the inequality on many instances and plot the observed ratios.
2. Positive definiteness of A = X^T D (I - gamma P) X (Section 5.2): always PD
   on-policy; with an arbitrary (off-policy) weighting d it can fail, expected TD
   can diverge, and the fixed point can violate the bound. The off-policy ratios
   are reported separately for instances where expected TD converges (all
   eigenvalues of A have non-negative real part) and where it diverges (the
   fixed point is then never reached). We also report how often P is
   non-expansive in the d-norm, ||D^{1/2} P D^{-1/2}||_2 <= 1 (the lemma of
   Section 5.2, which holds for the stationary weighting): it is only a
   *sufficient* condition for positive definiteness.
   NOTE: d ~ Dirichlet(0.1) is an arbitrary skewed weighting used as a stress
   test; these MRPs have no actions, so d need not be the state distribution of
   any behaviour policy.
3. Semi-gradient TD is not a gradient method (Section 3.3): the Jacobian of the
   expected update, -A, is not symmetric, so no objective has it as its gradient.
4. The futility of discounting in continuing tasks (Section 12.3):
       sum_s mu(s) v_gamma(s) = r(pi) / (1 - gamma)   for every gamma.
5. The Bellman error is not learnable (Section 15.2): two MRPs that generate the
   *same* distribution of observable data (features and rewards) have different
   BE minimisers, while their VE minimisers and TD fixed points coincide.
6. The geometry of VE, BE and PBE (Section 15.1): a 3-state MRP with a 2-D
   feature subspace, drawn exactly in mu-orthonormal coordinates (figure
   bellman_geometry.png).

Run:  python code/ch08_function_approximation/linear_td_theory.py [--quick]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread: the machine is shared
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse

import numpy as np
import scipy.linalg as sla

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def stationary(P):
    """Stationary distribution mu with mu^T P = mu^T (left eigenvector for eigenvalue 1)."""
    vals, vecs = np.linalg.eig(P.T)
    mu = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return mu / mu.sum()


def random_mrp(n, rng, k=2, eps=0.05, dense=False):
    """Random ergodic MRP: each state moves to k random successors (sparse, slowly mixing
    chains are where off-policy trouble shows up), mixed with eps of uniform jumps.
    With dense=True every row is a Dirichlet(0.5) distribution over all states instead."""
    if dense:
        return rng.dirichlet(0.5 * np.ones(n), size=n), rng.normal(size=n)
    P = np.zeros((n, n))
    for i in range(n):
        succ = rng.choice(n, size=k, replace=False)
        P[i, succ] = rng.dirichlet(np.ones(k))
    P = (1 - eps) * P + eps / n
    r = rng.normal(size=n)
    return P, r


def td_quantities(P, r, X, d, gamma):
    """Return w_TD, w_proj (min VE) and both VE values, all weighted by d."""
    n = len(r)
    v = np.linalg.solve(np.eye(n) - gamma * P, r)
    Dm = np.diag(d)
    A = X.T @ Dm @ (np.eye(n) - gamma * P) @ X
    b = X.T @ Dm @ r
    w_td = np.linalg.solve(A, b)
    w_proj = np.linalg.solve(X.T @ Dm @ X, X.T @ Dm @ v)
    ve = lambda w: float(d @ (v - X @ w) ** 2)
    return dict(A=A, v=v, w_td=w_td, w_proj=w_proj, ve_td=ve(w_td), ve_min=ve(w_proj))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figures")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dense", action="store_true",
                    help="dense, fast-mixing random chains instead of sparse ones (no figure)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    n, d_feat = 30, 4
    n_inst = 50 if args.quick else 300
    gammas = [0.5, 0.8, 0.9, 0.95, 0.99]
    kind = "dense (Dirichlet(0.5) rows)" if args.dense else "sparse (2 successors + 5% uniform)"
    print(f"seed={args.seed} quick={args.quick}; random {kind} MRPs with {n} states,"
          f" {d_feat} Gaussian features, {n_inst} instances per gamma; off-policy d ~ Dirichlet(0.1)\n")

    # ---------------- 1 & 2: bound and positive definiteness ----------------
    print("1/2. ratio VE(w_TD)/min VE.  on-policy weighting mu vs a random off-policy weighting d")
    print(f"{'gamma':>6s} {'1/(1-g^2)':>10s} {'1/(1-g)':>8s} | {'on: max ratio':>13s} {'viol.':>5s} {'A not PD':>8s}"
          f" | {'off: max ratio (all)':>20s} {'>1/(1-g)':>8s} {'A not PD':>8s} {'Re eig<0':>8s}")
    scatter = {"on": [], "off": [], "off_unstable": []}
    split_rows = []
    for g in gammas:
        on_ratios, off_ratios, off_unst_mask, nonexp, gains = [], [], [], [], []
        on_notpd = off_notpd = 0
        for _ in range(n_inst):
            P, r = random_mrp(n, rng, dense=args.dense)
            X = rng.normal(size=(n, d_feat))
            mu = stationary(P)
            q = td_quantities(P, r, X, mu, g)
            on_ratios.append(q["ve_td"] / q["ve_min"])
            on_notpd += np.linalg.eigvalsh(q["A"] + q["A"].T).min() <= 0
            dd = rng.dirichlet(0.1 * np.ones(n))                  # an arbitrary, skewed weighting
            qo = td_quantities(P, r, X, dd, g)
            off_ratios.append(qo["ve_td"] / qo["ve_min"])
            off_notpd += np.linalg.eigvalsh(qo["A"] + qo["A"].T).min() <= 0
            # expected TD diverges (from almost every start) iff some eigenvalue of A has Re < 0
            off_unst_mask.append(np.linalg.eigvals(qo["A"]).real.min() < 0)
            # the d-norm of P: ||P v||_d <= ||v||_d for all v  iff  ||D^{1/2} P D^{-1/2}||_2 <= 1
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                sq = np.sqrt(dd)
                K = sq[:, None] * P / sq[None, :]
            nonexp.append(np.linalg.norm(K, 2) if np.all(np.isfinite(K)) else np.inf)
            # ... but positive definiteness only needs P to be tame on the feature subspace:
            # gain = max_y ||P X y||_d / ||X y||_d, and y^T A y >= (1 - gamma*gain) ||X y||_d^2.
            G = X.T @ (dd[:, None] * X)
            H = (P @ X).T @ (dd[:, None] * (P @ X))
            gains.append(float(np.sqrt(max(sla.eigh(H, G, eigvals_only=True).max(), 0.0))))
        on_ratios, off_ratios = np.array(on_ratios), np.array(off_ratios)
        unst = np.array(off_unst_mask, dtype=bool)
        nonexp, gains = np.array(nonexp), np.array(gains)
        scatter["on"].append(on_ratios)
        scatter["off"].append(off_ratios)
        scatter["off_unstable"].append(unst)
        viol = int((on_ratios > 1 / (1 - g ** 2) + 1e-9).sum())
        print(f"{g:6.2f} {1 / (1 - g ** 2):10.2f} {1 / (1 - g):8.1f} | {on_ratios.max():13.3f} {viol:5d} {on_notpd:8d}"
              f" | {off_ratios.max():20.3g} {int((off_ratios > 1 / (1 - g)).sum()):8d} {off_notpd:8d} {int(unst.sum()):8d}")
        st = off_ratios[~unst]
        split_rows.append((g, int((~unst).sum()), st.max() if st.size else float("nan"),
                           int((st > 1 / (1 - g)).sum()), int(unst.sum()),
                           off_ratios[unst].max() if unst.any() else float("nan"),
                           int((nonexp > 1).sum()), float(np.median(nonexp)),
                           float(np.median(gains)), int((g * gains < 1).sum())))
    print("\n   off-policy weighting, split by whether expected TD converges (all Re eig(A) >= 0):")
    print(f"{'gamma':>6s} | {'#converge':>9s} {'max ratio':>10s} {'>1/(1-g)':>8s} | {'#diverge':>8s}"
          f" {'max ratio':>10s} | {'||D^.5 P D^-.5||>1':>18s} {'median norm':>11s} | {'median gain':>11s}"
          f" {'gamma*gain<1':>12s}")
    for g, nc, mxc, bc, nd, mxd, ne, med, mg, ng in split_rows:
        print(f"{g:6.2f} | {nc:9d} {mxc:10.3g} {bc:8d} | {nd:8d} {mxd:10.3g} | {ne:14d}/{n_inst:<3d} {med:11.3g}"
              f" | {mg:11.3f} {ng:8d}/{n_inst:<3d}")
    print("   gain = max_y ||P X y||_d / ||X y||_d: the d-norm gain of P on the feature subspace only;"
          " gamma*gain < 1 is sufficient for A to be PD")
    print("   (the 'diverge' fixed points are never reached by expected TD; d is an arbitrary Dirichlet(0.1)"
          " weighting, not necessarily any behaviour policy's state distribution)")

    # ---------------- 3: A is not symmetric ----------------
    P, r = random_mrp(n, rng)
    X = rng.normal(size=(n, d_feat))
    q = td_quantities(P, r, X, stationary(P), 0.9)
    A = q["A"]
    print(f"\n3. Jacobian of expected semi-gradient TD update is -A; ||A - A^T||_F / ||A||_F = "
          f"{np.linalg.norm(A - A.T) / np.linalg.norm(A):.3f} (0 would be needed for a gradient field)")

    # ---------------- 4: futility of discounting ----------------
    mu = stationary(P)
    avg_r = float(mu @ r)
    print(f"\n4. average reward r(pi) = {avg_r:.4f}")
    for g in (0.5, 0.9, 0.99):
        v = np.linalg.solve(np.eye(n) - g * P, r)
        print(f"   gamma={g}: sum_s mu(s) v_gamma(s) = {mu @ v:10.4f}   r(pi)/(1-gamma) = {avg_r / (1 - g):10.4f}")

    # ---------------- 5: the Bellman error is not learnable ----------------
    print("\n5. Two MRPs with identical observable data (Section 15.2)")
    g = 0.9
    # MRP 1: states A, B.  A->A (r=0) | A->B (r=0), each 1/2;  B->A (r=1) | B->B (r=0), each 1/2.
    P1 = np.array([[.5, .5], [.5, .5]])
    r1 = np.array([0.0, 0.5])                        # expected reward from each state
    X1 = np.eye(2)
    # MRP 2: states A, B, B'.  A->A 1/2, A->B 1/4, A->B' 1/4 (r=0);  B->A (r=1);
    #        B'->B 1/2, B'->B' 1/2 (r=0).  B and B' share the feature of B.
    P2 = np.array([[.5, .25, .25], [1, 0, 0], [0, .5, .5]])
    r2 = np.array([0.0, 1.0, 0.0])
    X2 = np.array([[1.0, 0], [0, 1], [0, 1]])

    def solutions(P, r, X):
        mu = stationary(P)
        Dm = np.diag(mu)
        M = (np.eye(len(r)) - g * P) @ X
        w_be = np.linalg.solve(M.T @ Dm @ M, M.T @ Dm @ r)          # argmin ||r - (I - gP) X w||_mu^2
        q = td_quantities(P, r, X, mu, g)
        be = float(mu @ (r - M @ w_be) ** 2)
        return mu, q["w_proj"], q["w_td"], w_be, be

    for name, (P, r, X) in {"MRP1": (P1, r1, X1), "MRP2": (P2, r2, X2)}.items():
        mu, w_ve, w_td, w_be, be = solutions(P, r, X)
        print(f"   {name}: mu = {np.round(mu, 3).tolist()}, argmin VE = {np.round(w_ve, 4).tolist()}, "
              f"w_TD = {np.round(w_td, 4).tolist()}, argmin BE = {np.round(w_be, 4).tolist()} (min BE = {be:.4f})")

    # sampled check that the observable data really have the same distribution
    def observe(P, X, reward_fn, T, rs):
        s, data = 0, []
        for _ in range(T):
            s2 = rs.choice(len(P), p=P[s])
            data.append((int(X[s].argmax()), reward_fn(s, s2), int(X[s2].argmax())))
            s = s2
        return data
    rw1 = lambda s, s2: 1.0 if (s == 1 and s2 == 0) else 0.0
    rw2 = lambda s, s2: 1.0 if (s == 1 and s2 == 0) else 0.0
    T = 20_000 if args.quick else 200_000
    rs = np.random.default_rng(args.seed + 5)
    o1, o2 = observe(P1, X1, rw1, T, rs), observe(P2, X2, rw2, T, rs)
    def pair_freqs(o):
        from collections import Counter
        c = Counter(zip(o[:-1], o[1:]))                # consecutive pairs of (x, r, x') triples
        return {k: v / (len(o) - 1) for k, v in c.items()}
    f1, f2 = pair_freqs(o1), pair_freqs(o2)
    keys = set(f1) | set(f2)
    print(f"   sampled {T} steps from each: max |difference| in frequencies of consecutive "
          f"(x, r, x') pairs = {max(abs(f1.get(k, 0) - f2.get(k, 0)) for k in keys):.4f} over {len(keys)} patterns")

    # ---------------- 6: geometry of VE, BE and PBE ----------------
    geo = geometry_example()

    if args.quick or args.dense:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    gg = np.linspace(0.45, 0.995, 200)
    jit = np.random.default_rng(1)
    for ax, key, title in [(axes[0], "on", "on-policy weighting (stationary $\\mu$)"),
                           (axes[1], "off", "arbitrary (off-policy) weighting $d$")]:
        for k, (g, ratios) in enumerate(zip(gammas, scatter[key])):
            xs = g + jit.normal(scale=0.004, size=ratios.size)
            unst = scatter["off_unstable"][k] if key == "off" else np.zeros(ratios.size, dtype=bool)
            ax.scatter(xs[~unst], ratios[~unst], s=4, alpha=0.4, color="tab:blue",
                       label="expected TD converges" if (k == 0 and key == "off") else None)
            if unst.any():
                ax.scatter(xs[unst], ratios[unst], s=9, alpha=0.8, color="tab:red", marker="x",
                           label="expected TD diverges\n(fixed point never reached)" if k == 0 else None)
        ax.plot(gg, 1 / (1 - gg ** 2), "k-", label=r"$1/(1-\gamma^2)$")
        ax.plot(gg, 1 / (1 - gg), "k--", label=r"$1/(1-\gamma)$")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\overline{VE}(\mathbf{w}_{TD}) \;/\; \min_{\mathbf{w}} \overline{VE}(\mathbf{w})$")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle(f"Linear TD fixed point vs best linear fit ({n_inst} random MRPs per $\\gamma$)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "td_fixed_point_bound.png"), dpi=110)
    plot_geometry(geo, plt)
    print(f"\nfigures saved to {FIG_DIR}")


def geometry_example():
    """Check 6 (Section 15.1): a 3-state MRP, gamma = 0.9, with features
    x(1) = (0, 0), x(2) = (1, 0), x(3) = (0, 1): state 1's estimate is pinned at 0 and
    w = (v_hat(2), v_hat(3)). Everything is exact. Coordinates: y = D^{1/2} v, so that
    the mu-norm becomes the Euclidean norm and the projection Pi becomes orthogonal."""
    g = 0.9
    P = np.array([[0.5, 0.5, 0.0], [0.0, 0.7, 0.3], [0.2, 0.1, 0.7]])
    r = np.array([0.0, 1.0, 3.0])
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    mu = stationary(P)
    Dm, I3 = np.diag(mu), np.eye(3)
    v = np.linalg.solve(I3 - g * P, r)
    Pi = X @ np.linalg.solve(X.T @ Dm @ X, X.T @ Dm)
    T = lambda u: r + g * P @ u                                  # Bellman operator
    w_ve = np.linalg.solve(X.T @ Dm @ X, X.T @ Dm @ v)           # Pi v_pi
    w_td = np.linalg.solve(X.T @ Dm @ (I3 - g * P) @ X, X.T @ Dm @ r)
    M = (I3 - g * P) @ X
    w_be = np.linalg.solve(M.T @ Dm @ M, M.T @ Dm @ r)           # argmin ||T v_w - v_w||_mu
    ve = lambda w: float(mu @ (v - X @ w) ** 2)
    be = lambda w: float(mu @ (T(X @ w) - X @ w) ** 2)
    pbe = lambda w: float(mu @ (Pi @ (T(X @ w) - X @ w)) ** 2)
    w_ex = np.array([14.0, 6.0])                                 # an arbitrary w for the arrows
    print("\n6. Geometry example (3 states, gamma=0.9, x(1)=0, x(2)=e1, x(3)=e2):"
          f" mu = {np.round(mu, 4).tolist()}, v_pi = {np.round(v, 3).tolist()}")
    for name, w in [("argmin VE (Pi v_pi)", w_ve), ("w_TD (PBE = 0)", w_td), ("argmin BE", w_be),
                    ("example w", w_ex)]:
        print(f"   {name:20s} w = {np.round(w, 3).tolist()!s:18s} sqrtVE = {np.sqrt(ve(w)):6.3f}"
              f"  sqrtBE = {np.sqrt(be(w)):6.3f}  sqrtPBE = {np.sqrt(pbe(w)):6.3f}")
    return dict(P=P, r=r, X=X, mu=mu, v=v, Pi=Pi, T=T, g=g, w_ve=w_ve, w_td=w_td, w_be=w_be,
                w_ex=w_ex, ve=ve, be=be, pbe=pbe)


def plot_geometry(G, plt):
    mu, X, v, P, r, g, Pi, T = (G[k] for k in ("mu", "X", "v", "P", "r", "g", "Pi", "T"))
    Dh = np.sqrt(mu)
    Q, R = np.linalg.qr(Dh[:, None] * X)                         # orthonormal basis of D^{1/2} X
    S = np.diag(np.sign(np.diag(R)))
    Q, R = Q @ S, S @ R
    nrm = np.cross(Q[:, 0], Q[:, 1])                             # unit normal to the plane
    if nrm @ (Dh * v) < 0:
        nrm = -nrm
    co = lambda u: np.array([Q[:, 0] @ (Dh * u), Q[:, 1] @ (Dh * u), nrm @ (Dh * u)])

    fig = plt.figure(figsize=(14, 5.8))
    # left: inside the plane, contours of the three objectives
    ax = fig.add_subplot(1, 2, 1)
    p_ve, p_td, p_be, p_ex = (R @ G[k] for k in ("w_ve", "w_td", "w_be", "w_ex"))
    pts = np.array([p_ve, p_td, p_be, p_ex])
    lo, hi = pts.min(0) - 1.0, pts.max(0) + 1.0
    Y1, Y2 = np.meshgrid(np.linspace(lo[0], hi[0], 300), np.linspace(lo[1], hi[1], 300))
    W = np.linalg.solve(R, np.stack([Y1.ravel(), Y2.ravel()])).T
    V = W @ X.T
    resid = r + g * V @ P.T - V                                  # Bellman error vectors (rows)
    Z = {"VE": np.sqrt(((v - V) ** 2) @ mu), "PBE": np.sqrt(((resid @ Pi.T) ** 2) @ mu),
         "BE": np.sqrt((resid ** 2) @ mu)}
    mins = {"VE": np.sqrt(G["ve"](G["w_ve"])), "PBE": 0.0, "BE": np.sqrt(G["be"](G["w_be"]))}
    for key, c in [("VE", "tab:green"), ("PBE", "tab:orange"), ("BE", "tab:red")]:
        cs = ax.contour(Y1, Y2, Z[key].reshape(Y1.shape), levels=mins[key] + np.array([0.3, 0.7, 1.2]),
                        colors=c, linewidths=1.1)
        ax.clabel(cs, fontsize=7, fmt="%.2f")
        ax.plot([], [], color=c, label=rf"$\sqrt{{\overline{{{key}}}}}$ contours")
    ax.plot(*p_ve, "o", color="tab:green", ms=9, label=r"argmin $\overline{VE}$ $=\Pi v_\pi$ (gradient MC)")
    ax.plot(*p_td, "s", color="tab:orange", ms=9, label=r"$\mathbf{w}_{TD}$: $\overline{PBE}=0$ (TD, LSTD, GTD2, TDC)")
    ax.plot(*p_be, "^", color="tab:red", ms=9, label=r"argmin $\overline{BE}$ (residual gradient)")
    e_in, pt_in = co(X @ G["w_ex"]), co(Pi @ T(X @ G["w_ex"]))
    ax.plot(*e_in[:2], "o", color="tab:purple", ms=6, label=r"example $v_{\mathbf{w}}$ and its PBE vector")
    ax.annotate("", xy=pt_in[:2], xytext=e_in[:2],
                arrowprops=dict(arrowstyle="->", color="tab:brown", lw=1.8))
    ax.set_aspect("equal")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_xlabel(r"$\sqrt{\mu(2)}\,w_1$  ($\mu$-orthonormal coordinate)")
    ax.set_ylabel(r"$\sqrt{\mu(3)}\,w_2$")
    ax.set_title("Inside the plane of representable value functions")
    # right: the plane seen in 3-D, with v_pi and T v_w above it
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    vw, Tvw, PTvw = co(X @ G["w_ex"]), co(T(X @ G["w_ex"])), co(Pi @ T(X @ G["w_ex"]))
    P3 = [(co(v), "k", r"$v_\pi$"), (co(X @ G["w_ve"]), "tab:green", r"$\Pi v_\pi$"),
          (co(X @ G["w_td"]), "tab:orange", r"$\mathbf{X}\mathbf{w}_{TD}$"),
          (co(X @ G["w_be"]), "tab:red", r"min $\overline{BE}$"),
          (vw, "tab:purple", r"$v_{\mathbf{w}}$"), (Tvw, "tab:purple", r"$\mathcal{T}^\pi v_{\mathbf{w}}$"),
          (PTvw, "tab:brown", r"$\Pi\mathcal{T}^\pi v_{\mathbf{w}}$")]
    allp = np.array([p for p, _, _ in P3])
    lo3, hi3 = allp[:, :2].min(0) - 0.5, allp[:, :2].max(0) + 0.5
    xx, yy = np.meshgrid(np.linspace(lo3[0], hi3[0], 2), np.linspace(lo3[1], hi3[1], 2))
    ax3.plot_surface(xx, yy, 0 * xx, alpha=0.18, color="tab:blue")
    for p, c, lab in P3:
        ax3.scatter(*p, color=c, s=40, depthshade=False)
        ax3.text(p[0], p[1], p[2] + 0.1, lab, fontsize=9, color=c)
    ax3.plot(*zip(co(v), co(X @ G["w_ve"])), "k:", lw=1)
    ax3.quiver(*vw, *(Tvw - vw), color="tab:purple", arrow_length_ratio=0.06, lw=2)
    ax3.quiver(*vw, *(PTvw - vw), color="tab:brown", arrow_length_ratio=0.06, lw=2)
    ax3.plot(*zip(Tvw, PTvw), ":", color="gray")
    ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
    ax3.set_zlim(0, allp[:, 2].max() * 1.1)
    ax3.view_init(elev=20, azim=-55)
    ax3.set_title("BE vector $\\mathcal{T}^\\pi v_{\\mathbf{w}} - v_{\\mathbf{w}}$ (purple) and its projection,\n"
                  "the PBE vector (brown); the plane is $\\{\\mathbf{X}\\mathbf{w}\\}$", fontsize=10)
    fig.suptitle(r"Geometry of linear value functions: 3-state MRP, 2 features, $\gamma = 0.9$ (exact)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "bellman_geometry.png"), dpi=100)


if __name__ == "__main__":
    main()
