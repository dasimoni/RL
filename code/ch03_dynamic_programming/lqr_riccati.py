"""Dynamic programming with continuous states: LQR, Riccati equations and iLQR (Sections 11.4-11.5).

The one continuous-state problem that DP solves exactly is the linear-quadratic regulator:
    S_{t+1} = F S_t + G A_t + W_t,      R_{t+1} = -(S_t' C_s S_t + A_t' C_a A_t),
with W_t zero-mean noise of covariance Sigma_w. Backward induction (Algorithm 3.6) with the guess
V_t(s) = -s' P_t s - c_t gives the discounted Riccati recursion (3.29)-(3.30). This script checks:

  A. the finite-horizon Riccati recursion against Monte Carlo, certainty equivalence (the gains do
     not depend on Sigma_w), convergence of P_t to the discounted DARE, and the scipy scaling trick
     solve_discrete_are(sqrt(gamma) F, sqrt(gamma) G, C_s, C_a);
  B. the scalar problem of Chapter 12, Section 2.6 (F = G = c_s = c_a = 1, gamma = 0.9): P, K, theta*;
  C. policy iteration on LQR (Hewer's algorithm: Lyapunov evaluation + greedy gain) against value
     iteration (the Riccati map iterated from P = 0), and Hewer = Newton's method on the DARE;
  D. tabular value iteration on a discretised state grid (1-D and 2-D) against the exact -s'Ps:
     the curse of dimensionality in numbers;
  E. iLQR (Li & Todorov, 2004; Tassa et al., 2012): an exactness check on an LQ problem, and a
     torque-limited pendulum swing-up with Gymnasium's Pendulum-v1 dynamics, followed by an
     LQR "catch" at the top, executed in the real environment.

Run:  python code/ch03_dynamic_programming/lqr_riccati.py [--quick]
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

SEED = 0


# ----------------------------------------------------------------------------------------------
# LQR building blocks.  Rewards are -(s'C_s s + a'C_a a); values are V(s) = -s'Ps - c.
# ----------------------------------------------------------------------------------------------
def greedy_gain(P, F, G, Ca, gamma):
    """K such that a = -K s maximises -a'C_a a + gamma * E[-(Fs+Ga+W)' P (Fs+Ga+W)]  (eq. 3.30)."""
    return gamma * np.linalg.solve(Ca + gamma * G.T @ P @ G, G.T @ P @ F)


def riccati_step(P, F, G, Cs, Ca, gamma):
    """One backward-induction step P_{t+1} -> P_t (eq. 3.29), i.e. one value-iteration sweep."""
    K = greedy_gain(P, F, G, Ca, gamma)
    Pn = Cs + gamma * F.T @ P @ F - gamma * F.T @ P @ G @ K
    return 0.5 * (Pn + Pn.T), K


def finite_horizon_lqr(F, G, Cs, Ca, gamma, H, Sigma_w, P_H=None):
    """Backward induction (Algorithm 3.6) for LQR: returns P_t, c_t (t = 0..H) and K_t (t = 0..H-1)."""
    d = F.shape[0]
    P = np.zeros((H + 1, d, d))
    c = np.zeros(H + 1)
    K = np.zeros((H, G.shape[1], d))
    P[H] = np.zeros((d, d)) if P_H is None else P_H
    for t in range(H - 1, -1, -1):
        P[t], K[t] = riccati_step(P[t + 1], F, G, Cs, Ca, gamma)
        c[t] = gamma * (c[t + 1] + np.trace(P[t + 1] @ Sigma_w))
    return P, c, K


def dare_discounted(F, G, Cs, Ca, gamma):
    """Stationary P of the discounted problem via scipy's (undiscounted) DARE solver."""
    sg = np.sqrt(gamma)
    P = solve_discrete_are(sg * F, sg * G, Cs, Ca)
    return 0.5 * (P + P.T)


def evaluate_gain(K, F, G, Cs, Ca, gamma):
    """Policy evaluation for a = -K s: the Lyapunov equation P = C_s + K'C_aK + gamma (F-GK)'P(F-GK).
    Returns None if the discounted closed loop sqrt(gamma)(F - GK) is not stable (infinite cost)."""
    Acl = np.sqrt(gamma) * (F - G @ K)
    if np.max(np.abs(np.linalg.eigvals(Acl))) >= 1:
        return None
    P = solve_discrete_lyapunov(Acl.T, Cs + K.T @ Ca @ K)   # solves X = a X a' + q with a = Acl'
    return 0.5 * (P + P.T)


def hewer(K0, F, G, Cs, Ca, gamma, P_ref, tol=1e-10, max_iter=100):
    """Policy iteration for LQR (Hewer, 1971). Returns relative errors ||P_k - P*||/||P*|| per evaluation."""
    K, errs = K0, []
    for _ in range(max_iter):
        P = evaluate_gain(K, F, G, Cs, Ca, gamma)
        errs.append(np.linalg.norm(P - P_ref) / np.linalg.norm(P_ref))
        if errs[-1] < tol:
            break
        K = greedy_gain(P, F, G, Ca, gamma)
    return errs


def riccati_vi(F, G, Cs, Ca, gamma, P_ref, tol=1e-10, max_iter=1_000_000):
    """Value iteration on LQR: iterate the Riccati map from P = 0 (= backward induction with H -> inf)."""
    P, errs = np.zeros_like(Cs), []
    nref = np.linalg.norm(P_ref)
    for _ in range(max_iter):
        P, _ = riccati_step(P, F, G, Cs, Ca, gamma)
        errs.append(np.linalg.norm(P - P_ref) / nref)
        if errs[-1] < tol:
            break
    return errs


def random_system(rng, d, m, rho):
    """Random (F, G) with spectral radius of F equal to rho."""
    F = rng.normal(size=(d, d))
    F *= rho / np.max(np.abs(np.linalg.eigvals(F)))
    G = rng.normal(size=(d, m)) / np.sqrt(d)
    return F, G


# ----------------------------------------------------------------------------------------------
# Part D: tabular value iteration on a grid with multilinear interpolation.
# ----------------------------------------------------------------------------------------------
def grid_value_iteration(F, G, Cs, Ca, gamma, L, ell, n_act, A_max, tol=1e-9):
    """Discretise [-L, L]^d with ell points per axis and the scalar action into n_act values in
    [-A_max, A_max]. Successor states are clipped to the box and spread over the 2^d surrounding grid
    points by multilinear interpolation, which gives a finite MDP with a stochastic-looking (but
    deterministic-dynamics) transition table. Then run synchronous VI until ||V_{k+1}-V_k|| < tol(1-g)/g.
    """
    d = F.shape[0]
    axis = np.linspace(-L, L, ell)
    h = axis[1] - axis[0]
    mesh = np.stack(np.meshgrid(*([axis] * d), indexing="ij"), -1).reshape(-1, d)
    n = mesh.shape[0]
    acts = np.linspace(-A_max, A_max, n_act)
    S = np.repeat(mesh, n_act, axis=0)
    Aa = np.tile(acts, n)[:, None]
    S2 = np.clip(S @ F.T + Aa @ G.T, -L, L)
    R = -(np.einsum("ij,jk,ik->i", S, Cs, S) + Ca[0, 0] * Aa[:, 0] ** 2)
    idx = np.minimum(((S2 + L) / h).astype(np.int64), ell - 2)
    w = (S2 - axis[idx]) / h
    m = S.shape[0]
    rows, cols, vals = [], [], []
    for corner in range(2 ** d):
        flat = np.zeros(m, dtype=np.int64)
        wt = np.ones(m)
        for k in range(d):
            bit = (corner >> k) & 1
            flat = flat * ell + idx[:, k] + bit
            wt = wt * (w[:, k] if bit else 1 - w[:, k])
        rows.append(np.arange(m)); cols.append(flat); vals.append(wt)
    P = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m, n))
    P.eliminate_zeros()
    V = np.zeros(n)
    t0 = time.perf_counter()
    sweeps = 0
    while True:
        Q = (R + gamma * (P @ V)).reshape(n, n_act)
        Vn = Q.max(axis=1)
        sweeps += 1
        done = np.max(np.abs(Vn - V)) < tol * (1 - gamma) / gamma
        V = Vn
        if done:
            break
    elapsed = time.perf_counter() - t0
    return mesh, V, acts[Q.argmax(axis=1)], sweeps, elapsed, P.nnz


# ----------------------------------------------------------------------------------------------
# Part E: iLQR for reward maximisation (Algorithm 3.7).
# ----------------------------------------------------------------------------------------------
def ilqr(f, derivs, terminal, s0, U0, gamma=1.0, lo=None, hi=None, max_iter=300, tol=1e-6, mu0=1.0):
    """Iterative LQR with Levenberg-Marquardt-style regularisation and a backtracking line search.

    f(s, a) -> next state; derivs(s, a) -> (r, r_s, r_a, r_ss, r_aa, r_as, f_s, f_a);
    terminal(s) -> (h, h_s, h_ss). Hessians of r must be negative semi-definite (we maximise).
    Action bounds [lo, hi] are handled by clamping k_t and zeroing the clamped rows of K_t (exact for a
    scalar action; Tassa, Mansard & Todorov, 2014, solve a small box QP in general).
    Policy at iteration end: a_t = abar_t + alpha k_t - K_t (s_t - sbar_t).
    Returns S, U, k, K, J, history of J, number of iterations.
    """
    def clip(u):
        return u if lo is None else np.clip(u, lo, hi)

    def rollout(U_new, S_old=None, U_old=None, k=None, K=None, alpha=0.0):
        S_new = np.zeros((H + 1, n)); S_new[0] = s0
        U_out = np.zeros_like(U_new); J_new = 0.0
        for t in range(H):
            if k is None:
                U_out[t] = clip(U_new[t])
            else:
                U_out[t] = clip(U_old[t] + alpha * k[t] - K[t] @ (S_new[t] - S_old[t]))
            J_new += gamma ** t * derivs(S_new[t], U_out[t])[0]
            S_new[t + 1] = f(S_new[t], U_out[t])
        return S_new, U_out, J_new + gamma ** H * terminal(S_new[H])[0]

    H, n, m = len(U0), len(s0), U0.shape[1]
    S, U, J = rollout(U0)
    mu, Delta, MU_MIN, D0 = mu0, 1.0, 1e-6, 2.0          # regularisation schedule of Tassa et al. (2012)
    hist = [J]
    it = 0
    for it in range(1, max_iter + 1):
        D = [derivs(S[t], U[t]) for t in range(H)]
        while True:                                       # backward pass, increasing mu until Q_aa < 0
            _, Vs, Vss = terminal(S[H])
            k = np.zeros((H, m)); K = np.zeros((H, m, n)); dV1 = dV2 = 0.0; ok = True
            for t in range(H - 1, -1, -1):
                _, rs, ra, rss, raa, ras, fs, fa = D[t]
                Qs = rs + gamma * fs.T @ Vs
                Qa = ra + gamma * fa.T @ Vs
                Qss = rss + gamma * fs.T @ Vss @ fs
                Qaa = raa + gamma * fa.T @ Vss @ fa
                Qas = ras + gamma * fa.T @ Vss @ fs
                Vreg = Vss - mu * np.eye(n)               # regularise the value Hessian (Tassa et al.)
                Qaa_r = raa + gamma * fa.T @ Vreg @ fa
                Qas_r = ras + gamma * fa.T @ Vreg @ fs
                try:
                    np.linalg.cholesky(-Qaa_r)
                except np.linalg.LinAlgError:
                    ok = False
                    break
                kt = -np.linalg.solve(Qaa_r, Qa)
                Kt = np.linalg.solve(Qaa_r, Qas_r)        # delta a = k - K delta s
                if lo is not None:
                    unc = U[t] + kt
                    cl = clip(unc)
                    Kt[cl != unc, :] = 0.0
                    kt = cl - U[t]
                Vs = Qs - Kt.T @ Qa - Kt.T @ Qaa @ kt + Qas.T @ kt
                Vss = Qss + Kt.T @ Qaa @ Kt - Kt.T @ Qas - Qas.T @ Kt
                Vss = 0.5 * (Vss + Vss.T)
                dV1 += Qa @ kt; dV2 += 0.5 * kt @ Qaa @ kt
                k[t], K[t] = kt, Kt
            if ok:
                break
            Delta = max(D0, Delta * D0); mu = max(MU_MIN, mu * Delta)
        accepted = False
        for alpha in 0.5 ** np.arange(11):                # forward pass with backtracking line search
            Sn, Un, Jn = rollout(U, S, U, k, K, alpha)
            expected = alpha * dV1 + alpha ** 2 * dV2
            if expected > 0 and (Jn - J) / expected > 0.1:
                accepted = True
                break
        if accepted:
            dJ = Jn - J
            S, U, J = Sn, Un, Jn
            hist.append(J)
            Delta = min(1 / D0, Delta / D0); mu = mu * Delta if mu * Delta > MU_MIN else 0.0
            if dJ < tol * abs(J):
                break
        else:
            if dV1 < 1e-10 * max(1.0, abs(J)):            # no predicted improvement: a stationary point
                break
            Delta = max(D0, Delta * D0); mu = max(MU_MIN, mu * Delta)
            if mu > 1e10:
                break
    return S, U, k, K, J, hist, it


# Pendulum-v1 dynamics (Gymnasium): theta = 0 upright, dt = 0.05, g = 10, m = l = 1, |torque| <= 2.
DT, GRAV, AMAX = 0.05, 10.0, 2.0


def pend_f(s, a):
    th, om = s
    om2 = om + (1.5 * GRAV * np.sin(th) + 3.0 * a[0]) * DT
    return np.array([th + om2 * DT, om2])


def pend_derivs(s, a):
    """Smooth stand-in for Pendulum-v1's reward -(theta^2 + 0.1 omega^2 + 0.001 a^2): we use
    2(1 - cos theta) = 4 sin^2(theta/2), which equals theta^2 to second order at the top and is periodic.
    Its Hessian is replaced by the Gauss-Newton term -2 cos^2(theta/2) <= 0 (the residual is 2 sin(theta/2))."""
    th, om = s
    fs = np.array([[1 + 1.5 * GRAV * DT * DT * np.cos(th), DT], [1.5 * GRAV * DT * np.cos(th), 1.0]])
    fa = np.array([[3 * DT * DT], [3 * DT]])
    r = -(2 * (1 - np.cos(th)) + 0.1 * om * om + 0.001 * a[0] ** 2)
    rs = np.array([-2 * np.sin(th), -0.2 * om])
    rss = np.diag([-2 * np.cos(th / 2) ** 2, -0.2])
    ra = np.array([-0.002 * a[0]])
    raa = np.array([[-0.002]])
    return r, rs, ra, rss, raa, np.zeros((1, 2)), fs, fa


def pend_terminal(s):
    th, om = s
    return (-(2 * (1 - np.cos(th)) + 0.1 * om * om), np.array([-2 * np.sin(th), -0.2 * om]),
            np.diag([-2 * np.cos(th / 2) ** 2, -0.2]))


def lq_problem(F, G, Cs, Ca):
    """Derivative oracle for a linear-quadratic problem (used to check iLQR against Riccati)."""
    def f(s, a):
        return F @ s + G @ a

    def derivs(s, a):
        r = -(s @ Cs @ s + a @ Ca @ a)
        return r, -2 * Cs @ s, -2 * Ca @ a, -2 * Cs, -2 * Ca, np.zeros((G.shape[1], F.shape[0])), F, G

    def terminal(s):
        d = len(s)
        return 0.0, np.zeros(d), np.zeros((d, d))
    return f, derivs, terminal


def wrap(th):
    return (th + np.pi) % (2 * np.pi) - np.pi


# ----------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    quick = args.quick
    rng = np.random.default_rng(SEED)
    n_mc = 2000 if quick else 20000
    print(f"lqr_riccati.py  seed={SEED}  quick={quick}  MC trajectories={n_mc}")
    t_start = time.perf_counter()
    figdata = {}

    # ------------------------------------------------------------------ Part A
    print("\n=== A. Finite-horizon Riccati recursion (backward induction with V_t(s) = -s'P_t s - c_t) ===")
    d, m, gamma, H = 4, 2, 0.95, 50
    F, G = random_system(rng, d, m, rho=1.1)
    Cs, Ca = np.eye(d), 0.5 * np.eye(m)
    Sigma_w = 0.1 * np.eye(d)
    P, c, K = finite_horizon_lqr(F, G, Cs, Ca, gamma, H, Sigma_w)
    _, _, K0 = finite_horizon_lqr(F, G, Cs, Ca, gamma, H, np.zeros((d, d)))
    print(f"random system: d_s={d}, d_a={m}, rho(F)=1.1 (unstable), gamma={gamma}, H={H}, Sigma_w=0.1 I")
    print(f"certainty equivalence: max |K_t(Sigma_w) - K_t(0)| over t = {np.max(np.abs(K - K0)):.1e}")
    s0 = np.ones(d)
    V0 = -s0 @ P[0] @ s0 - c[0]
    Lw = np.linalg.cholesky(Sigma_w)
    Sx = np.tile(s0, (n_mc, 1))
    G_ret = np.zeros(n_mc)
    for t in range(H):
        A_ = -Sx @ K[t].T
        G_ret += gamma ** t * -(np.einsum("ij,jk,ik->i", Sx, Cs, Sx) + np.einsum("ij,jk,ik->i", A_, Ca, A_))
        Sx = Sx @ F.T + A_ @ G.T + rng.normal(size=(n_mc, d)) @ Lw.T
    se = G_ret.std(ddof=1) / np.sqrt(n_mc)
    print(f"V_0(s0) = -s0'P_0 s0 - c_0 = {V0:.4f}  (quadratic part {-s0 @ P[0] @ s0:.4f}, noise part {-c[0]:.4f});"
          f"  Monte Carlo of the K_t policy: {G_ret.mean():.4f} +- {se:.4f}  ({(G_ret.mean() - V0) / se:+.2f} SE)")
    P_dare = dare_discounted(F, G, Cs, Ca, gamma)
    rel = [np.linalg.norm(P[t] - P_dare) / np.linalg.norm(P_dare) for t in range(H + 1)]
    Pv, steps = np.zeros((d, d)), 0
    while np.linalg.norm(Pv - P_dare) / np.linalg.norm(P_dare) >= 1e-10:
        Pv, _ = riccati_step(Pv, F, G, Cs, Ca, gamma)
        steps += 1
    K_dare = greedy_gain(P_dare, F, G, Ca, gamma)
    rho_cl = np.max(np.abs(np.linalg.eigvals(F - G @ K_dare)))
    print(f"||P_t - P_DARE||/||P_DARE|| with 1, 10, 50 steps to go: {rel[H - 1]:.2e}, {rel[H - 10]:.2e}, {rel[0]:.2e};"
          f" below 1e-10 after {steps} steps.  Asymptotic rate gamma*rho(F-GK)^2 = {gamma * rho_cl ** 2:.3f}")
    P_wrong = solve_discrete_are(F, G, Cs, Ca)
    print(f"scipy solve_discrete_are(sqrt(g)F, sqrt(g)G, C_s, C_a) vs the limit of the recursion: "
          f"{np.max(np.abs(P_dare - Pv)):.1e}.  Without the sqrt(g) scaling (undiscounted P): "
          f"relative difference {np.linalg.norm(P_wrong - P_dare) / np.linalg.norm(P_dare):.2f}")
    c_stat = gamma * np.trace(P_dare @ Sigma_w) / (1 - gamma)
    print(f"stationary noise constant c = gamma tr(P Sigma_w)/(1-gamma) = {c_stat:.4f}; c_0 for H={H}: {c[0]:.4f}")

    # ------------------------------------------------------------------ Part B
    print("\n=== B. The scalar problem of Chapter 12, Section 2.6 (F = G = c_s = c_a = 1, gamma = 0.9) ===")
    g = 0.9
    P_s = (0.8 + np.sqrt(0.64 + 3.6)) / 1.8                 # positive root of 0.9 P^2 - 0.8 P - 1 = 0
    P_scipy = dare_discounted(np.eye(1), np.eye(1), np.eye(1), np.eye(1), g)[0, 0]
    K_s = g * P_s / (1 + g * P_s)

    def J12(theta):                                         # Chapter 12's closed form, E[S_0^2] = 1
        return -(1 + theta ** 2) / (1 - 0.9 * (1 + theta) ** 2)
    opt = minimize_scalar(lambda th: -J12(th), bounds=(-1.5, -0.1), method="bounded",
                          options={"xatol": 1e-12})
    print(f"DARE by hand: P = {P_s:.6f}; scipy: {P_scipy:.6f}.  K = gamma P/(1 + gamma P) = {K_s:.6f}, "
          f"so theta* = -K = {-K_s:.6f} and J(theta*) = -P = {-P_s:.6f}")
    print(f"maximising Chapter 12's J(theta) numerically: theta* = {opt.x:.6f}, J = {J12(opt.x):.6f}; "
          f"J(-0.5) = {J12(-0.5):.6f}")
    print(f"with process noise sigma_w = 0.3: J(theta*) = -P - gamma P sigma_w^2/(1-gamma) = "
          f"{-P_s - g * P_s * 0.09 / (1 - g):.4f}")

    # ------------------------------------------------------------------ Part C
    print("\n=== C. Policy iteration (Hewer) vs value iteration (Riccati map from P = 0) ===")
    # Scalar: Hewer's iterates coincide with Newton's method on R(P) = T*(P) - P.
    Pk, Kk = None, 0.0
    hew, newt = [], []
    for _ in range(5):
        Pk = (1 + Kk ** 2) / (1 - g * (1 - Kk) ** 2)        # evaluate a = -K s
        hew.append(Pk)
        Kk = g * Pk / (1 + g * Pk)                           # greedy gain
    Pn = hew[0]
    newt.append(Pn)
    for _ in range(4):
        R = 1 + g * Pn - g ** 2 * Pn ** 2 / (1 + g * Pn) - Pn
        dR = g / (1 + g * Pn) ** 2 - 1                       # R'(P) = gamma (F - G K(P))^2 - 1
        Pn = Pn - R / dR
        newt.append(Pn)
    print("scalar, K_0 = 0:  Hewer P_k = " + ", ".join(f"{x:.6f}" for x in hew))
    print("                  Newton P_k = " + ", ".join(f"{x:.6f}" for x in newt)
          + f"   (max difference {np.max(np.abs(np.array(hew) - np.array(newt))):.1e})")
    gamma_c, rho_F = 0.95, 1.02
    dims = [2, 8] if quick else [2, 4, 8, 16, 32, 64]
    print(f"random systems: rho(F) = {rho_F} (unstable), d_a = d_s/2, C_s = I, C_a = I, gamma = {gamma_c}; "
          f"K_0 = 0 is admissible because sqrt(gamma) rho(F) = {np.sqrt(gamma_c) * rho_F:.3f} < 1")
    print(f"{'d_s':>4} {'PI evaluations':>15} {'PI time':>9} {'VI sweeps':>10} {'VI time':>9}  "
          f"{'gamma*rho_cl^2':>14}  PI relative errors")
    rows_c = []
    for dd in dims:
        mm = max(1, dd // 2)
        Fc, Gc = random_system(rng, dd, mm, rho_F)
        Csc, Cac = np.eye(dd), np.eye(mm)
        Pref = dare_discounted(Fc, Gc, Csc, Cac, gamma_c)
        t0 = time.perf_counter()
        e_pi = hewer(np.zeros((mm, dd)), Fc, Gc, Csc, Cac, gamma_c, Pref)
        t_pi = time.perf_counter() - t0
        t0 = time.perf_counter()
        e_vi = riccati_vi(Fc, Gc, Csc, Cac, gamma_c, Pref)
        t_vi = time.perf_counter() - t0
        rcl = np.max(np.abs(np.linalg.eigvals(Fc - Gc @ greedy_gain(Pref, Fc, Gc, Cac, gamma_c))))
        rows_c.append((dd, e_pi, e_vi))
        print(f"{dd:>4} {len(e_pi):>15} {t_pi * 1e3:>7.1f}ms {len(e_vi):>10} {t_vi * 1e3:>7.1f}ms  "
              f"{gamma_c * rcl ** 2:>14.3f}  " + " ".join(f"{e:.0e}" for e in e_pi))
    figdata["hewer"] = rows_c

    # ------------------------------------------------------------------ Part D
    print("\n=== D. Tabular VI on a state grid vs the exact quadratic value (curse of dimensionality) ===")
    one_d = []
    print("1-D: the Chapter 12 problem (gamma = 0.9), grid on [-2, 2], actions on [-2, 2] with the same spacing;"
          " errors on |s| <= 1")
    print(f"{'points':>7} {'sweeps':>7} {'time':>8} {'max |V + P s^2|':>16} {'rel.':>9} {'max |a - (-K s)|':>17}")
    for ell in ([21, 41, 81] if quick else [21, 41, 81, 161, 321, 641]):
        mesh, V, a, sw, el, nnz = grid_value_iteration(np.eye(1), np.eye(1), np.eye(1), np.eye(1), 0.9,
                                                       2.0, ell, ell, 2.0)
        inner = np.abs(mesh[:, 0]) <= 1 + 1e-12
        err = np.max(np.abs(V[inner] + P_s * mesh[inner, 0] ** 2))
        aerr = np.max(np.abs(a[inner] + K_s * mesh[inner, 0]))
        one_d.append((ell, err / P_s))
        print(f"{ell:>7} {sw:>7} {el * 1e3:>6.1f}ms {err:>16.2e} {err / P_s:>9.2e} {aerr:>17.3f}")
    dt2 = 0.5
    F2 = np.array([[1, dt2], [0, 1.0]]); G2 = np.array([[dt2 ** 2 / 2], [dt2]])
    Cs2, Ca2, g2 = np.eye(2), np.eye(1), 0.95
    P2 = dare_discounted(F2, G2, Cs2, Ca2, g2)
    K2 = greedy_gain(P2, F2, G2, Ca2, g2)
    t0 = time.perf_counter()
    for _ in range(100):
        dare_discounted(F2, G2, Cs2, Ca2, g2)
    t_dare = (time.perf_counter() - t0) / 100
    print(f"2-D: double integrator (dt = {dt2}), C_s = I, C_a = 1, gamma = {g2}; grid on [-3, 3]^2, "
          f"actions on [-3, 3]; errors on [-1, 1]^2.  Riccati: P = {np.round(P2, 3).tolist()}, "
          f"K = {np.round(K2, 3).tolist()}, scipy DARE {t_dare * 1e3:.2f} ms")
    print(f"{'points/axis':>11} {'states':>7} {'actions':>8} {'table nnz':>10} {'sweeps':>7} {'time':>8} "
          f"{'rel. value error':>17} {'max action error':>17}")
    two_d = []
    for ell in ([21, 41] if quick else [21, 41, 81, 161]):
        na = min(ell, 81)
        mesh, V, a, sw, el, nnz = grid_value_iteration(F2, G2, Cs2, Ca2, g2, 3.0, ell, na, 3.0)
        inner = np.all(np.abs(mesh) <= 1 + 1e-12, axis=1)
        Vt = -np.einsum("ij,jk,ik->i", mesh, P2, mesh)
        rel_err = np.max(np.abs(V[inner] - Vt[inner])) / np.max(np.abs(Vt[inner]))
        aerr = np.max(np.abs(a[inner] + mesh[inner] @ K2[0]))
        two_d.append((ell ** 2, rel_err))
        print(f"{ell:>11} {ell ** 2:>7} {na:>8} {nnz:>10,} {sw:>7} {el:>7.2f}s {rel_err:>17.2e} {aerr:>17.3f}")
    figdata["grid"] = (one_d, two_d)

    # ------------------------------------------------------------------ Part E
    print("\n=== E. iLQR ===")
    # (i) exactness on an LQ problem: one iteration from a zero nominal reproduces the Riccati gains.
    fl, dl, tl = lq_problem(F, G, Cs, Ca)
    Hl = 30
    _, _, Kl = finite_horizon_lqr(F, G, Cs, Ca, 1.0, Hl, np.zeros((d, d)))
    Pl, _, _ = finite_horizon_lqr(F, G, Cs, Ca, 1.0, Hl, np.zeros((d, d)))
    S_l, U_l, k_l, K_l, J_l, hist_l, it_l = ilqr(fl, dl, tl, s0, np.zeros((Hl, m)), gamma=1.0, tol=1e-12, mu0=0.0)
    print(f"LQ problem of Part A (gamma = 1, H = {Hl}): iLQR stops after {it_l} iterations; "
          f"return after iteration 1 = {hist_l[1]:.6f}, -s0'P_0 s0 = {-s0 @ Pl[0] @ s0:.6f}; "
          f"max |K_t(iLQR) - K_t(Riccati)| = {np.max(np.abs(K_l - Kl)):.1e}")
    # (ii) pendulum swing-up from hanging at rest.
    s_down = np.array([np.pi, 0.0])
    Hp = 100
    U_init = 0.1 * np.random.default_rng(SEED).normal(size=(Hp, 1))
    t0 = time.perf_counter()
    S_p, U_p, k_p, K_p, J_p, hist_p, it_p = ilqr(pend_f, pend_derivs, pend_terminal, s_down, U_init,
                                                 lo=-AMAX, hi=AMAX, max_iter=40 if quick else 300)
    t_p = time.perf_counter() - t0
    up = np.where(np.cos(S_p[:, 0]) > np.cos(0.1))[0]
    t_up = up[0] if len(up) else None
    sat = np.mean(np.abs(U_p[:, 0]) >= AMAX - 1e-9)
    print(f"pendulum swing-up, H = {Hp} steps ({Hp * DT:.0f} s), from (theta, omega) = (pi, 0), initial torques "
          f"N(0, 0.1^2): {it_p} iterations, {t_p:.2f} s; model return {J_p:.2f} (start {hist_p[0]:.2f}); "
          f"|theta mod 2pi| < 0.1 from step {t_up}; torque at the limit in {100 * sat:.0f}% of steps; "
          f"max |omega| = {np.max(np.abs(S_p[:, 1])):.2f}")
    _, _, _, _, J_z, hist_z, it_z = ilqr(pend_f, pend_derivs, pend_terminal, s_down, np.zeros((Hp, 1)),
                                         lo=-AMAX, hi=AMAX, max_iter=40 if quick else 300)
    print(f"  from zero torques: stops after {it_z} iteration(s) with return {J_z:.2f} "
          f"(hanging at rest is a stationary point: the gradient vanishes by symmetry)")
    figdata["ilqr"] = {"S": S_p, "U": U_p, "hist": hist_p}
    if not quick:
        # iLQR is a local method: repeat from 10 random initial torque sequences, for H = 100 and H = 200.
        for Hs in (Hp, 2 * Hp):
            res = []
            for seed in range(10):
                U0s = 0.1 * np.random.default_rng(seed).normal(size=(Hs, 1))
                S_s, _, _, _, J_s, _, it_s = ilqr(pend_f, pend_derivs, pend_terminal, s_down, U0s,
                                                  lo=-AMAX, hi=AMAX)
                swung = np.any(np.cos(S_s[:, 0]) > np.cos(0.1))
                res.append((seed, J_s, it_s, swung))
                if Hs == 2 * Hp and not swung and "S200" not in figdata["ilqr"]:
                    figdata["ilqr"]["S200"] = S_s
            ok = [r for r in res if r[3]]
            bad = [r for r in res if not r[3]]
            msg = (f"  H = {Hs}, initial torques N(0, 0.1^2), seeds 0-9: {len(ok)}/10 swing up"
                   + (f" (return {np.mean([r[1] for r in ok]):.2f}, {min(r[2] for r in ok)}-"
                      f"{max(r[2] for r in ok)} iterations)" if ok else ""))
            if bad:
                msg += (f"; {len(bad)}/10 stop at a local optimum that never lifts the pendulum (return "
                        f"{min(r[1] for r in bad):.2f} to {max(r[1] for r in bad):.2f}, "
                        f"{min(r[2] for r in bad)}-{max(r[2] for r in bad)} iterations)")
            print(msg)
    # (iii) execute in Gymnasium: iLQR's time-varying feedback, then an LQR "catch" at the top.
    import gymnasium as gym
    Fu = np.array([[1 + 1.5 * GRAV * DT * DT, DT], [1.5 * GRAV * DT, 1.0]])
    Gu = np.array([[3 * DT * DT], [3 * DT]])
    Cs_u, Ca_u = np.diag([1.0, 0.1]), np.array([[0.001]])
    P_u = solve_discrete_are(Fu, Gu, Cs_u, Ca_u)
    K_u = np.linalg.solve(Ca_u + Gu.T @ P_u @ Gu, Gu.T @ P_u @ Fu)
    env = gym.make("Pendulum-v1")
    env.reset(seed=SEED)
    env.unwrapped.state = s_down.copy()
    s = env.unwrapped.state.copy()
    ret, traj, acts_g, dev = 0.0, [s.copy()], [], 0.0
    for t in range(200):
        if t < Hp:
            a = np.clip(U_p[t] - K_p[t] @ (s - S_p[t]), -AMAX, AMAX)
        else:
            a = np.clip(-K_u @ np.array([wrap(s[0]), s[1]]), -AMAX, AMAX)
        _, r, _, _, _ = env.step(np.asarray(a, dtype=np.float32))
        ret += r
        s = env.unwrapped.state.copy()
        traj.append(s.copy()); acts_g.append(float(a[0]))
        if t < Hp:
            dev = max(dev, np.max(np.abs(s - S_p[t + 1])))
    env.close()
    print(f"LQR catch at the top (gamma = 1, C_s = diag(1, 0.1), C_a = 0.001, Pendulum-v1 linearised): "
          f"K = {np.round(K_u, 3).tolist()}")
    print(f"Pendulum-v1 (200 steps, its own reward): iLQR feedback policy for t < {Hp}, then LQR: return "
          f"{ret:.1f}, final cos(theta) = {np.cos(traj[-1][0]):.4f}; max |state - iLQR plan| over the first "
          f"{Hp} steps = {dev:.1e}")
    figdata["gym"] = (np.array(traj), np.array(acts_g), ret)

    print(f"\ntotal time {time.perf_counter() - t_start:.1f} s")
    if not quick:
        make_figure(figdata)


def make_figure(fd):
    from plotting import C, plt, save
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    rows = fd["hewer"]
    sel = [r for r in rows if r[0] in (2, 16, 64)]
    for (dd, e_pi, e_vi), col in zip(sel, [C[0], C[1], C[2]]):
        ax.loglog(range(1, len(e_pi) + 1), e_pi, "o-", color=col, lw=2, ms=6, label=f"$d_s={dd}$")
        ax.loglog(range(1, len(e_vi) + 1), e_vi, "--", color=col, lw=1.5)
    ax.plot([], [], "o-", color="0.3", lw=2, ms=6, label="policy iteration (Hewer)")
    ax.plot([], [], "--", color="0.3", lw=1.5, label="value iteration (Riccati map)")
    ax.set_xlabel("iteration (evaluations for PI, sweeps for VI)")
    ax.set_ylabel(r"$\Vert P_k - P\Vert\,/\,\Vert P\Vert$")
    ax.set_title("LQR: PI converges quadratically, VI linearly")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, lw=0.5)

    ax = axes[1]
    one_d, two_d = fd["grid"]
    ax.loglog([x[0] for x in one_d], [x[1] for x in one_d], "o-", color=C[0], lw=2, ms=6, label="1-D grid")
    ax.loglog([x[0] for x in two_d], [x[1] for x in two_d], "s-", color=C[1], lw=2, ms=6, label="2-D grid")
    ax.set_xlabel("number of grid states $n$")
    ax.set_ylabel(r"relative error of $V$ vs $-s^\top P s$")
    ax.set_title("Tabular VI on a state grid vs the exact value")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, lw=0.5, which="both")

    ax = axes[2]
    traj, acts, ret = fd["gym"]
    tt = np.arange(len(traj)) * DT
    ax.plot(tt, np.cos(traj[:, 0]), color=C[0], lw=2, label=r"$\cos\theta$ in Pendulum-v1 (1 = upright)")
    if "S200" in fd["ilqr"]:
        S2 = fd["ilqr"]["S200"]
        ax.plot(np.arange(len(S2)) * DT, np.cos(S2[:, 0]), ":", color=C[2], lw=2,
                label=r"$\cos\theta$, iLQR local optimum ($H=200$)")
    ax.step(tt[:-1], np.array(acts) / AMAX, where="post", color=C[1], lw=1.2, alpha=0.8,
            label="torque / 2")
    ax.axvline(100 * DT, color="0.5", lw=1, ls="--")
    ax.text(100 * DT + 0.1, -0.35, "LQR catch", fontsize=8, color="0.3")
    ax.set_xlabel("time (s)")
    ax.set_ylim(-1.15, 1.15)
    ax.set_title(f"iLQR swing-up, then LQR (return {ret:.0f})")
    ax.legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.62))
    fig.tight_layout()
    save(fig, "lqr_ilqr.png")


if __name__ == "__main__":
    main()
