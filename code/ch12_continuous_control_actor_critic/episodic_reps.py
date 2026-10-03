"""Episodic REPS vs reward-weighted regression, CEM and parameter-space ES (Chapter 12, Section 9).

Task: a planar two-link arm (links of length 1) moves its end-effector from a fixed start pose through
a via-point to a target. Each joint follows a discrete dynamic movement primitive (DMP):

    tau z' = a_z (b_z (g - y) - z) + f(x),   tau y' = z,   tau x' = -a_x x,
    f(x)   = a_z b_z x sum_k psi_k(x) w_k / sum_k psi_k(x)            (Gaussian bases psi_k in the phase x)

with K = 5 basis weights per joint and the joint-space goal g as parameters, so theta has
d = 2 x 5 + 2 = 12 entries, all in radians (w_k is an offset of the attractor). One rollout is
T = 100 Euler steps of dt = 0.01 s. The return is deterministic:

    R(theta) = -[ 1000 ||e(T) - target||^2 + 1000 ||e(T/2) - via||^2 + 1e-3 sum_t ||y''_t||^2 dt + ||y'(T)||^2 ]

where e(t) is the end-effector position. The "cost" -R starts at 5,500 (the arm does not move) and the
best value found (L-BFGS-B with finite-difference gradients, refined from the best point any method
reached) is about 0.11, almost all of it acceleration cost.

Every method searches with a Gaussian over theta, starts from N(theta_0, I) with theta_0 = "do not move",
and spends N = 50 rollouts per iteration (budget 4,000 rollouts = 80 iterations):

  reps   Episodic REPS (Peters, Mulling & Altun, 2010). E-step: weights q_i ∝ exp(R_i / eta*), with eta*
         minimizing the convex dual g(eta) = eta eps + eta log (1/N) sum_i exp(R_i / eta) (Eq. 12.37),
         solved with scipy.optimize (L-BFGS-B on log eta, exact gradient eta (eps - KL)).  Check: the
         sample KL(q || uniform) equals eps at eta*.  M-step: weighted maximum likelihood of a
         diagonal Gaussian. eps in {0.5, 1, 2}.
  rwr    the same E- and M-step with a FIXED temperature eta in {1, 10, 100, 1000} (reward-weighted
         regression; Dayan & Hinton, 1997; Peters & Schaal, 2007).
  cem    the same M-step with uniform weights on the K = 25 best of the N samples (cross-entropy method).
  es     antithetic Gaussian ES with centred-rank fitness shaping and Adam, fixed search std sigma
         (Algorithm 10.9 of Chapter 10, Section 15): 25 pairs per iteration. The Adam optimizer and the
         rank shaping are imported from code/ch10_policy_gradients/black_box_search.py, whose
         CartPole loop is written inline and so cannot be imported as a whole.

Settings chosen once on pilot seeds 100-105 (not the reported seeds): ES sigma = 0.03 and Adam lr 0.1
(best of sigma in {0.03, 0.1, 0.3} x lr in {0.03, 0.1, 0.3}), CEM K = 25 (best of K in {5, 10, 25}).
Performance is the cost of the MEAN of the search distribution, evaluated by one extra rollout that does
not count towards the budget. Reported: median and interquartile range over the seeds.

Run (from the repository root):
  python code/ch12_continuous_control_actor_critic/episodic_reps.py           # full: 20 seeds, figure
  python code/ch12_continuous_control_actor_critic/episodic_reps.py --quick   # 3 seeds, 1,000 rollouts, no figure
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")

# Adam and centred-rank fitness shaping from Chapter 10's black-box script (loaded by path so that
# Chapter 10's plot_style.py cannot shadow ours).
_spec = importlib.util.spec_from_file_location(
    "ch10_black_box_search", os.path.join(HERE, "..", "ch10_policy_gradients", "black_box_search.py"))
_bb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bb)
Adam, centred_ranks = _bb.Adam, _bb.centred_ranks

# ----------------------------------------------------------------------------------------------
# The reaching task
# ----------------------------------------------------------------------------------------------
DT, T, T_VIA = 0.01, 100, 50
A_Z, B_Z, A_X = 25.0, 25.0 / 4, 3.0
K_BASIS = 5
CENTRES = np.exp(-A_X * np.linspace(0, 1, K_BASIS))                    # in the phase variable x
WIDTHS = 1.0 / np.diff(CENTRES, append=CENTRES[-1] * 0.5) ** 2
Q_START = np.array([0.3, 1.2])                                         # joint angles (rad)
TARGET, VIA = np.array([-1.0, 1.0]), np.array([0.0, 1.8])
C_GOAL, C_VIA, C_ACC, C_VEL = 1000.0, 1000.0, 1e-3, 1.0
THETA0 = np.concatenate([np.zeros(2 * K_BASIS), Q_START])              # "do not move"
D = THETA0.size


def forward_kinematics(q):
    """End-effector position of the two-link arm (links of length 1) for joint angles q[..., 2]."""
    a, b = q[..., 0], q[..., 0] + q[..., 1]
    return np.stack([np.cos(a) + np.cos(b), np.sin(a) + np.sin(b)], -1)


def rollout(theta):
    """Integrate the two joint DMPs for a batch theta[n, 12]. Returns joint path Y[T+1, n, 2],
    the integrated squared acceleration and the final joint velocity."""
    theta = np.atleast_2d(theta)
    n = theta.shape[0]
    W, g = theta[:, :2 * K_BASIS].reshape(n, 2, K_BASIS), theta[:, 2 * K_BASIS:]
    y, z, x = np.tile(Q_START, (n, 1)), np.zeros((n, 2)), 1.0
    Y = np.empty((T + 1, n, 2))
    Y[0] = y
    acc = np.zeros(n)
    for t in range(T):
        psi = np.exp(-WIDTHS * (x - CENTRES) ** 2)
        f = A_Z * B_Z * x * (W @ psi) / psi.sum()
        zdot = A_Z * (B_Z * (g - y) - z) + f                           # tau = 1, so y'' = z'
        acc += (zdot ** 2).sum(1) * DT
        z = z + zdot * DT
        y = y + z * DT
        x = x - A_X * x * DT
        Y[t + 1] = y
    return Y, acc, z


def returns(theta):
    """R(theta) for a batch of parameter vectors (deterministic)."""
    Y, acc, z = rollout(theta)
    e = forward_kinematics(Y)
    return -(C_GOAL * ((e[-1] - TARGET) ** 2).sum(1) + C_VIA * ((e[T_VIA] - VIA) ** 2).sum(1)
             + C_ACC * acc + C_VEL * (z ** 2).sum(1))


def refine(theta, h=1e-5):
    """Local L-BFGS-B polish with batched central-difference gradients (for the reference optimum)."""
    E = np.eye(D) * h

    def fg(t):
        R = returns(np.vstack([t, t + E, t - E]))
        return -R[0], -(R[1:D + 1] - R[D + 1:]) / (2 * h)

    res = minimize(fg, theta, jac=True, method="L-BFGS-B", options=dict(maxiter=5000, gtol=1e-9))
    return res.x, -res.fun


# ----------------------------------------------------------------------------------------------
# The REPS E-step
# ----------------------------------------------------------------------------------------------
def reps_weights(R, eps):
    """Solve the episodic REPS dual on one batch of returns.

    g(eta) = eta*eps + eta*log((1/N) sum_i exp(R_i/eta)) is convex in eta > 0 with
    g'(eta) = eps - KL(q_eta || uniform), q_eta,i ∝ exp(R_i/eta). Minimized over log eta.
    The returns are first standardized, R_i = R_max + s * u_i with s = std(R), so that the solver sees
    the same well-scaled problem whether the returns spread over thousands or over 1e-9
    (g(eta) = R_max + s * g_u(eta / s), where g_u is the dual for the u_i).
    Returns eta*, the weights q, the sample KL and the dual value g(eta*)."""
    N, Rmax, s = len(R), R.max(), R.std()
    if s == 0.0:                                                       # all returns equal: q = p
        return np.inf, np.full(N, 1.0 / N), 0.0, Rmax
    u = (R - Rmax) / s

    def dual(x):
        eta = np.exp(x[0])
        a = u / eta
        lse = logsumexp(a)
        q = np.exp(a - lse)
        kl = float(q @ np.log(np.maximum(q * N, 1e-300)))
        g = eta * eps + eta * (lse - np.log(N))
        return g, np.array([eta * (eps - kl)])                         # d g / d log eta

    res = minimize(dual, np.array([0.0]), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=500, gtol=1e-12, ftol=1e-15))
    eta_u = float(np.exp(res.x[0]))
    a = u / eta_u
    q = np.exp(a - logsumexp(a))
    kl = float(q @ np.log(np.maximum(q * N, 1e-300)))
    return eta_u * s, q, kl, Rmax + s * float(res.fun)


def gauss_kl_diag(m1, v1, m0, v0):
    """KL(N(m1, diag v1) || N(m0, diag v0))."""
    return 0.5 * float(np.sum(v1 / v0 + (m0 - m1) ** 2 / v0 - 1 + np.log(v0 / v1)))


# ----------------------------------------------------------------------------------------------
# Search loops
# ----------------------------------------------------------------------------------------------
N_SAMPLES = 50
ES_SIGMA, ES_LR = 0.03, 0.1
CEM_ELITES = 25
VAR_FLOOR = 1e-300                                                     # keeps a collapsed Gaussian finite


def run(method, seed, budget, eps=1.0, eta=10.0):
    """One run; returns a dict with the cost of the mean after every iteration and diagnostics."""
    rng = np.random.default_rng(seed)
    N = N_SAMPLES
    m, v = THETA0.copy(), np.ones(D)
    cost = [-returns(m)[0]]
    diag = dict(eta=[], kl=[], dual_gap=[], gauss_kl=[], ess=[], std_R=[], sd=[])
    opt = Adam(ES_LR)
    for _ in range(budget // N):
        if method == "es":                                             # Algorithm 10.9, antithetic
            xi = rng.standard_normal((N // 2, D))
            R = returns(np.concatenate([m + ES_SIGMA * xi, m - ES_SIGMA * xi]))
            u = centred_ranks(R)
            grad = ((u[:N // 2] - u[N // 2:])[:, None] * xi).sum(0) / (N * ES_SIGMA)
            m = m + opt.step(grad)
        else:
            X = m + np.sqrt(v) * rng.standard_normal((N, D))
            R = returns(X)
            if method == "reps":
                e, q, kl, gval = reps_weights(R, eps)
                diag["eta"].append(e)
                diag["kl"].append(kl)
                diag["dual_gap"].append(gval - float(q @ R))           # strong duality: g(eta*) = E_q[R]
                diag["std_R"].append(R.std())
            elif method == "rwr":
                a = (R - R.max()) / eta
                q = np.exp(a - logsumexp(a))
            elif method == "cem":
                q = np.zeros(N)
                q[np.argsort(-R, kind="stable")[:CEM_ELITES]] = 1.0 / CEM_ELITES
            m_new = q @ X                                              # weighted maximum likelihood
            v_new = np.maximum(q @ (X - m_new) ** 2, VAR_FLOOR)
            diag["gauss_kl"].append(gauss_kl_diag(m_new, v_new, m, v))
            diag["ess"].append(1.0 / float(q @ q))
            m, v = m_new, v_new
            diag["sd"].append(float(np.sqrt(v).mean()))
        cost.append(-returns(m)[0])
    return dict(cost=np.array(cost), m=m, **{k: np.array(x) for k, x in diag.items()})


def fmt_iqr(x):
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return f"{med:9.3f} [{q1:.3g}, {q3:.3g}]"


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="first seed")
    args = ap.parse_args()
    t0 = time.time()
    n_seeds, budget = (3, 1000) if args.quick else (20, 4000)
    seeds = list(range(args.seed, args.seed + n_seeds))
    print(f"seeds {seeds[0]}..{seeds[-1]}  mode={'quick' if args.quick else 'full'}  d={D}  "
          f"N={N_SAMPLES} rollouts/iteration  budget={budget} rollouts")

    # ---- Part A: one REPS E-step, by hand ----------------------------------------------------
    rng = np.random.default_rng(args.seed)
    R0 = returns(THETA0 + rng.standard_normal((N_SAMPLES, D)))
    print("\nPart A: the REPS dual on the first batch (N = 50 samples from N(theta_0, I))")
    print(f"  returns: mean {R0.mean():.1f}, std {R0.std():.1f}, best {R0.max():.1f}")
    for eps in (0.5, 1.0, 2.0):
        e, q, kl, gval = reps_weights(R0, eps)
        print(f"  eps={eps:3.1f}: eta*={e:9.2f}  sample KL(q||p)={kl:.8f}  ESS={1 / (q @ q):5.1f}  "
              f"g(eta*)={gval:.4f}  E_q[R]={q @ R0:.4f}  "
              f"eta*/std(R)={e / R0.std():.3f} (Gaussian returns: 1/sqrt(2 eps)={1 / np.sqrt(2 * eps):.3f})")
    print(f"  (KL(q||p) <= log N = {np.log(N_SAMPLES):.3f} for any weights, so eps must be below it)")

    # ---- Part B: learning curves ----------------------------------------------------------------
    configs = [("REPS eps=0.5", "reps", dict(eps=0.5)), ("REPS eps=1", "reps", dict(eps=1.0)),
               ("REPS eps=2", "reps", dict(eps=2.0)),
               ("RWR eta=1", "rwr", dict(eta=1.0)), ("RWR eta=10", "rwr", dict(eta=10.0)),
               ("RWR eta=100", "rwr", dict(eta=100.0)), ("RWR eta=1000", "rwr", dict(eta=1000.0)),
               (f"CEM K={CEM_ELITES}", "cem", {}), (f"ES sigma={ES_SIGMA}", "es", {})]
    results = {}
    for name, method, kw in configs:
        results[name] = [run(method, s, budget, **kw) for s in seeds]
    checkpoints = [c for c in (500, 1000, 2000, 4000) if c <= budget]
    print("\nPart B: cost -R of the search mean, median [IQR] over seeds, by rollouts used")
    print(f"  {'method':14s}" + "".join(f"{c:>26d}" for c in checkpoints) + "   final sd of search dist.")
    for name, runs in results.items():
        C = np.array([r["cost"] for r in runs])
        row = "".join(f"{fmt_iqr(C[:, c // N_SAMPLES]):>26s}" for c in checkpoints)
        sd = np.median([r["sd"][-1] for r in runs]) if runs[0]["sd"].size else ES_SIGMA
        print(f"  {name:14s}{row}   {sd:.2g}")
    for name in ("RWR eta=1", "RWR eta=10"):
        it = [int(np.argmax(np.array(r["sd"]) < 1e-3)) + 1 if np.min(r["sd"]) < 1e-3 else None
              for r in results[name]]
        hit = [i for i in it if i is not None]
        print(f"  {name}: search sd < 1e-3 on {len(hit)}/{len(it)} seeds"
              + (f", after {min(hit)}-{max(hit)} iterations" if hit else ""))

    # ---- Part C: REPS diagnostics ---------------------------------------------------------------
    print("\nPart C: REPS diagnostics over all iterations and seeds")
    for name in ("REPS eps=0.5", "REPS eps=1", "REPS eps=2"):
        runs = results[name]
        eps = float(name.split("=")[1])
        kl = np.concatenate([r["kl"] for r in runs])
        gap = np.concatenate([r["dual_gap"] for r in runs])
        gk = np.concatenate([r["gauss_kl"] for r in runs])
        ess = np.concatenate([r["ess"] for r in runs])
        eta = np.array([r["eta"] for r in runs])
        ratio = np.concatenate([r["eta"] / np.maximum(r["std_R"], 1e-300) for r in runs])
        ratio = ratio[np.isfinite(ratio)]
        print(f"  {name}: max |sample KL - eps| = {np.abs(kl - eps).max():.1e};  "
              f"max |g(eta*) - E_q[R]| = {np.abs(gap).max():.1e};  median ESS {np.median(ess):.1f} of {N_SAMPLES}")
        print(f"      eta*: median {np.median(eta[:, 0]):.3g} at iteration 1 -> {np.median(eta[:, -1]):.3g} "
              f"at the last;  eta*/std(R): median {np.median(ratio):.3f} [{np.percentile(ratio, 25):.3f}, "
              f"{np.percentile(ratio, 75):.3f}] (Gaussian returns: {1 / np.sqrt(2 * eps):.3f})")
        print(f"      KL(p_new || p_old) of the FITTED Gaussians: median {np.median(gk):.2f}, "
              f"90th percentile {np.percentile(gk, 90):.2f}  (bound on the samples: {eps})")

    # ---- reference optimum ------------------------------------------------------------------------
    finals = [(r["cost"][-1], r["m"]) for runs in results.values() for r in runs]
    best_cost, best_m = min(finals, key=lambda t: t[0])
    th_ref, R_ref = refine(best_m)
    print(f"\nReference: best final mean of any run has cost {best_cost:.4f}; "
          f"L-BFGS-B polish from it reaches {-R_ref:.4f}")
    Y, acc, _ = rollout(th_ref)
    e = forward_kinematics(Y)[:, 0]
    print(f"  at the reference: target error {np.linalg.norm(e[-1] - TARGET) * 1000:.2f} mm, via error "
          f"{np.linalg.norm(e[T_VIA] - VIA) * 1000:.2f} mm, acceleration cost {C_ACC * acc[0]:.4f}")
    print(f"\nTotal time {time.time() - t0:.0f}s")

    if not args.quick:
        make_figure(results, -R_ref, seeds)


def make_figure(results, ref_cost, seeds):
    from plot_style import setup, C, GREY, INK
    plt = setup()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    show = [("REPS eps=1", C[0], "-"), ("RWR eta=10", C[1], "--"), ("RWR eta=1000", C[3], "-."),
            (f"CEM K={CEM_ELITES}", C[2], ":"), (f"ES sigma={ES_SIGMA}", C[4], (0, (5, 1, 1, 1)))]
    for name, col, ls in show:
        Cst = np.array([r["cost"] for r in results[name]])
        x = np.arange(Cst.shape[1]) * N_SAMPLES
        med = np.median(Cst, 0)
        lo, hi = np.percentile(Cst, [25, 75], axis=0)
        ax.plot(x, med, color=col, ls=ls, label=name.replace("eps", "$\\epsilon$").replace("eta", "$\\eta$")
                .replace("sigma", "$\\sigma$"))
        ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
    ax.axhline(ref_cost, color=GREY, lw=1.0, ls=":")
    ax.text(x[-1], ref_cost * 1.25, "best found", color=GREY, ha="right", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("rollouts")
    ax.set_ylabel("cost $-R$ of the search mean")
    ax.set_title(f"Learning curves (median, IQR; {len(seeds)} seeds)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)

    ax = axes[1]
    r = results["REPS eps=1"]
    eta = np.array([x["eta"] for x in r])
    sR = np.array([x["std_R"] for x in r])
    it = np.arange(1, eta.shape[1] + 1)
    ax.plot(it, np.median(sR, 0), color=C[6], ls="--", label="std of the batch returns")
    ax.plot(it, np.median(eta, 0), color=C[0], label="$\\eta^\\ast$ from the dual ($\\epsilon=1$)")
    ax.fill_between(it, *np.percentile(eta, [25, 75], axis=0), color=C[0], alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("iteration (50 rollouts each)")
    ax.set_ylabel("reward units")
    ax.set_title("REPS sets its temperature from the data")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[2]
    best = min(r, key=lambda x: x["cost"][-1])
    rng = np.random.default_rng(seeds[0])
    X0 = THETA0 + rng.standard_normal((12, D))
    for k, th in enumerate(X0):
        e = forward_kinematics(rollout(th)[0])[:, 0]
        ax.plot(e[:, 0], e[:, 1], color=GREY, lw=0.8, alpha=0.6, label="samples from $p_0$" if k == 0 else None)
    Y = rollout(best["m"])[0]
    e = forward_kinematics(Y)[:, 0]
    ax.plot(e[:, 0], e[:, 1], color=C[0], lw=2.0, label="REPS ($\\epsilon=1$) mean, 4,000 rollouts, best seed")
    for t, alpha in ((0, 0.35), (T_VIA, 0.6), (T, 1.0)):
        q = Y[t, 0]
        elbow = np.array([np.cos(q[0]), np.sin(q[0])])
        pts = np.array([[0, 0], elbow, e[t]])
        ax.plot(pts[:, 0], pts[:, 1], color=INK, lw=1.4, alpha=alpha, marker="o", ms=3)
    ax.plot(*TARGET, marker="*", ms=14, color=C[7], ls="none", label="target (t = 1 s)")
    ax.plot(*VIA, marker="D", ms=8, color=C[3], ls="none", label="via-point (t = 0.5 s)")
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.6, 2.3)
    ax.set_xlabel("x (link lengths)")
    ax.set_ylabel("y")
    ax.set_title("End-effector paths (arm at t = 0, 0.5, 1 s)")
    ax.legend(fontsize=7.5, loc="lower left")

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, "episodic_reps.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"figure written to {path}")


if __name__ == "__main__":
    main()
