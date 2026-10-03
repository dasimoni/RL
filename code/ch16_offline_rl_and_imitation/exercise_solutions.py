"""Numerical checks for the Chapter 16 exercises.

  Exercise 16.2   BC's T^2 bound is tight: exact excess cost of a learner that errs with
                  probability eps in the expert's only state and then falls into an absorbing
                  costly state.
  Exercise 16.9   expectiles of a two-point distribution: closed form vs numerical minimiser.
  Exercise 16.10  the KL-regularised improvement step has solution pi* ~ pi_beta exp(beta A).
  Exercise 16.13  tabular GAIL on the Section 4 gridworld (irl_gridworld.py), with exact
                  occupancy measures and exact policy gradients; a stable discriminator step
                  (main run) and one that is too large (contrast run).
  Exercise 16.14  offline policy selection: pick the best of several candidate policies from
                  logged data using an OPE estimator (ope_tabular.py's MDP).
  Exercise 16.15  mode averaging in one step: collision probability of MSE-BC, of a maximum-
                  likelihood Gaussian and of the bimodal expert (closed form and Monte Carlo).
  Exercise 16.16  the denoising chain as an MDP: the score-function gradient summed over the
                  per-step Gaussian log-likelihoods of a sampled chain is unbiased for the gradient
                  of the expected reward of the final sample (a 2-step linear-Gaussian chain).

Run:  python code/ch16_offline_rl_and_imitation/exercise_solutions.py [--quick]
(No figures.)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

import irl_gridworld as IG
import ope_tabular as OT


# ---------------------------------------------------------------- Exercise 16.2
def bc_tightness():
    print("Exercise 16.2: excess cost of the 'cliff' learner, J = T - (1 - (1-eps)^T) / eps")
    eps = 0.01
    print(f"  eps = {eps}")
    print(f"  {'T':>5s} {'exact (sim. of chain)':>22s} {'closed form':>12s} {'eps T(T-1)/2':>13s} {'bound eps T^2':>14s}")
    for T in [10, 30, 100, 300, 1000]:
        # exact propagation of P(in the bad state) through the 2-state chain
        p_bad, J = 0.0, 0.0
        for t in range(T):
            J += p_bad                       # cost 1 at step t if already in the bad state
            p_bad = p_bad + (1 - p_bad) * eps
        closed = T - (1 - (1 - eps) ** T) / eps
        print(f"  {T:5d} {J:22.3f} {closed:12.3f} {eps * T * (T - 1) / 2:13.3f} {eps * T * T:14.1f}")


# ---------------------------------------------------------------- Exercise 16.9
def expectile_check():
    print("\nExercise 16.9: tau-expectile of X in {0 (prob 1-p), 1 (prob p)}, p = 0.2")
    p = 0.2
    for tau in [0.5, 0.7, 0.9, 0.99]:
        closed = tau * p / (tau * p + (1 - tau) * (1 - p))
        loss = lambda m: (1 - p) * (1 - tau) * m ** 2 + p * tau * (1 - m) ** 2
        num = minimize_scalar(loss, bounds=(0, 1), method="bounded").x
        print(f"  tau={tau:4.2f}: closed form {closed:.4f}, numerical minimiser {num:.4f}")


# ---------------------------------------------------------------- Exercise 16.10
def awr_check(rng):
    print("\nExercise 16.10: argmax_pi E_pi[A] - (1/beta) KL(pi || pi_beta) over the simplex")
    pi_beta = np.array([0.6, 0.3, 0.1]); A = np.array([0.0, 0.5, 1.0]); beta = 2.0
    closed = pi_beta * np.exp(beta * A); closed /= closed.sum()
    obj = lambda z: -(np.exp(z - logsumexp(z)) @ A
                      - (1 / beta) * np.exp(z - logsumexp(z)) @ (z - logsumexp(z) - np.log(pi_beta)))
    best = min((minimize(obj, rng.standard_normal(3)) for _ in range(5)), key=lambda r: r.fun)
    num = np.exp(best.x - logsumexp(best.x))
    print(f"  closed form pi* = {np.round(closed, 4).tolist()}; numerical optimum = {np.round(num, 4).tolist()}")


# ---------------------------------------------------------------- Exercise 16.13
def occupancy(grid, pis):
    D = grid.visitation(pis)                                  # [H, S]
    return np.einsum("ts,tsa->sa", D, pis)                    # sum over t of P(s_t = s, a_t = a)


def soft_q(grid, pis, reward_sa, lam):
    """Entropy-regularised evaluation of a time-dependent policy: returns Q_t and V_t."""
    H = pis.shape[0]
    Q = np.zeros_like(pis); V = np.zeros((H + 1, grid.ns))
    for t in reversed(range(H)):
        Q[t] = reward_sa + grid.P @ V[t + 1]
        V[t] = (pis[t] * (Q[t] - lam * np.log(pis[t] + 1e-300))).sum(1)
    return Q, V[:-1]


def tabular_gail(grid, rho_E, iters, lam=0.1, lr_pi=0.5, lr_d=0.2, d_steps=5, track_state=None):
    """GAIL (Algorithm 16.5) with exact occupancy measures.  D(s,a) = P(expert | s, a) is a
    table of logits psi; the policy has time-dependent softmax logits theta[t, s, a].
      D step:  a few gradient-ascent steps on  sum rho_E log D + sum rho_pi log(1 - D)
      pi step: natural-gradient (soft policy iteration) step on E_pi[sum_t r] + lam H(pi),
               reward r(s,a) = -log(1 - D(s,a)), i.e. theta += lr * (Q_soft - lam log pi - V).
    Step size of the D step.  Per pair, d/dpsi = rho_E (1 - D) - rho_pi D, with fixed point
    D* = rho_E / (rho_E + rho_pi) (Eq. 16.19).  Linearised, one step with step size
    eta = lr_d * |S| multiplies the error by 1 - eta (rho_E + rho_pi) D*(1 - D*), so it is
    stable only if  eta (rho_E + rho_pi) D*(1 - D*) < 2.  The absorbing goal has by far the
    largest occupancy per (s, a) pair; with lr_d = 1 it violates this and D there flips
    between high and low values from one iteration to the next.  lr_d = 0.2 is stable.
    If track_state is given, D[track_state] is recorded at each of the last 3 iterations."""
    H, nA = IG.H, IG.NA
    theta = np.zeros((H, grid.ns, nA))
    psi = np.zeros((grid.ns, nA))
    hist, track = [], []
    rE = rho_E / rho_E.sum()
    for k in range(iters):
        pis = np.exp(theta - logsumexp(theta, axis=2, keepdims=True))
        rho = occupancy(grid, pis)
        rp = rho / rho.sum()
        for _ in range(d_steps):
            D = 1 / (1 + np.exp(-psi))
            psi += lr_d * (rE * (1 - D) - rp * D) * grid.ns   # gradient of the D objective
        D = 1 / (1 + np.exp(-psi))
        if track_state is not None and k >= iters - 3:
            track.append((k, D[track_state].copy()))
        reward = -np.log(1 - D + 1e-12)
        Q, V = soft_q(grid, pis, reward, lam)
        theta += lr_pi * (Q - lam * np.log(pis + 1e-300) - V[:, :, None])
        if k % max(iters // 10, 1) == 0 or k == iters - 1:
            m = 0.5 * (rE + rp)
            js = 0.5 * np.sum(rE[rE > 0] * np.log(rE[rE > 0] / m[rE > 0])) \
                + 0.5 * np.sum(rp[rp > 0] * np.log(rp[rp > 0] / m[rp > 0]))
            hist.append((k, np.abs(rE - rp).sum(), js, grid.value(pis)))
    pis = np.exp(theta - logsumexp(theta, axis=2, keepdims=True))
    # largest linearised stability factor eta (rho_E + rho_pi) / 4 (uses D*(1-D*) <= 1/4),
    # at the goal and over all other pairs, for the final policy
    rp = occupancy(grid, pis); rp = rp / rp.sum()
    fac = lr_d * grid.ns * (rE + rp) / 4
    other = np.ones(grid.ns, bool); other[grid.goal] = False
    stab = (fac[grid.goal].max(), fac[other].max())
    return pis, 1 / (1 + np.exp(-psi)), hist, track, stab


def gail_check(quick):
    print("\nExercise 16.13: tabular GAIL on the irl_gridworld.py training world")
    rng = np.random.default_rng(0)
    train = IG.Grid(IG.TRAIN_MAP, start=(0, 0))
    test = IG.Grid(IG.TEST_MAP, start=(6, 0))
    expert = train.soft_policy(train.phi @ IG.OMEGA_TRUE)
    demos = train.sample(expert, 30, rng)                      # same demonstrations as Section 4
    rho_E = np.zeros((train.ns, IG.NA))
    for traj in demos:
        for s, a in traj:
            rho_E[s, a] += 1.0 / len(demos)
    iters = 200 if quick else 3000
    supp = rho_E > 0
    off_goal = supp.copy(); off_goal[train.goal] = False
    goal_max = train.feature_expectations(train.optimal_policy(train.phi[:, 3]))[3]
    print(f"  iterations={iters}, entropy weight lam=0.1, 5 D steps per policy step")
    print(f"  expert's empirical goal visits {rho_E[train.goal].sum():.2f}; the most any policy "
          f"can achieve is {goal_max:.3f} (Section 4.4)")
    r_true_test = test.phi @ IG.OMEGA_TRUE
    print(f"  soft expert's true return in the training world: {train.value(expert):.2f}; "
          f"true optimum in the transfer world: {test.value(test.optimal_policy(r_true_test)):.2f}")
    for lr_d, label in [(0.2, "stable D step (main run)"), (1.0, "D step too large (contrast)")]:
        pis, D, hist, track, stab = tabular_gail(train, rho_E, iters, lr_d=lr_d, track_state=train.goal)
        print(f"\n  lr_d = {lr_d}: {label}; stability factor eta(rho_E+rho_pi)/4 at the goal "
              f"{stab[0]:.2f}, elsewhere at most {stab[1]:.2f} (stable if < 2)")
        print(f"  {'iter':>5s} {'L1(rho_E, rho_pi)':>18s} {'JS':>8s} {'true return':>12s}")
        for k, l1, js, v in hist:
            print(f"  {k:5d} {l1:18.4f} {js:8.4f} {v:12.2f}")
        print(f"  GAIL policy's goal visits {occupancy(train, pis)[train.goal].sum():.2f}")
        print(f"  D(s,a) on the expert's support: mean {D[supp].mean():.3f}, min {D[supp].min():.3f}, "
              f"max {D[supp].max():.3f}; at the goal {np.round(D[train.goal], 3).tolist()}; "
              f"elsewhere on the support [{D[off_goal].min():.3f}, {D[off_goal].max():.3f}]")
        print("  D at the goal over the last iterations: "
              + "; ".join(f"it. {k}: {np.round(d, 3).tolist()}" for k, d in track))
        # transfer: (i) replay the learned (state-indexed) policy; (ii) re-optimise the
        # discriminator reward r = -log(1 - D) in the new world
        r_d = -np.log(1 - D + 1e-12)
        V = np.zeros(test.ns); pis_t = np.zeros((IG.H, test.ns, IG.NA))
        for t in reversed(range(IG.H)):
            Qd = r_d + test.P @ V
            V = Qd.max(1)
            pis_t[t] = np.eye(IG.NA)[Qd.argmax(1)]
        mu_t = test.feature_expectations(pis_t)
        print(f"  reward -log(1-D): mean on the expert's support {r_d[supp].mean():.3f}, "
              f"at the goal {np.round(r_d[train.goal], 3).tolist()}")
        print(f"  transfer world: GAIL policy replayed {test.value(pis):.2f}; optimal for the "
              f"discriminator reward {test.value(pis_t):.2f} (expected visits road/grass/mud/goal "
              f"{np.round(mu_t, 2).tolist()})")
        # The discriminator is a table over grid cells, not terrain types: which cells does the
        # re-planned policy use, and what reward did the discriminator leave there?
        vis = test.visitation(pis_t).sum(0)
        cells = [f"({s // test.w},{s % test.w}) {IG.FEATURES[train.terrain[s]]}->{IG.FEATURES[test.terrain[s]]}"
                 f" r={r_d[s].max():.3f}{' [demo visits ' + format(rho_E[s].sum(), '.2f') + ']'}"
                 for s in np.flatnonzero(vis > 0.01)]
        print("  cells used by the re-planned policy (train->transfer terrain, max_a reward, demo visits):\n    "
              + "\n    ".join(cells))


# ---------------------------------------------------------------- Exercise 16.14
def policy_selection(quick):
    print("\nExercise 16.14: pick the best of 7 candidate policies (mixtures of b with the greedy or the worst policy) from logged data")
    rng = np.random.default_rng(1)
    H, n = 10, 200
    P, R, d0 = OT.make_mdp(0)
    b = np.exp(0.7 * np.random.default_rng(1).standard_normal((OT.NS, OT.NA)))
    b /= b.sum(1, keepdims=True)
    greedy = np.eye(OT.NA)[OT.optimal_q(P, R, H)[0].argmax(1)]
    # candidates: mixtures towards the greedy policy, and towards a deliberately bad policy
    worst = np.eye(OT.NA)[OT.optimal_q(P, -R, H)[0].argmax(1)]
    cands = [(1 - k) * b + k * greedy for k in (0.0, 0.25, 0.5, 0.75, 1.0)] + \
            [(1 - k) * b + k * worst for k in (0.5, 1.0)]
    v_true = np.array([OT.true_value(P, R, d0, c, H)[0] for c in cands])
    print("  true values of the 7 candidates:", np.round(v_true, 3).tolist())
    reps = 40 if quick else 300
    names = ["WIS", "WPDIS", "FQE", "DR", "WDR"]
    cols = [OT.NAMES.index(m) for m in names]
    chosen = {m: [] for m in names}
    for _ in range(reps):
        S, A, Rw = OT.generate(P, R, d0, b, n, H, rng)
        est = np.array([OT.all_estimates(S, A, Rw, c, b, H) for c in cands])   # [cand, est]
        for m, j in zip(names, cols):
            chosen[m].append(v_true[int(np.argmax(est[:, j]))])
    print(f"  over {reps} datasets of n = {n} episodes: mean true value of the selected policy "
          f"(best possible {v_true.max():.3f}, behaviour {v_true[0]:.3f})")
    for m in names:
        c = np.array(chosen[m])
        print(f"    {m:6s}: {c.mean():.3f}   (picked the best candidate in {np.mean(c == v_true.max()):.0%} of datasets)")


# ---------------------------------------------------------------- Exercise 16.15
def mode_averaging(rng):
    from scipy.stats import norm
    print("\nExercise 16.15: expert action a = +m or -m (prob 1/2 each) plus N(0, s^2) noise; |a| < c collides")
    m, s, c = 0.1, 0.01, 0.05
    p_expert = norm.cdf((c - m) / s) - norm.cdf((-c - m) / s)          # the same for both modes
    sd = np.sqrt(m ** 2 + s ** 2)                                     # ML Gaussian: moment matching
    p_gauss = 2 * norm.cdf(c / sd) - 1
    a = rng.choice([-m, m], 1_000_000) + s * rng.standard_normal(1_000_000)
    mu_hat, sd_hat = a.mean(), a.std()
    b = mu_hat + sd_hat * rng.standard_normal(1_000_000)
    print(f"  m = {m}, s = {s}, c = {c}")
    print(f"  MSE-BC (deterministic mean 0):  P(collide) = 1")
    print(f"  ML Gaussian N(0, m^2 + s^2), sd {sd:.4f}: P(collide) = {p_gauss:.4f}   "
          f"(Monte Carlo: fitted mean {mu_hat:+.5f}, sd {sd_hat:.4f}, P = {np.mean(np.abs(b) < c):.4f})")
    print(f"  expert (or an exact 2-component mixture): P(collide) = {p_expert:.2e}   "
          f"(Monte Carlo: {np.mean(np.abs(a) < c):.2e})")


# ---------------------------------------------------------------- Exercise 16.16
def denoising_chain_gradient(rng, quick):
    """Chain a2 ~ N(0, 1), a1 ~ N(th1 + 0.5 a2, s1^2), a0 ~ N(th0 + 0.5 a1, s0^2); reward
    R(a0) = -(a0 - 2)^2.  a0 is Gaussian with mean th0 + 0.5 th1 and variance s0^2 + 0.25 s1^2
    + 0.0625, so grad J is known in closed form.  Compare with the score-function estimate
    E[ R(a0) * sum_k grad log p(a^{k-1} | a^k) ]."""
    print("\nExercise 16.16: policy gradient through a 2-step denoising chain (score function of the chain)")
    th0, th1, s0, s1 = 0.3, -0.4, 0.5, 0.8
    mean0 = th0 + 0.5 * th1
    # J = E[-(a0 - 2)^2] = -(mean0 - 2)^2 - var0;  dJ/dth0 = -2 (mean0 - 2), dJ/dth1 = 0.5 dJ/dth0
    g_exact = np.array([-2 * (mean0 - 2), -(mean0 - 2)])
    n = 200_000 if quick else 2_000_000
    a2 = rng.standard_normal(n)
    a1 = th1 + 0.5 * a2 + s1 * rng.standard_normal(n)
    a0 = th0 + 0.5 * a1 + s0 * rng.standard_normal(n)
    R = -(a0 - 2) ** 2
    score0 = (a0 - th0 - 0.5 * a1) / s0 ** 2            # d/dth0 log N(a0; th0 + 0.5 a1, s0^2)
    score1 = (a1 - th1 - 0.5 * a2) / s1 ** 2            # d/dth1 log N(a1; th1 + 0.5 a2, s1^2)
    # th1 also moves a0's mean only through the sample a1, which the chain's score accounts for
    g_naive = np.array([np.mean(R * score0), np.mean(R * score1)])
    b = R.mean()                                         # a constant baseline (variance reduction)
    g_base = np.array([np.mean((R - b) * score0), np.mean((R - b) * score1)])
    se = np.array([np.std((R - b) * score0), np.std((R - b) * score1)]) / np.sqrt(n)
    print(f"  exact grad J = {np.round(g_exact, 4).tolist()}")
    print(f"  score-function estimate ({n:,} chains)           = {np.round(g_naive, 4).tolist()}")
    print(f"  with a constant baseline                          = {np.round(g_base, 4).tolist()}"
          f"  (standard errors {np.round(se, 4).tolist()})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    print(f"exercise_solutions: seed=0 quick={args.quick}")
    t0 = time.time()
    rng = np.random.default_rng(0)
    bc_tightness()
    expectile_check()
    awr_check(rng)
    gail_check(args.quick)
    policy_selection(args.quick)
    mode_averaging(np.random.default_rng(16))
    denoising_chain_gradient(np.random.default_rng(17), args.quick)
    print(f"total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
