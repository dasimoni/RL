"""Numerical checks for the Chapter 19 exercises (Exercises 2, 5, 7, 10, 13, 14 and 16).

  Ex. 2   the 1/n trap: iterations needed to halve the deterministic error of Q-learning
  Ex. 5   the sample size the toy theorem (19.19) asks for, versus what is actually needed
  Ex. 7   the combination lock: probability that eps-greedy reaches the goal in one episode,
          computed exactly by dynamic programming, under the idealised premise of the exercise that
          the greedy action is wrong in EVERY lock state (the agents of Section 7.6 do not satisfy
          it exactly; regret_experiment.py measures what they actually do)
  Ex. 10  one natural-policy-gradient step on a 3-armed bandit, by hand
  Ex. 13  asynchronous Q-learning along ONE trajectory (uniform behaviour) with five step sizes
  Ex. 14  LSVI-UCB with one-hot features (a tabular MDP is a linear MDP with d = S*A)
  Ex. 16  how good is the TD(lambda) fixed point? Tsitsiklis-Van Roy's constant (1-g lam)/(1-g)
          versus the Pythagorean constant 1/sqrt(1-g_lam^2), g_lam = g(1-lam)/(1-g lam): exact
          fixed points on random chains, a two-state chain on which the lambda = 0 constant is
          attained, and a random search for large ratios when lambda > 0 (calibrated at lambda = 0)

Seeds: --seed s (default 0) sets every random stream. Ex. 13 builds the MDP of q_learning_rates.py
from seed s and samples the trajectories with seed s + 5; Ex. 14 uses the two MDPs of
regret_experiment.py with the same --seed (random MDP from s + 7, lock from s + 3) and samples with
seed s + 11; Ex. 16 draws its chains with seed s + 13. The derived seeds are printed.

Run from the repository root:
  python code/ch19_rl_theory/exercise_solutions.py           # full (~1.5 min)
  python code/ch19_rl_theory/exercise_solutions.py --quick   # smoke test
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

from theory_lib import fh_optimal, random_mdp, value_iteration
from regret_experiment import eval_policies, make_lock_env, make_random_env


def ex2():
    print("Ex. 2: alpha_n = 1/n, error factor prod_{k<=n} (1 - (1-gamma)/k) ~ n^-(1-gamma) / Gamma(gamma)")
    for g in (0.9, 0.99):
        n_pred = (2.0 / math.gamma(g)) ** (1.0 / (1.0 - g))
        line = f"   gamma = {g}: predicted n to halve the error = {n_pred:.3g}"
        if n_pred < 1e7:
            k = np.arange(1, int(n_pred * 3) + 1)
            prod = np.exp(np.cumsum(np.log1p(-(1 - g) / k)))
            line += f"; exact first n with factor <= 1/2: {int(np.argmax(prod <= 0.5)) + 1}"
        print(line)
    for g in (0.9, 0.99):
        print(f"   rescaled linear 1/(1+(1-g)n), gamma = {g}: factor 1/(1+(1-g)n) = 1/2 at n = {1 / (1 - g):.0f}")


def ex5():
    S, A, H, eps, delta = 5, 2, 10, 0.1, 0.1
    N = H ** 2 * (H - 1) ** 2 * math.log(4 * S * A * H / delta) / (2 * eps ** 2)
    n_int = math.ceil(N)
    print(f"Ex. 5: toy theorem (19.19) with S={S}, A={A}, H={H}, eps={eps}, delta={delta}: "
          f"N >= {N:,.2f}, i.e. N = {n_int:,d} samples per (s,a,h), total {S * A * H * n_int:,d}")


def ex7(eps=0.1, A=2, L=6, H=9):
    q = eps / A                                    # prob. of a correct action when greedy is wrong
    # DP over (time, lock position): position resets to 0 on a wrong action.
    dist = np.zeros(L + 1)
    dist[0] = 1.0
    for _ in range(H):
        new = np.zeros(L + 1)
        new[L] += dist[L]
        new[1:L + 1] += q * dist[:L]
        new[0] += (1 - q) * dist[:L].sum()
        dist = new
    p = dist[L]
    print(f"Ex. 7: lock L={L}, H={H}, eps={eps}, A={A}, greedy action wrong in every lock state (idealised): "
          f"P(reach goal in an episode) = {p:.3e} "
          f"(approximation (H-L+1) q^L = {(H - L + 1) * q ** L:.3e}); expected episodes ~ {1 / p:.2e}")


def ex10():
    r = np.array([1.0, 0.5, 0.0])
    pi0 = np.full(3, 1 / 3)
    for step in (1.0, 5.0):        # step = eta / (1 - gamma) with gamma = 0 (a bandit)
        adv = r - pi0 @ r
        pi1 = pi0 * np.exp(step * adv)
        pi1 /= pi1.sum()
        gap = r.max() - pi1 @ r
        bound = math.log(3) / step + 1.0          # log|A|/(eta T) + 1/((1-gamma)^2 T), T = 1, gamma = 0
        print(f"Ex. 10: NPG step eta={step}: advantages {np.round(adv, 4)}, pi_1 = {np.round(pi1, 4)}, "
              f"gap = {gap:.4f}, NPG bound (19.29) for T=1 = {bound:.4f}")


SCHEDULES_ASYNC = [
    ("1/n", lambda n, g: 1.0 / n),
    ("1/n^0.8", lambda n, g: n ** -0.8),
    ("1/n^0.6", lambda n, g: n ** -0.6),
    ("1/(1+(1-g)n)", lambda n, g: 1.0 / (1.0 + (1.0 - g) * n)),
    ("constant 0.1", lambda n, g: 0.1 * np.ones_like(n, dtype=float)),
]


def ex13(steps, runs, seed=0):
    S, A, g = 8, 3, 0.9
    rng = np.random.default_rng(seed)
    P, r = random_mdp(rng, S, A, conc=1.0)          # the same MDP as q_learning_rates.py
    q_star, _ = value_iteration(P, r, g)
    cumP = np.cumsum(P, axis=-1)
    cumP[..., -1] = 1.0
    K = len(SCHEDULES_ASYNC)
    rng = np.random.default_rng(seed + 5)
    Q = np.zeros((K, runs, S, A))
    N = np.zeros((runs, S, A))
    s = rng.integers(0, S, size=runs)
    ridx = np.arange(runs)
    report = sorted({n for n in (10_000, 100_000, 1_000_000) if n <= steps} | {steps})
    print(f"Ex. 13: asynchronous Q-learning, one trajectory, uniform behaviour, S={S}, A={A}, gamma={g}, "
          f"{runs} runs; step size indexed by the pair's own visit count n (MDP seed {seed}, sampling seed {seed + 5})")
    for t in range(1, steps + 1):
        a = rng.integers(0, A, size=runs)
        rew = (rng.random(runs) < r[s, a]).astype(float)
        s2 = (rng.random(runs)[:, None] > cumP[s, a]).sum(axis=1)
        N[ridx, s, a] += 1
        n = N[ridx, s, a]
        target = rew[None] + g * Q[:, ridx, s2].max(axis=-1)          # (K, runs)
        for k, (_, f) in enumerate(SCHEDULES_ASYNC):
            Q[k, ridx, s, a] += f(n, g) * (target[k] - Q[k, ridx, s, a])
        s = s2
        if t in report:
            err = np.abs(Q - q_star).max(axis=(2, 3)).mean(axis=1)
            print(f"   t = {t:>9,d}: " + ", ".join(f"{name}: {e:.3f}" for (name, _), e in zip(SCHEDULES_ASYNC, err)))


def lsvi_ucb_onehot(P, r, d0, H, K, R, beta, lam, rng):
    """LSVI-UCB with phi(s,a) = e_{(s,a)}: Lambda_h is diagonal, so the ridge regression is
    w_h(s,a) = (sum of targets at (s,a,h)) / (lam + n_h(s,a)) and the bonus is beta / sqrt(lam + n_h(s,a))."""
    _, S, A, _ = P.shape
    _, V_star = fh_optimal(P, r)
    v_star = V_star[0] @ d0
    n = np.zeros((R, H, S, A))
    n_next = np.zeros((R, H, S, A, S))
    rsum = np.zeros((R, H, S, A))
    cumP = np.cumsum(P, axis=-1)
    cumP[..., -1] = 1.0
    cumd0 = np.cumsum(d0)
    cumd0[-1] = 1.0
    runs = np.arange(R)
    regret = np.zeros((R, K))
    for k in range(K):
        Qt = np.zeros((R, H, S, A))
        V = np.zeros((R, S))
        for h in range(H - 1, -1, -1):
            # targets r + V_{h+1}(s') summed over the data at (s,a,h): rsum + n_next @ V
            w = (rsum[:, h] + np.einsum("rsat,rt->rsa", n_next[:, h], V)) / (lam + n[:, h])
            Qh = np.minimum(w + beta / np.sqrt(lam + n[:, h]), H - h)
            Qt[:, h] = Qh
            V = Qh.max(axis=2)
        greedy_a = np.argmax(Qt + rng.random(Qt.shape) * 1e-9, axis=-1)
        regret[:, k] = v_star - eval_policies(P, r, d0, greedy_a, np.zeros(R))
        s = (rng.random(R)[:, None] > cumd0[None]).sum(axis=1)
        for h in range(H):
            a = greedy_a[runs, h, s]
            rew = (rng.random(R) < r[h, s, a]).astype(float)
            s2 = (rng.random(R)[:, None] > cumP[h, s, a]).sum(axis=1)
            n[runs, h, s, a] += 1
            n_next[runs, h, s, a, s2] += 1
            rsum[runs, h, s, a] += rew
            s = s2
    return regret


def ex14(K, R, seed=0):
    print(f"Ex. 14: LSVI-UCB with one-hot features, K = {K} episodes, {R} runs "
          f"(MDPs from seeds {seed + 7} and {seed + 3}, sampling seed {seed + 11})")
    for env_name, env in (("random", make_random_env(seed + 7)), ("lock", make_lock_env(seed + 3))):
        P, r, d0, H = env
        d = P.shape[1] * P.shape[2]
        iota = math.log(2 * d * K * H / 0.05)
        for beta, lam, label in ((0.5, 1.0, "beta = 0.5, lambda = 1"), (2.0, 1.0, "beta = 2, lambda = 1"),
                                 (2.0, 0.01, "beta = 2, lambda = 0.01"),
                                 (d * H * math.sqrt(iota), 1.0, "beta = dH sqrt(iota), lambda = 1")):
            reg = lsvi_ucb_onehot(P, r, d0, H, K, R, beta, lam, np.random.default_rng(seed + 11))
            cum = reg.sum(axis=1)
            print(f"   {env_name:>6s} (d = {d}, H = {H}), {label:<33s} (beta = {beta:6.1f}): "
                  f"Reg(K) = {cum.mean():8.1f} ± {cum.std(ddof=1) / np.sqrt(R):6.1f}, "
                  f"regret/episode over the last 10% = {reg[:, -K // 10:].mean():.4f}")


def _stationary(P):
    """Stationary distribution mu (mu^T P = mu^T) of an ergodic chain, by a linear solve."""
    n = P.shape[0]
    M = np.vstack([P.T - np.eye(n), np.ones(n)])
    return np.linalg.lstsq(M, np.r_[np.zeros(n), 1.0], rcond=None)[0]


def td_lambda_ratio(P, r, X, g, lam, mu=None):
    """Exact TD(lambda) fixed point (Chapter 08, Section 4.5) on a Markov reward process.
    Returns ||X w_TD(lam) - v||_mu / ||Pi v - v||_mu and the residual of X w = Pi T^lam X w."""
    n = len(r)
    mu = _stationary(P) if mu is None else mu
    I = np.eye(n)
    D = np.diag(mu)
    v = np.linalg.solve(I - g * P, r)
    M = np.linalg.inv(I - g * lam * P)
    w = np.linalg.solve(X.T @ D @ M @ (I - g * P) @ X, X.T @ D @ M @ r)        # A_lam w = b_lam
    proj = X @ np.linalg.solve(X.T @ D @ X, X.T @ D)                            # Pi (mu-orthogonal)
    T_lam = lambda u: M @ (r + g * (1 - lam) * P @ u)                           # T^lam (Ch. 08, 4.5)
    nrm = lambda u: float(np.sqrt(mu @ u ** 2))
    resid = float(np.max(np.abs(X @ w - proj @ T_lam(X @ w))))
    return nrm(X @ w - v) / nrm(proj @ v - v), resid


def ex16(n_chains, n_search, seed=0):
    """How good is the TD(lambda) fixed point? Tsitsiklis-Van Roy's constant (1-g lam)/(1-g) =
    1/(1-g_lam) versus the Pythagorean constant 1/sqrt(1-g_lam^2), g_lam = g(1-lam)/(1-g lam)."""
    print(f"Ex. 16: error of the TD(lambda) fixed point, ||X w - v||_mu / ||Pi v - v||_mu "
          f"(sampling seed {seed + 13})")
    gammas, lams = (0.9, 0.99), (0.0, 0.5, 0.9)
    rng = np.random.default_rng(seed + 13)
    # (i) random ergodic chains, generated as in Chapter 08's linear_td_theory.py:
    #     30 states, 2 random successors per state mixed with 5% uniform jumps, 4 Gaussian features.
    n, d = 30, 4
    chains = []
    for _ in range(n_chains):
        P = np.zeros((n, n))
        for i in range(n):
            P[i, rng.choice(n, size=2, replace=False)] = rng.dirichlet(np.ones(2))
        P = 0.95 * P + 0.05 / n
        chains.append((P, rng.normal(size=n), rng.normal(size=(n, d))))
    print(f"   (i) {n_chains} random chains (S = {n}, d = {d}); columns: TvR constant (1-g lam)/(1-g), "
          f"Pythagorean constant 1/sqrt(1-g_lam^2), max observed ratio, violations of the Pythagorean bound")
    worst_resid = 0.0
    for g in gammas:
        for lam in lams:
            gl = g * (1 - lam) / (1 - g * lam)
            sharp, tvr = 1 / math.sqrt(1 - gl ** 2), 1 / (1 - gl)
            ratios = []
            for P, r, X in chains:
                q, res = td_lambda_ratio(P, r, X, g, lam)
                ratios.append(q)
                worst_resid = max(worst_resid, res)
            ratios = np.array(ratios)
            print(f"      gamma = {g:4.2f}, lambda = {lam:3.1f}: g_lam = {gl:.4f}, TvR {tvr:7.3f}, "
                  f"Pythagorean {sharp:6.3f}, max ratio {ratios.max():6.3f} "
                  f"({ratios.max() / sharp:.3f} of Pythagorean), violations {int(np.sum(ratios > sharp + 1e-9))}")
    print(f"      max residual of X w = Pi T^lam X w over all solves: {worst_resid:.1e}")
    # (ii) the lambda = 0 constant is attained: two states that swap deterministically (mu uniform),
    #      one feature x = (1, c) with 2c/(1+c^2) = gamma, and v orthogonal to x, so Pi v = 0.
    print("   (ii) tight example for lambda = 0: swap chain, x = (1, c), c = (1 - sqrt(1-g^2))/g, v = (-c, 1)")
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    for g in gammas:
        c = (1 - math.sqrt(1 - g ** 2)) / g
        v = np.array([-c, 1.0])
        r = (np.eye(2) - g * swap) @ v
        X = np.array([[1.0], [c]])
        q, _ = td_lambda_ratio(swap, r, X, g, 0.0, mu=np.array([0.5, 0.5]))
        eps = 1e-3                                       # aperiodic version: 0.1% uniform jumps
        q_eps, _ = td_lambda_ratio((1 - eps) * swap + eps / 2, r, X, g, 0.0)
        print(f"      gamma = {g}: c = {c:.4f}, ratio {q:.6f} vs 1/sqrt(1-g^2) = {1 / math.sqrt(1 - g ** 2):.6f}; "
              f"with {eps:.1%} uniform jumps: {q_eps:.6f}")
    # (iii) lambda > 0: random search over small, nearly deterministic cycles (the instances that
    #       made lambda = 0 tight) for the largest ratio. The same search at lambda = 0, where (ii)
    #       shows the constant is attained, calibrates it: it is run last so that it does not change
    #       the random stream of the lambda > 0 searches.
    def cycle_search(g, lam):
        best = 0.0
        for _ in range(n_search):
            m = int(rng.integers(2, 7))
            eps = 10 ** rng.uniform(-4, -1)
            P = (1 - eps) * np.roll(np.eye(m), 1, axis=1) + eps / m
            X = rng.normal(size=(m, int(rng.integers(1, m))))
            best = max(best, td_lambda_ratio(P, rng.normal(size=m), X, g, lam)[0])
        return best

    print(f"   (iii) lambda > 0: best ratio found by random search over {n_search} perturbed cycles "
          f"(2-6 states, jumps 1e-4..1e-1, random features and rewards)")
    for lam_set in (lams[1:], lams[:1]):
        if lam_set == lams[:1]:
            print("      calibration: the same search at lambda = 0, where the constant is attained (ii)")
        for g in gammas:
            for lam in lam_set:
                gl = g * (1 - lam) / (1 - g * lam)
                sharp = 1 / math.sqrt(1 - gl ** 2)
                best = cycle_search(g, lam)
                print(f"      gamma = {g:4.2f}, lambda = {lam:3.1f}: best {best:.3f} = {best / sharp:.4f} "
                      f"of the Pythagorean constant {sharp:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test")
    ap.add_argument("--seed", type=int, default=0, help="base seed for Ex. 13, 14 and 16 (Ex. 2, 5, 7, 10 are exact)")
    args = ap.parse_args()
    t0 = time.time()
    print(f"seed={args.seed} (Ex. 13: MDP {args.seed}, samples {args.seed + 5}; Ex. 14: MDPs {args.seed + 7} "
          f"and {args.seed + 3}, samples {args.seed + 11}; Ex. 16: chains {args.seed + 13})  quick={args.quick}")
    ex2()
    ex5()
    ex7()
    ex10()
    ex13(steps=20_000 if args.quick else 1_000_000, runs=2 if args.quick else 10, seed=args.seed)
    ex14(K=200 if args.quick else 5000, R=2 if args.quick else 5, seed=args.seed)
    ex16(n_chains=20 if args.quick else 300, n_search=100 if args.quick else 2000, seed=args.seed)
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
