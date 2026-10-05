"""Numerical checks for the Chapter 17 exercises.

  Ex. 17.6   fictitious play is not a no-regret algorithm (adversary in matching pennies)
  Ex. 17.7   gradient descent-ascent on f(x, y) = x y: simultaneous, alternating, optimistic
  Ex. 17.8   the value -1/18 of Kuhn poker, deal by deal, for the alpha = 0 equilibrium
  Ex. 17.10  variance of the COMA and central-critic gradient estimators at a general p
  Ex. 17.12  distributed (optimistic) Q-learning on the climbing, penalty and
             partially stochastic climbing games
  Ex. 17.13  independent Q-learning in the iterated prisoner's dilemma, against another
             learner and against tit-for-tat
  Ex. 17.14  Shapley value iteration on a two-state zero-sum Markov game (by hand, checked)
  Ex. 17.15  one iteration of vanilla CFR on Kuhn poker from the uniform profile: the
             counterfactual values and regrets at four information sets

Run:  python code/ch17_multi_agent_rl/exercise_solutions.py [--quick]
"""
import argparse
import time

import numpy as np

import games
import kuhn


# ---------------------------------------------------------------------- Ex. 17.6
def fp_vs_adversary(T):
    """Row player runs fictitious play in matching pennies; the column player knows FP's
    deterministic next action and always mismatches it."""
    A, _, _ = games.matching_pennies()
    col_counts = np.zeros(2)
    payoff = 0.0
    for t in range(T):
        freq = col_counts / col_counts.sum() if col_counts.sum() > 0 else np.full(2, 0.5)
        a = int(np.argmax(A @ freq))          # FP: best response (ties -> H)
        o = 1 - a                             # adversary mismatches: row gets -1
        payoff += A[a, o]
        col_counts[o] += 1
    best_fixed = np.max(A @ col_counts)
    return payoff / T, best_fixed / T, (best_fixed - payoff) / T


def rm_vs_adversary(T):
    """Regret matching (row) against an adversary who sees its mixed strategy x_t (but not
    its coin flip) and plays the column that minimises x_t^T A e_o."""
    A, _, _ = games.matching_pennies()
    R = np.zeros(2)
    col_counts = np.zeros(2)
    payoff = 0.0
    for t in range(T):
        pos = np.maximum(R, 0)
        x = pos / pos.sum() if pos.sum() > 0 else np.full(2, 0.5)
        o = int(np.argmin(x @ A))             # ties -> column 0
        u = A[:, o]
        payoff += x @ u
        R += u - x @ u
        col_counts[o] += 1
    best_fixed = np.max(A @ col_counts)
    return payoff / T, best_fixed / T, (best_fixed - payoff) / T


# ---------------------------------------------------------------------- Ex. 17.7
def gda(kind, eta, T, x0=1.0, y0=1.0):
    """x maximises and y minimises f(x, y) = x y; the unique equilibrium is (0, 0)."""
    x, y = x0, y0
    xp, yp = x0, y0                           # previous iterate (optimistic variant)
    for _ in range(T):
        if kind == "simultaneous":
            x, y = x + eta * y, y - eta * x
        elif kind == "alternating":
            x = x + eta * y
            y = y - eta * x
        elif kind == "optimistic":
            xn = x + 2 * eta * y - eta * yp
            yn = y - 2 * eta * x + eta * xp
            xp, yp, x, y = x, y, xn, yn
    return x, y


# ---------------------------------------------------------------------- Ex. 17.10
def coma_variance_check(p, N, n, rng):
    a = (rng.random((n, N)) < p).astype(float)
    s = a - p
    g_coma = s[:, 0] ** 2
    g_central = s.sum(1) * s[:, 0]
    return (g_coma.mean(), g_coma.var(), p * (1 - p) * (1 - 2 * p) ** 2,
            g_central.mean(), g_central.var(), p * (1 - p) * (1 - 2 * p) ** 2 + (N - 1) * (p * (1 - p)) ** 2)


# ---------------------------------------------------------------------- Ex. 17.12
def distributed_q(game_mean, stochastic, T, R, rng, eps=0.2):
    """Lauer & Riedmiller (2000): Q_i(a_i) <- max(Q_i(a_i), r), and the greedy action is
    changed only when the maximum of Q_i strictly increases (this coordinates the agents
    when there are several optimal joint actions)."""
    n = 3
    idx = np.arange(R)
    Q1 = np.full((R, n), -np.inf)
    Q2 = np.full((R, n), -np.inf)
    pol1 = rng.integers(n, size=R)
    pol2 = rng.integers(n, size=R)
    for t in range(T):
        a1 = np.where(rng.random(R) < eps, rng.integers(n, size=R), pol1)
        a2 = np.where(rng.random(R) < eps, rng.integers(n, size=R), pol2)
        r = game_mean[a1, a2].copy()
        if stochastic:
            bb = (a1 == 1) & (a2 == 1)
            r[bb] = np.where(rng.random(bb.sum()) < 0.5, 14.0, 0.0)
        old1, old2 = Q1.max(1), Q2.max(1)
        Q1[idx, a1] = np.maximum(Q1[idx, a1], r)
        Q2[idx, a2] = np.maximum(Q2[idx, a2], r)
        pol1 = np.where(Q1.max(1) > old1, np.argmax(Q1, 1), pol1)
        pol2 = np.where(Q2.max(1) > old2, np.argmax(Q2, 1), pol2)
    return pol1, pol2


# ---------------------------------------------------------------------- Ex. 17.13
def ipd(opponent, T, R, rng, alpha=0.1, gamma=0.9, eps_end=0.01):
    """Iterated prisoner's dilemma. State = previous joint action (4 states) plus a start
    state. Learner(s): tabular Q-learning, epsilon decaying from 1 to eps_end.
    opponent = 'IQL' (another independent learner) or 'TFT' (tit-for-tat)."""
    A, B, _ = games.prisoners_dilemma()       # actions 0 = C, 1 = D
    Q1 = np.zeros((R, 5, 2))
    Q2 = np.zeros((R, 5, 2))
    idx = np.arange(R)
    s = np.full(R, 4)                         # start state
    coop = []
    for t in range(T):
        eps = max(eps_end, 1.0 - t / (0.5 * T))
        g1 = np.argmax(Q1[idx, s] + 1e-9 * rng.random((R, 2)), 1)
        a1 = np.where(rng.random(R) < eps, rng.integers(2, size=R), g1)
        if opponent == "IQL":
            g2 = np.argmax(Q2[idx, s] + 1e-9 * rng.random((R, 2)), 1)
            a2 = np.where(rng.random(R) < eps, rng.integers(2, size=R), g2)
        else:                                 # TFT: copy the learner's previous action
            a2 = np.where(s == 4, 0, s // 2)
        r1, r2 = A[a1, a2], B[a1, a2]
        s2 = 2 * a1 + a2
        Q1[idx, s, a1] += alpha * (r1 + gamma * Q1[idx, s2].max(1) - Q1[idx, s, a1])
        if opponent == "IQL":
            Q2[idx, s, a2] += alpha * (r2 + gamma * Q2[idx, s2].max(1) - Q2[idx, s, a2])
        s = s2
        coop.append(np.mean(a1 == 0))
    # greedy play from the start for 20 rounds, no exploration
    s = np.full(R, 4)
    both_coop = np.zeros(R)
    for k in range(20):
        a1 = np.argmax(Q1[idx, s], 1)
        a2 = np.argmax(Q2[idx, s], 1) if opponent == "IQL" else np.where(s == 4, 0, s // 2)
        if k >= 10:
            both_coop += (a1 == 0) & (a2 == 0)
        s = 2 * a1 + a2
    return both_coop / 10, np.mean(coop[-1000:])


# ---------------------------------------------------------------------- Ex. 17.14
def shapley_two_state(gamma=0.9, sweeps=60):
    """State 2: one-shot game [[3, 0], [0, 1]] then terminal. State 1: reward 0;
    (a1, o1) -> state 2, (a1, o2) and (a2, o1) -> terminal with reward +1 for A,
    (a2, o2) -> stay in state 1. Synchronous Shapley value iteration (Alg. 4.1) from V = 0."""
    V = np.zeros(2)                           # V[0] = state 1, V[1] = state 2
    hist = []
    for _ in range(sweeps):
        Q1 = np.array([[gamma * V[1], 1.0], [1.0, gamma * V[0]]])
        Q2 = np.array([[3.0, 0.0], [0.0, 1.0]])
        x1, y1, v1 = games.solve_zero_sum(Q1)
        x2, y2, v2 = games.solve_zero_sum(Q2)
        V = np.array([v1, v2])
        hist.append((V.copy(), x1, y1, x2, y2))
    return hist


def main():
    parser = argparse.ArgumentParser(description="Chapter 17 exercise checks")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    q = args.quick
    print(f"exercise_solutions.py | seed={args.seed} | quick={q}")
    t0 = time.time()

    T = 2_000 if q else 100_000
    print(f"\n[Ex 17.6] matching pennies, T = {T}")
    u, b, reg = fp_vs_adversary(T)
    print(f"  fictitious play vs adversary : avg payoff {u:+.3f}, best fixed action {b:+.4f}, avg regret {reg:.3f}")
    u, b, reg = rm_vs_adversary(T)
    print(f"  regret matching vs adversary : avg payoff {u:+.4f}, best fixed action {b:+.4f}, avg regret {reg:.4f}")

    print("\n[Ex 17.7] GDA on f(x, y) = x y from (1, 1), eta = 0.1")
    for steps in (100, 1000):
        for kind in ("simultaneous", "alternating", "optimistic"):
            x, y = gda(kind, 0.1, steps)
            extra = f", invariant x^2 + eta x y + y^2 = {x * x + 0.1 * x * y + y * y:.4f}" if kind == "alternating" else ""
            print(f"  {steps:5d} steps {kind:12s}: |(x, y)| = {np.hypot(x, y):.4e}{extra}")
    print(f"  predicted |(x,y)| simultaneous after 1000 steps: sqrt(2) * (1 + 0.01)^500 = {np.sqrt(2) * 1.01 ** 500:.4e}")
    s_ = np.sqrt(1 - 4 * 0.01)
    print(f"  optimistic: spectral radius sqrt((1 + sqrt(1 - 4 eta^2)) / 2) = {np.sqrt((1 + s_) / 2):.5f}")

    print("\n[Ex 17.8] Kuhn poker, alpha = 0 equilibrium, value of each deal for player 1")
    v, s1, s2 = kuhn.solve_sequence_form_lp()
    alpha = 0.0
    f1 = {"J:": np.array([1 - alpha, alpha]), "Q:": np.array([1.0, 0.0]),
          "K:": np.array([1 - 3 * alpha, 3 * alpha]), "J:pb": np.array([1.0, 0.0]),
          "Q:pb": np.array([2 / 3 - alpha, alpha + 1 / 3]), "K:pb": np.array([0.0, 1.0])}
    total = 0.0
    for c1, c2 in kuhn.DEALS:
        val = kuhn._value(c1, c2, "", f1, s2)
        total += val / 6
        print(f"  P1 {kuhn.CARDS[c1]} vs P2 {kuhn.CARDS[c2]}: {val:+.4f}")
    print(f"  average = {total:+.6f}  (-1/18 = {-1 / 18:+.6f})")

    n = 20_000 if q else 1_000_000
    print(f"\n[Ex 17.10] COMA vs central-critic estimator, p = 0.8, N = 8, {n} samples")
    m1, v1, f1v, m2, v2, f2v = coma_variance_check(0.8, 8, n, rng)
    print(f"  COMA          : mean {m1:.4f} (p(1-p) = 0.16), var {v1:.5f} (formula {f1v:.5f})")
    print(f"  central critic: mean {m2:.4f}, var {v2:.5f} (formula {f2v:.5f})")

    R = 200 if q else 1000
    T = 500 if q else 3000
    print(f"\n[Ex 17.12] distributed Q-learning, {R} runs x {T} steps, epsilon = 0.2")
    climb, _, _ = games.climbing_game()
    pen, _, _ = games.penalty_game(-100.0)
    for name, M, stoch, opt in (("climbing", climb, False, [(0, 0)]), ("penalty (k=-100)", pen, False, [(0, 0), (2, 2)]),
                                ("stochastic climbing", climb, True, [(0, 0)])):
        p1, p2 = distributed_q(M, stoch, T, R, np.random.default_rng(args.seed + 5))
        freq = {}
        for a, b_ in zip(p1, p2):
            k = "abc"[a] + "abc"[b_]
            freq[k] = freq.get(k, 0) + 1
        p_opt = np.mean([(a, b_) in opt for a, b_ in zip(p1, p2)])
        print(f"  {name:20s}: P(optimal) = {p_opt:.3f}; final joint actions "
              + ", ".join(f"{k}: {100 * v / R:.1f}%" for k, v in sorted(freq.items(), key=lambda kv: -kv[1])))

    R = 200 if q else 500
    T = 3_000 if q else 30_000
    print(f"\n[Ex 17.13] iterated prisoner's dilemma, Q-learning (alpha 0.1, gamma 0.9), {R} runs x {T} rounds")
    for opp in ("TFT", "IQL"):
        bc, c_last = ipd(opp, T, R, np.random.default_rng(args.seed + 9))
        print(f"  learner vs {opp}: fraction of runs whose greedy play is mutual cooperation "
              f"(rounds 11-20 from the start): {np.mean(bc == 1.0):.3f}; mean cooperation rate "
              f"{bc.mean():.3f}; learner's cooperation rate in the last 1000 training rounds {c_last:.3f}")
    print("\n[Ex 17.14] Shapley value iteration on the two-state game, gamma = 0.9, from V = 0")
    hist = shapley_two_state()
    v_star = hist[-1][0]
    for k in range(4):
        V = hist[k][0]
        print(f"  sweep {k + 1}: V(1) = {V[0]:.5f}, V(2) = {V[1]:.5f}, error in state 1 = {v_star[0] - V[0]:.5f}")
    root = (1.9325 - np.sqrt(1.9325 ** 2 - 3.6)) / 1.8
    _, x1, y1, x2, y2 = hist[-1]
    print(f"  fixed point: v*(1) = {v_star[0]:.6f} (root of 0.9 v^2 - 1.9325 v + 1 = 0: {root:.6f}), "
          f"v*(2) = {v_star[1]:.4f}")
    print(f"  state 1 strategies: A plays a1 w.p. {x1[0]:.4f}, B plays o1 w.p. {y1[0]:.4f}; "
          f"state 2: A plays a1 w.p. {x2[0]:.4f}, B plays o1 w.p. {y2[0]:.4f}")
    errs = [v_star[0] - h[0][0] for h in hist[:8]]
    print("  error ratios e_{k+1}/e_k: " + ", ".join(f"{errs[k + 1] / errs[k]:.3f}" for k in range(6))
          + f"; local rate gamma * x(a2) * y(o2) = {0.9 * x1[1] * y1[1]:.3f}; global bound gamma = 0.9")

    print("\n[Ex 17.15 and the worked example of section 10.3] one vanilla CFR iteration from uniform")
    import cfr_kuhn
    solver = cfr_kuhn.CFRSolver("cfr")
    solver.iterate()
    sig2 = solver.current_strategy()
    for I in ("K:", "K:pb", "Q:b", "J:p"):
        print(f"  {I:5s} cumulative regret (pass/check/fold, bet/call) = {np.round(solver.regret[I], 6)}; "
              f"strategy for iteration 2 = {np.round(sig2[I], 3)}")

    print(f"\ntotal time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
