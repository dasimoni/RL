"""Normal-form game toolkit for Chapter 17 (sections 2-3), with a self-test.

Conventions
-----------
A two-player game is a pair of payoff matrices (A, B): when the row player plays
action i and the column player plays action j, the row player receives A[i, j] and
the column player receives B[i, j]. Mixed strategies are probability vectors x (over
rows) and y (over columns), so expected payoffs are x^T A y and x^T B y.
A zero-sum game has B = -A and is described by A alone.

What is in here
---------------
* a catalogue of the games used in the chapter (matching pennies, rock-paper-scissors,
  weighted RPS, Shapley's game, prisoner's dilemma, stag hunt, chicken, climbing game,
  penalty game);
* best responses and NashConv / exploitability (section 3.5);
* zero-sum games solved by linear programming (minimax theorem, section 3.2), plus a
  small, fast tableau-simplex solver used inside Shapley value iteration and minimax-Q;
* all Nash equilibria of a small bimatrix game by support enumeration (section 3.1);
* correlated / coarse correlated equilibria by linear programming and their gaps
  (section 3.3);
* pure-outcome Pareto optimality (section 3.4).

Run `python code/ch17_multi_agent_rl/games.py` to print the solution concepts of every
game in the catalogue (the numbers quoted in sections 2-3 of the chapter).
"""
import argparse
import itertools
import time

import numpy as np
from scipy.optimize import linprog


# ----------------------------------------------------------------------------------
# Game catalogue
# ----------------------------------------------------------------------------------
def matching_pennies():
    A = np.array([[1.0, -1.0], [-1.0, 1.0]])
    return A, -A, ["H", "T"]


def rock_paper_scissors():
    A = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    return A, -A, ["R", "P", "S"]


def weighted_rps():
    """RPS in which Rock beating Scissors pays 2. Unique Nash: (1/4, 1/2, 1/4)."""
    A = np.array([[0.0, -1.0, 2.0], [1.0, 0.0, -1.0], [-2.0, 1.0, 0.0]])
    return A, -A, ["R", "P", "S"]


def shapley_game():
    """Shapley (1964): general-sum 3x3 game on which fictitious play cycles forever."""
    A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    B = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return A, B, ["0", "1", "2"]


def prisoners_dilemma():
    # actions: Cooperate, Defect. Payoffs T=5 > R=3 > P=1 > S=0.
    A = np.array([[3.0, 0.0], [5.0, 1.0]])
    return A, A.T.copy(), ["C", "D"]


def stag_hunt():
    # Stag needs both hunters; Hare is safe.
    A = np.array([[4.0, 0.0], [3.0, 3.0]])
    return A, A.T.copy(), ["Stag", "Hare"]


def chicken():
    """Aumann's (1974) chicken example (often told as a traffic light). Actions: Chicken (yield), Dare."""
    A = np.array([[6.0, 2.0], [7.0, 0.0]])
    return A, A.T.copy(), ["C", "D"]


def climbing_game():
    """Claus & Boutilier (1998): fully cooperative, both agents receive A."""
    A = np.array([[11.0, -30.0, 0.0], [-30.0, 7.0, 6.0], [0.0, 0.0, 5.0]])
    return A, A.copy(), ["a", "b", "c"]


def penalty_game(k=-100.0):
    """Claus & Boutilier (1998) penalty game, k <= 0. Both agents receive A."""
    A = np.array([[10.0, 0.0, k], [0.0, 2.0, 0.0], [k, 0.0, 10.0]])
    return A, A.copy(), ["a", "b", "c"]


CATALOGUE = {
    "matching pennies": matching_pennies,
    "rock-paper-scissors": rock_paper_scissors,
    "weighted RPS": weighted_rps,
    "Shapley's game": shapley_game,
    "prisoner's dilemma": prisoners_dilemma,
    "stag hunt": stag_hunt,
    "chicken": chicken,
    "climbing game": climbing_game,
    "penalty game (k=-100)": penalty_game,
}


# ----------------------------------------------------------------------------------
# Best responses and exploitability (section 3.5)
# ----------------------------------------------------------------------------------
def best_response_gains(A, B, x, y):
    """How much each player gains by switching to a best response (both >= 0)."""
    gain_row = np.max(A @ y) - x @ A @ y
    gain_col = np.max(x @ B) - x @ B @ y
    return gain_row, gain_col


def nash_conv(A, B, x, y):
    """NashConv(x, y) = sum of best-response gains. Zero iff (x, y) is a Nash equilibrium.

    For a two-player zero-sum game (B = -A) this equals max_i (A y)_i - min_j (x^T A)_j,
    the 'duality gap'; the chapter's exploitability is NashConv / 2.
    """
    g1, g2 = best_response_gains(A, B, x, y)
    return g1 + g2


# ----------------------------------------------------------------------------------
# Zero-sum games: linear programming (section 3.2)
# ----------------------------------------------------------------------------------
def _maximin_lp(A):
    """max_x min_j (x^T A)_j as an LP in the variables (x, v)."""
    m, n = A.shape
    c = np.zeros(m + 1)
    c[-1] = -1.0                                   # maximise v  <=> minimise -v
    A_ub = np.hstack([-A.T, np.ones((n, 1))])     # v - (x^T A)_j <= 0 for every column j
    b_ub = np.zeros(n)
    A_eq = np.hstack([np.ones((1, m)), np.zeros((1, 1))])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * m + [(None, None)], method="highs")
    assert res.success, res.message
    x = np.clip(res.x[:m], 0, None)
    return x / x.sum(), res.x[-1]


def solve_zero_sum(A):
    """Return (x, y, v): maximin strategy of the row player, minimax strategy of the
    column player, and the value v = max_x min_y x^T A y = min_y max_x x^T A y."""
    x, v_row = _maximin_lp(A)
    y, v_col = _maximin_lp(-A.T)                 # column player maximises -x^T A y
    assert abs(v_row + v_col) < 1e-7, (v_row, v_col)  # minimax theorem: the two LPs agree
    return x, y, v_row


def solve_zero_sum_pivot(A, tol=1e-12):
    """Same answer as solve_zero_sum, by a hand-written tableau simplex (Bland's rule).

    About 20x faster than scipy's linprog on tiny games because it has no setup
    overhead; used thousands of times by Shapley value iteration and minimax-Q.
    Method (e.g. Ferguson, 'Game Theory', Part II sec. 4): shift A so all entries are
    positive, solve  max 1^T u  s.t.  A' u <= 1, u >= 0; then the value is 1/sum(u)
    minus the shift, the column strategy is u/sum(u), and the row strategy comes from
    the dual variables (the reduced costs of the slack columns).
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    shift = 1.0 - A.min()
    T = np.zeros((m + 1, n + m + 1))
    T[:m, :n] = A + shift
    T[:m, n:n + m] = np.eye(m)
    T[:m, -1] = 1.0
    T[m, :n] = -1.0
    basis = list(range(n, n + m))
    for _ in range(50 * (m + n)):
        neg = np.nonzero(T[m, :-1] < -tol)[0]
        if neg.size == 0:
            break
        col = neg[0]                               # Bland: smallest improving index
        colv = T[:m, col]
        pos = colv > tol
        ratios = np.full(m, np.inf)
        ratios[pos] = T[:m, -1][pos] / colv[pos]
        rmin = ratios.min()
        cands = np.nonzero(ratios <= rmin + 1e-12)[0]
        row = min(cands, key=lambda r: basis[r])   # Bland tie-break on leaving variable
        T[row] /= T[row, col]
        for r in range(m + 1):
            if r != row and T[r, col] != 0.0:
                T[r] -= T[r, col] * T[row]
        basis[row] = col
    z = T[m, -1]
    u = np.zeros(n + m)
    for r, b in enumerate(basis):
        u[b] = T[r, -1]
    y = np.clip(u[:n], 0, None) / z
    x = np.clip(T[m, n:n + m], 0, None) / z
    return x / x.sum(), y / y.sum(), 1.0 / z - shift


# ----------------------------------------------------------------------------------
# All Nash equilibria of a small bimatrix game (section 3.1)
# ----------------------------------------------------------------------------------
def _indifferent_mix(M, rows, cols):
    """Solve for a mix z on `cols` (summing to 1) that makes every row in `rows` of M
    earn the same value v. Returns (z_full, v) or None."""
    k = len(cols)
    lhs = np.zeros((len(rows) + 1, k + 1))
    lhs[:len(rows), :k] = M[np.ix_(rows, cols)]
    lhs[:len(rows), k] = -1.0
    lhs[len(rows), :k] = 1.0
    rhs = np.zeros(len(rows) + 1)
    rhs[-1] = 1.0
    sol, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    if np.linalg.norm(lhs @ sol - rhs) > 1e-9:
        return None
    z = np.zeros(M.shape[1])
    z[list(cols)] = sol[:k]
    return z, sol[k]


def support_enumeration(A, B, tol=1e-9):
    """All Nash equilibria of a nondegenerate bimatrix game with equal-size supports.

    For every pair of supports (I, J) with |I| = |J|: choose y on J to make the row
    player indifferent on I, x on I to make the column player indifferent on J, and
    keep the pair if both are probability vectors and no action outside the
    supports does strictly better.
    """
    m, n = A.shape
    eqs = []
    for k in range(1, min(m, n) + 1):
        for I in itertools.combinations(range(m), k):
            for J in itertools.combinations(range(n), k):
                ry = _indifferent_mix(A, I, J)
                rx = _indifferent_mix(B.T, J, I)
                if ry is None or rx is None:
                    continue
                (y, v1), (x, v2) = ry, rx
                if (y < -tol).any() or (x < -tol).any():
                    continue
                if np.max(A @ y) > v1 + tol or np.max(x @ B) > v2 + tol:
                    continue
                x, y = np.clip(x, 0, None), np.clip(y, 0, None)
                if not any(np.allclose(x, e[0]) and np.allclose(y, e[1]) for e in eqs):
                    eqs.append((x, y))
    return eqs


# ----------------------------------------------------------------------------------
# Correlated and coarse correlated equilibria (section 3.3)
# ----------------------------------------------------------------------------------
def correlated_equilibrium(A, B, objective=None, coarse=False, maximize=True):
    """Optimise a linear objective over the set of CE (or CCE) of a bimatrix game.

    Variables: a joint distribution P[i, j] >= 0 summing to one. Constraints:
      CE : for every recommended i and deviation i':  sum_j P[i,j] (A[i,j] - A[i',j]) >= 0
           (and the analogue for the column player);
      CCE: for every deviation i':  sum_{i,j} P[i,j] (A[i,j] - A[i',j]) >= 0  (and analogue).
    Default objective: social welfare sum_{ij} P[i,j] (A + B)[i,j].
    """
    m, n = A.shape
    obj = (A + B) if objective is None else objective
    rows = []
    if coarse:
        for d in range(m):
            rows.append((A[d][None, :] - A).ravel())         # deviation gain <= 0
        for d in range(n):
            rows.append((B[:, d][:, None] - B).ravel())
    else:
        for i in range(m):
            for d in range(m):
                if d != i:
                    g = np.zeros((m, n))
                    g[i] = A[d] - A[i]
                    rows.append(g.ravel())
        for j in range(n):
            for d in range(n):
                if d != j:
                    g = np.zeros((m, n))
                    g[:, j] = B[:, d] - B[:, j]
                    rows.append(g.ravel())
    sign = -1.0 if maximize else 1.0
    res = linprog(sign * obj.ravel(), A_ub=np.array(rows), b_ub=np.zeros(len(rows)),
                  A_eq=np.ones((1, m * n)), b_eq=[1.0], bounds=[(0, None)] * (m * n),
                  method="highs")
    assert res.success, res.message
    P = np.clip(res.x, 0, None).reshape(m, n)
    return P / P.sum()


def cce_gap(A, B, P):
    """Largest gain any player gets by committing to a fixed action instead of
    following the joint recommendation P (<= 0 means P is a CCE)."""
    gain_row = np.max(A @ P.sum(axis=0)) - np.sum(P * A)
    gain_col = np.max(P.sum(axis=1) @ B) - np.sum(P * B)
    return max(gain_row, gain_col)


def ce_gap(A, B, P):
    """Largest gain any player gets from the best swap function phi(i) (>= 0;
    zero means P is a correlated equilibrium)."""
    gain_row = sum(max(0.0, np.max(A @ P[i]) - A[i] @ P[i]) for i in range(A.shape[0]))
    gain_col = sum(max(0.0, np.max(P[:, j] @ B) - P[:, j] @ B[:, j]) for j in range(A.shape[1]))
    return max(gain_row, gain_col)


# ----------------------------------------------------------------------------------
# Pareto optimality of pure outcomes (section 3.4)
# ----------------------------------------------------------------------------------
def pareto_optimal_pure(A, B):
    outcomes = [(i, j) for i in range(A.shape[0]) for j in range(A.shape[1])]
    po = []
    for (i, j) in outcomes:
        dominated = any(A[k, l] >= A[i, j] and B[k, l] >= B[i, j] and
                        (A[k, l] > A[i, j] or B[k, l] > B[i, j]) for (k, l) in outcomes)
        if not dominated:
            po.append((i, j))
    return po


# ----------------------------------------------------------------------------------
# Self-test / report
# ----------------------------------------------------------------------------------
def _fmt(v):
    return "(" + ", ".join(f"{a:.3f}" for a in v) + ")"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test (fewer random checks)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    print(f"games.py self-test | seed={args.seed} quick={args.quick}")

    # 1. The pivot solver agrees with the LP solver on random zero-sum games.
    n_checks = 50 if args.quick else 500
    worst = 0.0
    t_lp = t_piv = 0.0
    for _ in range(n_checks):
        m, n = rng.integers(2, 6, size=2)
        A = rng.integers(-3, 4, size=(m, n)).astype(float)   # integer entries: many ties
        t0 = time.perf_counter(); x1, y1, v1 = solve_zero_sum(A); t_lp += time.perf_counter() - t0
        t0 = time.perf_counter(); x2, y2, v2 = solve_zero_sum_pivot(A); t_piv += time.perf_counter() - t0
        worst = max(worst, abs(v1 - v2), nash_conv(A, -A, x2, y2))
    print(f"\n[pivot vs LP] {n_checks} random integer games up to 5x5: "
          f"max |v_LP - v_pivot| or NashConv(pivot) = {worst:.2e}; "
          f"time per game LP {1e3 * t_lp / n_checks:.2f} ms, pivot {1e3 * t_piv / n_checks:.2f} ms")
    assert worst < 1e-8

    # 2. Solution concepts of every catalogue game.
    for name, make in CATALOGUE.items():
        A, B, acts = make()
        print(f"\n=== {name} ===  actions {acts}")
        print("row payoffs A =\n" + np.array2string(A, precision=0))
        if np.allclose(A, -B):
            x, y, v = solve_zero_sum(A)
            print(f"zero-sum: value {v:+.4f}, maximin x = {_fmt(x)}, minimax y = {_fmt(y)}")
        else:
            print("col payoffs B =\n" + np.array2string(B, precision=0))
        for x, y in support_enumeration(A, B):
            print(f"  Nash: x = {_fmt(x)}, y = {_fmt(y)}  payoffs ({x @ A @ y:+.3f}, {x @ B @ y:+.3f})")
        if not np.allclose(A, -B):
            P = correlated_equilibrium(A, B)
            print(f"  max-welfare CE : payoffs ({np.sum(P * A):.3f}, {np.sum(P * B):.3f}), "
                  f"P = {np.array2string(P.ravel(), precision=3)}")
            Pc = correlated_equilibrium(A, B, coarse=True)
            print(f"  max-welfare CCE: payoffs ({np.sum(Pc * A):.3f}, {np.sum(Pc * B):.3f}), "
                  f"P = {np.array2string(Pc.ravel(), precision=3)}")
            po = pareto_optimal_pure(A, B)
            print(f"  Pareto-optimal pure outcomes: {[(acts[i], acts[j]) for i, j in po]}")

    # 3. A CCE that is not a CE (worked example in section 3.3): in RPS, the uniform
    #    distribution over the 6 off-diagonal outcomes; and Aumann's CE in chicken.
    A, B, _ = rock_paper_scissors()
    P = np.ones((3, 3)) - np.eye(3)
    P /= P.sum()
    print(f"\n[RPS] uniform over the 6 off-diagonal outcomes: CCE gap {cce_gap(A, B, P):+.3f}, "
          f"CE gap {ce_gap(A, B, P):+.3f}")
    A, B, _ = chicken()
    P = np.array([[1 / 3, 1 / 3], [1 / 3, 0.0]])
    print(f"[chicken] uniform over (C,C),(C,D),(D,C): CE gap {ce_gap(A, B, P):+.3f}, "
          f"payoffs ({np.sum(P * A):.3f}, {np.sum(P * B):.3f})")
    print("\nall self-tests passed")


if __name__ == "__main__":
    main()
