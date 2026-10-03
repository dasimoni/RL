"""Numerical checks and reference code for the Chapter 07 exercises.

  * Exercise 3:  random walk in a 1-wide corridor with 4 actions; checks the
                 closed form h(0) = 2 L (L + 1) against a linear solve.
  * Exercise 5:  constant step-size sample updates; checks the noise floor
                 E[(Q - mu)^2] -> alpha / (2 - alpha) * sigma_b^2 by simulation.
  * Exercise 12: tic-tac-toe positions up to the 8 board symmetries, and UCT
                 with a symmetry-canonical transposition table.
  * Exercise 13: PUCT (AlphaZero-style selection, edge statistics at the parent,
                 all children expanded at once with prior probabilities) with a
                 uniform prior, an "oracle" prior that favours minimax-optimal
                 moves, and a misleading prior that favours mistakes.

Run:  python code/ch07_planning_and_learning_tabular/exercise_solutions.py [--quick]
"""

from __future__ import annotations

import argparse
import math
import random
import time

import numpy as np

from mcts_tictactoe import (UCT, Node, legal_moves, nontrivial_positions, optimal_moves, play,
                            random_rollout, reward_for, solver_sanity_checks, to_move, winner)


# ------------------------------------------------------------ Exercise 3
def corridor_hitting_time(L):
    """Expected steps from the left end of a 1 x L corridor to the goal just
    right of cell L-1, when each of 4 actions (up/down bump) is uniform."""
    A, b = np.eye(L), np.ones(L)
    for i in range(L):
        for nxt, p in ((max(i - 1, 0), 0.25), (i, 0.5)):    # left (bump at 0), up/down bumps
            A[i, nxt] -= p
        if i + 1 < L:                                         # right (into the goal from L-1)
            A[i, i + 1] -= 0.25
    return np.linalg.solve(A, b)[0]


# ------------------------------------------------------------ Exercise 5
def constant_alpha_noise_floor(alpha, b, n_steps, n_trials, rng):
    v = rng.standard_normal((n_trials, b))
    mu = v.mean(axis=1)
    sigma_b2 = v.var(axis=1)                                  # (1/b) sum (v - mu)^2
    Q = mu + 1.0                                              # initial error 1
    for _ in range(n_steps):
        Q = Q + alpha * (v[np.arange(n_trials), rng.integers(0, b, n_trials)] - Q)
    return np.mean((Q - mu) ** 2), np.mean(alpha / (2 - alpha) * sigma_b2)


# ------------------------------------------------------------ Exercise 12
def _symmetries():
    """The 8 symmetries of the 3x3 board as index permutations."""
    perms, grid = [], [[3 * r + c for c in range(3)] for r in range(3)]
    for _ in range(4):
        for flip in (False, True):
            g = [row[::-1] for row in grid] if flip else grid
            perms.append(tuple(v for row in g for v in row))
        grid = [list(row) for row in zip(*grid[::-1])]       # rotate by 90 degrees
    return perms


PERMS = _symmetries()


def canonical(board):
    return min(tuple(board[p[i]] for i in range(9)) for p in PERMS)


class SymmetricUCT(UCT):
    """UCT whose transposition table is keyed by the canonical board, so the
    8 symmetric versions of a position share one node. Nodes below the root
    hold canonical boards; root moves stay in the real board's coordinates."""

    def __init__(self, n_simulations, **kw):
        super().__init__(n_simulations, transpositions=True, **kw)

    def _child(self, node, move):
        key = canonical(play(node.board, move))
        child = self.table.get(key)
        if child is None:
            child = self.table[key] = Node(key, self.rng)
        node.children[move] = child          # symmetric root moves may share a child
        return child


# ------------------------------------------------------------ Exercise 13
class PNode:
    __slots__ = ("board", "terminal", "P", "N", "W", "children")

    def __init__(self, board):
        self.board, self.terminal = board, winner(board)
        self.P, self.N, self.W, self.children = None, {}, {}, {}


def puct_search(root_board, n_sim, prior_fn, c_puct=1.5, rng=None):
    """Select argmax_a Q(s,a) + c_puct P(s,a) sqrt(N(s) + 1) / (1 + N(s,a)).
    (The +1 lets the priors act on the very first selection at a node.)
    Unvisited edges get the neutral value Q = 1/2. A new leaf is expanded with
    priors for all its moves and evaluated by a random rollout (AlphaZero would
    use a value network here). Negamax credit, as in mcts_tictactoe.py."""
    rng = rng or random.Random(0)
    root = PNode(root_board)
    for _ in range(n_sim):
        node, path = root, []
        while node.P is not None and node.terminal is None:
            n_s = sum(node.N.values())

            def score(m, node=node, n_s=n_s):
                q = node.W[m] / node.N[m] if node.N[m] else 0.5
                return q + c_puct * node.P[m] * math.sqrt(n_s + 1) / (1 + node.N[m])
            m = max(node.P, key=score)
            path.append((node, m))
            if m not in node.children:
                node.children[m] = PNode(play(node.board, m))
            node = node.children[m]
        if node.terminal is None:
            node.P = prior_fn(node.board)
            node.N = {m: 0 for m in node.P}
            node.W = {m: 0.0 for m in node.P}
            result = random_rollout(node.board, rng)
        else:
            result = node.terminal
        for parent, m in path:
            parent.N[m] += 1
            parent.W[m] += reward_for(result, to_move(parent.board))
    return max(root.N, key=root.N.get)


def uniform_prior(board):
    moves = legal_moves(board)
    return {m: 1 / len(moves) for m in moves}


def solver_prior(board, mass_on_optimal):
    """Put `mass_on_optimal` of the prior on minimax-optimal moves (0.9 = an
    'oracle' prior, 0.1 = a misleading prior that favours mistakes)."""
    moves, opt = legal_moves(board), optimal_moves(board)
    bad = [m for m in moves if m not in opt]
    if not bad:
        return uniform_prior(board)
    return {m: (mass_on_optimal / len(opt) if m in opt else (1 - mass_on_optimal) / len(bad))
            for m in moves}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    n_positions = 50 if args.quick else 500
    budgets = [100] if args.quick else [100, 300]
    print(f"Chapter 07 exercise checks | seed={args.seed} positions={n_positions} budgets={budgets}")

    print("\nExercise 3: corridor hitting time, linear solve vs 2L(L+1)")
    for L in (1, 2, 5, 9, 20):
        print(f"  L={L:>2}: {corridor_hitting_time(L):8.1f}  vs  {2 * L * (L + 1)}")

    n_trials = 2000 if args.quick else 20000
    print(f"\nExercise 5: constant-alpha noise floor (b = 10, 2000 steps, {n_trials} trials)")
    rng = np.random.default_rng(args.seed)
    for alpha in (0.5, 0.1, 0.02):
        emp, th = constant_alpha_noise_floor(alpha, 10, 2000, n_trials, rng)
        print(f"  alpha={alpha:<5} simulated MSE {emp:.4f}   alpha/(2-alpha)*E[sigma_b^2] {th:.4f}")

    print("\nExercise 12: symmetry")
    positions, _ = solver_sanity_checks()
    print(f"  {len(set(PERMS))} distinct symmetries; {len(positions)} reachable positions, "
          f"{len({canonical(b) for b in positions})} up to symmetry")

    print("\nExercises 12-13: fraction of minimax-optimal moves on random non-trivial positions")
    sample = random.Random(args.seed + 1).sample(nontrivial_positions(positions), n_positions)
    agents = {
        "UCT (tree)": lambda b, n, r: UCT(n, rng=r).search(b),
        "UCT + symmetric transpositions": lambda b, n, r: SymmetricUCT(n, rng=r).search(b),
        "PUCT, uniform prior": lambda b, n, r: puct_search(b, n, uniform_prior, rng=r),
        "PUCT, prior 90% on optimal moves": lambda b, n, r: puct_search(b, n, lambda x: solver_prior(x, 0.9), rng=r),
        "PUCT, prior 10% on optimal moves": lambda b, n, r: puct_search(b, n, lambda x: solver_prior(x, 0.1), rng=r),
    }
    print(f"  {'agent':<34}" + "".join(f"{n:>8}" for n in budgets))
    for name, agent in agents.items():
        t0 = time.time()
        accs = [np.mean([agent(b, n, random.Random(10_000 * n + k)) in optimal_moves(b)
                         for k, b in enumerate(sample)]) for n in budgets]
        print(f"  {name:<34}" + "".join(f"{a:8.3f}" for a in accs) + f"   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
