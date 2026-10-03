"""Monte Carlo Tree Search (UCT) for tic-tac-toe, verified against an exact solver.

Chapter 07, Sections 10-12.

  * An exact negamax solver (memoised minimax) labels every reachable position
    with its game-theoretic value and its set of optimal moves. We sanity-check
    it with the well-known counts of reachable positions and complete games.
  * UCT (Kocsis & Szepesvari 2006): selection by UCB1, expansion of one new
    node per simulation, uniformly random rollout, NEGAMAX backup (each node
    stores the value from the point of view of the player who moved into it).
    Two variants: a plain tree, and a DAG that merges transpositions (the same
    position reached by different move orders) through a hash table.
  * The rollout algorithm of Section 10 ("flat Monte Carlo"): split the budget
    evenly over the root moves, estimate each by random rollouts, pick the best.

Experiments
  1. Fraction of minimax-optimal moves chosen, vs simulation budget, on a random
     sample of non-trivial positions (positions in which some legal move is a
     mistake).
  2. Games: UCT vs a uniformly random player, UCT vs UCT, and UCT vs a perfect
     (minimax) player. Every UCT move is also checked against the solver.

Rewards are in [0, 1] (win 1, draw 1/2, loss 0), so UCB1's constant c = sqrt(2)
applies unchanged and the negamax flip of a value is v -> 1 - v.

Run:  python code/ch07_planning_and_learning_tabular/mcts_tictactoe.py [--quick]
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
EMPTY = (0,) * 9
X, O = 1, -1


# ---------------------------------------------------------------- game rules
def to_move(board):
    """X moves first, so X is to move iff the number of stones is even."""
    return X if sum(1 for v in board if v) % 2 == 0 else O


def winner(board):
    """+1 / -1 if X / O has three in a row, 0 for a full-board draw, None if
    the game is still running."""
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return 0 if all(board) else None


def legal_moves(board):
    return [i for i in range(9) if board[i] == 0]


def play(board, move):
    b = list(board)
    b[move] = to_move(board)
    return tuple(b)


# ------------------------------------------------------------- exact solver
@lru_cache(maxsize=None)
def negamax(board):
    """Game value for the player TO MOVE: +1 win, 0 draw, -1 loss, under
    perfect play by both sides (Section 12.1: v(s) = max_a -v(child))."""
    w = winner(board)
    if w is not None:
        # The game ended on the opponent's move, so the mover can only have lost or drawn.
        return 0 if w == 0 else -1
    return max(-negamax(play(board, m)) for m in legal_moves(board))


@lru_cache(maxsize=None)
def random_play_value(board):
    """E[result] (+1 X wins, -1 O wins, 0 draw) when BOTH sides play uniformly
    at random from `board`: the quantity flat Monte Carlo estimates."""
    w = winner(board)
    if w is not None:
        return float(w)
    moves = legal_moves(board)
    return sum(random_play_value(play(board, m)) for m in moves) / len(moves)


def flat_mc_limit_move(board):
    """The move flat Monte Carlo converges to with infinitely many rollouts:
    the best move for the mover if everybody plays randomly afterwards."""
    p = to_move(board)
    return max(legal_moves(board), key=lambda m: p * random_play_value(play(board, m)))


def optimal_moves(board):
    v = negamax(board)
    return {m for m in legal_moves(board) if -negamax(play(board, m)) == v}


def solver_sanity_checks():
    """Count reachable positions and complete games (perft-style check)."""
    seen, stack = {EMPTY}, [EMPTY]
    while stack:
        b = stack.pop()
        if winner(b) is None:
            for m in legal_moves(b):
                c = play(b, m)
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
    games = {X: 0, O: 0, 0: 0}

    def count(b):
        w = winner(b)
        if w is not None:
            games[w] += 1
            return
        for m in legal_moves(b):
            count(play(b, m))
    count(EMPTY)
    return seen, games


# --------------------------------------------------------------------- MCTS
class Node:
    __slots__ = ("board", "player", "terminal", "untried", "children", "N", "W")

    def __init__(self, board, rng):
        self.board = board
        self.player = to_move(board)          # player to move AT this node
        self.terminal = winner(board)
        self.untried = [] if self.terminal is not None else legal_moves(board)
        rng.shuffle(self.untried)             # expand in random order
        self.children = {}                    # move -> Node
        self.N = 0                            # visit count
        self.W = 0.0                          # total reward for the player who moved INTO this node


def reward_for(result, player):
    """Map a game result (+1 X wins, -1 O wins, 0 draw) to [0, 1] for `player`."""
    return 0.5 if result == 0 else (1.0 if result == player else 0.0)


def random_rollout(board, rng):
    """Default (rollout) policy: uniformly random legal moves to the end."""
    w = winner(board)
    b = list(board)
    player = to_move(board)
    while w is None:
        empties = [i for i in range(9) if b[i] == 0]
        b[rng.choice(empties)] = player
        player = -player
        w = winner(b)
    return w


class UCT:
    """UCT with negamax backups. transpositions=True merges identical positions
    into one node (a DAG); statistics are then shared by all move orders that
    lead to a position, and backups follow the path actually traversed."""

    def __init__(self, n_simulations, c=math.sqrt(2), transpositions=False, rng=None):
        self.n_sim, self.c, self.transpositions = n_simulations, c, transpositions
        self.rng = rng or random.Random(0)

    def _child(self, node, move):
        board = play(node.board, move)
        if self.transpositions:
            child = self.table.get(board)
            if child is None:
                child = self.table[board] = Node(board, self.rng)
        else:
            child = Node(board, self.rng)
        node.children[move] = child
        return child

    def _select(self, node):
        """UCB1 over children; Q is from the point of view of node.player
        because each child's W was accumulated for the player who moved into it."""
        log_n = math.log(node.N)
        best, best_score = None, -1.0
        for child in node.children.values():
            score = child.W / child.N + self.c * math.sqrt(log_n / child.N)
            if score > best_score:
                best, best_score = child, score
        return best

    def search(self, board):
        self.table = {}
        root = Node(board, self.rng)
        if self.transpositions:
            self.table[board] = root
        for _ in range(self.n_sim):
            node, path = root, [root]
            # 1. SELECTION (+ 2. EXPANSION of one new node)
            while node.terminal is None:
                if node.untried:
                    node = self._child(node, node.untried.pop())
                    path.append(node)
                    if node.N == 0:           # a genuinely new node: stop and simulate
                        break
                    # transposition: node already has statistics, keep descending
                else:
                    node = self._select(node)
                    path.append(node)
            # 3. SIMULATION (rollout) from the new node, or the terminal result
            result = node.terminal if node.terminal is not None else random_rollout(node.board, self.rng)
            # 4. BACKUP (negamax): each node is credited from the viewpoint of the
            #    player who moved into it, i.e. the player to move at its parent.
            for n in path:
                n.N += 1
                n.W += reward_for(result, -n.player)
        self.root = root
        # Final decision: the most-visited ("robust") child.
        return max(root.children.items(), key=lambda kv: kv[1].N)[0]


class FlatMonteCarlo:
    """The rollout algorithm of Section 10: Monte Carlo estimates of each root
    action under a uniformly random rollout policy, with an even budget split."""

    def __init__(self, n_simulations, rng=None):
        self.n_sim = n_simulations
        self.rng = rng or random.Random(0)

    def search(self, board):
        me = to_move(board)
        moves = legal_moves(board)
        per_move = max(1, self.n_sim // len(moves))
        best, best_value = None, -1.0
        for m in moves:
            child = play(board, m)
            total = sum(reward_for(random_rollout(child, self.rng), me) for _ in range(per_move))
            if total / per_move > best_value:
                best, best_value = m, total / per_move
        return best


# -------------------------------------------------------------- experiments
def nontrivial_positions(all_positions):
    """Non-terminal positions where at least one legal move is NOT optimal."""
    out = []
    for b in all_positions:
        if winner(b) is None and len(optimal_moves(b)) < len(legal_moves(b)):
            out.append(b)
    return sorted(out)


def accuracy_experiment(positions, budgets, rng_seed):
    methods = {
        "flat Monte Carlo (rollout algorithm)": lambda n, r: FlatMonteCarlo(n, r),
        "UCT (tree)": lambda n, r: UCT(n, rng=r),
        "UCT + transposition table": lambda n, r: UCT(n, transpositions=True, rng=r),
    }
    acc = {name: [] for name in methods}
    for name, make in methods.items():
        for n in budgets:
            rng = random.Random(rng_seed + n)
            agent = make(n, rng)
            correct = sum(agent.search(b) in optimal_moves(b) for b in positions)
            acc[name].append(correct / len(positions))
    return acc


def play_game(agent_x, agent_o, rng, mcts_check=None):
    """agent(board) -> move. Returns (result, list of (board, move) by UCT)."""
    board = EMPTY
    while winner(board) is None:
        agent = agent_x if to_move(board) == X else agent_o
        move = agent(board)
        if mcts_check is not None and agent in mcts_check:
            mcts_check[agent].append(move in optimal_moves(board))
        board = play(board, move)
    return winner(board)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    q = args.quick
    budgets = [10, 30, 100, 300] if q else [10, 30, 100, 300, 1000, 3000]
    n_positions = 40 if q else 1000
    game_sims = 300 if q else 3000
    n_vs_random = 10 if q else 200          # per colour
    n_self = 4 if q else 50
    n_vs_perfect = 4 if q else 50           # per colour
    print(f"MCTS for tic-tac-toe | seed={args.seed} c=sqrt(2) rewards in [0,1] | accuracy budgets={budgets} "
          f"on {n_positions} positions | games with {game_sims} simulations per move")

    # 1. exact solver
    t0 = time.time()
    positions, games = solver_sanity_checks()
    v0 = negamax(EMPTY)
    print(f"solver: value of the empty board for X = {v0:+d} ({'draw' if v0 == 0 else 'decisive'}); "
          f"{len(positions)} reachable positions; {sum(games.values())} complete games "
          f"(X wins {games[X]}, O wins {games[O]}, draws {games[0]}) [{time.time() - t0:.1f}s]")
    hard = nontrivial_positions(positions)
    n_nonterminal = sum(winner(b) is None for b in positions)
    print(f"{n_nonterminal} non-terminal positions, {len(hard)} of them non-trivial (some legal move is a mistake)")
    rng = random.Random(args.seed)
    sample = rng.sample(hard, n_positions)

    # speed check
    t0 = time.time()
    UCT(2000, rng=random.Random(1)).search(EMPTY)
    sims_per_s = 2000 / (time.time() - t0)
    print(f"speed: ~{sims_per_s:,.0f} UCT simulations per second from the empty board")

    # 2. accuracy vs budget
    t0 = time.time()
    acc = accuracy_experiment(sample, budgets, args.seed * 10_000)
    print(f"\nfraction of minimax-optimal moves on {n_positions} random non-trivial positions [{time.time() - t0:.1f}s]")
    print(f"{'budget (simulations)':<40}" + "".join(f"{n:>8}" for n in budgets))
    for name, vals in acc.items():
        print(f"{name:<40}" + "".join(f"{v:8.3f}" for v in vals))
    flat_limit = np.mean([flat_mc_limit_move(b) in optimal_moves(b) for b in sample])
    flat_limit_all = np.mean([flat_mc_limit_move(b) in optimal_moves(b) for b in hard])
    print(f"flat Monte Carlo with INFINITELY many rollouts (exact, via expectimax under random play): "
          f"{flat_limit:.3f} on this sample, {flat_limit_all:.3f} on all {len(hard)} non-trivial positions")

    # 2b. a concrete hard position: X has taken a corner; O's ONLY non-losing reply is the centre
    t0 = time.time()
    corner = play(EMPTY, 0)
    n_seeds = 10 if q else 50
    probe_budgets = [100, 300, 1000] if q else [100, 300, 1000, 3000]
    print(f"\nX opens in a corner; minimax-optimal replies for O: {sorted(optimal_moves(corner))} (centre). "
          f"Fraction of {n_seeds} seeds in which UCT (tree) plays it:")
    for n in probe_budgets:
        picks = [UCT(n, rng=random.Random(10_000 + k)).search(corner) for k in range(n_seeds)]
        print(f"   {n:>5} simulations: {sum(p in optimal_moves(corner) for p in picks) / n_seeds:.2f}")
    print(f"   [{time.time() - t0:.1f}s]")

    # 3. games
    t0 = time.time()
    rng = random.Random(args.seed + 1)
    uct = UCT(game_sims, rng=random.Random(args.seed + 2))
    uct_agent = uct.search
    rand_agent = lambda b: rng.choice(legal_moves(b))
    perfect_agent = lambda b: rng.choice(sorted(optimal_moves(b)))
    checks = {uct_agent: []}
    results = {}
    for label, ax_, ao_, n in [("UCT (X) vs random (O)", uct_agent, rand_agent, n_vs_random),
                               ("random (X) vs UCT (O)", rand_agent, uct_agent, n_vs_random),
                               ("UCT (X) vs perfect (O)", uct_agent, perfect_agent, n_vs_perfect),
                               ("perfect (X) vs UCT (O)", perfect_agent, uct_agent, n_vs_perfect)]:
        out = [play_game(ax_, ao_, rng, checks) for _ in range(n)]
        results[label] = (out.count(X), out.count(O), out.count(0))
    # UCT vs UCT: two independent searchers
    uct2 = UCT(game_sims, rng=random.Random(args.seed + 3))
    checks[uct2.search] = []
    out = [play_game(uct_agent, uct2.search, rng, checks) for _ in range(n_self)]
    results["UCT (X) vs UCT (O)"] = (out.count(X), out.count(O), out.count(0))
    print(f"\ngames ({game_sims} simulations per UCT move) [{time.time() - t0:.1f}s]")
    print(f"{'match':<26}{'X wins':>8}{'O wins':>8}{'draws':>8}")
    for label, (xw, ow, d) in results.items():
        print(f"{label:<26}{xw:>8}{ow:>8}{d:>8}")
    uct_losses = (results["UCT (X) vs random (O)"][1] + results["random (X) vs UCT (O)"][0]
                  + results["UCT (X) vs perfect (O)"][1] + results["perfect (X) vs UCT (O)"][0])
    all_checks = [c for lst in checks.values() for c in lst]
    print(f"UCT losses in all games vs random and perfect players: {uct_losses}")
    print(f"UCT moves checked against the solver: {len(all_checks)}; minimax-optimal: "
          f"{sum(all_checks)} ({100 * np.mean(all_checks):.2f}%)")
    print(f"UCT vs UCT: all {n_self} games drawn: {results['UCT (X) vs UCT (O)'][2] == n_self}")

    if q:
        print("quick mode: no figures written")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    styles = {"flat Monte Carlo (rollout algorithm)": ("tab:gray", "s"),
              "UCT (tree)": ("tab:blue", "o"),
              "UCT + transposition table": ("tab:red", "^")}
    for name, vals in acc.items():
        color, marker = styles[name]
        ax.plot(budgets, vals, marker=marker, color=color, label=name)
    base = np.mean([len(optimal_moves(b)) / len(legal_moves(b)) for b in sample])
    ax.axhline(flat_limit, color="tab:gray", ls="--", label=f"flat Monte Carlo, infinite budget (exact): {flat_limit:.3f}")
    ax.plot([], [], " ", label=f"(a uniformly random move: {base:.2f}, off scale)")
    ax.set_xscale("log")
    ax.set_ylim(0.75, 1.005)
    ax.set_xlabel("simulations per move (log scale)")
    ax.set_ylabel("fraction of minimax-optimal moves")
    ax.set_title(f"Tic-tac-toe: move quality vs search budget ({n_positions} non-trivial positions)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "mcts_tictactoe_accuracy.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
