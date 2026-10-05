"""AlphaZero-lite for tic-tac-toe: PUCT search + policy/value network + self-play.

Chapter 13, Sections 9.3-9.6 (Algorithms 13.8-13.9).  Everything AlphaZero does, at toy scale:

  * a single network f_theta(s) = (p, v): policy logits over the 9 cells (illegal moves masked) and a
    value v in [-1, 1] for the player to move.  Input: the board from the mover's point of view
    (two 3x3 planes: "my stones", "opponent's stones"), so one network plays both sides;
  * MCTS with the PUCT rule  a = argmax_a [ Q(s,a) + c_puct P(s,a) sqrt(N(s)) / (1 + N(s,a)) ]  (Eq. 13.15),
    leaves evaluated by v (no rollouts), negamax backups, Dirichlet noise at the root in self-play;
  * self-play games: move t is sampled from the visit counts with temperature 1 for the first
    ``temp_plies`` plies, then played greedily; each position stores (s_t, pi_t, z_t) with
    pi_t = N(s_t, .)/N(s_t) and z_t = final result from the mover's point of view;
  * training on  (z - v)^2 - pi^T log p + c ||theta||^2   (Eq. 13.17) from a replay buffer of recent games,
    with the 8 symmetries of the board as data augmentation (as in AlphaGo Zero; AlphaZero dropped it
    because chess and shogi are not symmetric; switch it off with --set augment=false).

Speed trick (tic-tac-toe only): there are only 4,520 non-terminal reachable positions, so after every
training phase we evaluate the network on ALL of them in one batch and cache (p, v).  Self-play then
uses that frozen snapshot, exactly as AlphaZero's actors use a recent copy of the network.  The
search algorithm is unchanged; only the bookkeeping is cheaper (measure the speed-up with --bench 40).

Evaluation is exact, using a negamax solver:
  * move accuracy: fraction of "non-trivial" positions (where some legal move is NOT minimax-optimal)
    on which a player picks an optimal move;
  * exhaustive verification: the evaluation player is deterministic (no noise, greedy, fixed
    tie-breaking), so we can enumerate EVERY possible opponent move sequence, as X and as O, and count
    the lines that the player loses.  Zero losing lines = it never loses against any opponent,
    in particular against a perfect (minimax) player.

Run (from the repository root):
  python code/ch13_model_based_rl/alphazero_tictactoe.py           # full: 3 seeds, evaluation, figure
  python code/ch13_model_based_rl/alphazero_tictactoe.py --seeds 1 --set augment=false sims=50   # ablation
  python code/ch13_model_based_rl/alphazero_tictactoe.py --bench 40   # time the snapshot cache only
  python code/ch13_model_based_rl/alphazero_tictactoe.py --quick   # smoke test, no figures
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True          # importing the sibling modules must not leave __pycache__/
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from dataclasses import asdict, dataclass, replace  # noqa: E402
from functools import lru_cache  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
RES_DIR = os.path.join(HERE, "results")

# ---------------------------------------------------------------------------------------------
# Game rules.  A board is a tuple of 9 ints: +1 = X, -1 = O, 0 = empty.  X moves first.
# ---------------------------------------------------------------------------------------------
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
EMPTY = (0,) * 9


def to_move(b):
    return 1 if sum(1 for x in b if x != 0) % 2 == 0 else -1


@lru_cache(maxsize=None)
def winner(b):
    """+1 / -1 if X / O has three in a row, 0 for a full board (draw), None if the game goes on."""
    for i, j, k in LINES:
        if b[i] != 0 and b[i] == b[j] == b[k]:
            return b[i]
    return 0 if all(x != 0 for x in b) else None


@lru_cache(maxsize=None)
def legal(b):
    return tuple(i for i in range(9) if b[i] == 0)


def play(b, a):
    lst = list(b)
    lst[a] = to_move(b)
    return tuple(lst)


@lru_cache(maxsize=None)
def solve(b):
    """Negamax value of b for the player to move: +1 win, 0 draw, -1 loss under perfect play."""
    w = winner(b)
    if w is not None:
        return 0 if w == 0 else -1          # if the game is over, the player to move did not win
    return max(-solve(play(b, a)) for a in legal(b))


@lru_cache(maxsize=None)
def optimal_moves(b):
    v = solve(b)
    return tuple(a for a in legal(b) if -solve(play(b, a)) == v)


def all_positions():
    seen, stack = {EMPTY}, [EMPTY]
    while stack:
        b = stack.pop()
        if winner(b) is None:
            for a in legal(b):
                c = play(b, a)
                if c not in seen:
                    seen.add(c); stack.append(c)
    return sorted(seen)


# ---------------------------------------------------------------------------------------------
# Network f_theta(s) = (policy logits, value)
# ---------------------------------------------------------------------------------------------
class PVNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(18, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, 9)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        h = self.body(x)
        logits = self.pi(h).masked_fill(~mask, -1e9)          # illegal moves get probability 0
        return logits, torch.tanh(self.v(h)).squeeze(-1)


def encode(boards):
    """Board -> 18 inputs from the mover's perspective: [my stones (9), opponent's stones (9)]."""
    arr = np.array(boards, dtype=np.float32)                   # (n, 9)
    me = np.array([to_move(b) for b in boards], dtype=np.float32)[:, None]
    can = arr * me                                             # +1 = mine, -1 = opponent's
    x = np.concatenate([(can == 1), (can == -1)], axis=1).astype(np.float32)
    mask = arr == 0
    return torch.from_numpy(x), torch.from_numpy(mask)


SYMS = []                                   # the 8 symmetries of the square, as cell permutations
_g = np.arange(9).reshape(3, 3)
for _k in range(4):
    _r = np.rot90(_g, _k)
    SYMS.append(_r.flatten()); SYMS.append(np.fliplr(_r).flatten())


@torch.no_grad()
def evaluate_all(net, positions):
    """Network snapshot on every non-terminal position: {board: (prior list[9], value)}."""
    xs, ms = encode(positions)
    logits, v = net(xs, ms)
    p = torch.softmax(logits, -1).numpy()
    return {b: (p[i].tolist(), float(v[i])) for i, b in enumerate(positions)}


# ---------------------------------------------------------------------------------------------
# PUCT Monte Carlo tree search (Algorithm 13.8)
# ---------------------------------------------------------------------------------------------
class Node:
    __slots__ = ("P", "N", "W", "children", "moves", "total")

    def __init__(self, prior, moves):
        self.P = prior            # prior P(s, a) for all 9 cells (0 for illegal)
        self.N = [0] * 9          # visit counts N(s, a)
        self.W = [0.0] * 9        # total value W(s, a), from the point of view of the player to move at s
        self.children = {}
        self.moves = moves
        self.total = 0            # N(s) = sum_a N(s, a)


def select(node, c_puct):
    sq = math.sqrt(max(node.total, 1))       # max(.,1): at a fresh node the prior alone decides
    best, best_a = -1e9, None
    for a in node.moves:
        n = node.N[a]
        q = node.W[a] / n if n else 0.0      # unvisited edges: Q = 0 (neutral; "first-play urgency")
        u = c_puct * node.P[a] * sq / (1 + n)
        if q + u > best:
            best, best_a = q + u, a
    return best_a


def mcts(root_board, n_sims, evaluator, c_puct, rng=None, dir_alpha=None, dir_eps=0.25):
    """Run n_sims PUCT simulations from root_board.  evaluator(board) -> (prior list, value).
    Returns the root node (visit counts in root.N)."""
    prior, _ = evaluator(root_board)
    moves = legal(root_board)
    if dir_alpha is not None:                # exploration noise at the root only (self-play)
        noise = rng.dirichlet([dir_alpha] * len(moves))
        prior = list(prior)
        for a, n in zip(moves, noise):
            prior[a] = (1 - dir_eps) * prior[a] + dir_eps * n
    root = Node(prior, moves)
    for _ in range(n_sims):
        node, board, path = root, root_board, []
        while True:                           # 1. selection
            a = select(node, c_puct)
            path.append((node, a))
            board = play(board, a)
            w = winner(board)
            if w is not None:                 # terminal: exact value for the player to move there
                value = 0.0 if w == 0 else -1.0
                break
            child = node.children.get(a)
            if child is None:                 # 2. expansion + evaluation by the network (no rollout)
                p, value = evaluator(board)
                node.children[a] = Node(p, legal(board))
                break
            node = child
        for node, a in reversed(path):        # 3. backup, negamax: flip the sign at every ply
            value = -value
            node.N[a] += 1
            node.W[a] += value
            node.total += 1
    return root


def greedy_from_counts(root):
    """Most-visited move; ties broken by the prior, then by the lowest index (deterministic)."""
    return max(root.moves, key=lambda a: (root.N[a], root.P[a], -a))


# ---------------------------------------------------------------------------------------------
# Self-play (Algorithm 13.9, inner loop)
# ---------------------------------------------------------------------------------------------
def self_play_game(evaluator, cfg, rng):
    board, history, ply = EMPTY, [], 0
    while winner(board) is None:
        root = mcts(board, cfg.sims, evaluator, cfg.c_puct, rng, cfg.dir_alpha, cfg.dir_eps)
        counts = np.array(root.N, dtype=np.float64)
        pi = counts / counts.sum()
        if ply < cfg.temp_plies:
            a = int(rng.choice(9, p=pi))                      # temperature 1: sample from visit counts
        else:
            best = np.flatnonzero(counts == counts.max())      # temperature -> 0: greedy
            a = int(rng.choice(best))
        history.append((board, pi))
        board = play(board, a)
        ply += 1
    w = winner(board)
    return [(b, pi, float(w * to_move(b))) for b, pi in history]   # z_t from the mover's viewpoint


# ---------------------------------------------------------------------------------------------
# Exact evaluation
# ---------------------------------------------------------------------------------------------
def exhaustive_losses(policy):
    """policy(board) -> move, deterministic.  Enumerate every opponent reply sequence with the
    policy playing X, then O.  Returns {'X': (wins, draws, losses), 'O': (...)} counted in lines."""
    out = {}
    for side, sgn in (("X", 1), ("O", -1)):
        res = [0, 0, 0]

        def rec(b):
            w = winner(b)
            if w is not None:
                res[0 if w == sgn else (1 if w == 0 else 2)] += 1
                return
            if to_move(b) == sgn:
                rec(play(b, policy(b)))
            else:
                for a in legal(b):
                    rec(play(b, a))
        rec(EMPTY)
        out[side] = tuple(res)
    return out


def show_mistakes(policy, evaluator, max_shown=6):
    """Print the positions, reached against some opponent line, where the deterministic policy plays
    a move that is not minimax-optimal: the board, the move, the optimal moves, the network's prior
    and value.  Used to diagnose failures (Exercise 14)."""
    seen = set()
    sym = {1: "X", -1: "O", 0: "."}

    def rec(b, sgn):
        w = winner(b)
        if w is not None:
            return
        if to_move(b) == sgn:
            a = policy(b)
            if a not in optimal_moves(b) and b not in seen and len(seen) < max_shown:
                seen.add(b)
                p, v = evaluator(b)
                rows = [" ".join(sym[b[3 * r + c]] for c in range(3)) for r in range(3)]
                print(f"    mistake ({'X' if sgn == 1 else 'O'} to move)  {' / '.join(rows)}  played cell {a}; "
                      f"optimal {optimal_moves(b)}; prior argmax {max(legal(b), key=lambda m: p[m])} "
                      f"(prior {[round(p[m], 2) for m in legal(b)]} on cells {list(legal(b))}); value {v:+.2f}")
            rec(play(b, a), sgn)
        else:
            for a in legal(b):
                rec(play(b, a), sgn)
    for sgn in (1, -1):
        rec(EMPTY, sgn)
    if not seen:
        print("    no mistakes on any line")


def move_accuracy(policy, positions):
    return float(np.mean([policy(b) in optimal_moves(b) for b in positions]))


def make_mcts_policy(evaluator, sims, c_puct):
    cache = {}

    def pol(b):
        if b not in cache:
            cache[b] = greedy_from_counts(mcts(b, sims, evaluator, c_puct))
        return cache[b]
    return pol


def make_raw_policy(evaluator):
    return lambda b: max(legal(b), key=lambda a: (evaluator(b)[0][a], -a))


def rollout_evaluator(rng):
    """'Pure MCTS' baseline: uniform prior, value = result of one uniformly random playout."""
    def ev(b):
        moves = legal(b)
        prior = [0.0] * 9
        for a in moves:
            prior[a] = 1.0 / len(moves)
        me, c = to_move(b), b
        while winner(c) is None:
            m = legal(c)
            c = play(c, m[rng.integers(len(m))])
        w = winner(c)
        return prior, float(w * me)
    return ev


def bench_snapshot_cache(cfg, n_games, seed=0):
    """Measure the speed-up of the snapshot cache (Section 9.6): play the same self-play games once
    with the cached snapshot (including the time to build it) and once calling the network on every
    newly expanded node.  Both evaluators return the same numbers, so the games are identical."""
    torch.manual_seed(seed)
    net = PVNet(cfg.hidden).eval()
    nonterminal = [b for b in all_positions() if winner(b) is None]
    calls = [0]

    @torch.no_grad()
    def net_evaluator(board):
        calls[0] += 1
        x, m = encode([board])
        logits, v = net(x, m)
        return torch.softmax(logits, -1)[0].tolist(), float(v[0])

    t0 = time.time()
    snap = evaluate_all(net, nonterminal)
    t_snap = time.time() - t0
    cached = snap.__getitem__

    def counted(board):
        calls[0] += 1
        return cached(board)

    times = {}
    for name, ev in (("cached snapshot", counted), ("network call per node", net_evaluator)):
        calls[0] = 0
        rng = np.random.default_rng(seed)
        t0 = time.time()
        n_pos = sum(len(self_play_game(ev, cfg, rng)) for _ in range(n_games))
        times[name] = (time.time() - t0, calls[0], n_pos)
    (t_c, n_c, p_c), (t_n, n_n, p_n) = times["cached snapshot"], times["network call per node"]
    assert (n_c, p_c) == (n_n, p_n), "the two evaluators should produce identical games"
    print(f"Snapshot-cache benchmark: {n_games} self-play games, {cfg.sims} simulations per move, "
          f"{p_c} positions, {n_c} evaluator calls")
    print(f"  cached snapshot:       {t_snap:.2f} s to evaluate all {len(nonterminal)} positions + "
          f"{t_c:.2f} s of self-play = {t_snap + t_c:.2f} s")
    print(f"  network call per node: {t_n:.2f} s ({1e6 * t_n / n_n:.0f} microseconds per call, search included)")
    print(f"  speed-up: {t_n / (t_snap + t_c):.1f}x including the snapshot, {t_n / t_c:.1f}x excluding it")


def play_vs_perfect(policy, n_games, rng):
    """Games against a perfect player that picks uniformly among minimax-optimal moves."""
    res = {"X": [0, 0, 0], "O": [0, 0, 0]}
    for side, sgn in (("X", 1), ("O", -1)):
        for _ in range(n_games):
            b = EMPTY
            while winner(b) is None:
                if to_move(b) == sgn:
                    b = play(b, policy(b))
                else:
                    opt = optimal_moves(b)
                    b = play(b, opt[rng.integers(len(opt))])
            w = winner(b)
            res[side][0 if w == sgn else (1 if w == 0 else 2)] += 1
    return res


# ---------------------------------------------------------------------------------------------
# Training loop (Algorithm 13.9)
# ---------------------------------------------------------------------------------------------
@dataclass
class AZConfig:
    seed: int = 0
    iterations: int = 60          # self-play / train cycles
    games_per_iter: int = 40
    sims: int = 100               # simulations per move in self-play
    c_puct: float = 1.5
    dir_alpha: float = 1.0        # Dirichlet(alpha) root noise; AlphaZero scales alpha ~ 10 / (#legal moves)
    dir_eps: float = 0.25
    temp_plies: int = 4           # sample moves with temperature 1 for the first plies, then greedy
    buffer_positions: int = 40_000  # positions (after augmentation)
    train_steps: int = 150        # gradient steps per iteration
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4    # the c ||theta||^2 term (coupled L2 via Adam's weight_decay)
    hidden: int = 128
    augment: bool = True          # 8-fold symmetry augmentation (AlphaGo Zero used it, AlphaZero did not)
    eval_sims: int = 100          # simulations per move for the evaluation player


def train(cfg: AZConfig, positions, nontrivial, verbose=True):
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    net = PVNet(cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    nonterminal = [b for b in positions if winner(b) is None]
    nt_x, nt_m = encode(nontrivial)
    true_v = torch.tensor([float(solve(b)) for b in nontrivial])
    buf = deque(maxlen=cfg.buffer_positions)
    log = []
    t0 = time.time()
    games = 0
    for it in range(cfg.iterations + 1):
        net.eval()
        snap = evaluate_all(net, nonterminal)
        ev = snap.__getitem__
        # ---- evaluation of the current snapshot ----
        with torch.no_grad():
            logits, v = net(nt_x, nt_m)
        raw_acc = float(np.mean([int(torch.argmax(logits[i])) in optimal_moves(b) for i, b in enumerate(nontrivial)]))
        v_mse = float(((v - true_v) ** 2).mean())
        ex_raw = exhaustive_losses(make_raw_policy(ev))
        ex_mcts = exhaustive_losses(make_mcts_policy(ev, cfg.eval_sims, cfg.c_puct))
        entry = {"iter": it, "games": games, "raw_acc": raw_acc, "value_mse": v_mse,
                 "raw_losing_lines": ex_raw["X"][2] + ex_raw["O"][2],
                 "mcts_losing_lines": ex_mcts["X"][2] + ex_mcts["O"][2], "time": time.time() - t0}
        log.append(entry)
        if verbose:
            print(f"  iter {it:2d} games {games:5d} | policy-net optimal-move acc {raw_acc:.3f} | value MSE "
                  f"{v_mse:.3f} | losing lines (all opponents): raw net {entry['raw_losing_lines']:4d}, "
                  f"MCTS-{cfg.eval_sims} {entry['mcts_losing_lines']:4d} | {entry['time']:.0f}s", flush=True)
        if it == cfg.iterations:
            break
        # ---- self-play with the frozen snapshot ----
        for _ in range(cfg.games_per_iter):
            for b, pi, z in self_play_game(ev, cfg, rng):
                if cfg.augment:
                    for perm in SYMS:
                        buf.append((tuple(b[j] for j in perm), pi[perm], z))
                else:
                    buf.append((b, pi, z))
            games += 1
        # ---- training on (s, pi, z) ----
        net.train()
        data = list(buf)
        for _ in range(cfg.train_steps):
            idx = rng.integers(0, len(data), size=cfg.batch_size)
            bs = [data[i][0] for i in idx]
            x, m = encode(bs)
            pi_t = torch.tensor(np.array([data[i][1] for i in idx]), dtype=torch.float32)
            z_t = torch.tensor([data[i][2] for i in idx], dtype=torch.float32)
            logits, v = net(x, m)
            loss = F.mse_loss(v, z_t) - (pi_t * F.log_softmax(logits, -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net, log, evaluate_all(net, nonterminal)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override AZConfig fields, e.g. --set augment=false sims=50 "
                         "(runs with overrides print results but save nothing)")
    ap.add_argument("--show-mistakes", action="store_true",
                    help="after training, print the evaluation player's non-optimal moves (diagnosis)")
    ap.add_argument("--bench", type=int, default=0, metavar="N_GAMES",
                    help="only time N_GAMES self-play games with the snapshot cache vs a network call per node")
    args = ap.parse_args()
    cfg = AZConfig()
    for kv in args.set:
        k, v = kv.split("=")
        typ = type(getattr(cfg, k))
        cfg = replace(cfg, **{k: (v.lower() == "true") if typ is bool else typ(v)})
    seeds = args.seeds
    if args.quick:
        cfg = replace(cfg, iterations=2, games_per_iter=8, sims=16, train_steps=20, eval_sims=8)
        seeds = [0]
    if args.bench:
        bench_snapshot_cache(cfg, args.bench if not args.quick else 2, seed=seeds[0])
        return
    print("AlphaZero-lite, tic-tac-toe | seeds", seeds, "| config:", asdict(cfg), flush=True)
    t_all = time.time()
    positions = all_positions()
    nonterminal = [b for b in positions if winner(b) is None]
    nontrivial = [b for b in nonterminal if len(optimal_moves(b)) < len(legal(b))]
    print(f"solver: {len(positions)} reachable positions, {len(nonterminal)} non-terminal, "
          f"{len(nontrivial)} non-trivial (some move is a mistake); value of the empty board = {solve(EMPTY)}")

    runs = []
    n_games = 200 if not args.quick else 4
    for seed in seeds:
        c = replace(cfg, seed=seed)
        print(f"--- seed {seed} ---", flush=True)
        t0 = time.time()
        net, log, snap = train(c, positions, nontrivial)
        ev = snap.__getitem__
        rng = np.random.default_rng(seed + 1)
        raw = make_raw_policy(ev)
        pol = make_mcts_policy(ev, c.eval_sims, c.c_puct)
        run = {"seed": seed, "log": log, "train_time": time.time() - t0,
               "raw_acc": move_accuracy(raw, nontrivial), "raw_exhaustive": exhaustive_losses(raw),
               "mcts_exhaustive": exhaustive_losses(pol), "vs_perfect": play_vs_perfect(pol, n_games, rng)}
        runs.append(run)
        print(f"  final (seed {seed}): raw policy network optimal-move accuracy {run['raw_acc']:.4f}; exhaustive "
              f"(wins, draws, losses) in lines: raw net as X {run['raw_exhaustive']['X']}, as O "
              f"{run['raw_exhaustive']['O']}; MCTS-{c.eval_sims} as X {run['mcts_exhaustive']['X']}, as O "
              f"{run['mcts_exhaustive']['O']}")
        print(f"  MCTS-{c.eval_sims} vs perfect player (random among optimal moves), {n_games} games per side, "
              f"(W, D, L): as X {tuple(run['vs_perfect']['X'])}, as O {tuple(run['vs_perfect']['O'])}", flush=True)
        if args.show_mistakes:
            print(f"  non-optimal moves of the MCTS-{c.eval_sims} player against some opponent line:")
            show_mistakes(pol, ev)
        if seed == seeds[0]:
            ev0 = ev

    # ---- search with and without learned guidance (first seed's network) ----
    rng = np.random.default_rng(seeds[0] + 2)
    sims_list = [1, 2, 4, 8, 16, 32, 64, 128] if not args.quick else [1, 4]
    n_sub = 600 if not args.quick else 40
    sub = [nontrivial[i] for i in rng.choice(len(nontrivial), size=min(n_sub, len(nontrivial)), replace=False)]
    acc_az, acc_pure = [], []
    for sm in sims_list:
        acc_az.append(move_accuracy(make_mcts_policy(ev0, sm, cfg.c_puct), sub))
        acc_pure.append(move_accuracy(make_mcts_policy(rollout_evaluator(rng), sm, cfg.c_puct), sub))
    print(f"\nOptimal-move accuracy vs simulations per move (seed {seeds[0]}'s network; "
          f"{len(sub)} random non-trivial positions):")
    print("    sims:                 " + "".join(f"{x:7d}" for x in sims_list))
    print("    AlphaZero-lite (net): " + "".join(f"{a:7.3f}" for a in acc_az))
    print("    pure MCTS (rollouts): " + "".join(f"{a:7.3f}" for a in acc_pure))
    print("\nSummary over seeds:")
    for r in runs:
        lost_raw = r["raw_exhaustive"]["X"][2] + r["raw_exhaustive"]["O"][2]
        lost_mcts = r["mcts_exhaustive"]["X"][2] + r["mcts_exhaustive"]["O"][2]
        last_m = max((e["games"] for e in r["log"] if e["mcts_losing_lines"] > 0), default=None)
        last_r = max((e["games"] for e in r["log"] if e["raw_losing_lines"] > 0), default=None)
        print(f"  seed {r['seed']}: losing lines vs all opponents: raw net {lost_raw}, MCTS-{cfg.eval_sims} {lost_mcts}; "
              f"losses vs perfect player {r['vs_perfect']['X'][2] + r['vs_perfect']['O'][2]}/{2 * n_games}; "
              f"last losing line seen at {last_m} games (MCTS), {last_r} games (raw net); "
              f"train+eval time {r['train_time']:.0f}s")
    print(f"Total wall-clock: {time.time() - t_all:.0f} s")

    if args.quick or args.set:
        return
    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "alphazero_tictactoe.json"), "w") as f:
        json.dump({"config": asdict(cfg), "runs": runs, "sims": sims_list, "acc_az": acc_az,
                   "acc_pure": acc_pure}, f)
    from plot_style import C, GREY, setup
    plt = setup()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax = axes[0]
    for i, r in enumerate(runs):
        g = [e["games"] for e in r["log"]]
        ax.plot(g, [e["raw_acc"] for e in r["log"]], color=C[0], lw=1.4, alpha=0.8,
                label="policy net: optimal-move accuracy" if i == 0 else None)
        ax.plot(g, [e["value_mse"] for e in r["log"]], color=C[1], lw=1.4, ls="--", alpha=0.8,
                label="value net: MSE vs minimax value" if i == 0 else None)
    ax.axhline(1.0, color=GREY, lw=0.6)
    ax.set_xlabel("self-play games"); ax.set_ylim(0, 1.02)
    ax.set_title(f"Network quality ({len(runs)} seeds, non-trivial positions)")
    ax.legend(fontsize=8, loc="center right")
    ax = axes[1]
    for i, r in enumerate(runs):
        g = [e["games"] for e in r["log"]]
        ax.plot(g, [e["raw_losing_lines"] for e in r["log"]], color=C[0], lw=1.4, alpha=0.8,
                label="raw policy net (no search)" if i == 0 else None)
        ax.plot(g, [e["mcts_losing_lines"] for e in r["log"]], color=C[2], lw=1.4, ls="--", alpha=0.9,
                label=f"PUCT search, {cfg.eval_sims} sims" if i == 0 else None)
    ax.set_yscale("symlog", linthresh=1); ax.set_xlabel("self-play games")
    ax.set_ylabel("losing lines vs ALL opponents (X + O)")
    ax.set_title("Exhaustive verification")
    ax.legend(fontsize=8)
    ax = axes[2]
    ax.plot(sims_list, acc_az, color=C[2], marker="^", label="AlphaZero-lite (prior + value net)")
    ax.plot(sims_list, acc_pure, color=C[3], ls="-.", marker="D", label="pure MCTS (uniform prior, rollouts)")
    ax.axhline(1.0, color=GREY, lw=0.6)
    ax.set_xscale("log", base=2); ax.set_xlabel("simulations per move"); ax.set_ylabel("optimal-move accuracy")
    ax.set_title("Search with and without learned guidance")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "alphazero_tictactoe.png")
    fig.savefig(out); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
