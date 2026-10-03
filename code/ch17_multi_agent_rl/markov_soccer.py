"""Zero-sum Markov games: Shapley value iteration, minimax-Q and independent Q-learning on
a small grid soccer game in the style of Littman (1994) (Chapter 17, section 4).

The game (our own small variant; not an exact copy of Littman's 4x5 field)
---------------------------------------------------------------------------
A 2 x 4 grid (2 rows, 4 columns), two players A and B, one ball. A scores by carrying the ball off the WEST
edge (any row), B by carrying it off the EAST edge. Actions: N, S, E, W, stay.
Each step a fair coin decides who moves first; moves are then executed one at a time:
  * a move off the grid is ignored, unless it is the ball carrier leaving through the
    goal it attacks: then the episode ends and the scorer receives +1, the other -1;
  * a move into the square occupied by the other player does not take place, and if
    the mover had the ball, possession passes to the (stationary) other player.
Rewards are zero-sum (r_B = -r_A), discount gamma = 0.9 (so scoring sooner is better).
Start: A at (0, 2), B at (0, 1) (row, column), ball to a random player.
112 non-terminal states.

What the script does
--------------------
1. Shapley value iteration (Eq. 4.2, Alg. 4.1): V(s) <- val[ r(s,.,.) + gamma sum_s' p(s'|s,.,.) V(s') ],
   solving a 5x5 zero-sum matrix game in every state with games.solve_zero_sum_pivot.
2. Minimax-Q (Littman, 1994; Alg. 4.2) in self-play, and independent Q-learning (each
   player ignores the other's action) in self-play, from the same experience budget.
   Step sizes: minimax-Q uses 1/n(s,a,o)^0.6 (it observes B's action); each independent
   learner uses 1/n(s,a^i)^0.6, counting only its own state-action pairs. In minimax-Q
   self-play both players share one table: B plays the minimax strategy of -Q, which is
   what a second minimax-Q learner given the same data would compute.
3. Exploitability of a learned policy for A, computed exactly: fix A's policy, solve B's
   best-response MDP by value iteration, and report A's worst-case value from the start.

Run:  python code/ch17_multi_agent_rl/markov_soccer.py [--quick]
"""
import argparse
import os
import time

import numpy as np

import games

ROWS, COLS = 2, 4
GAMMA = 0.9
MOVES = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]   # N, S, E, W, stay
ACT_NAMES = ["N", "S", "E", "W", "stay"]
NA = len(MOVES)
START_A, START_B = (0, 2), (0, 1)

# ---- state indexing ------------------------------------------------------------------
CELLS = [(r, c) for r in range(ROWS) for c in range(COLS)]
STATES = [(pa, pb, ball) for pa in range(len(CELLS)) for pb in range(len(CELLS)) if pa != pb
          for ball in (0, 1)]                        # ball: 0 = A has it, 1 = B has it
S_INDEX = {s: i for i, s in enumerate(STATES)}
NS = len(STATES)


def resolve(state, aA, aB, a_first):
    """Deterministic outcome of joint action (aA, aB) when player a_first (0=A, 1=B) moves
    first. Returns (next_state or None if terminal, reward for A)."""
    pa, pb, ball = state
    pos = [CELLS[pa], CELLS[pb]]
    acts = [aA, aB]
    for mover in (a_first, 1 - a_first):
        other = 1 - mover
        r, c = pos[mover]
        dr, dc = MOVES[acts[mover]]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < ROWS and 0 <= nc < COLS):
            scoring_edge = (nc < 0) if mover == 0 else (nc >= COLS)
            if ball == mover and scoring_edge:
                return None, (1.0 if mover == 0 else -1.0)
            continue                                 # bumped into the wall: stay
        if (nr, nc) == pos[other]:
            if ball == mover:
                ball = other                         # tackled: the stationary player gets the ball
            continue
        pos[mover] = (nr, nc)
    return (CELLS.index(pos[0]), CELLS.index(pos[1]), ball), 0.0


def build_model():
    """Tabulate the model: for each state and joint action, the two equally likely move
    orders give (next-state index or -1 for terminal, reward for A)."""
    nxt = np.full((NS, NA, NA, 2), -1, dtype=np.int64)
    rew = np.zeros((NS, NA, NA, 2))
    for i, s in enumerate(STATES):
        for a in range(NA):
            for o in range(NA):
                for first in (0, 1):
                    s2, r = resolve(s, a, o, first)
                    nxt[i, a, o, first] = -1 if s2 is None else S_INDEX[s2]
                    rew[i, a, o, first] = r
    return nxt, rew


def backup(V, nxt, rew):
    """Q(s, a, o) = E[ r + gamma V(s') ] for all states and joint actions (A's viewpoint)."""
    Vn = np.where(nxt >= 0, V[np.maximum(nxt, 0)], 0.0)
    return (0.5 * (rew + GAMMA * Vn)).sum(-1)


def shapley_value_iteration(nxt, rew, tol=1e-8, max_sweeps=500):
    V = np.zeros(NS)
    for sweep in range(1, max_sweeps + 1):
        Q = backup(V, nxt, rew)
        V_new = np.array([games.solve_zero_sum_pivot(Q[s])[2] for s in range(NS)])
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < tol:
            break
    Q = backup(V, nxt, rew)
    piA = np.array([games.solve_zero_sum_pivot(Q[s])[0] for s in range(NS)])
    piB = np.array([games.solve_zero_sum_pivot(Q[s])[1] for s in range(NS)])
    return V, Q, piA, piB, sweep, delta


def worst_case_value(piA, nxt, rew, tol=1e-9):
    """A's value when B best-responds to the fixed policy piA (B solves an MDP)."""
    V = np.zeros(NS)
    for _ in range(2000):
        Q = backup(V, nxt, rew)                      # [S, a, o], A's payoff
        V_new = np.min(np.einsum("sa,sao->so", piA, Q), axis=1)   # B minimises A's value
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return V


def value_against(piA, piB, nxt, rew, tol=1e-9):
    V = np.zeros(NS)
    for _ in range(2000):
        Q = backup(V, nxt, rew)
        V_new = np.einsum("sa,sao,so->s", piA, Q, piB)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
    return V


def start_value(V):
    """Expected value at kick-off (ball to a random player)."""
    pa, pb = CELLS.index(START_A), CELLS.index(START_B)
    return 0.5 * (V[S_INDEX[(pa, pb, 0)]] + V[S_INDEX[(pa, pb, 1)]])


def sample_start(rng):
    return S_INDEX[(CELLS.index(START_A), CELLS.index(START_B), int(rng.integers(2)))]


def train(method, steps, nxt, rew, rng, eps=0.2, eval_every=None, max_len=100):
    """Self-play training. method = 'minimax-Q' or 'IQL'. Returns A's policy (and a log)."""
    if method == "minimax-Q":
        # One shared table: A's Q(s, a, o). B plays the minimax strategy of -QA, which is
        # exactly what a second minimax-Q learner fed the same experience would compute.
        QA = np.zeros((NS, NA, NA))
        piA = np.full((NS, NA), 1 / NA)
        piB = np.full((NS, NA), 1 / NA)
        VA = np.zeros(NS)
        counts = np.zeros((NS, NA, NA))              # n(s, a, o): minimax-Q observes B's action
    else:
        QiA = np.zeros((NS, NA))                     # independent learners ignore the other
        QiB = np.zeros((NS, NA))
        nA = np.zeros((NS, NA))                      # n(s, a): an independent learner can only
        nB = np.zeros((NS, NA))                      # count its own (state, action) pairs
    log = []
    s = sample_start(rng)
    ep_len = 0

    def policy_A():
        if method == "minimax-Q":
            return piA.copy()
        g = np.zeros((NS, NA))
        g[np.arange(NS), np.argmax(QiA, 1)] = 1.0    # IQL acts greedily: a deterministic policy
        return g

    for t in range(1, steps + 1):
        # ---- act (epsilon-exploration around the current policies) --------------------
        if method == "minimax-Q":
            a = rng.integers(NA) if rng.random() < eps else rng.choice(NA, p=piA[s])
            o = rng.integers(NA) if rng.random() < eps else rng.choice(NA, p=piB[s])
        else:
            a = rng.integers(NA) if rng.random() < eps else int(np.argmax(QiA[s] + 1e-9 * rng.random(NA)))
            o = rng.integers(NA) if rng.random() < eps else int(np.argmax(QiB[s] + 1e-9 * rng.random(NA)))
        first = int(rng.integers(2))
        s2, r = nxt[s, a, o, first], rew[s, a, o, first]
        # ---- learn ----------------------------------------------------------------------
        if method == "minimax-Q":
            counts[s, a, o] += 1
            alpha = 1.0 / counts[s, a, o] ** 0.6    # Robbins-Monro step size per (s, a, o)
            target = r + (GAMMA * VA[s2] if s2 >= 0 else 0.0)
            QA[s, a, o] += alpha * (target - QA[s, a, o])
            x, y, v = games.solve_zero_sum_pivot(QA[s])   # Alg. 4.2: re-solve the stage game at s
            piA[s], piB[s], VA[s] = x, y, v
        else:
            nA[s, a] += 1                            # Alg. 6.1 with step size 1/n(s, a^i)^0.6:
            nB[s, o] += 1                            # each learner uses only its own action
            tA = r + (GAMMA * QiA[s2].max() if s2 >= 0 else 0.0)
            tB = -r + (GAMMA * QiB[s2].max() if s2 >= 0 else 0.0)
            QiA[s, a] += (tA - QiA[s, a]) / nA[s, a] ** 0.6
            QiB[s, o] += (tB - QiB[s, o]) / nB[s, o] ** 0.6
        ep_len += 1
        if s2 < 0 or ep_len >= max_len:              # goal (terminal) or time limit (truncation:
            s = sample_start(rng)                    # we bootstrapped through it above)
            ep_len = 0
        else:
            s = s2
        if eval_every and t % eval_every == 0:
            log.append((t, start_value(worst_case_value(policy_A(), nxt, rew))))
    return policy_A(), log


def main():
    parser = argparse.ArgumentParser(description="Shapley VI, minimax-Q and IQL on grid soccer")
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    steps = 6_000 if args.quick else 300_000
    n_seeds = 1 if args.quick else 3
    eval_every = 2_000 if args.quick else 20_000
    print(f"markov_soccer.py | seed={args.seed} | grid {ROWS}x{COLS}, {NS} states, gamma={GAMMA} | "
          f"training steps={steps}, epsilon=0.2, alpha=1/n(s,a,o)^0.6 (minimax-Q) or 1/n(s,a^i)^0.6 (IQL), "
          f"seeds={n_seeds} | quick={args.quick}")
    t0 = time.time()
    nxt, rew = build_model()

    V, Q, piA_star, piB_star, sweeps, delta = shapley_value_iteration(nxt, rew)
    v_star = start_value(V)
    pa, pb = CELLS.index(START_A), CELLS.index(START_B)
    print(f"\n[Shapley value iteration] {sweeps} sweeps, last max change {delta:.1e}, {time.time() - t0:.1f} s")
    print(f"  value for A at kick-off (ball random) = {v_star:+.4f};  "
          f"A has ball: {V[S_INDEX[(pa, pb, 0)]]:+.4f};  B has ball: {V[S_INDEX[(pa, pb, 1)]]:+.4f}")
    print(f"  exploitability check: worst-case value of A's equilibrium policy = "
          f"{start_value(worst_case_value(piA_star, nxt, rew)):+.4f}")
    # kick-off with A in possession: B stands directly in A's way, so mixing matters.
    # (In the top row, 'N' has the same effect as 'stay'.)
    s_demo = S_INDEX[(pa, pb, 0)]
    print(f"  equilibrium at kick-off with A in possession [A at (0,2), B at (0,1)]: value {V[s_demo]:+.4f}")
    print("    stage-game Q(s, a, o) (rows: A's N,S,E,W,stay; columns: B's):\n"
          + np.array2string(Q[s_demo], precision=4, suppress_small=True))
    print("    A's mixed strategy: " + ", ".join(f"{n}={p:.3f}" for n, p in zip(ACT_NAMES, piA_star[s_demo]) if p > 1e-6))
    print("    B's mixed strategy: " + ", ".join(f"{n}={p:.3f}" for n, p in zip(ACT_NAMES, piB_star[s_demo]) if p > 1e-6))
    det = np.zeros((NS, NA)); det[np.arange(NS), np.argmax(piA_star, 1)] = 1
    print(f"  A's equilibrium policy made deterministic (argmax per state): worst-case value "
          f"{start_value(worst_case_value(det, nxt, rew)):+.4f}")
    unif = np.full((NS, NA), 1 / NA)
    v_rand = start_value(worst_case_value(unif, nxt, rew))
    print(f"  uniformly random policy for A: worst-case value {v_rand:+.4f}")

    logs = {"minimax-Q": [], "IQL": []}
    finals = {"minimax-Q": [], "IQL": []}
    vs_eq = {"minimax-Q": [], "IQL": []}
    for method in ("minimax-Q", "IQL"):
        for k in range(n_seeds):
            t1 = time.time()
            rng = np.random.default_rng(args.seed + 100 * k + (0 if method == "minimax-Q" else 1))
            piA, log = train(method, steps, nxt, rew, rng, eval_every=eval_every)
            logs[method].append(log)
            wc = start_value(worst_case_value(piA, nxt, rew))
            ve = start_value(value_against(piA, piB_star, nxt, rew))
            finals[method].append(wc)
            vs_eq[method].append(ve)
            print(f"\n[{method}, seed {k}] {steps} steps in {time.time() - t1:.1f} s: worst-case value of A's policy "
                  f"{wc:+.4f} (minimax value {v_star:+.4f}); value vs B's equilibrium policy {ve:+.4f}")
    print(f"\nsummary: minimax value {v_star:+.4f} | worst case: minimax-Q "
          + ", ".join(f"{v:+.4f}" for v in finals["minimax-Q"]) + " | IQL "
          + ", ".join(f"{v:+.4f}" for v in finals["IQL"]) + f" | random {v_rand:+.4f}")
    print(f"total time {time.time() - t0:.1f} s")

    if not args.quick:
        import plot_style
        plt = plot_style.setup()
        figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
        os.makedirs(figdir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9.5, 4.3))
        st = {"minimax-Q": dict(color=plot_style.C[0], marker="o", ms=4),
              "IQL": dict(color=plot_style.C[1], marker="s", ms=4)}
        for method, ls in (("minimax-Q", "-"), ("IQL", "--")):
            for k, log in enumerate(logs[method]):
                tt, vv = zip(*log)
                ax.plot(tt, vv, ls=ls, label=f"{method} (seed {k})", alpha=1.0 if k == 0 else 0.6, **st[method])
        v_lab = 0.0 if abs(v_star) < 5e-4 else v_star            # avoid printing "-0.000"
        ax.axhline(v_star, color=plot_style.INK, lw=1.2, ls=":", label=f"minimax value {v_lab:.3f}")
        ax.axhline(v_rand, color=plot_style.GREY, lw=1.2, ls="-.", label=f"uniform random {v_rand:+.3f}")
        ax.set_xlabel("training steps (self-play)")
        ax.set_ylabel("A's value against a best-responding B")
        ax.set_title("Grid soccer: worst-case value of A's learned policy")
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "markov_soccer.png"))
        plt.close(fig)
        print(f"figure written to {figdir}/markov_soccer.png")


if __name__ == "__main__":
    main()
