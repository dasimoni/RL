"""Finite MDPs used in Chapter 03, all in one array format.

Every MDP is a `TabularMDP` (see dp.py) holding

    P[s, a, s']  = probability of moving s -> s' under a AND NOT terminating,
    R[s, a]      = r(s, a) = E[R_{t+1} | S_t = s, A_t = a]   (expected immediate reward),
    valid[s, a]  = True if a is in A(s),
    terminal[s]  = True for terminal states.

Why "and not terminating"? Episodic tasks end in a terminal state whose value is 0
by definition (Chapter 01, Section 4.4). Instead of keeping an absorbing terminal
state with a self-loop, we simply DROP the probability mass of transitions that end
the episode. Each row P[s, a, :] then sums to at most 1 (it is *substochastic*), the
missing mass is the termination probability, and every Bellman backup
r(s,a) + gamma * sum_s' P[s,a,s'] V(s') automatically treats the continuation value
after termination as 0.  This is exactly the "bootstrap through truncation, not
through termination" rule of NOTATION.md, applied to a model instead of to samples.

Terminal states are kept as (unreachable-after-termination) rows with one dummy
valid action, zero reward and an all-zero P row, so that every backup gives V = 0
there without special cases.
"""
from __future__ import annotations

import numpy as np

from dp import TabularMDP


# ---------------------------------------------------------------------------
# 1. The two-state machine-maintenance MDP of Chapter 01, Section 6.3
# ---------------------------------------------------------------------------
def maintenance_mdp(gamma: float = 0.8) -> TabularMDP:
    """States 0 = Good, 1 = Worn; actions 0 = run, 1 = fix.  Continuing task."""
    P = np.zeros((2, 2, 2))
    R = np.zeros((2, 2))
    P[0, 0] = [0.75, 0.25]; R[0, 0] = 2.0          # run a good machine
    P[0, 1] = [1.0, 0.0];   R[0, 1] = 0.0          # fix a good machine (wasted)
    P[1, 0] = [0.0, 1.0];   R[1, 0] = 0.5          # run a worn machine: +2 or -1, w.p. 1/2
    P[1, 1] = [1.0, 0.0];   R[1, 1] = -1.0         # repair
    return TabularMDP(P=P, R=R, gamma=gamma, name="maintenance",
                      state_names=["G", "W"], action_names=["run", "fix"])


# ---------------------------------------------------------------------------
# 2. Sutton & Barto Example 4.1: the 4x4 gridworld (gamma = 1, reward -1 per step)
# ---------------------------------------------------------------------------
GRID_ACTIONS = ["up", "down", "right", "left"]           # S&B order
GRID_DELTAS = [(-1, 0), (1, 0), (0, 1), (0, -1)]          # (row, col) changes
GRID_ARROWS = ["↑", "↓", "→", "←"]


def gridworld_4x4(gamma: float = 1.0) -> TabularMDP:
    """Cells numbered 0..15 row by row; cells 0 and 15 are the (single, shaded) terminal state.

    Every transition gives reward -1. Moves off the grid leave the state unchanged.
    Entering a terminal cell ends the episode, so that probability mass is dropped from P.
    """
    n = 4
    S, A = n * n, 4
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    terminal = np.zeros(S, dtype=bool)
    terminal[[0, S - 1]] = True
    valid = np.ones((S, A), dtype=bool)
    for s in range(S):
        if terminal[s]:
            valid[s] = False
            valid[s, 0] = True                     # dummy action: R = 0, no continuation
            continue
        r, c = divmod(s, n)
        for a, (dr, dc) in enumerate(GRID_DELTAS):
            r2, c2 = r + dr, c + dc
            if not (0 <= r2 < n and 0 <= c2 < n):
                r2, c2 = r, c                      # bump into the wall
            s2 = r2 * n + c2
            R[s, a] = -1.0
            if not terminal[s2]:
                P[s, a, s2] = 1.0                  # else: episode ends, mass dropped
    return TabularMDP(P=P, R=R, gamma=gamma, valid=valid, terminal=terminal,
                      name="gridworld4x4", action_names=GRID_ACTIONS, shape=(n, n))


# ---------------------------------------------------------------------------
# 3. Sutton & Barto Example 4.3: the gambler's problem
# ---------------------------------------------------------------------------
def gamblers_problem(ph: float = 0.4, goal: int = 100, allow_zero_stake: bool = False) -> TabularMDP:
    """States = capital 0..goal (0 and goal terminal). Action = stake a in {1, ..., min(s, goal - s)}.

    With probability ph the coin comes up heads and the gambler wins the stake; otherwise
    she loses it. Reward +1 on reaching `goal`, 0 otherwise; gamma = 1 (undiscounted episodic).
    Sutton & Barto allow a = 0 as well; a zero stake never changes the state, so a policy that
    uses it never terminates (an *improper* policy). We exclude it by default (see the chapter).
    """
    S = goal + 1
    A = goal // 2 + 1                               # stakes 0..goal//2
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    valid = np.zeros((S, A), dtype=bool)
    terminal = np.zeros(S, dtype=bool)
    terminal[[0, goal]] = True
    for s in range(S):
        if terminal[s]:
            valid[s, 0] = True                     # dummy action
            continue
        lo = 0 if allow_zero_stake else 1
        for a in range(lo, min(s, goal - s) + 1):
            valid[s, a] = True
            win, lose = s + a, s - a
            if a == 0:
                P[s, a, s] = 1.0
                continue
            # heads: win the stake
            if win == goal:
                R[s, a] += ph * 1.0                # reaching the goal pays +1 and ends the episode
            else:
                P[s, a, win] += ph
            # tails: lose the stake
            if lose != 0:
                P[s, a, lose] += 1.0 - ph          # ruin (lose == 0) ends the episode with reward 0
    return TabularMDP(P=P, R=R, gamma=1.0, valid=valid, terminal=terminal,
                      name=f"gambler(ph={ph})")


# ---------------------------------------------------------------------------
# 4. FrozenLake-v1 straight from Gymnasium's transition table env.unwrapped.P
# ---------------------------------------------------------------------------
FL_ARROWS = ["←", "↓", "→", "↑"]      # Gymnasium order: LEFT, DOWN, RIGHT, UP


def frozenlake_mdp(map_name: str = "4x4", gamma: float = 0.99, is_slippery: bool = True):
    """Read FrozenLake's model from env.unwrapped.P.

    env.unwrapped.P[s][a] is a list of (prob, next_state, reward, terminated) tuples.
    A `terminated` transition ends the episode, so its continuation value is 0: we add its
    reward to R but leave its probability out of P (see the module docstring). Gymnasium's
    TimeLimit wrapper (100 steps for FrozenLake-v1 with either map; the separately registered
    FrozenLake8x8-v1 uses 200) is a property of the *simulator*, not of the MDP, and is not part
    of this model (it returns in finite_horizon.py).
    Returns (mdp, desc) where desc is the map as a 2-D array of characters.
    """
    import gymnasium as gym

    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
    model = env.unwrapped.P
    desc = np.asarray(env.unwrapped.desc, dtype="U1")
    S = env.observation_space.n
    A = env.action_space.n
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    terminal = np.isin(desc.ravel(), ["H", "G"])
    valid = np.ones((S, A), dtype=bool)
    for s in range(S):
        for a in range(A):
            for prob, s2, r, done in model[s][a]:
                R[s, a] += prob * r
                if not done:
                    P[s, a, s2] += prob
    env.close()
    mdp = TabularMDP(P=P, R=R, gamma=gamma, valid=valid, terminal=terminal,
                     name=f"FrozenLake{map_name}", action_names=["left", "down", "right", "up"],
                     shape=desc.shape)
    return mdp, desc


# ---------------------------------------------------------------------------
# 5. Random MDPs (for timing and for checking theorems on many instances)
# ---------------------------------------------------------------------------
def random_mdp(S: int, A: int, gamma: float, branching: int = 5, seed: int = 0) -> TabularMDP:
    """Each (s, a) moves to `branching` distinct random successors with Dirichlet(1) weights.

    Rewards r(s, a) ~ Uniform[0, 1). Continuing task (rows of P sum to exactly 1).
    """
    rng = np.random.default_rng(seed)
    P = np.zeros((S, A, S))
    b = min(branching, S)
    for s in range(S):
        for a in range(A):
            succ = rng.choice(S, size=b, replace=False)
            P[s, a, succ] = rng.dirichlet(np.ones(b))
    R = rng.random((S, A))
    return TabularMDP(P=P, R=R, gamma=gamma, name=f"random(S={S},A={A},b={b})")
