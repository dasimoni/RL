"""The 19-state random walk (Sutton & Barto 2018, Example 7.1), shared by the Chapter 06 scripts.

    [T] - 1 - 2 - ... - 10 - ... - 18 - 19 - [T]
     0                 start                 20      <- state indices

* states 1..19 are non-terminal, 0 and 20 are terminal;
* every episode starts in the centre state 10;
* each step moves left or right with probability 1/2 (this is a Markov *reward*
  process: there are no actions, only a fixed policy, so this is pure prediction);
* the reward is -1 on entering the left terminal, +1 on entering the right
  terminal and 0 on every other transition; gamma = 1.

The true values are v(s) = (s - 10) / 10, i.e. -0.9, -0.8, ..., 0.9.

Because episodes do not depend on the value estimates, every prediction script
pre-generates its episodes once and feeds the *same* episodes to every algorithm
(common random numbers). The order of updates is still exactly that of the
online algorithms, which only ever look at the past.
"""
from __future__ import annotations

import numpy as np

N_STATES = 19
N_TOTAL = N_STATES + 2            # including the two terminal states
LEFT, RIGHT = 0, N_STATES + 1     # terminal indices
START = (N_STATES + 1) // 2       # 10
TRUE_V = (np.arange(1, N_STATES + 1) - START) / START   # v(1..19) = -0.9..0.9
TRUE_V_FULL = np.concatenate([[0.0], TRUE_V, [0.0]])    # with terminals (value 0)


def generate_episode(rng: np.random.Generator, start: int = START) -> tuple[np.ndarray, np.ndarray]:
    """One episode: states S_0..S_T (length T+1) and rewards R_1..R_T (length T).

    rewards[t] holds R_{t+1}, the reward for the transition S_t -> S_{t+1}.
    Steps are drawn in chunks so the walk is generated with a few numpy calls.
    """
    path = [start]
    pos = start
    while True:
        steps = rng.choice((-1, 1), size=256)
        walk = pos + np.cumsum(steps)
        hit = np.flatnonzero((walk == LEFT) | (walk == RIGHT))
        if hit.size:
            path.extend(walk[: hit[0] + 1].tolist())
            break
        path.extend(walk.tolist())
        pos = int(walk[-1])
    states = np.asarray(path, dtype=np.int64)
    rewards = np.zeros(len(states) - 1)
    rewards[-1] = 1.0 if states[-1] == RIGHT else -1.0
    return states, rewards


def generate_runs(n_runs: int, n_episodes: int, seed: int):
    """episodes[run][k] = (states, rewards) for the k-th episode of a run."""
    rng = np.random.default_rng(seed)
    return [[generate_episode(rng) for _ in range(n_episodes)] for _ in range(n_runs)]


def rms_error(V: np.ndarray) -> np.ndarray:
    """RMS error over the 19 non-terminal states; V has shape (..., 21)."""
    return np.sqrt(np.mean((V[..., 1:N_STATES + 1] - TRUE_V) ** 2, axis=-1))


def markov_chain():
    """Sub-stochastic transition matrix P (19x19) among non-terminal states and the
    expected one-step reward r(s) = E[R_{t+1} | S_t = s]. Used for exact calculations:
    v = (I - P)^{-1} r, and E[G_{t:t+n} | S_t] = sum_{k<n} P^k r + P^n V (gamma = 1)."""
    P = np.zeros((N_STATES, N_STATES))
    r = np.zeros(N_STATES)
    for i in range(N_STATES):          # i = s - 1
        for step in (-1, 1):
            j = i + step
            if 0 <= j < N_STATES:
                P[i, j] += 0.5
            else:
                r[i] += 0.5 * (1.0 if step == 1 else -1.0)
    return P, r
