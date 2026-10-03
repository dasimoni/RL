"""A slippery chain MDP for the off-policy experiments of Chapter 06 (Sections 7.3 and 14).

    [T] - 1 - 2 - ... - 14 - 15 - [T]
     0                             16     <- state indices

* actions: 0 = left, 1 = right; the move goes the intended way with probability
  0.85 and the opposite way with probability SLIP = 0.15;
* reward +1 on entering the right terminal, -1 on entering the left terminal,
  0 otherwise; gamma = 0.95;
* episodes start in a uniformly random non-terminal state.

Policies are arrays of shape (17, 2) with zero rows for the terminal states, so that
Vbar(terminal) = sum_a pi(a|terminal) Q(terminal, a) = 0 automatically.
Exact values q_pi are obtained by solving the Bellman equations as a linear system.
"""
from __future__ import annotations

import numpy as np

N = 15
N_TOTAL = N + 2
LEFT_T, RIGHT_T = 0, N + 1
GAMMA = 0.95
SLIP = 0.15
N_A = 2


def transitions(s: int, a: int):
    """[(probability, next state)] for action a in non-terminal state s."""
    d = 1 if a == 1 else -1
    return [(1.0 - SLIP, s + d), (SLIP, s - d)]


def reward(s1: int) -> float:
    return 1.0 if s1 == RIGHT_T else (-1.0 if s1 == LEFT_T else 0.0)


def is_terminal(s: int) -> bool:
    return s == LEFT_T or s == RIGHT_T


def policy(p_right: float) -> np.ndarray:
    """State-independent policy that goes right with probability p_right."""
    P = np.zeros((N_TOTAL, N_A))
    P[1:N + 1, 1] = p_right
    P[1:N + 1, 0] = 1.0 - p_right
    return P


def exact_q(pi: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """q_pi(s, a) for all s (terminal rows 0), from v_pi = (I - gamma P_pi)^{-1} r_pi."""
    P = np.zeros((N_TOTAL, N_TOTAL))
    r = np.zeros(N_TOTAL)
    for s in range(1, N + 1):
        for a in range(N_A):
            for p, s1 in transitions(s, a):
                r[s] += pi[s, a] * p * reward(s1)
                if not is_terminal(s1):
                    P[s, s1] += pi[s, a] * p
    v = np.linalg.solve(np.eye(N_TOTAL) - gamma * P, r)
    q = np.zeros((N_TOTAL, N_A))
    for s in range(1, N + 1):
        for a in range(N_A):
            q[s, a] = sum(p * (reward(s1) + gamma * (0.0 if is_terminal(s1) else v[s1]))
                          for p, s1 in transitions(s, a))
    return q


def exact_v(pi: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    return (pi * exact_q(pi, gamma)).sum(axis=1)


def generate_episode(rng: np.random.Generator, b: np.ndarray):
    """Episode under behaviour policy b: arrays S (T+1), A (T), R (T); R[t] = R_{t+1}."""
    s = int(rng.integers(1, N + 1))
    S, A, R = [s], [], []
    while not is_terminal(s):
        a = int(rng.random() < b[s, 1])
        d = 1 if a == 1 else -1
        s = s + (d if rng.random() >= SLIP else -d)
        S.append(s); A.append(a); R.append(reward(s))
    return np.array(S), np.array(A), np.array(R)


def rms_q_error(Q: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    """RMS error over the 2N non-terminal (s, a) pairs; Q has shape (..., N_TOTAL, 2)."""
    d = Q[..., 1:N + 1, :] - q_true[1:N + 1, :]
    return np.sqrt(np.mean(d ** 2, axis=(-2, -1)))
