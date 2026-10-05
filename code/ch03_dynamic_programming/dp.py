"""dp.py: tabular dynamic programming, written to mirror Chapter 03 line by line.

Everything works on a `TabularMDP` (arrays P[s,a,s'], R[s,a]; see mdps.py for the
"substochastic P" convention used for termination). Section numbers refer to
chapters/03-dynamic-programming.md.

    Bellman operators (Sec. 2)       q_from_v, bellman_expectation, bellman_optimality
    ... on action values (Sec. 2.6)  bellman_expectation_q, bellman_optimality_q, q_value_iteration
    Policy evaluation (Sec. 3)       evaluate_exact, evaluate_iterative (two-array / in-place)
    Policy improvement (Sec. 4)      greedy
    Policy iteration (Sec. 5)        policy_iteration
    Value iteration (Sec. 6)         value_iteration
    Modified PI (Sec. 7)             modified_policy_iteration
    Finite horizon (Sec. 11)         backward_induction, evaluate_finite_horizon

A policy is either an int array `actions[s]` (deterministic) or a row-stochastic
matrix `pi[s, a]`; `as_matrix` converts the former into the latter.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# The MDP container
# ---------------------------------------------------------------------------
@dataclass
class TabularMDP:
    P: np.ndarray                         # (S, A, S) continuation probabilities (substochastic)
    R: np.ndarray                         # (S, A) expected immediate reward r(s, a)
    gamma: float
    valid: np.ndarray | None = None       # (S, A) bool mask of A(s)
    terminal: np.ndarray | None = None    # (S,) bool
    name: str = ""
    state_names: list | None = None
    action_names: list | None = None
    shape: tuple | None = None            # grid shape for plotting, if any
    _flatP: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        S, A, S2 = self.P.shape
        assert S == S2 and self.R.shape == (S, A)
        if self.valid is None:
            self.valid = np.ones((S, A), dtype=bool)
        if self.terminal is None:
            self.terminal = np.zeros(S, dtype=bool)
        assert self.valid.any(axis=1).all(), "every state needs at least one action"
        row_sums = self.P.sum(axis=2)
        assert np.all(row_sums <= 1 + 1e-9) and np.all(self.P >= 0)
        # A (S*A, S) view makes every backup a single matrix-vector product.
        self._flatP = self.P.reshape(S * A, S)

    @property
    def S(self) -> int:
        return self.P.shape[0]

    @property
    def A(self) -> int:
        return self.P.shape[1]


def as_matrix(mdp: TabularMDP, policy) -> np.ndarray:
    """Deterministic action array -> one-hot (S, A) matrix; a matrix is returned unchanged."""
    policy = np.asarray(policy)
    if policy.ndim == 2:
        return policy
    pi = np.zeros((mdp.S, mdp.A))
    pi[np.arange(mdp.S), policy] = 1.0
    return pi


def uniform_policy(mdp: TabularMDP) -> np.ndarray:
    """Equiprobable random policy over A(s)."""
    return mdp.valid / mdp.valid.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Bellman operators (Section 2)
# ---------------------------------------------------------------------------
def q_from_v(mdp: TabularMDP, V: np.ndarray, mask: bool = True) -> np.ndarray:
    """One-step lookahead: Q[s, a] = r(s,a) + gamma * sum_s' p(s'|s,a) V(s')    (eq. 3.1)."""
    Q = mdp.R + mdp.gamma * (mdp._flatP @ V).reshape(mdp.S, mdp.A)
    if mask:
        Q = np.where(mdp.valid, Q, -np.inf)    # actions not in A(s) can never be chosen
    return Q


def bellman_optimality(mdp: TabularMDP, V: np.ndarray) -> np.ndarray:
    """(T* V)(s) = max_a Q[s, a]    (eq. 3.3)."""
    return q_from_v(mdp, V).max(axis=1)


def policy_model(mdp: TabularMDP, policy):
    """Induced Markov reward process: r_pi (S,) and P_pi (S, S)    (Chapter 01, eq. 1.17)."""
    policy = np.asarray(policy)
    if policy.ndim == 1:                       # deterministic: just pick one row per state, O(S^2)
        idx = np.arange(mdp.S)
        return mdp.R[idx, policy], mdp.P[idx, policy]
    pi = policy
    r_pi = (pi * mdp.R).sum(axis=1)
    P_pi = np.einsum("sa,sat->st", pi, mdp.P)
    return r_pi, P_pi


def bellman_expectation(mdp: TabularMDP, V: np.ndarray, policy) -> np.ndarray:
    """(T^pi V)(s) = sum_a pi(a|s) Q[s, a] = r_pi + gamma P_pi V    (eq. 3.2)."""
    pi = as_matrix(mdp, policy)
    return (pi * q_from_v(mdp, V, mask=False)).sum(axis=1)


# ---------------------------------------------------------------------------
# Bellman operators on action values (Section 2.6)
# ---------------------------------------------------------------------------
def bellman_optimality_q(mdp: TabularMDP, Q: np.ndarray) -> np.ndarray:
    """(T* Q)(s,a) = r(s,a) + gamma sum_s' p(s'|s,a) max_a' Q(s',a')    (eq. 3.13b).

    Invalid actions are masked out of the inner max; their own entries are set to -inf."""
    v_next = np.where(mdp.valid, Q, -np.inf).max(axis=1)        # max_a' Q(s', a') over A(s')
    return q_from_v(mdp, v_next)                                 # = q_V with V = max_a' Q(., a')


def bellman_expectation_q(mdp: TabularMDP, Q: np.ndarray, policy) -> np.ndarray:
    """(T^pi Q)(s,a) = r(s,a) + gamma sum_s' p(s'|s,a) sum_a' pi(a'|s') Q(s',a')    (eq. 3.13a)."""
    pi = as_matrix(mdp, policy)
    v_next = (pi * np.where(mdp.valid, Q, 0.0)).sum(axis=1)
    return q_from_v(mdp, v_next)


def q_value_iteration(mdp: TabularMDP, theta: float = 1e-10, Q0=None, max_sweeps: int = 10_000_000):
    """Q-value iteration Q_{k+1} = T* Q_k (Section 2.6). Stops when max |Q_{k+1} - Q_k| < theta.

    Greedy improvement from Q is a plain argmax over the table: no model is needed for it. That is
    why the model-free control methods of Chapters 04-05 store Q rather than V.
    Returns Q (invalid actions = -inf), number of sweeps.
    """
    Q = np.where(mdp.valid, 0.0, -np.inf) if Q0 is None else np.where(mdp.valid, Q0, -np.inf)
    for sweep in range(1, max_sweeps + 1):
        Q_new = bellman_optimality_q(mdp, Q)
        delta = float(np.max(np.abs(np.where(mdp.valid, Q_new - Q, 0.0))))
        Q = Q_new
        if delta < theta:
            break
    return Q, sweep


# ---------------------------------------------------------------------------
# Policy evaluation (Section 3)
# ---------------------------------------------------------------------------
def evaluate_exact(mdp: TabularMDP, policy) -> np.ndarray:
    """Solve (I - gamma P_pi) v = r_pi. Requires gamma < 1 or a proper policy (else singular)."""
    r_pi, P_pi = policy_model(mdp, policy)
    M = np.eye(mdp.S) - mdp.gamma * P_pi
    return np.linalg.solve(M, r_pi)


def evaluate_iterative(mdp: TabularMDP, policy, theta: float = 1e-8, inplace: bool = False,
                       V0=None, max_sweeps: int = 1_000_000, order=None, record: bool = False):
    """Iterative policy evaluation, Algorithm 3.1.

    two-array (inplace=False):  V_{k+1} = T^pi V_k, all states updated from the OLD array (Jacobi).
    in-place  (inplace=True):   one array; states updated in `order`, later states in the sweep
                                already see the new values of earlier ones (Gauss-Seidel).
    Stops when the largest change in a sweep, Delta, falls below theta.
    Returns V, number of sweeps, and (if record) the list of V after each sweep.
    """
    r_pi, P_pi = policy_model(mdp, policy)
    g = mdp.gamma
    V = np.zeros(mdp.S) if V0 is None else np.array(V0, dtype=float)
    order = np.arange(mdp.S) if order is None else order
    history = [V.copy()] if record else None
    for sweep in range(1, max_sweeps + 1):
        if inplace:
            delta = 0.0
            for s in order:
                v_old = V[s]
                V[s] = r_pi[s] + g * P_pi[s] @ V          # uses already-updated entries
                delta = max(delta, abs(V[s] - v_old))
        else:
            V_new = r_pi + g * P_pi @ V                  # uses only the old array
            delta = np.max(np.abs(V_new - V))
            V = V_new
        if record:
            history.append(V.copy())
        if delta < theta:
            break
    return V, sweep, history


# ---------------------------------------------------------------------------
# Policy improvement (Section 4)
# ---------------------------------------------------------------------------
def greedy(mdp: TabularMDP, V: np.ndarray, old_actions=None, tol: float = 1e-9) -> np.ndarray:
    """Deterministic greedy policy: pi'(s) in argmax_a Q[s, a].

    Tie-breaking matters for policy iteration (Section 5): if `old_actions` is given we KEEP
    the old action whenever it is within `tol` of the best one. Without this rule PI can
    cycle forever between equally good policies (S&B Exercise 4.4).
    """
    return greedy_from_q(q_from_v(mdp, V), old_actions, tol)


def greedy_from_q(Q: np.ndarray, old_actions=None, tol: float = 1e-9) -> np.ndarray:
    """Same as `greedy`, starting from an already computed Q (saves one backup)."""
    best = Q.max(axis=1)
    new = Q.argmax(axis=1)                       # first maximiser
    if old_actions is not None:
        old_q = Q[np.arange(Q.shape[0]), old_actions]
        keep = old_q >= best - tol * np.maximum(1.0, np.abs(best))
        new = np.where(keep, old_actions, new)
    return new


def greedy_actions_all(mdp: TabularMDP, V: np.ndarray, tol: float = 1e-9) -> list:
    """The full set argmax_a Q[s, a] (within tol) for each state, for plotting ties."""
    Q = q_from_v(mdp, V)
    best = Q.max(axis=1, keepdims=True)
    return [np.flatnonzero(Q[s] >= best[s] - tol) for s in range(mdp.S)]


# ---------------------------------------------------------------------------
# Policy iteration (Section 5)
# ---------------------------------------------------------------------------
def policy_iteration(mdp: TabularMDP, policy0, tol: float = 1e-9, max_iter: int = 10_000):
    """Algorithm 3.2 (Howard's policy iteration) with exact evaluation.

    policy0 may be stochastic (it is evaluated once, then replaced by its greedy policy).
    Returns (actions, V, history) where history[i] = dict(V, actions, n_changed, time).
    """
    t0 = time.perf_counter()
    history = []
    pi = np.asarray(policy0)
    actions = None if pi.ndim == 2 else pi.copy()
    for it in range(max_iter):
        V = evaluate_exact(mdp, actions if actions is not None else pi)      # E step: (I - g P_pi) V = r_pi
        new = greedy(mdp, V, old_actions=actions, tol=tol)                    # I step, ties keep the old action
        n_changed = mdp.S if actions is None else int(np.sum(new != actions))
        history.append(dict(V=V, actions=None if actions is None else actions.copy(),
                            n_changed=n_changed, time=time.perf_counter() - t0))
        if actions is not None and n_changed == 0:
            return actions, V, history                                       # policy stable => optimal
        actions = new
    raise RuntimeError("policy iteration did not converge")


# ---------------------------------------------------------------------------
# Value iteration (Section 6)
# ---------------------------------------------------------------------------
def value_iteration(mdp: TabularMDP, theta: float = 1e-10, V0=None, inplace: bool = False,
                    max_sweeps: int = 10_000_000, order=None, record: bool = False,
                    record_every: int = 1):
    """Algorithm 3.3. Repeatedly apply T* until the largest change Delta < theta.

    inplace=False: synchronous (Jacobi) V_{k+1} = T* V_k.  inplace=True: Gauss-Seidel sweep.
    Returns V, sweeps, deltas (list), history (list of V's if record).
    For gamma < 1, Delta < theta guarantees ||V - v_*|| < gamma*theta/(1-gamma)   (eq. 3.10),
    and the greedy policy loses at most 2*gamma*theta/(1-gamma)                   (eq. 3.21).
    """
    V = np.zeros(mdp.S) if V0 is None else np.array(V0, dtype=float)
    g = mdp.gamma
    order = np.arange(mdp.S) if order is None else order
    deltas = []
    history = [V.copy()] if record else None
    for sweep in range(1, max_sweeps + 1):
        if inplace:
            delta = 0.0
            for s in order:
                v_old = V[s]
                q = mdp.R[s] + g * mdp.P[s] @ V
                V[s] = np.max(np.where(mdp.valid[s], q, -np.inf))
                delta = max(delta, abs(V[s] - v_old))
        else:
            V_new = bellman_optimality(mdp, V)
            delta = float(np.max(np.abs(V_new - V)))
            V = V_new
        deltas.append(delta)
        if record and sweep % record_every == 0:
            history.append(V.copy())
        if delta < theta:
            break
    return V, sweep, deltas, history


# ---------------------------------------------------------------------------
# Modified (truncated) policy iteration (Section 7)
# ---------------------------------------------------------------------------
def modified_policy_iteration(mdp: TabularMDP, m: int, tol: float = 1e-6, V0=None,
                              max_iter: int = 10_000_000):
    """Algorithm 3.4. Greedy step, then m applications of T^pi (warm-started at V).

    m = 1 is exactly value iteration; m -> infinity is policy iteration.
    Stops when the residual bound ||T*V - V|| / (1 - gamma) < tol, which guarantees
    ||V - v_*|| < tol (eq. 3.11) for ANY V.  Returns V, actions, n_improve, n_eval_sweeps.
    """
    assert m >= 1 and mdp.gamma < 1
    g = mdp.gamma
    V = np.zeros(mdp.S) if V0 is None else np.array(V0, dtype=float)
    actions = None
    n_eval = 0
    for it in range(1, max_iter + 1):
        Q = q_from_v(mdp, V)
        TV = Q.max(axis=1)                                   # T* V  (one full backup)
        if np.max(np.abs(TV - V)) / (1 - g) < tol:           # residual bound (3.11)
            return V, greedy_from_q(Q, actions), it, n_eval
        actions = greedy_from_q(Q, actions)                  # pi_{k+1} greedy w.r.t. V_k
        V = TV                                               # first T^pi application = T* V
        if m > 1:
            r_pi, P_pi = policy_model(mdp, actions)
            for _ in range(m - 1):                           # m - 1 more cheap T^pi sweeps
                V = r_pi + g * (P_pi @ V)
            n_eval += m - 1
    raise RuntimeError("MPI did not converge")


# ---------------------------------------------------------------------------
# Finite horizon (Section 11)
# ---------------------------------------------------------------------------
def backward_induction(mdp: TabularMDP, H: int, terminal_value=None):
    """Algorithm 3.6. V_H = terminal value; V_t = max_a [r + gamma P V_{t+1}] for t = H-1..0.

    Returns V of shape (H+1, S) and the non-stationary policy pi of shape (H, S),
    pi[t, s] = optimal action at time t (with H - t steps to go).
    """
    V = np.zeros((H + 1, mdp.S))
    if terminal_value is not None:
        V[H] = terminal_value
    pi = np.zeros((H, mdp.S), dtype=int)
    for t in range(H - 1, -1, -1):
        Q = q_from_v(mdp, V[t + 1])
        pi[t] = Q.argmax(axis=1)
        V[t] = Q.max(axis=1)
    return V, pi


def evaluate_finite_horizon(mdp: TabularMDP, policy_t, H: int) -> np.ndarray:
    """Value of a (possibly non-stationary) policy over H steps: policy_t(t) -> action array."""
    V = np.zeros(mdp.S)
    for t in range(H - 1, -1, -1):
        V = bellman_expectation(mdp, V, policy_t(t))
    return V
