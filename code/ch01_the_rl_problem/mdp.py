"""Finite Markov decision processes: the objects of Chapter 01, in code.

This module is a small library used by every script in this folder.  It
implements, from scratch and with nothing but NumPy:

* ``FiniteMDP`` - a finite MDP stored through its *four-argument dynamics*
  p(s', r | s, a)  (chapter Section 6), together with the derived quantities
  p(s' | s, a), r(s, a) and r(s, a, s').
* ``policy_matrices`` - the Markov chain (P_pi, r_pi) that a stationary policy
  induces on the states (Section 7).
* ``evaluate_policy_exact`` - v_pi = (I - gamma P_pi)^{-1} r_pi (Section 10).
* ``q_from_v`` / ``v_from_q`` - the two one-step relations between state and
  action values (Section 8).
* ``value_iteration`` and ``greedy_actions`` - used to obtain v_* and the
  greedy (optimal) policy (Section 12).  Value iteration itself is the subject
  of Chapter 03; here it is a tool.

Policies are represented as |S| x |A| row-stochastic matrices ``pi[s, a]``;
a deterministic policy given as an integer array is converted with
``deterministic_policy``.

Running this file performs a quick self-test.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FiniteMDP:
    """A finite MDP specified by its four-argument dynamics (Section 6).

    ``outcomes[s][a]`` is a list of ``(probability, next_state, reward)``
    triples: one entry per (s', r) pair with p(s', r | s, a) > 0.  Duplicate
    (s', r) pairs are allowed and are summed, which is what the definition of
    p as a probability mass function requires.

    ``terminal[s]`` marks terminal states.  Following the absorbing-state
    convention of Section 4.4, a terminal state must loop to itself with
    reward 0 under every action; ``validate`` checks this.
    """

    n_states: int
    n_actions: int
    outcomes: list
    state_names: list = field(default_factory=list)
    action_names: list = field(default_factory=list)
    terminal: np.ndarray | None = None

    def __post_init__(self):
        self._P = None  # caches for the derived tensors
        self._R = None
        if not self.state_names:
            self.state_names = [str(s) for s in range(self.n_states)]
        if not self.action_names:
            self.action_names = [str(a) for a in range(self.n_actions)]
        if self.terminal is None:
            self.terminal = np.zeros(self.n_states, dtype=bool)
        self.validate()

    # ------------------------------------------------------------------ checks
    def validate(self, atol: float = 1e-12) -> None:
        """p(., . | s, a) must be a probability distribution for every (s, a)."""
        for s in range(self.n_states):
            for a in range(self.n_actions):
                outs = self.outcomes[s][a]
                total = sum(p for p, _, _ in outs)
                if abs(total - 1.0) > 1e-9:
                    raise ValueError(f"sum_{{s',r}} p(s',r|s={s},a={a}) = {total} != 1")
                for p, s2, _ in outs:
                    if p < -atol or not (0 <= s2 < self.n_states):
                        raise ValueError(f"bad outcome {(p, s2)} for (s={s}, a={a})")
                if self.terminal[s]:
                    for p, s2, r in outs:
                        if p > atol and (s2 != s or r != 0):
                            raise ValueError(f"terminal state {s} must be absorbing with reward 0")

    # ---------------------------------------------------- derived quantities
    def reward_support(self) -> np.ndarray:
        """The finite reward set R (sorted)."""
        return np.array(sorted({r for s in range(self.n_states)
                                for a in range(self.n_actions)
                                for _, _, r in self.outcomes[s][a]}), dtype=float)

    def dynamics_tensor(self) -> tuple[np.ndarray, np.ndarray]:
        """Dense four-argument dynamics p[s, a, s', k] = p(s', rewards[k] | s, a)."""
        rewards = self.reward_support()
        index = {r: k for k, r in enumerate(rewards)}
        p = np.zeros((self.n_states, self.n_actions, self.n_states, len(rewards)))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for prob, s2, r in self.outcomes[s][a]:
                    p[s, a, s2, index[float(r)]] += prob  # duplicates are summed
        return p, rewards

    def transition_tensor(self) -> np.ndarray:
        """p(s' | s, a) = sum_r p(s', r | s, a)   (state-transition probabilities).

        Computed once and cached (the MDP is not modified after construction);
        callers must treat the returned array as read-only.
        """
        if self._P is None:
            P = np.zeros((self.n_states, self.n_actions, self.n_states))
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    for prob, s2, _ in self.outcomes[s][a]:
                        P[s, a, s2] += prob
            self._P = P
        return self._P

    def expected_reward(self) -> np.ndarray:
        """r(s, a) = sum_{s', r} r p(s', r | s, a)   (expected immediate reward; cached, read-only)."""
        if self._R is None:
            R = np.zeros((self.n_states, self.n_actions))
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    R[s, a] = sum(prob * r for prob, _, r in self.outcomes[s][a])
            self._R = R
        return self._R

    def expected_reward_sas(self) -> np.ndarray:
        """r(s, a, s') = sum_r r p(s', r | s, a) / p(s' | s, a)  (nan where p(s'|s,a)=0)."""
        num = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for prob, s2, r in self.outcomes[s][a]:
                    num[s, a, s2] += prob * r
        P = self.transition_tensor()
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(P > 0, num / np.where(P > 0, P, 1.0), np.nan)

    # -------------------------------------------------------------- sampling
    def sample(self, s: int, a: int, rng: np.random.Generator) -> tuple[int, float]:
        """Draw (S_{t+1}, R_{t+1}) ~ p(., . | s, a): the environment's half of one step."""
        outs = self.outcomes[s][a]
        probs = np.array([p for p, _, _ in outs])
        k = rng.choice(len(outs), p=probs / probs.sum())
        return int(outs[k][1]), float(outs[k][2])


# ---------------------------------------------------------------- policies
def deterministic_policy(actions, n_actions: int) -> np.ndarray:
    """Turn a deterministic policy pi(s) (integer array) into a matrix pi[s, a]."""
    actions = np.asarray(actions, dtype=int)
    pi = np.zeros((len(actions), n_actions))
    pi[np.arange(len(actions)), actions] = 1.0
    return pi


def uniform_policy(n_states: int, n_actions: int) -> np.ndarray:
    """The equiprobable random policy pi(a | s) = 1 / |A|."""
    return np.full((n_states, n_actions), 1.0 / n_actions)


def policy_matrices(mdp: FiniteMDP, pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The Markov reward process induced by a stationary policy (Section 7).

    P_pi[s, s'] = sum_a pi(a|s) p(s'|s, a),     r_pi[s] = sum_a pi(a|s) r(s, a).
    """
    P = mdp.transition_tensor()
    R = mdp.expected_reward()
    P_pi = np.einsum("sa,sat->st", pi, P)      # P_pi[s, s'] = sum_a pi(a|s) p(s'|s,a)
    r_pi = np.einsum("sa,sa->s", pi, R)        # r_pi[s]     = sum_a pi(a|s) r(s,a)
    return P_pi, r_pi


def evaluate_policy_exact(mdp: FiniteMDP, pi: np.ndarray, gamma: float) -> np.ndarray:
    """Solve the Bellman expectation equation in matrix form (Section 10).

    For gamma < 1:   (I - gamma P_pi) v = r_pi   has a unique solution.
    For gamma = 1:   (episodic tasks) we pin v(terminal) = 0 and solve only on
    the non-terminal states, where the restricted matrix is invertible if and
    only if the policy terminates with probability 1 from every state.
    We call ``np.linalg.solve`` rather than forming the inverse: it is cheaper
    and numerically more accurate.
    """
    P_pi, r_pi = policy_matrices(mdp, pi)                    # equation (1.10)
    n = mdp.n_states
    if gamma < 1.0:
        return np.linalg.solve(np.eye(n) - gamma * P_pi, r_pi)   # equation (1.17)
    nt = ~mdp.terminal
    v = np.zeros(n)
    P_nt = P_pi[np.ix_(nt, nt)]  # P~_pi: transitions among non-terminal states (substochastic)
    v[nt] = np.linalg.solve(np.eye(nt.sum()) - P_nt, r_pi[nt])
    return v


def q_from_v(mdp: FiniteMDP, v: np.ndarray, gamma: float) -> np.ndarray:
    """q(s, a) = r(s, a) + gamma * sum_{s'} p(s'|s, a) v(s')   (Section 8)."""
    return mdp.expected_reward() + gamma * mdp.transition_tensor() @ v


def v_from_q(pi: np.ndarray, q: np.ndarray) -> np.ndarray:
    """v(s) = sum_a pi(a|s) q(s, a)   (Section 8)."""
    return np.sum(pi * q, axis=1)


# ---------------------------------------------------------------- optimality
def bellman_optimality_backup(mdp: FiniteMDP, v: np.ndarray, gamma: float) -> np.ndarray:
    """(T* v)(s) = max_a [ r(s, a) + gamma sum_{s'} p(s'|s,a) v(s') ]   (Section 12)."""
    return q_from_v(mdp, v, gamma).max(axis=1)


def value_iteration(mdp: FiniteMDP, gamma: float, tol: float = 1e-12,
                    max_iter: int = 100_000) -> tuple[np.ndarray, list[float]]:
    """Repeatedly apply the Bellman optimality backup until the change is < tol.

    Previewed here only as a tool to compute v_*; Chapter 03 derives it and
    proves that it converges (the backup is a gamma-contraction).
    Returns v and the list of max-norm changes ||v_{k+1} - v_k||_inf.
    """
    v = np.zeros(mdp.n_states)
    deltas = []
    for _ in range(max_iter):
        v_new = bellman_optimality_backup(mdp, v, gamma)
        deltas.append(float(np.max(np.abs(v_new - v))))
        v = v_new
        if deltas[-1] < tol:
            break
    return v, deltas


def greedy_actions(q: np.ndarray, tol: float = 1e-9) -> list[list[int]]:
    """All maximizing actions in each state (ties within tol are kept)."""
    return [list(np.flatnonzero(q[s] >= q[s].max() - tol)) for s in range(q.shape[0])]


def greedy_policy(q: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """A deterministic greedy policy (lowest-index maximizing action)."""
    return np.array([acts[0] for acts in greedy_actions(q, tol)])


# ------------------------------------------------------------------ self-test
def _two_state_example() -> FiniteMDP:
    """The machine-maintenance MDP used for hand calculations in the chapter."""
    G, W = 0, 1
    RUN, FIX = 0, 1
    outcomes = [[None, None], [None, None]]
    outcomes[G][RUN] = [(0.75, G, 2.0), (0.25, W, 2.0)]
    outcomes[G][FIX] = [(1.0, G, 0.0)]
    outcomes[W][RUN] = [(0.5, W, 2.0), (0.5, W, -1.0)]
    outcomes[W][FIX] = [(1.0, G, -1.0)]
    return FiniteMDP(2, 2, outcomes, ["Good", "Worn"], ["run", "fix"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smoke test (this file is fast anyway)")
    parser.parse_args()
    np.set_printoptions(precision=4, suppress=True)
    mdp = _two_state_example()
    gamma = 0.8
    print(f"mdp.py self-test on the machine-maintenance MDP, gamma={gamma}")
    pi_rf = deterministic_policy([0, 1], 2)            # run when Good, fix when Worn
    v = evaluate_policy_exact(mdp, pi_rf, gamma)
    q = q_from_v(mdp, v, gamma)
    assert np.allclose(v, [7.5, 5.0]), v
    assert np.allclose(v_from_q(pi_rf, q), v)
    v_star, deltas = value_iteration(mdp, gamma)
    assert np.allclose(v_star, v, atol=1e-9)
    print("v_pi(run/fix) =", v, " q_pi =", q.tolist())
    print(f"value iteration: {len(deltas)} sweeps, v_* = {v_star}")
    print("self-test passed")


if __name__ == "__main__":
    main()
