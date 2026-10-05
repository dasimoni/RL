"""Exact policy-gradient machinery for small finite MDPs (shared by the tabular Chapter 10 scripts).

Everything here is computed *exactly* with linear algebra, so the scripts can compare
Monte Carlo estimates against the truth:

* ``TabularMDP``      -- an episodic finite MDP with an absorbing terminal state;
* ``LinearSoftmax``   -- the policy  pi(a|s) = exp(theta . x(s,a)) / sum_b exp(theta . x(s,b))
                         (Section 2.1); one-hot features give the tabular softmax;
* ``exact_gradient``  -- the policy gradient theorem, Eq. (10.9):
                         grad J = sum_s eta_gamma(s) sum_a q_pi(s,a) grad pi(a|s);
* ``fd_gradient``     -- central finite differences of the closed-form J(theta), an
                         independent check that does not use the theorem at all.

Conventions (Chapter 10, Section 3): non-terminal states 0..S-1; the terminal state is
implicit -- the probability of terminating after (s, a) is 1 - sum_s' P[s, a, s'].
J(theta) = sum_s d0(s) v_pi(s), and eta_gamma(s) = sum_t gamma^t Pr{S_t = s} with S_0 ~ d0.

This file has no main block (it is not a script); it is imported by pg_theorem_check.py,
discount_bias.py and vtrace_tabular.py.
"""
from __future__ import annotations

import numpy as np


class TabularMDP:
    """Episodic finite MDP. P has shape (S, A, S) and may be sub-stochastic (mass to terminal)."""

    def __init__(self, P, r, d0, gamma, reward_sd=0.0):
        self.P = np.asarray(P, dtype=float)
        self.r = np.asarray(r, dtype=float)          # expected reward r(s, a)
        self.d0 = np.asarray(d0, dtype=float)
        self.gamma = float(gamma)
        self.reward_sd = float(reward_sd)            # Gaussian reward noise (sampling only)
        self.n_s, self.n_a, _ = self.P.shape
        self.p_term = 1.0 - self.P.sum(axis=2)       # Pr{terminate | s, a}
        assert np.all(self.p_term > -1e-12)

    # ---------- exact quantities ----------
    def policy_matrices(self, pi):
        """P_pi(s, s') = sum_a pi(a|s) P(s'|s,a);  r_pi(s) = sum_a pi(a|s) r(s,a)."""
        P_pi = np.einsum("sa,sat->st", pi, self.P)
        r_pi = (pi * self.r).sum(axis=1)
        return P_pi, r_pi

    def evaluate(self, pi, gamma=None):
        """Solve the Bellman equation v = r_pi + gamma P_pi v exactly; return v, q."""
        g = self.gamma if gamma is None else gamma
        P_pi, r_pi = self.policy_matrices(pi)
        v = np.linalg.solve(np.eye(self.n_s) - g * P_pi, r_pi)
        q = self.r + g * np.einsum("sat,t->sa", self.P, v)
        return v, q

    def visits(self, pi, gamma=None):
        """eta_gamma(s) = sum_t gamma^t Pr{S_t = s}:  the row vector d0^T (I - gamma P_pi)^{-1}."""
        g = self.gamma if gamma is None else gamma
        P_pi, _ = self.policy_matrices(pi)
        return np.linalg.solve((np.eye(self.n_s) - g * P_pi).T, self.d0)

    def J(self, pi, gamma=None):
        v, _ = self.evaluate(pi, gamma)
        return float(self.d0 @ v)

    # ---------- sampling ----------
    def sample_episodes(self, pi, M, rng, t_max=10_000):
        """Simulate M independent episodes in parallel (vectorized over episodes).

        Returns arrays of shape (T, M): S[t], A[t], R[t] (= R_{t+1}), alive[t] (step t exists),
        where T is the longest episode in the batch.  Dead episodes are padded with zeros.
        """
        s = rng.choice(self.n_s, size=M, p=self.d0)
        alive = np.ones(M, dtype=bool)
        S_list, A_list, R_list, alive_list = [], [], [], []
        cum_pi = np.cumsum(pi, axis=1)
        # cumulative next-state distribution including terminal (index n_s) as the last bin
        P_full = np.concatenate([self.P, self.p_term[..., None]], axis=2)
        cum_P = np.cumsum(P_full, axis=2)
        for _ in range(t_max):
            if not alive.any():
                break
            a = (rng.random(M)[:, None] > cum_pi[s]).sum(axis=1)
            a = np.minimum(a, self.n_a - 1)            # guard against round-off
            rew = self.r[s, a] + self.reward_sd * rng.standard_normal(M)
            s_next = (rng.random(M)[:, None] > cum_P[s, a]).sum(axis=1)
            s_next = np.minimum(s_next, self.n_s)       # n_s == terminal
            S_list.append(np.where(alive, s, 0))
            A_list.append(np.where(alive, a, 0))
            R_list.append(np.where(alive, rew, 0.0))
            alive_list.append(alive.copy())
            alive = alive & (s_next < self.n_s)
            s = np.where(alive, s_next, 0)
        else:
            raise RuntimeError("episodes did not terminate within t_max steps")
        return (np.array(S_list), np.array(A_list), np.array(R_list), np.array(alive_list))


class LinearSoftmax:
    """pi(a|s) = softmax_a( theta . x(s,a) ), Eq. (10.1).  X has shape (S, A, d)."""

    def __init__(self, X):
        self.X = np.asarray(X, dtype=float)
        self.d = self.X.shape[2]

    @staticmethod
    def tabular(n_s, n_a):
        X = np.zeros((n_s, n_a, n_s * n_a))
        for s in range(n_s):
            for a in range(n_a):
                X[s, a, s * n_a + a] = 1.0
        return LinearSoftmax(X)

    def probs(self, theta):
        h = self.X @ theta                              # action preferences h(s, a)
        h = h - h.max(axis=1, keepdims=True)            # numerically stable softmax
        e = np.exp(h)
        return e / e.sum(axis=1, keepdims=True)

    def score(self, theta):
        """psi(s,a) = grad log pi(a|s) = x(s,a) - sum_b pi(b|s) x(s,b)   (Eq. 10.2)."""
        pi = self.probs(theta)
        xbar = np.einsum("sa,sad->sd", pi, self.X)
        return self.X - xbar[:, None, :]


def exact_gradient(mdp, pol, theta, visit_gamma=None):
    """Policy gradient theorem (Eq. 10.9): sum_s eta(s) sum_a q(s,a) grad pi(a|s).

    visit_gamma=None uses the correct state weighting eta_gamma.  visit_gamma=1.0 gives the
    expected value of the common estimator that *drops* the gamma^t factor (Section 12):
    the states are weighted by undiscounted visit counts while q is still the discounted q.
    """
    pi = pol.probs(theta)
    _, q = mdp.evaluate(pi)
    eta = mdp.visits(pi, gamma=visit_gamma)
    grad_pi = pi[..., None] * pol.score(theta)          # grad pi(a|s) = pi(a|s) psi(s,a)
    return np.einsum("s,sa,sad->d", eta, q, grad_pi)


def fd_gradient(mdp, pol, theta, eps=1e-6):
    """Central finite differences of the closed-form J(theta) -- independent of the theorem."""
    g = np.zeros_like(theta)
    for i in range(theta.size):
        e = np.zeros_like(theta)
        e[i] = eps
        g[i] = (mdp.J(pol.probs(theta + e)) - mdp.J(pol.probs(theta - e))) / (2 * eps)
    return g


def random_mdp(n_s, n_a, gamma, rng, reward_mean=1.0, reward_sd=0.5, term_range=(0.05, 0.3)):
    """A random episodic MDP: Dirichlet transitions, per-(s,a) termination probability."""
    P = rng.dirichlet(np.ones(n_s), size=(n_s, n_a))
    p_term = rng.uniform(*term_range, size=(n_s, n_a))
    P = P * (1.0 - p_term)[..., None]
    r = reward_mean + rng.standard_normal((n_s, n_a))
    d0 = rng.dirichlet(np.ones(n_s))
    return TabularMDP(P, r, d0, gamma, reward_sd=reward_sd)


def short_corridor(gamma=1.0):
    """Sutton & Barto's Example 13.1 (Chapter 01, Exercise 12).

    States 0, 1, 2 (terminal to the right of 2); actions 0 = right, 1 = left; reward -1 per step.
    In state 1 the actions are reversed.  From state 0, 'left' stays put.
    """
    P = np.zeros((3, 2, 3))
    P[0, 0, 1] = 1.0   # 0 --right--> 1
    P[0, 1, 0] = 1.0   # 0 --left--> 0 (wall)
    P[1, 0, 0] = 1.0   # 1 --right--> 0 (reversed!)
    P[1, 1, 2] = 1.0   # 1 --left--> 2 (reversed!)
    #                    2 --right--> terminal (row left all zero)
    P[2, 1, 1] = 1.0   # 2 --left--> 1
    r = -np.ones((3, 2))
    return TabularMDP(P, r, d0=np.array([1.0, 0.0, 0.0]), gamma=gamma)


def corridor_policy():
    """All three states look identical: one feature, x(s,right) = 1, x(s,left) = 0,
    so pi(right|s) = sigmoid(theta) in every state."""
    X = np.zeros((3, 2, 1))
    X[:, 0, 0] = 1.0
    return LinearSoftmax(X)
