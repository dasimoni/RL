"""Shared tabular-MDP utilities for the Chapter 19 scripts (not run directly).

Conventions (Chapter 19, Section 1.2):
  * Discounted MDP: P has shape (S, A, S) with P[s, a, s'] = p(s' | s, a); r has shape (S, A)
    with r[s, a] = r(s, a) in [0, 1]; gamma in [0, 1).
  * Finite-horizon (episodic) MDP: P has shape (H, S, A, S) (time-inhomogeneous, step h uses
    P[h]); r has shape (H, S, A). Steps are indexed h = 0, ..., H-1 and V_H = 0.
  * A stochastic policy is an array pi of shape (S, A) (discounted) or (H, S, A) (finite
    horizon) whose rows sum to one.
Everything here is exact linear algebra or backward induction; no sampling.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: scripts only ever write PNG files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# A fixed categorical colour order (colour-blind friendly), used by every figure in this folder.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GREY = "#8a8985"


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 110, "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "legend.fontsize": 8.5, "lines.linewidth": 1.8,
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


# ----------------------------------------------------------------------------------------------
# Random MDPs
# ----------------------------------------------------------------------------------------------
def random_mdp(rng, S, A, conc=1.0, branching=None):
    """Random discounted MDP. Each P(.|s,a) is Dirichlet(conc) over `branching` random successors
    (all S states if branching is None); mean rewards r(s,a) ~ Uniform[0, 1]."""
    if branching is None:
        P = rng.dirichlet(np.full(S, conc), size=(S, A))
    else:
        P = np.zeros((S, A, S))
        for s in range(S):
            for a in range(A):
                succ = rng.choice(S, size=branching, replace=False)
                P[s, a, succ] = rng.dirichlet(np.full(branching, conc))
    r = rng.uniform(0.0, 1.0, size=(S, A))
    return P, r


def random_policy(rng, S, A, conc=1.0):
    return rng.dirichlet(np.full(A, conc), size=S)


# ----------------------------------------------------------------------------------------------
# Discounted MDPs
# ----------------------------------------------------------------------------------------------
def value_iteration(P, r, gamma, tol=1e-12, max_iter=1_000_000):
    """Return q*, v* (to sup-norm accuracy ~tol) by value iteration (Chapter 03)."""
    S = P.shape[0]
    v = np.zeros(S)
    for _ in range(max_iter):
        q = r + gamma * P @ v
        v_new = q.max(axis=1)
        if np.max(np.abs(v_new - v)) < tol * (1 - gamma):
            v = v_new
            break
        v = v_new
    q = r + gamma * P @ v
    return q, q.max(axis=1)


def policy_matrices(P, r, pi):
    """P_pi (S, S) and r_pi (S,) for a stochastic policy pi (S, A)."""
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = np.einsum("sa,sa->s", pi, r)
    return P_pi, r_pi


def policy_eval(P, r, gamma, pi):
    """Exact v_pi (S,) and q_pi (S, A) for a stochastic policy pi (S, A)."""
    S = P.shape[0]
    P_pi, r_pi = policy_matrices(P, r, pi)
    v = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)
    q = r + gamma * P @ v
    return v, q


def occupancy(P, gamma, pi, d0):
    """Normalised discounted state occupancy d^pi_{d0}(s) = (1-gamma) sum_t gamma^t Pr(S_t = s)."""
    S = P.shape[0]
    P_pi, _ = policy_matrices(P, np.zeros(P.shape[:2]), pi)
    return (1 - gamma) * np.linalg.solve((np.eye(S) - gamma * P_pi).T, d0)


def greedy(q):
    """Deterministic greedy policy as an (S, A) one-hot array (first maximiser on ties)."""
    pi = np.zeros_like(q)
    pi[np.arange(q.shape[0]), q.argmax(axis=1)] = 1.0
    return pi


# ----------------------------------------------------------------------------------------------
# Finite-horizon MDPs
# ----------------------------------------------------------------------------------------------
def fh_optimal(P, r):
    """Backward induction. P (H, S, A, S), r (H, S, A). Returns Q* (H, S, A) and V* (H+1, S)."""
    H, S, A, _ = P.shape
    V = np.zeros((H + 1, S))
    Q = np.zeros((H, S, A))
    for h in range(H - 1, -1, -1):
        Q[h] = r[h] + P[h] @ V[h + 1]
        V[h] = Q[h].max(axis=1)
    return Q, V


def fh_eval(P, r, pi):
    """Exact Q^pi (H, S, A) and V^pi (H+1, S) of a (possibly stochastic) policy pi (H, S, A)."""
    H, S, A, _ = P.shape
    V = np.zeros((H + 1, S))
    Q = np.zeros((H, S, A))
    for h in range(H - 1, -1, -1):
        Q[h] = r[h] + P[h] @ V[h + 1]
        V[h] = np.einsum("sa,sa->s", pi[h], Q[h])
    return Q, V


def fh_state_dists(P, pi, d0):
    """State distributions d_h(s) = Pr(S_h = s) for h = 0..H-1 when pi is run from S_0 ~ d0."""
    H, S, A, _ = P.shape
    d = np.zeros((H, S))
    d[0] = d0
    for h in range(H - 1):
        d[h + 1] = np.einsum("s,sa,sat->t", d[h], pi[h], P[h])
    return d


def loglog_slope(x, y):
    """Least-squares slope of log y against log x (used to report empirical rates)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = (x > 0) & (y > 0)
    return np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)[0]
