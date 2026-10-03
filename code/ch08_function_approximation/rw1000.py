"""The 1000-state random walk (Sutton & Barto, Example 9.1) and its exact quantities.

This module is imported by several scripts in this folder. It provides

* a fast pure-Python episode generator (the policy is fixed, so trajectories do
  not depend on the weights -- we can generate them up front);
* the exact substochastic transition matrix P, expected-reward vector r, true
  values v_pi = (I - P)^{-1} r and the on-policy distribution mu (Section 2 of the
  chapter, eq. for eta(s));
* feature matrices X (one row per state) for every representation used in the
  chapter: state aggregation, polynomials, Fourier cosines, tile coding, RBFs;
* exact linear solutions: the VE-minimiser (projection) and the TD fixed point
  w_TD = A^{-1} b (Section 4).

States are numbered 1..1000 as in the book; array index i holds state i+1.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread: the machine is shared
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import random

import numpy as np

N_STATES = 1000
START = 500
JUMP = 100


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def generate_episode(rng: random.Random) -> tuple[list[int], float]:
    """Return (list of visited non-terminal states S_0..S_{T-1}, final reward).

    Each step jumps uniformly to one of the 100 states on the left or one of the
    100 states on the right; jumping past an edge terminates with reward -1 (left)
    or +1 (right). All other rewards are 0 and gamma = 1, so G_t equals the final
    reward for every t of the episode.
    """
    s = START
    states = []
    while True:
        states.append(s)
        j = rng.randrange(2 * JUMP)          # 0..199
        s += (j - JUMP) if j < JUMP else (j - JUMP + 1)   # -100..-1, +1..+100
        if s < 1:
            return states, -1.0
        if s > N_STATES:
            return states, 1.0


# --------------------------------------------------------------------------- #
# Exact model quantities
# --------------------------------------------------------------------------- #
def model() -> tuple[np.ndarray, np.ndarray]:
    """Substochastic transition matrix P (among non-terminal states) and expected
    one-step reward r(s) = E[R_{t+1} | S_t = s]."""
    P = np.zeros((N_STATES, N_STATES))
    r = np.zeros(N_STATES)
    p = 1.0 / (2 * JUMP)
    for i in range(N_STATES):
        s = i + 1
        for k in range(1, JUMP + 1):
            for nxt in (s - k, s + k):
                if nxt < 1:
                    r[i] -= p
                elif nxt > N_STATES:
                    r[i] += p
                else:
                    P[i, nxt - 1] += p
    return P, r


_CACHE: dict = {}


def exact() -> dict:
    """True values, on-policy distribution and expected episode length."""
    if "v" not in _CACHE:
        P, r = model()
        I = np.eye(N_STATES)
        v = np.linalg.solve(I - P, r)                    # v = r + P v
        h = np.zeros(N_STATES)
        h[START - 1] = 1.0                               # start distribution
        eta = np.linalg.solve((I - P).T, h)              # eta = h + P^T eta
        _CACHE.update(P=P, r=r, v=v, eta=eta, mu=eta / eta.sum(),
                      mean_len=eta.sum())
    return _CACHE


# --------------------------------------------------------------------------- #
# Feature matrices: row i is x(state i+1)
# --------------------------------------------------------------------------- #
def unit(states: np.ndarray) -> np.ndarray:
    """Map states 1..1000 to [0, 1] (Section 6: normalise before polynomial/Fourier)."""
    return (states - 1) / (N_STATES - 1)


ALL = np.arange(1, N_STATES + 1)


def aggregation_features(n_groups: int) -> np.ndarray:
    size = N_STATES // n_groups
    X = np.zeros((N_STATES, n_groups))
    X[np.arange(N_STATES), (ALL - 1) // size] = 1.0
    return X


def polynomial_features(order: int) -> np.ndarray:
    u = unit(ALL)
    return np.stack([u ** i for i in range(order + 1)], axis=1)


def fourier_features(order: int) -> np.ndarray:
    u = unit(ALL)
    return np.stack([np.cos(i * np.pi * u) for i in range(order + 1)], axis=1)


def tile_features(n_tilings: int, width: int = 200, offset: int = 4) -> np.ndarray:
    """1-D tile coding: tilings of `width`-state tiles, tiling k shifted by k*offset states.
    Each tiling has ceil(N/width)+1 tiles so that every shifted tiling covers all states."""
    tiles_per = N_STATES // width + 1
    X = np.zeros((N_STATES, n_tilings * tiles_per))
    for k in range(n_tilings):
        idx = (ALL - 1 + k * offset) // width            # which tile in tiling k
        X[np.arange(N_STATES), k * tiles_per + idx] = 1.0
    return X


def rbf_features(n_centers: int, sigma: float) -> np.ndarray:
    u = unit(ALL)
    c = np.linspace(0, 1, n_centers)
    return np.exp(-((u[:, None] - c[None, :]) ** 2) / (2 * sigma ** 2))


# --------------------------------------------------------------------------- #
# Exact linear solutions (Section 4)
# --------------------------------------------------------------------------- #
def ve(X: np.ndarray, w: np.ndarray) -> float:
    """Mean squared value error VE(w) = sum_s mu(s) (v(s) - x(s)^T w)^2."""
    e = exact()
    return float(e["mu"] @ (e["v"] - X @ w) ** 2)


def rms_unweighted(X: np.ndarray, w: np.ndarray) -> float:
    """Root of the *unweighted* mean squared error over the 1000 states (used by S&B Fig 9.2/9.5)."""
    e = exact()
    return float(np.sqrt(np.mean((e["v"] - X @ w) ** 2)))


def projection_solution(X: np.ndarray) -> np.ndarray:
    """argmin_w VE(w): weighted least squares, w = (X^T D X)^{-1} X^T D v."""
    e = exact()
    D = e["mu"]
    return np.linalg.lstsq(X.T @ (D[:, None] * X), X.T @ (D * e["v"]), rcond=None)[0]


def td_fixed_point(X: np.ndarray) -> np.ndarray:
    """w_TD = A^{-1} b with A = X^T D (I - P) X and b = X^T D r (gamma = 1, episodic)."""
    e = exact()
    D = e["mu"]
    A = X.T @ (D[:, None] * (X - e["P"] @ X))
    b = X.T @ (D * e["r"])
    return np.linalg.lstsq(A, b, rcond=None)[0]


if __name__ == "__main__":
    e = exact()
    print(f"expected episode length = {e['mean_len']:.2f} steps")
    print(f"v(1) = {e['v'][0]:.4f}, v(500) = {e['v'][499]:.4f}, v(1000) = {e['v'][999]:.4f}")
    print(f"mu(500) = {e['mu'][499]:.5f}, its neighbours mu(499) = {e['mu'][498]:.6f}, "
          f"mu(501) = {e['mu'][500]:.6f}; mu(1) = {e['mu'][0]:.6f}")
    seed = 0
    print(f"seed={seed} for the simulated check below")
    rng = random.Random(seed)
    lens = [len(generate_episode(rng)[0]) for _ in range(20000)]
    print(f"simulated mean length (20000 episodes) = {np.mean(lens):.2f}")
