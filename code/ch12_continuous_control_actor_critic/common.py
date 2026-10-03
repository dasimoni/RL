"""Shared building blocks for the Chapter 12 deep-RL scripts (td3.py, sac.py and the experiments).

Nothing here is specific to one algorithm:

  * make_env            Gymnasium env whose action space is rescaled to [-1, 1]^d, so every agent
                        works with *normalised* actions (Section 6.3: the target entropy -dim(A)
                        and TD3's noise scales 0.1 / 0.2 / 0.5 are defined for that range).
  * ReplayBuffer        uniform experience replay storing (s, a, r, s', terminated).  We store
                        `terminated`, NOT `terminated or truncated`: the TD target bootstraps
                        through time-limit truncation (NOTATION.md, "Code conventions").
  * mlp, polyak_update  networks and the target-network averaging w_bar <- tau w + (1-tau) w_bar.
  * evaluate            average undiscounted return of a deterministic policy on fresh episodes.
  * pendulum_true_value exact discounted return of a deterministic policy from given Pendulum
                        states, computed with a batched NumPy copy of Pendulum-v1's dynamics.
                        Used to measure Q-value over-estimation (Section 4.6).
  * check_pendulum_true_value  compares that NumPy copy with step-by-step Gymnasium rollouts
                        (overestimation.py prints the result: max difference 6.2e-5 over 20
                        episodes of 200 steps).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import random  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

torch.set_num_threads(1)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(env_id: str = "Pendulum-v1", seed: int | None = None) -> gym.Env:
    """Environment with actions rescaled to [-1, 1]^d (the agent never sees the true bounds)."""
    env = gym.make(env_id)
    shape = env.action_space.shape
    env = gym.wrappers.RescaleAction(env, min_action=np.full(shape, -1.0, dtype=np.float32),
                                     max_action=np.full(shape, 1.0, dtype=np.float32))
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


# ---------------------------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------------------------
class ReplayBuffer:
    """Fixed-size circular buffer; uniform sampling (Section 3.1)."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int, seed: int = 0):
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros((capacity, act_dim), dtype=np.float32)
        self.r = np.zeros((capacity, 1), dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.term = np.zeros((capacity, 1), dtype=np.float32)  # 1.0 only for true termination
        self.capacity, self.ptr, self.size = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, terminated: bool) -> None:
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.term[i] = s, a, r, s2, float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]), torch.from_numpy(self.s2[idx]),
                torch.from_numpy(self.term[idx]))


# ---------------------------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------------------------
def mlp(in_dim: int, out_dim: int, hidden: int, n_hidden: int = 2) -> nn.Sequential:
    layers, d = [], in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


@torch.no_grad()
def polyak_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """w_bar <- tau * w + (1 - tau) * w_bar   (Eq. 12.7).  _foreach_ ops avoid a Python loop."""
    t_params, s_params = list(target.parameters()), list(source.parameters())
    torch._foreach_mul_(t_params, 1.0 - tau)
    torch._foreach_add_(t_params, s_params, alpha=tau)


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------
def evaluate(act_fn, env_id: str, n_episodes: int, seed: int) -> float:
    """Mean undiscounted return of the deterministic policy `act_fn(obs)->action in [-1,1]^d`."""
    env = make_env(env_id)
    total = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs, r, term, trunc, _ = env.step(act_fn(obs))
            total += float(r)
            done = term or trunc
    env.close()
    return total / n_episodes


# ---------------------------------------------------------------------------------------------
# Exact values for Pendulum (used by overestimation.py)
# ---------------------------------------------------------------------------------------------
def _angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def pendulum_true_value(obs: np.ndarray, act_fn_batch, gamma: float, horizon: int = 1000,
                        first_actions: np.ndarray | None = None) -> np.ndarray:
    """Discounted return of a deterministic policy from each Pendulum observation in `obs`.

    Pendulum-v1 is deterministic, so one rollout per start state gives the exact value
    (up to the horizon cut-off; with gamma = 0.99 and |r| <= 16.3 the neglected tail is at most
    16.3 * gamma**1000 / (1 - gamma) < 1e-1).  The value is the *infinite-horizon* discounted
    return, which is what a critic that bootstraps through the 200-step time limit estimates.

    obs:            (N, 3) array of (cos th, sin th, thdot)
    act_fn_batch:   maps an (N, 3) float32 array to (N, 1) normalised actions in [-1, 1]
    first_actions:  optional (N, 1) normalised actions for the first step (gives q(s, a)).
    """
    th = np.arctan2(obs[:, 1], obs[:, 0]).astype(np.float64)
    thdot = obs[:, 2].astype(np.float64)
    g, m, l, dt, max_speed, max_torque = 10.0, 1.0, 1.0, 0.05, 8.0, 2.0  # Pendulum-v1 defaults
    ret = np.zeros(len(obs))
    disc = 1.0
    for t in range(horizon):
        o = np.stack([np.cos(th), np.sin(th), thdot], axis=1).astype(np.float32)
        a = first_actions if (t == 0 and first_actions is not None) else act_fn_batch(o)
        u = np.clip(np.asarray(a, dtype=np.float64)[:, 0] * max_torque, -max_torque, max_torque)
        cost = _angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * u ** 2
        ret += disc * (-cost)
        thdot = np.clip(thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt,
                        -max_speed, max_speed)
        th = th + thdot * dt
        disc *= gamma
    return ret


def check_pendulum_true_value(n_episodes: int = 20, gamma: float = 0.99, horizon: int = 200,
                              seed: int = 777) -> float:
    """Compare pendulum_true_value with step-by-step Gymnasium rollouts of the same deterministic
    policy from the same start states.  Returns the largest absolute difference in the discounted
    `horizon`-step return.  The policy is a fixed random tanh-linear map of the observation, so the
    trajectories mix saturated and unsaturated torques and hit the velocity limit."""
    rng = np.random.default_rng(seed)
    W, b = rng.normal(0.0, 1.5, size=(3, 1)), rng.normal(0.0, 0.5, size=1)

    def act_batch(o):
        return np.tanh(np.asarray(o, dtype=np.float64) @ W + b).astype(np.float32)

    env = make_env("Pendulum-v1")
    starts, gym_returns = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        starts.append(obs)
        ret, disc = 0.0, 1.0
        for _ in range(horizon):
            obs, r, term, trunc, _ = env.step(act_batch(obs[None, :])[0])
            ret += disc * float(r)
            disc *= gamma
        gym_returns.append(ret)
    env.close()
    ours = pendulum_true_value(np.array(starts), act_batch, gamma, horizon=horizon)
    return float(np.max(np.abs(ours - np.array(gym_returns))))


def smooth(x, k: int = 1):
    """Centered moving average (k odd) used only for plotting."""
    if k <= 1:
        return np.asarray(x)
    x = np.asarray(x, dtype=float)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")
