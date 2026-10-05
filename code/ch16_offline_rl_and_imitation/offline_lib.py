"""Shared pieces for the Chapter 16 CartPole experiments (Sections 6-9).

What is in here
  expert_action / medium_action   the two scripted data-collection policies of Section 6.4
  collect_dataset(...)            roll out a (noisy) policy and return a transition dataset
  Normalizer                      per-feature state standardisation computed from the dataset
  EnsembleMLP(...)                E independent small MLPs (one per seed) trained at once
  evaluate(...)                   run greedy policies on several CartPole-v1 episodes in lockstep
  discounted_mc_returns(...)      per-transition Monte Carlo returns of the logged data

Termination vs truncation.  CartPole-v1 stops an episode when the pole falls or the cart leaves
the track (`terminated`: the next state is absorbing, value 0) or after 500 steps (`truncated`:
the next state is an ordinary state).  Datasets store `terminated` only, so every TD target in
this chapter bootstraps through truncation (NOTATION.md, code conventions).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

ENV_ID = "CartPole-v1"

# --------------------------------------------------------------------------------------------
# Data-collection policies
# --------------------------------------------------------------------------------------------
# The "expert" is a bang-bang LQR controller.  We linearised the CartPole-v1 equations of motion
# about the upright position (gravity 9.8, cart mass 1.0, pole mass 0.1, half-length 0.5) and
# solved the continuous-time Riccati equation with Q = diag(1, 1, 10, 1), R = 0.1 (scipy).  The
# gain below is the result; the expert pushes right whenever -K s > 0.  It balances for the full
# 500 steps from every start we tried, even when 20% of its actions are replaced by random ones.
EXPERT_K = np.array([-3.16227766, -5.91549435, -49.50716842, -12.84439629])


def expert_action(obs: np.ndarray) -> np.ndarray:
    """Expert action(s) for one observation (shape [4]) or a batch (shape [n, 4])."""
    return (obs @ -EXPERT_K > 0).astype(np.int64)


def medium_action(obs: np.ndarray) -> np.ndarray:
    """A consistently mediocre controller: it balances the pole but with a constant bias, so the
    cart drifts and leaves the track after roughly 170 steps (mean return ~170)."""
    return (obs[..., 2] + 0.3 * obs[..., 3] + 0.05 > 0).astype(np.int64)


# name -> (deterministic base policy or None for uniform random, probability of a random action)
BEHAVIOURS = {
    "expert": (expert_action, 0.0),
    "medium": (medium_action, 0.2),
    "random": (None, 1.0),
}


def collect_dataset(name: str, n_transitions: int, seed: int) -> dict:
    """Roll out behaviour `name` until `n_transitions` transitions are logged.

    Returns a dict of numpy arrays: s, a, r, s2, term (1.0 if the pole fell / cart left), ep (episode
    index), t (time step within the episode), plus `returns`, the undiscounted return of every
    completed episode (the last, possibly cut-off episode is not counted).
    """
    base, eps = BEHAVIOURS[name]
    env = gym.make(ENV_ID)
    rng = np.random.default_rng(seed)
    S, A, R, S2, TERM, EP, T = [], [], [], [], [], [], []
    returns = []
    ep = 0
    obs, _ = env.reset(seed=seed)
    ret, t = 0.0, 0
    while len(S) < n_transitions:
        if base is None or rng.random() < eps:
            a = int(rng.integers(2))
        else:
            a = int(base(obs))
        obs2, r, term, trunc, _ = env.step(a)
        S.append(obs); A.append(a); R.append(r); S2.append(obs2); TERM.append(float(term))
        EP.append(ep); T.append(t)
        ret += r; t += 1
        obs = obs2
        if term or trunc:
            returns.append(ret)
            ep += 1
            obs, _ = env.reset(seed=seed + ep)
            ret, t = 0.0, 0
    env.close()
    return dict(s=np.array(S, np.float32), a=np.array(A, np.int64), r=np.array(R, np.float32),
                s2=np.array(S2, np.float32), term=np.array(TERM, np.float32),
                ep=np.array(EP), t=np.array(T), returns=np.array(returns))


def discounted_mc_returns(data: dict, gamma: float) -> np.ndarray:
    """Discounted return-to-go of every logged transition (truncated at the end of the log).

    Episodes cut off by the 500-step limit are not continued, so for them this *under*-states the
    behaviour policy's value; we only use it as a rough scale for the Q-value diagnostics."""
    g = np.zeros(len(data["r"]), np.float64)
    running = 0.0
    for i in reversed(range(len(data["r"]))):
        last_of_episode = (i == len(data["r"]) - 1) or (data["ep"][i + 1] != data["ep"][i])
        if last_of_episode:
            running = 0.0
        running = data["r"][i] + gamma * running
        g[i] = running
    return g


class Normalizer:
    """Standardise states with the dataset mean and std (computed once, never updated)."""

    def __init__(self, s: np.ndarray):
        self.mean = torch.as_tensor(s.mean(0), dtype=torch.float32)
        self.std = torch.as_tensor(s.std(0) + 1e-3, dtype=torch.float32)

    def __call__(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.mean) / self.std


class EnsembleMLP(nn.Module):
    """E independent MLPs evaluated together with batched matrix products.

    Input [E, B, in_dim] -> output [E, B, out_dim]; member e only ever sees x[e].  Because the
    members share no parameters and Adam acts elementwise, training an EnsembleMLP on the sum of
    the members' losses is *exactly* E independent training runs (one per random seed), but
    costs little more than one: on a single CPU thread a small network's training step is
    dominated by per-call overhead, not arithmetic.  Each layer is initialised like nn.Linear.
    """

    def __init__(self, n_members: int, in_dim: int, out_dim: int, hidden=(64, 64)):
        super().__init__()
        dims = [in_dim, *hidden, out_dim]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            bound = 1.0 / np.sqrt(d_in)
            self.weights.append(nn.Parameter(torch.empty(n_members, d_in, d_out).uniform_(-bound, bound)))
            self.biases.append(nn.Parameter(torch.empty(n_members, 1, d_out).uniform_(-bound, bound)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_layers = len(self.weights)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = torch.baddbmm(b, x, w)
            if i < n_layers - 1:
                x = torch.relu(x)
        return x


def to_tensors(data: dict) -> dict:
    return {k: torch.as_tensor(data[k]) for k in ("s", "a", "r", "s2", "term")}


def evaluate(act_fn, n_members: int = 1, n_episodes: int = 10, seed: int = 100_000) -> np.ndarray:
    """Undiscounted returns of `n_members` policies, each on the same `n_episodes` CartPole-v1
    episodes (fixed reset seeds, so members are compared on identical start states).

    act_fn maps observations of shape [n_members, n_episodes, 4] to actions of shape
    [n_members, n_episodes].  All episodes are stepped in lockstep so the network is called once
    per time step.  Returns an array of shape [n_members, n_episodes]."""
    envs = [[gym.make(ENV_ID) for _ in range(n_episodes)] for _ in range(n_members)]
    obs = np.stack([np.stack([env.reset(seed=seed + i)[0] for i, env in enumerate(row)])
                    for row in envs])
    done = np.zeros((n_members, n_episodes), bool)
    rets = np.zeros((n_members, n_episodes))
    while not done.all():
        acts = act_fn(obs)
        for m, i in zip(*np.nonzero(~done)):
            o, r, term, trunc, _ = envs[m][i].step(int(acts[m, i]))
            rets[m, i] += r
            obs[m, i] = o
            done[m, i] = term or trunc
    for row in envs:
        for env in row:
            env.close()
    return rets
