"""Deep Q-Network (DQN) from scratch in PyTorch, with Double / Dueling / n-step / PER / Noisy flags.

Chapter 09, Sections 2-8 (Algorithm 9.1 is the DQN box; the flags switch on the extensions
of Sections 4-8).  This file is both a small library, imported by the other scripts of the
chapter, and a runnable demo that trains one agent on CartPole-v1.

What is in here
  QNetwork / DuelingQNetwork   Q(s, .; w) as an MLP; dueling head of Eq. (9.9)
  NoisyLinear                  factorised-Gaussian noisy layer (Section 8)
  ReplayBuffer                 uniform experience replay (Section 2.2)
  SumTree, PrioritizedReplay   proportional prioritized replay with IS weights (Section 6, Algorithm 9.2)
  NStepAccumulator             turns 1-step transitions into n-step ones (Section 7, Algorithm 9.3)
  DQNAgent                     action selection + the TD update (Eqs. 9.2, 9.5, 9.8)
  train(cfg)                   the interaction loop of Algorithm 9.1, with periodic evaluation
  evaluate(...)                greedy evaluation that ALSO measures Q-value bias against
                               Monte Carlo returns (Section 4.4)

Termination vs truncation (Section 2.5).  CartPole-v1 ends an episode either because the pole
fell (`terminated`: the next state is absorbing, its value is 0) or because the 500-step time
limit was hit (`truncated`: the next state is a perfectly ordinary state with a positive
value).  We store `terminated` -- not `terminated or truncated` -- in the replay buffer, so the
TD target bootstraps through truncation and stops only at real terminal states.

Noisy Nets (Section 8).  A noisy agent has no epsilon, so it acts uniformly at random until
learning starts and draws fresh weight noise before every action and every gradient step.
`--noisy --noisy-bug` reproduces our first, broken version (noise drawn only at gradient steps,
no random warm-up), which never learns: see Section 8 of the chapter.

Run the demo:  python code/ch09_deep_q_learning/dqn.py [--quick] [--double] [--dueling] ...
               (boolean flags also have --no-... forms, e.g. --no-use-target)
Output (full mode): figures/dqn_demo.png
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Config:
    env_id: str = "CartPole-v1"
    total_steps: int = 50_000        # environment steps (frames)
    gamma: float = 0.99
    lr: float = 5e-4                 # Adam step size
    batch_size: int = 64
    buffer_size: int = 50_000        # replay capacity N
    learning_starts: int = 1_000     # fill the buffer with this many steps before learning
    train_freq: int = 1              # one gradient step every `train_freq` env steps
    target_update: int = 250         # hard copy w^- <- w every this many env steps
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 10_000    # linear epsilon schedule (Section 2.6)
    hidden: int = 64
    loss: str = "huber"              # "huber" (Eq. 9.5) or "mse"
    double: bool = False             # Double DQN target (Eq. 9.8)
    dueling: bool = False            # dueling head (Eq. 9.9)
    noisy: bool = False              # NoisyNet layers instead of epsilon-greedy (Section 8)
    noisy_bug: bool = False          # True reproduces the bug of Section 8 (noise drawn only at
                                     #   gradient steps, no random warm-up); for that demo only
    n_step: int = 1                  # n-step returns (Section 7)
    prioritized: bool = False        # proportional PER (Section 6)
    per_alpha: float = 0.6
    per_beta0: float = 0.4           # annealed linearly to 1 over training
    per_eps: float = 1e-3
    use_target: bool = True          # False: bootstrap from the online network itself
    polyak: float = 0.0              # >0: soft update w^- <- tau w + (1 - tau) w^- every step instead
                                     #     of hard copies every `target_update` steps (Section 2.3)
    reward_noise: float = 0.0        # std of zero-mean Gaussian noise added to TRAINING rewards
    action_copies: int = 1           # >1: every real action is offered this many times (Section 4.4)
    terminal_bug: str = "none"       # "none" | "ignore_terminal" | "truncation_is_terminal"
    eval_every: int = 2_500
    eval_episodes: int = 5
    eval_max_steps: int = 1_000      # evaluation episodes may run past the 500-step limit, so that
    score_cap: int = 500             #   Monte Carlo returns approximate infinite-horizon values
    seed: int = 0


def epsilon_at(cfg: Config, t: int) -> float:
    """Linear annealing from eps_start to eps_end over eps_decay_steps, then constant."""
    frac = min(1.0, t / cfg.eps_decay_steps)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


# --------------------------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------------------------
class NoisyLinear(nn.Module):
    """Factorised-Gaussian noisy linear layer (Fortunato et al., 2018).

    y = (mu_W + sigma_W * eps_W) x + (mu_b + sigma_b * eps_b),  eps_W = f(eps_out) f(eps_in)^T,
    f(x) = sign(x) sqrt(|x|).  The sigmas are learned, so the network decides how much
    parameter noise (hence exploration) to keep in each part of the state space.
    In eval mode (`self.training == False`) the layer uses the mean weights only.
    """

    def __init__(self, n_in, n_out, sigma0=0.5):
        super().__init__()
        self.n_in, self.n_out = n_in, n_out
        bound = 1.0 / math.sqrt(n_in)
        self.w_mu = nn.Parameter(torch.empty(n_out, n_in).uniform_(-bound, bound))
        self.w_sigma = nn.Parameter(torch.full((n_out, n_in), sigma0 / math.sqrt(n_in)))
        self.b_mu = nn.Parameter(torch.empty(n_out).uniform_(-bound, bound))
        self.b_sigma = nn.Parameter(torch.full((n_out,), sigma0 / math.sqrt(n_in)))
        self.register_buffer("eps_in", torch.zeros(n_in))
        self.register_buffer("eps_out", torch.zeros(n_out))
        self.reset_noise()

    @staticmethod
    def _f(x):
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        self.eps_in.copy_(self._f(torch.randn(self.n_in)))
        self.eps_out.copy_(self._f(torch.randn(self.n_out)))

    def forward(self, x):
        if not self.training:
            return F.linear(x, self.w_mu, self.b_mu)
        w = self.w_mu + self.w_sigma * torch.outer(self.eps_out, self.eps_in)
        b = self.b_mu + self.b_sigma * self.eps_out
        return F.linear(x, w, b)


def _linear(n_in, n_out, noisy):
    return NoisyLinear(n_in, n_out) if noisy else nn.Linear(n_in, n_out)


class QNetwork(nn.Module):
    """Plain MLP: s -> [Q(s, a_1), ..., Q(s, a_|A|)]  (one output per action, as in DQN)."""

    def __init__(self, obs_dim, n_actions, hidden=128, noisy=False):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                  _linear(hidden, hidden, noisy), nn.ReLU())
        self.head = _linear(hidden, n_actions, noisy)

    def forward(self, x):
        return self.head(self.body(x))


class DuelingQNetwork(nn.Module):
    """Dueling head, Eq. (9.9):  Q(s,a) = V(s) + A(s,a) - mean_a' A(s,a').

    A shared torso feeds two streams.  Subtracting the mean advantage makes the split into
    V and A identifiable (Section 5.2): without it, adding a constant c to V and subtracting
    it from every A leaves Q unchanged, so neither stream would have a well-defined meaning.
    """

    def __init__(self, obs_dim, n_actions, hidden=128, noisy=False):
        super().__init__()
        self.torso = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.value = nn.Sequential(_linear(hidden, hidden, noisy), nn.ReLU(), _linear(hidden, 1, noisy))
        self.adv = nn.Sequential(_linear(hidden, hidden, noisy), nn.ReLU(),
                                 _linear(hidden, n_actions, noisy))

    def forward(self, x):
        h = self.torso(x)
        v, a = self.value(h), self.adv(h)
        return v + a - a.mean(dim=-1, keepdim=True)


def reset_noise(net: nn.Module):
    for m in net.modules():
        if isinstance(m, NoisyLinear):
            m.reset_noise()


def make_adam(params, lr):
    """Adam; the fused CPU kernel is ~3x cheaper per step for these tiny nets when available."""
    try:
        return torch.optim.Adam(params, lr=lr, fused=True)
    except (RuntimeError, TypeError):
        return torch.optim.Adam(params, lr=lr)


# --------------------------------------------------------------------------------------------
# Replay memories
# --------------------------------------------------------------------------------------------
class ReplayBuffer:
    """Uniform experience replay: a circular array of (s, a, G, s', terminated, discount).

    For 1-step transitions G = R_{t+1} and discount = gamma.  For n-step transitions
    (Section 7) G = G_{t:t+k} and discount = gamma^k, where k <= n is the number of real steps.
    """

    def __init__(self, capacity, obs_dim, rng):
        self.capacity, self.rng = capacity, rng
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros(capacity, np.int64)
        self.ret = np.zeros(capacity, np.float32)
        self.term = np.zeros(capacity, np.float32)
        self.disc = np.zeros(capacity, np.float32)
        self.pos, self.size = 0, 0

    def add(self, s, a, g, s2, term, disc):
        i = self.pos
        self.obs[i], self.act[i], self.ret[i] = s, a, g
        self.next_obs[i], self.term[i], self.disc[i] = s2, term, disc
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return i

    def _gather(self, idx):
        return dict(obs=torch.from_numpy(self.obs[idx]), act=torch.from_numpy(self.act[idx]),
                    ret=torch.from_numpy(self.ret[idx]), next_obs=torch.from_numpy(self.next_obs[idx]),
                    term=torch.from_numpy(self.term[idx]), disc=torch.from_numpy(self.disc[idx]))

    def sample(self, batch_size, beta=None):
        idx = self.rng.integers(0, self.size, size=batch_size)
        batch = self._gather(idx)
        batch["weights"] = torch.ones(batch_size)
        batch["idx"] = idx
        return batch

    def update_priorities(self, idx, td_abs):   # uniform replay ignores priorities
        pass


class SumTree:
    """Binary tree whose internal nodes store the sum of their children (Section 6.2).

    Leaves hold priorities p_i.  `find(u)` returns the leaf i with
    sum_{j<i} p_j <= u < sum_{j<=i} p_j  in O(log N), so sampling i with probability
    p_i / sum_j p_j costs O(log N).  Both operations are vectorised over a batch with numpy:
    every query descends one level per loop iteration.
    """

    def __init__(self, capacity):
        self.n_leaves = 1 << max(1, (capacity - 1).bit_length())   # next power of two
        self.tree = np.zeros(2 * self.n_leaves, np.float64)          # node 1 is the root

    @property
    def total(self):
        return self.tree[1]

    def update(self, idx, priorities):
        node = np.asarray(idx) + self.n_leaves
        self.tree[node] = priorities
        node = np.unique(node // 2)
        while True:                                   # recompute parents level by level
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
            if node[0] == 1:
                break
            node = np.unique(node // 2)

    def find(self, u):
        node = np.ones(len(u), np.int64)
        u = u.copy()
        while node[0] < self.n_leaves:                # all queries are at the same depth
            left = 2 * node
            go_right = u >= self.tree[left]
            u = np.where(go_right, u - self.tree[left], u)
            node = np.where(go_right, left + 1, left)
        return node - self.n_leaves

    def leaf(self, idx):
        return self.tree[np.asarray(idx) + self.n_leaves]

    # Scalar versions: for one transition at a time, plain Python is ~10x faster than numpy.
    def update_one(self, i, p):
        tree, node = self.tree, int(i) + self.n_leaves
        tree[node] = p
        node //= 2
        while node >= 1:
            tree[node] = tree[2 * node] + tree[2 * node + 1]
            node //= 2

    def find_one(self, u):
        tree, node = self.tree, 1
        while node < self.n_leaves:
            left = 2 * node
            if u >= tree[left]:
                u -= tree[left]
                node = left + 1
            else:
                node = left
        return node - self.n_leaves


class PrioritizedReplay(ReplayBuffer):
    """Proportional prioritized experience replay (Schaul et al., 2016), Section 6.

    P(i) = p_i^alpha / sum_k p_k^alpha,  p_i = |delta_i| + eps,
    importance-sampling weights  w_i = (N P(i))^(-beta) / max_j w_j  (max over the minibatch).
    New transitions get the maximum priority seen so far, so each is very likely to be replayed
    soon (a guarantee only for greedy prioritization; here sampling is stochastic).
    """

    def __init__(self, capacity, obs_dim, rng, alpha=0.6, eps=1e-3):
        super().__init__(capacity, obs_dim, rng)
        self.alpha, self.eps = alpha, eps
        self.tree = SumTree(capacity)
        self.max_p = 1.0                              # running max of p_i^alpha

    def add(self, *transition):
        i = super().add(*transition)
        self.tree.update(np.array([i]), np.array([self.max_p]))
        return i

    def sample(self, batch_size, beta=0.4):
        # Stratified sampling: split [0, total) into batch_size equal segments, one draw each.
        total = self.tree.total
        bounds = np.arange(batch_size) * (total / batch_size)
        u = bounds + self.rng.random(batch_size) * (total / batch_size)
        idx = np.minimum(self.tree.find(u), self.size - 1)
        p = self.tree.leaf(idx)
        p = np.where(p > 0, p, self.max_p)            # guard against float round-off at the edges
        prob = p / total
        w = (self.size * prob) ** (-beta)
        w = w / w.max()
        batch = self._gather(idx)
        batch["weights"] = torch.as_tensor(w, dtype=torch.float32)
        batch["idx"] = idx
        return batch

    def update_priorities(self, idx, td_abs):
        p = (np.asarray(td_abs, np.float64) + self.eps) ** self.alpha
        self.tree.update(idx, p)
        self.max_p = max(self.max_p, float(p.max()))


class NStepAccumulator:
    """Builds n-step transitions (s_t, a_t, G_{t:t+k}, s_{t+k}, terminated, gamma^k), Section 7.

    Normally k = n.  When an episode ends (terminated OR truncated) the pending transitions are
    flushed with k < n; for a truncated episode s_{t+k} is the real last observation and the
    target still bootstraps from it, for a terminated one the bootstrap term is switched off.
    """

    def __init__(self, n, gamma):
        self.n, self.gamma, self.q = n, gamma, deque()

    def _make(self, s_last, terminated):
        g, k = 0.0, len(self.q)
        for j, (_, _, r) in enumerate(self.q):
            g += (self.gamma ** j) * r
        s0, a0, _ = self.q[0]
        return (s0, a0, g, s_last, float(terminated), self.gamma ** k)

    def push(self, s, a, r, s_next, terminated, truncated):
        self.q.append((s, a, r))
        out = []
        if len(self.q) == self.n:
            out.append(self._make(s_next, terminated))
            self.q.popleft()
        if terminated or truncated:
            while self.q:
                out.append(self._make(s_next, terminated))
                self.q.popleft()
        return out


# --------------------------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------------------------
class DQNAgent:
    """Online network Q(.; w), target network Q(.; w^-), and the semi-gradient TD update."""

    def __init__(self, obs_dim, n_actions, cfg: Config):
        self.cfg, self.n_actions = cfg, n_actions
        Net = DuelingQNetwork if cfg.dueling else QNetwork
        self.q = Net(obs_dim, n_actions, cfg.hidden, cfg.noisy)
        self.q_target = Net(obs_dim, n_actions, cfg.hidden, cfg.noisy)
        self.sync_target()
        self.opt = make_adam(self.q.parameters(), cfg.lr)

    def sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())

    @torch.no_grad()
    def soft_update(self, tau):
        """Polyak averaging  w^- <- tau w + (1 - tau) w^-."""
        for p, p_targ in zip(self.q.parameters(), self.q_target.parameters()):
            p_targ.lerp_(p, tau)

    @torch.no_grad()
    def q_values(self, obs):
        return self.q(torch.as_tensor(obs, dtype=torch.float32))

    def act(self, obs, eps, rng):
        """epsilon-greedy, or greedy w.r.t. a freshly sampled noisy network when cfg.noisy."""
        if self.cfg.noisy:
            if not self.cfg.noisy_bug:
                reset_noise(self.q)        # new weight noise for every action (Section 8)
            return int(self.q_values(obs).argmax().item())
        if rng.random() < eps:
            return int(rng.integers(self.n_actions))
        return int(self.q_values(obs).argmax().item())

    def update(self, batch):
        cfg = self.cfg
        if cfg.noisy:                      # fresh, independent noise for the online and target nets
            reset_noise(self.q)
            reset_noise(self.q_target)
        s, a, g = batch["obs"], batch["act"], batch["ret"]
        s2, term, disc = batch["next_obs"], batch["term"], batch["disc"]

        q_sa = self.q(s).gather(1, a[:, None]).squeeze(1)              # Q(S_t, A_t; w)
        with torch.no_grad():
            boot_net = self.q_target if cfg.use_target else self.q
            q_next = boot_net(s2)                                     # Q(S_{t+n}, . ; w^-)
            if cfg.double:
                # Double DQN, Eq. (9.8): the ONLINE net selects, the TARGET net evaluates.
                a_star = self.q(s2).argmax(1, keepdim=True)
                v_next = q_next.gather(1, a_star).squeeze(1)
            else:
                v_next = q_next.max(1).values                         # Eq. (9.2)
            y = g + disc * (1.0 - term) * v_next                      # (1 - term): no bootstrap
        td = y - q_sa                                                 #   past a TRUE terminal
        if cfg.loss == "huber":
            per_sample = F.smooth_l1_loss(q_sa, y, reduction="none")  # Eq. (9.5), kappa = 1
        else:
            per_sample = 0.5 * td.pow(2)
        loss = (batch["weights"] * per_sample).mean()                 # IS weights = 1 if uniform
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return td.detach().abs().numpy(), float(loss.item()), float(q_sa.detach().mean().item())


# --------------------------------------------------------------------------------------------
# Evaluation: score AND value-estimation bias
# --------------------------------------------------------------------------------------------
def evaluate(agent, cfg: Config, env, seed, n_episodes=None, q_fn=None):
    """Run greedy episodes; return the mean score and the Q-value bias.

    Score: undiscounted return of the first cfg.score_cap steps (for CartPole-v1, whose official
           limit is 500 steps, this is the episode length capped at 500).
    Bias:  for every state S_t with t < score_cap visited by the greedy policy, compare the
           network's prediction  max_a Q(S_t, a)  with the discounted Monte Carlo return
           G_t = sum_k gamma^k R_{t+k+1}  actually obtained from S_t by the same greedy policy.
           Because we bootstrap through truncation, Q estimates the value of the task WITHOUT a
           time limit; the evaluation env therefore runs to cfg.eval_max_steps (1000), so the
           ignored tail is at most gamma^500 / (1 - gamma) = 0.66 for gamma = 0.99.
    `q_fn(obs_batch) -> Q values` lets the distributional agents reuse this (Q = mean of Z).
    """
    n_episodes = n_episodes or cfg.eval_episodes
    q_fn = q_fn or (lambda o: agent.q_values(o))
    was_training = agent.q.training
    agent.q.eval()                                     # noisy layers: use the mean weights
    scores, q_pred, mc = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        states, rewards, done = [], [], False
        while not done:
            states.append(obs)
            a = int(q_fn(obs).argmax().item())
            obs, r, terminated, truncated, _ = env.step(a)
            rewards.append(r)
            done = terminated or truncated
        scores.append(float(np.sum(rewards[:cfg.score_cap])))   # CartPole: min(length, 500)
        # discounted returns-to-go, backwards: G_t = R_{t+1} + gamma G_{t+1}
        G = np.zeros(len(rewards))
        run = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            run = rewards[t] + cfg.gamma * run
            G[t] = run
        m = min(len(rewards), cfg.score_cap)
        with torch.no_grad():
            qs = q_fn(np.asarray(states[:m], np.float32)).max(1).values.numpy()
        q_pred.append(qs)
        mc.append(G[:m])
    if was_training:
        agent.q.train()
    q_pred, mc = np.concatenate(q_pred), np.concatenate(mc)
    return dict(score=float(np.mean(scores)), q_mean=float(q_pred.mean()), mc_mean=float(mc.mean()),
                bias=float((q_pred - mc).mean()))


# --------------------------------------------------------------------------------------------
# Training loop (Algorithm 9.1)
# --------------------------------------------------------------------------------------------
class RedundantActions(gym.ActionWrapper):
    """Offer each of the n real actions `copies` times: action a means real action a mod n.

    The true action values are unchanged (q(s, a) = q(s, a mod n)), but the agent now maximises
    over copies * n separately estimated outputs.  Atari has the same structure: 18 actions, many
    of which are equivalent in most states.  Used to isolate the overestimation bias (Sec. 4.4).
    """

    def __init__(self, env, copies):
        super().__init__(env)
        self.n_real = env.action_space.n
        self.action_space = gym.spaces.Discrete(self.n_real * copies)

    def action(self, a):
        return int(a) % self.n_real


class OneHotObs(gym.ObservationWrapper):
    """Discrete observation i -> one-hot vector e_i (e.g. FrozenLake), so an MLP can read it."""

    def __init__(self, env):
        super().__init__(env)
        self.n = env.observation_space.n
        self.observation_space = gym.spaces.Box(0.0, 1.0, (self.n,), np.float32)

    def observation(self, s):
        x = np.zeros(self.n, np.float32)
        x[int(s)] = 1.0
        return x


def make_env(cfg: Config, max_steps=None):
    env = gym.make(cfg.env_id) if max_steps is None else gym.make(cfg.env_id, max_episode_steps=max_steps)
    if isinstance(env.observation_space, gym.spaces.Discrete):
        env = OneHotObs(env)
    return RedundantActions(env, cfg.action_copies) if cfg.action_copies > 1 else env


def train(cfg: Config, make_agent=None, verbose=True, label=""):
    """Train one agent; returns a dict of learning curves.

    `make_agent(obs_dim, n_actions, cfg)` builds the agent (default: DQNAgent).  Any object with
    the same interface (act, update, sync_target, q_values, attribute q) works, which is how
    distributional.py reuses this loop for C51 and QR-DQN.
    """
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    env = make_env(cfg)
    eval_env = make_env(cfg, cfg.eval_max_steps)
    obs, _ = env.reset(seed=cfg.seed)
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    agent = (make_agent or DQNAgent)(obs_dim, n_actions, cfg)
    Buf = PrioritizedReplay if cfg.prioritized else ReplayBuffer
    kw = dict(alpha=cfg.per_alpha, eps=cfg.per_eps) if cfg.prioritized else {}
    buffer = Buf(cfg.buffer_size, obs_dim, rng, **kw)
    nstep = NStepAccumulator(cfg.n_step, cfg.gamma)

    hist = dict(eval_step=[], eval_score=[], eval_q=[], eval_mc=[], eval_bias=[],
                train_q=[], loss=[], ep_end_step=[], ep_return=[])
    ep_ret, q_acc, loss_acc, n_acc = 0.0, 0.0, 0.0, 0
    t0 = time.perf_counter()
    for t in range(1, cfg.total_steps + 1):
        if cfg.noisy and not cfg.noisy_bug and t < cfg.learning_starts:
            # A noisy agent has no epsilon, so it fills the buffer with a uniformly random policy
            # (the epsilon-greedy agents already act with epsilon >= 0.9 during this warm-up).
            a = int(rng.integers(n_actions))
        else:
            a = agent.act(obs, epsilon_at(cfg, t), rng)
        next_obs, r, terminated, truncated, _ = env.step(a)
        ep_ret += r
        r_train = r + cfg.reward_noise * rng.standard_normal() if cfg.reward_noise else r
        # ---- what we store as "terminal" (deliberate bugs only in the pitfalls experiment) --
        if cfg.terminal_bug == "ignore_terminal":
            stored_term = False                               # BUG: bootstrap through the fall
        elif cfg.terminal_bug == "truncation_is_terminal":
            stored_term = terminated or truncated             # BUG: time limit treated as death
        else:
            stored_term = terminated                          # correct
        for tr in nstep.push(obs, a, r_train, next_obs, stored_term, terminated or truncated):
            buffer.add(*tr)
        obs = next_obs
        if terminated or truncated:
            hist["ep_end_step"].append(t)
            hist["ep_return"].append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()

        if t >= cfg.learning_starts and t % cfg.train_freq == 0:
            beta = cfg.per_beta0 + (1.0 - cfg.per_beta0) * t / cfg.total_steps
            batch = buffer.sample(cfg.batch_size, beta)
            td_abs, loss, q_mean = agent.update(batch)
            buffer.update_priorities(batch["idx"], td_abs)
            q_acc, loss_acc, n_acc = q_acc + q_mean, loss_acc + loss, n_acc + 1
        if cfg.use_target:
            if cfg.polyak > 0:
                agent.soft_update(cfg.polyak)
            elif t % cfg.target_update == 0:
                agent.sync_target()

        if t % cfg.eval_every == 0:
            ev = evaluate(agent, cfg, eval_env, seed=10_000 + cfg.seed * 100)
            hist["eval_step"].append(t)
            hist["eval_score"].append(ev["score"])
            hist["eval_q"].append(ev["q_mean"])
            hist["eval_mc"].append(ev["mc_mean"])
            hist["eval_bias"].append(ev["bias"])
            hist["train_q"].append(q_acc / max(n_acc, 1))
            hist["loss"].append(loss_acc / max(n_acc, 1))
            q_acc, loss_acc, n_acc = 0.0, 0.0, 0
            if verbose:
                eps_txt = "n/a " if cfg.noisy else f"{epsilon_at(cfg, t):.2f}"
                print(f"  {label}step {t:6d}  eps {eps_txt}  eval score {ev['score']:6.1f}"
                      f"  Q {ev['q_mean']:7.2f}  MC {ev['mc_mean']:6.2f}  bias {ev['bias']:+7.2f}"
                      f"  ({time.perf_counter() - t0:5.1f}s)", flush=True)
    hist["runtime_s"] = time.perf_counter() - t0
    hist["agent"] = agent
    env.close()
    eval_env.close()
    return hist


def config_from_args(argv=None, **overrides):
    """Shared command-line flags: every Config field can be set as --field value."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--quick", action="store_true", help="smoke test, no figures")
    for f in dataclasses.fields(Config):
        if f.type in ("bool", bool):             # gives --double / --no-double, --use-target / --no-use-target
            p.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name, action=argparse.BooleanOptionalAction,
                           default=None)
        else:
            typ = {"int": int, "float": float, "str": str}.get(f.type, f.type)
            p.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name, type=typ, default=None)
    args = p.parse_args(argv)
    cfg = Config(**overrides)
    for f in dataclasses.fields(Config):
        v = getattr(args, f.name)
        if v is not None:
            setattr(cfg, f.name, v)
    return cfg, args.quick


def main():
    cfg, quick = config_from_args()
    if quick:
        cfg.total_steps, cfg.eval_every, cfg.learning_starts, cfg.eps_decay_steps = 3_000, 1_000, 500, 2_000
    print("DQN demo on", cfg.env_id, "| seed", cfg.seed, "| quick" if quick else "| full")
    print("  config:", {k: v for k, v in dataclasses.asdict(cfg).items()})
    hist = train(cfg)
    scores = np.array(hist["eval_score"])
    print(f"Done in {hist['runtime_s']:.1f}s. Episodes: {len(hist['ep_return'])}. "
          f"Final eval score {scores[-1]:.1f}; best {scores.max():.1f}; "
          f"mean over last 5 evals {scores[-5:].mean():.1f}; final Q bias {hist['eval_bias'][-1]:+.2f}")
    if quick:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import setup, C, GREY
    setup()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(hist["ep_end_step"], hist["ep_return"], color=GREY, lw=0.8, alpha=0.6,
               label="training episode return (eps-greedy)")
    ax[0].plot(hist["eval_step"], hist["eval_score"], "o-", color=C[0], ms=4, label="greedy evaluation (5 episodes)")
    ax[0].set(xlabel="environment steps", ylabel="episode return", title="DQN on CartPole-v1: performance",
              ylim=(0, 640))
    ax[0].legend(loc="upper left", ncol=2, fontsize=8, frameon=True, framealpha=0.9, edgecolor="none")
    ax[1].plot(hist["eval_step"], hist["eval_q"], "o-", color=C[1], ms=4, label=r"predicted $\max_a Q(S_t,a)$")
    ax[1].plot(hist["eval_step"], hist["eval_mc"], "s--", color=C[2], ms=4, label=r"Monte Carlo return $G_t$")
    ax[1].axhline(1 / (1 - cfg.gamma), color=GREY, ls=":", lw=1, label=r"$1/(1-\gamma)=100$")
    ax[1].set(xlabel="environment steps", ylabel="discounted value",
              title="Value estimates on greedy-policy states")
    ax[1].legend(loc="lower right")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, "dqn_demo.png")
    fig.savefig(path)
    print("saved", path)


if __name__ == "__main__":
    main()
