"""DDPG and TD3 from scratch in PyTorch (Chapter 12, Sections 3-4, Algorithms 12.1 and 12.2).

One agent class covers both algorithms, because TD3 *is* DDPG plus three switches:

    twin_q        clipped double-Q learning: two critics, target uses their minimum   (Eq. 12.9)
    policy_delay  update the actor and the target networks once every d critic updates (Sec. 4.4)
    target_noise  target policy smoothing: clipped Gaussian noise on the target action (Eq. 12.10)

With twin_q=False, policy_delay=1, target_noise=0 the agent is "DDPG-style": the DDPG algorithm
of Lillicrap et al. (2016) with the modern hyperparameters of Fujimoto et al.'s "our DDPG"
(no batch norm, no critic weight decay, same networks and learning rates as TD3), so that any
difference from TD3 is due to the three switches alone.

Actions are normalised to [-1, 1] (common.make_env), so the noise scales 0.1 / 0.2 / 0.5 are the
TD3 paper's.  Exploration noise is Gaussian by default; Ornstein-Uhlenbeck is available.
Time-limit truncation is bootstrapped, termination is not (the buffer stores `terminated`).

Run (from the repository root):
  python code/ch12_continuous_control_actor_critic/td3.py            # TD3, 3 seeds, Pendulum, figure
  python code/ch12_continuous_control_actor_critic/td3.py --quick    # ~20 s smoke test, no figures
  python code/ch12_continuous_control_actor_critic/td3.py --algo ddpg --seeds 1
  python code/ch12_continuous_control_actor_critic/td3.py --noise ou     # Exercise 9 (no figure)
  python code/ch12_continuous_control_actor_critic/td3.py --expl-noise 0.38   # Exercise 9 (no figure)
  python code/ch12_continuous_control_actor_critic/td3.py --nstep 3      # Exercise 15 (no figure)
"""
from __future__ import annotations

import argparse
import collections
import copy
import dataclasses
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import FIG_DIR, ReplayBuffer, evaluate, make_env, mlp, polyak_update, set_seed


# ---------------------------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------------------------
@dataclass
class TD3Config:
    env_id: str = "Pendulum-v1"
    seed: int = 0
    total_steps: int = 15_000
    learning_starts: int = 1_000   # uniform-random actions (and no updates) until then
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005             # Polyak coefficient (Eq. 12.7); NOT a temperature here
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    hidden: int = 128              # two hidden layers of this width, actor and critics
    utd: int = 1                   # update-to-data ratio: gradient steps per environment step
    # exploration (Section 3.4)
    noise_type: str = "gaussian"   # "gaussian" or "ou"
    expl_noise: float = 0.1        # std of the Gaussian exploration noise
    ou_theta: float = 0.15         # OU mean-reversion rate kappa (DDPG paper value)
    ou_sigma: float = 0.2          # OU diffusion (DDPG paper value)
    # the three TD3 ingredients (Section 4)
    twin_q: bool = True
    policy_delay: int = 2
    target_noise: float = 0.2      # std of target-policy-smoothing noise; 0 disables it
    noise_clip: float = 0.5
    # evaluation
    eval_every: int = 1_000
    eval_episodes: int = 5
    # n-step returns (Exercise 15); n_step = 1 is the standard one-step target of Eqs. 12.5/12.9
    n_step: int = 1


def ddpg_config(**kw) -> TD3Config:
    """DDPG-style learner: TD3 with all three ingredients switched off."""
    base = dict(twin_q=False, policy_delay=1, target_noise=0.0)
    base.update(kw)
    return TD3Config(**base)


# ---------------------------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------------------------
class Actor(nn.Module):
    """Deterministic policy mu_theta(s) in [-1, 1]^d (tanh output layer)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(s))


class Critic(nn.Module):
    """Q_w(s, a): the action is concatenated with the state at the input."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))


class OUNoise:
    """Ornstein-Uhlenbeck process  x <- x + kappa (0 - x) dt + sigma sqrt(dt) xi  (Eq. 12.8).

    `theta` here is the OU mean-reversion rate (kappa in the chapter), not a policy parameter."""

    def __init__(self, dim: int, theta: float, sigma: float, rng: np.random.Generator, dt: float = 1.0):
        self.dim, self.theta, self.sigma, self.dt, self.rng = dim, theta, sigma, dt, rng
        self.x = np.zeros(dim)

    def reset(self) -> None:
        self.x = np.zeros(self.dim)

    def __call__(self) -> np.ndarray:
        self.x = self.x - self.theta * self.x * self.dt + \
            self.sigma * np.sqrt(self.dt) * self.rng.standard_normal(self.dim)
        return self.x


# ---------------------------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------------------------
class TD3Agent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: TD3Config):
        self.cfg = cfg
        self.actor = Actor(obs_dim, act_dim, cfg.hidden)
        self.actor_targ = copy.deepcopy(self.actor)
        n_critics = 2 if cfg.twin_q else 1
        self.q = nn.ModuleList([Critic(obs_dim, act_dim, cfg.hidden) for _ in range(n_critics)])
        self.q_targ = copy.deepcopy(self.q)
        for p in list(self.actor_targ.parameters()) + list(self.q_targ.parameters()):
            p.requires_grad_(False)  # target networks are never trained by gradient descent
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.critic_lr)
        self.n_updates = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()

    def update(self, batch) -> dict:
        cfg = self.cfg
        if len(batch) == 5:                  # one-step transitions: bootstrap discount is gamma
            s, a, r, s2, term = batch
            disc = cfg.gamma
        else:                                # n-step transitions carry their own gamma^k (Ex. 15)
            s, a, r, s2, term, disc = batch

        # ---- critic: regress every Q_j onto one shared target y (Eqs. 12.5, 12.9, 12.10) ----
        with torch.no_grad():
            a2 = self.actor_targ(s2)
            if cfg.target_noise > 0:   # target policy smoothing
                eps = (torch.randn_like(a2) * cfg.target_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
                a2 = (a2 + eps).clamp(-1.0, 1.0)
            q_next = torch.min(torch.cat([qt(s2, a2) for qt in self.q_targ], dim=1), dim=1, keepdim=True)[0]
            y = r + disc * (1.0 - term) * q_next           # no bootstrap only on *termination*
        q_preds = [q(s, a) for q in self.q]
        critic_loss = sum(F.mse_loss(qp, y) for qp in q_preds)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.n_updates += 1
        info = {"critic_loss": critic_loss.item(), "q": q_preds[0].mean().item()}

        # ---- actor and targets: every `policy_delay` critic updates (Section 4.4) ----
        if self.n_updates % cfg.policy_delay == 0:
            # Deterministic policy gradient (Eq. 12.6): ascend Q_1(s, mu_theta(s)).  Autograd
            # computes grad_theta mu(s) * grad_a Q(s, a)|_{a = mu(s)} by the chain rule.
            actor_loss = -self.q[0](s, self.actor(s)).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            # the critic received gradients from actor_loss too; they are discarded by
            # critic_opt.zero_grad() before its next step.
            polyak_update(self.actor_targ, self.actor, cfg.tau)
            polyak_update(self.q_targ, self.q, cfg.tau)
            info["actor_loss"] = actor_loss.item()
        return info


# ---------------------------------------------------------------------------------------------
# n-step returns (Exercise 15)
# ---------------------------------------------------------------------------------------------
class NStepReplayBuffer(ReplayBuffer):
    """Replay buffer whose transitions (s, a, R, s_k, term, gamma^k) carry their own bootstrap
    discount, because a window cut short by the end of an episode holds k < n rewards."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int, seed: int = 0):
        super().__init__(obs_dim, act_dim, capacity, seed)
        self.disc = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, s, a, r, s2, terminated: bool, disc: float = 1.0) -> None:
        self.disc[self.ptr] = disc           # written before super().add advances the pointer
        super().add(s, a, r, s2, terminated)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]), torch.from_numpy(self.s2[idx]),
                torch.from_numpy(self.term[idx]), torch.from_numpy(self.disc[idx]))


class NStepAccumulator:
    """Turns one-step transitions into n-step ones:
        (S_t, A_t, R_{t+1} + gamma R_{t+2} + ... + gamma^{k-1} R_{t+k}, S_{t+k}, term, gamma^k).

    The window never crosses an episode boundary.  When the episode ends, every pending start
    state is flushed with a *shorter* window (k < n) that ends at the last next-state:
      * terminated: term = 1, so the target is the k-step reward sum alone;
      * truncated:  term = 0, so the target still bootstraps from S_{t+k} with gamma^k.
    Treating truncation like termination here would reintroduce the classic bug (Section 1.3)."""

    def __init__(self, n: int, gamma: float):
        self.n, self.gamma, self.window = n, gamma, collections.deque()

    def push(self, s, a, r, s2, terminated: bool, truncated: bool) -> list:
        self.window.append((s, a, float(r)))
        out = []
        if terminated or truncated:
            while self.window:                   # flush: windows of length k = n-1, ..., 1
                out.append(self._make(s2, terminated))
                self.window.popleft()
        elif len(self.window) == self.n:
            out.append(self._make(s2, False))
            self.window.popleft()
        return out

    def _make(self, s_last, terminated: bool):
        ret = sum(self.gamma ** k * r for k, (_, _, r) in enumerate(self.window))
        s0, a0, _ = self.window[0]
        return s0, a0, ret, s_last, terminated, self.gamma ** len(self.window)


# ---------------------------------------------------------------------------------------------
# Training loop (Algorithm 12.2)
# ---------------------------------------------------------------------------------------------
def train(cfg: TD3Config, measure=None, verbose: bool = True, label: str = "TD3") -> dict:
    """Train one agent.  `measure(agent, buffer, step)` (optional) is called at every evaluation
    and its returned dict is appended to the history (overestimation.py uses this)."""
    set_seed(cfg.seed)
    env = make_env(cfg.env_id, seed=cfg.seed)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = TD3Agent(obs_dim, act_dim, cfg)
    if cfg.n_step == 1:
        buf = ReplayBuffer(obs_dim, act_dim, cfg.buffer_size, seed=cfg.seed)
    else:
        buf = NStepReplayBuffer(obs_dim, act_dim, cfg.buffer_size, seed=cfg.seed)
        nstep = NStepAccumulator(cfg.n_step, cfg.gamma)
    rng = np.random.default_rng(cfg.seed + 12345)
    ou = OUNoise(act_dim, cfg.ou_theta, cfg.ou_sigma, rng)

    hist = {"step": [], "eval_return": [], "train_returns": [], "measure": []}
    obs, _ = env.reset(seed=cfg.seed)
    ep_ret, t0 = 0.0, time.time()
    for t in range(1, cfg.total_steps + 1):
        # ---- act ----
        if t <= cfg.learning_starts:
            a = env.action_space.sample()                        # uniform warm-up
        else:
            noise = ou() if cfg.noise_type == "ou" else rng.normal(0.0, cfg.expl_noise, act_dim)
            a = np.clip(agent.act(obs) + noise, -1.0, 1.0)
        obs2, r, term, trunc, _ = env.step(a)
        if cfg.n_step == 1:
            buf.add(obs, a, r, obs2, term)                       # store `terminated` only
        else:
            for tr in nstep.push(obs, a, r, obs2, term, trunc):
                buf.add(*tr)
        ep_ret += float(r)
        obs = obs2
        if term or trunc:
            hist["train_returns"].append((t, ep_ret))
            obs, _ = env.reset()
            ou.reset()
            ep_ret = 0.0

        # ---- learn ----
        if t >= cfg.learning_starts:
            for _ in range(cfg.utd):
                agent.update(buf.sample(cfg.batch_size))

        # ---- evaluate ----
        if t % cfg.eval_every == 0:
            ret = evaluate(agent.act, cfg.env_id, cfg.eval_episodes, seed=10_000)
            hist["step"].append(t)
            hist["eval_return"].append(ret)
            if measure is not None:
                hist["measure"].append(measure(agent, buf, t))
            if verbose:
                extra = ""
                if measure is not None:
                    m = hist["measure"][-1]
                    extra = "  " + "  ".join(f"{k}={v:8.1f}" for k, v in m.items())
                print(f"  [{label} seed {cfg.seed}] step {t:6d}  eval return {ret:8.1f}{extra}"
                      f"  ({time.time() - t0:5.1f}s)", flush=True)
    env.close()
    hist["agent"] = agent
    hist["time"] = time.time() - t0
    return hist


# ---------------------------------------------------------------------------------------------
# Main: TD3 (or DDPG) on Pendulum-v1 over several seeds
# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--algo", choices=["td3", "ddpg"], default="td3")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--noise", choices=["gaussian", "ou"], default="gaussian",
                    help="exploration noise process (Section 3.4; Exercise 9)")
    ap.add_argument("--expl-noise", type=float, default=None,
                    help="std of the Gaussian exploration noise (default 0.1; Exercise 9)")
    ap.add_argument("--nstep", type=int, default=1, help="n-step returns (Exercise 15)")
    args = ap.parse_args()

    make = TD3Config if args.algo == "td3" else ddpg_config
    if args.quick:
        cfg = make(total_steps=1_500, learning_starts=500, eval_every=500, eval_episodes=1, seed=0)
        seeds = [0]
    else:
        cfg = make()
        seeds = list(range(args.seeds))
    if args.total_steps:
        cfg = dataclasses.replace(cfg, total_steps=args.total_steps)
    cfg = dataclasses.replace(cfg, noise_type=args.noise, n_step=args.nstep)
    if args.expl_noise is not None:
        cfg = dataclasses.replace(cfg, expl_noise=args.expl_noise)
    name = args.algo.upper() + (" (OU noise)" if args.noise == "ou" else "")
    if args.noise == "gaussian" and args.expl_noise is not None:
        name += f" (Gaussian noise {args.expl_noise:g})"
    if args.nstep > 1:
        name += f" ({args.nstep}-step)"
    # only the default configuration writes the chapter's figure and curves
    default_run = args.noise == "gaussian" and args.expl_noise is None and args.nstep == 1 \
        and args.total_steps is None
    print(f"{name} on {cfg.env_id} | seeds {seeds} | {dataclasses.asdict(cfg)}")

    runs = []
    n_final = 1 if args.quick else 20      # fresh evaluation episodes for the final score
    for sd in seeds:
        h = train(dataclasses.replace(cfg, seed=sd), label=name)
        final = evaluate(h["agent"].act, cfg.env_id, n_final, seed=20_000)
        h["final"] = final
        runs.append(h)
        print(f"{name} seed {sd}: final return over {n_final} eval episodes "
              f"= {final:.1f}  (train time {h['time']:.0f}s)")

    finals = np.array([h["final"] for h in runs])
    curves = np.array([h["eval_return"] for h in runs])
    steps = np.array(runs[0]["step"])
    first_hit = []
    for c in curves:   # first evaluation at which the 5-episode mean return is >= -200
        hits = np.nonzero(c >= -200)[0]
        first_hit.append(int(steps[hits[0]]) if len(hits) else None)
    print("\n=== Summary ===")
    print(f"{name}: final return ({n_final} eval episodes per seed) mean {finals.mean():.1f}, "
          f"per seed {np.round(finals, 1).tolist()}")
    print(f"{name}: first evaluation with return >= -200 at steps {first_hit}")
    print(f"{name}: mean eval return over the last 5 evaluations "
          f"{curves[:, -5:].mean():.1f}; wall-clock per seed {np.mean([h['time'] for h in runs]):.0f}s")

    if not args.quick and default_run:
        from plot_style import C, kfmt, setup
        plt = setup()
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for i, c in enumerate(curves):
            ax.plot(steps, c, color=C[0], lw=0.9, alpha=0.45, label="individual seeds" if i == 0 else None)
        ax.plot(steps, curves.mean(0), color=C[0], lw=2.4, label=f"{name}, mean of {len(seeds)} seeds")
        ax.axhline(-200, color="#8a8985", ls="--", lw=1, label="-200")
        ax.set_xlabel("environment steps")
        ax.set_ylabel("evaluation return (5 episodes)")
        ax.set_title(f"{name} on Pendulum-v1")
        ax.set_ylim(-1700, 0)
        kfmt(ax)
        ax.legend(loc="lower right")
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, f"{args.algo}_pendulum.png")
        fig.tight_layout()
        fig.savefig(path)
        np.savez(os.path.join(FIG_DIR, f"{args.algo}_pendulum_curves.npz"), steps=steps, curves=curves,
                 finals=finals)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
