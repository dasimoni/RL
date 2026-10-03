"""Soft Actor-Critic (SAC) from scratch in PyTorch (Chapter 12, Section 6, Algorithm 12.4).

This is the "applications" version of SAC (Haarnoja et al., 2018b): no separate state-value
network, twin Q-networks with Polyak-averaged targets, a reparameterised tanh-Gaussian policy,
and automatic tuning of the entropy temperature alpha towards a target entropy.

    critic target  y = r + gamma (1 - terminated) [ min_j Q_bar_j(s', a') - alpha log pi(a'|s') ],
                   a' ~ pi(.|s')                                                     (Eq. 12.28)
    actor loss     E_{s~D, xi~N(0,I)} [ alpha log pi(a_theta(s,xi)|s) - min_j Q_j(s, a_theta(s,xi)) ]
                                                                                     (Eq. 12.30)
    alpha loss     E_{s~D, a~pi} [ -alpha (log pi(a|s) + H_bar) ],  H_bar = -dim(A)  (Eq. 12.33)

NOTE on symbols: alpha is the entropy temperature (NOTATION.md flags the collision with the step
size); learning rates are called lr here and eta in the chapter.

Actions are normalised to [-1, 1]^d by common.make_env, so log pi is the density of the squashed
action in [-1, 1]^d and the target entropy -dim(A) refers to that box (Section 6.3).

Run (from the repository root):
  python code/ch12_continuous_control_actor_critic/sac.py             # 3 seeds on Pendulum, figure
  python code/ch12_continuous_control_actor_critic/sac.py --quick     # ~20 s smoke test, no figures
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import FIG_DIR, ReplayBuffer, evaluate, make_env, mlp, polyak_update, set_seed

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


# ---------------------------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------------------------
@dataclass
class SACConfig:
    env_id: str = "Pendulum-v1"
    seed: int = 0
    total_steps: int = 15_000
    learning_starts: int = 1_000   # uniform-random actions (and no updates) until then
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005             # Polyak coefficient (Eq. 12.7)
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    alpha_lr: float = 1e-3
    hidden: int = 128
    utd: int = 1                   # gradient steps per environment step
    autotune: bool = True          # learn alpha (Section 6.5); otherwise alpha = init_alpha forever
    init_alpha: float = 1.0
    target_entropy: float | None = None   # None -> -dim(A)
    reward_scale: float = 1.0      # multiply rewards before storing (used only in experiments)
    eval_every: int = 1_000
    eval_episodes: int = 5


# ---------------------------------------------------------------------------------------------
# The tanh-Gaussian policy (Section 6.3)
# ---------------------------------------------------------------------------------------------
def tanh_log_det_jacobian(u: torch.Tensor) -> torch.Tensor:
    """log |d tanh(u) / du| = log(1 - tanh(u)^2), summed over action dimensions.

    Computed as 2 (log 2 - u - softplus(-2u)), which is exact (Eq. 12.27) and does not lose
    precision when |u| is large, unlike log(1 - tanh(u)**2 + 1e-6)."""
    return (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(dim=-1, keepdim=True)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden)
        self.act_dim = act_dim

    def forward(self, s: torch.Tensor, deterministic: bool = False, with_logp: bool = True):
        mean, log_std = self.net(s).chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        if deterministic:
            u = mean
        else:
            u = mean + std * torch.randn_like(mean)       # reparameterisation: u = mu + sigma * xi
        a = torch.tanh(u)
        if not with_logp:
            return a, None
        # log pi(a|s) = log N(u; mu, sigma^2) - sum_i log(1 - tanh(u_i)^2)          (Eq. 12.26)
        logp_u = (-0.5 * ((u - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        logp = logp_u - tanh_log_det_jacobian(u)
        return a, logp


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


# ---------------------------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------------------------
class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: SACConfig):
        self.cfg = cfg
        self.actor = SquashedGaussianActor(obs_dim, act_dim, cfg.hidden)
        self.q = nn.ModuleList([Critic(obs_dim, act_dim, cfg.hidden) for _ in range(2)])
        self.q_targ = copy.deepcopy(self.q)
        for p in self.q_targ.parameters():
            p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.critic_lr)
        # optimise log(alpha) so that alpha stays positive
        self.log_alpha = torch.tensor(math.log(cfg.init_alpha), requires_grad=cfg.autotune)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr) if cfg.autotune else None
        self.target_entropy = -float(act_dim) if cfg.target_entropy is None else cfg.target_entropy

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        a, _ = self.actor(torch.as_tensor(obs, dtype=torch.float32), deterministic, with_logp=False)
        return a.numpy()

    def act_det(self, obs: np.ndarray) -> np.ndarray:
        """Evaluation policy: the mode-ish action tanh(mean)."""
        return self.act(obs, deterministic=True)

    def update(self, batch) -> dict:
        cfg = self.cfg
        s, a, r, s2, term = batch
        alpha = self.log_alpha.exp().detach()

        # ---- critics: soft Bellman backup with a fresh action sample a' ~ pi(.|s') (Eq. 12.28) ----
        with torch.no_grad():
            a2, logp2 = self.actor(s2)
            q_next = torch.min(self.q_targ[0](s2, a2), self.q_targ[1](s2, a2))
            y = r + cfg.gamma * (1.0 - term) * (q_next - alpha * logp2)
        q1, q2 = self.q[0](s, a), self.q[1](s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # ---- actor: reparameterised KL projection onto exp(Q / alpha) (Eq. 12.30) ----
        a_new, logp = self.actor(s)
        q_new = torch.min(self.q[0](s, a_new), self.q[1](s, a_new))
        actor_loss = (alpha * logp - q_new).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # ---- temperature: dual gradient step towards E[-log pi] = H_bar (Eq. 12.33) ----
        entropy_est = -logp.detach().mean()
        if cfg.autotune:
            alpha_loss = -(self.log_alpha.exp() * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()

        polyak_update(self.q_targ, self.q, cfg.tau)
        return {"critic_loss": critic_loss.item(), "entropy": entropy_est.item(), "alpha": self.alpha,
                "q": q1.mean().item()}


# ---------------------------------------------------------------------------------------------
# Training loop (Algorithm 12.4)
# ---------------------------------------------------------------------------------------------
def train(cfg: SACConfig, verbose: bool = True, label: str = "SAC") -> dict:
    set_seed(cfg.seed)
    env = make_env(cfg.env_id, seed=cfg.seed)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = SACAgent(obs_dim, act_dim, cfg)
    buf = ReplayBuffer(obs_dim, act_dim, cfg.buffer_size, seed=cfg.seed)

    hist = {"step": [], "eval_return": [], "alpha": [], "entropy": [], "train_returns": []}
    obs, _ = env.reset(seed=cfg.seed)
    ep_ret, t0, recent = 0.0, time.time(), []
    for t in range(1, cfg.total_steps + 1):
        a = env.action_space.sample() if t <= cfg.learning_starts else agent.act(obs)  # no extra noise
        obs2, r, term, trunc, _ = env.step(a)
        buf.add(obs, a, cfg.reward_scale * r, obs2, term)
        ep_ret += float(r)
        obs = obs2
        if term or trunc:
            hist["train_returns"].append((t, ep_ret))
            obs, _ = env.reset()
            ep_ret = 0.0

        if t >= cfg.learning_starts:
            for _ in range(cfg.utd):
                info = agent.update(buf.sample(cfg.batch_size))
            recent.append(info["entropy"])

        if t % cfg.eval_every == 0:
            ret = evaluate(agent.act_det, cfg.env_id, cfg.eval_episodes, seed=10_000)
            ent = float(np.mean(recent)) if recent else float("nan")
            recent = []
            hist["step"].append(t)
            hist["eval_return"].append(ret)
            hist["alpha"].append(agent.alpha)
            hist["entropy"].append(ent)
            if verbose:
                print(f"  [{label} seed {cfg.seed}] step {t:6d}  eval return {ret:8.1f}  "
                      f"alpha {agent.alpha:7.4f}  entropy {ent:6.2f}  ({time.time() - t0:5.1f}s)", flush=True)
    env.close()
    hist["agent"] = agent
    hist["time"] = time.time() - t0
    return hist


# ---------------------------------------------------------------------------------------------
# Main: SAC on Pendulum-v1 over several seeds
# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--total-steps", type=int, default=None)
    args = ap.parse_args()

    if args.quick:
        cfg = SACConfig(total_steps=1_500, learning_starts=500, eval_every=500, eval_episodes=1)
        seeds = [0]
    else:
        cfg = SACConfig()
        seeds = list(range(args.seeds))
    if args.total_steps:
        cfg = dataclasses.replace(cfg, total_steps=args.total_steps)
    print(f"SAC on {cfg.env_id} | seeds {seeds} | {dataclasses.asdict(cfg)}")

    runs = []
    n_final = 1 if args.quick else 20
    for sd in seeds:
        h = train(dataclasses.replace(cfg, seed=sd))
        h["final"] = evaluate(h["agent"].act_det, cfg.env_id, n_final, seed=20_000)
        runs.append(h)
        print(f"SAC seed {sd}: final return over {n_final} eval episodes = {h['final']:.1f} "
              f"(alpha {h['alpha'][-1]:.4f}; train time {h['time']:.0f}s)")

    finals = np.array([h["final"] for h in runs])
    curves = np.array([h["eval_return"] for h in runs])
    steps = np.array(runs[0]["step"])
    first_hit = []
    for c in curves:
        hits = np.nonzero(c >= -200)[0]
        first_hit.append(int(steps[hits[0]]) if len(hits) else None)
    print("\n=== Summary ===")
    print(f"SAC: final return ({n_final} eval episodes per seed) mean {finals.mean():.1f}, "
          f"per seed {np.round(finals, 1).tolist()}")
    print(f"SAC: first evaluation with return >= -200 at steps {first_hit}")
    print(f"SAC: mean eval return over the last 5 evaluations {curves[:, -5:].mean():.1f}; "
          f"final alpha per seed {[round(h['alpha'][-1], 4) for h in runs]}; "
          f"wall-clock per seed {np.mean([h['time'] for h in runs]):.0f}s")

    if not args.quick:
        from plot_style import C, kfmt, setup
        plt = setup()
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
        ax = axes[0]
        for i, c in enumerate(curves):
            ax.plot(steps, c, color=C[2], lw=0.9, alpha=0.45, label="individual seeds" if i == 0 else None)
        ax.plot(steps, curves.mean(0), color=C[2], lw=2.4, label=f"SAC, mean of {len(seeds)} seeds")
        # overlay TD3 and DDPG from td3.py if their full runs have been made (same budget, seeds and
        # evaluation protocol: 5 fixed start states every 1,000 steps)
        for algo, col, ls in [("td3", C[0], "--"), ("ddpg", C[1], ":")]:
            f = os.path.join(FIG_DIR, f"{algo}_pendulum_curves.npz")
            if os.path.exists(f):
                d = np.load(f)
                ax.plot(d["steps"], d["curves"].mean(0), color=col, lw=1.8, ls=ls,
                        label=f"{algo.upper()} (td3.py), mean of {len(d['curves'])} seeds")
        ax.axhline(-200, color="#8a8985", ls="--", lw=1)
        ax.set_ylim(-1700, 0)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("evaluation return (5 episodes)")
        ax.set_title("SAC (auto-tuned alpha) vs TD3 and DDPG, Pendulum-v1")
        ax.legend(loc="lower right")
        kfmt(ax)
        ax = axes[1]
        for i, h in enumerate(runs):
            ax.semilogy(steps, h["alpha"], color=C[2], lw=1.2, label="alpha" if i == 0 else None)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("temperature alpha (log scale)", color=C[2])
        ax2 = ax.twinx()
        for i, h in enumerate(runs):
            ax2.plot(steps, h["entropy"], color=C[1], lw=1.2, ls="--",
                     label="policy entropy estimate" if i == 0 else None)
        ax2.axhline(-1.0, color=C[1], lw=0.8, ls=":", label="target entropy -dim(A) = -1")
        ax2.set_ylabel("entropy  E[-log pi] (nats)", color=C[1])
        ax2.spines["right"].set_visible(True)
        ax.set_title("temperature and entropy (each line one seed)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right")
        kfmt(ax)
        fig.tight_layout()
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, "sac_pendulum.png")
        fig.savefig(path)
        np.savez(os.path.join(FIG_DIR, "sac_pendulum_curves.npz"), steps=steps, curves=curves, finals=finals)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
