"""A small, instrumented A2C for CartPole-v1 (Chapter 20, Sections 5.4 and 7.4).

Used by seed_variance.py. Running this file directly trains one agent and
prints the diagnostics of Section 7.4 every few updates:

  episode return and length, policy entropy, approximate KL between the
  policy before and after each update (k3 estimator), explained variance of
  the critic, actor and critic gradient norms (before clipping), and the
  critic's prediction V(s_0) at the start of each episode next to the
  discounted return that episode actually obtained (both in original reward
  units, although the agent learns from rewards scaled by reward_scale).

Algorithm (Chapter 10): N parallel environments, n-step rollouts, GAE(lambda)
advantages, one gradient step per rollout. Truncation (the 500-step limit) is
handled by bootstrapping: the reward of the truncated step is augmented with
gamma * V(final observation) and the GAE recursion is cut, exactly as for a
terminal step but without dropping the future value.

Three independent random streams are exposed so that seed_variance.py can
vary them one at a time: network initialization (init_seed), environment
resets (env_seed) and action sampling (act_seed).

After training, the FINAL policy is evaluated (Section 6.1) on a separate,
separately seeded evaluation environment for eval_episodes episodes, both
greedily (argmax action) and stochastically (sampling, with its own generator),
so evaluation never touches the training streams.

Run:  python code/ch20_deep_rl_in_practice/cartpole_a2c.py [--quick]
"""
import argparse
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)


@dataclass
class Config:
    total_steps: int = 80_000
    n_envs: int = 8
    n_steps: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 1e-3
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5          # applied to actor and critic SEPARATELY (Chapter 11, Sec. 8.3)
    reward_scale: float = 0.1           # the agent learns from c * r; returns are logged in original units
    init_seed: int = 0
    env_seed: int = 0
    act_seed: int = 0
    eval_episodes: int = 10


def layer(i, o, gain):
    lin = nn.Linear(i, o)
    nn.init.orthogonal_(lin.weight, gain)
    nn.init.zeros_(lin.bias)
    return lin


def make_nets(obs_dim, n_actions):
    g = np.sqrt(2)
    actor = nn.Sequential(layer(obs_dim, 64, g), nn.Tanh(), layer(64, 64, g), nn.Tanh(), layer(64, n_actions, 0.01))
    critic = nn.Sequential(layer(obs_dim, 64, g), nn.Tanh(), layer(64, 64, g), nn.Tanh(), layer(64, 1, 1.0))
    return actor, critic


def evaluate(actor, cfg: Config, greedy: bool):
    """Returns of the final policy on a separate evaluation environment (no learning)."""
    env = gym.make("CartPole-v1")
    gen = torch.Generator().manual_seed(20_000 + cfg.act_seed)
    out = []
    for i in range(cfg.eval_episodes):
        o, _ = env.reset(seed=1_000_000 + 1000 * cfg.env_seed + i)   # never used in training
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                logits = actor(torch.as_tensor(o, dtype=torch.float32)[None])
                a = int(logits.argmax(-1)) if greedy else int(torch.multinomial(torch.softmax(logits, -1), 1, generator=gen))
            o, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        out.append(total)
    env.close()
    return np.array(out)


def grad_norm(params):
    return float(torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None)))


def train(cfg: Config, log_every=0):
    torch.manual_seed(cfg.init_seed)
    act_gen = torch.Generator().manual_seed(10_000 + cfg.act_seed)
    envs = [gym.make("CartPole-v1") for _ in range(cfg.n_envs)]
    obs = np.stack([e.reset(seed=1000 * cfg.env_seed + i)[0] for i, e in enumerate(envs)]).astype(np.float32)
    actor, critic = make_nets(4, 2)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=cfg.lr)
    N, n = cfg.n_envs, cfg.n_steps
    ep_ret, ep_len, ep_disc = np.zeros(N), np.zeros(N, dtype=int), np.zeros(N)
    with torch.no_grad():
        v_start = critic(torch.as_tensor(obs)).squeeze(-1).numpy().copy()   # V(s_0) of the current episodes
    log = {k: [] for k in ("step", "entropy", "approx_kl", "explained_var", "gn_actor", "gn_critic", "value_mean")}
    episodes = {k: [] for k in ("step", "return", "length", "v_start", "disc_return")}
    n_updates = cfg.total_steps // (N * n)
    step = 0
    for upd in range(n_updates):
        b_obs = np.zeros((n, N, 4), np.float32)
        b_act = np.zeros((n, N), np.int64)
        b_rew = np.zeros((n, N), np.float32)
        b_cut = np.zeros((n, N), np.float32)          # 1 where the episode ended (terminated OR truncated)
        b_val = np.zeros((n, N), np.float32)
        b_logp = np.zeros((n, N), np.float32)
        for t in range(n):
            with torch.no_grad():
                o = torch.as_tensor(obs)
                logits = actor(o)
                a = torch.multinomial(torch.softmax(logits, -1), 1, generator=act_gen).squeeze(-1)
                b_logp[t] = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(1).numpy()
                b_val[t] = critic(o).squeeze(-1).numpy()
            b_obs[t], b_act[t] = obs, a.numpy()
            for i, e in enumerate(envs):
                o2, r, term, trunc, _ = e.step(int(a[i]))
                step += 1
                ep_ret[i] += r
                ep_disc[i] += cfg.gamma ** ep_len[i] * r
                ep_len[i] += 1
                r_learn = cfg.reward_scale * r
                if trunc and not term:
                    # bootstrap through truncation: the future value is not zero
                    with torch.no_grad():
                        r_learn += cfg.gamma * float(critic(torch.as_tensor(o2, dtype=torch.float32)))
                b_rew[t, i] = r_learn
                if term or trunc:
                    b_cut[t, i] = 1.0
                    episodes["step"].append(step)
                    episodes["return"].append(ep_ret[i])
                    episodes["length"].append(ep_len[i])
                    episodes["v_start"].append(v_start[i] / cfg.reward_scale)   # original reward units
                    episodes["disc_return"].append(ep_disc[i])
                    ep_ret[i], ep_len[i], ep_disc[i] = 0.0, 0, 0.0
                    o2, _ = e.reset()
                    with torch.no_grad():
                        v_start[i] = float(critic(torch.as_tensor(o2, dtype=torch.float32)))
                obs[i] = o2
        # GAE(lambda) backwards; the recursion is cut wherever an episode ended
        with torch.no_grad():
            v_last = critic(torch.as_tensor(obs)).squeeze(-1).numpy()
        adv = np.zeros((n, N), np.float32)
        last = np.zeros(N, np.float32)
        for t in reversed(range(n)):
            v_next = v_last if t == n - 1 else b_val[t + 1]
            nonterminal = 1.0 - b_cut[t]
            delta = b_rew[t] + cfg.gamma * v_next * nonterminal - b_val[t]
            last = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last
            adv[t] = last
        ret = adv + b_val
        # one A2C update
        o = torch.as_tensor(b_obs.reshape(-1, 4))
        a = torch.as_tensor(b_act.reshape(-1))
        A = torch.as_tensor(adv.reshape(-1))
        R = torch.as_tensor(ret.reshape(-1))
        logits = actor(o)
        logp_all = torch.log_softmax(logits, -1)
        logp = logp_all.gather(1, a[:, None]).squeeze(1)
        entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
        v = critic(o).squeeze(-1)
        loss = -(logp * A).mean() + cfg.vf_coef * ((v - R) ** 2).mean() - cfg.ent_coef * entropy
        opt.zero_grad()
        loss.backward()
        gn_a, gn_c = grad_norm(actor.parameters()), grad_norm(critic.parameters())
        nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
        nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
        opt.step()
        with torch.no_grad():
            logp_new = torch.log_softmax(actor(o), -1).gather(1, a[:, None]).squeeze(1)
            log_ratio = logp_new - torch.as_tensor(b_logp.reshape(-1))
            approx_kl = float(((log_ratio.exp() - 1) - log_ratio).mean())     # k3 estimator of KL(old || new)
        var_r = float(np.var(ret))
        ev = 1.0 - float(np.var(ret - b_val)) / var_r if var_r > 1e-8 else float("nan")
        for k, val in zip(log, (step, float(entropy.detach()), approx_kl, ev, gn_a, gn_c, float(b_val.mean()) / cfg.reward_scale)):
            log[k].append(val)
        if log_every and (upd + 1) % log_every == 0:
            recent = episodes["return"][-20:]
            print(f"  step {step:6d}  return(last 20 eps) {np.mean(recent) if recent else float('nan'):6.1f}"
                  f"  entropy {float(entropy.detach()):.3f}  approx KL {approx_kl:.2e}  expl. var {ev:+.2f}"
                  f"  |g_actor| {gn_a:.3f}  |g_critic| {gn_c:7.2f}  mean V {b_val.mean() / cfg.reward_scale:6.1f}")
    for e in envs:
        e.close()
    episodes = {k: np.array(v) for k, v in episodes.items()}
    if cfg.eval_episodes:
        episodes["eval_greedy"] = evaluate(actor, cfg, greedy=True)
        episodes["eval_stochastic"] = evaluate(actor, cfg, greedy=False)
    return {k: np.array(v) for k, v in log.items()}, episodes


def curve(episodes, total_steps, n_points=40, window=10_000):
    """Mean return of the episodes that ENDED in each window of steps (a common reporting choice)."""
    grid = np.linspace(total_steps / n_points, total_steps, n_points)
    out = np.full(n_points, np.nan)
    for i, s in enumerate(grid):
        m = (episodes["step"] > s - window) & (episodes["step"] <= s)
        if m.any():
            out[i] = episodes["return"][m].mean()
    return grid, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = Config(total_steps=6_000 if args.quick else 80_000)
    print("A2C on CartPole-v1:", cfg)
    t0 = time.time()
    log, eps = train(cfg, log_every=5 if args.quick else 50)
    print(f"done in {time.time() - t0:.1f} s; {len(eps['return'])} episodes; "
          f"mean return of the last 20 training episodes {eps['return'][-20:].mean():.1f}; final policy on "
          f"{cfg.eval_episodes} evaluation episodes: greedy {eps['eval_greedy'].mean():.1f}, "
          f"stochastic {eps['eval_stochastic'].mean():.1f}")
