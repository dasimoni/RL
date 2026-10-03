"""Proximal Policy Optimization (PPO-clip) from scratch, in one readable file.

Chapter 11, Sections 7-8 (Algorithm 11.7).  CleanRL-style: everything that matters is in
this file, in the order it happens.  Works for

  * discrete actions   (CartPole-v1):  categorical policy, logits = MLP(s)
  * continuous actions (Pendulum-v1):  Gaussian policy, mean = MLP(s), state-independent
                                        log-std vector (Section 8.6)

and logs the PPO diagnostics discussed in Section 8.5: the KL between the data-collecting
policy and the updated one (exact, on the whole batch, after the K epochs), the per-minibatch
approximate-KL estimators k1 and k3 (computed before each minibatch step, as CleanRL logs them
and as early stopping uses them), clip fraction, policy entropy and the explained variance of
the value function.

The implementation details from Section 8 are all here and switchable by flag:
advantage normalisation, observation normalisation, reward scaling, value clipping,
orthogonal initialisation, learning-rate annealing, global gradient-norm clipping and
early stopping on KL.  Time-limit truncation is bootstrapped (r += gamma * V(s_final)),
termination is not (NOTATION.md, "Code conventions").  Two options exist for the exercises:
the adaptive KL-penalty variant of PPO (Algorithm 11.6; kl_penalty_target) and the
"truncation = termination" bug (bootstrap_truncation=False).

Run (from the repository root):
  python code/ch11_trust_regions_and_ppo/ppo.py               # full: CartPole + Pendulum, 3 seeds, figures
  python code/ch11_trust_regions_and_ppo/ppo.py --quick       # ~10 s smoke test, no figures
  python code/ch11_trust_regions_and_ppo/ppo.py --env Pendulum-v1 --seeds 1 --total-steps 200000
  python code/ch11_trust_regions_and_ppo/ppo.py --env CartPole-v1 --no-figures \
         --set update_epochs=10 lr=1e-3 total_steps=80000 clip_coef=inf kl_penalty_target=0.01
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one CPU thread for numpy/BLAS as well as torch
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.distributions import Categorical, Normal, kl_divergence  # noqa: E402

torch.set_num_threads(1)
# Skip torch.distributions' argument checks (simplex/positivity tests on every construction).
# They never change a result, only cost time: about 15% of a CartPole run.
torch.distributions.Distribution.set_default_validate_args(False)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ---------------------------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------------------------
@dataclass
class PPOConfig:
    env_id: str = "CartPole-v1"
    seed: int = 1
    total_steps: int = 100_000
    num_envs: int = 4              # N parallel environments (stepped in a plain Python loop)
    num_steps: int = 128           # T: rollout length per environment; batch = N*T transitions
    gamma: float = 0.99
    gae_lambda: float = 0.95       # lambda of GAE (Chapter 10)
    lr: float = 2.5e-4
    anneal_lr: bool = True         # linear decay of the learning rate to 0 (Section 8.2)
    update_epochs: int = 4         # K: passes over the batch
    num_minibatches: int = 4
    clip_coef: float = 0.2         # epsilon of the clipped surrogate; math.inf disables clipping
    clip_vloss: bool = True        # PPO2-style value clipping (Section 8.2) ...
    vclip_coef: float = 0.2        # ... with its own range, so "no policy clip" leaves it unchanged
    norm_adv: bool = True          # per-minibatch advantage normalisation (Section 8.2)
    ent_coef: float = 0.01         # c2
    vf_coef: float = 0.5           # c1
    max_grad_norm: float = 0.5     # global gradient-norm clipping
    target_kl: float | None = None  # early stopping of the epoch loop when approx KL exceeds it
    kl_penalty_target: float | None = None  # if set: PPO-penalty (Eq. 11.33, Alg. 11.6) instead of the clip
    kl_beta_init: float = 1.0      # initial penalty coefficient beta for the adaptive KL variant
    bootstrap_truncation: bool = True  # False reproduces the common bug: treat time limits as terminal
    norm_obs: bool = False         # running mean/std observation normalisation (+ clip to [-10, 10])
    norm_reward: bool = False      # divide rewards by running std of the discounted return
    ortho_init: bool = True        # orthogonal init, gains sqrt(2) / 0.01 (policy) / 1 (value)
    hidden: int = 64
    log_std_init: float = 0.0      # Gaussian policies only

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


# Presets used for the chapter's results.  CartPole follows CleanRL's ppo.py defaults plus
# reward scaling: without it the value targets reach ~100, the critic cannot fit them in the
# budget, its gradient dominates the *global* gradient norm that is clipped to 0.5, and the
# policy's updates shrink (KL per update ~60x smaller); learning is much slower (Section 8.3
# reports the measurements).  Pendulum uses the continuous-control
# recipe (obs/reward normalisation, no entropy bonus, longer rollouts, more epochs, a larger
# learning rate than MuJoCo's 3e-4 because the budget is only 200k steps; Section 8.6).
PRESETS = {
    "CartPole-v1": dict(total_steps=150_000, norm_reward=True),
    "Pendulum-v1": dict(total_steps=200_000, num_envs=4, num_steps=512, lr=1e-3,
                        update_epochs=10, num_minibatches=16, gamma=0.99, gae_lambda=0.95,
                        ent_coef=0.0, norm_obs=True, norm_reward=True, target_kl=None),
}


def make_config(env_id: str, **overrides) -> PPOConfig:
    kw = dict(env_id=env_id)
    kw.update(PRESETS.get(env_id, {}))
    kw.update(overrides)
    return PPOConfig(**kw)


# ---------------------------------------------------------------------------------------------
# Running statistics for observation / reward normalisation (Section 8.2)
# ---------------------------------------------------------------------------------------------
class RunningMeanStd:
    """Running mean and variance, merged batch by batch (Chan et al.'s parallel formula)."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        b_mean, b_var, b_count = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean = self.mean + delta * b_count / tot
        m2 = self.var * self.count + b_var * b_count + delta**2 * self.count * b_count / tot
        self.var = m2 / tot
        self.count = tot


class ObsNormalizer:
    def __init__(self, shape, enabled: bool, clip: float = 10.0):
        self.rms, self.enabled, self.clip = RunningMeanStd(shape), enabled, clip

    def __call__(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        if not self.enabled:
            return obs.astype(np.float32)
        if update:
            self.rms.update(obs)
        z = (obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)


class RewardScaler:
    """Scale (do not shift) rewards by the running std of the discounted return.

    This is the 'reward scaling' of Engstrom et al. (2020) / VecNormalize: it keeps the value
    targets O(1) without changing which policy is optimal (a positive rescaling of all rewards).
    """

    def __init__(self, num_envs: int, gamma: float, enabled: bool, clip: float = 10.0):
        self.ret = np.zeros(num_envs)
        self.rms, self.gamma, self.enabled, self.clip = RunningMeanStd(()), gamma, enabled, clip

    def __call__(self, r: np.ndarray, done: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return r.astype(np.float32)
        self.ret = self.ret * self.gamma + r
        self.rms.update(self.ret)
        self.ret[done] = 0.0
        return np.clip(r / np.sqrt(self.rms.var + 1e-8), -self.clip, self.clip).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Networks (Section 8.2: orthogonal initialisation; Section 8.6: state-independent log-std)
# ---------------------------------------------------------------------------------------------
def layer_init(layer: nn.Linear, gain: float, ortho: bool) -> nn.Linear:
    if ortho:
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs (64-64, tanh), as in the PPO paper's continuous experiments."""

    def __init__(self, obs_dim: int, act_dim: int, discrete: bool, cfg: PPOConfig):
        super().__init__()
        h, o, s2 = cfg.hidden, cfg.ortho_init, math.sqrt(2)
        self.discrete = discrete
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, h), s2, o), nn.Tanh(),
            layer_init(nn.Linear(h, h), s2, o), nn.Tanh(),
            layer_init(nn.Linear(h, 1), 1.0, o))
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, h), s2, o), nn.Tanh(),
            layer_init(nn.Linear(h, h), s2, o), nn.Tanh(),
            # gain 0.01: the initial policy is almost uniform / almost centred, so early updates
            # are not dominated by an arbitrary initial preference.
            layer_init(nn.Linear(h, act_dim), 0.01, o))
        if not discrete:
            self.log_std = nn.Parameter(torch.full((act_dim,), cfg.log_std_init))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def dist(self, obs: torch.Tensor):
        out = self.actor(obs)
        if self.discrete:
            return Categorical(logits=out)
        return Normal(out, self.log_std.exp().expand_as(out))

    def evaluate(self, obs, actions):
        """log pi(a|s), entropy H[pi(.|s)], v(s) for a batch (summing over action dimensions)."""
        d = self.dist(obs)
        if self.discrete:
            return d.log_prob(actions), d.entropy(), self.value(obs)
        return d.log_prob(actions).sum(-1), d.entropy().sum(-1), self.value(obs)


# ---------------------------------------------------------------------------------------------
# Generalised advantage estimation (Chapter 10), with episode boundaries
# ---------------------------------------------------------------------------------------------
def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """rewards, values, dones: arrays (T, N); dones[t] = 1 if the transition at t ended an episode.
    rewards[t] is the reward R_{t+1} that follows the action at step t (stored at index t).

    delta_t = r_t + gamma (1 - done_t) V(s_{t+1}) - V(s_t)
    A_t     = delta_t + gamma lambda (1 - done_t) A_{t+1}
    Truncated episodes already carry their bootstrap gamma*V(s_final) inside r_t.
    """
    T = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nonterminal * next_v - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    return adv, adv + values  # advantages and lambda-returns (value targets)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """1 - Var[y - y_hat] / Var[y]: 1 = perfect, 0 = no better than a constant, < 0 = worse."""
    var_y = np.var(y_true)
    return float("nan") if var_y == 0 else float(1.0 - np.var(y_true - y_pred) / var_y)


# ---------------------------------------------------------------------------------------------
# PPO training loop (Algorithm 11.7)
# ---------------------------------------------------------------------------------------------
def train(cfg: PPOConfig, verbose: bool = True, log_every: int = 10) -> dict:
    rng_seed = cfg.seed
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    envs = [gym.make(cfg.env_id) for _ in range(cfg.num_envs)]
    discrete = isinstance(envs[0].action_space, gym.spaces.Discrete)
    obs_dim = envs[0].observation_space.shape[0]
    act_dim = envs[0].action_space.n if discrete else envs[0].action_space.shape[0]
    if not discrete:
        a_low, a_high = envs[0].action_space.low, envs[0].action_space.high

    agent = ActorCritic(obs_dim, act_dim, discrete, cfg)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)
    obs_norm = ObsNormalizer((obs_dim,), cfg.norm_obs)
    rew_scale = RewardScaler(cfg.num_envs, cfg.gamma, cfg.norm_reward)

    N, T = cfg.num_envs, cfg.num_steps
    num_updates = cfg.total_steps // cfg.batch_size
    # Rollout storage (time-major, shape (T, N, ...))
    b_obs = np.zeros((T, N, obs_dim), dtype=np.float32)
    b_act = np.zeros((T, N) if discrete else (T, N, act_dim), dtype=np.int64 if discrete else np.float32)
    b_logp = np.zeros((T, N), dtype=np.float32)
    b_rew = np.zeros((T, N), dtype=np.float32)
    b_done = np.zeros((T, N), dtype=np.float32)
    b_val = np.zeros((T, N), dtype=np.float32)

    raw_obs = np.stack([env.reset(seed=cfg.seed * 1000 + i)[0] for i, env in enumerate(envs)])
    for i, env in enumerate(envs):
        env.action_space.seed(cfg.seed * 1000 + i)
    obs = obs_norm(raw_obs)
    ep_ret, ep_len = np.zeros(N), np.zeros(N, dtype=int)
    episodes = []   # (global_step, undiscounted return, length)
    logs = {k: [] for k in ["step", "approx_kl_k1", "approx_kl_k3", "clipfrac", "entropy",
                            "explained_var", "value_loss", "policy_loss", "epochs_run",
                            "max_ratio_dev", "lr", "grad_norm_actor", "grad_norm_critic",
                            "target_std", "kl_beta", "kl_post", "kl_k3_post",
                            "max_ratio_dev_post", "clipfrac_post"]}
    global_step, t0 = 0, time.time()
    kl_beta = cfg.kl_beta_init

    for update in range(1, num_updates + 1):
        # -- learning-rate annealing: alpha_k = alpha_0 * (1 - (k-1)/U) --
        lr_now = cfg.lr * (1.0 - (update - 1.0) / num_updates) if cfg.anneal_lr else cfg.lr
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        # ------------------------------ 1. collect a rollout with pi_old ------------------
        for t in range(T):
            global_step += N
            b_obs[t] = obs
            with torch.no_grad():
                o = torch.as_tensor(obs)
                d = agent.dist(o)
                a = d.sample()
                logp = d.log_prob(a) if discrete else d.log_prob(a).sum(-1)
                v = agent.value(o)
            a_np = a.numpy()
            b_act[t], b_logp[t], b_val[t] = a_np, logp.numpy(), v.numpy()

            rew = np.zeros(N)
            term = np.zeros(N, dtype=bool)
            trunc = np.zeros(N, dtype=bool)
            final_obs = {}
            for i, env in enumerate(envs):
                # Continuous actions: sample from the unbounded Gaussian, clip only when sending
                # to the environment; the stored action (and its log-prob) stays unclipped.
                act_i = int(a_np[i]) if discrete else np.clip(a_np[i], a_low, a_high)
                o_i, r_i, te_i, tr_i, _ = env.step(act_i)
                rew[i], term[i], trunc[i] = r_i, te_i, tr_i
                ep_ret[i] += r_i
                ep_len[i] += 1
                if te_i or tr_i:
                    episodes.append((global_step, ep_ret[i], ep_len[i]))
                    ep_ret[i], ep_len[i] = 0.0, 0
                    if tr_i and not te_i:
                        final_obs[i] = o_i
                    o_i, _ = env.reset()
                raw_obs[i] = o_i
            done = term | trunc
            r_scaled = rew_scale(rew, done)
            # Bootstrap through truncation: the time limit is not part of the MDP, so the
            # truncated state still has value.  Add gamma * V(s_final) to the last reward.
            if final_obs and cfg.bootstrap_truncation:
                idx = list(final_obs)
                fo = obs_norm(np.stack([final_obs[i] for i in idx]), update=False)
                with torch.no_grad():
                    fv = agent.value(torch.as_tensor(fo)).numpy()
                r_scaled[idx] += cfg.gamma * fv
            b_rew[t], b_done[t] = r_scaled, done.astype(np.float32)
            obs = obs_norm(raw_obs)

        # ------------------------------ 2. advantages and value targets -------------------
        with torch.no_grad():
            last_v = agent.value(torch.as_tensor(obs)).numpy()
        adv, ret = compute_gae(b_rew, b_val, b_done, last_v, cfg.gamma, cfg.gae_lambda)

        # flatten (T, N) -> (T*N,)
        f_obs = torch.as_tensor(b_obs.reshape(-1, obs_dim))
        f_act = torch.as_tensor(b_act.reshape(-1) if discrete else b_act.reshape(-1, act_dim))
        f_logp = torch.as_tensor(b_logp.reshape(-1))
        f_adv = torch.as_tensor(adv.reshape(-1))
        f_ret = torch.as_tensor(ret.reshape(-1))
        f_val = torch.as_tensor(b_val.reshape(-1))

        # The data-collecting policy pi_old on the whole batch, kept for the exact KL after the
        # update (the parameters have not changed since the rollout, so this *is* pi_old).
        with torch.no_grad():
            old_dist = agent.dist(f_obs)

        # ------------------------------ 3. K epochs of minibatch SGD on L^CLIP+VF+S --------
        B, M = cfg.batch_size, cfg.minibatch_size
        use_penalty = cfg.kl_penalty_target is not None
        clipfracs, kl1s, kl3s, max_dev = [], [], [], 0.0
        gn_actor, gn_critic = [], []
        epochs_run = 0
        for epoch in range(cfg.update_epochs):
            perm = np.random.permutation(B)
            for start in range(0, B, M):
                mb = perm[start:start + M]
                new_logp, entropy, new_v = agent.evaluate(f_obs[mb], f_act[mb])
                logratio = new_logp - f_logp[mb]
                ratio = logratio.exp()                      # rho_t(theta) = pi_theta / pi_old

                with torch.no_grad():
                    # Schulman's KL(pi_old || pi_theta) estimators from samples a ~ pi_old, measured
                    # on this minibatch BEFORE its gradient step (so they lag the final policy)
                    kl1s.append((-logratio).mean().item())                    # k1: unbiased, noisy
                    kl3s.append(((ratio - 1.0) - logratio).mean().item())     # k3: unbiased, >= 0
                    # fraction of samples outside [1-eps, 1+eps]; when clipping is disabled we
                    # still measure it with eps = 0.2 so runs can be compared.
                    eps_diag = cfg.clip_coef if math.isfinite(cfg.clip_coef) else 0.2
                    clipfracs.append(((ratio - 1.0).abs() > eps_diag).float().mean().item())
                    max_dev = max(max_dev, (ratio - 1.0).abs().max().item())

                mb_adv = f_adv[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Clipped surrogate (Eq. 11.30), written as a loss to *minimise*:
                # -min(rho A, clip(rho) A) = max(-rho A, -clip(rho) A)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                if use_penalty:
                    # PPO-penalty: unclipped surrogate + beta * KL(pi_old || pi_theta), with the KL
                    # estimated per sample by k3 (its gradient is also unbiased; Section 8.5)
                    pg_loss = pg_loss1.mean() + kl_beta * ((ratio - 1.0) - logratio).mean()

                # Value loss, optionally clipped around the old prediction (Section 8.2)
                if cfg.clip_vloss and math.isfinite(cfg.vclip_coef):
                    v_unclipped = (new_v - f_ret[mb]) ** 2
                    v_clipped = f_val[mb] + torch.clamp(new_v - f_val[mb], -cfg.vclip_coef, cfg.vclip_coef)
                    v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - f_ret[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((new_v - f_ret[mb]) ** 2).mean()

                ent = entropy.mean()
                loss = pg_loss - cfg.ent_coef * ent + cfg.vf_coef * v_loss   # Eq. 11.34

                optimizer.zero_grad()
                loss.backward()
                with torch.no_grad():   # pre-clipping gradient norms of the two networks (Section 8.3)
                    gn_actor.append(math.sqrt(sum(float((q.grad ** 2).sum()) for n, q in agent.named_parameters()
                                                  if not n.startswith("critic"))))
                    gn_critic.append(math.sqrt(sum(float((q.grad ** 2).sum()) for q in agent.critic.parameters())))
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()
            epochs_run += 1
            # Early stopping: the mean k3 estimate over this epoch's minibatches (each measured
            # before its own step -- the cheap estimate the reference implementations use)
            if cfg.target_kl is not None and np.mean(kl3s[-cfg.num_minibatches:]) > cfg.target_kl:
                break

        # The policy this update produced, on the whole batch: the realised KL(pi_old || pi_new)
        # (exact, as TRPO measures it in Section 6.7, and its k3 estimate), the ratios, the entropy.
        with torch.no_grad():
            new_dist = agent.dist(f_obs)
            kl_exact = kl_divergence(old_dist, new_dist)
            ent_all = new_dist.entropy()
            lp_post = new_dist.log_prob(f_act)
            if not discrete:
                kl_exact, ent_all, lp_post = kl_exact.sum(-1), ent_all.sum(-1), lp_post.sum(-1)
            lr_post = lp_post - f_logp
            ratio_post = lr_post.exp()
            kl_k3_post = float(((ratio_post - 1.0) - lr_post).mean())
        if use_penalty:
            # Algorithm 11.6: adapt beta so that the realised KL (k3 on the whole batch) tracks the target
            d = kl_k3_post
            if d < cfg.kl_penalty_target / 1.5:
                kl_beta /= 2.0
            elif d > cfg.kl_penalty_target * 1.5:
                kl_beta *= 2.0

        # ------------------------------ 4. diagnostics ------------------------------------
        logs["step"].append(global_step)
        logs["approx_kl_k1"].append(float(np.mean(kl1s[-cfg.num_minibatches:])))
        logs["approx_kl_k3"].append(float(np.mean(kl3s[-cfg.num_minibatches:])))
        logs["clipfrac"].append(float(np.mean(clipfracs)))
        logs["kl_post"].append(float(kl_exact.mean()))     # realised KL per update (exact)
        logs["kl_k3_post"].append(kl_k3_post)
        logs["max_ratio_dev_post"].append(float((ratio_post - 1.0).abs().max()))
        eps_diag = cfg.clip_coef if math.isfinite(cfg.clip_coef) else 0.2
        logs["clipfrac_post"].append(float(((ratio_post - 1.0).abs() > eps_diag).float().mean()))
        logs["entropy"].append(float(ent_all.mean()))      # batch-mean entropy of the new policy
        logs["explained_var"].append(explained_variance(b_val.reshape(-1), ret.reshape(-1)))
        logs["target_std"].append(float(ret.std()))       # spread of the value targets in the batch
        logs["kl_beta"].append(kl_beta)
        logs["value_loss"].append(float(v_loss.item()))
        logs["policy_loss"].append(float(pg_loss.item()))
        logs["epochs_run"].append(epochs_run)
        logs["max_ratio_dev"].append(max_dev)
        logs["lr"].append(lr_now)
        logs["grad_norm_actor"].append(float(np.mean(gn_actor)))     # mean over this update's minibatches
        logs["grad_norm_critic"].append(float(np.mean(gn_critic)))
        if verbose and (update % log_every == 0 or update == num_updates):
            recent = [e[1] for e in episodes[-20:]]
            print(f"  [{cfg.env_id} seed {cfg.seed}] step {global_step:7d}  "
                  f"ret(last20) {np.mean(recent) if recent else float('nan'):8.1f}  "
                  f"KL {logs['kl_post'][-1]:.4f}  clipfrac {logs['clipfrac'][-1]:.3f}  "
                  f"H {logs['entropy'][-1]:.3f}  EV {logs['explained_var'][-1]:.3f}  "
                  f"({global_step / (time.time() - t0):.0f} steps/s)", flush=True)

    for env in envs:
        env.close()
    return {"cfg": dataclasses.asdict(cfg), "episodes": np.array(episodes), "logs": logs,
            "agent": agent, "obs_norm": obs_norm, "time": time.time() - t0}


def evaluate_policy(result: dict, n_episodes: int = 10, seed: int = 10_000) -> float:
    """Average undiscounted return of the *deterministic* policy (argmax / Gaussian mean)."""
    cfg, agent, obs_norm = result["cfg"], result["agent"], result["obs_norm"]
    env = gym.make(cfg["env_id"])
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    rets = []
    for ep in range(n_episodes):
        o, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                out = agent.actor(torch.as_tensor(obs_norm(o[None], update=False)))[0]
            a = int(out.argmax()) if discrete else np.clip(out.numpy(), env.action_space.low,
                                                           env.action_space.high)
            o, r, te, tr, _ = env.step(a)
            total += r
            done = te or tr
        rets.append(total)
    env.close()
    return float(np.mean(rets))


# ---------------------------------------------------------------------------------------------
# Plotting helpers (shared with ppo_clip_ablation.py and ppo_details_ablation.py)
# ---------------------------------------------------------------------------------------------
def smoothed_curve(episodes: np.ndarray, grid: np.ndarray, window: int = 20) -> np.ndarray:
    """Mean return of the last `window` finished episodes at each grid step (NaN before any)."""
    out = np.full(len(grid), np.nan)
    if len(episodes) == 0:
        return out
    steps, rets = episodes[:, 0], episodes[:, 1]
    for j, g in enumerate(grid):
        k = np.searchsorted(steps, g, side="right")
        if k > 0:
            out[j] = rets[max(0, k - window):k].mean()
    return out


def first_reach(curve: np.ndarray, grid: np.ndarray, level: float) -> float:
    """First grid step at which a smoothed learning curve reaches `level` (NaN if never)."""
    hit = np.nonzero(np.nan_to_num(curve, nan=-np.inf) >= level)[0]
    return float(grid[hit[0]]) if len(hit) else float("nan")


def largest_drop(curve: np.ndarray) -> float:
    """Largest fall of a smoothed learning curve below its running maximum (NaNs ignored)."""
    with np.errstate(invalid="ignore"):
        return float(np.nanmax(np.fmax.accumulate(curve) - curve)) if np.isfinite(curve).any() else float("nan")


def plot_diagnostics(results: list[dict], title: str, path: str):
    from plot_style import C, kfmt, setup
    plt = setup()
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    total = results[0]["cfg"]["total_steps"]
    grid = np.linspace(0, total, 200)
    panels = [("return", "episodic return (last 20 episodes)"),
              ("kl_post", "KL(pi_old || pi_new) after each update"),
              ("clipfrac", "clip fraction |rho-1| > eps"),
              ("entropy", "policy entropy (nats)"),
              ("explained_var", "explained variance of V"),
              ("approx_kl_k1", "k1 KL estimate, pre-step minibatches (can be < 0)")]
    styles = ["-", "--", ":"]
    for ax, (key, label) in zip(axes.flat, panels):
        for i, res in enumerate(results):
            lab = f"seed {res['cfg']['seed']}"
            if key == "return":
                ax.plot(grid, smoothed_curve(res["episodes"], grid), color=C[i], ls=styles[i % 3], lw=1.6, label=lab)
            else:
                ax.plot(res["logs"]["step"], res["logs"][key], color=C[i], ls=styles[i % 3], lw=1.2, label=lab)
        ax.set_title(label)
        ax.set_xlabel("environment steps")
        kfmt(ax)
        if key == "explained_var":
            ax.set_ylim(-0.5, 1.05)
        if key == "approx_kl_k1":
            ax.axhline(0.0, color="#8a8985", lw=0.8)
    axes[0, 0].legend(loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
def parse_overrides(items: list[str]) -> dict:
    """Turn ["update_epochs=10", "clip_coef=inf", "target_kl=None"] into typed PPOConfig overrides."""
    fields = {f.name: f for f in dataclasses.fields(PPOConfig)}
    out = {}
    for item in items:
        key, val = item.split("=", 1)
        if key not in fields:
            raise SystemExit(f"unknown PPOConfig field: {key}")
        default = getattr(PPOConfig(), key)
        if val in ("None", "none"):
            out[key] = None
        elif isinstance(default, bool):
            out[key] = val.lower() in ("1", "true", "yes")
        elif isinstance(default, int) and not isinstance(default, bool):
            out[key] = int(float(val))
        elif isinstance(default, str):
            out[key] = val
        else:
            out[key] = float(val)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="smoke test (~20 s), writes no figures")
    p.add_argument("--env", default=None, help="CartPole-v1 or Pendulum-v1 (default: both)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override PPOConfig fields, e.g. --set update_epochs=10 clip_coef=inf "
                        "kl_penalty_target=0.01 bootstrap_truncation=False")
    p.add_argument("--no-figures", action="store_true", help="full-length runs without writing figures")
    a = p.parse_args()
    user_over = parse_overrides(a.set)

    envs = [a.env] if a.env else ["CartPole-v1", "Pendulum-v1"]
    seeds = list(range(1, (1 if a.quick else a.seeds) + 1))
    summary = []
    t_start = time.time()
    for env_id in envs:
        over = {}
        if a.quick:
            over["total_steps"] = 8_192
        if a.total_steps:
            over["total_steps"] = a.total_steps
        over.update(user_over)
        results = []
        for s in seeds:
            cfg = make_config(env_id, seed=s, **over)
            if s == seeds[0]:
                print(f"\n=== PPO on {env_id} ===")
                print("  " + ", ".join(f"{k}={v}" for k, v in dataclasses.asdict(cfg).items() if k != "seed"))
                print(f"  seeds={seeds}  batch={cfg.batch_size}  minibatch={cfg.minibatch_size}  "
                      f"updates={cfg.total_steps // cfg.batch_size}")
            res = train(cfg, log_every=5 if a.quick else 25)
            res["eval"] = evaluate_policy(res, n_episodes=3 if a.quick else 10)
            last = res["episodes"][-20:, 1].mean() if len(res["episodes"]) else float("nan")
            grid = np.linspace(0, cfg.total_steps, 160)        # same grid as ppo_clip_ablation.py
            curve = smoothed_curve(res["episodes"], grid)
            reach = first_reach(curve, grid, 500.0) if env_id == "CartPole-v1" else float("nan")
            summary.append((env_id, s, last, res["eval"], reach, largest_drop(curve), res["time"]))
            results.append(res)
        # diagnostics over all seeds: first quarter vs last quarter of the updates
        L = [r["logs"] for r in results]
        q = len(L[0]["step"]) // 4
        cat = lambda k, sl: np.concatenate([np.array(l[k])[sl] for l in L])  # noqa: E731
        first, lastq = slice(0, q), slice(-q, None)
        print(f"  diagnostics (median over seeds; first quarter -> last quarter of updates):")
        print(f"    realised KL(pi_old || pi_new) after each update (exact) "
              f"{np.median(cat('kl_post', first)):.2e} -> {np.median(cat('kl_post', lastq)):.2e}"
              f" (max {np.max(cat('kl_post', slice(None))):.2e});  pre-step k3 estimate "
              f"{np.median(cat('approx_kl_k3', first)):.2e} -> {np.median(cat('approx_kl_k3', lastq)):.2e}")
        print(f"    clip fraction {np.median(cat('clipfrac', first)):.3f} -> {np.median(cat('clipfrac', lastq)):.3f}"
              f" (max {np.max(cat('clipfrac', slice(None))):.3f});  "
              f"entropy {np.median(cat('entropy', slice(0, 1))):.3f} (first update) -> "
              f"{np.median(cat('entropy', slice(-1, None))):.3f} (last)")
        print(f"    explained variance {np.median(cat('explained_var', first)):.3f} -> "
              f"{np.median(cat('explained_var', lastq)):.3f} (max {np.nanmax(cat('explained_var', slice(None))):.3f}, "
              f"> 0.5 on {np.mean(cat('explained_var', slice(None)) > 0.5):.0%} of updates);  "
              f"std of value targets {np.median(cat('target_std', first)):.3f} -> {np.median(cat('target_std', lastq)):.3f}")
        print(f"    k1 estimate negative on {np.mean(cat('approx_kl_k1', slice(None)) < 0):.0%} of updates; "
              f"k3 negative on {np.mean(cat('approx_kl_k3', slice(None)) < 0):.0%}")
        if user_over.get("kl_penalty_target") is not None:
            print(f"    KL-penalty coefficient beta after each update: range "
                  f"{np.min(cat('kl_beta', slice(None))):g} .. {np.max(cat('kl_beta', slice(None))):g}, "
                  f"median over the last quarter {np.median(cat('kl_beta', lastq)):g}")
        if not a.quick and not a.no_figures:
            os.makedirs(FIG_DIR, exist_ok=True)
            name = "ppo_cartpole.png" if env_id == "CartPole-v1" else "ppo_pendulum.png"
            plot_diagnostics(results, f"PPO on {env_id}: learning curve and diagnostics", os.path.join(FIG_DIR, name))
            print(f"  wrote figures/{name}")

    print("\n=== Summary ===")
    print("(smoothed = mean of the last 20 episodes; 'steps to 500' only for CartPole)")
    print(f"{'env':14s} {'seed':>4s} {'train return (last 20 eps)':>28s} {'deterministic eval':>19s} "
          f"{'steps to 500':>13s} {'largest drop':>13s} {'time (s)':>9s}")
    for env_id, s, last, ev, reach, drop, tm in summary:
        print(f"{env_id:14s} {s:4d} {last:28.1f} {ev:19.1f} {reach:13.0f} {drop:13.1f} {tm:9.1f}")
    print(f"total wall time {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
