"""A2C with Generalized Advantage Estimation on CartPole-v1, with vectorized environments.

Chapter 10, Sections 8-10 (Algorithms 10.5 and 10.7), PyTorch, one CPU thread.

* N_ENV copies of CartPole-v1 run in lock-step in a gymnasium.vector.SyncVectorEnv; every
  update uses a rollout of n steps from each copy (a batch of N_ENV * n transitions).
* Advantages: GAE(gamma, lambda), Eq. (10.21), computed by the backward recursion
      A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}.
* Critic target: the lambda-return  A_t + V(S_t)  (Eq. 10.24).
* Loss: -mean(A_t log pi(A_t|S_t)) + c_v * mean((V(S_t) - target_t)^2) - c_H * mean(entropy).
* Truncation vs termination (Section 8.4): the vector env runs in SAME_STEP autoreset mode, so
  when an episode ends the returned observation is already the first one of the next episode
  and the real last observation is in info["final_obs"].  The TD error bootstraps from
  V(final_obs) when the episode was TRUNCATED (500-step limit) and uses 0 when it TERMINATED.

Experiments (full mode)
  1. lambda ablation: lambda in {0, 0.5, 0.9, 0.95, 0.99, 1} x n_seeds seeds.
  2. entropy-bonus ablation at lambda = 0.95: c_H in {0, 0.01, 0.05}.
  3. clipping the gradient norm of actor and critic separately vs jointly.
The role of parallel actors (how a batch is collected) is studied in a2c_parallel_actors.py,
which imports train() from this file.

Outputs (full mode): figures/a2c_gae_lambda.png, figures/a2c_ablations.png
Run:  python code/ch10_policy_gradients/a2c_gae_cartpole.py [--quick]
"""
from __future__ import annotations

import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def mlp(n_in, n_out, hidden=64, out_gain=1.0):
    net = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                        nn.Linear(hidden, n_out))
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
    nn.init.orthogonal_(net[-1].weight, gain=out_gain)   # small last layer => near-uniform pi_0
    return net


class NumpyActor:
    """Frozen numpy copy of the actor, used only to choose actions during the rollout.

    For a batch of 8 observations a numpy forward pass is several times cheaper than a PyTorch
    call; log-probabilities for the gradient are recomputed by PyTorch on the whole batch.
    """

    def __init__(self, actor):
        self.W = [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())
                  for m in actor if isinstance(m, nn.Linear)]

    def sample(self, obs, rng):
        h = obs
        for i, (W, b) in enumerate(self.W):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
        p = np.exp(h - h.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        u = rng.random(len(p))[:, None]
        return np.minimum((u > np.cumsum(p, 1)).sum(1), p.shape[1] - 1)   # inverse-CDF sampling


def compute_gae(rewards, values, next_values, terminated, done, gamma, lam):
    """GAE(gamma, lambda) by the backward recursion (10.23) of Algorithm 10.5 (arrays of shape (n, N_ENV)).

    next_values[t] = V(S_{t+1}) of the SAME episode (V(final_obs) if the episode ended at t).
    delta_t = R_{t+1} + gamma * (1 - terminated_t) * V(S_{t+1}) - V(S_t)   -- no bootstrap past a
                                                                              true termination
    A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}            -- the sum stops at ANY
                                                                              episode boundary
    """
    n = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1])
    for t in range(n - 1, -1, -1):
        delta = rewards[t] + gamma * (1.0 - terminated[t]) * next_values[t] - values[t]
        last = delta + gamma * lam * (1.0 - done[t]) * last
        adv[t] = last
    return adv


def train(lam, seed, total_steps, n_env=8, n_steps=16, gamma=0.99, lr=1e-3, c_v=0.5, c_ent=0.01,
          max_grad_norm=0.5, clip_separately=True, segments=1):
    """A2C with GAE (Algorithm 10.7).  Returns an array of (env step, return) of finished episodes.

    segments > 1 collects that many CONSECUTIVE rollouts of n_steps per environment (each with
    its own GAE, bootstrapped at its end) before one update.  With n_env = 1 and segments = 8
    this produces the same estimator and batch size as n_env = 8, segments = 1, except that the
    8 segments are consecutive pieces of one environment's trajectory instead of 8
    independent environments: it isolates the decorrelation effect of parallel actors.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(n_env)],
                                    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    obs, _ = envs.reset(seed=seed)
    actor, critic = mlp(4, 2, out_gain=0.01), mlp(4, 1, out_gain=1.0)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    ep_ret = np.zeros(n_env)
    finished = []                                   # (env step at which it ended, episode return)
    norms = []                                      # (||grad actor||, ||grad critic||) per update
    step = 0
    while step < total_steps:
        acting = NumpyActor(actor)                  # the policy is fixed during the rollout
        batch_O, batch_A, batch_adv, batch_target = [], [], [], []
        for _ in range(segments):
            O = np.zeros((n_steps, n_env, 4), np.float32)
            Act = np.zeros((n_steps, n_env), np.int64)
            Rw = np.zeros((n_steps, n_env), np.float32)
            Term = np.zeros((n_steps, n_env), np.float32)
            Done = np.zeros((n_steps, n_env), np.float32)
            NextO = np.zeros((n_steps, n_env, 4), np.float32)   # S_{t+1} of the same episode
            for t in range(n_steps):
                a = acting.sample(obs, rng)             # A_t ~ pi(.|S_t) for every env
                nobs, r, term, trunc, info = envs.step(a)
                O[t], Act[t], Rw[t] = obs, a, r
                Term[t], Done[t] = term, term | trunc
                NextO[t] = nobs
                if "final_obs" in info:             # SAME_STEP autoreset: nobs is a reset obs
                    for i in np.flatnonzero(info["_final_obs"]):
                        NextO[t, i] = info["final_obs"][i]
                ep_ret += r
                step += n_env
                for i in np.flatnonzero(term | trunc):
                    finished.append((step, ep_ret[i]))
                    ep_ret[i] = 0.0
                obs = nobs
            # advantages and lambda-return targets for this segment (critic fixed)
            with torch.no_grad():
                V = critic(torch.as_tensor(O.reshape(-1, 4))).squeeze(-1).numpy().reshape(n_steps, n_env)
                V_next = critic(torch.as_tensor(NextO.reshape(-1, 4))).squeeze(-1).numpy().reshape(n_steps, n_env)
            adv = compute_gae(Rw, V, V_next, Term, Done, gamma, lam)
            batch_O.append(O.reshape(-1, 4))
            batch_A.append(Act.reshape(-1))
            batch_adv.append(adv.reshape(-1))
            batch_target.append((adv + V).reshape(-1))            # lambda-return, Eq. (10.24)
        # ----- learning step -----
        O_t = torch.as_tensor(np.concatenate(batch_O))
        adv_t = torch.as_tensor(np.concatenate(batch_adv), dtype=torch.float32)
        target = torch.as_tensor(np.concatenate(batch_target), dtype=torch.float32)
        dist = torch.distributions.Categorical(logits=actor(O_t))
        logp = dist.log_prob(torch.as_tensor(np.concatenate(batch_A)))
        loss_pi = -(adv_t * logp).mean()            # advantages are constants (no gradient)
        loss_v = ((critic(O_t).squeeze(-1) - target) ** 2).mean()
        entropy = dist.entropy().mean()
        loss = loss_pi + c_v * loss_v - c_ent * entropy
        opt.zero_grad()
        loss.backward()
        norms.append((grad_norm(actor), grad_norm(critic)))   # before clipping (diagnostic)
        if clip_separately:
            # Clip actor and critic gradients SEPARATELY.  The critic's gradient is typically
            # much larger (value errors scale with returns ~ 100).  Clipping the JOINT norm
            # multiplies the actor's gradient by a factor set by the critic's error, which varies
            # from batch to batch (Adam undoes a constant rescaling, not a varying one) --
            # ablation [3] measures the norms and the effect.
            nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
        else:
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), max_grad_norm)
        opt.step()
    envs.close()
    train.last_grad_norms = np.array(norms)         # read by ablation [3]
    return np.array(finished)


def grad_norm(net):
    return float(torch.sqrt(sum((p.grad ** 2).sum() for p in net.parameters() if p.grad is not None)))


def curve(finished, total_steps, window=10_000, grid=None):
    """Mean return of the episodes that ended within the trailing `window` env steps."""
    grid = np.arange(window, total_steps + 1, window // 2) if grid is None else grid
    out = []
    for g in grid:
        m = (finished[:, 0] > g - window) & (finished[:, 0] <= g)
        out.append(finished[m, 1].mean() if m.any() else np.nan)
    return grid, np.array(out)


def plot_mean_sd(ax, results, total_steps, color, ls, label, band_alpha=0.12):
    """Mean learning curve over seeds, with a +-1 s.d. band (seed-to-seed spread)."""
    cs = np.array([curve(f, total_steps)[1] for f in results])
    g = curve(results[0], total_steps)[0]
    m, sd = np.nanmean(cs, 0), np.nanstd(cs, 0, ddof=1) if len(cs) > 1 else np.zeros(cs.shape[1])
    ax.plot(g, m, color=color, ls=ls, label=label)
    if band_alpha > 0:
        ax.fill_between(g, m - sd, m + sd, color=color, alpha=band_alpha, lw=0)


def summarize(results, total_steps):
    """Final performance (episodes ending in the last 20% of training) and mean over training."""
    finals, aucs = [], []
    for f in results:
        finals.append(f[f[:, 0] > 0.8 * total_steps, 1].mean())
        _, c = curve(f, total_steps, window=min(10_000, total_steps // 4))
        aucs.append(np.nanmean(c))
    return np.array(finals), np.array(aucs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.quick:
        lams, n_seeds, total = [0.0, 0.95], 1, 8_000
    else:
        lams, n_seeds, total = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0], 5, 150_000
    base = dict(n_env=8, n_steps=16, gamma=0.99, lr=1e-3, c_v=0.5, c_ent=0.01, max_grad_norm=0.5)
    print(f"seed={args.seed}  seeds={n_seeds}  env steps/run={total}  lambdas={lams}  {base}")
    t0 = time.time()

    print("\n[1] lambda ablation")
    res_lam = {}
    for lam in lams:
        res_lam[lam] = [train(lam, args.seed + i, total, **base) for i in range(n_seeds)]
        fin, auc = summarize(res_lam[lam], total)
        print(f"  lambda={lam:4.2f}: final return {fin.mean():6.1f} +- {fin.std(ddof=1) if n_seeds > 1 else 0:5.1f} "
              f"(per seed {np.round(fin).astype(int).tolist()}),  mean over training {auc.mean():6.1f}  "
              f"[{time.time() - t0:.0f}s]")

    res_ent, res_clip = {}, {}
    if not args.quick:
        def ablation(title, results, key, **kw):
            results[key] = res_lam[0.95] if not kw else \
                [train(0.95, args.seed + i, total, **{**base, **kw}) for i in range(n_seeds)]
            fin, auc = summarize(results[key], total)
            print(f"  {title:24s}: final return {fin.mean():6.1f} +- {fin.std(ddof=1):5.1f} "
                  f"(per seed {np.round(fin).astype(int).tolist()}),  mean over training {auc.mean():6.1f}"
                  f"  [{time.time() - t0:.0f}s]")

        print("\n[2] entropy coefficient (lambda = 0.95)")
        for c in [0.0, 0.01, 0.05]:
            ablation(f"c_H = {c}", res_ent, c, **({} if c == 0.01 else {"c_ent": c}))
        print("\n[3] gradient clipping (lambda = 0.95, max norm 0.5)")
        ablation("actor/critic separately", res_clip, "separate")
        ablation("joint norm", res_clip, "joint", clip_separately=False)
        gn = train.last_grad_norms                  # from the last joint-clipping run
        ratio = gn[:, 1] / gn[:, 0]
        scale = np.minimum(1.0, 0.5 / np.sqrt((gn ** 2).sum(1)))
        print(f"  joint-clipping run (seed {args.seed + n_seeds - 1}): ||grad critic|| / ||grad actor|| before "
              f"clipping: median {np.median(ratio):.1f}, 90th percentile {np.quantile(ratio, 0.9):.1f}; "
              f"the actor's gradient is multiplied by {np.median(scale):.3f} (median) by the joint clip, "
              f"vs {np.median(np.minimum(1.0, 0.5 / gn[:, 0])):.3f} if clipped alone")
    print(f"\nTotal time {time.time() - t0:.0f}s")

    if not args.quick:
        from plot_style import setup, C
        from matplotlib.ticker import FuncFormatter
        plt = setup()
        kfmt = FuncFormatter(lambda x, _: f"{x / 1000:.0f}k")
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw=dict(width_ratios=[1.5, 1]))
        ax = axes[0]
        styles = ["-", "--", "-.", "-", ":", "--"]
        for j, lam in enumerate(lams):
            # bands only for lambda = 0 and 0.95; six overlapping bands would hide the means
            plot_mean_sd(ax, res_lam[lam], total, C[j], styles[j], f"$\\lambda$ = {lam}",
                         band_alpha=0.13 if lam in (0.0, 0.95) else 0.0)
        ax.set_xlabel("environment steps")
        ax.xaxis.set_major_formatter(kfmt)
        ax.set_ylabel("episode return (10k-step window)")
        ax.set_title(f"A2C + GAE on CartPole-v1, mean of {n_seeds} seeds\n"
                     "(bands: $\\pm$1 s.d. over seeds for $\\lambda$ = 0 and 0.95)", fontsize=10)
        ax.set_ylim(0, 520)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax = axes[1]
        fins = [summarize(res_lam[l], total)[0] for l in lams]
        aucs = [summarize(res_lam[l], total)[1] for l in lams]
        x = np.arange(len(lams))
        # lambda values are categorical here (unevenly spaced): markers only, no connecting lines
        ax.errorbar(x - 0.1, [f.mean() for f in fins], yerr=[f.std(ddof=1) for f in fins], fmt="o",
                    color=C[0], capsize=3, label="final (last 20% of steps)")
        ax.errorbar(x + 0.1, [a.mean() for a in aucs], yerr=[a.std(ddof=1) for a in aucs], fmt="s",
                    color=C[1], capsize=3, label="mean over training")
        ax.set_xticks(x, [str(l) for l in lams])
        ax.set_xlabel("GAE $\\lambda$ (categorical axis)")
        ax.set_ylabel("episode return (mean $\\pm$ sd over seeds)")
        ax.set_title("Bias-variance trade-off in $\\lambda$")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "a2c_gae_lambda.png"))
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        panels = ((axes[0], res_ent, "entropy bonus ($\\lambda$=0.95)", lambda k: f"$c_{{\\mathcal{{H}}}}$ = {k}"),
                  (axes[1], res_clip, "gradient clipping ($\\lambda$=0.95)",
                   lambda k: {"separate": "actor, critic clipped separately",
                              "joint": "joint norm of actor + critic"}[k]))
        for ax, res, title, fmt in panels:
            for j, (k, rs) in enumerate(res.items()):
                plot_mean_sd(ax, rs, total, C[j], styles[j], fmt(k))
            ax.set_xlabel("environment steps")
            ax.xaxis.set_major_formatter(kfmt)
            ax.set_ylabel("episode return (10k-step window)")
            ax.set_title(title + " (band: $\\pm$1 s.d. over seeds)", fontsize=10)
            ax.set_ylim(0, 520)
            ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "a2c_ablations.png"))
        plt.close(fig)
        print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
