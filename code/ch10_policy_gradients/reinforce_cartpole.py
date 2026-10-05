"""REINFORCE on CartPole-v1: total return vs reward-to-go vs reward-to-go with a learned baseline.

Chapter 10, Sections 5-6 (Algorithms 10.1 and 10.2), PyTorch, one CPU thread.

Three single-episode gradient estimators (Eqs. 10.11, 10.13 and 10.16), identical otherwise:
    total     : sum_t G_0              * grad log pi(A_t|S_t)   ("vanilla" trajectory form)
    rtg       : sum_t G_t              * grad log pi(A_t|S_t)   (causality / reward-to-go)
    baseline  : sum_t (G_t - v_hat(S_t)) * grad log pi(A_t|S_t) (learned state-value baseline)
As in nearly all implementations, the gamma^t factor is dropped (Section 12 explains what that
means).  Every update uses ONE episode.  The per-step terms are SUMMED and divided by the
CONSTANT T_MAX = 500 (the time limit), which only rescales the step size.  Dividing by the
episode's own length T (i.e. .mean()) would NOT be a rescaling: T depends on the actions, so
for the total-return variant E[(G_0/T) sum_t psi_t] = grad E[G_0/T], and with gamma = 0.99
G_0/T decreases with T -- the agent would learn to fail fast (Section 5.4, pitfall 7).

Two experiments
  1. Learning curves: each variant x n_seeds seeds x n_episodes episodes.
  2. Gradient variance: at checkpoints of one 'baseline' run, freeze the policy (and its value
     net), sample K episodes, and compute each estimator's per-episode gradient vector g_i.
     We report the total variance tr Cov(g) and the signal-to-noise ratio ||mean g||^2 / tr Cov(g).

Truncation (500 steps) vs termination: Gymnasium's CartPole-v1 truncates at 500 steps.  A Monte
Carlo return cannot bootstrap, so a truncated episode's G_t simply stops at the cut-off: we
optimize the time-limited return that CartPole scores, and the time-unaware baseline fits it
less well near the limit (it cannot bias the gradient).  A2C (a2c_gae_cartpole.py) bootstraps
through truncation.

--mean-loss reproduces the per-episode-mean surrogate (.mean() over the episode's T steps)
for comparison; it prints the learning results only and writes no figure.

Outputs (full mode): figures/reinforce_cartpole.png
Run:  python code/ch10_policy_gradients/reinforce_cartpole.py [--quick] [--mean-loss]
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
VARIANTS = ["total", "rtg", "baseline"]
T_MAX = 500                                        # CartPole-v1 time limit: a CONSTANT normalizer


def mlp(n_in, n_out, hidden=64):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, n_out))


class NumpyPolicy:
    """A frozen numpy copy of the policy MLP, used only to ACT during rollouts.

    A per-step PyTorch forward pass on a single observation costs ~50 us; this numpy version
    costs a few us, which makes the full experiment several times faster.  Gradients are always
    computed by PyTorch on the whole episode afterwards, with the same weights.
    """

    def __init__(self, policy):
        lin = [m for m in policy if isinstance(m, nn.Linear)]
        self.W = [(m.weight.detach().double().numpy().copy(), m.bias.detach().double().numpy().copy())
                  for m in lin]

    def probs(self, obs):
        h = obs
        for i, (W, b) in enumerate(self.W):
            h = W @ h + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
        e = np.exp(h - h.max())
        return e / e.sum()


def run_episode(env, policy, rng, seed=None):
    """Roll out one episode with the current (frozen) policy. Returns obs (T,4), actions (T,), rewards (T,)."""
    actor = NumpyPolicy(policy)
    obs, _ = env.reset(seed=seed)
    O, A, R = [], [], []
    while True:
        p = actor.probs(obs.astype(np.float64))
        a = int(rng.random() < p[1])                  # sample A_t ~ pi(.|S_t) (two actions)
        O.append(obs)
        A.append(a)
        obs, r, terminated, truncated, _ = env.step(a)
        R.append(r)
        if terminated or truncated:
            break
    return np.array(O, dtype=np.float32), np.array(A), np.array(R, dtype=np.float32)


def rewards_to_go(R, gamma):
    G = np.zeros_like(R)
    run = 0.0
    for t in range(len(R) - 1, -1, -1):
        run = R[t] + gamma * run
        G[t] = run
    return G


def weights_for(variant, R, obs_t, value, gamma):
    """The scalar multiplying grad log pi(A_t|S_t) at each step, for each estimator."""
    G = torch.as_tensor(rewards_to_go(R, gamma))
    if variant == "total":
        return G[0].expand_as(G)                   # every action credited with the whole return
    if variant == "rtg":
        return G
    with torch.no_grad():
        return G - value(obs_t).squeeze(-1)        # advantage estimate G_t - v_hat(S_t)


def train(variant, seed, n_episodes, gamma, lr_pi, lr_v, checkpoints=(), mean_loss=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")
    policy, value = mlp(4, 2), mlp(4, 1)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=lr_pi)
    opt_v = torch.optim.Adam(value.parameters(), lr=lr_v)
    returns, steps, saved = np.zeros(n_episodes), 0, {}
    for ep in range(n_episodes):
        if ep in checkpoints:
            saved[ep] = ({k: v.clone() for k, v in policy.state_dict().items()},
                         {k: v.clone() for k, v in value.state_dict().items()})
        O, A, R = run_episode(env, policy, rng, seed=seed * 100_000 + ep)
        returns[ep], steps = R.sum(), steps + len(R)
        obs_t = torch.as_tensor(O)
        w = weights_for(variant, R, obs_t, value, gamma)
        logp = torch.distributions.Categorical(logits=policy(obs_t)).log_prob(torch.as_tensor(A))
        # Surrogate: its gradient is the PG estimate sum_t w_t psi_t, divided by a CONSTANT.
        # NOT .mean(): dividing by this episode's own T reweights episodes by 1/T and changes
        # the expected direction (Section 5.4).
        loss_pi = -(w * logp).mean() if mean_loss else -(w * logp).sum() / T_MAX
        opt_pi.zero_grad()
        loss_pi.backward()
        opt_pi.step()
        if variant == "baseline":                  # regress v_hat(S_t) on the MC return G_t
            G = torch.as_tensor(rewards_to_go(R, gamma))
            loss_v = 0.5 * ((value(obs_t).squeeze(-1) - G) ** 2).mean()
            opt_v.zero_grad()
            loss_v.backward()
            opt_v.step()
    env.close()
    with torch.no_grad():                          # how deterministic is the final policy?
        maxprob = torch.softmax(policy(obs_t), -1).max(-1).values.mean().item()
    return returns, steps, saved, maxprob


def gradient_variance(saved, K, gamma, seed):
    """Per-episode gradient estimates of all three estimators at frozen checkpoints."""
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(10_000 + seed)
    policy, value = mlp(4, 2), mlp(4, 1)
    out = {}
    for ep, (pi_sd, v_sd) in sorted(saved.items()):
        policy.load_state_dict(pi_sd)
        value.load_state_dict(v_sd)
        grads = {v: [] for v in VARIANTS}
        lens = []
        for k in range(K):
            O, A, R = run_episode(env, policy, rng, seed=10_000_000 + seed * 1000 + k)
            lens.append(len(R))
            obs_t = torch.as_tensor(O)
            logp = torch.distributions.Categorical(logits=policy(obs_t)).log_prob(torch.as_tensor(A))
            for v in VARIANTS:
                w = weights_for(v, R, obs_t, value, gamma)
                g = torch.autograd.grad((w * logp).sum(), list(policy.parameters()), retain_graph=True)
                grads[v].append(torch.cat([x.flatten() for x in g]).numpy())
        res = {}
        for v in VARIANTS:
            Gm = np.array(grads[v])
            tr = Gm.var(0, ddof=1).sum()
            m2 = (Gm.mean(0) ** 2).sum() - tr / K      # unbiased estimate of ||E g||^2
            res[v] = (tr, m2 / tr)
        out[ep] = (res, float(np.mean(lens)))
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mean-loss", action="store_true",
                    help="divide each episode's surrogate by its own length T (changes the objective)")
    args = ap.parse_args()
    if args.quick:
        n_seeds, n_eps, K, ckpts = 1, 60, 10, (0, 40)
    else:
        n_seeds, n_eps, K, ckpts = 5, 1000, 100, (0, 100, 200, 400)
    gamma, lr_pi, lr_v = 0.99, 1e-3, 5e-3
    print(f"seed={args.seed}  seeds={n_seeds}  episodes/run={n_eps}  gamma={gamma}  "
          f"lr_pi={lr_pi}  lr_v={lr_v}  net=2x64 tanh  variance episodes K={K} at {ckpts}  "
          f"policy loss = {'per-episode MEAN (1/T)' if args.mean_loss else 'sum / T_MAX (constant)'}")
    t0 = time.time()
    curves, steps, maxp, saved0 = {}, {}, {}, None
    for v in VARIANTS:
        rs, ss, mp = [], [], []
        for i in range(n_seeds):
            r, s, saved, m = train(v, args.seed + i, n_eps, gamma, lr_pi, lr_v,
                                   checkpoints=ckpts if (v == "baseline" and i == 0) else (),
                                   mean_loss=args.mean_loss)
            rs.append(r)
            ss.append(s)
            mp.append(m)
            if saved:
                saved0 = saved
        curves[v], steps[v], maxp[v] = np.array(rs), np.array(ss), np.array(mp)
        print(f"  {v:9s} done ({time.time() - t0:.0f}s)")

    last = max(1, n_eps // 10)
    print(f"\nLearning (mean over {n_seeds} seeds; return of the last {last} episodes; "
          "first episode whose 20-episode moving average reaches 475):")
    for v in VARIANTS:
        R = curves[v]
        ma = np.array([np.convolve(r, np.ones(20) / 20, mode="valid") for r in R])
        first = [int(np.argmax(m >= 475)) + 20 if (m >= 475).any() else None for m in ma]
        print(f"  {v:9s} last-{last} mean {R[:, -last:].mean():6.1f} (per seed "
              f"{np.round(R[:, -last:].mean(1)).astype(int).tolist()}),  mean over all episodes "
              f"{R.mean():6.1f},  solved at {first},  env steps/run {steps[v].mean():.0f},  "
              f"final mean max_a pi(a|s) {maxp[v].mean():.3f}")
    if args.mean_loss:
        print(f"\n(--mean-loss: gradient-variance study and figure skipped)  Total time {time.time() - t0:.0f}s")
        return

    print(f"\nGradient variance at frozen checkpoints of the baseline run (K = {K} episodes each):")
    gv = gradient_variance(saved0, K, gamma, args.seed)
    print("  episode | mean len |  tr Cov: total     rtg  baseline | SNR: total    rtg  baseline")
    for ep, (res, mlen) in gv.items():
        print(f"  {ep:7d} | {mlen:8.1f} | {res['total'][0]:12.3g} {res['rtg'][0]:9.3g} "
              f"{res['baseline'][0]:9.3g} | {res['total'][1]:9.3f} {res['rtg'][1]:6.3f} "
              f"{res['baseline'][1]:9.3f}")
    print(f"\nTotal time {time.time() - t0:.0f}s")

    if not args.quick:
        from plot_style import setup, C
        plt = setup()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw=dict(width_ratios=[1.4, 1]))
        ax = axes[0]
        labels = {"total": "total return $G_0$", "rtg": "reward-to-go $G_t$",
                  "baseline": r"$G_t-\hat v(S_t,\mathbf{w})$ (learned baseline)"}
        styles = {"total": ":", "rtg": "--", "baseline": "-"}
        for j, v in enumerate(VARIANTS):
            R = curves[v]
            ma = np.array([np.convolve(r, np.ones(20) / 20, mode="valid") for r in R])
            x = np.arange(20, n_eps + 1)
            ax.plot(x, ma.mean(0), color=C[j], ls=styles[v], label=labels[v])
            ax.fill_between(x, ma.min(0), ma.max(0), color=C[j], alpha=0.12)
        ax.set_xlabel("episode")
        ax.set_ylabel("return (20-episode moving average)")
        ax.set_title(f"REINFORCE on CartPole-v1 ({n_seeds} seeds; band = min/max)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8, frameon=False)
        ax = axes[1]
        eps = list(gv.keys())
        xs = np.arange(len(eps))
        for j, v in enumerate(VARIANTS):
            ax.bar(xs + (j - 1) * 0.27, [gv[e][0][v][0] for e in eps], width=0.27, color=C[j],
                   label=labels[v])
        ax.set_yscale("log")
        top = max(gv[e][0][v][0] for e in eps for v in VARIANTS)
        ax.set_ylim(top=top * 60)                  # headroom so the legend does not cover any bar
        ax.set_xticks(xs, [f"ep {e}\n(len {gv[e][1]:.0f})" for e in eps])
        ax.set_ylabel("tr Cov of one-episode gradient")
        ax.set_title("Gradient variance at frozen policies")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "reinforce_cartpole.png"))
        plt.close(fig)
        print(f"figure written to {FIG_DIR}/reinforce_cartpole.png")


if __name__ == "__main__":
    main()
