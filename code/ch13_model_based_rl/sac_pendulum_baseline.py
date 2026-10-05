"""A compact Soft Actor-Critic baseline on Pendulum-v1, for the sample-efficiency comparison of
Chapter 13, Section 5.2.  SAC itself is taught in Chapter 12; this file is deliberately minimal
(standard hyperparameters, no tricks) and exists only so that the model-based vs model-free
comparison uses numbers produced in this repository.

  * tanh-squashed Gaussian actor, twin Q critics with Polyak-averaged targets (tau = 0.005 here is the
    Polyak coefficient, NOTATION.md), automatic entropy-temperature tuning (target entropy = -1).
  * 1,000 uniformly random steps first, then one gradient update per environment step.
  * Pendulum never terminates, only truncates (200 steps), so the TD target always bootstraps:
    y = r + gamma * (min Q_target(s', a') - alpha log pi(a'|s')),  done mask = `terminated` = 0.
  * Every 500 steps the deterministic policy tanh(mean) is evaluated on 5 fresh episodes.

Writes code/ch13_model_based_rl/results/sac_pendulum.json, which pets_pendulum.py overlays on
its learning-curve figure.  No figure of its own.

Run (from the repository root):
  python code/ch13_model_based_rl/sac_pendulum_baseline.py           # full: 3 seeds x 10,000 steps
  python code/ch13_model_based_rl/sac_pendulum_baseline.py --quick   # smoke test, writes nothing
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True          # importing the sibling modules must not leave __pycache__/
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, "results")


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_scale):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim)
        self.act_scale = act_scale

    def forward(self, s):
        mu, log_std = self.net(s).chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 2)
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)                     # reparameterised sample
        a = torch.tanh(u)
        # log pi(a|s) with the change of variables a = scale * tanh(u) (Chapter 12):
        # log|da/du| = log scale + log(1 - tanh(u)^2),  log(1 - tanh^2 u) = 2(log 2 - u - softplus(-2u))
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - (2 * (np.log(2) - u - F.softplus(-2 * u)) + np.log(self.act_scale)).sum(-1)
        return a * self.act_scale, logp, torch.tanh(mu) * self.act_scale


def run_sac(seed, total_steps, eval_every, start_steps=1000, batch=256, gamma=0.99, tau=0.005, lr=3e-4,
            eval_episodes=5):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")
    od, ad, scale = 3, 1, 2.0
    actor = Actor(od, ad, scale)
    q1, q2 = mlp(od + ad, 1), mlp(od + ad, 1)
    q1_t, q2_t = mlp(od + ad, 1), mlp(od + ad, 1)
    q1_t.load_state_dict(q1.state_dict()); q2_t.load_state_dict(q2.state_dict())
    log_alpha = torch.zeros(1, requires_grad=True)
    opt_pi = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    opt_a = torch.optim.Adam([log_alpha], lr=lr)
    target_entropy = -float(ad)
    buf_s = np.zeros((total_steps, od), np.float32); buf_a = np.zeros((total_steps, ad), np.float32)
    buf_r = np.zeros(total_steps, np.float32); buf_s2 = np.zeros((total_steps, od), np.float32)
    buf_d = np.zeros(total_steps, np.float32)
    eval_steps, eval_returns = [], []
    s, _ = env.reset(seed=seed)
    eval_env.reset(seed=10_000 + seed)
    t0 = time.time()
    for t in range(total_steps):
        if t < start_steps:
            a = rng.uniform(-scale, scale, size=ad).astype(np.float32)
        else:
            with torch.no_grad():
                a = actor(torch.as_tensor(s).unsqueeze(0))[0][0].numpy()
        s2, r, terminated, truncated, _ = env.step(a)
        buf_s[t], buf_a[t], buf_r[t], buf_s2[t] = s, a, r, s2
        buf_d[t] = float(terminated)             # bootstrap through truncation, not termination
        s = s2
        if terminated or truncated:
            s, _ = env.reset()
        if t + 1 >= start_steps:
            idx = rng.integers(0, t + 1, size=batch)
            bs, ba = torch.as_tensor(buf_s[idx]), torch.as_tensor(buf_a[idx])
            br, bs2, bd = torch.as_tensor(buf_r[idx]), torch.as_tensor(buf_s2[idx]), torch.as_tensor(buf_d[idx])
            alpha = log_alpha.exp().detach()
            with torch.no_grad():
                a2, logp2, _ = actor(bs2)
                sa2 = torch.cat([bs2, a2], -1)
                qt = torch.min(q1_t(sa2), q2_t(sa2)).squeeze(-1) - alpha * logp2
                y = br + gamma * (1 - bd) * qt
            sa = torch.cat([bs, ba], -1)
            loss_q = F.mse_loss(q1(sa).squeeze(-1), y) + F.mse_loss(q2(sa).squeeze(-1), y)
            opt_q.zero_grad(); loss_q.backward(); opt_q.step()
            an, logp, _ = actor(bs)
            san = torch.cat([bs, an], -1)
            loss_pi = (alpha * logp - torch.min(q1(san), q2(san)).squeeze(-1)).mean()
            opt_pi.zero_grad(); loss_pi.backward(); opt_pi.step()
            loss_a = -(log_alpha * (logp.detach() + target_entropy)).mean()
            opt_a.zero_grad(); loss_a.backward(); opt_a.step()
            with torch.no_grad():
                for net, tgt in ((q1, q1_t), (q2, q2_t)):
                    for p, pt in zip(net.parameters(), tgt.parameters()):
                        pt.mul_(1 - tau).add_(tau * p)
        if (t + 1) % eval_every == 0:
            rets = []
            for _ in range(eval_episodes):
                es, _ = eval_env.reset()
                done, R = False, 0.0
                while not done:
                    with torch.no_grad():
                        ea = actor(torch.as_tensor(es).unsqueeze(0))[2][0].numpy()
                    es, er, te, tr, _ = eval_env.step(ea)
                    R += er
                    done = te or tr
                rets.append(R)
            eval_steps.append(t + 1); eval_returns.append(float(np.mean(rets)))
            print(f"  seed {seed} step {t + 1:6d}  eval return {np.mean(rets):8.1f}  "
                  f"alpha {log_alpha.exp().item():.3f}  [{time.time() - t0:.0f}s]", flush=True)
    return {"seed": seed, "eval_steps": eval_steps, "eval_returns": eval_returns, "time": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=10_000)
    args = ap.parse_args()
    seeds, steps, every, start = args.seeds, args.steps, 500, 1000
    if args.quick:
        seeds, steps, every, start = [0], 1500, 500, 1000
    print(f"SAC baseline on Pendulum-v1 | seeds {seeds} | steps {steps} | random steps {start} | "
          "batch 256 | lr 3e-4 | gamma 0.99 | Polyak tau 0.005 | 256x256 MLPs", flush=True)
    runs = [run_sac(s, steps, every, start_steps=start) for s in seeds]
    E = np.array([r["eval_returns"] for r in runs])
    print("\nEval return (deterministic policy, mean of 5 episodes) at steps", runs[0]["eval_steps"])
    for r in runs:
        print(f"  seed {r['seed']}: " + " ".join(f"{x:7.0f}" for x in r["eval_returns"]))
    print("  mean   : " + " ".join(f"{x:7.0f}" for x in E.mean(0)))
    print(f"Wall-clock per seed: {np.mean([r['time'] for r in runs]):.0f} s")
    if not args.quick:
        os.makedirs(RES_DIR, exist_ok=True)
        with open(os.path.join(RES_DIR, "sac_pendulum.json"), "w") as f:
            json.dump({"runs": runs}, f)
        print("saved", os.path.join(RES_DIR, "sac_pendulum.json"))


if __name__ == "__main__":
    main()
