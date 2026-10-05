"""Distributional DQN from scratch: C51 (categorical) and QR-DQN (quantile regression).

Chapter 09, Section 9.

  C51 (Bellemare, Dabney & Munos, 2017): Z(s,a) is a categorical distribution on fixed atoms
      z_0 < ... < z_{N-1}; the target r + gamma z_j is projected back onto the atoms (Eq. 9.16)
      and the loss is the cross-entropy  -sum_i m_i log p_i(s,a)  (Eq. 9.17; Algorithm 9.4).
  QR-DQN (Dabney, Rowland, Bellemare & Munos, 2018): Z(s,a) is N equally-weighted Diracs at
      learned locations theta_i(s,a), trained with the quantile Huber loss at the quantile
      midpoints tau_i = (2i-1)/(2N)  (Eqs. 9.18-9.19; Algorithm 9.5).
Both reuse the interaction loop of dqn.py (replay, target network, epsilon-greedy on the mean).

Experiments
  0. The categorical projection worked example of Section 9.3, computed by the same function the
     agent uses.
  1. FrozenLake-v1 (4x4, slippery), 20k steps per agent: returns are genuinely random (falling into a hole gives 0,
     reaching the goal after T steps gives gamma^(T-1)), so the return distribution is bimodal.
     We compare the learned Z(s, greedy action) with the exact distribution of the learned greedy
     policy's return, estimated from 100,000 simulated episodes (model-based Monte Carlo), and
     report the Wasserstein-1 distance W1.
  2. CartPole-v1, 40k steps per agent: C51 and QR-DQN learn the task; the learned distributions at three hand-picked
     states are compared with the (deterministic) return of each agent's greedy policy.

Outputs (full mode): figures/distributional_frozenlake.png, figures/distributional_cartpole.png
Run:  python code/ch09_deep_q_learning/distributional.py [--quick]
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dqn import Config, FIG_DIR, make_adam, train

torch.set_num_threads(1)
SEED = 0


# --------------------------------------------------------------------------------------------
# Networks and agents
# --------------------------------------------------------------------------------------------
class DistNet(nn.Module):
    """MLP with |A| x N outputs: N logits (C51) or N quantile locations (QR) per action."""

    def __init__(self, obs_dim, n_actions, n_out, hidden):
        super().__init__()
        self.n_actions, self.n_out = n_actions, n_out
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, n_actions * n_out)

    def forward(self, x):
        return self.head(self.body(x)).view(*x.shape[:-1], self.n_actions, self.n_out)


def project_categorical(tz, p, z):
    """Cramer projection Pi_C of sum_j p_j delta_{tz_j} onto the atoms z (Eq. 9.16).

    m_i = sum_j [1 - |clip(tz_j) - z_i| / dz]_0^1  p_j : each target atom's mass is split
    between its two neighbouring support atoms in proportion to closeness (the triangular kernel
    sums to one over i, so total mass is preserved).  Shapes: tz, p (..., N_j); z (N_i,).
    """
    dz = z[1] - z[0]
    tz = tz.clamp(z[0], z[-1])
    w = (1.0 - (tz[..., :, None] - z).abs() / dz).clamp(min=0.0)       # (..., N_j, N_i)
    return (p[..., :, None] * w).sum(-2)


class C51Agent:
    def __init__(self, obs_dim, n_actions, cfg: Config, v_min, v_max, n_atoms=51):
        self.cfg, self.n_actions = cfg, n_actions
        self.z = torch.linspace(v_min, v_max, n_atoms)
        self.q = DistNet(obs_dim, n_actions, n_atoms, cfg.hidden)
        self.q_target = DistNet(obs_dim, n_actions, n_atoms, cfg.hidden)
        self.sync_target()
        self.opt = make_adam(self.q.parameters(), cfg.lr)

    def sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())

    @torch.no_grad()
    def dist(self, obs):
        """p_i(s, a) for all actions: shape (..., |A|, N)."""
        return F.softmax(self.q(torch.as_tensor(obs, dtype=torch.float32)), dim=-1)

    @torch.no_grad()
    def q_values(self, obs):                     # Q(s,a) = sum_i z_i p_i(s,a)
        return (self.dist(obs) * self.z).sum(-1)

    def act(self, obs, eps, rng):
        if rng.random() < eps:
            return int(rng.integers(self.n_actions))
        return int(self.q_values(obs).argmax().item())

    def update(self, b):
        s, a, g, s2, term, disc = b["obs"], b["act"], b["ret"], b["next_obs"], b["term"], b["disc"]
        B = len(a)
        log_p = F.log_softmax(self.q(s)[torch.arange(B), a], dim=-1)          # (B, N)
        with torch.no_grad():
            p_next = F.softmax(self.q_target(s2), dim=-1)                       # (B, |A|, N)
            a_star = (p_next * self.z).sum(-1).argmax(1)                         # greedy on the mean
            p_star = p_next[torch.arange(B), a_star]                            # (B, N)
            tz = g[:, None] + (disc * (1.0 - term))[:, None] * self.z            # r + gamma z_j
            m = project_categorical(tz, p_star, self.z)                         # (B, N)
        loss = -(m * log_p).sum(-1).mean()                                      # cross-entropy
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        q_sa = (log_p.detach().exp() * self.z).sum(-1)
        return np.zeros(B), float(loss.item()), float(q_sa.mean().item())


class QRAgent:
    def __init__(self, obs_dim, n_actions, cfg: Config, n_quantiles=51, kappa=1.0):
        self.cfg, self.n_actions, self.kappa = cfg, n_actions, kappa
        self.N = n_quantiles
        self.tau_hat = (2 * torch.arange(n_quantiles, dtype=torch.float32) + 1) / (2 * n_quantiles)
        self.q = DistNet(obs_dim, n_actions, n_quantiles, cfg.hidden)
        self.q_target = DistNet(obs_dim, n_actions, n_quantiles, cfg.hidden)
        self.sync_target()
        self.opt = make_adam(self.q.parameters(), cfg.lr)

    def sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())

    @torch.no_grad()
    def quantiles(self, obs):
        return self.q(torch.as_tensor(obs, dtype=torch.float32))               # (..., |A|, N)

    @torch.no_grad()
    def q_values(self, obs):                     # Q(s,a) = (1/N) sum_i theta_i(s,a)
        return self.quantiles(obs).mean(-1)

    def act(self, obs, eps, rng):
        if rng.random() < eps:
            return int(rng.integers(self.n_actions))
        return int(self.q_values(obs).argmax().item())

    def update(self, b):
        s, a, g, s2, term, disc = b["obs"], b["act"], b["ret"], b["next_obs"], b["term"], b["disc"]
        B = len(a)
        theta = self.q(s)[torch.arange(B), a]                                   # (B, N_i)
        with torch.no_grad():
            th_next = self.q_target(s2)                                         # (B, |A|, N)
            a_star = th_next.mean(-1).argmax(1)
            target = g[:, None] + (disc * (1.0 - term))[:, None] * th_next[torch.arange(B), a_star]
        u = target[:, None, :] - theta[:, :, None]           # u_ij = T theta_j - theta_i  (B, N_i, N_j)
        k = self.kappa
        # Huber part l_k(u) (Eq. 9.5); F.huber_loss is a fused kernel, ~4x faster than torch.where
        huber = F.huber_loss(theta[:, :, None].expand_as(u), target[:, None, :].expand_as(u),
                             reduction="none", delta=k)
        weight = (self.tau_hat[None, :, None] - (u.detach() < 0).float()).abs()   # |tau_i - 1[u<0]|
        loss = (weight * huber / k).mean(2).sum(1).mean()    # sum over i, mean over j, mean over batch
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return np.zeros(B), float(loss.item()), float(theta.detach().mean().item())


# --------------------------------------------------------------------------------------------
# Distances between 1-D distributions
# --------------------------------------------------------------------------------------------
def w1(x1, p1, x2, p2):
    """Exact Wasserstein-1 distance between two discrete distributions: integral of |F1 - F2|."""
    pts = np.unique(np.concatenate([x1, x2]))
    F1 = np.array([p1[x1 <= t].sum() for t in pts])
    F2 = np.array([p2[x2 <= t].sum() for t in pts])
    return float(np.sum(np.abs(F1 - F2)[:-1] * np.diff(pts)))


# --------------------------------------------------------------------------------------------
# FrozenLake: exact Monte Carlo ground truth from the known model
# --------------------------------------------------------------------------------------------
def frozenlake_returns(policy, start, gamma, n_episodes, rng, max_steps=3000):
    """Discounted returns of a deterministic tabular policy, simulated from env.unwrapped.P."""
    env = gym.make("FrozenLake-v1")
    P = env.unwrapped.P
    nS = env.observation_space.n
    nxt = np.zeros((nS, 4, 3), np.int64)
    prob = np.zeros((nS, 4, 3))
    rew = np.zeros((nS, 4, 3))
    done = np.zeros((nS, 4, 3), bool)
    for s in range(nS):
        for a in range(4):
            outs = P[s][a]
            outs = outs + [outs[-1]] * (3 - len(outs)) if len(outs) < 3 else outs
            for k, (pr, s2, r, d) in enumerate(outs[:3]):
                nxt[s, a, k], prob[s, a, k], rew[s, a, k], done[s, a, k] = s2, pr, r, d
            prob[s, a] /= prob[s, a].sum()
    s = np.full(n_episodes, start)
    G = np.zeros(n_episodes)
    disc = np.ones(n_episodes)
    alive = np.ones(n_episodes, bool)
    cum = prob.cumsum(-1)
    for _ in range(max_steps):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        a = policy[s[idx]]
        u = rng.random(idx.size)
        k = np.minimum((u[:, None] > cum[s[idx], a]).sum(1), 2)
        G[idx] += disc[idx] * rew[s[idx], a, k]
        disc[idx] *= gamma
        alive[idx] = ~done[s[idx], a, k]
        s[idx] = nxt[s[idx], a, k]
    return G


def frozenlake_experiment(quick, rng):
    # 20k steps: all three agents reach their final greedy success rate long before that.
    cfg = Config(env_id="FrozenLake-v1", total_steps=1_500 if quick else 20_000, lr=1e-3,
                 learning_starts=500 if quick else 1_000, eps_decay_steps=1_000 if quick else 10_000,
                 buffer_size=20_000, eval_every=1_500 if quick else 2_500, eval_episodes=5 if quick else 20,
                 loss="n/a (distributional loss)", seed=SEED)
    print(f"\n[1] FrozenLake-v1 4x4 slippery | config: {dataclasses.asdict(cfg)}")
    agents = {}
    # Returns lie in [0, 1].  With kappa = 1 every |u| < kappa, so the quantile Huber loss is the
    # asymmetric SQUARED loss |tau - 1[u<0]| u^2 / 2, whose minimisers are expectiles, not quantiles.
    for name, factory in (("C51", lambda o, n, c: C51Agent(o, n, c, 0.0, 1.0, 51)),
                          ("QR-DQN k=1", lambda o, n, c: QRAgent(o, n, c, 51, kappa=1.0)),
                          ("QR-DQN k=0.01", lambda o, n, c: QRAgent(o, n, c, 51, kappa=0.01))):
        h = train(cfg, make_agent=factory, verbose=False)
        agents[name] = h["agent"]
        print(f"  {name}: {h['runtime_s']:.0f}s; greedy success rate at the last 4 evaluations: "
              f"{np.round(h['eval_score'][-4:], 2)}", flush=True)
    eye = np.eye(16, dtype=np.float32)
    n_mc = 2_000 if quick else 100_000
    results = {}
    for name, ag in agents.items():
        policy = ag.q_values(eye).argmax(1).numpy()
        for state in (0, 14):
            G = frozenlake_returns(policy, state, cfg.gamma, n_mc, rng)
            a = policy[state]
            if name == "C51":
                x, p = ag.z.numpy(), ag.dist(eye[state])[a].numpy()
            else:
                x, p = np.sort(ag.quantiles(eye[state])[a].numpy()), np.full(ag.N, 1.0 / ag.N)
            xs, ps = np.unique(G, return_counts=True)
            d = w1(x, p, xs, ps / ps.sum())
            # Reference: the Cramer projection Pi_C of the Monte Carlo distribution onto the same 51
            # atoms (what C51 targets).  NOT the W1-best 51-atom categorical in general.
            best = project_categorical(torch.tensor(xs, dtype=torch.float32), torch.tensor(ps / ps.sum(),
                                       dtype=torch.float32), torch.linspace(0, 1, 51)).numpy()
            d_best = w1(np.linspace(0, 1, 51), best, xs, ps / ps.sum())
            results[(name, state)] = dict(x=x, p=p, G=G, w1=d, w1_best=d_best, policy=policy,
                                          mean_learned=float((x * p).sum()))
            print(f"  {name} state {state:2d}, greedy action {a}: learned mean {float((x * p).sum()):.3f}, "
                  f"MC mean {G.mean():.3f} (P(G=0) = {np.mean(G == 0):.3f}); W1(learned, MC) = {d:.4f}"
                  + (f"; W1(Pi_C MC, MC) = {d_best:.4f}" if name == "C51" else "")
                  + f"\n      learned P(G <= 0.01) = {p[x <= 0.01].sum():.3f}; learned support with p > 0.001: "
                    f"[{x[p > 1e-3].min():.3f}, {x[p > 1e-3].max():.3f}]")
        print(f"  {name} greedy policy (rows of the 4x4 map; 0=L 1=D 2=R 3=U): {policy.reshape(4, 4).tolist()}")
    return results, list(agents)


# --------------------------------------------------------------------------------------------
# CartPole: learned distributions at chosen states
# --------------------------------------------------------------------------------------------
PROBE_STATES = {
    "start state": np.array([0.02, 0.01, -0.03, 0.02], np.float32),
    "pole leaning, falling": np.array([0.0, 0.0, 0.12, 0.9], np.float32),
    "beyond recovery": np.array([0.0, 0.0, 0.19, 1.6], np.float32),
}


def cartpole_return_from(agent, state, gamma, max_steps=1000):
    """Discounted return of the agent's greedy policy started in `state` (deterministic dynamics)."""
    env = gym.make("CartPole-v1", max_episode_steps=max_steps)
    env.reset(seed=0)
    env.unwrapped.state = state.astype(np.float64).copy()
    obs, G, disc = state, 0.0, 1.0
    for _ in range(max_steps):
        a = int(agent.q_values(obs).argmax().item())
        obs, r, term, trunc, _ = env.step(a)
        G += disc * r
        disc *= gamma
        if term or trunc:
            break
    return G


def cartpole_experiment(quick):
    # 40k steps (not 50k as for DQN) keeps the full run of this script under ~7 minutes.
    cfg = Config(seed=SEED, total_steps=1_500 if quick else 40_000, eval_every=1_500 if quick else 2_500,
                 learning_starts=500 if quick else 1_000, eps_decay_steps=1_000 if quick else 10_000,
                 loss="n/a (distributional loss)")
    print(f"\n[2] CartPole-v1 | config: {dataclasses.asdict(cfg)}")
    out = {}
    for name, factory in (("C51", lambda o, n, c: C51Agent(o, n, c, 0.0, 100.0, 51)),
                          ("QR-DQN", lambda o, n, c: QRAgent(o, n, c, 51, kappa=1.0))):
        h = train(cfg, make_agent=factory, verbose=False)
        ag = h["agent"]
        print(f"  {name}: {h['runtime_s']:.0f}s; greedy scores every {cfg.eval_every} steps: "
              f"{np.round(h['eval_score']).astype(int).tolist()}", flush=True)
        probes = {}
        for label, st in PROBE_STATES.items():
            a = int(ag.q_values(st).argmax().item())
            if name == "C51":
                x, p = ag.z.numpy(), ag.dist(st)[a].numpy()
            else:
                x, p = np.sort(ag.quantiles(st)[a].numpy()), np.full(ag.N, 1.0 / ag.N)
            G = cartpole_return_from(ag, st, cfg.gamma)
            sd = float(np.sqrt((p * (x - (x * p).sum()) ** 2).sum()))
            probes[label] = dict(x=x, p=p, G=G)
            print(f"    {label:22s}: learned mean {float((x * p).sum()):6.2f}, std {sd:5.2f}; "
                  f"actual return of the greedy policy {G:6.2f}; W1 {w1(x, p, np.array([G]), np.array([1.0])):6.2f}; "
                  f"support with p > 0.001: [{x[p > 1e-3].min():.1f}, {x[p > 1e-3].max():.1f}]")
        out[name] = dict(hist=h, probes=probes)
    return out


def worked_example():
    z = torch.arange(5, dtype=torch.float32)                 # atoms 0,1,2,3,4 (dz = 1)
    p = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1])
    r, gamma = 0.5, 0.9
    tz = r + gamma * z
    m = project_categorical(tz, p, z)
    print("[0] Categorical projection worked example (Section 9.3): atoms 0..4, p =", p.tolist(),
          f"r = {r}, gamma = {gamma}")
    print("    r + gamma z_j =", [round(v, 2) for v in tz.tolist()], "-> projected m =",
          [round(v, 4) for v in m.tolist()], f"(sum {m.sum():.4f}, mean {float((m * z).sum()):.4f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    print(f"Seed {SEED}; quick={args.quick}")
    t0 = time.perf_counter()
    worked_example()
    fl, fl_names = frozenlake_experiment(args.quick, rng)
    cp = cartpole_experiment(args.quick)
    print(f"Total runtime {time.perf_counter() - t0:.0f}s")
    if args.quick:
        return
    plot(fl, fl_names, cp)


def _cdf_xy(x, p):
    order = np.argsort(x)
    return np.r_[x[order][0], x[order]], np.r_[0.0, np.cumsum(p[order])]


def plot(fl, fl_names, cp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import setup, C, GREY
    setup()
    os.makedirs(FIG_DIR, exist_ok=True)
    # ---- FrozenLake: rows = states, columns = agents ---------------------------------------
    fig, ax = plt.subplots(2, len(fl_names), figsize=(4.3 * len(fl_names), 7), sharex=True, sharey=True)
    for r, state in enumerate((0, 14)):
        for c, name in enumerate(fl_names):
            rr = fl[(name, state)]
            axc = ax[r, c]
            xs, cnt = np.unique(rr["G"], return_counts=True)
            xg, Fg = _cdf_xy(xs, cnt / cnt.sum())
            axc.step(xg, Fg, where="post", color="k", lw=1.2, label="Monte Carlo (100k episodes)")
            xg, Fg = _cdf_xy(rr["x"], rr["p"])
            axc.step(xg, Fg, where="post", color=C[c], lw=2.0, ls=["-", "--", "-."][c], label=f"{name}, learned")
            axc.set_title(f"{name}, state {state}: W1 = {rr['w1']:.3f}", fontsize=10)
            if r == 1:
                axc.set_xlabel("discounted return")
            if c == 0:
                axc.set_ylabel("cumulative probability")
            axc.set_xlim(-0.05, 1.05)
            axc.legend(loc="upper left", fontsize=8)
    fig.suptitle("FrozenLake 4x4 (slippery): learned return distribution of the greedy action vs the "
                 "Monte Carlo distribution of the same agent's greedy policy", fontsize=10.5)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "distributional_frozenlake.png")
    fig.savefig(path)
    print("saved", path)
    # ---- CartPole ---------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.6))
    for i, name in enumerate(("C51", "QR-DQN")):
        h = cp[name]["hist"]
        ax[0].plot(np.array(h["eval_step"]) / 1000, h["eval_score"], ["-o", "--s"][i], ms=3.5, color=C[i], label=name)
    ax[0].set(xlabel="environment steps (thousands)", ylabel="greedy score", title="CartPole-v1 (1 seed each)",
              ylim=(0, 520))
    ax[0].legend(loc="lower right")
    for j, label in enumerate(PROBE_STATES):
        axj = ax[j + 1]
        for i, name in enumerate(("C51", "QR-DQN")):
            pr = cp[name]["probes"][label]
            xg, Fg = _cdf_xy(pr["x"], pr["p"])
            axj.step(xg, Fg, where="post", color=C[i], ls=["-", "--"][i], label=f"{name} learned")
            axj.axvline(pr["G"], color=C[i], ls=":", lw=1.3, label=f"{name}: actual return {pr['G']:.1f}")
        lo = min(-2.0, min(float(cp[n]["probes"][label]["x"].min()) for n in ("C51", "QR-DQN")) - 2.0)
        axj.set(xlabel="discounted return", ylabel="cumulative probability", title=f"{label}", xlim=(lo, 102))
        axj.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "distributional_cartpole.png")
    fig.savefig(path)
    print("saved", path)


if __name__ == "__main__":
    main()
