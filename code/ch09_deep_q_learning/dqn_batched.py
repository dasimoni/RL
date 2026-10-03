"""K independent DQN agents trained in lock-step: cheap multi-seed experiments on one CPU thread.

Chapter 09, "In code".  Small networks on a CPU are dominated by Python/PyTorch overhead, not
arithmetic: one gradient step on a 64 x 4 minibatch costs about the same as one on a
5 x 64 x 4 batch.  So instead of running five seeds one after the other we stack the weights of
K networks into tensors of shape (K, n_in, n_out) and use batched matrix products.

Why these are K independent copies of Algorithm 9.1 (not an ensemble that shares anything):
  * member k only ever sees its own environment, its own replay buffer and its own minibatch;
  * the total loss is  sum_k  mean_b  loss_{k,b},  so the gradient w.r.t. member k's weights is
    the gradient of member k's own loss;
  * Adam is an elementwise optimiser, so member k's update uses only member k's gradients.
`--quick` checks this numerically: a member of the batched network, initialised with the
weights of a dqn.py network and fed the same minibatches, follows dqn.py's update to ~1e-6.
So a batched run is equivalent IN DISTRIBUTION to K separate dqn.py runs (same algorithm and
hyperparameters, independent weights and data).  It does not reproduce `dqn.py --seed k`
exactly: the members share one numpy generator, use different environment seeds and draw their
initial weights in a different order.

Diagnostic for Exercise 9.15: at every evaluation we also record the mean predicted max_a Q at
the most recent "fallen" states, i.e. next states S' of transitions the ENVIRONMENT marked as
terminated (whatever flag the agent stored).  No transition ever starts in such a state, so the
network is never trained there; the measurement shows what the bootstrap would see.

Supported flags: double, dueling, n_step, use_target, polyak, loss, reward_noise, action_copies,
terminal_bug.
(PER and NoisyNets are only in the single-agent dqn.py.)

Run:  python code/ch09_deep_q_learning/dqn_batched.py [--quick]
      quick: equivalence check + a 2,000-step smoke run of K=2 agents;
      full:  equivalence check with more updates + cost per gradient step, single vs batched.
      No figures.  The experiments that use this module are compare_variants.py,
      overestimation.py and ablate_stabilizers.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dqn import (Config, DQNAgent, NStepAccumulator, epsilon_at, make_adam, make_env)

torch.set_num_threads(1)


class BatchedLinear(nn.Module):
    """K independent linear layers: (K, N, n_in) -> (K, N, n_out) with one baddbmm."""

    def __init__(self, K, n_in, n_out):
        super().__init__()
        bound = 1.0 / math.sqrt(n_in)            # same distribution as nn.Linear's default init
        self.weight = nn.Parameter(torch.empty(K, n_in, n_out).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(K, 1, n_out).uniform_(-bound, bound))

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)


class BatchedQNet(nn.Module):
    """K copies of dqn.QNetwork (or dqn.DuelingQNetwork, Eq. 9.9) with stacked weights."""

    def __init__(self, K, obs_dim, n_actions, hidden=128, dueling=False):
        super().__init__()
        self.dueling = dueling
        if dueling:
            self.l1 = BatchedLinear(K, obs_dim, hidden)
            self.v1, self.v2 = BatchedLinear(K, hidden, hidden), BatchedLinear(K, hidden, 1)
            self.a1, self.a2 = BatchedLinear(K, hidden, hidden), BatchedLinear(K, hidden, n_actions)
        else:
            self.l1 = BatchedLinear(K, obs_dim, hidden)
            self.l2 = BatchedLinear(K, hidden, hidden)
            self.l3 = BatchedLinear(K, hidden, n_actions)

    def forward(self, x):                        # x: (K, N, obs_dim) -> (K, N, n_actions)
        h = F.relu(self.l1(x))
        if self.dueling:
            v = self.v2(F.relu(self.v1(h)))
            a = self.a2(F.relu(self.a1(h)))
            return v + a - a.mean(dim=-1, keepdim=True)
        return self.l3(F.relu(self.l2(h)))

    @torch.no_grad()
    def load_member(self, k, single):
        """Copy the weights of a dqn.py network into member k (nn.Linear stores W as (out, in))."""
        if self.dueling:
            pairs = [(self.l1, single.torso[0]), (self.v1, single.value[0]), (self.v2, single.value[2]),
                     (self.a1, single.adv[0]), (self.a2, single.adv[2])]
        else:
            pairs = [(self.l1, single.body[0]), (self.l2, single.body[2]), (self.l3, single.head)]
        for mine, theirs in pairs:
            mine.weight[k].copy_(theirs.weight.T)
            mine.bias[k, 0].copy_(theirs.bias)


class BatchedDQN:
    """K agents; every tensor carries a leading member dimension K."""

    def __init__(self, K, obs_dim, n_actions, cfg: Config):
        self.K, self.cfg, self.n_actions = K, cfg, n_actions
        self.q = BatchedQNet(K, obs_dim, n_actions, cfg.hidden, cfg.dueling)
        self.q_target = BatchedQNet(K, obs_dim, n_actions, cfg.hidden, cfg.dueling)
        self.sync_target()
        self.opt = make_adam(self.q.parameters(), cfg.lr)

    def sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())

    @torch.no_grad()
    def soft_update(self, tau):
        for p, p_targ in zip(self.q.parameters(), self.q_target.parameters()):
            p_targ.lerp_(p, tau)

    @torch.no_grad()
    def greedy(self, obs):                       # obs: (K, obs_dim) -> (K,) greedy actions
        return self.q(torch.as_tensor(obs, dtype=torch.float32)[:, None, :])[:, 0].argmax(-1).numpy()

    def update(self, b):
        cfg = self.cfg
        q_sa = self.q(b["obs"]).gather(2, b["act"][..., None]).squeeze(2)          # (K, B)
        with torch.no_grad():
            q_next = (self.q_target if cfg.use_target else self.q)(b["next_obs"])  # (K, B, A)
            if cfg.double:                                                          # Eq. (9.8)
                a_star = self.q(b["next_obs"]).argmax(2, keepdim=True)
                v_next = q_next.gather(2, a_star).squeeze(2)
            else:                                                                   # Eq. (9.2)
                v_next = q_next.max(2).values
            y = b["ret"] + b["disc"] * (1.0 - b["term"]) * v_next
        if cfg.loss == "huber":
            per = F.smooth_l1_loss(q_sa, y, reduction="none")
        else:
            per = 0.5 * (y - q_sa).pow(2)
        loss = per.mean(1).sum()               # sum over members of each member's mean loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return q_sa.detach().mean(1).numpy()


class BatchedReplay:
    """K circular buffers with their own write positions (n-step flushes desynchronise them)."""

    def __init__(self, K, capacity, obs_dim, rng):
        self.K, self.cap, self.rng = K, capacity, rng
        self.obs = np.zeros((K, capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((K, capacity, obs_dim), np.float32)
        self.act = np.zeros((K, capacity), np.int64)
        self.ret = np.zeros((K, capacity), np.float32)
        self.term = np.zeros((K, capacity), np.float32)
        self.disc = np.zeros((K, capacity), np.float32)
        self.pos = np.zeros(K, np.int64)
        self.size = np.zeros(K, np.int64)
        self.last = np.zeros(K, np.int64)       # index of the newest transition of each member

    def add(self, k, s, a, g, s2, term, disc):
        i = self.pos[k]
        self.obs[k, i], self.act[k, i], self.ret[k, i] = s, a, g
        self.next_obs[k, i], self.term[k, i], self.disc[k, i] = s2, term, disc
        self.last[k] = i
        self.pos[k] = (i + 1) % self.cap
        self.size[k] = min(self.size[k] + 1, self.cap)

    def sample(self, B):
        if self.cap == 1:                        # "no replay": train on the newest transition only
            idx = np.zeros((self.K, B), np.int64)
        else:
            idx = (self.rng.random((self.K, B)) * self.size[:, None]).astype(np.int64)
        k = np.arange(self.K)[:, None]
        t = torch.from_numpy
        return dict(obs=t(self.obs[k, idx]), act=t(self.act[k, idx]), ret=t(self.ret[k, idx]),
                    next_obs=t(self.next_obs[k, idx]), term=t(self.term[k, idx]), disc=t(self.disc[k, idx]))


@torch.no_grad()
def evaluate_batched(agent: BatchedDQN, cfg: Config, envs, seed):
    """Greedy evaluation of all K members at once; same measurements as dqn.evaluate, plus the
    fraction of visited states whose predicted max_a Q exceeds 1/(1-gamma), a value no policy
    can achieve in CartPole."""
    K = agent.K
    q_bound = 1.0 / (1.0 - cfg.gamma)       # CartPole: every reward is +1, so no return exceeds this
    scores = np.zeros((K, cfg.eval_episodes))
    q_sum, mc_sum, n, n_over = np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K)
    for ep in range(cfg.eval_episodes):
        obs = np.stack([env.reset(seed=seed + ep)[0] for env in envs])
        active = np.ones(K, bool)
        states = [[] for _ in range(K)]
        rewards = [[] for _ in range(K)]
        while active.any():
            a = agent.greedy(obs)
            for k in np.flatnonzero(active):
                states[k].append(obs[k].copy())
                o, r, term, trunc, _ = envs[k].step(int(a[k]))
                rewards[k].append(r)
                obs[k] = o
                if term or trunc:
                    active[k] = False
        lengths = [min(len(r), cfg.score_cap) for r in rewards]
        L = max(lengths)
        pad = np.zeros((K, L, obs.shape[1]), np.float32)
        for k in range(K):
            pad[k, :lengths[k]] = np.asarray(states[k][:lengths[k]])
        qmax = agent.q(torch.from_numpy(pad)).max(2).values.numpy()
        for k in range(K):
            G, run = np.zeros(len(rewards[k])), 0.0
            for t in range(len(rewards[k]) - 1, -1, -1):
                run = rewards[k][t] + cfg.gamma * run
                G[t] = run
            m = lengths[k]
            scores[k, ep] = np.sum(rewards[k][:cfg.score_cap])     # CartPole: = m
            q_sum[k] += qmax[k, :m].sum()
            mc_sum[k] += G[:m].sum()
            n_over[k] += (qmax[k, :m] > q_bound).sum()
            n[k] += m
    return scores.mean(1), q_sum / n, mc_sum / n, n_over / n


def train_batched(cfg: Config, K: int, verbose=True, label=""):
    """Train K seeds (cfg.seed, ..., cfg.seed + K - 1) of the configuration `cfg` in lock-step."""
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    envs = [make_env(cfg) for _ in range(K)]
    eval_envs = [make_env(cfg, cfg.eval_max_steps) for _ in range(K)]
    obs = np.stack([env.reset(seed=cfg.seed + k)[0] for k, env in enumerate(envs)])
    obs_dim, n_actions = obs.shape[1], envs[0].action_space.n
    agent = BatchedDQN(K, obs_dim, n_actions, cfg)
    buf = BatchedReplay(K, cfg.buffer_size, obs_dim, rng)
    nsteps = [NStepAccumulator(cfg.n_step, cfg.gamma) for _ in range(K)]
    n_eval = cfg.total_steps // cfg.eval_every
    hist = dict(eval_step=np.zeros(n_eval, int), score=np.zeros((K, n_eval)), q=np.zeros((K, n_eval)),
                mc=np.zeros((K, n_eval)), frac_over=np.zeros((K, n_eval)), train_q=np.zeros((K, n_eval)),
                ep_returns=[[] for _ in range(K)], ep_steps=[[] for _ in range(K)])
    hist["q_fallen"] = np.full((K, n_eval), np.nan)
    fallen = [[] for _ in range(K)]               # recent next states with terminated=True (env flag)
    ep_ret = np.zeros(K)
    q_acc, n_acc, e = np.zeros(K), 0, 0
    t0 = time.perf_counter()
    for t in range(1, cfg.total_steps + 1):
        eps = epsilon_at(cfg, t)
        actions = agent.greedy(obs)
        explore = rng.random(K) < eps
        actions[explore] = rng.integers(n_actions, size=explore.sum())
        for k in range(K):
            o2, r, terminated, truncated, _ = envs[k].step(int(actions[k]))
            ep_ret[k] += r
            r_tr = r + cfg.reward_noise * rng.standard_normal() if cfg.reward_noise else r
            if cfg.terminal_bug == "ignore_terminal":
                stored_term = False
            elif cfg.terminal_bug == "truncation_is_terminal":
                stored_term = terminated or truncated
            else:
                stored_term = terminated
            for tr in nsteps[k].push(obs[k].copy(), int(actions[k]), r_tr, o2, stored_term,
                                     terminated or truncated):
                buf.add(k, *tr)
            if terminated:
                fallen[k].append(o2.copy())
                if len(fallen[k]) > 500:
                    fallen[k].pop(0)
            if terminated or truncated:
                hist["ep_returns"][k].append(ep_ret[k])
                hist["ep_steps"][k].append(t)
                ep_ret[k] = 0.0
                o2, _ = envs[k].reset()
            obs[k] = o2
        if t >= cfg.learning_starts and t % cfg.train_freq == 0:
            q_acc += agent.update(buf.sample(cfg.batch_size))
            n_acc += 1
        if cfg.use_target:
            if cfg.polyak > 0:
                agent.soft_update(cfg.polyak)
            elif t % cfg.target_update == 0:
                agent.sync_target()
        if t % cfg.eval_every == 0:
            sc, qm, mc, fo = evaluate_batched(agent, cfg, eval_envs, seed=10_000 + 100 * cfg.seed)
            hist["eval_step"][e] = t
            hist["score"][:, e], hist["q"][:, e], hist["mc"][:, e], hist["frac_over"][:, e] = sc, qm, mc, fo
            hist["train_q"][:, e] = q_acc / max(n_acc, 1)
            if all(fallen):
                L = min(len(f) for f in fallen)
                xf = torch.as_tensor(np.stack([np.asarray(f[-L:]) for f in fallen]), dtype=torch.float32)
                with torch.no_grad():
                    hist["q_fallen"][:, e] = agent.q(xf).max(2).values.mean(1).numpy()
            q_acc, n_acc = np.zeros(K), 0
            if verbose:
                print(f"  {label}step {t:6d}  score {np.round(sc).astype(int)}  "
                      f"Q {np.round(qm, 1)}  MC {np.round(mc, 1)}  ({time.perf_counter() - t0:5.1f}s)",
                      flush=True)
            e += 1
    hist["runtime_s"] = time.perf_counter() - t0
    return hist


def check_equivalence(n_updates=50, dueling=False, double=False):
    """Max |difference| between dqn.py's update and member 1 of a K=3 batched agent."""
    cfg = Config(dueling=dueling, double=double)
    torch.manual_seed(123)
    single = DQNAgent(4, 2, cfg)
    batched = BatchedDQN(3, 4, 2, cfg)
    batched.q.load_member(1, single.q)
    batched.q_target.load_member(1, single.q_target)
    g = torch.Generator().manual_seed(0)
    for _ in range(n_updates):
        B = 32
        s, s2 = torch.randn(3, B, 4, generator=g), torch.randn(3, B, 4, generator=g)
        a = torch.randint(0, 2, (3, B), generator=g)
        r = torch.ones(3, B)
        term = (torch.rand(3, B, generator=g) < 0.1).float()
        disc = torch.full((3, B), cfg.gamma)
        batched.update(dict(obs=s, act=a, ret=r, next_obs=s2, term=term, disc=disc))
        single.update(dict(obs=s[1], act=a[1], ret=r[1], next_obs=s2[1], term=term[1], disc=disc[1],
                           weights=torch.ones(B)))
    probe = torch.randn(1, 64, 4, generator=g)
    q_b = batched.q(probe.expand(3, 64, 4))[1]
    q_s = single.q(probe[0])
    return float((q_b - q_s).abs().max().detach())


def time_updates(n=300):
    """Seconds per gradient step: one dqn.py agent vs K stacked agents (batch 64 each)."""
    cfg = Config()
    out = {}
    single = DQNAgent(4, 2, cfg)
    b = dict(obs=torch.randn(64, 4), act=torch.randint(0, 2, (64,)), ret=torch.ones(64),
             next_obs=torch.randn(64, 4), term=torch.zeros(64), disc=torch.full((64,), 0.99),
             weights=torch.ones(64))
    for _ in range(20):
        single.update(b)
    t0 = time.perf_counter()
    for _ in range(n):
        single.update(b)
    out["dqn.py (1 agent)"] = (time.perf_counter() - t0) / n
    for K in (1, 5, 10):
        ag = BatchedDQN(K, 4, 2, cfg)
        bb = {k: v.expand(K, *v.shape).contiguous() for k, v in b.items() if k != "weights"}
        for _ in range(20):
            ag.update(bb)
        t0 = time.perf_counter()
        for _ in range(n):
            ag.update(bb)
        out[f"batched K={K}"] = (time.perf_counter() - t0) / n
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    n_upd = 50 if args.quick else 200
    print(f"Seed 123 (networks), 0 (minibatches).  Equivalence check: member 1 of a K=3 batched agent "
          f"vs dqn.py after {n_upd} identical updates")
    for dueling in (False, True):
        for double in (False, True):
            err = check_equivalence(n_updates=n_upd, dueling=dueling, double=double)
            print(f"  dueling={dueling!s:5} double={double!s:5}  max |Q_batched - Q_single| = {err:.2e}")
            assert err < 1e-4, "batched implementation diverged from dqn.py"
    if args.quick:
        cfg = Config(seed=0, total_steps=2_000, eval_every=1_000, learning_starts=500, eps_decay_steps=1_000)
        print("Smoke test: K=2 seeds, 2,000 steps")
        h = train_batched(cfg, 2)
        print(f"Done in {h['runtime_s']:.1f}s.")
        return
    print("Cost of one gradient step (minibatch 64 per agent, hidden 64, fused Adam):")
    for name, sec in time_updates().items():
        print(f"  {name:18s} {1e3 * sec:6.2f} ms")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------------------------
# Helpers shared by ablate_stabilizers.py and pitfalls.py
# --------------------------------------------------------------------------------------------
def run_suite(configs: dict, base: Config, K: int):
    """Train K seeds of every configuration in `configs` ({name: Config overrides})."""
    results = {}
    for name, kw in configs.items():
        h = train_batched(dataclasses.replace(base, **kw), K, verbose=False)
        results[name] = h
        last = slice(-2, None) if h["score"].shape[1] >= 2 else slice(None)
        print(f"  {name:38s} {h['runtime_s']:4.0f}s | final score per seed "
              f"{np.round(h['score'][:, last].mean(1)).astype(int)} | max over run of mean Q "
              f"{h['q'].mean(0).max():8.1f} | final mean Q {h['q'][:, last].mean():8.1f}", flush=True)
    return results


def print_suite_summary(results, last_n=2):
    print("\nMean (over seeds) of predicted max_a Q at each evaluation:")
    for name, h in results.items():
        print(f"  {name:38s} steps {h['eval_step'].tolist()}\n  {'':38s} Q     {np.round(h['q'].mean(0), 1).tolist()}")
    print("\nMean (over seeds) of predicted max_a Q at recent 'fallen' next states (env terminated=True):")
    for name, h in results.items():
        print(f"  {name:38s} Q_fallen {np.round(h['q_fallen'].mean(0), 1).tolist()}")
    print(f"\n{'configuration':38s} {'final score':>16s} {'mean score over run':>20s} "
          f"{'final mean Q':>13s} {'final MC':>9s} {'Q>100':>7s}   (+- = sample std over seeds, ddof=1)")
    for name, h in results.items():
        last = slice(-last_n, None)
        fin, auc = h["score"][:, last].mean(1), h["score"].mean(1)
        print(f"{name:38s} {fin.mean():7.1f} +- {fin.std(ddof=1):5.1f} {auc.mean():11.1f} +- {auc.std(ddof=1):5.1f} "
              f"{h['q'][:, last].mean():13.1f} {h['mc'][:, last].mean():9.1f} "
              f"{100 * h['frac_over'][:, last].mean():6.1f}%")


def plot_suite(results, path, K, suptitle, q_log=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import setup, C, GREY
    setup()
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
    styles, markers = ["-", "--", "-.", ":", "-"], ["o", "s", "^", "D", "v"]
    for i, (name, h) in enumerate(results.items()):
        x = h["eval_step"] / 1000
        m, se = h["score"].mean(0), h["score"].std(0, ddof=1) / np.sqrt(K)
        ax[0].plot(x, m, styles[i], marker=markers[i], ms=3.5, color=C[i], label=name)
        ax[0].fill_between(x, m - se, m + se, color=C[i], alpha=0.12, lw=0)
        ax[1].plot(x, h["q"].mean(0), styles[i], marker=markers[i], ms=3.5, color=C[i], label=name)
    ax[0].set(xlabel="environment steps (thousands)", ylabel="greedy score (max 500)", ylim=(0, 520),
              title=f"Performance: mean $\\pm$ s.e. over {K} seeds")
    ax[0].legend(loc="upper left", fontsize=8.5)
    ax[1].axhline(100, color=GREY, ls=":", lw=1.2)
    if q_log:
        ax[1].set_yscale("log")
    ax[1].set(xlabel="environment steps (thousands)", ylabel=r"mean predicted $\max_a Q$ (greedy states)",
              title=r"Value estimates (dotted: $1/(1-\gamma)=100$)")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    print("saved", path)
