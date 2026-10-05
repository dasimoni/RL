"""Offline RL on CartPole-v1: behaviour cloning vs naive DQN vs discrete BCQ, CQL and IQL.

Chapter 16, Sections 6-9.  We log three fixed datasets of equal size with three behaviour
policies (Section 6.4), train five learners on each WITHOUT touching the environment again,
and only then evaluate their greedy policies online.

  BC          Algorithm 16.1   cross-entropy on (s, a) pairs
  naive DQN   Algorithm 16.7   DQN's TD update run on the fixed dataset (Section 6.5): the
                               max over next actions queries actions the data never contains
  BCQ         Algorithm 16.8   discrete BCQ: the max only ranges over actions a behaviour-
                               cloned model deems likely, pi_hat(a|s) / max_a' pi_hat(a'|s) > threshold
  CQL         Algorithm 16.10  CQL(H): TD loss + alpha * E_s[logsumexp_a Q(s,a) - Q(s,a_data)]
  IQL         Algorithm 16.12  expectile regression for V, TD to V(s') for Q, AWR for the policy

Seeds as an ensemble.  Every network is an EnsembleMLP with one member per random seed; each
member draws its own minibatches and is initialised independently, so this is exactly
`n_seeds` independent runs, computed with batched matrix products (see offline_lib.py).
Tensors therefore carry a leading "member" dimension E: states are [E, B, 4], Q-values [E, B, 2].

Diagnostics.  Every `eval_every` gradient steps we record each member's greedy return
(10 online episodes) and the average of max_a Q(s, a) over 2,000 dataset states.  With reward
+1 per step and gamma = 0.99, no true action value can exceed 1 / (1 - gamma) = 100, so any
estimate above 100 is an overestimate (extrapolation error, Section 6.2).

Run:  python code/ch16_offline_rl_and_imitation/offline_cartpole.py [--quick]
Output (full mode): figures/offline_cartpole_returns.png, figures/offline_cartpole_qvalues.png
"""
from __future__ import annotations

import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from offline_lib import (EnsembleMLP, Normalizer, collect_dataset, discounted_mc_returns,
                         evaluate, to_tensors)

torch.set_num_threads(1)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

GAMMA = 0.99
LR = 3e-4
BATCH = 128
POLYAK = 0.02           # target-network averaging coefficient (tau in Chapter 12's notation)
CQL_ALPHA = 1.0         # weight of the conservative penalty (Eq. 16.31)
IQL_EXPECTILE = 0.7     # expectile tau of Eq. (16.33)  (IQL paper's locomotion default)
IQL_BETA = 3.0          # inverse temperature of the advantage weights (Eq. 16.36)  (ditto)
BCQ_THRESHOLD = 0.3     # discrete BCQ action filter (Eq. 16.23)


def pick(x, a):
    """x[e, b, a[e, b]]: the value of the logged action."""
    return x.gather(2, a.unsqueeze(2)).squeeze(2)


def polyak_update(target, online):
    with torch.no_grad():
        torch._foreach_lerp_(list(target.parameters()), list(online.parameters()), POLYAK)


# --------------------------------------------------------------------------------------------
# Learners.  Each has update(batch) -> dict of scalars, act(obs [E, n, 4]) -> actions [E, n],
# and max_q(states [E, N, 4]) -> per-member mean of max_a Q(s, a) (NaN for BC, which has no Q).
# (Figure 2 plots the absolute value of that signed mean, |mean_s max_a Q(s, a)|, on a log scale.)
# --------------------------------------------------------------------------------------------
class BC:
    """Algorithm 16.1: maximum likelihood on the logged actions."""

    def __init__(self, norm, E):
        self.norm = norm
        self.pi = EnsembleMLP(E, 4, 2)
        self.opt = torch.optim.Adam(self.pi.parameters(), lr=LR, foreach=True)
        self.E = E

    def update(self, b):
        logits = self.pi(self.norm(b["s"]))                          # [E, B, 2]
        loss = F.cross_entropy(logits.reshape(-1, 2), b["a"].reshape(-1)) * self.E
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return {"loss": loss.item() / self.E}

    @torch.no_grad()
    def act(self, obs):
        return self.pi(self.norm(torch.as_tensor(obs, dtype=torch.float32))).argmax(2).numpy()

    def max_q(self, s):
        return np.full(self.E, np.nan)


class DQN:
    """Algorithm 16.7: off-policy Q-learning on a fixed dataset.  Subclasses change only the
    bootstrap action set (BCQ) or add a regulariser (CQL)."""

    def __init__(self, norm, E):
        self.norm = norm
        self.E = E
        self.q = EnsembleMLP(E, 4, 2)
        self.q_targ = copy.deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=LR, foreach=True)

    def next_value(self, s2):
        # y = r + gamma (1 - term) max_a' Q_targ(s', a')   -- the max may pick an action that
        # never appears in the data in (or near) s', whose value is pure extrapolation.
        return self.q_targ(self.norm(s2)).max(2).values

    def extra_loss(self, q_all, b):
        return torch.zeros(())

    def update(self, b):
        with torch.no_grad():
            y = b["r"] + GAMMA * (1.0 - b["term"]) * self.next_value(b["s2"])
        q_all = self.q(self.norm(b["s"]))                             # [E, B, 2]
        td = 0.5 * (pick(q_all, b["a"]) - y).pow(2).mean(1).sum()     # summed over members
        reg = self.extra_loss(q_all, b)
        loss = td + reg + self.aux_loss(b)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        polyak_update(self.q_targ, self.q)
        return {"td": td.item() / self.E, "reg": reg.item() / self.E}

    def aux_loss(self, b):
        return torch.zeros(())

    @torch.no_grad()
    def act(self, obs):
        return self.q(self.norm(torch.as_tensor(obs, dtype=torch.float32))).argmax(2).numpy()

    @torch.no_grad()
    def max_q(self, s):
        return self.q(self.norm(s)).max(2).values.mean(1).numpy()


class BCQ(DQN):
    """Algorithm 16.8 (discrete BCQ, Fujimoto, Conti, Ghavamzadeh & Pineau 2019): a behaviour
    model pi_hat is trained by BC; both the bootstrap max and the final policy only consider
    actions with pi_hat(a|s) / max_a' pi_hat(a'|s) > threshold."""

    def __init__(self, norm, E):
        super().__init__(norm, E)
        self.bc = EnsembleMLP(E, 4, 2)
        params = [*self.q.parameters(), *self.bc.parameters()]
        self.opt = torch.optim.Adam(params, lr=LR, foreach=True)

    def aux_loss(self, b):   # the behaviour model is trained alongside Q (plain BC)
        logits = self.bc(self.norm(b["s"]))
        return F.cross_entropy(logits.reshape(-1, 2), b["a"].reshape(-1)) * self.E

    def allowed(self, s_normed):
        probs = F.softmax(self.bc(s_normed), dim=2)
        return probs / probs.max(2, keepdim=True).values > BCQ_THRESHOLD

    def next_value(self, s2):
        s2n = self.norm(s2)
        mask = self.allowed(s2n)
        a2 = self.q(s2n).masked_fill(~mask, -1e8).argmax(2, keepdim=True)  # best *allowed* a'
        return self.q_targ(s2n).gather(2, a2).squeeze(2)                   # evaluated by target

    @torch.no_grad()
    def act(self, obs):
        sn = self.norm(torch.as_tensor(obs, dtype=torch.float32))
        return self.q(sn).masked_fill(~self.allowed(sn), -1e8).argmax(2).numpy()

    @torch.no_grad()
    def max_q(self, s):
        sn = self.norm(s)
        return self.q(sn).masked_fill(~self.allowed(sn), -1e8).max(2).values.mean(1).numpy()


class CQL(DQN):
    """Algorithm 16.10, CQL(H) for discrete actions (Kumar, Zhou, Tucker & Levine 2020):
    push down a soft maximum of Q over all actions, push up Q on the logged action."""

    def extra_loss(self, q_all, b):
        gap = torch.logsumexp(q_all, dim=2) - pick(q_all, b["a"])    # >= 0, Eq. (16.31)
        return CQL_ALPHA * gap.mean(1).sum()


class IQL:
    """Algorithm 16.12 (Kostrikov, Nair & Levine 2022), discrete-action version.

      V:  expectile regression of Q_targ(s, a_data) onto V(s)            (Eq. 16.33)
      Q:  regression onto r + gamma (1 - term) V(s')                      (Eq. 16.34)
      pi: advantage-weighted BC, weights exp(beta (Q_targ - V)) <= 100    (Eq. 16.36)
    No value is ever queried at an action outside the dataset.  The three losses touch disjoint
    parameters (cross terms are computed under no_grad), so one optimiser step on their sum is
    the same as three separate steps taken simultaneously."""

    def __init__(self, norm, E):
        self.norm = norm
        self.E = E
        self.q = EnsembleMLP(E, 4, 2)
        self.q_targ = copy.deepcopy(self.q)
        self.v = EnsembleMLP(E, 4, 1)
        self.pi = EnsembleMLP(E, 4, 2)
        params = [*self.q.parameters(), *self.v.parameters(), *self.pi.parameters()]
        self.opt = torch.optim.Adam(params, lr=LR, foreach=True)

    def update(self, b):
        s, s2 = self.norm(b["s"]), self.norm(b["s2"])
        with torch.no_grad():
            q_t = pick(self.q_targ(s), b["a"])                         # [E, B]
            y = b["r"] + GAMMA * (1.0 - b["term"]) * self.v(s2).squeeze(2)
        # V step: asymmetric squared loss |tau - 1(u < 0)| u^2 with u = Q - V
        v = self.v(s).squeeze(2)
        u = q_t - v
        w = torch.where(u < 0, 1.0 - IQL_EXPECTILE, IQL_EXPECTILE)
        v_loss = (w * u.pow(2)).mean(1).sum()
        # Q step: TD target uses V(s'), an in-sample quantity -- no max over actions
        q_loss = 0.5 * (pick(self.q(s), b["a"]) - y).pow(2).mean(1).sum()
        # policy step: advantage-weighted regression (AWR) on logged actions only
        with torch.no_grad():
            weight = torch.exp(IQL_BETA * (q_t - v)).clamp(max=100.0)
        logp = pick(F.log_softmax(self.pi(s), dim=2), b["a"])
        pi_loss = -(weight * logp).mean(1).sum()
        self.opt.zero_grad(); (v_loss + q_loss + pi_loss).backward(); self.opt.step()
        polyak_update(self.q_targ, self.q)
        return {"v": v_loss.item() / self.E, "q": q_loss.item() / self.E}

    @torch.no_grad()
    def act(self, obs):
        return self.pi(self.norm(torch.as_tensor(obs, dtype=torch.float32))).argmax(2).numpy()

    @torch.no_grad()
    def max_q(self, s):
        return self.q(self.norm(s)).max(2).values.mean(1).numpy()


METHODS = {"BC": BC, "naive DQN": DQN, "BCQ": BCQ, "CQL": CQL, "IQL": IQL}


@torch.no_grad()
def bcq_both_allowed(agent, data):
    """Per ensemble member: fraction of the dataset's states at which BCQ's filter (Eq. 16.23)
    allows both actions, i.e. at which BCQ can do anything other than clone the likelier one."""
    s = agent.norm(torch.as_tensor(data["s"])).unsqueeze(0).expand(agent.E, -1, -1)
    return agent.allowed(s).all(2).float().mean(1).numpy()


# --------------------------------------------------------------------------------------------
def train(method: str, data: dict, n_seeds: int, steps: int, eval_every: int,
          eval_episodes: int, seed: int, return_agent: bool = False):
    """Train `n_seeds` independent copies of one learner on one dataset (as one ensemble).
    Returns logged steps, greedy returns [seeds, evals] and max-Q diagnostics [seeds, evals]
    (and the trained agent if return_agent)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tens = to_tensors(data)
    n = len(data["a"])
    agent = METHODS[method](Normalizer(data["s"]), n_seeds)
    probe = tens["s"][rng.choice(n, size=min(2000, n), replace=False)].expand(n_seeds, -1, -1)
    curve, qcurve, steps_logged = [], [], []
    for step in range(1, steps + 1):
        idx = torch.as_tensor(rng.integers(0, n, size=(n_seeds, BATCH)))  # own batch per member
        agent.update({k: v[idx] for k, v in tens.items()})
        if step % eval_every == 0 or step == steps:
            curve.append(evaluate(agent.act, n_seeds, eval_episodes).mean(1))
            qcurve.append(agent.max_q(probe))
            steps_logged.append(step)
    out = (np.array(steps_logged), np.array(curve).T, np.array(qcurve, dtype=float).T)
    return (*out, agent) if return_agent else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    quick = args.quick

    n_data = 2_000 if quick else 20_000
    steps = 300 if quick else 8_000
    eval_every = 150 if quick else 1_000
    eval_episodes = 3 if quick else 10
    n_seeds = 2 if quick else args.seeds
    datasets = ["expert", "medium", "random"]
    methods = list(METHODS)
    print(f"offline_cartpole: quick={quick} seeds=0..{n_seeds - 1} (ensemble members) "
          f"dataset_size={n_data} grad_steps={steps} batch={BATCH} gamma={GAMMA} lr={LR} "
          f"polyak={POLYAK} cql_alpha={CQL_ALPHA} iql_expectile={IQL_EXPECTILE} "
          f"iql_beta={IQL_BETA} bcq_threshold={BCQ_THRESHOLD}")
    t0 = time.time()

    results, data_info = {}, {}
    for d_i, dname in enumerate(datasets):
        data = collect_dataset(dname, n_data, seed=1000 * (d_i + 1))
        mc = discounted_mc_returns(data, GAMMA)
        data_info[dname] = dict(mean_return=data["returns"].mean())
        print(f"\n[{dname}] {n_data} transitions, {len(data['returns'])} complete episodes, "
              f"behaviour return {data['returns'].mean():.1f} +- {data['returns'].std():.1f}, "
              f"P(a=1) {data['a'].mean():.2f}, mean discounted MC return-to-go {mc.mean():.1f}")
        results[dname] = {}
        for m_i, m in enumerate(methods):
            ts = time.time()
            st, rets, qs, agent = train(m, data, n_seeds, steps, eval_every, eval_episodes,
                                        seed=100 * d_i + m_i, return_agent=True)
            results[dname][m] = (st, rets, qs)
            qtxt = "      n/a" if np.isnan(qs).all() else \
                f"{np.mean(qs[:, -1]):9.1f} (max over run {np.max(np.abs(qs)):.3g})"
            print(f"  {m:10s} final return {rets[:, -1].mean():6.1f} +- {rets[:, -1].std():5.1f} "
                  f"per seed {np.round(rets[:, -1]).astype(int).tolist()}  mean max-Q {qtxt}  "
                  f"[{time.time() - ts:.0f}s]")
            if m == "BCQ":
                # Diagnostic for Section 7.3: how often does the filter (Eq. 16.23) let BOTH
                # actions through?  (Computed after training; uses no random numbers, so it
                # does not change any other result.)
                both = bcq_both_allowed(agent, data)
                print(f"  {'':10s} BCQ filter: fraction of dataset states where both actions are "
                      f"allowed {both.mean():.3f} (per seed {', '.join(f'{x:.3f}' for x in both)})")

    print(f"\nSummary: final greedy return, mean (sd) over seeds, {eval_episodes} eval episodes per seed")
    print(f"  {'dataset':8s} {'behaviour':>9s} " + " ".join(f"{m:>13s}" for m in methods))
    for dname in datasets:
        row = " ".join(f"{results[dname][m][1][:, -1].mean():7.1f} ({results[dname][m][1][:, -1].std():3.0f})"
                       for m in methods)
        print(f"  {dname:8s} {data_info[dname]['mean_return']:9.1f} {row}")
    print("Summary: mean over the last 3 evaluations (smooths evaluation noise)")
    for dname in datasets:
        row = " ".join(f"{results[dname][m][1][:, -3:].mean():13.1f}" for m in methods)
        print(f"  {dname:8s} {data_info[dname]['mean_return']:9.1f} {row}")
    print(f"total time {time.time() - t0:.0f}s")

    if quick:
        return
    import plot_style
    plt = plot_style.setup()
    C = plot_style.C
    colors = {m: C[i] for i, m in enumerate(methods)}
    markers = {"BC": "o", "naive DQN": "s", "BCQ": "^", "CQL": "D", "IQL": "v"}

    # Figure 1: learning curves of the greedy return, one panel per dataset
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for ax, dname in zip(axes, datasets):
        for m in methods:
            st, rets, _ = results[dname][m]
            mu, sd = rets.mean(0), rets.std(0)
            ax.plot(st, mu, marker=markers[m], ms=4, color=colors[m], label=m)
            ax.fill_between(st, mu - sd, mu + sd, color=colors[m], alpha=0.12, lw=0)
        ax.axhline(data_info[dname]["mean_return"], color=plot_style.GREY, ls="--", lw=1.2,
                   label="behaviour policy")
        ax.set_title(f"{dname} dataset ({n_data // 1000}k transitions)")
        ax.set_xlabel("gradient steps")
        ax.set_ylim(0, 520)
    axes[0].set_ylabel("greedy return (10 episodes)")
    axes[0].legend(loc="center right", fontsize=8)
    fig.suptitle("Offline RL on CartPole-v1: no environment interaction during training "
                 f"(mean $\\pm$ sd over {n_seeds} seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "offline_cartpole_returns.png"))
    plt.close(fig)

    # Figure 2: Q-value diagnostic (log scale), one panel per dataset
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for ax, dname in zip(axes, datasets):
        for m in methods[1:]:
            st, _, qs = results[dname][m]
            for k in range(n_seeds):
                ax.plot(st, np.abs(qs[k]), color=colors[m], marker=markers[m], ms=3, lw=1.1,
                        alpha=0.75, label=m if k == 0 else None)
        ax.axhline(1 / (1 - GAMMA), color=plot_style.INK, ls=":", lw=1.2,
                   label="$1/(1-\\gamma)$ = largest true value")
        ax.set_yscale("log")
        ax.set_title(f"{dname} dataset")
        ax.set_xlabel("gradient steps")
    axes[0].set_ylabel("$|$mean over data states of $\\max_a Q(s,a)|$")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Extrapolation error: Q estimates on the dataset's own states (one line per seed)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "offline_cartpole_qvalues.png"))
    plt.close(fig)
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
