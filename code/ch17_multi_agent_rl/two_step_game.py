"""Value decomposition in a SEQUENTIAL task: the two-step game of the QMIX paper
(Rashid et al., 2018), learned with TD targets, a replay buffer and target networks
(Chapter 17, section 7.5; Algorithm 7.1 on a toy scale).

The game (2 agents, 2 actions each, gamma = 1, every episode lasts two steps)
-----------------------------------------------------------------------------
  state 1 : reward 0. Agent 1's action chooses the next state: A -> state 2A, B -> 2B.
            (Agent 2's action has no effect.)
  state 2A: every joint action pays 7, then the episode terminates.
  state 2B: pays  [[0, 1], [1, 8]]  (rows: agent 1, columns: agent 2), then terminates.
The optimal joint policy goes to 2B and plays (B, B): return 8. Going to 2A is a safe 7.

Learners (all act on per-agent utilities Q_i(s, a^i); the agents observe s):
  central  one Q-table over joint actions, Q(s, a1, a2) (needs a joint argmax)
  IQL      each agent runs Q-learning on the team reward with its own Q_i(s, a^i)
  VDN      Q_tot = Q_1(s, a1) + Q_2(s, a2), trained end to end on the TD loss
  QMIX     Q_tot = f_mix(Q_1, Q_2; s) with a 1-hidden-layer ELU mixer whose weights are
           |hypernet(s)| (non-negative) and whose final bias is a small state-value net
Every factorised learner bootstraps from the DECENTRALISED greedy actions:
  y = r + gamma * (1 - terminated) * f(max_b Q_1^-(s', b), max_b Q_2^-(s', b); s')   (Eq. 7.5)
with target copies Q^- refreshed every C updates.

Two data regimes: uniform exploration (epsilon = 1 throughout, as in the paper's table)
and epsilon-greedy exploration annealed from 1 to 0.05 over the first half of training.

Run:  python code/ch17_multi_agent_rl/two_step_game.py [--quick]
"""
import argparse
import copy
import os
import time

import numpy as np
import torch

torch.set_num_threads(1)

S1, S2A, S2B = 0, 1, 2
NS, NA = 3, 2
PAYOFF_2B = np.array([[0.0, 1.0], [1.0, 8.0]])
METHODS = ["central", "IQL", "VDN", "QMIX"]


def env_step(s, a1, a2):
    """Returns (reward, next state or -1 if terminal)."""
    if s == S1:
        return 0.0, (S2A if a1 == 0 else S2B)
    if s == S2A:
        return 7.0, -1
    return PAYOFF_2B[a1, a2], -1


class Learner(torch.nn.Module):
    def __init__(self, method, hidden=8):
        super().__init__()
        self.method = method
        if method == "central":
            self.q = torch.nn.Parameter(0.1 * torch.randn(NS, NA, NA))
            return
        self.q1 = torch.nn.Parameter(0.1 * torch.randn(NS, NA))     # agent utilities Q_i(s, a^i)
        self.q2 = torch.nn.Parameter(0.1 * torch.randn(NS, NA))
        if method == "QMIX":
            self.H = hidden
            self.hyper_w1 = torch.nn.Linear(NS, 2 * hidden)            # state -> mixer weights
            self.hyper_b1 = torch.nn.Linear(NS, hidden)
            self.hyper_w2 = torch.nn.Linear(NS, hidden)
            self.hyper_v = torch.nn.Sequential(torch.nn.Linear(NS, hidden), torch.nn.ReLU(),
                                               torch.nn.Linear(hidden, 1))   # final bias V(s)

    def mix(self, u1, u2, s):
        """Q_tot from the two utilities u1, u2 (shape [B]) in states s (shape [B])."""
        if self.method == "VDN":
            return u1 + u2
        x = torch.nn.functional.one_hot(s, NS).float()
        w1 = self.hyper_w1(x).abs().view(-1, 2, self.H)                 # |W1| >= 0: monotone
        b1 = self.hyper_b1(x)
        w2 = self.hyper_w2(x).abs()                                     # |w2| >= 0: monotone
        h = torch.nn.functional.elu(torch.einsum("bk,bkh->bh", torch.stack([u1, u2], 1), w1) + b1)
        return (h * w2).sum(1) + self.hyper_v(x).squeeze(1)

    def q_taken(self, s, a1, a2):
        """Training-time value of the joint action taken (per agent for IQL)."""
        if self.method == "central":
            return self.q[s, a1, a2]
        u1, u2 = self.q1[s, a1], self.q2[s, a2]
        if self.method == "IQL":
            return torch.stack([u1, u2], 1)
        return self.mix(u1, u2, s)

    def q_greedy(self, s):
        """Value of the decentralised greedy joint action in s (for TD targets)."""
        if self.method == "central":
            return self.q[s].flatten(1).max(1).values
        u1, u2 = self.q1[s].max(1).values, self.q2[s].max(1).values
        if self.method == "IQL":
            return torch.stack([u1, u2], 1)
        return self.mix(u1, u2, s)

    def greedy_actions(self, s):
        with torch.no_grad():
            if self.method == "central":
                k = int(torch.argmax(self.q[s]))
                return k // NA, k % NA
            return int(torch.argmax(self.q1[s])), int(torch.argmax(self.q2[s]))

    def q_tot_table(self, s):
        """Q_tot(s, a1, a2) for all joint actions (for printing)."""
        with torch.no_grad():
            a1 = torch.arange(NA).repeat_interleave(NA)
            a2 = torch.arange(NA).repeat(NA)
            ss = torch.full((NA * NA,), s, dtype=torch.long)
            q = self.q_taken(ss, a1, a2)
            if self.method == "IQL":
                q = q.sum(1)                                            # (not a joint value)
            return q.view(NA, NA).numpy()


def greedy_return(model):
    a1, a2 = model.greedy_actions(S1)
    r1, s2 = env_step(S1, a1, a2)
    b1, b2 = model.greedy_actions(s2)
    r2, _ = env_step(s2, b1, b2)
    return r1 + r2


def train(method, regime, seed, episodes, batch=64, buffer_cap=2000, C=100, lr=0.03, gamma=1.0, eval_every=25):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = Learner(method)
    target = copy.deepcopy(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    buf = np.zeros((buffer_cap, 6))                  # s, a1, a2, r, s', terminated
    n_buf, ptr = 0, 0
    curve = []
    for ep in range(episodes):
        eps = 1.0 if regime == "uniform" else max(0.05, 1.0 - ep / (0.5 * episodes))
        s = S1
        while s >= 0:                                # one episode: two steps
            g1, g2 = model.greedy_actions(s)
            a1 = int(rng.integers(NA)) if rng.random() < eps else g1
            a2 = int(rng.integers(NA)) if rng.random() < eps else g2
            r, s2 = env_step(s, a1, a2)
            buf[ptr] = (s, a1, a2, r, max(s2, 0), float(s2 < 0))
            ptr = (ptr + 1) % buffer_cap
            n_buf = min(n_buf + 1, buffer_cap)
            s = s2
        # one gradient step on a minibatch of transitions (Algorithm 7.1, Eq. 7.5)
        b = buf[rng.integers(n_buf, size=batch)]
        bs, ba1, ba2 = (torch.tensor(b[:, k], dtype=torch.long) for k in range(3))
        br, bs2, bterm = (torch.tensor(b[:, k], dtype=torch.float32) for k in (3, 4, 5))
        with torch.no_grad():
            nxt = target.q_greedy(bs2.long())
            if method == "IQL":
                y = br[:, None] + gamma * (1 - bterm[:, None]) * nxt   # each agent: own target
            else:
                y = br + gamma * (1 - bterm) * nxt                     # terminal: y = r only
        loss = ((model.q_taken(bs, ba1, ba2) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % C == 0:
            target.load_state_dict(model.state_dict())
        if (ep + 1) % eval_every == 0:
            curve.append(greedy_return(model))
    return model, np.array(curve)


def main():
    parser = argparse.ArgumentParser(description="Two-step game: IQL / VDN / QMIX with TD learning")
    parser.add_argument("--quick", action="store_true", help="smoke test")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    episodes = 300 if args.quick else 1500
    n_seeds = 2 if args.quick else 10
    print(f"two_step_game.py | seed={args.seed} | {n_seeds} seeds x {episodes} episodes (one Adam step "
          f"per episode, batch 64, lr 0.03, buffer 2000 transitions, target copy every 100 updates) | "
          f"gamma=1 | QMIX hidden=8 | quick={args.quick}")
    t0 = time.time()
    summary, curves = {}, {}
    for regime in ("uniform", "eps-greedy"):
        print(f"\n=== data regime: {regime} "
              f"({'epsilon = 1 throughout' if regime == 'uniform' else 'epsilon 1 -> 0.05 over the first half'}) ===")
        for method in METHODS:
            rets, models, cs = [], [], []
            for k in range(n_seeds):
                m, c = train(method, regime, args.seed + k, episodes)
                rets.append(greedy_return(m))
                models.append(m)
                cs.append(c)
            rets = np.array(rets)
            summary[(regime, method)] = rets
            curves[(regime, method)] = np.mean(cs, axis=0)
            print(f"  {method:8s} greedy decentralised return per seed: {rets.astype(int).tolist()} "
                  f"-> optimal (8) in {np.mean(rets == 8):.0%} of seeds, mean {rets.mean():.2f}")
            if method != "IQL":
                m = models[0]
                t1, t2b = m.q_tot_table(S1), m.q_tot_table(S2B)
                print(f"     seed 0 Q_tot  state 1: [[{t1[0, 0]:5.2f} {t1[0, 1]:5.2f}] [{t1[1, 0]:5.2f} {t1[1, 1]:5.2f}]]"
                      f"   state 2B: [[{t2b[0, 0]:5.2f} {t2b[0, 1]:5.2f}] [{t2b[1, 0]:5.2f} {t2b[1, 1]:5.2f}]]"
                      f"   state 2A max: {m.q_tot_table(S2A).max():5.2f}")
            else:
                m = models[0]
                with torch.no_grad():
                    print(f"     seed 0 agent 1 utilities: state 1 {np.round(m.q1[S1].numpy(), 2)}, "
                          f"state 2B {np.round(m.q1[S2B].numpy(), 2)}; agent 2 in 2B {np.round(m.q2[S2B].numpy(), 2)}")
    print("\nsummary (fraction of seeds whose greedy decentralised policy earns the optimal 8):")
    for regime in ("uniform", "eps-greedy"):
        print(f"  {regime:10s} " + "  ".join(f"{mth} {np.mean(summary[(regime, mth)] == 8):.0%}" for mth in METHODS))
    print(f"total time {time.time() - t0:.1f} s")

    if not args.quick:
        import plot_style
        plt = plot_style.setup()
        figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
        os.makedirs(figdir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        st = {"central": dict(color=plot_style.C[0], ls="-"), "IQL": dict(color=plot_style.C[1], ls="--"),
              "VDN": dict(color=plot_style.C[2], ls="-."), "QMIX": dict(color=plot_style.C[6], ls=":", lw=2.6)}
        for ax, regime in zip(axes, ("uniform", "eps-greedy")):
            for method in METHODS:
                c = curves[(regime, method)]
                ax.plot(25 * np.arange(1, len(c) + 1), c, label=method, **st[method])
            ax.axhline(8, color=plot_style.GREY, lw=1, ls=(0, (1, 2)))
            ax.set_xlabel("episodes (one gradient step each)")
            ax.set_title("uniform exploration (epsilon = 1)" if regime == "uniform"
                         else "epsilon-greedy, 1 -> 0.05 over the first half")
        axes[0].set_ylabel(f"greedy decentralised return (mean of {n_seeds} seeds)")
        axes[0].set_ylim(5.8, 8.3)
        axes[0].legend(loc="lower right")
        fig.suptitle("Two-step game (Rashid et al., 2018): optimal return 8, safe return 7", y=1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "two_step_game.png"))
        plt.close(fig)
        print(f"figure written to {figdir}/two_step_game.png")


if __name__ == "__main__":
    main()
