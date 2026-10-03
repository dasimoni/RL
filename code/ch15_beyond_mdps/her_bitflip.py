"""Hindsight Experience Replay (HER) on the bit-flipping environment, with a DQN learner.

Chapter 15, Section 4 (Algorithm 15.3).  Reproduces the qualitative result of Andrychowicz et
al. (2017, Fig. 1) at a CPU-sized budget: with a sparse "did you reach the goal?" reward, DQN
alone essentially never succeeds once n is around 10 or more, while DQN + HER keeps working.

Environment (Andrychowicz et al., 2017, Sec. 3.1).  State s in {0,1}^n, goal g in {0,1}^n, both
drawn uniformly at random (g != s).  Action i flips bit i.  Reward r = 0 if the NEXT state equals
the goal and -1 otherwise.  Reaching the goal TERMINATES the episode (every action flips a bit,
so the agent could not stay there anyway); otherwise the episode is TRUNCATED after T = n steps.
Bootstrapping follows the course convention: through truncation, never through termination.

Learner.  Goal-conditioned DQN (a UVFA, Section 4.2): Q(s, a, g; w) is an MLP on the concatenation
[s, g] with n outputs (one hidden layer of 256 units, as in the HER paper's bit-flipping setup).
Episodes are collected 16 at a time (one "cycle"), then 40 minibatch updates are made; the target
network is Polyak-averaged after every cycle.  Batch 128, Adam 1e-3, Polyak 0.95, gamma 0.98,
20% random actions and targets clipped to [-1/(1-gamma), 0] follow the general training details
in the appendix of the HER paper.

HER.  At the end of each episode, every transition (s_t, a_t, s_{t+1}) is stored with the
original goal AND with k extra goals g' taken from states the episode actually achieved; the
reward and the termination flag are recomputed for g'.  Strategies (Section 4.3; Andrychowicz et al., 2017):
  final   : g' = the last state of the episode (k = 1)
  future  : g' = s_{t'} for t' sampled uniformly from t+1..T (the states reached AFTER step t)
  episode : g' = any state reached in the episode
  random  : g' = any state stored in the replay buffer so far

Run:  python code/ch15_beyond_mdps/her_bitflip.py [--quick]
Full mode writes figures/her_bitflip_scaling.png and figures/her_strategies.png.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one CPU thread: multithreaded BLAS on a
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # shared machine makes small solves ~1000x slower
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

GAMMA = 0.98
N_ENVS = 16           # episodes per cycle
UPDATES_PER_CYCLE = 40
BATCH = 128
LR = 1e-3
POLYAK = 0.95         # target <- 0.95 target + 0.05 online, once per cycle
EPS = 0.2             # epsilon-greedy exploration during training
HIDDEN = 256
BUFFER = 300_000


# ----------------------------------------------------------------------------------------------
# Environment (vectorised: B independent copies stepped in lock-step)
# ----------------------------------------------------------------------------------------------
def sample_tasks(rng, B, n):
    s = rng.integers(0, 2, size=(B, n), dtype=np.int8)
    g = rng.integers(0, 2, size=(B, n), dtype=np.int8)
    same = (s == g).all(axis=1)
    while same.any():                            # make sure the goal is not already achieved
        g[same] = rng.integers(0, 2, size=(int(same.sum()), n), dtype=np.int8)
        same = (s == g).all(axis=1)
    return s, g


def step(s, a):
    s2 = s.copy()
    s2[np.arange(len(s)), a] ^= 1
    return s2


# ----------------------------------------------------------------------------------------------
# Goal-conditioned Q-network (UVFA) and replay buffer
# ----------------------------------------------------------------------------------------------
class QNet(nn.Module):
    def __init__(self, n, hidden=HIDDEN):
        super().__init__()
        self.l1 = nn.Linear(2 * n, hidden)
        self.l2 = nn.Linear(hidden, n)

    def forward(self, s, g):
        return self.l2(F.relu(self.l1(torch.cat([s, g], dim=-1))))


class Replay:
    def __init__(self, n, capacity):
        self.s = np.zeros((capacity, n), dtype=np.int8)
        self.s2 = np.zeros((capacity, n), dtype=np.int8)
        self.g = np.zeros((capacity, n), dtype=np.int8)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.cap, self.size, self.ptr = capacity, 0, 0

    def add(self, s, a, s2, g):
        """Store a batch of transitions; reward and termination are recomputed from (s2, g)."""
        m = len(s)
        idx = (self.ptr + np.arange(m)) % self.cap
        self.s[idx], self.a[idx], self.s2[idx], self.g[idx] = s, a, s2, g
        self.ptr = (self.ptr + m) % self.cap
        self.size = min(self.size + m, self.cap)

    def sample(self, rng, batch):
        idx = rng.integers(0, self.size, size=batch)
        s, a, s2, g = self.s[idx], self.a[idx], self.s2[idx], self.g[idx]
        done = (s2 == g).all(axis=1)             # goal reached -> terminal transition
        r = np.where(done, 0.0, -1.0)            # r = -[s' != g]
        return s, a, s2, g, r, done

    def random_achieved(self, rng, m):
        """'random' strategy: achieved states drawn from the whole buffer."""
        return self.s2[rng.integers(0, self.size, size=m)]


def relabel(rng, states, actions, goal, strategy, k, replay):
    """Hindsight goals for ONE episode (Algorithm 15.3, the inner loop over t).

    states: (T+1, n) array s_0..s_T; actions: (T,).  Returns arrays (s, a, s', g) to store:
    the T original transitions plus k relabelled copies of each (one for 'final').
    """
    T = len(actions)
    S, A, S2 = states[:-1], actions, states[1:]
    out_s, out_a, out_s2, out_g = [S], [A], [S2], [np.repeat(goal[None], T, axis=0)]
    if strategy == "none":
        pass
    elif strategy == "final":
        out_s.append(S); out_a.append(A); out_s2.append(S2)
        out_g.append(np.repeat(states[-1][None], T, axis=0))
    else:
        for _ in range(k):
            if strategy == "future":
                # t' uniform in {t+1, ..., T}: a state achieved at or after s_{t+1}
                tp = np.array([rng.integers(t + 1, T + 1) for t in range(T)])
                g_new = states[tp]
            elif strategy == "episode":
                g_new = states[rng.integers(1, T + 1, size=T)]
            elif strategy == "random":
                g_new = replay.random_achieved(rng, T) if replay.size > 0 else states[rng.integers(1, T + 1, size=T)]
            else:
                raise ValueError(strategy)
            out_s.append(S); out_a.append(A); out_s2.append(S2); out_g.append(g_new)
    return (np.concatenate(out_s), np.concatenate(out_a), np.concatenate(out_s2), np.concatenate(out_g))


def to_t(x):
    return torch.as_tensor(x, dtype=torch.float32)


def evaluate(net, rng, n, episodes=256):
    """Greedy success rate: fraction of fresh (s0, g) pairs solved within T = n steps."""
    s, g = sample_tasks(rng, episodes, n)
    active = np.ones(episodes, dtype=bool)
    with torch.no_grad():
        for _ in range(n):
            a = net(to_t(s), to_t(g)).argmax(dim=1).numpy()
            s = np.where(active[:, None], step(s, a), s)
            active &= ~(s == g).all(axis=1)
    return 1.0 - active.mean()


def random_policy_success(rng, n, episodes=20_000):
    s, g = sample_tasks(rng, episodes, n)
    active = np.ones(episodes, dtype=bool)
    for _ in range(n):
        s = np.where(active[:, None], step(s, rng.integers(0, n, size=episodes)), s)
        active &= ~(s == g).all(axis=1)
    return 1.0 - active.mean()


def train(n, strategy, k, cycles, seed, eval_every=10, eval_episodes=256):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net, target = QNet(n), QNet(n)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    replay = Replay(n, BUFFER)
    clip_lo = -1.0 / (1.0 - GAMMA)
    curve = []
    train_success = []
    for cycle in range(1, cycles + 1):
        # ---- collect N_ENVS episodes in lock-step (epsilon-greedy behaviour policy) ----------
        s, g = sample_tasks(rng, N_ENVS, n)
        traj = [s.copy()]
        acts = []
        active = np.ones(N_ENVS, dtype=bool)
        length = np.full(N_ENVS, n)
        for t in range(n):
            with torch.no_grad():
                greedy = net(to_t(s), to_t(g)).argmax(dim=1).numpy()
            explore = rng.random(N_ENVS) < EPS
            a = np.where(explore, rng.integers(0, n, size=N_ENVS), greedy)
            s = np.where(active[:, None], step(s, a), s)
            traj.append(s.copy())
            acts.append(a)
            reached = active & (s == g).all(axis=1)
            length[reached] = t + 1                 # terminated: episode ends here
            active &= ~reached
            if not active.any():
                break
        traj = np.stack(traj)                       # (steps+1, N_ENVS, n)
        acts = np.stack(acts)                       # (steps, N_ENVS)
        train_success.append(float((~active).mean()))   # episodes that reached their goal
        for e in range(N_ENVS):                     # store each episode, with hindsight goals
            T = length[e]
            batch = relabel(rng, traj[:T + 1, e], acts[:T, e], g[e], strategy, k, replay)
            replay.add(*batch)
        # ---- learn ---------------------------------------------------------------------------
        for _ in range(UPDATES_PER_CYCLE):
            bs, ba, bs2, bg, br, bdone = replay.sample(rng, BATCH)
            q = net(to_t(bs), to_t(bg)).gather(1, torch.as_tensor(ba)[:, None]).squeeze(1)
            with torch.no_grad():
                q_next = target(to_t(bs2), to_t(bg)).max(dim=1).values
                y = to_t(br) + GAMMA * (1.0 - to_t(bdone.astype(np.float32))) * q_next
                y = y.clamp(clip_lo, 0.0)           # returns lie in [-1/(1-gamma), 0]
            loss = F.mse_loss(q, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():                       # Polyak-averaged target network
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.mul_(POLYAK).add_((1 - POLYAK) * p)
        if cycle % eval_every == 0 or cycle == cycles:
            curve.append((cycle, evaluate(net, rng, n, eval_episodes)))
    return np.array(curve), np.array(train_success)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figures")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    if args.quick:
        ns, seeds, cycles_for = [8], [0], (lambda n: 30)
        strat_n, strat_cycles, strategies = 8, 20, [("future", 4)]
    else:
        ns, seeds = [5, 10, 15, 20], list(range(args.seeds))
        cycles_for = lambda n: {5: 50, 10: 120, 15: 200, 20: 300}[n]
        strat_n, strat_cycles = 15, 200
        strategies = [("final", 1), ("future", 4), ("episode", 4), ("random", 4)]
    print(f"HER bit-flipping | seeds={seeds} gamma={GAMMA} episodes/cycle={N_ENVS} "
          f"updates/cycle={UPDATES_PER_CYCLE} batch={BATCH} lr={LR} eps={EPS} hidden={HIDDEN} "
          f"polyak={POLYAK} quick={args.quick}")
    rng = np.random.default_rng(123)
    rand_succ = {n: random_policy_success(rng, n) for n in ns}
    print("random-policy success rate within n steps: "
          + ", ".join(f"n={n}: {p:.4f}" for n, p in rand_succ.items()))

    # ---- experiment 1: success vs number of bits, with and without HER (future, k = 4) -------
    scaling = {}
    for n in ns:
        for label, strategy, k in [("DQN", "none", 0), ("DQN + HER (future, k=4)", "future", 4)]:
            finals, curves = [], []
            t1 = time.time()
            for seed in seeds:
                curve, _ = train(n, strategy, k, cycles_for(n), seed)
                finals.append(curve[-1, 1])
                curves.append(curve)
            scaling[(n, label)] = (np.array(finals), curves)
            print(f"n={n:2d} {label:<24} cycles={cycles_for(n):4d} ({cycles_for(n) * N_ENVS} episodes): "
                  f"final greedy success = {np.mean(finals):.3f} "
                  f"(per seed: {', '.join(f'{f:.2f}' for f in finals)})  [{time.time() - t1:.0f} s]")

    # ---- experiment 2: relabelling strategies at n = strat_n ---------------------------------
    strat = {}
    for strategy, k in strategies:
        curves = []
        t1 = time.time()
        for seed in seeds:
            if (strat_n, "DQN + HER (future, k=4)") in scaling and strategy == "future" and k == 4:
                curve = scaling[(strat_n, "DQN + HER (future, k=4)")][1][seeds.index(seed)]
            else:
                curve, _ = train(strat_n, strategy, k, strat_cycles, seed)
            curves.append(curve)
        strat[(strategy, k)] = curves
        finals = [c[-1, 1] for c in curves]
        first = [str(int(c[np.argmax(c[:, 1] >= 0.9), 0])) if (c[:, 1] >= 0.9).any() else "never"
                 for c in curves]
        # budget-based summaries: the threshold above hides how fast each strategy gets going
        ys = np.stack([c[:, 1] for c in curves])
        cyc = curves[0][:, 0]
        at = {cy: ys[:, cyc == cy].mean() for cy in (40, 60, 80, 100) if (cyc == cy).any()}
        print(f"strategy {strategy:<8} k={k}: final success at n={strat_n} = {np.mean(finals):.3f} "
              f"(per seed: {', '.join(f'{f:.2f}' for f in finals)}); first evaluation (every 10 cycles) "
              f"with success >= 0.9 at cycle: {', '.join(first)}  [{time.time() - t1:.0f} s]")
        print(f"         mean success over all {ys.shape[1]} evaluations (area under the curve) = "
              f"{ys.mean():.3f}; mean success at cycle "
              + ", ".join(f"{cy} ({cy * N_ENVS} episodes): {v:.3f}" for cy, v in at.items()))

    if not args.quick:
        from plot_style import setup, C
        plt = setup()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        ax = axes[0]
        for i, label in enumerate(["DQN", "DQN + HER (future, k=4)"]):
            m = [scaling[(n, label)][0].mean() for n in ns]
            lo = [scaling[(n, label)][0].min() for n in ns]
            hi = [scaling[(n, label)][0].max() for n in ns]
            ax.plot(ns, m, marker="os"[i], color=C[i], ls=["--", "-"][i], label=label)
            ax.fill_between(ns, lo, hi, color=C[i], alpha=0.15)
        ax.plot(ns, [rand_succ[n] for n in ns], marker="^", color=C[3], ls=":", label="uniform random policy")
        ax.set_xlabel("number of bits n")
        ax.set_ylabel("greedy success rate at end of training")
        ax.set_title(f"Bit flipping: mean over {len(seeds)} seeds (band: min-max)")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(ns)
        ax.legend()
        ax = axes[1]
        n_show = ns[-2] if len(ns) > 1 else ns[0]
        for i, label in enumerate(["DQN", "DQN + HER (future, k=4)"]):
            curves = scaling[(n_show, label)][1]
            x = curves[0][:, 0] * N_ENVS
            ys = np.stack([c[:, 1] for c in curves])
            ax.plot(x, ys.mean(0), color=C[i], ls=["--", "-"][i], label=label)
            ax.fill_between(x, ys.min(0), ys.max(0), color=C[i], alpha=0.15)
        ax.set_xlabel("training episodes")
        ax.set_ylabel("greedy success rate")
        ax.set_title(f"Learning curves, n = {n_show}")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(loc="center right")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "her_bitflip_scaling.png"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.2))
        styles = ["-.", "-", "--", ":"]
        for i, (key, curves) in enumerate(strat.items()):
            x = curves[0][:, 0] * N_ENVS
            ys = np.stack([c[:, 1] for c in curves])
            ax.plot(x, ys.mean(0), color=C[i + 2], ls=styles[i], label=f"HER '{key[0]}', k={key[1]}")
            ax.fill_between(x, ys.min(0), ys.max(0), color=C[i + 2], alpha=0.12)
        dq = scaling[(strat_n, "DQN")][1]
        ax.plot(dq[0][:, 0] * N_ENVS, np.stack([c[:, 1] for c in dq]).mean(0), color=C[0], ls="--",
                label="no HER")
        ax.set_xlabel("training episodes")
        ax.set_ylabel("greedy success rate")
        ax.set_title(f"Goal-relabelling strategies, n = {strat_n} bits ({len(seeds)} seeds)")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "her_strategies.png"))
        plt.close(fig)
        print(f"figures written to {FIG_DIR}")
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
