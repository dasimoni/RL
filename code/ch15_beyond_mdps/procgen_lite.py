"""Generalization across environments on a "Procgen-lite" maze family; uniform vs PLR level sampling.

Chapter 15, Section 10 (Eq. 15.33, Algorithm 15.14). A contextual MDP: each LEVEL c (a random seed)
defines a maze - interior size 7x7, 8x8 or 9x9, each interior cell a wall with probability 0.25, a
start and a goal at least half the size apart (Manhattan), regenerated until the goal is reachable.
The agent sees only an egocentric 5x5 window (walls and goal) plus the signs of the goal's
row/column offset (6 bits), so nothing in the observation names the level: generalization to a new
maze has to come from the policy itself, not from a lookup. Actions: up/down/left/right; reward +1
for reaching the goal (the episode ends), 0 otherwise; gamma = 0.99; time limit 60 steps.

The agent is PPO with a 2 x 64 tanh MLP (32 parallel environments, rollouts of 32 steps), trained
for a FIXED budget of environment steps on a training set of N levels, N in {1, 4, 16, 64, 256}
or "unlimited" (a fresh level every episode).  We report the success rate on the training levels
and on 200 held-out levels (sampled policy, as in Cobbe et al.'s Procgen evaluation).

Level sampling: "uniform" draws training levels uniformly.  "PLR" (Prioritized Level Replay,
Jiang, Grefenstette & Rocktaschel, 2021) keeps a score per level and replays levels from the
distribution below.  The score is the "positive value loss", the mean of max(GAE advantage, 0) over
the level's last episode (segment), the regret proxy of robust PLR (Jiang et al., NeurIPS 2021);
the original PLR paper's default score is the L1 value loss, the mean of |GAE advantage|.  Replay
distribution:
    P_replay = (1 - lam) P_score + lam P_stale,  P_score(i) ~ (1 / rank(S_i))^(1/beta),
    P_stale(i) ~ (episodes so far - episode count when i was last played),
with beta = 0.1 and lam = 0.1 (the chapter's beta_PLR and rho_PLR); unseen training levels are
drawn with probability 0.5 until all have been seen.

Run:  python code/ch15_beyond_mdps/procgen_lite.py [--quick]
Full mode writes figures/procgen_lite.png.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
MAXN, VIEW, PAD = 9, 5, 2                 # largest interior size, egocentric window, wall padding
GRID = MAXN + 2 + 2 * PAD                 # padded array side
T_MAX, GAMMA, LAM_GAE = 60, 0.99, 0.95
MOVES = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])
TEST_SEED0 = 10_000_000                   # held-out levels use seeds TEST_SEED0 + j
OBS_DIM = 2 * VIEW * VIEW + 6


def make_level(seed):
    """Walls (GRID x GRID bool, padded), start and goal (padded coordinates)."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(7, MAXN + 1))
    while True:
        walls = np.ones((GRID, GRID), dtype=bool)
        inner = rng.random((n, n)) < 0.25
        walls[PAD + 1:PAD + 1 + n, PAD + 1:PAD + 1 + n] = inner
        free = np.argwhere(~inner)
        if len(free) < 2:
            continue
        i, j = rng.choice(len(free), size=2, replace=False)
        s, g = free[i] + PAD + 1, free[j] + PAD + 1
        if np.abs(s - g).sum() < n // 2:
            continue
        # BFS reachability
        seen = {tuple(s)}
        q = deque([tuple(s)])
        while q:
            r, c = q.popleft()
            for dr, dc in MOVES:
                nb = (r + dr, c + dc)
                if not walls[nb] and nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        if tuple(g) in seen:
            return walls, s, g


class LevelCache:
    def __init__(self):
        self.cache = {}

    def get(self, seed):
        if seed not in self.cache:
            self.cache[seed] = make_level(seed)
        return self.cache[seed]


class VecMaze:
    """n parallel maze environments; levels are assigned by the caller at every reset."""

    def __init__(self, n, cache):
        self.n, self.cache = n, cache
        self.walls = np.ones((n, GRID, GRID), dtype=bool)
        self.goal_map = np.zeros((n, GRID, GRID), dtype=bool)
        self.pos = np.zeros((n, 2), dtype=int)
        self.goal = np.zeros((n, 2), dtype=int)
        self.t = np.zeros(n, dtype=int)
        self.level = np.zeros(n, dtype=np.int64)
        off = np.arange(-PAD, PAD + 1)
        self.dr, self.dc = np.meshgrid(off, off, indexing="ij")

    def reset_one(self, k, seed):
        walls, s, g = self.cache.get(seed)
        self.walls[k] = walls
        self.goal_map[k] = False
        self.goal_map[k, g[0], g[1]] = True
        self.pos[k], self.goal[k], self.t[k], self.level[k] = s, g, 0, seed

    def obs(self):
        rr = self.pos[:, 0, None, None] + self.dr[None]
        cc = self.pos[:, 1, None, None] + self.dc[None]
        ar = np.arange(self.n)[:, None, None]
        w = self.walls[ar, rr, cc].reshape(self.n, -1)
        g = self.goal_map[ar, rr, cc].reshape(self.n, -1)
        d = np.sign(self.goal - self.pos)                                  # (n, 2) in {-1, 0, 1}
        dirs = np.concatenate([np.eye(3)[d[:, 0] + 1], np.eye(3)[d[:, 1] + 1]], axis=1)
        return np.concatenate([w, g, dirs], axis=1).astype(np.float32)

    def step(self, a):
        nxt = self.pos + MOVES[a]
        blocked = self.walls[np.arange(self.n), nxt[:, 0], nxt[:, 1]]
        self.pos = np.where(blocked[:, None], self.pos, nxt)
        self.t += 1
        success = np.all(self.pos == self.goal, axis=1)
        timeout = (self.t >= T_MAX) & ~success
        return success.astype(np.float32), success, timeout


class ActorCritic(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                nn.Linear(hidden, 4))
        self.v = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, 1))

    def forward(self, x):
        return self.pi(x), self.v(x).squeeze(-1)


class UniformSampler:
    def __init__(self, levels, rng):
        self.levels, self.rng = levels, rng

    def sample(self):
        if self.levels is None:                                  # unlimited: a fresh level every time
            return int(self.rng.integers(TEST_SEED0))
        return int(self.levels[self.rng.integers(len(self.levels))])

    def update(self, seed, score):
        pass


class PLRSampler:
    """Prioritized Level Replay (Jiang, Grefenstette & Rocktaschel, 2021), rank prioritization."""

    def __init__(self, levels, rng, beta=0.1, lam=0.1, p_new=0.5):
        self.levels, self.rng = list(levels), rng
        self.beta, self.lam, self.p_new = beta, lam, p_new
        self.score = {}
        self.last = {}
        self.episodes = 0

    def probs(self):
        seen = list(self.score)
        S = np.array([self.score[i] for i in seen])
        rank = np.empty(len(S))
        rank[np.argsort(-S)] = np.arange(1, len(S) + 1)
        h = (1.0 / rank) ** (1.0 / self.beta)
        Ps = h / h.sum()
        stale = np.array([self.episodes - self.last[i] for i in seen], dtype=float)
        Pc = stale / stale.sum() if stale.sum() > 0 else np.full(len(S), 1.0 / len(S))
        return seen, (1 - self.lam) * Ps + self.lam * Pc

    def sample(self):
        self.episodes += 1
        unseen = len(self.score) < len(self.levels)
        if unseen and (not self.score or self.rng.random() < self.p_new):
            pool = [i for i in self.levels if i not in self.score]
            lvl = int(pool[self.rng.integers(len(pool))])
            self.score[lvl] = 0.0                                # placeholder until its first episode ends
        else:
            seen, p = self.probs()
            lvl = int(seen[self.rng.choice(len(seen), p=p)])
        self.last[lvl] = self.episodes
        return lvl

    def update(self, seed, score):
        self.score[seed] = score


def evaluate(net, cache, seeds, episodes_per_seed, rng, n_env=50):
    """Success rate of the SAMPLED policy, episodes_per_seed episodes on each level."""
    jobs = [s for s in seeds for _ in range(episodes_per_seed)]
    succ = 0
    for k0 in range(0, len(jobs), n_env):
        batch = jobs[k0:k0 + n_env]
        env = VecMaze(len(batch), cache)
        for k, s in enumerate(batch):
            env.reset_one(k, s)
        done = np.zeros(len(batch), dtype=bool)
        for _ in range(T_MAX):
            with torch.no_grad():
                logits, _ = net(torch.from_numpy(env.obs()))
            a = torch.distributions.Categorical(logits=logits).sample().numpy()
            _, success, timeout = env.step(a)
            succ += int((success & ~done).sum())
            done |= success | timeout
            if done.all():
                break
    return succ / len(jobs)


def train(n_levels, sampler_name, seed, total_steps, cache, n_env=32, n_roll=32, epochs=4, n_mb=4,
          lr=7e-4, clip=0.2, ent=0.01):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    levels = None if n_levels is None else list(range(seed * 100_000, seed * 100_000 + n_levels))
    sampler = (PLRSampler if sampler_name == "PLR" else UniformSampler)(levels, rng)
    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    env = VecMaze(n_env, cache)
    plays = {}                                                   # episodes started per level (diagnostic)

    def next_level():
        lvl = sampler.sample()
        plays[lvl] = plays.get(lvl, 0) + 1
        return lvl

    for k in range(n_env):
        env.reset_one(k, next_level())
    obs = env.obs()
    n_updates = total_steps // (n_env * n_roll)
    for u in range(n_updates):
        for g in opt.param_groups:
            g["lr"] = lr * (1 - u / n_updates)
        O = np.zeros((n_roll, n_env, OBS_DIM), dtype=np.float32)
        A = np.zeros((n_roll, n_env), dtype=np.int64)
        R = np.zeros((n_roll, n_env), dtype=np.float32)
        D = np.zeros((n_roll, n_env), dtype=np.float32)          # 1 if the episode TERMINATED (goal)
        TR = np.zeros((n_roll, n_env), dtype=bool)               # episode ended (goal or time-out)
        V = np.zeros((n_roll + 1, n_env), dtype=np.float32)
        LP = np.zeros((n_roll, n_env), dtype=np.float32)
        LV = np.zeros((n_roll, n_env), dtype=np.int64)
        V_TRUNC = np.zeros((n_roll, n_env), dtype=np.float32)    # bootstrap value at time-outs
        for t in range(n_roll):
            with torch.no_grad():
                logits, v = net(torch.from_numpy(obs))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            O[t], A[t], V[t], LP[t], LV[t] = obs, a.numpy(), v.numpy(), dist.log_prob(a).numpy(), env.level
            r, success, timeout = env.step(a.numpy())
            R[t], D[t], TR[t] = r, success, success | timeout
            if timeout.any():                                    # truncation: bootstrap from the last state
                with torch.no_grad():
                    _, v_last = net(torch.from_numpy(env.obs()))
                V_TRUNC[t] = np.where(timeout, v_last.numpy(), 0.0)
            for k in np.where(success | timeout)[0]:
                env.reset_one(k, next_level())
            obs = env.obs()
        with torch.no_grad():
            _, v = net(torch.from_numpy(obs))
        V[n_roll] = v.numpy()
        # GAE; at an episode end the next value is 0 (goal) or V(last state) (time-out), and the
        # advantage recursion is cut
        adv = np.zeros((n_roll, n_env), dtype=np.float32)
        last = np.zeros(n_env, dtype=np.float32)
        for t in reversed(range(n_roll)):
            v_next = np.where(TR[t], V_TRUNC[t] * (1 - D[t]), V[t + 1])
            delta = R[t] + GAMMA * v_next - V[t]
            last = delta + GAMMA * LAM_GAE * np.where(TR[t], 0.0, last)
            adv[t] = last
        ret = adv + V[:n_roll]
        # PLR score: positive value loss = mean max(advantage, 0) over each level's segment in this rollout
        if sampler_name == "PLR":
            for k in range(n_env):
                start = 0
                for t in range(n_roll):
                    if TR[t, k] or t == n_roll - 1:
                        seg = adv[start:t + 1, k]
                        sampler.update(int(LV[t, k]), float(np.maximum(seg, 0).mean()))
                        start = t + 1
        # PPO update
        b_o = torch.from_numpy(O.reshape(-1, OBS_DIM))
        b_a = torch.from_numpy(A.reshape(-1))
        b_lp = torch.from_numpy(LP.reshape(-1))
        b_adv = torch.from_numpy(adv.reshape(-1))
        b_ret = torch.from_numpy(ret.reshape(-1))
        n = b_o.shape[0]
        for _ in range(epochs):
            perm = torch.from_numpy(rng.permutation(n))
            for mb in perm.chunk(n_mb):
                logits, v = net(b_o[mb])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(b_a[mb]) - b_lp[mb])
                a_mb = b_adv[mb]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg = -torch.min(ratio * a_mb, ratio.clamp(1 - clip, 1 + clip) * a_mb).mean()
                loss = pg + 0.5 * ((v - b_ret[mb]) ** 2).mean() - ent * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
    return net, levels, plays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figures")
    ap.add_argument("--steps", type=int, default=200_000, help="environment steps per run")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    t0 = time.time()
    if args.quick:
        Ns, plr_Ns, seeds, steps, n_test = [1, 64], [64], [0], 20_000, 50
    else:
        Ns, plr_Ns, seeds, steps, n_test = [1, 4, 16, 64, 256, None], [16, 64, 256], args.seeds, args.steps, 200
    print(f"Procgen-lite mazes | PPO, MLP 2x64, {steps} env steps per run, seeds={seeds}, "
          f"N={['unlimited' if n is None else n for n in Ns]}, PLR at N={plr_Ns}, held-out levels={n_test}, "
          f"quick={args.quick}")
    cache = LevelCache()
    test_seeds = [TEST_SEED0 + j for j in range(n_test)]
    ev_rng = np.random.default_rng(12345)
    res = {}
    for sampler in ["uniform", "PLR"]:
        for N in (Ns if sampler == "uniform" else plr_Ns):
            rows = []
            for seed in seeds:
                t1 = time.time()
                net, levels, plays = train(N, sampler, seed, steps, cache)
                if levels is None:
                    train_succ = float("nan")
                else:
                    reps = max(1, 200 // len(levels))
                    train_succ = evaluate(net, cache, levels[:200], reps, ev_rng)
                test_succ = evaluate(net, cache, test_seeds, 2, ev_rng)
                # concentration of training: share of episodes on the most-played 10% of levels
                counts = np.sort(np.array(list(plays.values())))[::-1]
                top = counts[:max(1, len(counts) // 10)].sum() / counts.sum()
                rows.append((train_succ, test_succ, top, counts.sum()))
                name = "unlimited" if N is None else N
                print(f"[{sampler:>7}] N={name!s:>9} seed={seed}: success train {train_succ:.3f}, "
                      f"held-out {test_succ:.3f}; {counts.sum()} training episodes, "
                      f"{top:.2f} of them on the most-played 10% of levels  ({time.time() - t1:.1f} s)")
            res[(sampler, N)] = np.array(rows)
    print(f"\nSummary: success rate (mean over seeds [min, max]); held-out = {n_test} new levels x 2 episodes")
    print(f"{'sampler':<9}{'N':>10}{'train':>24}{'held-out':>24}{'gap':>8}{'top-10% share':>15}")
    for (sampler, N), r in res.items():
        name = "unlimited" if N is None else str(N)
        tr = "     n/a (= held-out)" if np.isnan(r[:, 0]).all() else \
            f"{r[:, 0].mean():.3f} [{r[:, 0].min():.3f},{r[:, 0].max():.3f}]"
        gap = "" if np.isnan(r[:, 0]).all() else f"{(r[:, 0] - r[:, 1]).mean():8.3f}"
        gap = gap if gap else " " * 8
        print(f"{sampler:<9}{name:>10}{tr:>24}{r[:, 1].mean():>9.3f} [{r[:, 1].min():.3f},{r[:, 1].max():.3f}]{gap}"
              f"{r[:, 2].mean():>15.2f}")

    if not args.quick:
        from plot_style import setup, C
        plt = setup()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
        ax = axes[0]
        finite = [n for n in Ns if n is not None]
        xs = finite + [finite[-1] * 4]
        xl = [str(n) for n in finite] + ["unlimited"]
        tr = [res[("uniform", n)][:, 0].mean() for n in finite]
        te = [res[("uniform", n)][:, 1].mean() for n in Ns]
        ax.plot(finite, tr, marker="o", color=C[0], label="training levels")
        ax.plot(xs, te, marker="s", color=C[1], ls="--", label="200 held-out levels")
        for j, n in enumerate(Ns):
            r = res[("uniform", n)]
            ax.vlines(xs[j], r[:, 1].min(), r[:, 1].max(), color=C[1], lw=1)
            if n is not None:
                ax.vlines(xs[j], r[:, 0].min(), r[:, 0].max(), color=C[0], lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels(xl)
        ax.set_xlabel("number of training levels N")
        ax.set_ylabel("success rate")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"PPO, uniform level sampling, {steps // 1000}k steps ({len(seeds)} seeds)")
        ax.legend(loc="lower right")
        ax = axes[1]
        x = np.arange(len(plr_Ns))
        for j, (sampler, col) in enumerate([("uniform", C[1]), ("PLR", C[2])]):
            m = [res[(sampler, n)][:, 1].mean() for n in plr_Ns]
            lo = [res[(sampler, n)][:, 1].min() for n in plr_Ns]
            hi = [res[(sampler, n)][:, 1].max() for n in plr_Ns]
            ax.bar(x + (j - 0.5) * 0.36, m, width=0.36, color=col, label=sampler,
                   yerr=[np.subtract(m, lo), np.subtract(hi, m)], capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"N = {n}" for n in plr_Ns])
        ax.set_ylabel("held-out success rate")
        ax.set_ylim(0, 1.02)
        ax.set_title("Uniform vs prioritized level replay (bars: min-max over seeds)")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "procgen_lite.png"))
        plt.close(fig)
        print(f"figures written to {FIG_DIR}")
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
