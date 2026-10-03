"""Episodic semi-gradient SARSA with tile coding on Gymnasium's MountainCar-v0
(Chapter 08, Section 11; cf. Sutton & Barto Example 10.1, Figs 10.1-10.2).

* Features: 8 tilings of 8x8 tiles over (position, velocity), asymmetric offsets
  (tiles.py), one weight vector per action: q_hat(s,a,w) = sum of the 8 active
  weights of action a.
* Exploration: epsilon = 0 (greedy!). Rewards are -1 per step and w starts at 0,
  so q_hat = 0 is wildly *optimistic*; every visited action looks worse
  afterwards, which drives systematic exploration.
* Gymnasium's MountainCar-v0 ends episodes in two ways (Section 11.2):
    terminated -> the car reached the goal: target is R alone;
    truncated  -> the 200-step time limit fired: the next state still has
                  value, so we BOOTSTRAP: target = R + gamma q_hat(S', A').
  Experiment (c) shows what happens if you treat truncation as termination.

Experiments
  (a) steps per episode for alpha in {0.1, 0.2, 0.5}/8, with the time limit raised to
      10,000 steps so that early episodes are not cut off (as in Sutton & Barto);
  (b) cost-to-go surfaces -max_a q_hat(s,a) at several points of one run (same limit);
  (c) correct vs incorrect handling of truncation under the default 200-step limit;
  (d) optional (--hashing-demo): hash the 648 tiles per action into smaller tables.

Run:  python code/ch08_function_approximation/mountain_car_sarsa.py [--quick] [--hashing-demo]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread: the machine is shared
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import time

import gymnasium as gym
import numpy as np

from tiles import TileCoder

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
LOWS, HIGHS = np.array([-1.2, -0.07]), np.array([0.6, 0.07])


class SarsaAgent:
    """Linear q_hat over tile-coded features, one weight row per action."""

    def __init__(self, n_actions=3, n_tilings=8, tiles=8, alpha=0.5, epsilon=0.0,
                 gamma=1.0, seed=0, hash_size=None, hashing="mix"):
        self.coder = TileCoder(LOWS, HIGHS, n_tilings=n_tilings, tiles_per_dim=tiles,
                               hash_size=hash_size, hashing=hashing)
        self.w = np.zeros((n_actions, self.coder.n_features))
        self.alpha = alpha / n_tilings        # alpha per active tile (Section 7: alpha = 1/(tau E[x^T x]))
        self.epsilon, self.gamma = epsilon, gamma
        self.rng = np.random.default_rng(seed)

    def features(self, s):
        return self.coder.active(s)

    def q(self, idx):
        return self.w[:, idx].sum(axis=1)      # q_hat(s, a, w) for all a

    def act(self, idx):
        """epsilon-greedy w.r.t. q_hat(s,.,w); returns (action, q-values of s)."""
        q = self.q(idx)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(q.size)), q
        best = np.flatnonzero(q == q.max())    # random tie-breaking matters with q = 0 init
        a = best[0] if best.size == 1 else best[int(self.rng.random() * best.size)]
        return int(a), q

    def update(self, idx, a, target):
        # grad q_hat(S,A,w) is 1 on the active tiles of action a, 0 elsewhere
        delta = target - self.w[a, idx].sum()
        self.w[a, idx] += self.alpha * delta


def run_episode(env, agent, seed=None, truncation_is_terminal=False, on_step=None):
    """One episode of episodic semi-gradient SARSA (Section 11 pseudocode).
    Returns the number of steps taken. `on_step(t)` is an optional hook (used to
    snapshot the value function in the middle of the first episode)."""
    obs, _ = env.reset(seed=seed)
    idx = agent.features(obs)
    a, _ = agent.act(idx)
    steps = 0
    while True:
        obs2, r, terminated, truncated, _ = env.step(a)
        steps += 1
        if terminated or (truncated and truncation_is_terminal):
            agent.update(idx, a, r)                         # no bootstrap
            return steps
        idx2 = agent.features(obs2)
        a2, q2 = agent.act(idx2)
        agent.update(idx, a, r + agent.gamma * q2[a2])      # bootstrap (also on truncation)
        if on_step is not None:
            on_step(steps)
        if truncated:
            return steps
        idx, a = idx2, a2


def learning_curves(alphas, n_runs, n_episodes, max_steps, seed, truncation_is_terminal=False,
                    hash_size=None, hashing="mix"):
    """Steps per episode, shape (len(alphas), n_runs, n_episodes); also returns the final agents."""
    curves = np.zeros((len(alphas), n_runs, n_episodes))
    agents = []
    for i, alpha in enumerate(alphas):
        for run in range(n_runs):
            env = gym.make("MountainCar-v0", max_episode_steps=max_steps)
            agent = SarsaAgent(alpha=alpha, seed=seed + run, hash_size=hash_size, hashing=hashing)
            for ep in range(n_episodes):
                s = seed + 10_000 * run if ep == 0 else None      # seed the env once per run
                curves[i, run, ep] = run_episode(env, agent, seed=s,
                                                 truncation_is_terminal=truncation_is_terminal)
            env.close()
            agents.append(agent)
    return curves, agents


def greedy_steps_from(agent, position, max_steps=2000):
    """Actual number of steps the greedy policy needs from (position, 0), no time limit."""
    env = gym.make("MountainCar-v0").unwrapped
    env.reset(seed=0)
    env.state = np.array([position, 0.0])
    s, n = env.state.copy(), 0
    while n < max_steps:
        s, _, terminated, _, _ = env.step(int(agent.q(agent.features(s)).argmax()))
        n += 1
        if terminated:
            break
    env.close()
    return n


def cost_to_go(agent, n=60):
    pos = np.linspace(LOWS[0], 0.5, n)
    vel = np.linspace(LOWS[1], HIGHS[1], n)
    Z = np.zeros((n, n))
    for i, v in enumerate(vel):
        for j, p in enumerate(pos):
            Z[i, j] = -agent.q(agent.features((p, v))).max()
    return pos, vel, Z


def hashing_demo(args):
    """(d) Fold the 8 x 81 = 648 grid tiles (per action) into tables of H entries.
    "mix": a stateless pseudo-random hash -- collisions make unrelated regions share weights
    (a few collide even when H is several times larger than 648);
    "iht": an index hash table (tiles.py) -- fresh indices on first visit, so with H >= 648
    there are no collisions at all and learning is identical to no hashing."""
    n_runs, n_episodes = (1, 10) if args.quick else (3, 150)
    sizes = [(None, "mix"), (1024, "iht"), (4096, "mix"), (512, "mix"), (128, "mix"), (32, "mix")]
    if args.quick:
        sizes = [(None, "mix"), (1024, "iht"), (128, "mix")]
    max_steps = 2000          # bounds the runtime of badly-hashed agents (truncation is bootstrapped)
    print(f"seed={args.seed} quick={args.quick}; (d) hashing demo, alpha=0.5/8, {n_runs} runs x "
          f"{n_episodes} episodes, max_episode_steps={max_steps}")
    from tiles import _mix32
    for H, scheme in sizes:
        t0 = time.time()
        c, ags = learning_curves([0.5], n_runs, n_episodes, max_steps, args.seed, hash_size=H,
                                 hashing=scheme)
        coder = ags[0].coder
        grid = np.arange(coder.n_grid_features)
        if H is None:
            used, label = coder.n_grid_features, "no hashing (648)"
        elif scheme == "iht":
            used, label = len(coder._iht), f"IHT {H}"       # tiles actually visited, one slot each
        else:
            used, label = len(np.unique((_mix32(grid) % np.uint64(H)))), f"mix hash {H}"
        m = c[0].mean(0)
        print(f"  table {label:>18s}: distinct slots used by the 648 tiles ="
              f" {used:4d}; steps per episode: first 10 = {m[:10].mean():7.1f}, last 50 = "
              f"{m[-50:].mean():6.1f}, episodes truncated = {(c[0] >= max_steps).mean():.2f}  [{time.time() - t0:.1f}s]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figures")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hashing-demo", action="store_true",
                    help="only run experiment (d): tile coding with small hash tables")
    args = ap.parse_args()
    if args.hashing_demo:
        hashing_demo(args)
        return
    alphas = [0.1, 0.2, 0.5]
    n_runs, n_episodes = (1, 15) if args.quick else (10, 300)
    abl_runs, abl_episodes = (1, 15) if args.quick else (10, 300)
    snap_episodes = [12, 104, 1000] if not args.quick else [3]
    print(f"seed={args.seed} quick={args.quick}; 8 tilings of 8x8 tiles, epsilon=0, gamma=1, w0=0")

    # ---------------- (a) step-size comparison, time limit 10,000 ----------------
    print(f"(a) alpha in {alphas} (per tile: alpha/8), {n_runs} runs x {n_episodes} episodes,"
          f" max_episode_steps=10000")
    t0 = time.time()
    curves, _ = learning_curves(alphas, n_runs, n_episodes, 10_000, args.seed)
    t_a = time.time() - t0
    for i, a in enumerate(alphas):
        m = curves[i].mean(0)
        print(f"    alpha={a}/8: ep 1 = {m[0]:7.1f}, eps 1-10 = {m[:10].mean():6.1f}, "
              f"eps 91-100 = {m[90:100].mean() if m.size >= 100 else float('nan'):6.1f}, last 50 = {m[-50:].mean():6.1f}, "
              f"total steps/run = {curves[i].sum(1).mean():8.0f}")
    print(f"    ({t_a:.1f}s)")

    # ---------------- (b) cost-to-go snapshots from one run ----------------------
    t0 = time.time()
    env = gym.make("MountainCar-v0", max_episode_steps=10_000)
    agent = SarsaAgent(alpha=0.5, seed=args.seed)
    snaps = {}

    def snap_mid(t):
        if t == 428:
            snaps["step 428 (episode 1)"] = cost_to_go(agent)

    first_len = run_episode(env, agent, seed=args.seed, on_step=snap_mid)
    lens = [first_len]
    for ep in range(2, max(snap_episodes) + 1):
        lens.append(run_episode(env, agent))
        if ep in snap_episodes:
            snaps[f"episode {ep}"] = cost_to_go(agent)
    env.close()
    pos_g, vel_g, Z = list(snaps.values())[-1]
    iv, ip = np.unravel_index(np.argmax(Z), Z.shape)
    unvisited = float(np.mean(Z == 0.0))
    print(f"(b) single run alpha=0.5/8: episode 1 took {first_len} steps; mean of last 100 episodes ="
          f" {np.mean(lens[-100:]):.1f}; max cost-to-go after episode {max(snap_episodes)} ="
          f" {Z.max():.1f} at position {pos_g[ip]:.2f}, velocity {vel_g[iv]:.4f};"
          f" fraction of grid still at the initial value 0: {unvisited:.2f} ({time.time() - t0:.1f}s)")

    # ---------------- (c) truncation handling, default 200-step limit ------------
    t0 = time.time()
    abl = {}
    for label, wrong in [("bootstrap on truncation (correct)", False),
                         ("truncation treated as terminal (wrong)", True)]:
        c, ags = learning_curves([0.5], abl_runs, abl_episodes, 200, args.seed + 7,
                                 truncation_is_terminal=wrong)
        abl[label] = c = c[0]
        starts = (-0.6, -0.5, -0.4)
        pred = np.mean([-ag.q(ag.features((p, 0.0))).max() for ag in ags for p in starts])
        actual = np.mean([greedy_steps_from(ag, p) for ag in ags for p in starts])
        print(f"(c) {label:40s}: steps (last 100 eps) = {c[:, -100:].mean():6.1f}, "
              f"episodes reaching goal = {(c < 200).mean():.2f}, predicted cost-to-go from "
              f"x0 in {starts} = {pred:6.1f} vs actual greedy steps = {actual:6.1f}")
        trunc = c >= 200                                      # truncated episodes (time limit fired)
        last_tr = [int(np.flatnonzero(row).max()) + 1 if row.any() else 0 for row in trunc]
        print(f"    truncated episodes per run = {trunc.sum(1).astype(int).tolist()}; "
              f"last truncated episode per run = {last_tr}; "
              f"truncations in episodes 1-100 / 101-{abl_episodes}: {int(trunc[:, :100].sum())} / "
              f"{int(trunc[:, 100:].sum())}")
    print(f"    ({time.time() - t0:.1f}s)")

    if args.quick:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    ep = np.arange(1, n_episodes + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for i, a in enumerate(alphas):
        axes[0].plot(ep, curves[i].mean(0), label=rf"$\alpha$ = {a}/8")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel(f"steps per episode (log scale, mean of {n_runs} runs)")
    axes[0].set_title("Semi-gradient SARSA, 8 tilings (time limit 10,000)")
    axes[0].legend()
    k = 10
    for label, c in abl.items():
        m = c.mean(0)
        sm = np.convolve(m, np.ones(k) / k, mode="valid")
        axes[1].plot(np.arange(k, abl_episodes + 1), sm, label=label)
    axes[1].axhline(200, color="gray", ls=":", lw=1)
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel(f"steps per episode ({k}-episode moving avg.)")
    axes[1].set_title(rf"MountainCar-v0 default 200-step limit, $\alpha$=0.5/8 ({abl_runs} runs)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mountain_car_learning_curves.png"), dpi=110)

    fig = plt.figure(figsize=(15, 4.8))
    for i, (title, (pos, vel, Z)) in enumerate(snaps.items()):
        ax = fig.add_subplot(1, len(snaps), i + 1, projection="3d")
        P, V = np.meshgrid(pos, vel)
        ax.plot_surface(P, V, Z, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel("position", fontsize=8)
        ax.set_ylabel("velocity", fontsize=8)
        ax.set_zlabel(r"$-\max_a \hat q$", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_title(f"{title}\nmax = {Z.max():.0f}", fontsize=9)
        ax.view_init(elev=35, azim=-125)
    fig.suptitle(r"Cost-to-go $-\max_a \hat q(s,a,\mathbf{w})$ learned by semi-gradient SARSA, one run, $\alpha$=0.5/8")
    fig.tight_layout()
    fig.subplots_adjust(top=0.80, left=0.01, right=0.97, wspace=0.12)
    fig.savefig(os.path.join(FIG_DIR, "mountain_car_cost_to_go.png"), dpi=110)

    # greedy policy map after the last snapshot
    pos = np.linspace(LOWS[0], 0.5, 120)
    vel = np.linspace(LOWS[1], HIGHS[1], 120)
    A = np.array([[agent.q(agent.features((p, v))).argmax() for p in pos] for v in vel])
    fig, ax = plt.subplots(figsize=(6, 4.3))
    im = ax.imshow(A, origin="lower", aspect="auto", extent=[pos[0], pos[-1], vel[0], vel[-1]],
                   cmap=plt.matplotlib.colors.ListedColormap(["tab:red", "lightgray", "tab:blue"]))
    cb = fig.colorbar(im, ticks=[0.33, 1, 1.67])
    cb.ax.set_yticklabels(["push left", "no push", "push right"])
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.set_title(f"Greedy policy after {max(snap_episodes)} episodes")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mountain_car_policy.png"), dpi=110)
    print(f"figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
