"""Off-policy Monte Carlo control with weighted importance sampling (Sec. 8).

The target policy pi is greedy w.r.t. Q; the behavior policy b that generates the episodes
is soft (here: uniformly random, or epsilon-greedy w.r.t. the current Q). Returns are
weighted incrementally (Sec. 7): C(s,a) accumulates the weights W, and
Q(s,a) <- Q(s,a) + W / C(s,a) * (G - Q(s,a)). Because pi is deterministic, the ratio
pi(A_t|S_t)/b(A_t|S_t) is 0 as soon as b takes a non-greedy action, so each episode only
teaches us about its *tail* after the last non-greedy action. We measure how much of each
episode is actually used.

    python code/ch04_monte_carlo/off_policy_mc_control.py [--quick]
"""
import argparse
import os
import random
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blackjack import Blackjack, HIT, STICK, STATE_SHAPE, flat_index, solve

HERE = os.path.dirname(os.path.abspath(__file__))


def off_policy_mc_control(n_episodes, seed, behavior="random", eps_b=0.3, checkpoints=()):
    """Off-policy MC control (pseudocode of Sec. 8). behavior: 'random' or 'eps-greedy'."""
    env, rng = Blackjack(seed), random.Random(seed)
    Q = [[0.0, 0.0] for _ in range(200)]
    C = [[0.0, 0.0] for _ in range(200)]
    pi = [STICK] * 200                        # greedy w.r.t. Q = 0 (ties -> stick)
    snaps, checkpoints = {}, set(checkpoints)
    steps_total = steps_used = full_episodes = 0
    for ep in range(1, n_episodes + 1):
        # Generate an episode with b; remember b(A_t|S_t) for the importance weights.
        traj, done = [], False
        obs = env.reset()
        while not done:
            si = flat_index(obs)
            if behavior == "random" or rng.random() < eps_b:
                a = rng.randrange(2)
            else:
                a = pi[si]
            if behavior == "random":
                b_prob = 0.5
            else:                              # epsilon-greedy w.r.t. pi (fixed during the episode)
                b_prob = 1 - eps_b + eps_b / 2 if a == pi[si] else eps_b / 2
            obs, r, done = env.step(a)
            traj.append((si, a, r, b_prob))
        G, W = 0.0, 1.0
        used = 0
        for t in range(len(traj) - 1, -1, -1):
            si, a, r, b_prob = traj[t]
            G = G + r                          # gamma = 1
            C[si][a] += W
            Q[si][a] += (W / C[si][a]) * (G - Q[si][a])
            pi[si] = HIT if Q[si][HIT] > Q[si][STICK] else STICK
            used += 1
            if a != pi[si]:                    # pi(A_t|S_t) = 0: earlier steps get weight 0
                break
            W = W / b_prob                     # pi(A_t|S_t) = 1 for the greedy action
        steps_total += len(traj)
        steps_used += used
        full_episodes += used == len(traj)    # the update reached S_0 with nonzero weight
        if ep in checkpoints:
            snaps[ep] = np.array(pi).reshape(STATE_SHAPE)
    stats = dict(frac_steps_used=steps_used / steps_total, frac_full=full_episodes / n_episodes,
                 mean_len=steps_total / n_episodes)
    return (np.array(Q).reshape(STATE_SHAPE + (2,)), np.array(C).reshape(STATE_SHAPE + (2,)),
            np.array(pi).reshape(STATE_SHAPE), snaps, stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n = 100_000 if args.quick else 5_000_000
    seeds = [args.seed] if args.quick else [args.seed, args.seed + 1, args.seed + 2]
    behaviors = [("random", None), ("eps-greedy", 0.3)]
    print(f"Off-policy MC control (weighted IS) | seeds={seeds} episodes={n:,} gamma=1 "
          f"behaviors={behaviors}")
    opt = solve(mode="optimal")
    print(f"exact J* = {opt['J']:+.5f}")
    grid = sorted(set(np.round(np.logspace(3, np.log10(n), 12)).astype(int)))
    t0 = time.time()
    curves = {}
    for behavior, eps_b in behaviors:
        curves[behavior] = []
        for seed in seeds:
            ts = time.time()
            Q, C, pi, snaps, st = off_policy_mc_control(n, seed, behavior, eps_b or 0.0, grid)
            J = solve(pi.astype(float))["J"]
            wrong = np.argwhere(pi != opt["greedy"])
            print(f"\n[{behavior}] seed {seed} ({time.time() - ts:.1f}s): J(pi) = {J:+.5f} "
                  f"(J* - J = {opt['J'] - J:.5f}); {len(wrong)} states differ from pi_*")
            print(f"  mean episode length {st['mean_len']:.2f} decisions; fraction of steps that "
                  f"updated Q {st['frac_steps_used']:.3f}; episodes used to the start "
                  f"{st['frac_full']:.3f}")
            for s, d, u in wrong:
                idx = (s, d, u)
                print(f"    sum {s + 12:2d} dealer {d + 1:2d} usable {u}: learned "
                      f"{'HIT' if pi[idx] else 'STICK'} | q*(hit)-q*(stick) = "
                      f"{opt['Q'][idx + (HIT,)] - opt['Q'][idx + (STICK,)]:+.4f} | "
                      f"C = {C[idx + (HIT,)]:.3g}/{C[idx + (STICK,)]:.3g}")
            curves[behavior].append([(solve(snaps[g].astype(float))["J"], int((snaps[g] != opt["greedy"]).sum()))
                                     for g in grid])
    print("\nlearning curve (seed-mean): episodes | " +
          " | ".join(f"{b}: J(pi), #wrong" for b, _ in behaviors))
    for i, g in enumerate(grid):
        row = " | ".join(f"{np.mean([c[i][0] for c in curves[b]]):+.5f}, {np.mean([c[i][1] for c in curves[b]]):5.1f}"
                         for b, _ in behaviors)
        print(f"  {g:>10,} | {row}")
    print(f"total elapsed {time.time() - t0:.1f}s")

    if args.quick:
        return
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, (b, eps_b) in enumerate(behaviors):
        c = np.array([[x[0] for x in run] for run in curves[b]])
        label = "b = uniform random" if b == "random" else f"b = {eps_b}-greedy w.r.t. Q"
        ax.semilogx(grid, c.mean(0), "o-" if k == 0 else "s-", ms=4, label=label)
        ax.fill_between(grid, c.min(0), c.max(0), alpha=0.2)
    ax.axhline(opt["J"], color="k", ls="--", label=r"$J(\pi_*)$")
    ax.set_xlabel("episodes")
    ax.set_ylabel(r"exact expected return of greedy target $\pi$")
    ax.set_title(f"Off-policy MC control on Blackjack ({len(seeds)} seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "blackjack_off_policy_control.png"), dpi=110)
    print("saved figures/blackjack_off_policy_control.png")


if __name__ == "__main__":
    main()
