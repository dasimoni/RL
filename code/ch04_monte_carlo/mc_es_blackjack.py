"""Monte Carlo control with Exploring Starts on Blackjack (Sec. 4.2-4.3; S&B Figure 5.2).

Every episode starts in a uniformly random (state, action) pair -- the "exploring start" --
and then follows the current greedy policy. After the episode, first-visit returns update
Q(s, a) by sample averaging and the policy is made greedy in every visited state.

Because blackjack.solve(mode="optimal") gives the exact q_* and pi_*, we can say precisely
which states MC ES gets wrong, how close those decisions are, and what the mistakes cost.

    python code/ch04_monte_carlo/mc_es_blackjack.py [--quick]
"""
import argparse
import os
import random
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blackjack import Blackjack, HIT, STICK, STATE_SHAPE, all_states, flat_index, solve

HERE = os.path.dirname(os.path.abspath(__file__))
STATES = all_states()


def mc_es(n_episodes, seed, checkpoints=()):
    """Monte Carlo ES (first-visit, sample averages) -- mirrors the pseudocode of Sec. 4.2.

    Returns Q (STATE_SHAPE + (2,)), visit counts N, the greedy policy and snapshots of Q at
    the requested checkpoints."""
    env, rng = Blackjack(seed), random.Random(seed)
    Q = [[0.0, 0.0] for _ in STATES]
    N = [[0, 0] for _ in STATES]
    # Initial policy (as in S&B): stick only on 20 or 21.
    pi = [HIT if s < 20 else STICK for (s, d, u) in STATES]
    snaps, checkpoints = {}, set(checkpoints)
    for ep in range(1, n_episodes + 1):
        # Exploring start: every (s, a) pair has probability 1/400 of starting the episode.
        s0, a = rng.randrange(200), rng.randrange(2)
        obs = env.reset(start=STATES[s0])
        pairs, rewards, done = [], [], False
        while not done:
            si = flat_index(obs)
            pairs.append((si, a))
            obs, r, done = env.step(a)
            rewards.append(r)
            if not done:
                a = pi[flat_index(obs)]                 # afterwards: follow pi
        G = 0.0
        for t in range(len(pairs) - 1, -1, -1):
            G = G + rewards[t]                          # gamma = 1
            si, at = pairs[t]
            if (si, at) in pairs[:t]:                   # first-visit check
                continue
            N[si][at] += 1
            Q[si][at] += (G - Q[si][at]) / N[si][at]   # incremental sample average
            # Greedy improvement in S_t; ties keep the current action.
            if Q[si][1 - pi[si]] > Q[si][pi[si]]:
                pi[si] = 1 - pi[si]
        if ep in checkpoints:
            snaps[ep] = (np.array(Q).reshape(STATE_SHAPE + (2,)), np.array(pi).reshape(STATE_SHAPE))
    Qa = np.array(Q).reshape(STATE_SHAPE + (2,))
    return Qa, np.array(N).reshape(STATE_SHAPE + (2,)), np.array(pi).reshape(STATE_SHAPE), snaps


def plot_policy_and_value(Q, pi, opt, path, n_episodes):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(11, 8.5))
    for col, (usable, label) in enumerate(((1, "usable ace"), (0, "no usable ace"))):
        ax = fig.add_subplot(2, 2, col + 1)
        P = pi[:, :, usable]
        ax.imshow(P, origin="lower", cmap="coolwarm", vmin=-0.6, vmax=1.6,
                  extent=(0.5, 10.5, 11.5, 21.5), aspect="auto")
        wrong = np.argwhere(P != opt["greedy"][:, :, usable])
        for si, di in wrong:
            ax.plot(di + 1, si + 12, "kx", ms=12, mew=2)
        # exact optimal decision boundary: for each dealer card the smallest sum we stick on
        # (drawn as a step line between the hit region and the stick region)
        for di in range(10):
            col_opt = opt["greedy"][:, di, usable]
            for si in range(10):
                if si + 1 < 10 and col_opt[si] != col_opt[si + 1]:
                    ax.plot([di + 0.5, di + 1.5], [si + 12.5] * 2, "k-", lw=2.5)
            if di + 1 < 10:
                for si in range(10):
                    if opt["greedy"][si, di, usable] != opt["greedy"][si, di + 1, usable]:
                        ax.plot([di + 1.5] * 2, [si + 11.5, si + 12.5], "k-", lw=2.5)
        ax.set_xticks(range(1, 11))
        ax.set_xticklabels(["A"] + [str(i) for i in range(2, 11)])
        ax.set_yticks(range(12, 22))
        ax.set_xlabel("dealer showing")
        ax.set_ylabel("player sum")
        ax.set_title(f"MC ES policy, {label}\n(red = HIT, blue = STICK; black line = exact "
                     f"$\\pi_*$ boundary; x = disagrees)", fontsize=9)
        ax3 = fig.add_subplot(2, 2, col + 3, projection="3d")
        dealer, player = np.meshgrid(np.arange(1, 11), np.arange(12, 22))
        ax3.plot_surface(dealer, player, Q.max(-1)[:, :, usable], cmap="viridis",
                         vmin=-1, vmax=1, edgecolor="k", linewidth=0.2)
        ax3.set_zlim(-1, 1)
        ax3.set_xlabel("dealer showing", fontsize=8)
        ax3.set_ylabel("player sum", fontsize=8)
        ax3.view_init(elev=28, azim=-125)
        ax3.set_title(f"$\\max_a Q(s,a)$ ({label})", fontsize=9)
    fig.suptitle(f"Blackjack, Monte Carlo ES after {n_episodes:,} episodes")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=None)
    args = ap.parse_args()
    n = args.episodes or (100_000 if args.quick else 5_000_000)
    seeds = [args.seed] if args.quick else [args.seed, args.seed + 1, args.seed + 2]
    print(f"MC ES on Blackjack | seeds={seeds} episodes={n:,} gamma=1 Q0=0 "
          f"initial policy=stick on 20/21")
    opt = solve(mode="optimal")
    gap = np.abs(opt["Q"][..., HIT] - opt["Q"][..., STICK])
    print(f"exact optimal J* = {opt['J']:+.5f}")
    t0 = time.time()
    grid = sorted(set(np.round(np.logspace(3, np.log10(n), 12)).astype(int)))
    results = []
    for seed in seeds:
        ts = time.time()
        Q, N, pi, snaps = mc_es(n, seed, grid)
        wrong = np.argwhere(pi != opt["greedy"])
        J = solve(pi.astype(float))["J"]             # exact value of the learned policy
        curve = [(g, int((snaps[g][1] != opt["greedy"]).sum()), solve(snaps[g][1].astype(float))["J"])
                 for g in grid]
        v_err = np.abs(Q.max(-1) - opt["Q"].max(-1))
        results.append(dict(seed=seed, Q=Q, N=N, pi=pi, wrong=wrong, J=J, curve=curve))
        print(f"\nseed {seed}: {time.time() - ts:.1f}s; greedy policy differs from pi_* in "
              f"{len(wrong)}/200 states; exact J(greedy) = {J:+.5f} (J* - J = {opt['J'] - J:.5f})")
        print(f"  max |max_a Q - v_*| = {v_err.max():.4f}, mean = {v_err.mean():.4f}; "
              f"visits per (s,a): min {N.min():,}, median {int(np.median(N)):,}")
        for si, di, u in wrong:
            idx = (si, di, u)
            print(f"  wrong at sum {si + 12}, dealer {di + 1}, usable={u}: learned "
                  f"{'HIT' if pi[idx] else 'STICK'}, exact q*(hit)-q*(stick) = "
                  f"{opt['Q'][idx + (HIT,)] - opt['Q'][idx + (STICK,)]:+.4f}, estimated "
                  f"{Q[idx + (HIT,)] - Q[idx + (STICK,)]:+.4f} (N = {N[idx + (HIT,)]:,} / {N[idx + (STICK,)]:,})")
    print("\nlearning curve (seed-mean): episodes | #states wrong | J(greedy)")
    for i, g in enumerate(grid):
        w = np.mean([r["curve"][i][1] for r in results])
        j = np.mean([r["curve"][i][2] for r in results])
        print(f"  {g:>10,} | {w:5.1f} | {j:+.5f}")
    smallest = np.sort(gap.ravel())[:6]
    print(f"\nsmallest exact action gaps |q*(s,hit)-q*(s,stick)|: {np.round(smallest, 4)}")
    print(f"total elapsed {time.time() - t0:.1f}s")

    if args.quick:
        return
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    r0 = results[0]
    plot_policy_and_value(r0["Q"], r0["pi"], opt, os.path.join(HERE, "figures", "blackjack_mc_es.png"), n)
    print("saved figures/blackjack_mc_es.png")


if __name__ == "__main__":
    main()
