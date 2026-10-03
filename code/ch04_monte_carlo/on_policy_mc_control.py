"""On-policy first-visit MC control with epsilon-soft (epsilon-greedy) policies (Sec. 5).

No exploring starts: every episode begins with a normal deal, and exploration comes from
the policy itself, which picks a uniformly random action with probability epsilon.

Section 5 proves that this procedure can at best find the optimal policy *among
epsilon-soft policies*. blackjack.solve(mode="eps") computes that best epsilon-soft policy
exactly, so we can check the claim: the epsilon-greedy policy we follow should approach
the best epsilon-soft value, and its greedy part should approach the greedy part of the
best epsilon-soft policy -- which, for epsilon = 0.1, is NOT quite pi_*.

    python code/ch04_monte_carlo/on_policy_mc_control.py [--quick]
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


def eps_greedy_phit(greedy, eps):
    """P(hit) of the epsilon-greedy policy whose greedy action table is `greedy`."""
    return np.where(greedy == HIT, 1 - eps + eps / 2, eps / 2)


def on_policy_mc_control(n_episodes, seed, eps=0.1, checkpoints=(), glie=None):
    """On-policy first-visit MC control for epsilon-soft policies (pseudocode of Sec. 5.4).

    The policy is stored implicitly: pi(a|s) = 1 - eps + eps/|A| for a = A*(s) (the greedy
    action) and eps/|A| otherwise, so we only keep the table A*.
    If `glie` = k0 is given, epsilon decays as eps_k = k0 / (k0 + k) in episode k (Exercise 12;
    eps has halved at k = k0)."""
    env, rng = Blackjack(seed), random.Random(seed)
    Q = [[0.0, 0.0] for _ in range(200)]
    N = [[0, 0] for _ in range(200)]
    # Arbitrary initial epsilon-soft policy: epsilon-greedy around "stick on 20 or 21"
    # (the same starting point as our MC ES run, for comparability).
    greedy = [HIT if s < 20 else STICK for s in range(12, 22) for d in range(10) for u in range(2)]
    snaps, checkpoints = {}, set(checkpoints)
    for ep in range(1, n_episodes + 1):
        if glie is not None:
            eps = glie / (glie + ep)
        pairs, rewards = [], []
        obs, done = env.reset(), False
        while not done:
            si = flat_index(obs)
            a = rng.randrange(2) if rng.random() < eps else greedy[si]   # epsilon-greedy
            pairs.append((si, a))
            obs, r, done = env.step(a)
            rewards.append(r)
        G = 0.0
        for t in range(len(pairs) - 1, -1, -1):
            G = G + rewards[t]                      # gamma = 1
            si, at = pairs[t]
            if (si, at) in pairs[:t]:
                continue
            N[si][at] += 1
            Q[si][at] += (G - Q[si][at]) / N[si][at]
            greedy[si] = HIT if Q[si][HIT] > Q[si][STICK] else STICK   # A* (ties -> stick)
        if ep in checkpoints:
            snaps[ep] = np.array(greedy).reshape(STATE_SHAPE)
    return (np.array(Q).reshape(STATE_SHAPE + (2,)), np.array(N).reshape(STATE_SHAPE + (2,)),
            np.array(greedy).reshape(STATE_SHAPE), snaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--glie", type=float, default=None,
                    help="decay epsilon as k0/(k0+k) with k0 = GLIE (Exercise 12); reports use the final epsilon")
    args = ap.parse_args()
    n = 100_000 if args.quick else 5_000_000
    seeds = [args.seed] if args.quick else [args.seed, args.seed + 1, args.seed + 2]
    eps = args.eps if args.glie is None else args.glie / (args.glie + n)
    sched = f"eps={eps}" if args.glie is None else f"GLIE eps_k={args.glie:g}/({args.glie:g}+k), final eps={eps:.5f}"
    print(f"On-policy first-visit MC control | seeds={seeds} episodes={n:,} {sched} gamma=1 Q0=0")
    opt, best_soft = solve(mode="optimal"), solve(mode="eps", eps=eps)
    J_opt_eps = solve(eps_greedy_phit(opt["greedy"], eps))["J"]
    print(f"exact references: J* = {opt['J']:+.5f} | best eps-soft J = {best_soft['J']:+.5f} | "
          f"eps-greedy version of pi_* J = {J_opt_eps:+.5f}")
    diff = np.argwhere(best_soft["greedy"] != opt["greedy"])
    print("greedy part of the best eps-soft policy differs from pi_* at: " +
          ", ".join(f"(sum {s + 12}, dealer {d + 1}, usable {u})" for s, d, u in diff))
    t0 = time.time()
    grid = sorted(set(np.round(np.logspace(3, np.log10(n), 12)).astype(int)))
    curves = []
    for seed in seeds:
        ts = time.time()
        Q, N, greedy, snaps = on_policy_mc_control(n, seed, eps, grid, glie=args.glie)
        J_soft = solve(eps_greedy_phit(greedy, eps))["J"]
        J_greedy = solve(greedy.astype(float))["J"]
        print(f"\nseed {seed} ({time.time() - ts:.1f}s): J(eps-greedy policy followed) = {J_soft:+.5f}"
              f" | J(its greedy part) = {J_greedy:+.5f}")
        print(f"  greedy part vs pi_*: {(greedy != opt['greedy']).sum()} states differ; "
              f"vs greedy part of best eps-soft policy: {(greedy != best_soft['greedy']).sum()} differ")
        for s, d, u in np.argwhere(greedy != opt["greedy"]):
            idx = (s, d, u)
            print(f"    sum {s + 12:2d} dealer {d + 1:2d} usable {u}: learned {'HIT' if greedy[idx] else 'STICK'}"
                  f" | q*(hit)-q*(stick) = {opt['Q'][idx + (HIT,)] - opt['Q'][idx + (STICK,)]:+.4f}"
                  f" | eps-soft q(hit)-q(stick) = {best_soft['Q'][idx + (HIT,)] - best_soft['Q'][idx + (STICK,)]:+.4f}"
                  f" | N = {N[idx + (HIT,)]:,}/{N[idx + (STICK,)]:,}")
        print(f"  visits per (s,a): min {N.min():,}, median {int(np.median(N)):,}")
        curves.append([(solve(eps_greedy_phit(snaps[g], eps))["J"], solve(snaps[g].astype(float))["J"])
                       for g in grid])
    curves = np.array(curves)                     # seeds x grid x 2
    print("\nlearning curve (seed-mean): episodes | J(eps-greedy) | J(greedy part)")
    for i, g in enumerate(grid):
        print(f"  {g:>10,} | {curves[:, i, 0].mean():+.5f} | {curves[:, i, 1].mean():+.5f}")
    print(f"total elapsed {time.time() - t0:.1f}s")

    if args.quick or args.glie is not None:     # the GLIE variant only prints its results
        return
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, (label, style) in enumerate((("eps-greedy policy actually followed", "C0o-"),
                                        ("its greedy part", "C1s-"))):
        ax.semilogx(grid, curves[:, :, k].mean(0), style, ms=4, label=label)
        ax.fill_between(grid, curves[:, :, k].min(0), curves[:, :, k].max(0), color=style[:2], alpha=0.2)
    ax.axhline(opt["J"], color="k", ls="--", label=r"$J(\pi_*)$ (optimal)")
    ax.axhline(best_soft["J"], color="C0", ls=":", label=f"best {eps}-soft policy")
    ax.set_xlabel("episodes")
    ax.set_ylabel("exact expected return per game")
    ax.set_title(f"On-policy MC control, epsilon = {eps} ({len(seeds)} seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "blackjack_on_policy_control.png"), dpi=110)
    print("saved figures/blackjack_on_policy_control.png")


if __name__ == "__main__":
    main()
