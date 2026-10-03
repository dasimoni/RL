"""Monte Carlo check: v_pi(s) really is the expected return, so averaging sampled returns recovers it.

Chapter 01, Section 11 (a preview of Chapter 04).  Gridworld of Example 3.5,
equiprobable random policy, gamma = 0.9.

1. For every state, roll out N independent trajectories of H steps and average
   the truncated discounted returns  G = sum_{k<H} gamma^k R_{k+1}.  Truncation
   bias is at most gamma^H R_max / (1 - gamma) (Section 4).
2. Watch the running average at A and B converge to the exact values from the
   linear solve, with the 1/sqrt(N) shrinking of the 95% confidence band.
3. Look at the *distribution* of returns from A: v_pi(A) is only its mean.
4. The "discount = probability of surviving another step" reading of gamma (Section 4.3):
   averaging UNdiscounted returns of episodes that end with probability 1 - gamma
   after every step gives the same expectation.

Usage:  python code/ch01_the_rl_problem/monte_carlo_returns.py [--quick]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from gridworld import GridWorld
from mdp import evaluate_policy_exact, uniform_policy


def discounted_returns(nxt, rew, starts, horizon, gamma, rng):
    """Sample one truncated discounted return per entry of ``starts`` (all in parallel).

    Mirrors the pseudocode of Section 11: S_0 = s; for t = 0..H-1: A_t ~ pi(.|S_t),
    observe R_{t+1}, S_{t+1}; accumulate G += gamma^t R_{t+1}.
    """
    s = starts.copy()
    G = np.zeros(len(s))
    disc = 1.0
    for _ in range(horizon):
        a = rng.integers(4, size=len(s))      # equiprobable random policy
        G += disc * rew[s, a]                 # discount^t * R_{t+1}
        s = nxt[s, a]
        disc *= gamma
    return G


def geometric_lifetime_returns(nxt, rew, starts, gamma, rng, cap=600):
    """Undiscounted returns of episodes that continue with probability gamma after each reward.

    E[sum_{k=0}^{K-1} R_{k+1}] with Pr{K > k} = gamma^k equals sum_k gamma^k E[R_{k+1}].
    ``cap`` only guards the loop: Pr{K > 600} = 0.9^600 ~ 3e-28.
    """
    s = starts.copy()
    G = np.zeros(len(s))
    alive = np.ones(len(s), dtype=bool)
    lengths = np.zeros(len(s), dtype=int)
    for _ in range(cap):
        if not alive.any():
            break
        a = rng.integers(4, size=len(s))
        G += np.where(alive, rew[s, a], 0.0)
        lengths += alive
        s = nxt[s, a]
        alive &= rng.random(len(s)) < gamma  # the episode survives this step with prob. gamma
    return G, lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smoke test: fewer samples, no figures")
    args = parser.parse_args()
    t0 = time.perf_counter()
    seed, gamma, horizon = 0, 0.9, 100
    n_per_state = 500 if args.quick else 5_000
    n_long = 10_000 if args.quick else 200_000
    print(f"monte_carlo_returns.py | seed={seed} gamma={gamma} horizon H={horizon} "
          f"N per state={n_per_state} N at A,B={n_long} quick={args.quick}")
    rng = np.random.default_rng(seed)

    world = GridWorld()
    mdp = world.to_mdp()
    nxt, rew = world.tables()
    v_exact = evaluate_policy_exact(mdp, uniform_policy(mdp.n_states, 4), gamma)
    A, B = world.index(*world.a_pos), world.index(*world.b_pos)
    r_max = np.abs(rew).max()
    print(f"truncation bias bound gamma^H R_max/(1-gamma) = {gamma ** horizon * r_max / (1 - gamma):.4f}")

    # ---- 1. every state
    starts = np.repeat(np.arange(mdp.n_states), n_per_state)
    G = discounted_returns(nxt, rew, starts, horizon, gamma, rng).reshape(mdp.n_states, n_per_state)
    mc_mean = G.mean(axis=1)
    mc_se = G.std(axis=1, ddof=1) / np.sqrt(n_per_state)
    inside = np.abs(mc_mean - v_exact) <= 1.96 * mc_se
    print(f"\n[1] Monte Carlo estimate of v_pi from {n_per_state} returns per state:")
    print(np.round(mc_mean.reshape(5, 5), 2))
    print(f"    max |MC - exact| = {np.abs(mc_mean - v_exact).max():.3f}; "
          f"typical standard error = {np.median(mc_se):.3f}; "
          f"exact value inside the 95% CI for {inside.sum()}/25 states")

    # ---- 2. convergence at A and B
    print(f"\n[2] running estimate with N returns (exact v(A) = {v_exact[A]:.4f}, v(B) = {v_exact[B]:.4f}):")
    long = {}
    for name, s0 in [("A", A), ("B", B)]:
        g = discounted_returns(nxt, rew, np.full(n_long, s0), horizon, gamma, rng)
        long[name] = g
        for n in [100, 1_000, 10_000, 100_000, 200_000]:
            if n <= n_long:
                m, se = g[:n].mean(), g[:n].std(ddof=1) / np.sqrt(n)
                print(f"    {name}: N={n:>7}: estimate {m:7.4f} +- {1.96 * se:.4f} (95% CI half-width), "
                      f"error {m - v_exact[s0]:+.4f}")

    # ---- 3. the distribution behind the mean
    gA = long["A"]
    q = np.percentile(gA, [5, 25, 50, 75, 95])
    print(f"\n[3] distribution of the discounted return from A ({n_long} samples): "
          f"mean {gA.mean():.3f}, std {gA.std():.3f}, min {gA.min():.2f}, max {gA.max():.2f}")
    print(f"    percentiles 5/25/50/75/95: {np.round(q, 2).tolist()}")

    # ---- 4. discounting as a survival probability
    gl, lengths = geometric_lifetime_returns(nxt, rew, np.full(n_long, A), gamma, rng)
    se_l = gl.std(ddof=1) / np.sqrt(n_long)
    print(f"\n[4] undiscounted returns with termination probability 1-gamma={1 - gamma:.1f} per step, from A:")
    print(f"    mean {gl.mean():.4f} +- {1.96 * se_l:.4f} (95% CI half-width)  (exact discounted value {v_exact[A]:.4f}); "
          f"mean episode length {lengths.mean():.2f} (1/(1-gamma) = {1 / (1 - gamma):.0f}); "
          f"std of returns {gl.std():.3f} vs {gA.std():.3f} for discounted returns")

    if not args.quick:
        import matplotlib.pyplot as plt
        from plotting import AQUA, BLUE, INK, INK2, ORANGE, setup_style
        setup_style()
        figdir = Path(__file__).parent / "figures"
        figdir.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw={"width_ratios": [1.25, 1, 1]})
        ax = axes[0]
        for name, s0, col in [("A", A, BLUE), ("B", B, ORANGE)]:
            g = long[name]
            n = np.arange(1, len(g) + 1)
            run = np.cumsum(g) / n
            se = g.std() / np.sqrt(n)
            ax.fill_between(n, run - 1.96 * se, run + 1.96 * se, color=col, alpha=0.18, lw=0)
            ax.plot(n, run, color=col, lw=1.6, label=f"running mean, state {name}")
            ax.axhline(v_exact[s0], color=col, ls="--", lw=1.2)
            ax.text(len(g) * 1.08, v_exact[s0], f"exact\n{v_exact[s0]:.3f}",
                    color=INK2, fontsize=8.5, va="center")
        ax.set_xscale("log")
        ax.set_xlim(10, len(gA) * 1.0)
        ax.set_ylim(min(v_exact[A], v_exact[B]) - 3, max(v_exact[A], v_exact[B]) + 3)
        ax.set_xlabel("number of sampled returns N")
        ax.set_ylabel("estimate of $v_\\pi(s)$")
        ax.set_title("Sample averages converge to $v_\\pi$ (band: 95% CI)")
        ax.legend(loc="lower right")

        ax = axes[1]
        ax.hist(gA, bins=120, color=BLUE, alpha=0.85, density=True)
        ax.axvline(v_exact[A], color=INK, lw=1.5, ls="--")
        ax.text(v_exact[A] + 0.4, ax.get_ylim()[1] * 0.9, f"mean = $v_\\pi(A)$ = {v_exact[A]:.2f}",
                color=INK, fontsize=9)
        ax.set_xlim(np.percentile(gA, 0.05), np.percentile(gA, 99.95))
        ax.set_xlabel("discounted return $G_0$ from A ($H$ = 100)")
        ax.set_ylabel("density")
        ax.set_title(f"Returns from A vary a lot (std {gA.std():.2f})")

        ax = axes[2]
        for g, col, lab in [(gA, BLUE, "discounted, $\\gamma$ = 0.9"),
                            (gl, AQUA, "survival: undiscounted,\nstops w.p. 0.1 per step")]:
            xs = np.sort(g)
            ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=col, lw=2, label=lab, drawstyle="steps-post")
        ax.axvline(v_exact[A], color=INK, lw=1.2, ls="--")
        ax.set_xlim(np.percentile(gl, 0.5), np.percentile(gl, 99.5))
        ax.set_xlabel("return from A")
        ax.set_ylabel("fraction of samples $\\leq$ x")
        ax.set_title("Same mean, different spread (CDFs)")
        # centre right is empty (both CDFs are near 1 there); the legend no longer hides the curves'
        # upper parts, such as the survival return's jump at 10
        ax.legend(loc="center left", bbox_to_anchor=(0.57, 0.42), fontsize=8.5, frameon=True,
                  facecolor="white", edgecolor="none", framealpha=0.85)
        fig.tight_layout()
        fig.savefig(figdir / "monte_carlo_returns.png")
        plt.close(fig)
        print(f"\nsaved figure to {figdir / 'monte_carlo_returns.png'}")

    print(f"\nSUMMARY: MC v(A) = {gA.mean():.3f} vs exact {v_exact[A]:.3f}; "
          f"MC v(B) = {long['B'].mean():.3f} vs exact {v_exact[B]:.3f}; "
          f"geometric-lifetime estimate of v(A) = {gl.mean():.3f}")
    print(f"done in {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
