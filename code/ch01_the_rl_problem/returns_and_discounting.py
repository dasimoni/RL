"""Returns and discounting: the numbers behind Chapter 01, Section 4.

1. The worked example of Section 4.2: returns of a short episode computed backwards
   with G_t = R_{t+1} + gamma G_{t+1} (G_T = 0) and checked against the direct sums.
2. The absorbing-state view (Section 4.4): padding the episode with an absorbing
   state that pays 0 forever and using the infinite-horizon formula gives the same returns.
3. Constant rewards: sum_k gamma^k c = c / (1 - gamma).
4. Effective horizons: how much of the total weight sum_k gamma^k sits in the first
   1/(1-gamma) steps, and how long until gamma^k falls below 1%.
5. Discount factor <-> interest rate: gamma = 1 / (1 + rate).

Usage:  python code/ch01_the_rl_problem/returns_and_discounting.py [--quick]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def returns_backward(rewards, gamma: float) -> np.ndarray:
    """G_t for t = 0..T-1 given rewards [R_1, ..., R_T], via G_t = R_{t+1} + gamma G_{t+1}.

    rewards[t] holds R_{t+1}: the reward that *follows* the action at time t.
    One backward pass costs O(T) instead of the O(T^2) of summing each G_t directly.
    """
    G = np.zeros(len(rewards) + 1)          # G[T] = 0: nothing more to collect after the end
    for t in reversed(range(len(rewards))):
        G[t] = rewards[t] + gamma * G[t + 1]
    return G


def returns_direct(rewards, gamma: float) -> np.ndarray:
    T = len(rewards)
    return np.array([sum(gamma ** (k - t - 1) * rewards[k - 1] for k in range(t + 1, T + 1))
                     for t in range(T + 1)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smoke test: no figures")
    args = parser.parse_args()
    t0 = time.perf_counter()
    seed = 0
    print(f"returns_and_discounting.py | seed={seed} quick={args.quick}")

    # ---- 1. worked example
    gamma, rewards = 0.9, [1.0, 0.0, -2.0, 10.0]
    Gb, Gd = returns_backward(rewards, gamma), returns_direct(rewards, gamma)
    print(f"\n[1] episode with R_1..R_4 = {rewards}, gamma = {gamma}:")
    for t in reversed(range(len(rewards) + 1)):
        print(f"    G_{t} = {Gb[t]:7.4f}   (direct sum: {Gd[t]:7.4f})")
    assert np.allclose(Gb, Gd)

    # ---- 2. absorbing state: pad with zeros and use a (long) infinite-horizon sum
    padded = rewards + [0.0] * 200
    G_inf = returns_backward(padded, gamma)
    print(f"[2] same episode continued in an absorbing zero-reward state: G_0 = {G_inf[0]:.4f} "
          f"(identical: {np.isclose(G_inf[0], Gb[0])}); with gamma = 1 the episodic return is "
          f"{returns_backward(rewards, 1.0)[0]:.1f}")

    # ---- 3. constant rewards
    for g in (0.5, 0.9, 0.99):
        approx = returns_backward([1.0] * 5000, g)[0]
        print(f"[3] gamma = {g}: sum of 5000 discounted rewards of 1 = {approx:.4f}; 1/(1-gamma) = {1 / (1 - g):.4f}")

    # ---- 4. effective horizon
    print("\n[4] effective horizon")
    print(f"    {'gamma':>6} {'1/(1-g)':>8} {'weight in first 1/(1-g) steps':>31} {'steps until g^k < 0.01':>23}")
    for g in (0.5, 0.9, 0.99, 0.999):
        h = 1 / (1 - g)
        frac = 1 - g ** round(h)          # sum_{k<h} g^k / sum_k g^k
        k01 = int(np.ceil(np.log(0.01) / np.log(g)))
        print(f"    {g:>6} {h:>8.0f} {frac:>31.3f} {k01:>23d}")
    print(f"    (limit of 1 - gamma^(1/(1-gamma)) as gamma -> 1 is 1 - 1/e = {1 - np.exp(-1):.3f})")

    # ---- 5. interest rates
    for rate in (0.01, 0.05, 0.11):
        print(f"[5] an interest rate of {rate:.0%} per step corresponds to gamma = 1/(1+rate) = {1 / (1 + rate):.4f}")

    if not args.quick:
        import matplotlib.pyplot as plt
        from plotting import AQUA, BLUE, INK2, ORANGE, setup_style
        setup_style()
        figdir = Path(__file__).parent / "figures"
        figdir.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
        k = np.arange(0, 301)
        m = np.unique(np.round(np.logspace(0, 3, 400)).astype(int))   # number of rewards counted
        for g, col in [(0.5, BLUE), (0.9, ORANGE), (0.99, AQUA)]:
            h = 1 / (1 - g)
            axes[0].semilogy(k, g ** k, color=col, label=f"$\\gamma$ = {g}  (1/(1-$\\gamma$) = {h:.0f})")
            axes[0].plot([h], [g ** h], "o", color=col, ms=6)
            axes[1].semilogx(m, 1 - g ** m, color=col, label=f"$\\gamma$ = {g}")
            axes[1].plot([h], [1 - g ** h], "o", color=col, ms=6)
        axes[0].set_ylim(1e-4, 1.5)
        axes[0].set_xlabel("delay k (steps)")
        axes[0].set_ylabel(r"weight $\gamma^k$ of reward $R_{t+k+1}$")
        axes[0].set_title("Discount weights decay geometrically")
        axes[0].legend(fontsize=8.5, loc="upper right")
        axes[1].axhline(1 - np.exp(-1), color=INK2, ls=":", lw=1)
        axes[1].text(1.1, 1 - np.exp(-1) + 0.02, "1 - 1/e", color=INK2, fontsize=9)
        axes[1].set_xlabel("number m of first rewards counted (log scale)")
        axes[1].set_ylabel(r"share of total weight  $1-\gamma^{m}$")
        axes[1].set_title("Share of the weight in the first m rewards\n(dots: m = 1/(1-$\\gamma$))")
        axes[1].legend(fontsize=8.5, loc="lower right")
        fig.tight_layout()
        fig.savefig(figdir / "discount_weights.png")
        plt.close(fig)
        print(f"\nsaved figure to {figdir / 'discount_weights.png'}")

    print(f"\nSUMMARY: returns of the worked example G_0..G_4 = {np.round(Gb, 4).tolist()}; "
          f"at least 1 - 1/e (about 63%) of the discount weight lies in the first 1/(1-gamma) steps")
    print(f"done in {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
