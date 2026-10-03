"""Side-by-side: the three Monte Carlo control algorithms of this chapter on Blackjack.

  * Monte Carlo ES                    (mc_es_blackjack.py)       -- needs exploring starts
  * on-policy MC control, eps = 0.1   (on_policy_mc_control.py)  -- normal deals
  * off-policy MC control, random b   (off_policy_mc_control.py) -- normal deals

At log-spaced checkpoints we take each method's current greedy policy and compute its EXACT
expected return per game with blackjack.solve, so the curves contain no evaluation noise.

    python code/ch04_monte_carlo/compare_mc_control.py [--quick]
"""
import argparse
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blackjack import solve, stick_on_20_policy
from mc_es_blackjack import mc_es
from on_policy_mc_control import on_policy_mc_control
from off_policy_mc_control import off_policy_mc_control

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n = 30_000 if args.quick else 1_000_000
    seeds = [args.seed] if args.quick else [args.seed + k for k in range(3)]
    print(f"MC control comparison | seeds={seeds} episodes={n:,} (gamma=1, eps=0.1, b=uniform)")
    opt = solve(mode="optimal")
    grid = sorted(set(np.round(np.logspace(2, np.log10(n), 15)).astype(int)))
    methods = {
        "Monte Carlo ES": lambda s: mc_es(n, s, grid)[3],
        "on-policy, eps=0.1 (greedy part)": lambda s: on_policy_mc_control(n, s, 0.1, grid)[3],
        "off-policy, weighted IS, random b": lambda s: off_policy_mc_control(n, s, "random", 0.0, grid)[3],
    }
    t0 = time.time()
    res = {}
    for name, run in methods.items():
        Js, wrongs = [], []
        for s in seeds:
            snaps = run(s)
            pol = {g: (snaps[g][1] if isinstance(snaps[g], tuple) else snaps[g]) for g in grid}
            Js.append([solve(pol[g].astype(float))["J"] for g in grid])
            wrongs.append([int((pol[g] != opt["greedy"]).sum()) for g in grid])
        res[name] = (np.array(Js), np.array(wrongs))
        print(f"{name:36s}: final J = {np.mean(Js, 0)[-1]:+.5f} (seed range {np.min(Js, 0)[-1]:+.5f}.."
              f"{np.max(Js, 0)[-1]:+.5f}); states != pi_*: {np.mean(wrongs, 0)[-1]:.1f}  [{time.time() - t0:.0f}s]")
    print(f"exact J* = {opt['J']:+.5f};  J(stick on 20/21) = {solve(stick_on_20_policy())['J']:+.5f}")
    print("\n  episodes | " + " | ".join(f"{k[:22]:>22s}" for k in res))
    for i, g in enumerate(grid):
        print(f"  {g:>8,} | " + " | ".join(f"{res[k][0][:, i].mean():+.4f} ({res[k][1][:, i].mean():5.1f})".rjust(22)
                                          for k in res))
    print(f"(entries: seed-mean exact J of the greedy policy, and in brackets #states that differ from pi_*)")
    print(f"total elapsed {time.time() - t0:.1f}s")

    if args.quick:
        return
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for k, (name, (J, W)) in enumerate(res.items()):
        ax[0].semilogx(grid, J.mean(0), "o-", ms=3, color=f"C{k}", label=name)
        ax[0].fill_between(grid, J.min(0), J.max(0), color=f"C{k}", alpha=0.2)
        ax[1].semilogx(grid, W.mean(0), "o-", ms=3, color=f"C{k}", label=name)
    ax[0].axhline(opt["J"], color="k", ls="--", label=r"$J(\pi_*)$")
    ax[0].set_ylim(-0.25, -0.035)
    ax[0].set_ylabel("exact expected return of greedy policy")
    ax[1].set_ylabel(r"number of states (of 200) where policy $\neq \pi_*$")
    for a in ax:
        a.set_xlabel("episodes")
        a.legend(fontsize=8)
    fig.suptitle(f"Monte Carlo control on Blackjack ({len(seeds)} seeds; band = min..max)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "blackjack_control_comparison.png"), dpi=110)
    print("saved figures/blackjack_control_comparison.png")


if __name__ == "__main__":
    main()
