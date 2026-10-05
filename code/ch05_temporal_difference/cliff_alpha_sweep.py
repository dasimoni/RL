"""Step-size sensitivity of SARSA, Q-learning and Expected SARSA on cliff walking.

Chapter 05, Section 10.3, in the spirit of Sutton & Barto (2018), Figure 6.3
(van Seijen, van Hasselt, Whiteson & Wiering, 2009).

For each alpha in {0.1, 0.2, ..., 1.0} and each method we measure
  * interim performance:  mean sum of rewards per episode over the first 100 episodes,
                          averaged over N_INTERIM independent runs;
  * long-run performance: mean sum of rewards per episode over the first N_LONG episodes,
                          averaged over a few runs.
S&B use 100,000 episodes for the "asymptotic" curve; we use far fewer to stay within a
small CPU budget, so our "long-run" numbers are an approximation of theirs (stated in
the chapter).

Everything is online performance of the epsilon-greedy behaviour (epsilon = 0.1).

Run:  python code/ch05_temporal_difference/cliff_alpha_sweep.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.dont_write_bytecode = True        # importing the sibling script must not leave __pycache__/

import numpy as np                     # noqa: E402

from cliff_walking import ALGORITHMS, MAX_STEPS, train   # noqa: E402  (same environment and agents)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    alphas = [0.1, 0.5, 1.0] if args.quick else [round(0.1 * k, 1) for k in range(1, 11)]
    n_interim_runs = 5 if args.quick else 50
    n_long, n_long_runs = (500, 1) if args.quick else (5000, 5)
    eps = 0.1
    print(f"[cliff_alpha_sweep] seed={args.seed} epsilon={eps} alphas={alphas}\n"
          f"  interim: first 100 episodes x {n_interim_runs} runs; "
          f"long-run: first {n_long} episodes x {n_long_runs} runs")
    t0 = time.time()

    interim = {m: [] for m in ALGORITHMS}
    long_run = {m: [] for m in ALGORITHMS}
    capped = {}                       # (method, alpha) -> episodes abandoned at MAX_STEPS
    for m in ALGORITHMS:
        for a in alphas:
            r_int, r_long, n_cap = [], [], 0
            for run in range(n_interim_runs):
                _, R, c = train(m, 100, a, eps, seed=args.seed + run)
                r_int.append(R.mean()); n_cap += c
            for run in range(n_long_runs):
                _, R, c = train(m, n_long, a, eps, seed=args.seed + 10_000 + run)
                r_long.append(R.mean()); n_cap += c
            interim[m].append(np.mean(r_int))
            long_run[m].append(np.mean(r_long))
            if n_cap:
                capped[(m, a)] = n_cap

    print("\nMean sum of rewards per episode (online, eps-greedy):")
    header = "  alpha        " + "".join(f"{a:>7.1f}" for a in alphas)
    print(header)
    for m in ALGORITHMS:
        print(f"  {m[:14]:14s} interim " + "".join(f"{x:7.1f}" for x in interim[m]))
        print(f"  {m[:14]:14s} long    " + "".join(f"{x:7.1f}" for x in long_run[m]))

    if capped:
        print(f"\nEpisodes abandoned at the {MAX_STEPS}-step cap (their partial return is still counted):")
        for (m, a), c in capped.items():
            print(f"  {m}, alpha={a}: {c}")

    if not args.quick:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIG_DIR, exist_ok=True)
        colors = {"SARSA": "tab:blue", "Q-learning": "tab:red", "Expected SARSA": "tab:green"}
        fig, ax = plt.subplots(figsize=(9.0, 4.4))
        for m in ALGORITHMS:
            ax.plot(alphas, interim[m], "--o", color=colors[m], ms=4, label=f"{m}, interim (first 100 ep.)")
            ax.plot(alphas, long_run[m], "-s", color=colors[m], ms=4, label=f"{m}, long-run (first {n_long} ep.)")
        ax.set_xlabel("step size α")
        ax.set_ylabel("mean sum of rewards per episode")
        ax.set_title("Cliff walking: sensitivity to α (ε-greedy, ε = 0.1)")
        ax.set_ylim(-160, 0)
        for m in ALGORITHMS:          # annotate points that fall below the plotted range
            for a, v in zip(alphas, long_run[m]):
                if v < -160:
                    ax.annotate(f"{m} long-run\nα={a}: {v:.0f}", xy=(a, -158), xytext=(a - 0.3, -145),
                                fontsize=7, color=colors[m], arrowprops=dict(arrowstyle="->", color=colors[m]))
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "cliff_alpha_sweep.png"), dpi=110)
        plt.close(fig)
        print(f"\nFigure written to {FIG_DIR}/cliff_alpha_sweep.png")
    print(f"\nRuntime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
