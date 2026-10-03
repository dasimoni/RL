"""Regret as a function of the gap: why the worst case is ~ sqrt(kT) (Chapter 02, Section 11.7).

Two Bernoulli arms with means 0.5 and 0.5 - Delta.  For a fixed horizon T we sweep the gap
Delta over three orders of magnitude and measure E[Reg(T)]:
  * tiny gaps:  the arms are indistinguishable within T pulls, but mistakes are cheap,
                so Reg(T) <= Delta * T is small;
  * large gaps: the better arm is found after ~ 1/Delta^2 pulls, Reg(T) ~ log(T) / Delta is small;
  * in between, around Delta ~ sqrt(k/T), both effects meet and Reg(T) peaks at ~ sqrt(kT).
We repeat for several T and check that the WORST-CASE regret over Delta grows like sqrt(T)
(up to log factors), not like log T: the log T rates of Section 11 hide 1/Delta constants.

All (gap, run) pairs are simulated in parallel: row = one run at one gap.

Explore-then-commit's regret is extremely noisy (a run either commits correctly or pays Delta per
step for ~T steps), and the maximum of 19 noisy points is biased upward.  For ETC we therefore also
compute the EXACT expected regret from binomial distributions (etc_exact_regret_two_arms, the
formula of Exercise 2.9) and use it for the worst-case numbers; the simulation is a check.

Run:  python code/ch02_multi_armed_bandits/regret_vs_gap.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from bandits import (FIG_DIR, BernoulliBandit, ExploreThenCommit, ThompsonBeta, UCB,
                     etc_exact_regret_two_arms, run, setup_matplotlib, style)


def sweep(make_agent, gaps, T, runs, seed):
    probs = np.stack([np.full_like(gaps, 0.5), 0.5 - gaps], axis=1)       # [n_gaps, 2]
    probs = np.repeat(probs, runs, axis=0)                                 # [n_gaps * runs, 2]
    env = BernoulliBandit(probs, len(probs), np.random.default_rng(seed))
    res = run(env, make_agent(), T, np.random.default_rng(seed + 1))
    reg = res.final_regret.reshape(len(gaps), runs)
    return reg.mean(axis=1), reg.std(axis=1, ddof=1) / np.sqrt(runs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    seed = 0
    gaps = np.geomspace(0.002, 0.5, 13 if args.quick else 19)
    horizons = [500, 2000] if args.quick else [1000, 4000, 16000]
    runs = 50 if args.quick else 200
    print(f"Seed {seed}; 2 Bernoulli arms (0.5, 0.5-Δ); {len(gaps)} gaps in [{gaps[0]}, {gaps[-1]}]; "
          f"horizons {horizons}; {runs} runs per (gap, T)")
    t0 = time.time()
    m_etc = 50
    algos = {
        "UCB1": lambda: UCB(2, c=np.sqrt(2)),
        "Thompson (Beta)": lambda: ThompsonBeta(2),
        f"explore-then-commit m={m_etc}": lambda: ExploreThenCommit(2, m=m_etc),
    }
    etc_name = f"explore-then-commit m={m_etc}"
    exact_etc = {T: np.array([etc_exact_regret_two_arms(0.5, 0.5 - g, m_etc, T) for g in gaps]) for T in horizons}
    table = {}
    worst = {}                  # (name, T) -> (worst-case value used in the summary, its s.e., worst gap)
    for name, make in algos.items():
        for T in horizons:
            table[name, T] = sweep(make, gaps, T, runs, seed)
            m, se = table[name, T]
            j = int(np.argmax(m))
            line = (f"  {name:>25}, T={T:6d}: simulated worst gap Δ={gaps[j]:.4f}, E[Reg]={m[j]:7.1f} ± {se[j]:4.1f}; "
                    f"worst/√T = {m[j] / np.sqrt(T):.2f}")
            worst[name, T] = (m[j], se[j], gaps[j])
            if name == etc_name:
                ex = exact_etc[T]
                je = int(np.argmax(ex))
                worst[name, T] = (ex[je], 0.0, gaps[je])
                edge = " (edge of the gap grid)" if je == len(gaps) - 1 else ""
                line += (f"\n  {'':>25}  EXACT: worst gap Δ={gaps[je]:.4f}{edge}, E[Reg]={ex[je]:7.1f} "
                         f"(simulated there {m[je]:.1f} ± {se[je]:.1f}); worst/√T = {ex[je] / np.sqrt(T):.2f}; "
                         f"max |sim - exact| / s.e. over the grid = {np.max(np.abs(m - ex)[se > 1e-6] / se[se > 1e-6]):.1f}")
            print(line)
    # ETC's exact curve on a fine grid: its first local maximum is the "needle in a haystack" peak;
    # at large gaps the curve rises again only because exploration costs m * Delta.
    fine = np.geomspace(gaps[0], gaps[-1], 2000)
    peaks = []
    for T in horizons:
        v = np.array([etc_exact_regret_two_arms(0.5, 0.5 - g, m_etc, T) for g in fine])
        i = int(np.argmax((v[1:-1] >= v[:-2]) & (v[1:-1] >= v[2:]))) + 1      # first local maximum
        peaks.append(v[i])
        print(f"  ETC exact, fine grid, T={T}: first (interior) peak {v[i]:.1f} at Δ={fine[i]:.4f}; "
              f"value at Δ=0.5 is mΔ = {v[-1]:.1f}")
    print("  ETC interior-peak growth per ×4 in T: " + ", ".join(f"×{b / a:.2f}" for a, b in zip(peaks[:-1], peaks[1:])))
    print("\nsqrt(k/T) for k=2:", ", ".join(f"T={T}: {np.sqrt(2 / T):.4f}" for T in horizons))
    print("growth of the worst-case regret when T is multiplied by 4 (√T scaling -> ×2.0; ETC uses exact values;"
          " the max over noisy points is biased upward, so read these as ± several %):")
    for name in algos:
        w = [worst[name, T][0] for T in horizons]
        print(f"  {name:>25}: " + ", ".join(f"×{b / a:.2f}" for a, b in zip(w[:-1], w[1:])))

    if not args.quick:
        plt = setup_matplotlib()
        fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
        T = horizons[-1]
        for i, name in enumerate(algos):
            m, se = table[name, T]
            ax[0].plot(gaps, m, marker="o", ms=3.5, label=name + (" (simulated)" if name == etc_name else ""),
                       **style(i))
            ax[0].fill_between(gaps, m - 2 * se, m + 2 * se, color=style(i)["color"], alpha=0.15, lw=0)
            if name == etc_name:
                fine = np.geomspace(gaps[0], gaps[-1], 300)
                ax[0].plot(fine, [etc_exact_regret_two_arms(0.5, 0.5 - g, m_etc, T) for g in fine],
                           color=style(i)["color"], lw=1.0, ls="-", alpha=0.8, label=f"{name} (exact)")
        ax[0].plot(gaps, gaps * T / 2, color="#8a8984", lw=1, ls=(0, (6, 3)), label="ΔT/2 (pull arms at random)")
        ax[0].axvline(np.sqrt(2 / T), color="#0b0b0b", lw=0.8, ls=":")
        ax[0].text(np.sqrt(2 / T) * 1.08, 0.04, "Δ = √(k/T)", fontsize=8, color="#0b0b0b",
                   transform=ax[0].get_xaxis_transform())
        ax[0].set_xscale("log")
        ax[0].set_ylim(0, 1.25 * max(table[n, T][0].max() for n in algos))
        ax[0].set_ylim(0, max(ax[0].get_ylim()[1], 1.15 * exact_etc[T].max()))
        ax[0].set(xlabel="gap Δ (log scale)", ylabel=f"E[Reg(T)] at T = {T}",
                  title=f"(a) Regret vs gap, two arms, T={T} ({runs} runs/point)")
        ax[0].legend(loc="upper left", fontsize=8)
        Ts = np.array(horizons, dtype=float)
        for i, name in enumerate(algos):
            w = [worst[name, T][0] for T in horizons]
            ax[1].plot(Ts, w, marker="o", label=name + (" (exact)" if name == etc_name else ""), **style(i))
        ref = worst["UCB1", horizons[0]][0]
        ax[1].plot(Ts, ref * np.sqrt(Ts / Ts[0]), color="#8a8984", lw=1, ls=(0, (6, 3)), label="∝ √T reference")
        ax[1].set_xscale("log")
        ax[1].set_yscale("log")
        from matplotlib.ticker import FuncFormatter
        plain = FuncFormatter(lambda v, _: f"{v:g}")
        ax[1].yaxis.set_major_formatter(plain)
        ax[1].yaxis.set_minor_formatter(plain)
        ax[1].tick_params(axis="y", which="minor", labelsize=7)
        ax[1].set(xlabel="horizon T (log scale)", ylabel="worst-case E[Reg(T)] over the Δ grid",
                  title="(b) Worst case over the Δ grid: ∝ √T, except ETC")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "regret_vs_gap.png")
        fig.savefig(out)
        print(f"\nsaved {out}")
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
