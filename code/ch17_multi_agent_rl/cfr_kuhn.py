"""Counterfactual regret minimisation on Kuhn poker (Chapter 17, section 10).

Implements, mirroring Algorithm 10.1 of the chapter:
  CFR      vanilla CFR (Zinkevich et al., 2007): full-tree traversal, simultaneous
           updates, regret matching, average strategy weighted by own reach probability
  CFR+     (Tammelin, 2014): alternating updates, regret matching+ (cumulative regrets
           floored at 0), linearly weighted average (iteration t gets weight t)
  CS-MCCFR chance-sampled Monte Carlo CFR: one random deal per iteration instead of all six

Exploitability is computed exactly with the best response of kuhn.py, and the value of
the average strategy profile is compared with the game value -1/18.

Run:  python code/ch17_multi_agent_rl/cfr_kuhn.py [--quick]
"""
import argparse
import os
import time

import numpy as np

import kuhn
from kuhn import ACTIONS, DEALS, infoset_key, player, utility


def regret_matching(cum_regret):
    """sigma(a|I) proportional to the positive part of the cumulative regret (Eq. 10.4)."""
    pos = np.maximum(cum_regret, 0.0)
    s = pos.sum()
    return pos / s if s > 0 else np.full(len(cum_regret), 1.0 / len(cum_regret))


class CFRSolver:
    def __init__(self, variant="cfr", seed=0):
        assert variant in ("cfr", "cfr+", "cs-mccfr")
        self.variant = variant
        self.rng = np.random.default_rng(seed)
        all_I = kuhn.infosets(1) + kuhn.infosets(2)
        self.regret = {I: np.zeros(2) for I in all_I}     # R^T(I, a)
        self.strat_sum = {I: np.zeros(2) for I in all_I}  # sum_t w_t eta_i^t(I) sigma^t(I)
        self.t = 0

    # -- strategies -------------------------------------------------------------------
    def current_strategy(self):
        return {I: regret_matching(r) for I, r in self.regret.items()}

    def average_strategy(self):
        avg = {}
        for I, s in self.strat_sum.items():
            avg[I] = s / s.sum() if s.sum() > 0 else np.array([0.5, 0.5])
        return avg

    def split(self, strategy):
        s1 = {I: strategy[I] for I in kuhn.infosets(1)}
        s2 = {I: strategy[I] for I in kuhn.infosets(2)}
        return s1, s2

    # -- one traversal ------------------------------------------------------------------
    def _traverse(self, c1, c2, h, reach, sigma, update, chance, avg_weight):
        """Return the expected utility for player 1 of the subtree at h under sigma, and
        add counterfactual regrets / average-strategy contributions for the players in
        `update`. reach = [eta_1(h), eta_2(h)] (each player's own contribution);
        chance = probability of the deal (or 1 under sampling, see Eq. 10.8)."""
        p = player(h)
        if p == 0:
            return utility(c1, c2, h)
        I = infoset_key(c1 if p == 1 else c2, h)
        s = sigma[I]
        child = np.empty(2)
        for k, a in enumerate(ACTIONS):
            new_reach = reach.copy()
            new_reach[p - 1] *= s[k]
            child[k] = self._traverse(c1, c2, h + a, new_reach, sigma, update, chance, avg_weight)
        v = s @ child
        if p in update:
            sign = 1.0 if p == 1 else -1.0                # utilities are stored for player 1
            cf_reach = chance * reach[2 - p]              # eta_{-i}(h): chance x opponent
            self.regret[I] += cf_reach * sign * (child - v)   # Eq. (10.3) increment
            self.strat_sum[I] += avg_weight * reach[p - 1] * s  # Eq. (10.6) increment
        return v

    def iterate(self):
        self.t += 1
        if self.variant == "cfr":
            sigma = self.current_strategy()               # fixed during the whole iteration
            for c1, c2 in DEALS:
                self._traverse(c1, c2, "", np.ones(2), sigma, (1, 2), 1 / len(DEALS), 1.0)
        elif self.variant == "cs-mccfr":
            sigma = self.current_strategy()
            c1, c2 = DEALS[self.rng.integers(len(DEALS))]
            # sampled deal: weight 1 instead of 1/6, i.e. divide by the sampling probability
            self._traverse(c1, c2, "", np.ones(2), sigma, (1, 2), 1.0, 1.0)
        else:  # CFR+: alternating updates, RM+, linear averaging
            for p in (1, 2):
                sigma = self.current_strategy()           # player 2 sees player 1's new strategy
                for c1, c2 in DEALS:
                    self._traverse(c1, c2, "", np.ones(2), sigma, (p,), 1 / len(DEALS), float(self.t))
                for I in kuhn.infosets(p):
                    np.maximum(self.regret[I], 0.0, out=self.regret[I])   # RM+ floor


def run(variant, T, checkpoints, seed=0):
    solver = CFRSolver(variant, seed=seed)
    rec = {k: [] for k in ("t", "expl_avg", "expl_cur", "ev_avg", "alpha", "betK", "callQ")}
    cps = set(checkpoints)
    for _ in range(T):
        solver.iterate()
        if solver.t in cps:
            a1, a2 = solver.split(solver.average_strategy())
            c1, c2 = solver.split(solver.current_strategy())
            rec["t"].append(solver.t)
            rec["expl_avg"].append(kuhn.exploitability(a1, a2))
            rec["expl_cur"].append(kuhn.exploitability(c1, c2))
            rec["ev_avg"].append(kuhn.expected_value(a1, a2))
            rec["alpha"].append(a1["J:"][1])
            rec["betK"].append(a1["K:"][1])
            rec["callQ"].append(a1["Q:pb"][1])
    return {k: np.array(v) for k, v in rec.items()}, solver


def main():
    parser = argparse.ArgumentParser(description="CFR / CFR+ / MCCFR on Kuhn poker")
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    T = 300 if args.quick else 20_000
    n_mc_seeds = 2 if args.quick else 5
    print(f"cfr_kuhn.py | seed={args.seed} | T={T} iterations | MCCFR seeds={n_mc_seeds} | quick={args.quick}")
    checkpoints = sorted(set(np.unique(np.logspace(0, np.log10(T), 60).astype(int)).tolist())
                         | {t for t in (10, 100, 1000, 10000) if t <= T})
    t0 = time.time()

    res = {}
    for variant in ("cfr", "cfr+"):
        t1 = time.time()
        res[variant], solver = run(variant, T, checkpoints)
        r = res[variant]
        a1, a2 = solver.split(solver.average_strategy())
        print(f"\n[{variant.upper()}] {T} iterations in {time.time() - t1:.1f} s")
        print(f"  exploitability of average strategy  = {r['expl_avg'][-1]:.2e}")
        print(f"  exploitability of current strategy  = {r['expl_cur'][-1]:.2e}")
        print(f"  value of average profile for P1      = {r['ev_avg'][-1]:+.6f}  (game value -1/18 = {kuhn.GAME_VALUE:+.6f})")
        for k, v in kuhn.describe(a1, a2).items():
            print(f"    {k:28s} {v:.4f}")
        alpha = a1["J:"][1]
        print(f"  equilibrium-family check: alpha = {alpha:.4f}, 3*alpha = {3 * alpha:.4f} "
              f"(P1 bets K {a1['K:'][1]:.4f}), alpha + 1/3 = {alpha + 1 / 3:.4f} "
              f"(P1 calls Q {a1['Q:pb'][1]:.4f})")
        sel = (r["t"] >= T / 100)
        slope = np.polyfit(np.log(r["t"][sel]), np.log(r["expl_avg"][sel]), 1)[0]
        print(f"  log-log slope of exploitability(avg) over the last two decades: {slope:.2f}")
        for tt in (10, 100, 1000, 10000):
            if tt in r["t"]:
                i = list(r["t"]).index(tt)
                print(f"    t = {tt:6d}: exploitability(avg) = {r['expl_avg'][i]:.2e}, "
                      f"exploitability(current) = {r['expl_cur'][i]:.2e}")

    # chance-sampled MCCFR: 6 sampled iterations cost about one vanilla iteration
    T_mc = 6 * T
    cps_mc = sorted(set(np.unique(np.logspace(0, np.log10(T_mc), 60).astype(int)).tolist()))
    mc = []
    t1 = time.time()
    for s in range(n_mc_seeds):
        r, _ = run("cs-mccfr", T_mc, cps_mc, seed=args.seed + s)
        mc.append(r)
    mc_expl = np.array([r["expl_avg"] for r in mc])
    print(f"\n[CS-MCCFR] {T_mc} sampled iterations x {n_mc_seeds} seeds in {time.time() - t1:.1f} s: "
          f"final exploitability(avg) mean {mc_expl[:, -1].mean():.2e} "
          f"(min {mc_expl[:, -1].min():.2e}, max {mc_expl[:, -1].max():.2e})")
    print(f"  vanilla CFR after the same number of dealt hands ({T} iterations x 6): "
          f"{res['cfr']['expl_avg'][-1]:.2e}")

    if not args.quick:
        import plot_style
        plt = plot_style.setup()
        figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
        os.makedirs(figdir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
        ax = axes[0]
        C = plot_style.C
        ax.loglog(res["cfr"]["t"], res["cfr"]["expl_avg"], color=C[0], label="CFR, average strategy")
        ax.loglog(res["cfr"]["t"], res["cfr"]["expl_cur"], color=C[0], ls=":", lw=1.3,
                  label="CFR, current strategy")
        ax.loglog(res["cfr+"]["t"], res["cfr+"]["expl_avg"], color=C[1], ls="--",
                  label="CFR+, average strategy")
        ax.loglog(res["cfr+"]["t"], res["cfr+"]["expl_cur"], color=C[1], ls="-.", lw=1.3,
                  label="CFR+, current strategy")
        tmc = np.array(mc[0]["t"]) / 6.0
        keep = tmc >= 1
        ax.loglog(tmc[keep], mc_expl.mean(axis=0)[keep], color=C[2], lw=1.5,
                  label=f"CS-MCCFR avg ({n_mc_seeds} seeds; x-axis = deals/6)")
        tt = np.array(res["cfr"]["t"], dtype=float)
        ax.loglog(tt, 0.5 / np.sqrt(tt), color=plot_style.GREY, ls=(0, (1, 2)), lw=1, label=r"$\propto 1/\sqrt{T}$")
        ax.loglog(tt, 0.5 / tt, color=plot_style.GREY, ls=(0, (4, 2)), lw=1, label=r"$\propto 1/T$")
        ax.set_xlabel("iteration T (full-tree traversals)")
        ax.set_ylabel("exploitability (chips per hand)")
        ax.set_title("(a) Exploitability, computed exactly")
        ax.legend(fontsize=7.5, loc="lower left")

        ax = axes[1]
        ax.semilogx(res["cfr"]["t"], res["cfr"]["ev_avg"], color=C[0], label="CFR")
        ax.semilogx(res["cfr+"]["t"], res["cfr+"]["ev_avg"], color=C[1], ls="--", label="CFR+")
        ax.axhline(kuhn.GAME_VALUE, color=plot_style.INK, lw=1, ls=":", label="game value $-1/18$")
        ax.set_xlabel("iteration T")
        ax.set_ylabel("value of average profile for player 1")
        ax.set_title("(b) Value of the average strategy profile")
        ax.set_ylim(-0.12, 0.05)
        ax.legend()

        ax = axes[2]
        r = res["cfr"]
        ax.semilogx(r["t"], r["alpha"], color=C[0], label=r"P1 bets J ($\alpha$)")
        ax.semilogx(r["t"], r["betK"] / 3, color=C[1], ls="--", label=r"P1 bets K, divided by 3")
        ax.semilogx(r["t"], r["callQ"] - 1 / 3, color=C[2], ls="-.", label=r"P1 calls with Q, minus 1/3")
        ax.set_xlabel("iteration T")
        ax.set_ylabel("probability")
        ax.set_title(r"(c) CFR average strategy: the three curves meet at $\alpha$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "cfr_kuhn.png"))
        plt.close(fig)
        print(f"\nfigure written to {figdir}/cfr_kuhn.png")
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
