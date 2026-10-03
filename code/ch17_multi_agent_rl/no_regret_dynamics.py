"""Learning in matrix games: averages converge, last iterates cycle (Chapter 17, section 5).

Both players run the same learning rule against each other ("self-play") with
full-information feedback: after each round, player 1 sees the vector of expected
payoffs u1_t = A y_t of all of its actions, and player 2 sees u2_t = x_t^T B.

Algorithms (section 5):
  FP     fictitious play: best response to the opponent's empirical average (Alg. 5.1);
         the starting strategies serve as one fictitious round of prior counts, and the
         first real round is a pure best response to them
  RM     regret matching: play proportional to positive cumulative regret (Alg. 5.2)
  RM+    regret matching+: cumulative regrets clipped at zero every step
  Hedge  exponential weights with decreasing step eta_t = 1/sqrt(t) (Alg. 5.3)
  OMWU   optimistic exponential weights, constant step eta = 0.1 (Alg. 5.3)

Experiments:
  1. Zero-sum games (matching pennies, RPS, weighted RPS): NashConv of the time-averaged
     strategies goes to zero; NashConv of the current ("last") iterates does not, except
     for OMWU. Simplex plot of the trajectories on weighted RPS.
  2. Shapley's general-sum game: the empirical joint distribution of RM and Hedge play
     converges to the set of coarse correlated equilibria (CCE gap -> 0), while the
     marginal averages do not converge to the unique Nash equilibrium.

Run:  python code/ch17_multi_agent_rl/no_regret_dynamics.py [--quick]
"""
import argparse
import os
import time

import numpy as np

import games

ALGOS = ["FP", "RM", "RM+", "Hedge", "OMWU"]


def run_dynamics(A, B, algo, T, x1, y1, eta_omwu=0.1, record=False, track_cce=False):
    """Simulate T rounds of self-play. Returns a dict of per-round diagnostics.

    x_t, y_t are the strategies used in round t (the 'last iterates'); the averages are
    xbar_t = (1/t) sum_{s<=t} x_s.  P_t = (1/t) sum_s x_s y_s^T is the empirical joint
    distribution of play (the expectation of the empirical frequency of sampled joint
    actions), which is what the no-regret -> CCE theorem (Eq. 5.5) is about.
    """
    m, n = A.shape
    x, y = x1.astype(float).copy(), y1.astype(float).copy()
    if algo == "FP":
        # Alg. 5.1: x1, y1 are the initial beliefs (one fictitious round, counted once in the
        # pseudo-counts below); the first real round is already a pure best response to them.
        x, y = np.eye(m)[np.argmax(A @ y1)], np.eye(n)[np.argmax(x1 @ B)]
    # state of each learner
    R1, R2 = np.zeros(m), np.zeros(n)            # cumulative regrets (RM / RM+)
    U1, U2 = np.zeros(m), np.zeros(n)            # cumulative payoff vectors (Hedge / OMWU)
    last_u1, last_u2 = np.zeros(m), np.zeros(n)  # previous payoff vector (OMWU optimism)
    cnt1, cnt2 = x1.astype(float).copy(), y1.astype(float).copy()  # FP pseudo-counts (prior once)
    xbar, ybar = np.zeros(m), np.zeros(n)
    P = np.zeros((m, n))
    nc_last = np.empty(T)
    nc_avg = np.empty(T)
    cce = np.empty(T)
    reg1 = np.empty(T)
    xs = np.empty((T, m)) if record else None      # player 1's trajectory (for the simplex plot)
    xbars = np.empty((T, m)) if record else None
    for t in range(1, T + 1):
        # --- play round t with (x, y) and record diagnostics -----------------------
        xbar += (x - xbar) / t
        ybar += (y - ybar) / t
        P += (np.outer(x, y) - P) / t
        if record:
            xs[t - 1], xbars[t - 1] = x, xbar
        nc_last[t - 1] = games.nash_conv(A, B, x, y)
        nc_avg[t - 1] = games.nash_conv(A, B, xbar, ybar)
        if track_cce:
            cce[t - 1] = games.cce_gap(A, B, P)
        u1, u2 = A @ y, x @ B                    # full-information feedback
        r1, r2 = u1 - x @ u1, u2 - u2 @ y        # instantaneous regrets r_t(a)
        R1 += r1
        R2 += r2
        reg1[t - 1] = R1.max() / t               # average external regret of player 1
        # --- each learner chooses its next strategy --------------------------------
        if algo == "FP":
            cnt1 += x
            cnt2 += y
            x = np.eye(m)[np.argmax(A @ (cnt2 / cnt2.sum()))]   # BR to opponent's average
            y = np.eye(n)[np.argmax((cnt1 / cnt1.sum()) @ B)]
        elif algo in ("RM", "RM+"):
            if algo == "RM+":
                np.maximum(R1, 0, out=R1)        # RM+: forget negative regret immediately
                np.maximum(R2, 0, out=R2)
            p1, p2 = np.maximum(R1, 0), np.maximum(R2, 0)
            x = p1 / p1.sum() if p1.sum() > 0 else np.full(m, 1 / m)
            y = p2 / p2.sum() if p2.sum() > 0 else np.full(n, 1 / n)
        elif algo == "Hedge":
            U1 += u1
            U2 += u2
            eta = 1.0 / np.sqrt(t)
            z1 = eta * U1 + np.log(x1)
            z2 = eta * U2 + np.log(y1)
            x = np.exp(z1 - z1.max()); x /= x.sum()
            y = np.exp(z2 - z2.max()); y /= y.sum()
        elif algo == "OMWU":
            U1 += u1
            U2 += u2
            z1 = eta_omwu * (U1 + u1) + np.log(x1)   # optimism: count the last payoff twice
            z2 = eta_omwu * (U2 + u2) + np.log(y1)
            x = np.exp(z1 - z1.max()); x /= x.sum()
            y = np.exp(z2 - z2.max()); y /= y.sum()
        else:
            raise ValueError(algo)
    return dict(nc_last=nc_last, nc_avg=nc_avg, cce=cce, reg1=reg1, xbar=xbar, ybar=ybar,
                P=P, x_last=x, y_last=y, xs=xs, xbars=xbars)


def loglog_slope(t, v, lo, hi):
    """Least-squares slope of log v against log t for t in [lo, hi]."""
    sel = (t >= lo) & (t <= hi) & (v > 0)
    return np.polyfit(np.log(t[sel]), np.log(v[sel]), 1)[0]


def to_xy(p):
    """Barycentric coordinates on the probability simplex -> 2-D point in a triangle."""
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    return p @ corners


def main():
    parser = argparse.ArgumentParser(description="No-regret dynamics in matrix games")
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0, help="unused: all dynamics are deterministic")
    args = parser.parse_args()
    T = 2_000 if args.quick else 100_000
    T_shapley = 2_000 if args.quick else 100_000
    print(f"no_regret_dynamics.py | seed={args.seed} (dynamics are deterministic) | T={T} "
          f"| Hedge eta_t=1/sqrt(t), OMWU eta=0.1 | quick={args.quick}")
    t0 = time.time()

    zs_games = {"matching pennies": games.matching_pennies(),
                "RPS": games.rock_paper_scissors(),
                "weighted RPS": games.weighted_rps()}
    # Non-equilibrium starting strategies (from the uniform point RM and Hedge would never move).
    starts = {2: (np.array([0.8, 0.2]), np.array([0.3, 0.7])),
              3: (np.array([0.6, 0.3, 0.1]), np.array([0.2, 0.2, 0.6]))}
    results = {}
    ts = np.arange(1, T + 1)
    print("\nExperiment 1: zero-sum games. NashConv = sum of best-response gains (0 at Nash).")
    print(f"{'game':18s} {'algo':6s} {'NashConv(avg) @T':>17s} {'slope(avg)':>10s} "
          f"{'NashConv(last) median of last 10%':>34s} {'avg regret P1':>14s} {'RM bound':>9s}")
    for gname, (A, B, _) in zs_games.items():
        x1, y1 = starts[A.shape[0]]
        delta = A.max() - A.min()                    # payoff range: |r_t(a)| <= delta
        for algo in ALGOS:
            r = run_dynamics(A, B, algo, T, x1, y1)
            results[(gname, algo)] = r
            slope = loglog_slope(ts, r["nc_avg"], T / 100, T)
            last_med = np.median(r["nc_last"][-T // 10:])
            bound = delta * np.sqrt(A.shape[0] / T)  # Eq. (5.3): RM average regret bound
            print(f"{gname:18s} {algo:6s} {r['nc_avg'][-1]:17.2e} {slope:10.2f} "
                  f"{last_med:34.2e} {r['reg1'][-1]:14.2e} {bound:9.2e}")

    # ---------------- Experiment 2: Shapley's game (general sum) ---------------------
    A, B, _ = games.shapley_game()
    x1, y1 = np.array([0.6, 0.3, 0.1]), np.array([0.2, 0.2, 0.6])
    print("\nExperiment 2: Shapley's game (unique Nash = uniform, Nash payoff 1/3 each).")
    print(f"{'algo':6s} {'CCE gap(P_T)':>13s} {'CE gap(P_T)':>12s} {'NashConv(avg)':>14s} "
          f"{'xbar_T':>24s} {'welfare(P_T)':>13s} {'payoffs (P1, P2)':>17s}")
    shap = {}
    for algo in ["FP", "RM", "Hedge"]:
        r = run_dynamics(A, B, algo, T_shapley, x1, y1, track_cce=True)
        shap[algo] = r
        P = r["P"]
        print(f"{algo:6s} {max(r['cce'][-1], 0):13.2e} {games.ce_gap(A, B, P):12.3f} "
              f"{r['nc_avg'][-1]:14.3f} {np.array2string(r['xbar'], precision=3):>24s} "
              f"{np.sum(P * (A + B)):13.3f}    ({np.sum(P * A):.3f}, {np.sum(P * B):.3f})")
    tsh = np.arange(1, T_shapley + 1)
    lo_fit = min(1_000, T_shapley // 10)
    for algo in ["FP", "RM"]:
        cg = shap[algo]["cce"]
        marks = [t for t in (1_000, 10_000, 100_000) if t <= T_shapley]
        print(f"  {algo}: CCE gap at t = " + ", ".join(f"{t}: {cg[t - 1]:.2e}" for t in marks)
              + f"; fitted log-log slope over [{lo_fit}, {T_shapley}] = {loglog_slope(tsh, cg, lo_fit, T_shapley):.2f}")
    print("FP empirical joint distribution P_T (rows = player 1):\n"
          + np.array2string(shap["FP"]["P"], precision=3))
    print("RM empirical joint distribution P_T:\n" + np.array2string(shap["RM"]["P"], precision=3))

    if not args.quick:
        import plot_style
        plt = plot_style.setup()
        figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
        os.makedirs(figdir, exist_ok=True)
        styles = {"FP": dict(color=plot_style.C[0], ls="-"), "RM": dict(color=plot_style.C[1], ls="-"),
                  "RM+": dict(color=plot_style.C[2], ls="--"), "Hedge": dict(color=plot_style.C[3], ls="-."),
                  "OMWU": dict(color=plot_style.C[6], ls=":")}
        sub = np.unique(np.logspace(0, np.log10(T), 400).astype(int)) - 1

        # Figure 1: NashConv of averages (top) and last iterates (bottom)
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True, sharey="row")
        for c, gname in enumerate(zs_games):
            for algo in ALGOS:
                r = results[(gname, algo)]
                axes[0, c].loglog(ts[sub], r["nc_avg"][sub], label=algo, lw=1.8, **styles[algo])
                axes[1, c].loglog(ts[sub], np.maximum(r["nc_last"][sub], 1e-12), label=algo,
                                  lw=1.3, **styles[algo])
            axes[0, c].loglog(ts[sub], 2 * np.sqrt(1.0 / ts[sub]), color=plot_style.GREY, lw=1,
                              ls=(0, (1, 2)), label=r"$\propto 1/\sqrt{t}$")
            axes[0, c].set_title(f"{gname}: time-averaged strategies")
            axes[1, c].set_title(f"{gname}: current (last) iterates\n(values below 1e-12 drawn at 1e-12)",
                                 fontsize=10)
            axes[1, c].set_xlabel("iteration t")
        axes[0, 0].set_ylabel("NashConv")
        axes[1, 0].set_ylabel("NashConv")
        axes[0, 0].legend(loc="lower left", ncol=2)
        fig.suptitle("Self-play in zero-sum games: averages converge to Nash, last iterates cycle "
                     "(except optimistic MWU)", y=0.995)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "nashconv_zero_sum.png"))
        plt.close(fig)

        # Figure 2: simplex trajectories of player 1 on weighted RPS
        A, B, _ = games.weighted_rps()
        x1, y1 = starts[3]
        nash = np.array([0.25, 0.5, 0.25])
        fig, axes = plt.subplots(1, 4, figsize=(14, 4.3))
        Ttraj = 3000
        for ax, algo in zip(axes, ["FP", "RM", "Hedge", "OMWU"]):
            tr = run_dynamics(A, B, algo, Ttraj, x1, y1, record=True)
            xs, xbars = tr["xs"], tr["xbars"]
            tri = to_xy(np.vstack([np.eye(3), np.eye(3)[:1]]))
            ax.plot(tri[:, 0], tri[:, 1], color=plot_style.GREY, lw=1)
            p = to_xy(xs)
            if algo != "FP":
                ax.plot(p[:, 0], p[:, 1], color=plot_style.C[0], lw=0.6, alpha=0.8)
            else:   # FP's current iterate is a pure strategy: a vertex of the simplex
                ax.scatter(p[1:, 0], p[1:, 1], color=plot_style.C[0], s=40, zorder=3)
            ax.scatter([p[0, 0]], [p[0, 1]], marker="o", facecolor="white", edgecolor=plot_style.INK,
                       s=40, zorder=5)
            pb = to_xy(xbars)
            ax.plot(pb[:, 0], pb[:, 1], color=plot_style.C[1], lw=2.0)
            ns = to_xy(nash)
            ax.scatter([ns[0]], [ns[1]], marker="*", s=180, color=plot_style.INK, zorder=4)
            for k, lab in enumerate(["R", "P", "S"]):
                q = to_xy(np.eye(3)[k])
                ax.annotate(lab, q, xytext=(q[0] + (-0.06 if k == 0 else 0.03), q[1] + (0.02 if k == 2 else -0.06)))
            ax.set_title(f"{algo} ({Ttraj} rounds)")
            ax.set_aspect("equal"); ax.axis("off")
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=plot_style.C[0], lw=1, label="current iterate $x_t$ (FP: a vertex)"),
                   Line2D([], [], color=plot_style.C[1], lw=2, label=r"time average $\bar x_t$"),
                   Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=plot_style.INK,
                          label="start $x_1$"),
                   Line2D([], [], marker="*", ls="", markersize=12, color=plot_style.INK,
                          label="Nash (1/4, 1/2, 1/4)")]
        fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle("Weighted RPS: player 1's strategy in the probability simplex", y=0.99)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(os.path.join(figdir, "simplex_weighted_rps.png"))
        plt.close(fig)

        # Figure 3: Shapley's game
        subs = np.unique(np.logspace(0, np.log10(T_shapley), 400).astype(int)) - 1
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for algo in ["FP", "RM", "Hedge"]:
            r = shap[algo]
            axes[0].loglog(tsh[subs], np.maximum(r["cce"][subs], 1e-6), label=algo, **styles[algo])
            axes[1].semilogx(tsh[subs], r["nc_avg"][subs], label=algo, **styles[algo])
        axes[0].text(1.5, 2e-6, "gap <= 0 (strictly inside the CCE set) drawn at 1e-6",
                     fontsize=8, color=plot_style.GREY)
        axes[0].set_title("(a) CCE gap of the empirical joint distribution $P_t$")
        axes[0].set_xlabel("iteration t"); axes[0].set_ylabel("max gain from a fixed deviation")
        axes[1].set_title(r"(b) NashConv of the marginal averages $(\bar x_t, \bar y_t)$")
        axes[1].set_xlabel("iteration t"); axes[1].set_ylabel("NashConv")
        axes[0].legend(); axes[1].legend()
        fig.suptitle("Shapley's game: no-regret play reaches the CCE set, not Nash", y=1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "shapley_cce.png"))
        plt.close(fig)
        print(f"\nfigures written to {figdir}")
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
