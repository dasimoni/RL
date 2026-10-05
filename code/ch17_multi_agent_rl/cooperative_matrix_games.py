"""Independent learners vs a centralized learner on cooperative matrix games
(Chapter 17, section 6).

Two agents repeatedly play a fully cooperative 3x3 game: both receive the same reward
r(a1, a2). Learners (all stateless Q-learning with step size alpha, Eq. 6.1):

  central      one learner over the 9 joint actions (the "team as a single agent")
  JAL          Claus & Boutilier's joint-action learners: each agent learns Q_i(a1, a2),
               models its partner by empirical action frequencies, and acts greedily
               on the expected value of its own actions
  IQL-eps      independent Q-learners, epsilon-greedy
  IQL-Boltz    independent Q-learners, Boltzmann exploration with decaying temperature
  hysteretic   independent learners with a large step for positive TD errors and a small
               one for negative TD errors (Matignon, Laurent & Le Fort-Piat, 2007)

Games: the climbing game, the penalty game (k = -100), and a partially stochastic
climbing game in which (b, b) pays 14 or 0 with probability 1/2 each (mean 7).

Everything is vectorised over independent runs (axis 0 of every array).
Run:  python code/ch17_multi_agent_rl/cooperative_matrix_games.py [--quick]
"""
import argparse
import os
import time

import numpy as np

import games

LEARNERS = ["central", "JAL", "IQL-eps", "IQL-Boltz", "hysteretic"]


def make_games():
    climb, _, _ = games.climbing_game()
    pen, _, _ = games.penalty_game(-100.0)
    return {
        "climbing": dict(mean=climb, stochastic=False, optimal=[(0, 0)]),
        "penalty (k=-100)": dict(mean=pen, stochastic=False, optimal=[(0, 0), (2, 2)]),
        "stochastic climbing": dict(mean=climb, stochastic=True, optimal=[(0, 0)]),
    }


def sample_reward(game, a1, a2, rng):
    r = game["mean"][a1, a2].copy()
    if game["stochastic"]:
        bb = (a1 == 1) & (a2 == 1)                         # (b, b): 14 or 0, mean 7
        r[bb] = np.where(rng.random(bb.sum()) < 0.5, 14.0, 0.0)
    return r


def argmax_random(Q, rng):
    """Greedy action per row with uniformly random tie-breaking."""
    noise = rng.random(Q.shape) * 1e-9
    return np.argmax(Q + noise, axis=-1)


def eps_greedy(Q, eps, rng):
    greedy = argmax_random(Q, rng)
    explore = rng.random(Q.shape[0]) < eps
    return np.where(explore, rng.integers(Q.shape[-1], size=Q.shape[0]), greedy)


def run(learner, game, T, R, rng, alpha=0.1, beta_hyst=0.01, eps_decay=0.998, eps_min=0.01,
        tau0=20.0, tau_decay=0.997, tau_min=0.05):
    """Simulate R independent runs of T steps. Returns final greedy joint actions,
    the mean reward of the greedy joint action over time, and agent 1's mean Q-values."""
    n = 3
    idx = np.arange(R)
    Qc = np.zeros((R, n * n))                 # central learner
    Q1, Q2 = np.zeros((R, n)), np.zeros((R, n))
    J1, J2 = np.zeros((R, n, n)), np.zeros((R, n, n))   # JAL: Q_i(a1, a2)
    C1, C2 = np.ones((R, n)), np.ones((R, n))             # JAL: partner action counts (+1 prior)
    curve = np.empty(T)
    q1_trace = np.empty((T, n))
    for t in range(T):
        eps = max(eps_min, eps_decay ** t)
        if learner == "central":
            a = eps_greedy(Qc, eps, rng)
            a1, a2 = a // n, a % n
        elif learner == "JAL":
            ev1 = np.einsum("rij,rj->ri", J1, C2 / C2.sum(1, keepdims=True))   # E_{a2~model}[Q1(a1,a2)]
            ev2 = np.einsum("rij,ri->rj", J2, C1 / C1.sum(1, keepdims=True))
            a1, a2 = eps_greedy(ev1, eps, rng), eps_greedy(ev2, eps, rng)
        elif learner == "IQL-Boltz":
            tau = max(tau_min, tau0 * tau_decay ** t)
            p1 = np.exp((Q1 - Q1.max(1, keepdims=True)) / tau); p1 /= p1.sum(1, keepdims=True)
            p2 = np.exp((Q2 - Q2.max(1, keepdims=True)) / tau); p2 /= p2.sum(1, keepdims=True)
            a1 = (rng.random((R, 1)) > np.cumsum(p1, 1)).sum(1)
            a2 = (rng.random((R, 1)) > np.cumsum(p2, 1)).sum(1)
            a1, a2 = np.minimum(a1, n - 1), np.minimum(a2, n - 1)
        else:  # IQL-eps and hysteretic
            a1, a2 = eps_greedy(Q1, eps, rng), eps_greedy(Q2, eps, rng)
        r = sample_reward(game, a1, a2, rng)
        # ---- updates ----------------------------------------------------------------
        if learner == "central":
            a = a1 * n + a2
            Qc[idx, a] += alpha * (r - Qc[idx, a])
        elif learner == "JAL":
            J1[idx, a1, a2] += alpha * (r - J1[idx, a1, a2])
            J2[idx, a1, a2] += alpha * (r - J2[idx, a1, a2])
            C1[idx, a1] += 1
            C2[idx, a2] += 1
        else:
            d1, d2 = r - Q1[idx, a1], r - Q2[idx, a2]          # TD errors (one-step game)
            if learner == "hysteretic":
                Q1[idx, a1] += np.where(d1 >= 0, alpha, beta_hyst) * d1
                Q2[idx, a2] += np.where(d2 >= 0, alpha, beta_hyst) * d2
            else:
                Q1[idx, a1] += alpha * d1
                Q2[idx, a2] += alpha * d2
        # ---- greedy joint action now ----------------------------------------------------
        if learner == "central":
            g = np.argmax(Qc, 1)
            g1, g2 = g // n, g % n
        elif learner == "JAL":
            g1 = np.argmax(np.einsum("rij,rj->ri", J1, C2 / C2.sum(1, keepdims=True)), 1)
            g2 = np.argmax(np.einsum("rij,ri->rj", J2, C1 / C1.sum(1, keepdims=True)), 1)
        else:
            g1, g2 = np.argmax(Q1, 1), np.argmax(Q2, 1)
        curve[t] = game["mean"][g1, g2].mean()
        q1_trace[t] = Q1.mean(0)
    return (g1, g2), curve, q1_trace


def main():
    parser = argparse.ArgumentParser(description="IQL vs centralized learners on cooperative matrix games")
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    R = 100 if args.quick else 1000
    T = 1500 if args.quick else 5000
    print(f"cooperative_matrix_games.py | seed={args.seed} | runs={R} | steps={T} | alpha=0.1 | "
          f"eps_t=max(0.01, 0.998^t) | Boltzmann tau_t=max(0.05, 20*0.997^t) | hysteretic beta=0.01 | quick={args.quick}")
    t0 = time.time()
    gms = make_games()
    names = "abc"
    out = {}
    for gname, game in gms.items():
        print(f"\n=== {gname} ===")
        print("mean payoffs:\n" + np.array2string(game["mean"], precision=0))
        print(f"{'learner':11s} {'P(optimal)':>10s}  greedy joint action frequencies (a1a2: %)"
              f"                        mean greedy payoff")
        for k, learner in enumerate(LEARNERS):
            rng = np.random.default_rng(args.seed + 1000 * k + 7)
            (g1, g2), curve, q1 = run(learner, game, T, R, rng)
            freq = np.zeros((3, 3))
            np.add.at(freq, (g1, g2), 1)
            freq /= R
            p_opt = sum(freq[i, j] for i, j in game["optimal"])
            out[(gname, learner)] = dict(p_opt=p_opt, freq=freq, curve=curve, q1=q1)
            top = sorted(((freq[i, j], f"{names[i]}{names[j]}") for i in range(3) for j in range(3)), reverse=True)
            top_s = ", ".join(f"{s}: {100 * f:5.1f}" for f, s in top if f > 0)
            print(f"{learner:11s} {p_opt:10.3f}  {top_s:60s} {curve[-1]:7.2f}")
    print(f"\ntotal time {time.time() - t0:.1f} s")

    if not args.quick:
        import plot_style
        plt = plot_style.setup()
        figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
        os.makedirs(figdir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        ax = axes[0]
        width = 0.26
        xs = np.arange(len(LEARNERS))
        hatches = ["", "//", ".."]
        for gi, gname in enumerate(gms):
            vals = [out[(gname, l)]["p_opt"] for l in LEARNERS]
            ax.bar(xs + (gi - 1) * width, vals, width * 0.92, color=plot_style.C[gi], hatch=hatches[gi],
                   edgecolor="white", label=gname)
            for xv, v in zip(xs + (gi - 1) * width, vals):
                ax.text(xv, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels(LEARNERS, rotation=15)
        ax.set_ylabel("fraction of runs whose greedy joint\naction is optimal at the end")
        ax.set_ylim(0, 1.38)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(f"(a) Coordination success ({R} runs each)")
        ax.legend(loc="upper center", fontsize=8, ncol=3)

        ax = axes[1]
        sty = {"central": dict(color=plot_style.C[0], ls="-"), "JAL": dict(color=plot_style.C[1], ls="--"),
               "IQL-eps": dict(color=plot_style.C[2], ls="-."), "IQL-Boltz": dict(color=plot_style.C[3], ls=":"),
               "hysteretic": dict(color=plot_style.C[6], ls=(0, (5, 1)))}
        for learner in LEARNERS:
            ax.plot(out[("climbing", learner)]["curve"], label=learner, **sty[learner])
        ax.axhline(11, color=plot_style.GREY, lw=1, ls=":")
        ax.set_xlabel("step")
        ax.set_ylabel("mean payoff of greedy joint action")
        ax.set_title("(b) Climbing game: learning curves")
        ax.set_ylim(-12, 12)
        ax.legend(fontsize=8, loc="lower right")

        ax = axes[2]
        q1 = out[("climbing", "IQL-eps")]["q1"]
        for a, ls in zip(range(3), ["-", "--", "-."]):
            ax.plot(q1[:, a], ls=ls, color=plot_style.C[a], label=f"$Q_1$({names[a]})")
        ax.set_xlabel("step")
        ax.set_ylabel("mean over runs")
        ax.set_title("(c) Climbing game, IQL-eps: agent 1's Q-values")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "cooperative_matrix_games.png"))
        plt.close(fig)
        print(f"figure written to {figdir}/cooperative_matrix_games.png")


if __name__ == "__main__":
    main()
