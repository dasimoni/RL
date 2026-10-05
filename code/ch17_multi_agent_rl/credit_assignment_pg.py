"""Multi-agent credit assignment for policy gradients: why COMA's counterfactual
baseline helps (Chapter 17, section 8).

N agents each choose a_i in {0, 1} with a Bernoulli policy pi_i(1) = sigmoid(theta_i).
The team receives one shared reward
        R = sum_j a_j + sigma * noise,       noise ~ N(0, 1),
so the expected reward is Q(a) = sum_j a_j and every agent should learn a_i = 1.
The score function of agent i is d log pi_i(a_i) / d theta_i = a_i - p_i.

Four unbiased estimators of agent i's policy gradient (all have mean p_i (1 - p_i)):
  REINFORCE          g = R (a_i - p_i)
  + value baseline   g = (R - V)(a_i - p_i),           V = E[R] (the best state baseline)
  central critic     g = (Q(a) - V)(a_i - p_i)         (critic removes reward noise only)
  COMA               g = (Q(a) - sum_{a'} pi_i(a') Q(a', a_{-i})) (a_i - p_i)
                     (counterfactual baseline also removes the other agents' noise)
The critic is exact here so that only the estimator differs (in COMA it is learned).

Part 1: exact and Monte Carlo variance at p = 1/2 for N = 2 ... 64 (Eq. 8.5).
Part 2: learning with each estimator (same step size 0.5, 500 runs per setting).

Run:  python code/ch17_multi_agent_rl/credit_assignment_pg.py [--quick]
"""
import argparse
import os
import time

import numpy as np

ESTIMATORS = ["REINFORCE", "+ value baseline", "central critic", "COMA"]


def gradient_samples(est, a, p, noise, sigma):
    """Agent-wise gradient estimates for a batch: a, p have shape [runs, N]."""
    score = a - p
    Q = a.sum(1, keepdims=True)                       # exact critic: expected team reward
    R = Q + sigma * noise                             # sampled team reward
    V = p.sum(1, keepdims=True)                       # E[R] under the current policies
    if est == "REINFORCE":
        adv = np.broadcast_to(R, a.shape)
    elif est == "+ value baseline":
        adv = np.broadcast_to(R - V, a.shape)
    elif est == "central critic":
        adv = np.broadcast_to(Q - V, a.shape)
    else:
        # counterfactual baseline: replace a_i by every a_i' weighted by pi_i(a_i')
        Q_if0 = Q - a                                 # Q(0, a_{-i}) for each i
        Q_if1 = Q - a + 1.0                           # Q(1, a_{-i})
        baseline = (1 - p) * Q_if0 + p * Q_if1
        adv = Q - baseline
    return adv * score


def exact_variance(est, N, sigma):
    """Closed-form variance at p = 1/2 (derived in section 8.3, Eq. 8.5)."""
    if est == "REINFORCE":
        # g = (N/2 + X) s with X = sum_j s_j + sigma*eps, s_j = a_j - 1/2, s^2 = 1/4
        return 0.25 * (N ** 2 / 4 + N / 4 + sigma ** 2) - 1 / 16
    if est == "+ value baseline":
        return (N - 1) / 16 + sigma ** 2 / 4
    if est == "central critic":
        return (N - 1) / 16
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Credit assignment: counterfactual baselines")
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    sigma = 1.0
    Ns = [2, 4, 8, 16, 32, 64]
    n_mc = 20_000 if args.quick else 200_000
    runs = 50 if args.quick else 500
    K = 150 if args.quick else 400
    alpha = 0.5
    print(f"credit_assignment_pg.py | seed={args.seed} | reward noise sigma={sigma} | "
          f"MC samples={n_mc} | learning: runs={runs}, steps={K}, step size={alpha} | quick={args.quick}")
    t0 = time.time()

    # ---------------- Part 1: variance of agent 1's gradient estimate at p = 1/2 -------
    print("\nPart 1: agent 1's gradient estimate at p = 1/2 (true gradient = 0.25 for every N)")
    print(f"{'N':>4s} " + " ".join(f"{e:>26s}" for e in ESTIMATORS))
    var_mc = {e: [] for e in ESTIMATORS}
    for N in Ns:
        p = np.full((n_mc, N), 0.5)
        a = (rng.random((n_mc, N)) < p).astype(float)
        noise = rng.standard_normal((n_mc, 1))
        row = []
        for e in ESTIMATORS:
            g = gradient_samples(e, a, p, noise, sigma)[:, 0]
            var_mc[e].append(g.var())
            row.append(f"mean {g.mean():.3f} var {g.var():6.3f} ({exact_variance(e, N, sigma):6.3f})")
        print(f"{N:4d} " + " ".join(f"{r:>26s}" for r in row))
    print("(numbers in brackets: exact variance from Eq. 8.5)")

    # ---------------- Part 2: learning -------------------------------------------------
    print(f"\nPart 2: mean P(a_i = 1) over agents and runs after {K} updates "
          f"(t90 = number of updates after which it first exceeds 0.9)")
    curves = {}
    locked = {}
    print(f"{'N':>4s} " + " ".join(f"{e:>22s}" for e in ESTIMATORS))
    for N in Ns:
        row = []
        for e in ESTIMATORS:
            rng_l = np.random.default_rng(args.seed + 10_000 + N)   # common random numbers
            theta = np.zeros((runs, N))
            curve = np.empty(K)
            for k in range(K):
                p = 1 / (1 + np.exp(-theta))
                a = (rng_l.random((runs, N)) < p).astype(float)
                noise = rng_l.standard_normal((runs, 1))
                theta += alpha * gradient_samples(e, a, p, noise, sigma)
                curve[k] = (1 / (1 + np.exp(-theta))).mean()
            curves[(N, e)] = curve
            p_final = 1 / (1 + np.exp(-theta))
            locked[(N, e)] = ((p_final < 0.01).mean(), (p_final > 0.99).mean())
            # curve[k] is measured after k + 1 updates, so the number of updates is index + 1
            hit = int(np.argmax(curve > 0.9)) + 1 if (curve > 0.9).any() else -1
            row.append(f"{curve[-1]:.3f} (t90={hit if hit > 0 else 'never'})")
        print(f"{N:4d} " + " ".join(f"{r:>22s}" for r in row))
    lo, hi = locked[(Ns[-1], "REINFORCE")]
    print(f"REINFORCE without baseline, N = {Ns[-1]}: {100 * lo:.1f}% of agents end with P(a_i=1) < 0.01 "
          f"and {100 * hi:.1f}% with P(a_i=1) > 0.99 (every reward is positive, so the first action taken "
          f"is reinforced)")
    print(f"\ntotal time {time.time() - t0:.1f} s")

    if not args.quick:
        import plot_style
        plt = plot_style.setup()
        figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
        os.makedirs(figdir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        st = {"REINFORCE": dict(color=plot_style.C[0], ls="-", marker="o"),
              "+ value baseline": dict(color=plot_style.C[1], ls="--", marker="s"),
              "central critic": dict(color=plot_style.C[2], ls="-.", marker="^"),
              "COMA": dict(color=plot_style.C[6], ls=":", marker="D")}
        ax = axes[0]
        for e in ESTIMATORS[:3]:
            ax.loglog(Ns, var_mc[e], label=e, **st[e])
        ax.text(10, 0.07, "COMA: variance exactly 0 at p = 1/2\n(cannot be drawn on a log axis)", fontsize=8)
        from matplotlib.ticker import NullFormatter
        ax.set_xticks(Ns)
        ax.set_xticklabels([str(n) for n in Ns])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("number of agents N")
        ax.set_ylabel("variance of agent 1's gradient estimate")
        ax.set_title("(a) Variance at p = 1/2 (true gradient 0.25)")
        ax.legend(fontsize=8)
        for ax, N in zip(axes[1:], [8, 64]):
            for e in ESTIMATORS:
                s = dict(st[e]); s.pop("marker")
                ax.plot(np.arange(1, K + 1), curves[(N, e)], label=e, **s)
            ax.set_xlabel("number of updates")
            ax.set_ylabel("mean P(a_i = 1)")
            ax.set_ylim(0.4, 1.0)
            ax.set_title(f"({'b' if N == 8 else 'c'}) Learning, N = {N} agents ({runs} runs)")
            ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "credit_assignment.png"))
        plt.close(fig)
        print(f"figure written to {figdir}/credit_assignment.png")


if __name__ == "__main__":
    main()
