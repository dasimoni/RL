"""Dropping gamma^t from the policy gradient: a small MDP where it converges to the WORST policy.

Chapter 10, Section 12 (in the spirit of Nota & Thomas, 2020, "Is the policy gradient a gradient?").

The exact discounted policy gradient (Eqs. 10.13, 10.15) weights the update at time t by gamma^t.
Almost every implementation drops that factor and uses  sum_t G_t grad log pi(A_t|S_t).
Its expectation is  sum_s eta_1(s) sum_a q_gamma(s,a) grad pi(a|s):  undiscounted state weights
with discounted action values -- the gradient of neither the discounted nor the undiscounted
objective.

The MDP (gamma = 0.5).  All decision states look the same, so pi(R|s) = p = sigmoid(theta):

    s0 --R: +1--> s1          s0 --L: 0--> s1
    s1 --L: +a = 2.5, terminate
    s1 --R: 0--> c1 --0--> c2 --0--> c3 --(+B = 8)--> terminate   (chain: both actions equal)

  * discounted objective J_gamma(p) = p + gamma[(1-p) a + p gamma^3 B]:  slope +0.25 -> best p = 1
  * undiscounted objective J_1(p)   = p + (1-p) a + p B:                 slope +6.5  -> best p = 1
  * expected no-gamma^t update  ~  p(1-p)[1 + (gamma^3 B - a)] = -0.5 p(1-p)       -> drives p to 0,
    the policy that is WORST for both objectives.

The script checks those three directions exactly and then runs sampled REINFORCE with and
without the gamma^t factor.

Outputs (full mode): figures/discount_bias.png
Run:  python code/ch10_policy_gradients/discount_bias.py [--quick]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from tabular_pg import LinearSoftmax, TabularMDP, exact_gradient

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
GAMMA, A_REW, B_REW = 0.5, 2.5, 8.0


def build():
    # states: 0 = s0, 1 = s1, 2 = c1, 3 = c2, 4 = c3;  actions: 0 = R, 1 = L
    P = np.zeros((5, 2, 5))
    r = np.zeros((5, 2))
    P[0, :, 1] = 1.0
    r[0, 0] = 1.0                       # s0: R pays +1, both actions lead to s1
    P[1, 0, 2] = 1.0                    # s1 --R--> c1 (reward 0)
    r[1, 1] = A_REW                     # s1 --L--> terminate with +a
    P[2, :, 3] = 1.0
    P[3, :, 4] = 1.0
    r[4, :] = B_REW                     # c3 --> terminate with +B
    mdp = TabularMDP(P, r, d0=np.array([1.0, 0, 0, 0, 0]), gamma=GAMMA)
    X = np.zeros((5, 2, 1))
    X[:, 0, 0] = 1.0                    # one shared feature: pi(R|s) = sigmoid(theta) everywhere
    return mdp, LinearSoftmax(X)


def run_reinforce(n_runs, n_episodes, alpha, seed, keep_gamma_t, theta0=0.0):
    """Sampled REINFORCE (Alg. 10.1) for n_runs independent learners at once (vectorized over runs).

    One update per episode, with theta fixed during the episode.  An episode is
    s0 -> s1 -> (terminate after L) or (c1 -> c2 -> c3 -> terminate after R).
    The chain states' actions do not matter, but REINFORCE does not know that: their score
    terms are included, exactly as a real implementation would include them.
    """
    rng = np.random.default_rng(seed)
    g = GAMMA
    theta = np.full(n_runs, theta0)
    ps = np.empty((n_runs, n_episodes))
    for ep in range(n_episodes):
        p = 1.0 / (1.0 + np.exp(-theta))
        ps[:, ep] = p
        right = rng.random((n_runs, 5)) < p[:, None]          # actions at t = 0..4 (R = True)
        psi = np.where(right, 1.0 - p[:, None], -p[:, None])   # d/dtheta log pi(A_t)
        r0 = right[:, 0] * 1.0
        long = right[:, 1]                                      # took R at s1: the 5-step episode
        # discounted reward-to-go G_t for t = 0..4 (zero after the episode has ended)
        G = np.zeros((n_runs, 5))
        G[:, 4] = np.where(long, B_REW, 0.0)
        G[:, 3] = np.where(long, g * B_REW, 0.0)
        G[:, 2] = np.where(long, g ** 2 * B_REW, 0.0)
        G[:, 1] = np.where(long, g ** 3 * B_REW, A_REW)
        G[:, 0] = r0 + g * G[:, 1]
        alive = np.ones((n_runs, 5), dtype=bool)
        alive[:, 2:] = long[:, None]
        w = g ** np.arange(5) if keep_gamma_t else np.ones(5)
        theta = theta + alpha * (w * G * psi * alive).sum(1)
    return ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n_runs, n_eps, alpha = (20, 2000, 0.05) if args.quick else (100, 20000, 0.05)
    print(f"seed={args.seed}  gamma={GAMMA}  a={A_REW}  B={B_REW}  runs={n_runs}  "
          f"episodes={n_eps}  alpha={alpha}  theta0=0 (p0 = 0.5)")
    t0 = time.time()
    mdp, pol = build()
    mdp1 = TabularMDP(mdp.P, mdp.r, mdp.d0, gamma=1.0)

    print("\n   p   | J_gamma | J_1  | dJ_gamma/dtheta | dJ_1/dtheta | E[no-gamma^t update]")
    grid = np.linspace(-4, 4, 161)
    rows = []
    for th in grid:
        theta = np.array([th])
        pi = pol.probs(theta)
        rows.append((pi[0, 0], mdp.J(pi), mdp1.J(pi), exact_gradient(mdp, pol, theta)[0],
                     exact_gradient(mdp1, pol, theta)[0],
                     exact_gradient(mdp, pol, theta, visit_gamma=1.0)[0]))
    rows = np.array(rows)
    for th in (-2.0, 0.0, 2.0):
        i = np.argmin(np.abs(grid - th))
        p, Jg, J1, gg, g1, gb = rows[i]
        print(f" {p:.3f} | {Jg:7.3f} | {J1:4.2f} | {gg:15.4f} | {g1:11.4f} | {gb:20.4f}")
    p = rows[:, 0]
    print("closed forms: dJ_gamma/dtheta = 0.25 p(1-p), dJ_1/dtheta = 6.5 p(1-p), "
          "no-gamma = -0.5 p(1-p);  max deviations:",
          f"{np.abs(rows[:, 3] - 0.25 * p * (1 - p)).max():.1e},",
          f"{np.abs(rows[:, 4] - 6.5 * p * (1 - p)).max():.1e},",
          f"{np.abs(rows[:, 5] + 0.5 * p * (1 - p)).max():.1e}")

    runs = {}
    for keep in (True, False):
        runs[keep] = run_reinforce(n_runs, n_eps, alpha, args.seed + (0 if keep else 1), keep)
    print("\nSampled REINFORCE, final p = pi(R) (mean over runs, [min, max]):")
    for keep, lab in ((True, "with gamma^t (correct)"), (False, "without gamma^t (common)")):
        fin = runs[keep][:, -1]
        Jg = np.mean([mdp.J(np.array([[f, 1 - f]] * 5)) for f in fin])
        J1 = np.mean([mdp1.J(np.array([[f, 1 - f]] * 5)) for f in fin])
        print(f"  {lab:26s} p = {fin.mean():.3f} [{fin.min():.3f}, {fin.max():.3f}]   "
              f"J_gamma = {Jg:.3f}   J_1 = {J1:.2f}   runs with p < 0.5: {(fin < 0.5).sum()}/{len(fin)}")
    print(f"  (best: p = 1, J_gamma = {1 + GAMMA ** 4 * B_REW:.3f}, J_1 = {1 + B_REW:.2f};  "
          f"worst: p = 0, J_gamma = {GAMMA * A_REW:.3f}, J_1 = {A_REW:.2f})")
    print(f"Time {time.time() - t0:.1f}s")

    if not args.quick:
        from plot_style import setup, C, GREY
        plt = setup()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
        ax = axes[0]
        ax.plot(p, rows[:, 3], color=C[0], label=r"$dJ_\gamma/d\theta$ (true discounted gradient)")
        ax.plot(p, rows[:, 4] / 10, color=C[2], ls="--", label=r"$dJ_1/d\theta$ (undiscounted) $\div 10$")
        ax.plot(p, rows[:, 5], color=C[7], ls="-.", label=r"E[update] without $\gamma^t$")
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_xlabel(r"$p=\pi(R)$")
        ax.set_ylabel("exact expected update direction")
        ax.set_title(r"Dropping $\gamma^t$ flips the sign of the update")
        ax.legend(fontsize=8)
        ax = axes[1]
        x = np.arange(1, n_eps + 1)
        for j, (keep, lab) in enumerate(((True, r"REINFORCE with $\gamma^t$"),
                                         (False, r"REINFORCE without $\gamma^t$"))):
            R = runs[keep]
            col = C[0] if keep else C[7]
            ax.plot(x, R.mean(0), color=col, label=lab + f" (mean of {n_runs} runs)",
                    ls="-" if keep else "-.")
            ax.fill_between(x, np.quantile(R, 0.1, 0), np.quantile(R, 0.9, 0), color=col, alpha=0.15)
        ax.set_ylim(0, 1)
        ax.set_xlabel("episode")
        ax.set_ylabel(r"$p=\pi(R)$  (shaded: 10-90% of runs)")
        ax.set_title("Sampled REINFORCE converges to opposite policies")
        ax.legend(fontsize=8, loc="center right")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "discount_bias.png"))
        plt.close(fig)
        print(f"figure written to {FIG_DIR}/discount_bias.png")


if __name__ == "__main__":
    main()
