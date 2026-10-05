"""The short corridor with aliased states: why stochastic policies, and REINFORCE with/without baseline.

Chapter 10, Sections 1, 5 and 6 (Sutton & Barto, 2018, Example 13.1; Chapter 01, Exercise 12).

Three non-terminal states that all LOOK THE SAME to the agent; in the middle state the actions
are reversed.  Reward -1 per step, gamma = 1.  A policy can only choose one probability
p = pi(right) for all states, here parameterized as p = sigmoid(theta) (one feature,
x(s,right) = 1, x(s,left) = 0; Section 2.1).

What the script does
  1. Exact J(p) = -2(2-p)/(p(1-p)) on a grid; optimum p* = 2 - sqrt(2), J* = -11.66; the two
     epsilon-greedy policies (epsilon = 0.1) a value-based method could end up with.
  2. REINFORCE (Algorithm 10.1) from p0 = 0.05 for three step sizes, 100 independent runs each.
  3. REINFORCE with a learned baseline (Algorithm 10.2).  The baseline is a single number w,
     because the agent cannot tell the states apart either.
  4. The same, but with the whole episode's updates SUMMED (theta, w fixed) and applied once
     at the end, as deep-RL code does: the baseline's effective step becomes alpha_w * T.

Outputs (full mode): figures/short_corridor.png

Run:  python code/ch10_policy_gradients/short_corridor.py [--quick]
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time

import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
RIGHT, LEFT = 0, 1


def J_exact(p):
    """Value of the start state for pi(right) = p (Chapter 01, Exercise 12)."""
    return -2.0 * (2.0 - p) / (p * (1.0 - p))


def step(s, a):
    """Corridor dynamics. Returns next state (3 = terminal)."""
    if s == 0:
        return 1 if a == RIGHT else 0
    if s == 1:                       # reversed
        return 0 if a == RIGHT else 2
    return 3 if a == RIGHT else 1    # s == 2


def run_episode(p, rng, max_steps=1000):
    """Return the list of actions (states are irrelevant to the aliased agent).

    Episodes are cut off after max_steps; this only happens when a run has diverged to a
    (near-)deterministic policy, which can loop forever (J -> -infinity at p = 0 and p = 1)."""
    s, actions = 0, []
    while s != 3:
        a = RIGHT if rng.random() < p else LEFT
        actions.append(a)
        s = step(s, a)
        if len(actions) >= max_steps:
            break
    return actions


def reinforce(n_episodes, alpha_theta, alpha_w, p0, seed, use_baseline, summed=False):
    """REINFORCE (Alg. 10.1) or REINFORCE with baseline (Alg. 10.2) on the aliased corridor.

    With one feature, psi(right) = d/dtheta log sigmoid(theta) = 1 - p and psi(left) = -p.
    Rewards are -1 per step and gamma = 1, so G_t = -(T - t).
    As in Sutton & Barto's boxed algorithms, theta and w are updated at every step t of the
    (already generated) episode, so later steps use the slightly updated parameters.
    summed=True instead sums the whole episode's updates with theta and w FIXED and applies
    them once.  The baseline update is then w += alpha_w * sum_t (G_t - w), an effective step
    of alpha_w * T on the regression of w, which is unstable (|1 - alpha_w T| > 1) once
    T > 2 / alpha_w = 128 steps with alpha_w = 2^-6.
    """
    rng = random.Random(seed)
    theta = math.log(p0 / (1.0 - p0))
    w = 0.0                                   # baseline v_hat(s, w) = w for every (aliased) state
    returns = np.empty(n_episodes)
    probs = np.empty(n_episodes)
    for ep in range(n_episodes):
        p = 1.0 / (1.0 + math.exp(-theta))
        acts = run_episode(p, rng)
        T = len(acts)
        returns[ep] = -T
        probs[ep] = p
        if summed:                            # one update per episode, parameters fixed
            d_theta = d_w = 0.0
            for t, a in enumerate(acts):
                G = -(T - t)
                psi = (1.0 - p) if a == RIGHT else -p
                delta = G - w if use_baseline else G
                d_w += delta
                d_theta += delta * psi
            if use_baseline:
                w += alpha_w * d_w
            theta = max(-30.0, min(30.0, theta + alpha_theta * d_theta))
            continue
        for t, a in enumerate(acts):
            G = -(T - t)                      # reward-to-go from step t
            p = 1.0 / (1.0 + math.exp(-theta))
            psi = (1.0 - p) if a == RIGHT else -p
            if use_baseline:
                delta = G - w
                w += alpha_w * delta          # semi-gradient step on (G - w)^2 / 2
                theta += alpha_theta * delta * psi
            else:
                theta += alpha_theta * G * psi
            theta = max(-30.0, min(30.0, theta))   # keep exp() finite if a run diverges
    reinforce.last_w = w                      # final baseline weight (diagnostic)
    return returns, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n_runs, n_eps = (10, 300) if args.quick else (100, 1000)
    p0 = 0.05
    alphas = [2 ** -12, 2 ** -13, 2 ** -14]
    bl = dict(alpha_theta=2 ** -9, alpha_w=2 ** -6)
    print(f"seed={args.seed}  runs={n_runs}  episodes={n_eps}  p0={p0}  "
          f"REINFORCE alphas={['2^%d' % round(math.log2(a)) for a in alphas]}  "
          f"with baseline: alpha_theta=2^-9, alpha_w=2^-6")
    t0 = time.time()

    p_star = 2 - math.sqrt(2)
    print(f"\nExact: p* = 2 - sqrt(2) = {p_star:.4f}, J(p*) = {J_exact(p_star):.3f}; "
          f"eps-greedy (eps=0.1): J(0.95) = {J_exact(0.95):.1f}, J(0.05) = {J_exact(0.05):.1f}")

    curves, finals = {}, {}
    for a in alphas:
        R = np.array([reinforce(n_eps, a, 0.0, p0, args.seed * 1000 + i, False)[0] for i in range(n_runs)])
        curves[f"REINFORCE, alpha=2^{round(math.log2(a))}"] = R.mean(0)
    Rb, Pb = zip(*[reinforce(n_eps, bl["alpha_theta"], bl["alpha_w"], p0, args.seed * 1000 + i, True)
                   for i in range(n_runs)])
    Rb = np.array(Rb)
    curves["REINFORCE with baseline"] = Rb.mean(0)
    # the best plain step size, rerun at the baseline's step size for a like-for-like comparison
    R9, P9 = zip(*[reinforce(n_eps, 2 ** -9, 0.0, p0, args.seed * 1000 + i, False) for i in range(n_runs)])
    R9, P9 = np.array(R9), np.array(P9)
    curves["REINFORCE, alpha=2^-9 (no baseline)"] = R9.mean(0)
    stuck = np.mean((P9[:, -1] < 0.01) | (P9[:, -1] > 0.99))
    Pb = np.array(Pb)
    # with baseline, but the episode's updates summed and applied once (Pitfall 7)
    Rs, Ps, Ws = [], [], []
    for i in range(n_runs):
        r_, p_ = reinforce(n_eps, bl["alpha_theta"], bl["alpha_w"], p0, args.seed * 1000 + i, True, summed=True)
        Rs.append(r_)
        Ps.append(p_)
        Ws.append(reinforce.last_w)
    Rs, Ps, Ws = np.array(Rs), np.array(Ps), np.array(Ws)
    div = ~np.isfinite(Ws) | (np.abs(Ws) > 1e3) | (Ps[:, -1] < 0.01) | (Ps[:, -1] > 0.99)

    last = max(1, n_eps // 10)
    print(f"\nMean return over runs (first 50 episodes | last {last} episodes):")
    for k, c in curves.items():
        print(f"  {k:38s}  {c[:50].mean():8.1f} | {c[-last:].mean():7.2f}")
    print(f"  (optimum {J_exact(p_star):.2f})")
    print(f"with baseline: final p = {Pb[:, -1].mean():.3f} +- {Pb[:, -1].std():.3f} (p* = {p_star:.3f}); "
          f"range over runs [{Pb[:, -1].min():.3f}, {Pb[:, -1].max():.3f}]")
    print(f"no baseline, alpha=2^-9: {stuck:.0%} of runs ended with p < 0.01 or p > 0.99 "
          f"(episodes capped at 1000 steps)")
    ok = ~div
    print(f"with baseline, per-episode SUMMED updates (same step sizes): {div.sum()}/{n_runs} runs diverged "
          f"(|w| > 1000 or final p < 0.01 or > 0.99); the others: last-{last} return "
          f"{Rs[ok][:, -last:].mean() if ok.any() else float('nan'):.2f}, final p "
          f"{Ps[ok][:, -1].mean() if ok.any() else float('nan'):.3f}")
    print(f"\nTime {time.time() - t0:.1f}s")

    if not args.quick:
        from plot_style import setup, C, GREY
        plt = setup()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        ax = axes[0]
        ps = np.linspace(0.01, 0.99, 400)
        ax.plot(ps, J_exact(ps), color=C[0])
        ax.axvline(p_star, color=GREY, ls=":", lw=1)
        ax.plot([p_star], [J_exact(p_star)], "o", color=C[0])
        ax.annotate(f"optimal stochastic policy\np = {p_star:.3f}, J = {J_exact(p_star):.2f}",
                    (p_star, J_exact(p_star)), xytext=(0.33, -35), fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=GREY))
        for pe, lab in [(0.05, "$\\varepsilon$-greedy left"), (0.95, "$\\varepsilon$-greedy right")]:
            ax.plot([pe], [J_exact(pe)], "s", color=C[1])
            ax.annotate(f"{lab}\nJ = {J_exact(pe):.1f}", (pe, J_exact(pe)),
                        xytext=(pe + (0.06 if pe < 0.5 else -0.3), J_exact(pe) - 12), fontsize=9)
        ax.set_ylim(-100, 0)
        ax.set_xlabel("probability of 'right', p (same in every state)")
        ax.set_ylabel("J(p) = value of the start state")
        ax.set_title("Short corridor: the best policy is stochastic")
        ax = axes[1]
        x = np.arange(1, n_eps + 1)
        styles = ["-", "--", ":", "-", "-."]
        for j, (k, c) in enumerate(curves.items()):
            if "2^-9" in k:
                continue                      # diverged: off the scale, reported in the text box
            sm = np.convolve(c, np.ones(10) / 10, mode="valid")
            lab = k.replace("alpha=2^", "$\\alpha=2^{").replace("REINFORCE, $", "REINFORCE, $")
            lab = lab + "}$" if "$" in lab else lab
            if "baseline" in lab:
                lab += " ($\\alpha^{\\theta}=2^{-9}$, $\\alpha^{w}=2^{-6}$)"
            ax.plot(x[9:], sm, label=lab, color=C[j], ls=styles[j], lw=1.6)
        ax.text(0.97, 0.42, "no baseline at the baseline run's\n$\\alpha=2^{-9}$: diverges "
                f"({stuck:.0%} of runs\nend near-deterministic), mean\nreturn {R9[:, -last:].mean():.0f} (off scale)",
                transform=ax.transAxes, ha="right", fontsize=8, color=C[4])
        ax.axhline(J_exact(p_star), color=GREY, ls=":", lw=1, label="optimum $-11.66$")
        ax.set_ylim(-90, -5)
        ax.set_xlabel("episode")
        ax.set_ylabel(f"total reward per episode (mean of {n_runs} runs)")
        ax.set_title("REINFORCE from p = 0.05, with and without baseline")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "short_corridor.png"))
        plt.close(fig)
        print(f"figure written to {FIG_DIR}/short_corridor.png")


if __name__ == "__main__":
    main()
