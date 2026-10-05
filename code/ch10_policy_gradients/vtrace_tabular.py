"""V-trace (IMPALA's off-policy value target) on a small MDP: what does it converge to?

Chapter 10, Section 14 (Espeholt et al., 2018).

Data come from a behavior policy b; we want the value of a target policy pi.  V-trace targets
(Eq. 10.34), written with Sutton & Barto's reward index:

    v_s = V(S_s) + sum_{t >= s} gamma^{t-s} (c_s ... c_{t-1}) rho_t (R_{t+1} + gamma V(S_{t+1}) - V(S_t))
    rho_t = min(rho_bar, pi(A_t|S_t) / b(A_t|S_t)),   c_i = min(c_bar, pi(A_i|S_i) / b(A_i|S_i)).

Claim (Espeholt et al., 2018, Theorem 1): the fixed point is the value function of the policy

    pi_rho_bar(a|s) = min(rho_bar b(a|s), pi(a|s)) / sum_a' min(rho_bar b(a'|s), pi(a'|s)),

which equals pi when rho_bar = infinity and moves towards b as rho_bar shrinks; c_bar affects only
the speed of convergence.  We run tabular V-trace learning from behavior data for several
(rho_bar, c_bar) and compare the result with v_pi, v_b and v_{pi_rho_bar} computed exactly.

Outputs (full mode): figures/vtrace_tabular.png
Run:  python code/ch10_policy_gradients/vtrace_tabular.py [--quick]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from tabular_pg import random_mdp

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def pi_rho_bar(pi, b, rho_bar):
    m = np.minimum(rho_bar * b, pi)
    return m / m.sum(1, keepdims=True)


def vtrace_targets(V, S, A, R, alive, pi, b, gamma, rho_bar, c_bar):
    """V-trace targets for a batch of episodes (arrays (T, M)), by the backward recursion
    v_s - V(S_s) = rho_s delta_s + gamma c_s (v_{s+1} - V(S_{s+1}))."""
    T, M = S.shape
    ratio = pi[S, A] / b[S, A]
    rho = np.minimum(rho_bar, ratio)
    c = np.minimum(c_bar, ratio)
    V_s = V[S]
    V_next = np.zeros((T, M))
    V_next[:-1] = np.where(alive[1:], V[S[1:]], 0.0)          # V(terminal) = 0
    delta = rho * (R + gamma * V_next - V_s)
    acc = np.zeros(M)                                           # v_{s+1} - V(S_{s+1})
    out = np.zeros((T, M))
    for t in range(T - 1, -1, -1):
        nxt_alive = alive[t + 1] if t + 1 < T else np.zeros(M, dtype=bool)
        acc = delta[t] + gamma * c[t] * np.where(nxt_alive, acc, 0.0)
        out[t] = V_s[t] + acc
    return out


def learn(mdp, pi, b, rho_bar, c_bar, n_iters, batch, rng, alpha=0.5):
    V = np.zeros(mdp.n_s)
    errs = []
    for it in range(n_iters):
        S, A, R, alive = mdp.sample_episodes(b, batch, rng)
        vt = vtrace_targets(V, S, A, R, alive, pi, b, mdp.gamma, rho_bar, c_bar)
        # average the target - V over all visits of each state in the batch, then step
        num = np.bincount(S[alive], weights=(vt - V[S])[alive], minlength=mdp.n_s)
        cnt = np.bincount(S[alive], minlength=mdp.n_s)
        step = alpha / (1 + it / 50)                                # decaying step size
        V += step * num / np.maximum(cnt, 1)
        errs.append(V.copy())
    return V, np.array(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n_iters, batch = (60, 500) if args.quick else (400, 2000)
    print(f"seed={args.seed}  iterations={n_iters}  episodes/iteration={batch}")
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    mdp = random_mdp(5, 3, gamma=0.9, rng=rng, reward_mean=0.0, reward_sd=0.5)
    # target pi: fairly greedy; behavior b: close to uniform
    pi = np.exp(2.0 * rng.standard_normal((5, 3)))
    pi /= pi.sum(1, keepdims=True)
    b = np.exp(0.3 * rng.standard_normal((5, 3)))
    b /= b.sum(1, keepdims=True)
    v_pi, _ = mdp.evaluate(pi)
    v_b, _ = mdp.evaluate(b)
    print("max importance ratio pi/b =", np.round((pi / b).max(), 2))
    print("v_pi =", np.round(v_pi, 3))
    print("v_b  =", np.round(v_b, 3))

    settings = [(np.inf, 1.0), (2.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.5, 0.5)]
    results = {}
    print("\n rho_bar  c_bar | max|V - v_pi_rho_bar| | max|V - v_pi| | max|V - v_b| | "
          "max|v_pi_rho_bar - v_pi|")
    for rho_bar, c_bar in settings:
        V, traj = learn(mdp, pi, b, rho_bar, c_bar, n_iters, batch, rng)
        v_star, _ = mdp.evaluate(pi_rho_bar(pi, b, rho_bar))
        results[(rho_bar, c_bar)] = (traj, v_star)
        print(f" {rho_bar:7.2f} {c_bar:6.2f} | {np.abs(V - v_star).max():21.4f} | "
              f"{np.abs(V - v_pi).max():13.4f} | {np.abs(V - v_b).max():12.4f} | "
              f"{np.abs(v_star - v_pi).max():.4f}")
    print(f"Time {time.time() - t0:.1f}s")

    if not args.quick:
        from plot_style import setup, C
        plt = setup()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
        styles = ["-", "--", "-.", ":", "-"]
        for j, ((rb, cb), (traj, v_star)) in enumerate(results.items()):
            rb_txt = r"$\infty$" if np.isinf(rb) else f"{rb:g}"
            lab = f"$\\bar\\rho$={rb_txt}, $\\bar c$={cb:g}"
            axes[0].semilogy(np.abs(traj - v_star).max(1), color=C[j], ls=styles[j], label=lab)
            axes[1].semilogy(np.abs(traj - v_pi).max(1), color=C[j], ls=styles[j], label=lab)
        axes[0].set_title(r"distance to $v_{\pi_{\bar\rho}}$ (the predicted fixed point)")
        axes[1].set_title(r"distance to $v_\pi$ (the target policy's value)")
        for ax in axes:
            ax.set_xlabel("iteration (batch of behavior episodes)")
            ax.set_ylabel("max-norm error")
        # one shared legend below the panels, so that no curve is hidden behind it
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(os.path.join(FIG_DIR, "vtrace_tabular.png"))
        plt.close(fig)
        print(f"figure written to {FIG_DIR}/vtrace_tabular.png")


if __name__ == "__main__":
    main()
