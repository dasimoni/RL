"""V-trace (Espeholt et al. 2018) on the slippery chain: what does it converge to?

Chapter 06, Section 14.4 (preview of IMPALA). State values are learned off-policy
from episodes of the uniform behaviour policy b (b(right) = 0.5; IMPALA writes mu) for the target
policy pi (pi(right) = 0.9). For each episode, with V held fixed, the V-trace
targets are computed backwards (IMPALA, Remark 1):

    v_s = V(x_s) + rho_s delta_s + gamma c_s (v_{s+1} - V(x_{s+1})),
    delta_s = r_s + gamma V(x_{s+1}) - V(x_s),
    rho_s = min(rho_bar, pi/b),   c_s = min(c_bar, pi/b),

and V(x_s) moves towards v_s (increments summed over the episode, step size alpha).
The claim to check: the fixed point is v of the policy
    pi_rho_bar(a|x) = min(rho_bar b(a|x), pi(a|x)) / sum_a' min(rho_bar b(a'|x), pi(a'|x)),
which equals pi only when rho_bar >= max pi/b (here 1.8); c_bar does not move the
fixed point, only the speed of contraction.

Outputs (full mode): figures/vtrace_fixed_point.png

Run:  python code/ch06_n_step_and_eligibility_traces/vtrace_fixed_point.py [--quick]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

import chain_mdp as cm

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def pi_rho_bar(pi, b, rho_bar):
    m = np.minimum(rho_bar * b, pi)
    tot = m.sum(axis=1, keepdims=True)
    return np.divide(m, tot, out=np.zeros_like(m), where=tot > 0)


def vtrace_targets(V, S, A, R, pi, b, rho_bar, c_bar, gamma=cm.GAMMA):
    T = len(R)
    ratio = pi[S[:-1], A] / b[S[:-1], A]
    rho = np.minimum(rho_bar, ratio)
    c = np.minimum(c_bar, ratio)
    v = np.zeros(T + 1)                       # v_T = V(terminal) = 0
    for s in range(T - 1, -1, -1):
        delta = R[s] + gamma * V[S[s + 1]] - V[S[s]]
        v[s] = V[S[s]] + rho[s] * delta + gamma * c[s] * (v[s + 1] - V[S[s + 1]])
    return v[:T]


def learn(rng, pi, b, rho_bar, c_bar, alpha, n_eps, n_avg):
    V = np.zeros(cm.N_TOTAL)
    V_avg = np.zeros(cm.N_TOTAL)
    for k in range(n_eps):
        S, A, R = cm.generate_episode(rng, b)
        v = vtrace_targets(V, S, A, R, pi, b, rho_bar, c_bar)
        dV = np.zeros_like(V)
        np.add.at(dV, S[:-1], alpha * (v - V[S[:-1]]))
        V += dV
        if k >= n_eps - n_avg:                # average the last n_avg iterates to remove noise
            V_avg += V / n_avg
    return V_avg


def rms(u, w):
    return float(np.sqrt(np.mean((u[1:cm.N + 1] - w[1:cm.N + 1]) ** 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    n_eps, n_avg, n_runs, alpha = (300, 100, 1, 0.05) if args.quick else (6000, 3000, 3, 0.02)
    pi, b = cm.policy(0.9), cm.policy(0.5)
    configs = [(2.0, 1.0), (1.0, 1.0), (1.0, 0.5), (0.5, 0.5)]
    print(f"V-trace on the slippery {cm.N}-state chain | seed={args.seed} pi(right)=0.9 b(right)=0.5 "
          f"gamma={cm.GAMMA} alpha={alpha} episodes={n_eps} (average of last {n_avg}) runs={n_runs}")
    print(f"max pi/b = {np.max(pi[1:-1] / b[1:-1]):.2f}")
    t0 = time.time()
    v_pi, v_b = cm.exact_v(pi), cm.exact_v(b)
    rng = np.random.default_rng(args.seed)
    results = {}
    print("\n  rho_bar  c_bar  pi_rho_bar(right)   RMS to v_pi   RMS to v_pi_rho_bar   RMS to v_b")
    for rho_bar, c_bar in configs:
        p = pi_rho_bar(pi, b, rho_bar)
        v_fix = cm.exact_v(p)
        V = np.mean([learn(rng, pi, b, rho_bar, c_bar, alpha, n_eps, n_avg) for _ in range(n_runs)], axis=0)
        results[(rho_bar, c_bar)] = (V, v_fix, p[1, 1])
        print(f"  {rho_bar:6.2f}  {c_bar:5.2f}   {p[1, 1]:10.4f}          {rms(V, v_pi):.4f}        "
              f"{rms(V, v_fix):.4f}              {rms(V, v_b):.4f}")
    print(f"\n  RMS(v_pi - v_b) = {rms(v_pi, v_b):.4f}")
    print(f"elapsed {time.time() - t0:.1f} s")

    if args.quick:
        return
    from plot_style import setup, C, MARKERS, GREY
    plt = setup()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(1, cm.N + 1)
    ax.plot(x, v_pi[1:-1], color=C[0], lw=2, label=r"$v_\pi$ (π(right) = 0.9)")
    ax.plot(x, v_b[1:-1], color=GREY, lw=1.5, ls=":", label=r"$v_b$ (behaviour b, 0.5)")
    styles = {(2.0, 1.0): (C[0], MARKERS[0]), (1.0, 1.0): (C[1], MARKERS[1]),
              (1.0, 0.5): (C[1], MARKERS[3]), (0.5, 0.5): (C[2], MARKERS[2])}
    drawn = set()
    for (rb, cb), (V, v_fix, p_right) in results.items():
        col, mk = styles[(rb, cb)]
        if rb < 1.8 and rb not in drawn:
            ax.plot(x, v_fix[1:-1], color=col, lw=1.2, ls="--",
                    label=rf"$v_{{\pi_{{\bar\rho}}}}$, $\bar\rho$={rb:g} (π$_{{\bar\rho}}$(right) = {p_right:.3f})")
            drawn.add(rb)
        ax.plot(x, V[1:-1], ls="none", color=col, marker=mk, ms=6, mfc="none" if cb < rb else col,
                label=rf"V-trace estimate, $\bar\rho$={rb:g}, $\bar c$={cb:g}")
    ax.set_xlabel("state")
    ax.set_ylabel("value")
    ax.set_title(r"V-trace converges to $v_{\pi_{\bar\rho}}$, not $v_\pi$, when $\bar\rho < \max\,\pi/b$")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, "vtrace_fixed_point.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
