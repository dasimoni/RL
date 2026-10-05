"""Emergent communication in the Lewis signalling game (Chapter 17, section 11).

A sender observes a state s (uniform over N states) and sends a message m from a
vocabulary of size M; a receiver sees only m and chooses an action. Both receive
reward 1 if the action equals the state, 0 otherwise. Nothing in the setup gives the
messages a meaning: any meaning must emerge from learning.

Both agents are independent softmax policies trained with REINFORCE (Chapter 10) and a
running-average reward baseline, from the shared reward only. A *signalling system*
(a bijection between states and messages, decoded correctly) earns 1. *Partial pooling*
equilibria, where the sender uses one message for several states, earn less and are
stable local optima that gradient learning can get stuck in.

Run:  python code/ch17_multi_agent_rl/lewis_signaling.py [--quick]
"""
import argparse
import os
import time

import numpy as np


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def expected_success(theta_s, theta_r):
    """Exact probability of success: (1/N) sum_s sum_m pi_S(m|s) pi_R(s|m), per run."""
    ps = softmax(theta_s)                         # [R, N, M]
    pr = softmax(theta_r)                         # [R, M, N]
    N = ps.shape[1]
    return np.einsum("rsm,rms->r", ps, pr) / N


def greedy_success(theta_s, theta_r):
    """Success of the greedy protocol: sender sends argmax_m, receiver answers argmax_a.
    Takes values k/N: k = number of states that are communicated correctly."""
    m = np.argmax(theta_s, -1)                    # [R, N]
    a = np.take_along_axis(np.argmax(theta_r, -1), m, axis=1)   # receiver's answer to m(s)
    return (a == np.arange(theta_s.shape[1])[None, :]).mean(1)


def train(N, M, runs, steps, rng, lr=0.5, baseline_lr=0.01):
    theta_s = np.zeros((runs, N, M))
    theta_r = np.zeros((runs, M, N))
    b = np.zeros(runs)                            # running-average baseline
    idx = np.arange(runs)
    curve = []
    for t in range(steps):
        s = rng.integers(N, size=runs)
        ps = softmax(theta_s[idx, s])             # [R, M]
        m = (rng.random((runs, 1)) > np.cumsum(ps, 1)).sum(1).clip(max=M - 1)
        pr = softmax(theta_r[idx, m])             # [R, N]
        a = (rng.random((runs, 1)) > np.cumsum(pr, 1)).sum(1).clip(max=N - 1)
        r = (a == s).astype(float)
        adv = r - b
        b += baseline_lr * (r - b)
        # REINFORCE: grad of log softmax wrt logits = onehot(chosen) - probs
        gs = -ps
        gs[idx, m] += 1.0
        gr = -pr
        gr[idx, a] += 1.0
        theta_s[idx, s] += lr * adv[:, None] * gs
        theta_r[idx, m] += lr * adv[:, None] * gr
        if t % 100 == 0 or t == steps - 1:
            curve.append((t, expected_success(theta_s, theta_r).mean()))
    return greedy_success(theta_s, theta_r), theta_s, theta_r, curve


def main():
    parser = argparse.ArgumentParser(description="Lewis signalling game with REINFORCE")
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    runs = 100 if args.quick else 1000
    steps = 2_000 if args.quick else 10_000
    print(f"lewis_signaling.py | seed={args.seed} | runs={runs} | steps={steps} | REINFORCE lr=0.5, "
          f"running baseline lr=0.01 | quick={args.quick}")
    t0 = time.time()
    settings = [(2, 2), (3, 3), (4, 4), (8, 8), (3, 6), (4, 8), (8, 16)]
    results = {}
    print("success of the final greedy protocol = fraction of states communicated correctly")
    print(f"\n{'states N':>8s} {'messages M':>10s} {'mean success':>12s} {'P(perfect)':>11s} "
          f"{'mean expected success':>22s}  distribution of final greedy success")
    for N, M in settings:
        rng = np.random.default_rng(args.seed + 100 * N + M)
        succ, ts, tr, curve = train(N, M, runs, steps, rng)
        results[(N, M)] = (succ, ts, tr, curve)
        levels, counts = np.unique(np.round(succ, 3), return_counts=True)
        order = np.argsort(-levels)
        lv = ", ".join(f"{levels[i]:.3f}: {100 * counts[i] / runs:.1f}%" for i in order)
        print(f"{N:8d} {M:10d} {succ.mean():12.3f} {np.mean(succ == 1.0):11.3f} "
              f"{curve[-1][1]:22.3f}  {lv}")
    print(f"\ntotal time {time.time() - t0:.1f} s")

    if not args.quick:
        import plot_style
        plt = plot_style.setup()
        figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
        os.makedirs(figdir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        ax = axes[0]
        for k, (N, M) in enumerate([(8, 8), (8, 16)]):
            succ = results[(N, M)][0]
            lv = np.arange(1, N + 1) / N
            frac = [np.mean(np.isclose(succ, v)) for v in lv]
            ax.bar(np.arange(1, N + 1) + (k - 0.5) * 0.38, frac, 0.36, color=plot_style.C[k],
                   hatch="" if k == 0 else "//", edgecolor="white", label=f"N = {N} states, M = {M} messages")
        ax.set_xlabel("states communicated correctly by the final greedy protocol (of 8)")
        ax.set_ylabel("fraction of runs")
        ax.set_title("(a) Where learning ends up (N = 8)")
        ax.legend()
        ax = axes[1]
        for k, (N, M) in enumerate([(3, 3), (4, 4), (8, 8), (8, 16)]):
            t, v = zip(*results[(N, M)][3])
            ax.plot(t, v, color=plot_style.C[k], ls=["-", "--", "-.", ":"][k], label=f"N = {N}, M = {M}")
        ax.set_xlabel("training step")
        ax.set_ylabel("mean expected success over runs")
        ax.set_title("(b) Learning curves")
        ax.set_ylim(0, 1.02)
        ax.legend()
        ax = axes[2]
        succ, ts, tr, _ = results[(4, 4)]
        bad = int(np.argmax(succ < 1.0)) if (succ < 1.0).any() else 0
        P = softmax(ts[bad])
        ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if P[i, j] > 0.6 else "black")
        ax.set_xlabel("message")
        ax.set_ylabel("state")
        ax.set_xticks(range(P.shape[1])); ax.set_yticks(range(P.shape[0]))
        ax.grid(False)
        ax.set_title(f"(c) A partial-pooling sender (N = M = 4), success {succ[bad]:.2f}")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "lewis_signaling.png"))
        plt.close(fig)
        print(f"figure written to {figdir}/lewis_signaling.png")


if __name__ == "__main__":
    main()
