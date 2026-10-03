"""lambda-return methods on the 19-state random walk (S&B 2018, Figures 12.3, 12.6, 12.8).

Chapter 06, Sections 8-12. For each lambda and step size alpha we run, from V = 0,
for 10 episodes, and report the RMS error over the 19 states averaged over the
first 10 episodes and over 100 runs:

  * offline lambda-return algorithm (forward view, Section 8.3): targets
        G_t^lambda = R_{t+1} + gamma[(1-lambda) V(S_{t+1}) + lambda G_{t+1}^lambda]   (Eq. 6.18)
    computed at the end of the episode from the values frozen during the episode,
    then applied as a sequence of updates V(S_t) <- V(S_t) + alpha[G_t^lambda - V(S_t)];
  * TD(lambda) with accumulating traces (backward view, Section 9), updating online;
  * TD(lambda) with replacing traces;
  * true online TD(lambda) with dutch traces (Section 12);
  * truncated TD(lambda), TTD(lambda): the truncated lambda-return G^lambda_{t:t+K}
    with horizon K = 8, applied K - 1 steps late (Section 11).

All (lambda, alpha) pairs run in parallel as arrays of shape (n_lambda, n_alpha, 21),
and every method sees the same episodes.

Outputs (full mode): figures/lambda_methods_random_walk.png

Run:  python code/ch06_n_step_and_eligibility_traces/lambda_methods_random_walk.py [--quick]
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one CPU thread (shared machine)
import numpy as np  # noqa: E402

import random_walk19 as rw  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
TTD_K = 8


# Every function below updates V in place. Shapes: V (L, A, 21), lam (L, 1), alphas (1, A).
def offline_lambda_return_episode(V, states, rewards, lam, alphas, gamma=1.0):
    """Offline lambda-return algorithm: no change during the episode; at the end, the
    lambda-returns are computed backwards with the frozen values V0, then applied in
    time order (S&B Eq. 12.4)."""
    T = len(rewards)
    V0 = V.copy()                                   # values that were in force during the episode
    targets = np.empty((T,) + V.shape[:2])
    G = np.zeros(V.shape[:2])                       # G^lambda_T = 0 (beyond the end)
    for t in range(T - 1, -1, -1):
        G = rewards[t] + gamma * ((1 - lam) * V0[..., states[t + 1]] + lam * G)
        targets[t] = G
    for t in range(T):
        s = states[t]
        V[..., s] += alphas * (targets[t] - V[..., s])


def td_lambda_episode(V, states, rewards, lam, alphas, gamma=1.0, replacing=False):
    """Online tabular TD(lambda), backward view. The trace z does not depend on alpha,
    so one trace per lambda (shape (L, 21)) serves all step sizes."""
    z = np.zeros((V.shape[0], V.shape[2]))
    for t in range(len(rewards)):
        s, s1 = states[t], states[t + 1]
        delta = rewards[t] + gamma * V[..., s1] - V[..., s]    # TD error, (L, A)
        z *= gamma * lam                                         # decay every trace
        if replacing:
            z[:, s] = 1.0                                        # replacing trace
        else:
            z[:, s] += 1.0                                       # accumulating trace
        V += (alphas * delta)[..., None] * z[:, None, :]


def true_online_td_lambda_episode(V, states, rewards, lam, alphas, gamma=1.0):
    """True online TD(lambda) (van Seijen & Sutton 2014) in tabular form, i.e. with
    one-hot features x(s) = e_s:
        z <- gamma lambda z + (1 - alpha gamma lambda z(S)) e_S          (dutch trace)
        V <- V + alpha (delta + V(S) - V_old) z - alpha (V(S) - V_old) e_S
    The dutch trace depends on alpha, so z has the full shape (L, A, 21)."""
    gl = gamma * lam
    z = np.zeros_like(V)
    v_old = np.zeros(V.shape[:2])
    for t in range(len(rewards)):
        s, s1 = states[t], states[t + 1]
        v, v1 = V[..., s].copy(), V[..., s1].copy()
        delta = rewards[t] + gamma * v1 - v
        zs = z[..., s].copy()
        z *= gl[..., None]
        z[..., s] += 1.0 - alphas * gl * zs
        V += (alphas * (delta + v - v_old))[..., None] * z
        V[..., s] -= alphas * (v - v_old)
        v_old = v1


def ttd_lambda_episode(V, states, rewards, lam, alphas, K, gamma=1.0):
    """Truncated TD(lambda): like n-step TD with n = K, but the target is the truncated
    lambda-return G^lambda_{tau:h}, h = min(tau + K, T), computed with the current V."""
    T = len(rewards)
    for t in range(T + K - 1):
        tau = t - K + 1
        if tau < 0:
            continue
        h = min(tau + K, T)
        G = rewards[h - 1] + gamma * V[..., states[h]]          # G^lambda_{h-1:h} (V(terminal) = 0)
        for i in range(h - 2, tau - 1, -1):                       # recursion of Eq. 6.18, truncated at h
            G = rewards[i] + gamma * ((1 - lam) * V[..., states[i + 1]] + lam * G)
        s = states[tau]
        V[..., s] += alphas * (G - V[..., s])


METHODS = {
    "offline λ-return": lambda V, S, R, l, a: offline_lambda_return_episode(V, S, R, l, a),
    "TD(λ), accumulating": lambda V, S, R, l, a: td_lambda_episode(V, S, R, l, a),
    "TD(λ), replacing": lambda V, S, R, l, a: td_lambda_episode(V, S, R, l, a, replacing=True),
    "true online TD(λ)": lambda V, S, R, l, a: true_online_td_lambda_episode(V, S, R, l, a),
    f"TTD(λ), K={TTD_K}": lambda V, S, R, l, a: ttd_lambda_episode(V, S, R, l, a, TTD_K),
}


def run(lams, alphas, episodes):
    lam = np.asarray(lams)[:, None]
    al = np.asarray(alphas)[None, :]
    out = {}
    for name, fn in METHODS.items():
        t0 = time.time()
        err = np.zeros((len(lams), len(alphas)))
        with np.errstate(over="ignore", invalid="ignore"):
            for run_eps in episodes:
                V = np.zeros((len(lams), len(alphas), rw.N_TOTAL))
                for states, rewards in run_eps:
                    fn(V, states, rewards, lam, al)
                    err += rw.rms_error(V)
        err /= len(episodes) * len(episodes[0])
        err[~np.isfinite(err)] = np.inf                       # diverged runs
        out[name] = err
        print(f"  {name:22s} done in {time.time() - t0:5.1f} s")
    return out


def plot(lams, alphas, res, path):
    from plot_style import setup, C, MARKERS
    plt = setup()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharey=True)
    # Eight lambda values: one categorical slot each (fixed order, never cycled) plus a
    # distinct marker, so that neighbouring lambdas such as 0.95 and 0.975 stay distinguishable.
    assert len(lams) <= len(C)
    cols = C[:len(lams)]
    for ax, (name, err) in zip(axes.flat, res.items()):
        for i, lam in enumerate(lams):
            ax.plot(alphas, np.where(np.isfinite(err[i]), err[i], np.nan), color=cols[i],
                    marker=MARKERS[i], markersize=3.5, lw=1.5, label=f"λ = {lam:g}")
        ax.set_title(name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.23, 0.55)
        ax.set_xlabel(r"step size $\alpha$")
    for ax in axes[:, 0]:
        ax.set_ylabel("RMS error, first 10 episodes")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(lams), bbox_to_anchor=(0.5, 0.955), fontsize=9)
    # summary panel: best alpha for every lambda
    ax = axes[1, 2]
    # Colour means lambda in the other panels, so here the methods are told apart by
    # line style and marker in neutral ink instead.
    inks = ["#0b0b0b", "#52514e", "#0b0b0b", "#52514e", "#898781"]
    styles = ["-", "--", ":", "-.", (0, (6, 2))]
    for k, (name, err) in enumerate(res.items()):
        ax.plot(lams, err.min(axis=1), color=inks[k], ls=styles[k], marker=MARKERS[k], ms=5,
                mfc="none" if k % 2 else inks[k], label=name)
    ax.set_xscale("function", functions=(lambda x: -np.log10(1.001 - x), lambda y: 1.001 - 10 ** (-y)))
    ax.set_xticks(lams)
    ax.set_xticklabels([f"{x:g}" for x in lams], fontsize=7.5, rotation=45)
    ax.set_xlim(-0.02, 1.0)
    ax.set_xlabel(r"$\lambda$ (axis stretched near 1)")
    ax.set_title(r"best $\alpha$ for each $\lambda$")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle("19-state random walk, 100 runs: offline λ-return vs TD(λ) vs true online TD(λ)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    lams = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0]
    if args.quick:
        alphas = np.round(np.linspace(0.1, 1.0, 4), 3)
        n_runs = 5
    else:
        alphas = np.unique(np.round(np.concatenate([[0.01, 0.02, 0.03, 0.04],
                                                    np.linspace(0.05, 1.0, 20)]), 3))
        n_runs = 100
    n_episodes = 10
    print(f"lambda methods on the 19-state random walk | seed={args.seed} runs={n_runs} "
          f"episodes={n_episodes} gamma=1 V0=0 TTD K={TTD_K}")
    print(f"lambda in {lams}\nalpha in {alphas.tolist()}")
    t0 = time.time()
    episodes = rw.generate_runs(n_runs, n_episodes, args.seed)
    res = run(lams, alphas, episodes)

    print("\nBest alpha and RMS error (averaged over first 10 episodes) for each lambda:")
    header = "  lambda " + "".join(f"| {name:^22s}" for name in res)
    print(header)
    for i, lam in enumerate(lams):
        row = f"  {lam:6.3f} "
        for err in res.values():
            j = int(np.argmin(err[i]))
            row += f"| a={alphas[j]:4.2f} rms={err[i, j]:.4f}   "
        print(row)
    print("\nOverall best per method:")
    for name, err in res.items():
        i, j = np.unravel_index(np.argmin(err), err.shape)
        print(f"  {name:22s} lambda={lams[i]:<5g} alpha={alphas[j]:.2f} RMS={err[i, j]:.4f}")
    print("\nLargest alpha with RMS < 0.55 (i.e. still learning) at lambda = 0.95 and 0.99:")
    for name, err in res.items():
        msg = []
        for lam in (0.95, 0.99):
            i = lams.index(lam)
            ok = np.flatnonzero(err[i] < 0.55)
            msg.append(f"lambda={lam}: {alphas[ok[-1]]:.2f}" if ok.size else f"lambda={lam}: none")
        print(f"  {name:22s} " + ",  ".join(msg))
    print(f"\nelapsed {time.time() - t0:.1f} s")

    if not args.quick:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, "lambda_methods_random_walk.png")
        plot(lams, alphas, res, path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
