"""Discounting-aware and per-decision importance sampling (Sec. 9): exact variances.

A fixed-length task: T = 10 decisions (T is the episode length, as in the chapter), state =
time step, actions {0, 1}, reward R_{t+1} = 1 if A_t = 0 else 0, then termination at time T.
Target pi(0|s) = 0.9, behavior b(0|s) = 0.5. Because there are only 2^T = 1024 action
sequences, we can enumerate them and compute the EXACT mean and variance of each
single-episode estimator of v_pi(s_0):

  ordinary IS          rho_{0:T-1} G_0
  per-decision IS      sum_k gamma^k rho_{0:k} R_{k+1}                              (S&B eq. 5.14 idea)
                       (equivalently, backwards: G~_t = rho_t (R_{t+1} + gamma G~_{t+1}), G~_T = 0)
  discounting-aware IS (1-gamma) sum_{h=1}^{T-1} gamma^{h-1} rho_{0:h-1} Gbar_{0:h}
                         + gamma^{T-1} rho_{0:T-1} Gbar_{0:T}                         (S&B eq. 5.9)

All three are unbiased; they differ (a lot) in variance. We also simulate MSE vs number of
episodes, adding the weighted (normalized) versions of ordinary and discounting-aware IS.

    python code/ch04_monte_carlo/per_decision_is.py [--quick]
"""
import argparse
import itertools
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
T, PI0, B0 = 10, 0.9, 0.5


def estimators(actions, gamma):
    """Per-episode quantities for one action sequence (array of 0/1, length T).

    Returns dict with the numerators of the ordinary, per-decision and discounting-aware
    estimators and the denominators used by the weighted versions."""
    actions = np.asarray(actions)
    R = (actions == 0).astype(float)                       # R_{k+1}
    ratio = np.where(actions == 0, PI0 / B0, (1 - PI0) / (1 - B0))
    rho = np.cumprod(ratio, axis=-1)                       # rho_{0:k}, k = 0..T-1
    disc = gamma ** np.arange(T)
    G = (disc * R).sum(-1)
    flat = np.cumsum(R, axis=-1)                           # Gbar_{0:h} for h = 1..T
    da_w = np.concatenate([(1 - gamma) * gamma ** np.arange(T - 1), [gamma ** (T - 1)]])
    return dict(
        ordinary=rho[..., -1] * G,
        per_decision=(disc * rho * R).sum(-1),
        da=(da_w * rho * flat).sum(-1),                    # rho_{0:h-1} pairs with Gbar_{0:h}
        w_ordinary_den=rho[..., -1],
        w_da_den=(da_w * rho).sum(-1),
    )


def exact_table(gamma):
    seqs = np.array(list(itertools.product((0, 1), repeat=T)))
    pb = np.prod(np.where(seqs == 0, B0, 1 - B0), axis=1)          # probability under b
    est = estimators(seqs, gamma)
    v = PI0 * (gamma ** np.arange(T)).sum()                        # v_pi(s_0)
    rows = {}
    for name in ("ordinary", "per_decision", "da"):
        mean = (pb * est[name]).sum()
        var = (pb * (est[name] - mean) ** 2).sum()
        rows[name] = (mean, var)
    return v, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    runs = 200 if args.quick else 2000
    n_ep = 300 if args.quick else 3000
    gamma_sim = 0.9
    print(f"Per-decision / discounting-aware IS | T={T} pi(0)={PI0} b(0)={B0} seed={args.seed} "
          f"simulation: gamma={gamma_sim}, runs={runs}, episodes/run={n_ep}")
    t0 = time.time()
    print(f"E_b[rho_(0:T-1)^2] = {(0.5 * (PI0 / B0) ** 2 + 0.5 * ((1 - PI0) / (1 - B0)) ** 2) ** T:.1f}")
    print("\nExact single-episode variance (all three estimators are unbiased: mean = v_pi):")
    print("  gamma |   v_pi  | ordinary IS | discounting-aware | per-decision")
    for gamma in (0.0, 0.5, 0.9, 1.0):
        v, rows = exact_table(gamma)
        assert all(abs(m - v) < 1e-9 for m, _ in rows.values()), "estimator is biased?!"
        print(f"  {gamma:5.2f} | {v:7.4f} | {rows['ordinary'][1]:11.3f} | {rows['da'][1]:17.3f} | "
              f"{rows['per_decision'][1]:12.3f}")

    # Simulation: MSE vs number of episodes for gamma = 0.9.
    rng = np.random.default_rng(args.seed)
    v, _ = exact_table(gamma_sim)
    n = np.arange(1, n_ep + 1)
    names = ("ordinary IS", "discounting-aware IS", "per-decision IS", "weighted IS",
             "weighted discounting-aware IS")
    sq_err = {k: np.zeros(n_ep) for k in names}
    for start in range(0, runs, 100):                              # chunks keep memory small
        r = min(100, runs - start)
        acts = (rng.random((r, n_ep, T)) >= B0).astype(np.int8)    # action 1 w.p. 1 - b(0)
        est = estimators(acts, gamma_sim)
        curves = {
            "ordinary IS": np.cumsum(est["ordinary"], 1) / n,
            "discounting-aware IS": np.cumsum(est["da"], 1) / n,
            "per-decision IS": np.cumsum(est["per_decision"], 1) / n,
            "weighted IS": np.cumsum(est["ordinary"], 1) / np.cumsum(est["w_ordinary_den"], 1),
            "weighted discounting-aware IS": np.cumsum(est["da"], 1) / np.cumsum(est["w_da_den"], 1),
        }
        for k, c in curves.items():
            sq_err[k] += ((c - v) ** 2).sum(0)
    mse = {k: s / runs for k, s in sq_err.items()}
    print(f"\nSimulated MSE, gamma = {gamma_sim} (v_pi = {v:.4f}), {runs} runs:")
    print("  episodes | " + " | ".join(f"{k:>14s}" for k in mse))
    for m in (1, 10, 100, 1000, 3000):
        if m <= n_ep:
            print(f"  {m:8d} | " + " | ".join(f"{mse[k][m - 1]:14.4f}" for k in mse))
    print(f"elapsed {time.time() - t0:.1f}s")

    if args.quick:
        return
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, style in zip(mse, ("C3-", "C1-", "C0-", "C2--", "C4--")):
        ax.loglog(n, mse[k], style, label=k)
    ax.set_xlabel("episodes")
    ax.set_ylabel(f"MSE (average over {runs} runs)")
    ax.set_title(rf"Off-policy estimators, T={T}, $\gamma$={gamma_sim}, $\pi(0)$={PI0}, $b(0)$={B0}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "per_decision_is.png"), dpi=110)
    print("saved figures/per_decision_is.png")


if __name__ == "__main__":
    main()
