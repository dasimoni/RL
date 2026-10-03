"""Bias and variance of n-step and lambda-return targets on the 19-state random walk.

Chapter 06, Sections 2.3 and 15.1. Two things are computed for a *fixed* value
estimate V (no learning happens in this script):

1. The error-reduction property (Eq. 6.4), computed exactly with the transition
   matrix P of the walk (gamma = 1):
       E[G_{t:t+n} | S_t = s] - v(s) = (P^n (V - v))(s),
   so  max_s |E[G_{t:t+n}|S_t=s] - v(s)| <= ||P^n||_inf * max_s |V(s) - v(s)|,
   where ||P^n||_inf = max_s Pr{episode still running after n steps | S_0 = s} <= gamma^n = 1.

2. Monte Carlo estimates of bias^2 and variance of the targets G_{t:t+n} and
   G_t^lambda, averaged uniformly over the 19 start states, for two estimates:
       V = 0 (where every experiment in this chapter starts), and
       V = v + noise (a "half-learned" estimate, noise ~ N(0, 0.15^2), fixed draw).
   The mean-squared error of a target about v(s) is bias^2 + variance.

Outputs (full mode): figures/error_reduction.png, figures/return_bias_variance.png

Run:  python code/ch06_n_step_and_eligibility_traces/return_bias_variance.py [--quick]
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one CPU thread (shared machine)
import numpy as np  # noqa: E402

import random_walk19 as rw

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
N_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
LAM_LIST = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0]


# ---------------------------------------------------------------- 1. exact error reduction
def error_reduction(V_list, ns):
    """For each V (array over the 19 states) return max-error ratios e_n / e_0, and the
    survival bound max_s Pr{alive after n}."""
    P, r = rw.markov_chain()
    v = np.linalg.solve(np.eye(rw.N_STATES) - P, r)          # exact values
    assert np.allclose(v, rw.TRUE_V)
    ratios = np.zeros((len(V_list), len(ns)))
    survival = np.zeros(len(ns))
    for j, n in enumerate(ns):
        Pn = np.linalg.matrix_power(P, n)
        survival[j] = Pn.sum(axis=1).max()
        for i, V in enumerate(V_list):
            # expected n-step return, computed term by term (not via the identity) as a check
            expected_G = sum(np.linalg.matrix_power(P, k) @ r for k in range(n)) + Pn @ V \
                if n <= 64 else None
            err_n = Pn @ (V - v)
            if expected_G is not None:
                assert np.allclose(expected_G - v, err_n)
            ratios[i, j] = np.abs(err_n).max() / np.abs(V - v).max()
    return ratios, survival


# ---------------------------------------------------------------- 2. Monte Carlo bias/variance
def sample_walks(rng, start, m, max_len=4000, chunk=250):
    """m random walks from `start`, padded to a common length.

    Returns S (m, L) state matrix (terminal state repeated after absorption),
    T (m,) episode lengths and R_T (m,) terminal rewards (+-1). A walk that has not
    terminated after max_len steps raises an error (it never happens; max_len is generous).

    Memory: the +-1 steps are drawn in blocks of `chunk` walks and kept as int8, and states
    as int16 (indices 0..20). A block of rows consumes exactly the random numbers that the
    same rows of one big (m, max_len) draw would, so the results do not depend on `chunk`.
    The two walk arrays take about 50 MB instead of about 260 MB as int64; the whole script
    peaks at about 180 MB of resident memory (measured), most of it NumPy and Matplotlib."""
    steps = np.empty((m, max_len), dtype=np.int8)
    for i in range(0, m, chunk):
        steps[i:i + chunk] = rng.choice((-1, 1), size=(min(chunk, m - i), max_len))
    S = np.empty((m, max_len + 1), dtype=np.int16)
    S[:, 0] = start
    pos = np.full(m, start, dtype=np.int16)
    alive = np.ones(m, dtype=bool)
    T = np.full(m, -1)
    for t in range(max_len):
        pos = np.where(alive, pos + steps[:, t], pos)
        S[:, t + 1] = pos
        ended = alive & ((pos == rw.LEFT) | (pos == rw.RIGHT))
        T[ended] = t + 1
        alive &= ~ended
        if not alive.any():
            S = S[:, : t + 2]
            break
    if alive.any():
        raise RuntimeError("increase max_len")
    RT = np.where(S[np.arange(m), T] == rw.RIGHT, 1.0, -1.0)
    return S, T, RT


def targets_from_walks(S, T, RT, V_full, ns, lams):
    """n-step returns G_{0:n} and lambda-returns G_0^lambda for every sampled walk (gamma = 1).

    Rewards are zero except R_T, so  G_{0:n} = R_T if T <= n else V(S_n),  and
    G_0^lambda = (1-lambda) sum_{k=1}^{T-1} lambda^{k-1} V(S_k) + lambda^{T-1} R_T   (Eq. 6.17)."""
    m, L = S.shape
    Vs = V_full[S]                                   # V(S_k), zero at terminal states
    cols = np.arange(L)
    G_n = np.empty((m, len(ns)))
    for j, n in enumerate(ns):
        idx = np.minimum(n, L - 1)
        G_n[:, j] = np.where(T <= n, RT, Vs[:, idx])
    G_lam = np.empty((m, len(lams)))
    k = cols[1:]                                     # k = 1 .. L-1
    inside = k[None, :] < T[:, None]                 # only k <= T-1 contribute
    for j, lam in enumerate(lams):
        if lam == 1.0:
            G_lam[:, j] = RT
            continue
        w = (1 - lam) * lam ** (k - 1.0)
        G_lam[:, j] = (Vs[:, 1:] * inside) @ w + lam ** (T - 1.0) * RT
    return G_n, G_lam


def bias_variance(V_full, rng, m, ns, lams):
    """Average over the 19 start states of bias^2 and variance of each target."""
    b2_n = np.zeros(len(ns)); var_n = np.zeros(len(ns))
    b2_l = np.zeros(len(lams)); var_l = np.zeros(len(lams))
    for s in range(1, rw.N_STATES + 1):
        S, T, RT = sample_walks(rng, s, m)
        G_n, G_l = targets_from_walks(S, T, RT, V_full, ns, lams)
        v = rw.TRUE_V_FULL[s]
        # (mean - v)^2 overestimates bias^2 by Var/m on average; subtract that so the
        # estimate is unbiased (it can then be slightly negative, which we clip at 0 below).
        vn, vl = G_n.var(0, ddof=1), G_l.var(0, ddof=1)
        b2_n += (G_n.mean(0) - v) ** 2 - vn / m
        var_n += vn
        b2_l += (G_l.mean(0) - v) ** 2 - vl / m
        var_l += vl
    k = rw.N_STATES
    return np.maximum(b2_n / k, 0.0), var_n / k, np.maximum(b2_l / k, 0.0), var_l / k


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    m = 300 if args.quick else 4000
    noise_sd = 0.15
    print(f"Bias/variance of n-step and lambda-return targets | seed={args.seed} "
          f"walks per start state={m} gamma=1 noisy-V sd={noise_sd}")
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    # ---- 1. exact error reduction
    V_zero = np.zeros(rw.N_STATES)
    V_rand = [rng.uniform(-1, 1, rw.N_STATES) for _ in range(3)]
    ratios, survival = error_reduction([V_zero] + V_rand, N_LIST)
    print("\n1) Error-reduction property (exact): max_s|E[G_{t:t+n}]-v| / max_s|V-v|")
    print("     n   V=0     rand V#1  rand V#2  rand V#3   bound gamma^n   survival bound")
    for j, n in enumerate(N_LIST):
        print(f"  {n:4d}  {ratios[0, j]:.4f}   {ratios[1, j]:.4f}    {ratios[2, j]:.4f}    "
              f"{ratios[3, j]:.4f}    1.0000          {survival[j]:.4f}")
    assert np.all(ratios <= survival[None, :] + 1e-12)
    print("  all ratios <= survival bound <= 1: OK")

    # ---- 2. Monte Carlo bias / variance
    V0_full = np.zeros(rw.N_TOTAL)
    Vn_full = rw.TRUE_V_FULL.copy()
    Vn_full[1:-1] += noise_sd * rng.standard_normal(rw.N_STATES)
    results = {}
    for name, Vf in [("V = 0", V0_full), (f"V = v + N(0,{noise_sd}^2)", Vn_full)]:
        results[name] = bias_variance(Vf, rng, m, N_LIST, LAM_LIST)
        b2_n, var_n, b2_l, var_l = results[name]
        mse_n, mse_l = b2_n + var_n, b2_l + var_l
        print(f"\n2) {name}: target error about v(s), averaged over the 19 states")
        print("     n    bias^2    variance   MSE")
        for j, n in enumerate(N_LIST):
            print(f"  {n:4d}   {b2_n[j]:.4f}    {var_n[j]:.4f}    {mse_n[j]:.4f}")
        print(f"  -> best n = {N_LIST[int(np.argmin(mse_n))]} (MSE {mse_n.min():.4f})")
        print("   lambda  bias^2    variance   MSE")
        for j, lam in enumerate(LAM_LIST):
            print(f"  {lam:6.3f}   {b2_l[j]:.4f}    {var_l[j]:.4f}    {mse_l[j]:.4f}")
        print(f"  -> best lambda = {LAM_LIST[int(np.argmin(mse_l))]} (MSE {mse_l.min():.4f})")
    print(f"\nelapsed {time.time() - t0:.1f} s")

    if args.quick:
        return
    from plot_style import setup, C, GREY
    plt = setup()
    os.makedirs(FIG_DIR, exist_ok=True)

    # figure 1: error reduction
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = ["V = 0"] + [f"random V #{i}" for i in (1, 2, 3)]
    for i in range(4):
        ax.plot(N_LIST, ratios[i], color=C[i], marker="os^D"[i], label=labels[i])
    ax.plot(N_LIST, survival, color=GREY, ls="--", label=r"bound $\max_s \Pr\{T > n \mid S_0=s\}$")
    ax.axhline(1.0, color=GREY, ls=":", lw=1.2, label=r"bound $\gamma^n = 1$")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n (steps before bootstrapping)")
    ax.set_ylabel(r"$\max_s|\mathbb{E}[G_{t:t+n}]-v(s)|\;/\;\max_s|V(s)-v(s)|$")
    ax.set_title("Error reduction of the expected n-step return (exact)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    path1 = os.path.join(FIG_DIR, "error_reduction.png")
    fig.savefig(path1)
    plt.close(fig)

    # figure 2: bias^2 / variance / MSE. The lambda panels use the x-coordinate
    # 1/(1 - lambda), the mean of the geometric weights (1-lambda) lambda^{n-1} over n
    # (Section 8.1), so the two columns share a horizon scale; lambda = 1 is drawn at 1024.
    lam_x = [1.0 / (1.0 - lam) if lam < 1 else 1024.0 for lam in LAM_LIST]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.0), sharey=True)
    floor = 1e-4
    for row, (name, (b2_n, var_n, b2_l, var_l)) in enumerate(results.items()):
        for col, (x, b2, var) in enumerate([(N_LIST, b2_n, var_n), (lam_x, b2_l, var_l)]):
            ax = axes[row, col]
            mse = b2 + var
            ax.plot(x, np.where(b2 > floor, b2, np.nan), color=C[0], marker="o", label=r"bias$^2$")
            ax.plot(x, var, color=C[1], marker="s", label="variance")
            ax.plot(x, mse, color=INK_COLOR, marker="^", label="MSE = bias$^2$ + variance")
            j = int(np.argmin(mse))
            best = f"n = {N_LIST[j]}" if col == 0 else rf"$\lambda$ = {LAM_LIST[j]}"
            ax.plot(x[j], mse[j], "o", mfc="none", mec=INK_COLOR, ms=11, mew=1.2)
            ax.annotate(f"min MSE at {best}", (x[j], mse[j]), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=8.5, color=INK_COLOR)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_ylim(floor, 1.5)
            if col == 0:
                ax.set_xlabel("n")
                ax.set_ylabel("average over the 19 states")
                ax.set_title(f"n-step return, {name}")
            else:
                ticks = [0.0, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 1.0]
                ax.set_xticks([1.0 / (1.0 - l) if l < 1 else 1024.0 for l in ticks])
                ax.set_xticklabels([f"{l:g}" for l in ticks], fontsize=8)
                ax.minorticks_off()
                ax.set_xlabel(r"$\lambda$  (axis spaced by $1/(1-\lambda)$)")
                ax.set_title(rf"$\lambda$-return, {name}")
            ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    path2 = os.path.join(FIG_DIR, "return_bias_variance.png")
    fig.savefig(path2)
    plt.close(fig)
    print(f"saved {path1}\nsaved {path2}")


INK_COLOR = "#52514e"

if __name__ == "__main__":
    main()
