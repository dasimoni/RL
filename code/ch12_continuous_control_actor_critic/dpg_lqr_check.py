"""The deterministic policy gradient theorem, checked exactly on a scalar linear-quadratic problem
(Chapter 12, Section 2).

Problem (everything scalar):
    dynamics   S_{t+1} = F S_t + G A_t + W_t,   W_t ~ N(0, sw^2),   S_0 ~ N(0, s0^2)
    reward     R_{t+1} = -(c_s S_t^2 + c_a A_t^2)
    policy     mu_theta(s) = theta * s        (closed loop: S_{t+1} = k S_t + W_t,  k = F + G theta)

Everything is available in closed form (Section 2.6 derives it):
    v_mu(s)    = -P s^2 - c,            P = (c_s + c_a theta^2) / (1 - gamma k^2)
    J(theta)   = -P s0^2 - gamma P sw^2 / (1 - gamma)
    q_mu(s,a)  = -(c_s s^2 + c_a a^2) - gamma P ((F s + G a)^2 + sw^2) - gamma c
    DPG        dJ/dtheta = sum_t gamma^t E[ grad_theta mu(S_t) * d/da q_mu(S_t, a)|_{a = mu(S_t)} ]  (Eq. 12.3)

We check
  1. the DPG formula against central finite differences of J(theta) at several theta;
  2. a Monte Carlo version of the DPG estimator (sampled trajectories, exact dQ/da);
  3. Silver et al.'s limit theorem: the stochastic policy gradient of the Gaussian policy
     N(theta s, sigma^2) converges to the DPG as sigma -> 0, while the variance of its
     single-sample score-function estimator grows like 1/sigma^2.  The DPG estimator's does not.
  4. Exercise 2's closed-form variances of the score-function estimator on a one-step problem
     r(a) = -(a - a*)^2, by Monte Carlo.

Run (from the repository root):
  python code/ch12_continuous_control_actor_critic/dpg_lqr_check.py          # full, figure
  python code/ch12_continuous_control_actor_critic/dpg_lqr_check.py --quick  # fewer samples
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from common import FIG_DIR

F_, G_, CS, CA, GAMMA = 1.0, 1.0, 1.0, 1.0, 0.9


def P_of(theta):
    k = F_ + G_ * theta
    assert GAMMA * k ** 2 < 1, "closed loop not stable enough: value is infinite"
    return (CS + CA * theta ** 2) / (1 - GAMMA * k ** 2)


def J_det(theta, s0, sw):
    """Exact objective of the deterministic policy (Section 2.6)."""
    P = P_of(theta)
    return -P * s0 ** 2 - GAMMA * P * sw ** 2 / (1 - GAMMA)


def dpg_exact(theta, s0, sw):
    """DPG theorem: sum_t gamma^t E[S_t * dq/da(S_t, theta S_t)]  in closed form."""
    k = F_ + G_ * theta
    P = P_of(theta)
    coef = -2.0 * (CA * theta + GAMMA * P * G_ * k)        # dq/da at a = theta s equals coef * s
    disc_second_moment = (s0 ** 2 + GAMMA * sw ** 2 / (1 - GAMMA)) / (1 - GAMMA * k ** 2)
    return coef * disc_second_moment                        # grad_theta mu(s) = s


def J_stoch(theta, sigma, s0, sw):
    """Exact objective of the Gaussian policy A ~ N(theta s, sigma^2): extra noise G sigma xi in the
    dynamics and an extra expected cost c_a sigma^2 every step (Exercise 3)."""
    P = P_of(theta)
    c = (CA * sigma ** 2 + GAMMA * P * (G_ ** 2 * sigma ** 2 + sw ** 2)) / (1 - GAMMA)
    return -P * s0 ** 2 - c


def sample_discounted_states(theta, sigma, s0, sw, n, rng):
    """Draw S ~ d (normalised discounted state distribution) for the policy N(theta s, sigma^2):
    draw t ~ Geometric, then S_t ~ N(0, var_t) exactly (the closed loop is linear-Gaussian)."""
    k = F_ + G_ * theta
    t = rng.geometric(1 - GAMMA, size=n) - 1                # P(t) = (1 - gamma) gamma^t, t >= 0
    noise_var = sw ** 2 + (G_ * sigma) ** 2
    if abs(k) < 1e-12:
        var = np.where(t == 0, s0 ** 2, noise_var)
    else:
        var = k ** (2 * t) * s0 ** 2 + noise_var * (1 - k ** (2 * t)) / (1 - k ** 2)
    return rng.normal(0.0, np.sqrt(var))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seed = 0
    rng = np.random.default_rng(seed)
    s0, sw = 1.0, 0.0
    n_traj = 20_000 if args.quick else 200_000
    n_samp = 200_000 if args.quick else 2_000_000
    print(f"DPG check on scalar LQR | seed {seed} | F={F_} G={G_} c_s={CS} c_a={CA} gamma={GAMMA} "
          f"s0={s0} sw={sw}")

    # ---- 1. exact DPG vs finite differences ----
    print("\n1) DPG theorem vs central finite differences of J (h = 1e-5):")
    print(f"   {'sw':>4s} {'theta':>6s} {'J(theta)':>10s} {'DPG formula':>12s} {'finite diff':>12s} {'abs diff':>9s}")
    max_fd_err = 0.0
    for sw_ in [0.0, 0.3]:
        for th in [-1.5, -1.0, -0.8, -0.5, -0.2]:
            h = 1e-5
            fd = (J_det(th + h, s0, sw_) - J_det(th - h, s0, sw_)) / (2 * h)
            g = dpg_exact(th, s0, sw_)
            max_fd_err = max(max_fd_err, abs(g - fd))
            print(f"   {sw_:4.1f} {th:6.2f} {J_det(th, s0, sw_):10.5f} {g:12.6f} {fd:12.6f} {abs(g - fd):9.1e}")
    ths = np.linspace(-1.9, -0.05, 2001)
    th_star = ths[np.argmax([J_det(t, s0, sw) for t in ths])]
    print(f"   (the optimal gain on a grid is theta* = {th_star:.3f}; the DPG is positive below it, "
          f"negative above it)")

    # ---- 2. Monte Carlo DPG with sampled trajectories ----
    theta = -0.5
    T = 120   # gamma^120 ~ 3e-6: truncation is negligible
    S = rng.normal(0.0, s0, n_traj)
    est = np.zeros(n_traj)
    k = F_ + G_ * theta
    P = P_of(theta)
    for t in range(T):
        dq_da = -2 * CA * theta * S - 2 * GAMMA * P * G_ * (F_ * S + G_ * theta * S)
        est += GAMMA ** t * S * dq_da
        S = k * S + rng.normal(0.0, sw, n_traj) if sw > 0 else k * S
    se = est.std() / np.sqrt(n_traj)
    z_mc = abs(est.mean() - dpg_exact(theta, s0, sw)) / se
    print(f"\n2) theta = {theta}: Monte Carlo DPG over {n_traj:,} trajectories = "
          f"{est.mean():.5f} +- {se:.5f};  exact {dpg_exact(theta, s0, sw):.5f}  "
          f"(|MC - exact| = {z_mc:.2f} standard errors)")

    # ---- 3. stochastic PG -> DPG as sigma -> 0; estimator variances ----
    # Process noise sw3 > 0 here so that sampled returns are noisy, as in any real environment.
    sw3, T3 = 0.3, 100
    print(f"\n3) theta = {theta}, process noise sw = {sw3}: Gaussian policy N(theta s, sigma^2) vs "
          f"deterministic policy ({n_samp:,} states from the discounted state distribution; "
          f"{n_samp // 10:,} for the sampled-return estimator)")
    print(f"   {'sigma':>6s} {'exact SPG':>10s} {'MC mean (A-estimator)':>22s} | per-sample std:"
          f" {'score*q':>8s} {'score*A':>8s} {'score*(G-v)':>11s} {'DPG':>6s}")
    sigmas = [1.0, 0.5, 0.3, 0.1, 0.03, 0.01]
    rows = []
    Ps = P_of(theta)
    for sigma in sigmas:
        h = 1e-5
        spg_exact = (J_stoch(theta + h, sigma, s0, sw3) - J_stoch(theta - h, sigma, s0, sw3)) / (2 * h)
        s = sample_discounted_states(theta, sigma, s0, sw3, n_samp, rng)
        a = theta * s + sigma * rng.standard_normal(n_samp)
        # exact q and v of the Gaussian policy (Exercise 3)
        c_sig = (CA * sigma ** 2 + GAMMA * Ps * (G_ ** 2 * sigma ** 2 + sw3 ** 2)) / (1 - GAMMA)
        q = -(CS * s ** 2 + CA * a ** 2) - GAMMA * Ps * ((F_ * s + G_ * a) ** 2 + sw3 ** 2) - GAMMA * c_sig
        v = -Ps * s ** 2 - c_sig
        score = (a - theta * s) * s / sigma ** 2              # d/dtheta log N(a; theta s, sigma^2)
        g_q = score * q / (1 - GAMMA)                         # (i)  1/(1-gamma) E_d[score * q]
        g_A = score * (q - v) / (1 - GAMMA)                   # (ii) exact advantage (ideal critic)
        # (iii) REINFORCE-style: one sampled return G from (s, a) minus the exact baseline v(s)
        m = n_samp // 10
        sg, ag = s[:m], a[:m]
        G = -(CS * sg ** 2 + CA * ag ** 2)
        x = F_ * sg + G_ * ag + rng.normal(0.0, sw3, m)
        for t in range(1, T3):
            u = theta * x + sigma * rng.standard_normal(m)
            G += GAMMA ** t * -(CS * x ** 2 + CA * u ** 2)
            x = F_ * x + G_ * u + rng.normal(0.0, sw3, m)
        g_G = score[:m] * (G - v[:m]) / (1 - GAMMA)
        # (iv) DPG single-sample estimator at the same states (gradient of the deterministic q)
        dq_da = -2 * CA * theta * s - 2 * GAMMA * Ps * G_ * (F_ * s + G_ * theta * s)
        g_dpg = s * dq_da / (1 - GAMMA)
        rows.append((sigma, spg_exact, g_A.mean(), g_q.std(), g_A.std(), g_G.std(), g_dpg.std()))
        print(f"   {sigma:6.2f} {spg_exact:10.5f} {g_A.mean():12.4f} +- {g_A.std() / np.sqrt(n_samp):.4f} |"
              f"                {g_q.std():8.1f} {g_A.std():8.2f} {g_G.std():11.1f} {g_dpg.std():6.2f}")
    print(f"   deterministic policy gradient (sigma = 0) with sw = {sw3}: {dpg_exact(theta, s0, sw3):.5f}")

    # ---- 4. Exercise 2: one-step problem r(a) = -(a - a*)^2, Gaussian policy a = theta + sigma xi ----
    # Separate RNG, so sections 1-3 (and the figure) do not depend on this section.
    rng4 = np.random.default_rng(seed + 4)
    n4 = 400_000 if args.quick else 4_000_000
    delta = 0.7                                             # delta = theta - a*
    print(f"\n4) Exercise 2: score-function estimator on r(a) = -(a - a*)^2 with delta = theta - a* = "
          f"{delta}, {n4:,} samples")
    print(f"   {'sigma':>6s} {'baseline':>9s} {'MC mean':>9s} {'exact':>7s} {'MC var':>9s} {'formula':>9s}")
    for sig in [1.0, 0.1]:
        xi = rng4.standard_normal(n4)
        r = -(delta + sig * xi) ** 2
        for use_b in [False, True]:
            b = -(delta ** 2 + sig ** 2) if use_b else 0.0     # b = J_sigma(theta)
            g = xi / sig * (r - b)
            formula = 8 * delta ** 2 + 10 * sig ** 2 if use_b else \
                delta ** 4 / sig ** 2 + 14 * delta ** 2 + 15 * sig ** 2
            print(f"   {sig:6.2f} {'J_sigma' if use_b else 'none':>9s} {g.mean():9.4f} {-2 * delta:7.3f} "
                  f"{g.var():9.3f} {formula:9.3f}")

    print("\n=== Summary ===")
    print(f"DPG formula matches finite differences to {max_fd_err:.1e} (max over 10 cases); "
          f"MC DPG is {z_mc:.2f} standard errors from the exact value; "
          f"exact SPG -> DPG as sigma -> 0 ({rows[0][1]:.4f} at sigma=1 -> {rows[-1][1]:.5f} at sigma=0.01 vs "
          f"{dpg_exact(theta, s0, sw3):.5f}); per-sample std of the score-function estimator grows "
          f"(sampled returns) from {rows[0][5]:.1f} to {rows[-1][5]:.1f} while DPG's stays {rows[-1][6]:.2f} "
          f"and the exact-advantage estimator's stays {rows[-1][4]:.2f}")

    if not args.quick:
        from plot_style import C, setup
        plt = setup()
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
        ax = axes[0]
        J = np.array([J_det(t, s0, sw) for t in ths])
        ax.plot(ths, J, color=C[0], label="J(theta), exact")
        for t0 in [-1.5, -0.5]:
            g = dpg_exact(t0, s0, sw)
            xs = np.linspace(t0 - 0.25, t0 + 0.25, 10)
            ax.plot(xs, J_det(t0, s0, sw) + g * (xs - t0), color=C[1], ls="--", lw=1.6,
                    label="tangent with slope = DPG" if t0 == -1.5 else None)
            ax.plot([t0], [J_det(t0, s0, sw)], "o", color=C[1])
        ax.axvline(th_star, color="#8a8985", lw=1, ls=":", label=f"optimum theta* = {th_star:.3f}")
        ax.set_xlabel("policy gain theta  (mu(s) = theta s)")
        ax.set_ylabel("J(theta)")
        ax.set_ylim(-6, -1)
        ax.set_title("DPG theorem gives the exact slope")
        ax.legend(loc="lower center")
        ax = axes[1]
        r = np.array(rows)
        ax.loglog(r[:, 0], r[:, 3], "o-", color=C[1], label="score fn x q (no baseline)")
        ax.loglog(r[:, 0], r[:, 5], "d-", color=C[4], label="score fn x (sampled return - v)")
        ax.loglog(r[:, 0], r[:, 4], "s--", color=C[3], label="score fn x exact advantage")
        ax.loglog(r[:, 0], r[:, 6], "^-", color=C[0], label="DPG: grad mu x grad_a q")
        ax.loglog(r[:, 0], r[-1, 5] * (r[-1, 0] / r[:, 0]), color="#8a8985", lw=1, ls=":", label="slope -1 (1/sigma)")
        ax.set_xlabel("exploration std sigma of the Gaussian policy")
        ax.set_ylabel("std of a single-sample gradient estimate")
        ax.set_title("per-sample gradient noise vs sigma")
        ax.legend(fontsize=8)
        fig.tight_layout()
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, "dpg_lqr_check.png")
        fig.savefig(path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
