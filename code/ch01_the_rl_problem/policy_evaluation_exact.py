"""Exact policy evaluation by solving the Bellman expectation equation as a linear system.

Chapter 01, Sections 8-10.  For the gridworld of Example 3.5 (gamma = 0.9) and
the equiprobable random policy we

1. build the induced Markov chain P_pi and expected rewards r_pi       (Section 7);
2. solve (I - gamma P_pi) v = r_pi                                        (Section 10);
3. compare with the values printed in Sutton & Barto (2018), Figure 3.2;
4. check the one-step relations v <-> q and the Bellman equation by hand  (Sections 8-9);
5. look at why I - gamma P_pi is invertible: spectrum, the Neumann series
   v = sum_k gamma^k P_pi^k r_pi, and the discounted occupancy (1-gamma)(I-gamma P_pi)^{-1};
6. sweep gamma to see how values and conditioning change (infinity-norm and 2-norm
   condition numbers, number of positive values, and (1-gamma) v_pi -> average reward).

Usage:  python code/ch01_the_rl_problem/policy_evaluation_exact.py [--quick]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from gridworld import ACTION_NAMES, GridWorld
from mdp import evaluate_policy_exact, policy_matrices, q_from_v, uniform_policy, v_from_q

# Sutton & Barto (2018), Figure 3.2: v_pi of the equiprobable random policy, rounded to 0.1.
SB_FIG_3_2 = np.array([
    [3.3, 8.8, 4.4, 5.3, 1.5],
    [1.5, 3.0, 2.3, 1.9, 0.5],
    [0.1, 0.7, 0.7, 0.4, -0.4],
    [-1.0, -0.4, -0.4, -0.6, -1.2],
    [-1.9, -1.3, -1.2, -1.4, -2.0],
])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smoke test: no figures")
    args = parser.parse_args()
    t0 = time.perf_counter()
    seed, gamma = 0, 0.9  # no randomness is used here; the seed is printed for uniformity
    np.set_printoptions(precision=3, suppress=True, linewidth=110)
    print(f"policy_evaluation_exact.py | seed={seed} gamma={gamma} quick={args.quick}")

    world = GridWorld()
    mdp = world.to_mdp()
    n = mdp.n_states
    pi = uniform_policy(n, mdp.n_actions)
    idx = world.index
    A, A2, B, B2 = idx(*world.a_pos), idx(*world.a_prime), idx(*world.b_pos), idx(*world.b_prime)

    # ---- 1-2. the induced chain and the linear solve (Section 10)
    P_pi, r_pi = policy_matrices(mdp, pi)
    assert np.allclose(P_pi.sum(axis=1), 1.0)  # P_pi is row-stochastic
    v = evaluate_policy_exact(mdp, pi, gamma)
    grid = v.reshape(world.size, world.size)
    print("\nv_pi for the equiprobable random policy (exact, linear solve):")
    print(np.round(grid, 1))

    # ---- 3. compare with the book
    diff = np.abs(np.round(grid, 1) - SB_FIG_3_2)
    print(f"max |round(v,1) - S&B Fig 3.2| = {diff.max():.3f}   "
          f"(max |v - Fig 3.2| = {np.abs(grid - SB_FIG_3_2).max():.3f}, i.e. rounding only)")
    print(f"v(A) = {v[A]:.4f}, v(A') = {v[A2]:.4f}, v(B) = {v[B]:.4f}, v(B') = {v[B2]:.4f}")

    # ---- 4. Bellman equation checks by hand (Section 9)
    print("\nBellman expectation equation, checked at individual states:")
    print(f"  A : 10 + gamma v(A') = 10 + 0.9 * ({v[A2]:.4f}) = {10 + gamma * v[A2]:.4f}  vs v(A) = {v[A]:.4f}")
    print(f"  B :  5 + gamma v(B') =  5 + 0.9 * ({v[B2]:.4f}) = {5 + gamma * v[B2]:.4f}  vs v(B) = {v[B]:.4f}")
    c = idx(2, 2)
    nbrs = [idx(1, 2), idx(3, 2), idx(2, 3), idx(2, 1)]
    print(f"  centre (2,2): 0.25 * 0.9 * (v(1,2) + v(3,2) + v(2,3) + v(2,1)) = 0.225 * "
          f"({' + '.join(f'{v[s]:.3f}' for s in nbrs)}) = {0.225 * v[nbrs].sum():.4f}  vs v = {v[c]:.4f}")
    residual = np.max(np.abs(r_pi + gamma * P_pi @ v - v))
    print(f"  max_s |r_pi + gamma P_pi v - v| = {residual:.2e}")

    q = q_from_v(mdp, v, gamma)  # q(s,a) = r(s,a) + gamma sum_s' p(s'|s,a) v(s')
    print(f"\nq_pi(s, a) from v_pi:  max |sum_a pi(a|s) q(s,a) - v(s)| = {np.max(np.abs(v_from_q(pi, q) - v)):.2e}")
    for s in [A, idx(0, 0), idx(4, 4)]:
        cells = ", ".join(f"{ACTION_NAMES[a]} {q[s, a]:+.3f}" for a in range(4))
        adv = ", ".join(f"{q[s, a] - v[s]:+.3f}" for a in range(4))
        print(f"  s = {world.state_label(s):<6} v = {v[s]:+.3f} | q: {cells} | advantages: {adv}")

    # ---- 5. why the system is invertible (Section 10)
    eig = np.linalg.eigvals(P_pi)
    Mmat = np.linalg.inv(np.eye(n) - gamma * P_pi)  # formed explicitly only to inspect it
    occupancy_rows = (1 - gamma) * Mmat
    M_sys = np.eye(n) - gamma * P_pi
    cond = np.linalg.cond(M_sys)                                         # 2-norm condition number
    kappa_inf = np.abs(M_sys).sum(axis=1).max() * np.abs(Mmat).sum(axis=1).max()   # inf-norm one
    print("\nInvertibility of I - gamma P_pi:")
    print(f"  spectral radius of P_pi = {np.max(np.abs(eig)):.6f}; of gamma P_pi = {gamma * np.max(np.abs(eig)):.6f} < 1")
    print(f"  ||(I - gamma P_pi)^-1||_inf = {np.abs(Mmat).sum(axis=1).max():.4f}  (1/(1-gamma) = {1 / (1 - gamma):.4f})")
    print(f"  (1-gamma)(I - gamma P_pi)^-1: min entry {occupancy_rows.min():.2e} >= 0, "
          f"row sums in [{occupancy_rows.sum(axis=1).min():.6f}, {occupancy_rows.sum(axis=1).max():.6f}]")
    print(f"  condition numbers: kappa_inf = {kappa_inf:.3f} (bound (1+gamma)/(1-gamma) = "
          f"{(1 + gamma) / (1 - gamma):.3f}); kappa_2 = {cond:.2f} (not covered by that bound)")
    d_A = occupancy_rows[A]
    print(f"  discounted occupancy from A: d_A(A') = {d_A[A2]:.3f}, d_A(A) = {d_A[A]:.3f}; "
          f"check v(A) = sum_s d_A(s) r_pi(s) / (1-gamma) = {d_A @ r_pi / (1 - gamma):.4f}")

    # Neumann series: v_M = sum_{k<M} gamma^k P^k r  and  v - v_M = gamma^M P^M v
    M_max = 200
    vM, term, errors = np.zeros(n), r_pi.copy(), []
    for _ in range(M_max):
        vM += term
        term = gamma * P_pi @ term
        errors.append(np.max(np.abs(v - vM)))
    errors = np.array(errors)
    M_needed = int(np.argmax(errors < 1e-6)) + 1
    print(f"  Neumann partial sums: error after M=10 terms: {errors[9]:.3f}, M=50: {errors[49]:.2e}; "
          f"error < 1e-6 first at M = {M_needed}")

    # ---- 6. gamma sweep
    print("\nEffect of gamma on v_pi and on conditioning (#v>0 = number of states with positive value):")
    print(f"  {'gamma':>6} {'v(A)':>9} {'v(B)':>9} {'max|v|':>8} {'#v>0':>5} {'(1-g)v(A)':>10} "
          f"{'kappa_inf':>10} {'(1+g)/(1-g)':>12} {'kappa_2':>9}")
    sweep = {}
    for g in [0.0, 0.5, 0.9, 0.99, 0.995, 0.999, 0.9999]:
        vg = evaluate_policy_exact(mdp, pi, g)
        Mg = np.eye(n) - g * P_pi
        k2 = np.linalg.cond(Mg)
        kinf = np.abs(Mg).sum(axis=1).max() * np.abs(np.linalg.inv(Mg)).sum(axis=1).max()
        sweep[g] = vg
        print(f"  {g:>6} {vg[A]:>9.3f} {vg[B]:>9.3f} {np.abs(vg).max():>8.3f} {int((vg > 0).sum()):>5d} "
              f"{(1 - g) * vg[A]:>10.4f} {kinf:>10.1f} {(1 + g) / (1 - g):>12.1f} {k2:>9.1f}")
    # Long-run average reward of the policy: r_bar = sum_s d(s) r_pi(s), d = stationary distribution
    # of P_pi (left eigenvector for eigenvalue 1).  As gamma -> 1, (1-gamma) v_pi(s) -> r_bar for every s.
    w_eig, V_eig = np.linalg.eig(P_pi.T)
    d_stat = np.real(V_eig[:, np.argmin(np.abs(w_eig - 1.0))])
    d_stat /= d_stat.sum()
    r_bar = d_stat @ r_pi
    print(f"  long-run average reward of the random policy r_bar = {r_bar:.4f} per step; "
          f"(1-gamma) v(A) = {(1 - 0.999) * sweep[0.999][A]:.4f} at gamma=0.999 and "
          f"{(1 - 0.9999) * sweep[0.9999][A]:.4f} at gamma=0.9999 (the limit is approached slowly)")

    if not args.quick:
        import matplotlib.pyplot as plt
        from plotting import BLUE, INK2, ORANGE, grid_heatmap, setup_style
        setup_style()
        figdir = Path(__file__).parent / "figures"
        figdir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.3))
        im = grid_heatmap(axes[0], grid, r"$v_\pi$, equiprobable random policy ($\gamma=0.9$)",
                          diverging=True)
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        im2 = grid_heatmap(axes[1], d_A.reshape(world.size, world.size),
                           r"discounted occupancy $d_A(s)$ starting from A",
                           fmt="{:.2f}", fontsize=9)
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(figdir / "random_policy_values.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        Ms = np.arange(1, M_max + 1)
        ax.semilogy(Ms, errors, color=BLUE, label=r"$\|v_\pi - \sum_{k<M}\gamma^k P_\pi^k r_\pi\|_\infty$")
        ax.semilogy(Ms, gamma ** Ms * np.abs(v).max(), color=ORANGE, ls="--",
                    label=r"bound $\gamma^M \|v_\pi\|_\infty$")
        ax.set_xlabel("number of terms M")
        ax.set_ylabel("max-norm error", color=INK2)
        ax.set_title("Neumann series for $(I-\\gamma P_\\pi)^{-1} r_\\pi$ converges like $\\gamma^M$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figdir / "neumann_series_convergence.png")
        plt.close(fig)
        print(f"\nsaved figures to {figdir}")

    print(f"\nSUMMARY: v_pi(A) = {v[A]:.3f} (book 8.8), v_pi(B) = {v[B]:.3f} (book 5.3); "
          f"all 25 values match Fig. 3.2 after rounding: {diff.max() < 1e-9}")
    print(f"done in {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
