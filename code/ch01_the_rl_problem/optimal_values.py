"""Optimal values and optimal policies of the Example 3.5 gridworld.

Chapter 01, Sections 12-13.  We compute v_* three ways and check they agree:

1. value iteration  v_{k+1}(s) = max_a [ r(s,a) + gamma sum_s' p(s'|s,a) v_k(s') ]   (Chapter 03 proves it converges);
2. the linear program  min sum_s v(s)  s.t.  v(s) >= r(s,a) + gamma sum_s' p(s'|s,a) v(s')  for all (s,a)
   (an exact solution of the Bellman optimality equation, solved by SciPy's HiGHS);
3. the closed form v_*(A) = 10 / (1 - gamma^5) from the 5-step A -> A' -> A cycle.

Then we extract the greedy policies from q_*, check by exact policy evaluation
that they achieve v_* (Section 12.4), compare with Sutton & Barto (2018) Figure 3.5,
show one step of greedy improvement of the random policy (a preview of Chapter 03),
estimate how long brute-force enumeration of all 4^25 deterministic policies would take,
and sweep gamma to show that the discount factor changes *what is optimal*.

Usage:  python code/ch01_the_rl_problem/optimal_values.py [--quick]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# One BLAS thread, set before NumPy loads: the brute-force timing in step [6] is then a
# single-core figure (it still varies with the load on a shared machine).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
from scipy.optimize import linprog  # noqa: E402

from gridworld import ARROWS, GridWorld, draw_policy  # noqa: E402
from mdp import (  # noqa: E402
deterministic_policy, evaluate_policy_exact, greedy_actions, greedy_policy,
                 policy_matrices, q_from_v, uniform_policy, value_iteration)

# Sutton & Barto (2018), Figure 3.5 (middle): v_* rounded to 0.1.
SB_FIG_3_5 = np.array([
    [22.0, 24.4, 22.0, 19.4, 17.5],
    [19.8, 22.0, 19.8, 17.8, 16.0],
    [17.8, 19.8, 17.8, 16.0, 14.4],
    [16.0, 17.8, 16.0, 14.4, 13.0],
    [14.4, 16.0, 14.4, 13.0, 11.7],
])


def solve_bellman_optimality_lp(mdp, gamma: float) -> np.ndarray:
    """v_* as the smallest v satisfying v >= T* v componentwise (Section 13)."""
    P, R = mdp.transition_tensor(), mdp.expected_reward()
    n, k = mdp.n_states, mdp.n_actions
    A_ub = np.zeros((n * k, n))
    b_ub = np.zeros(n * k)
    for s in range(n):
        for a in range(k):
            # r(s,a) + gamma P[s,a] v <= v(s)   <=>   (gamma P[s,a] - e_s) v <= -r(s,a)
            A_ub[s * k + a] = gamma * P[s, a]
            A_ub[s * k + a, s] -= 1.0
            b_ub[s * k + a] = -R[s, a]
    res = linprog(c=np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * n, method="highs")
    assert res.success, res.message
    return res.x


def first_special_state_reached(world: GridWorld, policy: np.ndarray, start: int) -> str:
    """Follow a deterministic policy from ``start`` until it collects A's or B's reward."""
    s = start
    for _ in range(4 * world.n_states):
        label = world.state_label(s)
        if label in ("A", "B"):
            return label
        s, _ = world.transition(s, int(policy[s]))
    return "neither"


def policy_grid_text(world: GridWorld, action_sets) -> str:
    rows = []
    for r in range(world.size):
        rows.append("  " + " ".join("".join(ARROWS[a] for a in action_sets[world.index(r, c)]).ljust(4)
                                    for c in range(world.size)))
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smoke test: smaller sweeps, no figures")
    args = parser.parse_args()
    t0 = time.perf_counter()
    seed, gamma = 0, 0.9
    rng = np.random.default_rng(seed)
    np.set_printoptions(precision=2, suppress=True, linewidth=110)
    print(f"optimal_values.py | seed={seed} gamma={gamma} quick={args.quick}")

    world = GridWorld()
    mdp = world.to_mdp()
    n, k = mdp.n_states, mdp.n_actions
    A, B, B2 = (world.index(*world.a_pos), world.index(*world.b_pos), world.index(*world.b_prime))

    # ---- 1. value iteration
    v_vi, deltas = value_iteration(mdp, gamma, tol=1e-10)
    print(f"\n[1] value iteration: {len(deltas)} sweeps until max-change < 1e-10")
    print(np.round(v_vi.reshape(5, 5), 1))
    print(f"    max |round(v_*,1) - S&B Fig 3.5| = {np.abs(np.round(v_vi.reshape(5, 5), 1) - SB_FIG_3_5).max():.3f}")

    # ---- 2. linear programming
    t_lp = time.perf_counter()
    v_lp = solve_bellman_optimality_lp(mdp, gamma)
    t_lp = time.perf_counter() - t_lp
    print(f"[2] linear program ({n} variables, {n * k} constraints, {t_lp * 1e3:.1f} ms): "
          f"max |v_LP - v_VI| = {np.abs(v_lp - v_vi).max():.2e}")

    # ---- 3. closed form at A and B
    vA_closed = 10 / (1 - gamma ** 5)
    print(f"[3] closed form: v_*(A) = 10/(1-0.9^5) = {vA_closed:.4f} (VI: {v_vi[A]:.4f}); "
          f"v_*(B) = 5 + 0.9^5 v_*(A) = {5 + gamma ** 5 * vA_closed:.4f} (VI: {v_vi[B]:.4f})")

    # ---- 4. q_*, greedy policies, and their exact values
    v_star = v_vi
    q_star = q_from_v(mdp, v_star, gamma)
    acts = greedy_actions(q_star, tol=1e-6)
    print("\n[4] optimal actions (all maximizers of q_*(s, .)):")
    print(policy_grid_text(world, acts))
    resid = np.abs(q_star.max(axis=1) - v_star).max()
    print(f"    Bellman optimality residual max_s |max_a q_*(s,a) - v_*(s)| = {resid:.2e}")
    pi_g = greedy_policy(q_star, tol=1e-6)
    v_g = evaluate_policy_exact(mdp, deterministic_policy(pi_g, k), gamma)
    # a stochastic policy that randomizes uniformly over the tied optimal actions is optimal too
    pi_mix = np.zeros((n, k))
    for s, a_set in enumerate(acts):
        pi_mix[s, a_set] = 1.0 / len(a_set)
    v_mix = evaluate_policy_exact(mdp, pi_mix, gamma)
    print(f"    greedy deterministic policy:   max |v_pi - v_*| = {np.abs(v_g - v_star).max():.2e}")
    print(f"    uniform mix of optimal actions: max |v_pi - v_*| = {np.abs(v_mix - v_star).max():.2e}")
    n_opt = int(np.prod([len(a) for a in acts]))
    print(f"    number of optimal deterministic policies = product of tie-set sizes = {n_opt}")

    # ---- 5. one greedy improvement step from the random policy (preview of Chapter 03)
    v_rand = evaluate_policy_exact(mdp, uniform_policy(n, k), gamma)
    pi_1 = greedy_policy(q_from_v(mdp, v_rand, gamma))
    v_1 = evaluate_policy_exact(mdp, deterministic_policy(pi_1, k), gamma)
    print(f"\n[5] greedy w.r.t. v_random: v improves in every state: {bool(np.all(v_1 >= v_rand - 1e-12))}; "
          f"v(A): random {v_rand[A]:.2f} -> greedy {v_1[A]:.2f} -> optimal {v_star[A]:.2f}; "
          f"already optimal everywhere: {bool(np.allclose(v_1, v_star))}")

    # ---- 6. brute force is hopeless
    n_pol = k ** n
    batch = 200 if args.quick else 4000
    pols = rng.integers(k, size=(batch, n))
    P, R = mdp.transition_tensor(), mdp.expected_reward()
    t_b = time.perf_counter()
    P_b = P[np.arange(n)[None, :], pols]              # (batch, n, n): P_pi for each policy
    r_b = R[np.arange(n)[None, :], pols]              # (batch, n)
    V_b = np.linalg.solve(np.eye(n)[None] - gamma * P_b, r_b[..., None])[..., 0]
    per_policy = (time.perf_counter() - t_b) / batch
    years = per_policy * n_pol / (3600 * 24 * 365.25)
    print(f"\n[6] brute force: |A|^|S| = 4^25 = {n_pol:.3e} deterministic policies; "
          f"batched exact evaluation costs {per_policy * 1e6:.1f} us/policy here "
          f"-> about {years:,.0f} years on this CPU")
    print(f"    (none of {batch} random deterministic policies beats v_* in any state: "
          f"{bool(np.all(V_b <= v_star + 1e-9))}; best state-averaged value among them "
          f"{V_b.mean(axis=1).max():.2f} vs {v_star.mean():.2f} for v_*)")

    # ---- 7. gamma changes what is optimal
    gammas = np.round(np.arange(0.50, 0.991, 0.05 if args.quick else 0.01), 4)
    target = []
    vb2 = []
    for g in gammas:
        vg, _ = value_iteration(mdp, g, tol=1e-10)
        pol = greedy_policy(q_from_v(mdp, vg, g), tol=1e-9)
        target.append(first_special_state_reached(world, pol, B2))
        vb2.append(vg[B2])
    root = [r.real for r in np.roots([1, 1, 1, -1, -1]) if abs(r.imag) < 1e-12 and 0 < r.real < 1][0]
    switch = next(g for g, t in zip(gammas, target) if t == "A")
    print(f"\n[7] from B' the optimal policy heads for: "
          + ", ".join(f"{g:.2f}:{t}" for g, t in zip(gammas, target) if abs((g * 100) % 5) < 1e-6))
    print(f"    first gamma on the grid where B' heads for A: {switch:.2f}; analytic switch point "
          f"(root of g^4+g^3+g^2-g-1=0, Exercise 11): {root:.4f}")
    # B' is not the only state that changes its mind: compare just below and just above the root.
    lo_g, hi_g = 0.848, 0.849
    sets, heads = {}, {}
    for g in (lo_g, hi_g):
        vg, _ = value_iteration(mdp, g, tol=1e-12)
        qg = q_from_v(mdp, vg, g)
        sets[g] = greedy_actions(qg, tol=1e-7)
        pol = greedy_policy(qg, tol=1e-9)
        heads[g] = [first_special_state_reached(world, pol, s) for s in range(n)]
    changed_sets = [world.state_label(s) for s in range(n) if sets[lo_g][s] != sets[hi_g][s]]
    changed_heads = [world.state_label(s) for s in range(n) if heads[lo_g][s] != heads[hi_g][s]]
    print(f"    between gamma = {lo_g} and {hi_g}: the set of optimal actions changes in {len(changed_sets)} "
          f"states {changed_sets};")
    print(f"    {len(changed_heads)} states switch from heading for B to heading for A: {changed_heads}")

    if not args.quick:
        import matplotlib.pyplot as plt
        from plotting import BLUE, INK, INK2, ORANGE, grid_heatmap, setup_style
        setup_style()
        figdir = Path(__file__).parent / "figures"
        figdir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
        im = grid_heatmap(axes[0], v_star.reshape(5, 5), r"$v_\ast$ ($\gamma = 0.9$)")
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        ax = axes[1]
        grid_heatmap(ax, np.zeros((5, 5)), r"$\pi_\ast$: all actions maximizing $q_\ast(s,\cdot)$",
                     fmt="", vmin=0, vmax=1)
        draw_policy(ax, world, acts, color=INK)
        for pos, lab in [(world.a_pos, "A"), (world.a_prime, "A'"), (world.b_pos, "B"), (world.b_prime, "B'")]:
            ax.text(pos[1] - 0.42, pos[0] - 0.42, lab, fontsize=9, color=INK2, ha="left", va="top")
        fig.tight_layout()
        fig.savefig(figdir / "optimal_values_policy.png")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), gridspec_kw={"width_ratios": [1, 1, 1.35]})
        for ax, g in zip(axes[:2], [0.8, 0.9]):
            vg, _ = value_iteration(mdp, g, tol=1e-10)
            a_g = greedy_actions(q_from_v(mdp, vg, g), tol=1e-6)
            grid_heatmap(ax, np.zeros((5, 5)), f"optimal policy, $\\gamma$ = {g}", fmt="", vmin=0, vmax=1)
            draw_policy(ax, world, a_g, color=INK)
            for s in range(n):
                r_, c_ = world.coords(s)
                ax.text(c_ + 0.46, r_ + 0.46, f"{vg[s]:.1f}", fontsize=7.5, color=INK2, ha="right", va="bottom")
            for pos, lab in [(world.a_pos, "A"), (world.b_pos, "B"), (world.b_prime, "B'")]:
                ax.text(pos[1] - 0.44, pos[0] - 0.44, lab, fontsize=9, color=ORANGE, ha="left", va="top",
                        fontweight="bold")
        ax = axes[2]
        gg = np.linspace(0.5, 0.995, 400)
        # Multiplying by (1 - gamma) keeps the curves bounded: as gamma -> 1 they tend to the
        # reward *rate* of each cycle, 10/5 = 2 (A) and 5/3 (B).
        ax.plot(gg, (1 - gg) * gg ** 4 * 10 / (1 - gg ** 5), color=BLUE,
                label=r"go to A, then cycle: $\gamma^4\,10/(1-\gamma^5)$")
        ax.plot(gg, (1 - gg) * 5 * gg ** 2 / (1 - gg ** 3), color=ORANGE,
                label=r"back to B, then cycle: $5\gamma^2/(1-\gamma^3)$")
        ax.plot(gammas, (1 - gammas) * np.array(vb2), "o", ms=4, color=INK,
                label=r"$v_\ast(B')$ from value iteration")
        ax.axvline(root, color=INK2, ls="--", lw=1)
        ax.text(root - 0.01, 0.35, f"switch at\n$\\gamma$ = {root:.4f}", color=INK2, fontsize=9, ha="right")
        ax.set_xlabel(r"discount factor $\gamma$")
        ax.set_ylabel(r"$(1-\gamma)\times$ value of B'")
        ax.set_title("Which bonus to chase from B'?")
        ax.legend(loc="upper left", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(figdir / "optimal_policy_vs_gamma.png")
        plt.close(fig)
        print(f"\nsaved figures to {figdir}")

    print(f"\nSUMMARY: v_*(A) = {v_star[A]:.3f} (book 24.4, closed form {vA_closed:.3f}); "
          f"v_*(B) = {v_star[B]:.3f}; VI and LP agree to {np.abs(v_lp - v_vi).max():.1e}; "
          f"greedy policy is optimal; optimal behaviour at B' switches at gamma ~ {root:.4f}")
    print(f"done in {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
