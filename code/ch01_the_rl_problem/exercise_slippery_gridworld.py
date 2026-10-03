"""Exercise 13 of Chapter 01: a slippery version of the Example 3.5 gridworld.

The deterministic gridworld (gridworld.py) is made stochastic by writing its
four-argument dynamics p(s', r | s, a) directly (Section 6): the intended move
happens with probability 0.8, and each of the two perpendicular moves with
probability 0.1.  The usual rules still apply to whichever move happens (bumping
into a wall: stay and pay -1); in A and B every action still teleports.

We then recompute, with the tools of this chapter,
1. v_pi of the equiprobable random policy (it does not change: why?);
2. v_* and the optimal actions at gamma = 0.9 (Sections 12.3-12.5);
3. the discount factor at which the optimal behaviour from B' switches from
   chasing B to chasing A (0.8484 in the deterministic world; Exercise 11);
4. the long-run reward rate of each bonus cycle and how far each bonus is from B',
   which explain the shift.

"Heads for A" is measured exactly: under the greedy policy, the probability of
collecting A's bonus before B's when starting from B' (an absorption probability,
solved as a linear system like the ones of Section 10.3).

Usage:  python code/ch01_the_rl_problem/exercise_slippery_gridworld.py [--quick]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from gridworld import ARROWS, GridWorld
from mdp import (FiniteMDP, deterministic_policy, evaluate_policy_exact, greedy_actions,
                 greedy_policy, policy_matrices, q_from_v, uniform_policy, value_iteration)

NORTH, SOUTH, EAST, WEST = 0, 1, 2, 3
PERPENDICULAR = {NORTH: (EAST, WEST), SOUTH: (EAST, WEST), EAST: (NORTH, SOUTH), WEST: (NORTH, SOUTH)}


def slippery_mdp(world: GridWorld, p_intended: float = 0.8) -> FiniteMDP:
    """Four-argument dynamics of the slippery gridworld, written outcome by outcome.

    outcomes[s][a] lists (probability, next_state, reward) triples.  Several entries can
    share the same (s', r) - e.g. both slips in a corner bump into walls - and FiniteMDP
    adds them up, as the definition of p(s', r | s, a) as a pmf requires (Section 6.5).
    """
    p_side = (1.0 - p_intended) / 2.0
    outcomes = []
    for s in range(world.n_states):
        row = []
        for a in range(world.n_actions):
            outs = [(p_intended, *world.transition(s, a))]
            outs += [(p_side, *world.transition(s, d)) for d in PERPENDICULAR[a]]
            row.append(outs)
        outcomes.append(row)
    mdp = FiniteMDP(world.n_states, world.n_actions, outcomes,
                    state_names=[world.state_label(s) for s in range(world.n_states)])
    mdp.validate()  # every p(., . | s, a) sums to 1
    return mdp


def prob_A_before_B(mdp: FiniteMDP, world: GridWorld, gamma: float, start: int) -> float:
    """Pr{collect A's bonus before B's | S_0 = start} under the greedy optimal policy at gamma.

    Make A and B absorbing; h(s) = Pr{hit A first} solves h = P_pi h on the other states,
    with h(A) = 1 and h(B) = 0, i.e. (I - P_pi restricted) h = P_pi[:, A] restricted.
    """
    v, _ = value_iteration(mdp, gamma, tol=1e-11)
    pol = greedy_policy(q_from_v(mdp, v, gamma), tol=1e-9)
    P_pi, _ = policy_matrices(mdp, deterministic_policy(pol, mdp.n_actions))
    A, B = world.index(*world.a_pos), world.index(*world.b_pos)
    other = np.array([s not in (A, B) for s in range(mdp.n_states)])
    h = np.zeros(mdp.n_states)
    h[A] = 1.0
    h[other] = np.linalg.solve(np.eye(other.sum()) - P_pi[np.ix_(other, other)], P_pi[other][:, A])
    return float(h[start])


def switch_gamma(mdp: FiniteMDP, world: GridWorld, lo: float, hi: float, iters: int) -> tuple[float, float]:
    """Bisection on gamma for the point where B' starts heading for A (probability crosses 1/2)."""
    B2 = world.index(*world.b_prime)
    assert prob_A_before_B(mdp, world, lo, B2) < 0.5 < prob_A_before_B(mdp, world, hi, B2)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if prob_A_before_B(mdp, world, mid, B2) > 0.5:
            hi = mid
        else:
            lo = mid
    return lo, hi


def expected_steps(P_pi: np.ndarray, target: int, start: int) -> float:
    """E[number of steps to reach ``target`` from ``start``] under the chain P_pi.

    t(s) = 1 + sum_s' P_pi(s, s') t(s') for s != target, t(target) = 0: a linear system on the
    other states, like the expected-episode-length computation of Section 10.3 / Exercise 9.
    """
    other = np.arange(P_pi.shape[0]) != target
    t = np.zeros(P_pi.shape[0])
    t[other] = np.linalg.solve(np.eye(other.sum()) - P_pi[np.ix_(other, other)], np.ones(other.sum()))
    return float(t[start])


def cycle_stats(world: GridWorld, mdp: FiniteMDP, bonus_pos, landing_pos, gamma: float = 0.999):
    """Statistics of the policy that is optimal at gamma close to 1 (only one bonus is active).

    Returns its long-run reward per step (the 'gain': stationary distribution d of P_pi, the left
    eigenvector for eigenvalue 1, dotted with r_pi), the expected number of steps from B' to the
    bonus cell, and the expected length of one bonus cycle (1 step to jump + the walk back).
    """
    v, _ = value_iteration(mdp, gamma, tol=1e-9)
    pol = greedy_policy(q_from_v(mdp, v, gamma), tol=1e-9)
    P_pi, r_pi = policy_matrices(mdp, deterministic_policy(pol, mdp.n_actions))
    w, V = np.linalg.eig(P_pi.T)
    d = np.real(V[:, np.argmin(np.abs(w - 1.0))])
    rate = float((d / d.sum()) @ r_pi)
    bonus, landing, B2 = world.index(*bonus_pos), world.index(*landing_pos), world.index(*world.b_prime)
    return rate, expected_steps(P_pi, bonus, B2), 1.0 + expected_steps(P_pi, bonus, landing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smoke test: coarser bisection")
    args = parser.parse_args()
    t0 = time.perf_counter()
    seed, gamma, p_int = 0, 0.9, 0.8
    iters = 12 if args.quick else 40
    print(f"exercise_slippery_gridworld.py | seed={seed} (no randomness used) gamma={gamma} "
          f"p(intended move)={p_int} bisection steps={iters} quick={args.quick}")

    world = GridWorld()
    det, slip = world.to_mdp(), slippery_mdp(world, p_int)
    n, k = slip.n_states, slip.n_actions
    A, B, B2 = (world.index(*world.a_pos), world.index(*world.b_pos), world.index(*world.b_prime))
    example = [(round(pr, 3), world.state_label(s2), r) for pr, s2, r in slip.outcomes[world.index(1, 3)][NORTH]]
    print("\nexample: outcomes (probability, s', r) of s = (1,3), a = north:", example)

    # ---- 1. the random policy
    pi = uniform_policy(n, k)
    v_det, v_slip = evaluate_policy_exact(det, pi, gamma), evaluate_policy_exact(slip, pi, gamma)
    P_det, _ = policy_matrices(det, pi)
    P_slip, _ = policy_matrices(slip, pi)
    print(f"\n[1] random policy: v(A) = {v_slip[A]:.4f} (deterministic world {v_det[A]:.4f}); "
          f"max |v_slip - v_det| = {np.abs(v_slip - v_det).max():.1e}; "
          f"max |P_pi,slip - P_pi,det| = {np.abs(P_slip - P_det).max():.1e}")

    # ---- 2. optimal values and actions at gamma = 0.9
    vs_det, _ = value_iteration(det, gamma, tol=1e-11)
    vs_slip, sweeps = value_iteration(slip, gamma, tol=1e-11)
    acts = greedy_actions(q_from_v(slip, vs_slip, gamma), tol=1e-7)
    print(f"\n[2] v_* at gamma = {gamma} ({len(sweeps)} value-iteration sweeps):")
    print(np.round(vs_slip.reshape(5, 5), 2))
    print(f"    v_*(A) = {vs_slip[A]:.3f} (deterministic {vs_det[A]:.3f}), v_*(B) = {vs_slip[B]:.3f} "
          f"(deterministic {vs_det[B]:.3f}), v_*(B') = {vs_slip[B2]:.3f} (deterministic {vs_det[B2]:.3f})")
    print("    optimal actions:")
    for r in range(5):
        print("     ", " ".join("".join(ARROWS[a] for a in acts[world.index(r, c)]).ljust(4) for c in range(5)))
    print(f"    Pr(collect A before B from B') under the optimal policy at gamma = {gamma}: "
          f"slippery {prob_A_before_B(slip, world, gamma, B2):.3f}, "
          f"deterministic {prob_A_before_B(det, world, gamma, B2):.3f}")

    # ---- 3. where does B' switch from B to A?
    lo_d, hi_d = switch_gamma(det, world, 0.80, 0.90, iters)
    lo_s, hi_s = switch_gamma(slip, world, 0.90, 0.995, iters)
    print(f"\n[3] switch point (B' starts heading for A): deterministic in [{lo_d:.5f}, {hi_d:.5f}] "
          f"(Exercise 11: 0.84837); slippery in [{lo_s:.5f}, {hi_s:.5f}]")
    for g in (round(lo_s - 0.001, 3), round(hi_s + 0.001, 3)):
        print(f"    gamma = {g}: Pr(A before B from B') = {prob_A_before_B(slip, world, g, B2):.3f}")

    # ---- 4. why: each bonus cycle on its own (the other bonus switched off), near gamma = 1
    print("\n[4] each bonus on its own (other bonus set to 0; policy optimal at gamma = 0.999):")
    rates = {}
    for name, kw, pos, landing in (("A", dict(r_b=0.0), world.a_pos, world.a_prime),
                                   ("B", dict(r_a=0.0), world.b_pos, world.b_prime)):
        w2 = GridWorld(**kw)
        for label, m2 in (("deterministic", w2.to_mdp()), ("slippery", slippery_mdp(w2, p_int))):
            rate, to_bonus, cycle = cycle_stats(w2, m2, pos, landing)
            rates[name, label] = rate
            print(f"    {name} only, {label:<13}: reward per step {rate:.4f}; expected steps from B' to {name} "
                  f"{to_bonus:.3f}; expected cycle length {cycle:.3f}")
    for label in ("deterministic", "slippery"):
        print(f"    A's advantage in reward rate, {label}: {rates['A', label] / rates['B', label]:.3f}x")

    print(f"\nSUMMARY: slipping leaves the random policy's values unchanged (v(A) = {v_slip[A]:.4f}); "
          f"v_*(A) drops from {vs_det[A]:.2f} to {vs_slip[A]:.2f}; the B' switch moves from gamma = "
          f"{0.5 * (lo_d + hi_d):.4f} to {0.5 * (lo_s + hi_s):.4f}")
    print(f"done in {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
