"""Numerical checks for the Chapter 03 exercises, and the two coding exercises.

  Ex. 1   first in-place sweep on the 4x4 gridworld, by hand vs code
  Ex. 3   predicted vs measured number of value-iteration sweeps
  Ex. 4   greedy policy w.r.t. an arbitrary V on the maintenance MDP, and the Singh-Yee bound
  Ex. 5   with termination probability >= beta, T* contracts with modulus gamma (1 - beta)
  Ex. 8   dual LP by hand with mu = (1, 0)
  Ex. 9   finite horizon on the maintenance MDP = value iteration iterates
  Ex. 11  the policy improvement theorem fails for gamma = 1 when pi' is improper
  Ex. 12  Jack's car rental (S&B Example 4.2) by policy iteration
  Ex. 13  "simple" (one-switch) policy iteration vs Howard's policy iteration
  Ex. 14  policy iteration and value iteration written in terms of action values Q (S&B Ex. 4.5, 4.10)

Run:  python code/ch03_dynamic_programming/exercise_solutions.py [--quick]
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from dp import (TabularMDP, as_matrix, backward_induction, bellman_optimality, evaluate_exact,  # noqa: E402
                evaluate_iterative, greedy, greedy_from_q, policy_iteration, q_from_v, q_value_iteration,
                uniform_policy, value_iteration)
from mdps import frozenlake_mdp, gridworld_4x4, maintenance_mdp, random_mdp  # noqa: E402

SEED = 0


def ex1():
    mdp = gridworld_4x4()
    V, _, hist = evaluate_iterative(mdp, uniform_policy(mdp), theta=0.0, inplace=True, max_sweeps=1,
                                    record=True)
    print("Ex 1: first in-place sweep (order 0..15), states 1..5:", np.round(hist[1][1:6], 4).tolist())


def ex3():
    print("Ex 3: VI sweeps until ||V - v*|| <= 1e-6 is guaranteed by the a-priori bound, R_max = 1, V_0 = 0:")
    for g in (0.9, 0.99, 0.999):
        k = math.log(1 / (1e-6 * (1 - g))) / math.log(1 / g)
        print(f"   gamma = {g}: k >= {k:.0f}   (approximation log(1/(eps(1-gamma)))/(1-gamma) = "
              f"{math.log(1 / (1e-6 * (1 - g))) / (1 - g):.0f})")


def ex4():
    mdp = maintenance_mdp(0.8)
    V = np.array([0.0, 10.0])
    pi = greedy(mdp, V)
    v_pi = evaluate_exact(mdp, pi)
    v_star = np.array([7.5, 5.0])
    eps = np.max(np.abs(V - v_star))
    print(f"Ex 4: greedy w.r.t. V = (0, 10): {[mdp.action_names[a] for a in pi]}, q_V = "
          f"{np.round(q_from_v(mdp, V), 3).tolist()}, v_pi = {v_pi.tolist()}, loss = {np.max(v_star - v_pi)}, "
          f"bound 2 gamma eps/(1-gamma) = {2 * 0.8 * eps / 0.2}")


def ex5(rng):
    beta, gamma = 0.3, 0.9
    mdp = random_mdp(20, 3, gamma, branching=4, seed=SEED)
    mdp = TabularMDP(P=mdp.P * (1 - beta), R=mdp.R, gamma=gamma)       # terminate w.p. beta each step
    worst = 0.0
    for _ in range(2000):
        u = rng.normal(0, 5, 20)
        w = u + rng.normal(0, 3, 20)
        worst = max(worst, np.max(np.abs(bellman_optimality(mdp, u) - bellman_optimality(mdp, w)))
                    / np.max(np.abs(u - w)))
        shift = np.max(np.abs(bellman_optimality(mdp, u + 1.0) - bellman_optimality(mdp, u)))
    print(f"Ex 5: beta = {beta}, gamma = {gamma}: max observed ratio {worst:.4f}; constant shift ratio "
          f"{shift:.4f} = gamma (1 - beta) = {gamma * (1 - beta):.4f}")


def ex8():
    mdp = maintenance_mdp(0.8)
    P_pi = mdp.P[[0, 1], [0, 1]]                                        # (run, fix)
    x = np.linalg.solve((np.eye(2) - 0.8 * P_pi).T, np.array([1.0, 0.0]))
    obj = x[0] * mdp.R[0, 0] + x[1] * mdp.R[1, 1]
    print(f"Ex 8: mu = (1, 0): x(G, run) = {x[0]:.4f}, x(W, fix) = {x[1]:.4f}, sum = {x.sum():.4f}, "
          f"objective = {obj:.4f}")


def ex9():
    mdp = maintenance_mdp(0.8)
    V, pi = backward_induction(mdp, 4)
    for t in range(4):
        print(f"Ex 9: {4 - t} steps to go: V = {np.round(V[t], 4).tolist()}, "
              f"policy = {[mdp.action_names[a] for a in pi[t]]}")


def ex11():
    # One state s, actions exit (+1, terminate) and stay (0, back to s); gamma = 1.
    P = np.zeros((1, 2, 1)); P[0, 1, 0] = 1.0
    R = np.array([[1.0, 0.0]])
    mdp = TabularMDP(P=P, R=R, gamma=1.0)
    v_exit = evaluate_exact(mdp, np.array([0]))
    q = q_from_v(mdp, v_exit)
    print(f"Ex 11: v_exit = {v_exit[0]}, q_exit(s, stay) = {q[0, 1]} >= v_exit(s): hypothesis holds; "
          f"but 'stay' never terminates and collects 0 forever: v_stay = 0 < 1")


# ---------------------------------------------------------------------------------------------
# Ex. 12: Jack's car rental
# ---------------------------------------------------------------------------------------------
def poisson(lam, n):
    return np.array([math.exp(-lam) * lam ** k / math.factorial(k) for k in range(n)])


def location_model(lam_req, lam_ret, max_cars=20, n_trunc=30):
    """For each morning count m: expected rentals E[min(req, m)] and P(evening count n' | m)."""
    p_req, p_ret = poisson(lam_req, n_trunc), poisson(lam_ret, n_trunc)
    p_req[-1] += 1 - p_req.sum(); p_ret[-1] += 1 - p_ret.sum()        # fold the tail into the last bin
    exp_rent = np.zeros(max_cars + 1)
    trans = np.zeros((max_cars + 1, max_cars + 1))
    for m in range(max_cars + 1):
        for req, pq in enumerate(p_req):
            rented = min(req, m)
            exp_rent[m] += pq * rented
            for ret, pr in enumerate(p_ret):
                trans[m, min(m - rented + ret, max_cars)] += pq * pr  # cars above 20 disappear
    return exp_rent, trans


def jacks_car_rental(max_cars=20, max_move=5, gamma=0.9, n_trunc=30):
    e1, T1 = location_model(3, 3, max_cars, n_trunc)
    e2, T2 = location_model(4, 2, max_cars, n_trunc)
    n = max_cars + 1
    S, A = n * n, 2 * max_move + 1                       # action index k <-> move a = k - max_move (1 -> 2)
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    valid = np.zeros((S, A), dtype=bool)
    for n1 in range(n):
        for n2 in range(n):
            s = n1 * n + n2
            for k in range(A):
                a = k - max_move
                if a > n1 or -a > n2:                     # cannot move cars you do not have
                    continue
                valid[s, k] = True
                m1, m2 = min(n1 - a, max_cars), min(n2 + a, max_cars)
                R[s, k] = 10 * (e1[m1] + e2[m2]) - 2 * abs(a)
                P[s, k] = np.outer(T1[m1], T2[m2]).ravel()
    return TabularMDP(P=P, R=R, gamma=gamma, valid=valid, name="jack", shape=(n, n))


def ex12(quick):
    t0 = time.perf_counter()
    mdp = jacks_car_rental()
    zero = np.full(mdp.S, 5)                              # pi_0: never move a car (index 5 <-> a = 0)
    acts, V, hist = policy_iteration(mdp, zero)
    print(f"Ex 12: Jack's car rental: |S| = {mdp.S}, |A| = {mdp.A}; policy iteration from 'never move' needs "
          f"{len(hist)} evaluations ({len(hist) - 1} improvements), {time.perf_counter() - t0:.1f} s")
    print(f"   actions changed per improvement: {[h['n_changed'] for h in hist[:-1]]}")
    print(f"   v_* ranges from {V.min():.1f} to {V.max():.1f}; v_*(10, 10) = {V[10 * 21 + 10]:.1f}")
    # Sensitivity to the Poisson truncation (tails folded into the last bin): 12 instead of 30 terms.
    m12 = jacks_car_rental(n_trunc=12)
    a12, V12, _ = policy_iteration(m12, zero)
    print(f"   truncating the Poisson tails at 12 instead of 30 terms: max |v_* difference| = "
          f"{np.max(np.abs(V12 - V)):.3f}, optimal actions differ in {int(np.sum(a12 != acts))} states")
    moves = (acts - 5).reshape(21, 21)
    print("   optimal move (cars from 1 to 2), rows n1 = 20..0 (top to bottom), columns n2 = 0..20:")
    for n1 in range(20, -1, -4):
        print(f"     n1={n1:2d}: " + " ".join(f"{x:+d}" for x in moves[n1]))
    if not quick:
        from plotting import plt, save
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        im = axes[0].imshow(moves, origin="lower", cmap="RdBu_r", vmin=-5, vmax=5)
        axes[0].set_xlabel("cars at location 2"); axes[0].set_ylabel("cars at location 1")
        axes[0].set_title(r"$\pi_\ast$: cars moved from 1 to 2"); fig.colorbar(im, ax=axes[0])
        im = axes[1].imshow(V.reshape(21, 21), origin="lower", cmap="viridis")
        axes[1].set_xlabel("cars at location 2"); axes[1].set_ylabel("cars at location 1")
        axes[1].set_title(r"$v_\ast$"); fig.colorbar(im, ax=axes[1])
        for ax in axes:                                   # car counts are integers
            ax.set_xticks(range(0, 21, 5)); ax.set_yticks(range(0, 21, 5))
        save(fig, "jack_car_rental.png")


def simple_pi(mdp, tol=1e-9):
    """One-switch policy iteration: change only the state with the largest improvement q(s,a) - v(s)."""
    acts = np.zeros(mdp.S, dtype=int)
    for it in range(1, 10 ** 6):
        v = evaluate_exact(mdp, acts)
        Q = q_from_v(mdp, v)
        gain = Q.max(axis=1) - v
        s = int(np.argmax(gain))
        if gain[s] <= tol * max(1.0, abs(v[s])):
            return acts, v, it
        acts[s] = int(np.argmax(Q[s]))


def ex13(quick):
    n = 5 if quick else 20
    howard, simple, n_diff = [], [], []
    for i in range(n):
        mdp = random_mdp(50, 5, 0.9, branching=5, seed=1000 + i)
        a1, v1, h = policy_iteration(mdp, np.zeros(mdp.S, dtype=int))
        a2, v2, it = simple_pi(mdp)
        assert np.allclose(v1, v2, atol=1e-8)
        howard.append(len(h)); simple.append(it)
        n_diff.append(int(np.sum(a1 != 0)))              # states whose optimal action differs from the start
    print(f"Ex 13: {n} random MDPs (S=50, A=5, gamma=0.9), evaluations until optimal: Howard PI "
          f"mean {np.mean(howard):.1f} (max {max(howard)}), one-switch PI mean {np.mean(simple):.1f} "
          f"(max {max(simple)}); states whose optimal action differs from the initial one: mean "
          f"{np.mean(n_diff):.1f}")


def q_policy_iteration(mdp, actions0, tol=1e-9):
    """Ex. 14: policy iteration on action values. Evaluate q_pi EXACTLY by solving the |S||A| linear system
    q = r + gamma P Pi q, with (P Pi)[(s,a), (s',a')] = p(s'|s,a) pi(a'|s'); improve with argmax_a q(s, a)."""
    S, A = mdp.S, mdp.A
    acts, n_eval = np.array(actions0), 0
    while True:
        Pi = as_matrix(mdp, acts)                                       # (S, A) one-hot
        PPi = (mdp.P.reshape(S * A, S)[:, :, None] * Pi[None, :, :]).reshape(S * A, S * A)
        q = np.linalg.solve(np.eye(S * A) - mdp.gamma * PPi, mdp.R.ravel()).reshape(S, A)
        n_eval += 1
        new = greedy_from_q(np.where(mdp.valid, q, -np.inf), acts, tol)  # model-free improvement step
        if np.array_equal(new, acts):
            return acts, q, n_eval
        acts = new


def ex14():
    for name, mdp in (("maintenance", maintenance_mdp(0.8)), ("FrozenLake 8x8", frozenlake_mdp("8x8", 0.99)[0])):
        start = np.full(mdp.S, 1 if name == "maintenance" else 0)          # (fix, fix) / always left
        a_q, q, n_q = q_policy_iteration(mdp, start)
        a_v, v, hist = policy_iteration(mdp, start)
        Qvi, sweeps = q_value_iteration(mdp, theta=1e-12)
        print(f"Ex 14: {name}: q-policy iteration {n_q} evaluations (v-policy iteration {len(hist)}), "
              f"same policy: {np.array_equal(a_q, a_v)};  max|max_a q - v*| = {np.max(np.abs(q.max(axis=1) - v)):.1e};"
              f"  q-value iteration: {sweeps} sweeps, max|Q - q*| = "
              f"{np.max(np.abs(np.where(mdp.valid, Qvi - q, 0))):.1e}")
    q = q_policy_iteration(maintenance_mdp(0.8), np.array([1, 1]))[1]
    print(f"   maintenance q* = [[G: run {q[0, 0]:.3f}, fix {q[0, 1]:.3f}], [W: run {q[1, 0]:.3f}, fix {q[1, 1]:.3f}]]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(f"exercise_solutions.py  seed={SEED}  quick={args.quick}")
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    ex1(); ex3(); ex4(); ex5(rng); ex8(); ex9(); ex11()
    ex12(args.quick)
    ex13(args.quick)
    ex14()
    print(f"\nDone in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
