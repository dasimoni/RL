"""Policies are only partially ordered, yet one deterministic stationary policy beats them all.

Chapter 01, Sections 6.3, 10 and 12.

Part 1 - the two-state machine-maintenance MDP worked by hand in the chapter
         (gamma = 0.8): its four-argument dynamics, derived quantities, the
         values of its four deterministic policies, q_*, and the discount factor
         at which "repair the worn machine" stops being optimal (gamma = 4/7).
Part 2 - the *value polytope*: the set {v_pi} of all stationary stochastic
         policies drawn in the (v(Good), v(Worn)) plane.  Many pairs of
         policies are incomparable, but one corner dominates every point.
Part 3 - brute force on random MDPs (|S| = 5, |A| = 3, gamma = 0.9): enumerate all
         3^5 = 243 deterministic policies, check that one of them dominates all the
         others in every state, measure how many pairs are incomparable, and check
         that random stochastic and random history-dependent policies never beat it.

Usage:  python code/ch01_the_rl_problem/policy_ordering.py [--quick]
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np

from mdp import (FiniteMDP, deterministic_policy, evaluate_policy_exact, greedy_actions,
                 q_from_v, uniform_policy, value_iteration)

G, W = 0, 1
RUN, FIX = 0, 1


def maintenance_mdp() -> FiniteMDP:
    """Section 6.3.  Good machine: running earns 2 and wears it out w.p. 0.25.
    Worn machine: running earns +2 or -1 with equal probability (defects);
    fixing costs 1 and restores it.  Fixing a good machine earns nothing."""
    outcomes = [[None, None], [None, None]]
    outcomes[G][RUN] = [(0.75, G, 2.0), (0.25, W, 2.0)]
    outcomes[G][FIX] = [(1.0, G, 0.0)]
    outcomes[W][RUN] = [(0.5, W, 2.0), (0.5, W, -1.0)]
    outcomes[W][FIX] = [(1.0, G, -1.0)]
    return FiniteMDP(2, 2, outcomes, ["Good", "Worn"], ["run", "fix"])


def batched_values(P, R, policies, gamma):
    """Exact v_pi for a batch of deterministic policies (rows of ``policies``)."""
    n = P.shape[0]
    idx = np.arange(n)[None, :]
    P_b = P[idx, policies]                       # (batch, n, n)
    r_b = R[idx, policies]                       # (batch, n)
    return np.linalg.solve(np.eye(n)[None] - gamma * P_b, r_b[..., None])[..., 0]


def stochastic_values(P, R, pis, gamma):
    """Exact v_pi for a batch of stochastic policies pis[b, s, a]."""
    n = P.shape[0]
    P_b = np.einsum("bsa,sat->bst", pis, P)
    r_b = np.einsum("bsa,sa->bs", pis, R)
    return np.linalg.solve(np.eye(n)[None] - gamma * P_b, r_b[..., None])[..., 0]


def history_dependent_values(P, R, pis, gamma):
    """Exact values of policies pi(a | s_prev, s) that remember the previous state.

    Such a policy is not Markov in s, but it is Markov in the augmented state
    (x, s) with x in {start} U S, so its value can be computed exactly on that
    larger chain.  pis[b, x, s, a] with x = n meaning "no previous state yet".
    Returns the value of starting in each s at time 0, i.e. at (start, s).
    """
    batch, n_x, n, k = pis.shape
    m = n_x * n                                  # augmented states (x, s) -> x * n + s
    P_aug = np.zeros((batch, m, m))
    r_aug = np.einsum("bxsa,sa->bxs", pis, R).reshape(batch, m)
    for x in range(n_x):
        for s in range(n):
            # next augmented state is (s, s') with probability sum_a pi(a|x,s) p(s'|s,a)
            row = np.einsum("ba,at->bt", pis[:, x, s, :], P[s])
            P_aug[:, x * n + s, s * n:(s + 1) * n] = row
    V = np.linalg.solve(np.eye(m)[None] - gamma * P_aug, r_aug[..., None])[..., 0]
    return V.reshape(batch, n_x, n)[:, n_x - 1, :]   # the "start" slice


def best_memory_policy_values(P, R, gamma, tol=1e-12):
    """Optimal value over ALL policies that may use the previous state: value iteration
    on the augmented MDP with states (x, s), x in {start} U S.  Returns values at (start, s)."""
    n, k, _ = P.shape
    V = np.zeros((n + 1, n))                      # V[x, s]
    while True:
        # Q[x, s, a] = r(s,a) + gamma * sum_s' p(s'|s,a) V[s, s']  (next augmented state is (s, s'))
        cont = np.einsum("sat,st->sa", P, V[:n])   # depends on s only, not on x
        Q = R[None] + gamma * cont[None]
        V_new = np.broadcast_to(Q.max(axis=2), V.shape).copy()
        if np.max(np.abs(V_new - V)) < tol:
            return V_new[n]
        V = V_new


def random_mdp(rng, n=5, k=3):
    P = rng.dirichlet(np.full(n, 0.5), size=(n, k))  # p(.|s,a), fairly sparse
    R = rng.normal(size=(n, k))                       # r(s,a)
    outcomes = [[[(P[s, a, t], t, 0.0) for t in range(n)] for a in range(k)] for s in range(n)]
    for s in range(n):  # put the expected reward on the outcomes so r(s,a) = R[s,a]
        for a in range(k):
            outcomes[s][a] = [(p, t, R[s, a]) for p, t, _ in outcomes[s][a]]
    return FiniteMDP(n, k, outcomes), P, R


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smoke test: fewer MDPs, no figures")
    args = parser.parse_args()
    t0 = time.perf_counter()
    seed = 0
    n_mdps = 10 if args.quick else 200
    n_stoch, n_hist = (200, 50) if args.quick else (2000, 300)
    rng = np.random.default_rng(seed)
    np.set_printoptions(precision=4, suppress=True)
    print(f"policy_ordering.py | seed={seed} random MDPs={n_mdps} (|S|=5,|A|=3,gamma=0.9) "
          f"stochastic/history policies per MDP={n_stoch}/{n_hist} quick={args.quick}")

    # ------------------------------------------------------------- Part 1
    mdp = maintenance_mdp()
    gamma = 0.8
    print(f"\n[Part 1] machine-maintenance MDP, gamma = {gamma}")
    print("  four-argument dynamics p(s', r | s, a):")
    for s in range(2):
        for a in range(2):
            outs = ", ".join(f"p(s'={mdp.state_names[t]}, r={r:+g}) = {p:g}" for p, t, r in mdp.outcomes[s][a])
            print(f"    s={mdp.state_names[s]:<4} a={mdp.action_names[a]:<3}: {outs}")
    print("  r(s,a)    =", mdp.expected_reward().tolist())
    print("  p(G|s,a)  =", mdp.transition_tensor()[:, :, G].tolist())
    names = {}
    for acts in itertools.product(range(2), repeat=2):
        label = "".join("rf"[a] for a in acts)  # e.g. "rf" = run when Good, fix when Worn
        names[label] = evaluate_policy_exact(mdp, deterministic_policy(acts, 2), gamma)
    for label, v in names.items():
        print(f"  pi_{label}: v(Good) = {v[G] + 0.0:.4f}, v(Worn) = {v[W] + 0.0:.4f}")
    v_rand = evaluate_policy_exact(mdp, uniform_policy(2, 2), gamma)
    print(f"  equiprobable random policy: v = {v_rand}")
    labels = list(names)
    for i, j in itertools.combinations(range(4), 2):
        vi, vj = names[labels[i]], names[labels[j]]
        rel = ">=" if np.all(vi >= vj) else "<=" if np.all(vi <= vj) else "incomparable"
        print(f"    pi_{labels[i]} {rel} pi_{labels[j]}")
    v_star, _ = value_iteration(mdp, gamma)
    q_star = q_from_v(mdp, v_star, gamma)
    print(f"  v_* = {v_star},  q_* = {q_star.tolist()}")
    print(f"  greedy actions: Good -> {[mdp.action_names[a] for a in greedy_actions(q_star)[G]]}, "
          f"Worn -> {[mdp.action_names[a] for a in greedy_actions(q_star)[W]]}")

    # where does fixing stop paying?  (Exercise 6: gamma = 4/7)
    # For each gamma, evaluate all four deterministic policies exactly and pick the one
    # that dominates (Section 12 guarantees it exists); record whether it fixes when Worn.
    P2, R2 = mdp.transition_tensor(), mdp.expected_reward()
    four = np.array(list(itertools.product(range(2), repeat=2)))
    gs = np.linspace(0.01, 0.99, 9801)
    fix_opt = []
    for g in gs:
        V4 = batched_values(P2, R2, four, g)
        best = np.flatnonzero(np.all(V4 >= V4.max(axis=0) - 1e-9, axis=1))
        assert best.size > 0, g        # Theorem 1.1: a dominating policy exists; never test an empty set
        fix_opt.append(bool(np.all(four[best, W] == FIX)))
    fix_opt = np.array(fix_opt)
    g_switch = gs[np.argmax(fix_opt)]
    print(f"  smallest gamma on a 0.0001 grid where fixing the worn machine is optimal: {g_switch:.4f} "
          f"(analytic 4/7 = {4 / 7:.4f}); at gamma=0.5 the optimal policy is "
          f"{'run/fix' if fix_opt[np.argmin(np.abs(gs - 0.5))] else 'run/run'}")

    # ------------------------------------------------------------- Part 2 (value polytope)
    grid = np.linspace(0, 1, 41 if args.quick else 121)
    al, be = np.meshgrid(grid, grid)                   # al = pi(run|Good), be = pi(run|Worn)
    pis = np.zeros((al.size, 2, 2))
    pis[:, G, RUN], pis[:, G, FIX] = al.ravel(), 1 - al.ravel()
    pis[:, W, RUN], pis[:, W, FIX] = be.ravel(), 1 - be.ravel()
    polytopes = {}
    for g in (0.5, 0.8):
        V = stochastic_values(P2, R2, pis, g)
        vs, _ = value_iteration(mdp, g)
        dominated = np.all(V <= vs + 1e-9, axis=1).mean()
        polytopes[g] = (V, vs)
        print(f"\n[Part 2] value polytope, gamma={g}: {len(V)} stochastic policies; "
              f"fraction with v_pi <= v_* in both states = {dominated:.3f}; v_* = {vs}")

    # ------------------------------------------------------------- Part 3 (random MDPs)
    gamma3, n, k = 0.9, 5, 3
    all_pols = np.array(list(itertools.product(range(k), repeat=n)))   # 243 x 5
    stats = dict(dominating_exists=0, n_optimal=[], incomparable=[], vi_match=0,
                 stoch_beats=0, hist_beats=0, worst_stoch=-np.inf, worst_hist=-np.inf, mem_gap=0.0)
    iu = np.triu_indices(len(all_pols), 1)
    for _ in range(n_mdps):
        mdp_r, P, R = random_mdp(rng, n, k)
        V = batched_values(P, R, all_pols, gamma3)                     # 243 x 5
        upper = V.max(axis=0)                                          # pointwise best over all policies
        dom = np.all(V >= upper - 1e-9, axis=1)                        # policies achieving it everywhere
        stats["dominating_exists"] += int(dom.any())
        stats["n_optimal"].append(int(dom.sum()))
        ge = np.all(V[:, None, :] >= V[None, :, :] - 1e-12, axis=2)
        incomp = ~(ge | ge.T)
        stats["incomparable"].append(incomp[iu].mean())
        v_vi, _ = value_iteration(mdp_r, gamma3, tol=1e-12)
        stats["vi_match"] += int(np.allclose(v_vi, upper, atol=1e-8))
        pis = rng.dirichlet(np.full(k, 0.3), size=(n_stoch, n))       # random stochastic policies ...
        pi_opt = deterministic_policy(all_pols[np.argmax(dom)], k)
        eps = rng.uniform(0, 0.2, size=(n_stoch, 1, 1))                # ... half of them close to optimal
        pis[: n_stoch // 2] = ((1 - eps) * pi_opt[None] + eps * pis)[: n_stoch // 2]
        Vs = stochastic_values(P, R, pis, gamma3)
        gap = (Vs - upper).max()
        stats["worst_stoch"] = max(stats["worst_stoch"], gap)
        stats["stoch_beats"] += int(gap > 1e-9)
        hpis = rng.dirichlet(np.full(k, 0.3), size=(n_hist, n + 1, n))  # pi(a | s_prev, s)
        Vh = history_dependent_values(P, R, hpis, gamma3)
        gap_h = (Vh - upper).max()
        stats["worst_hist"] = max(stats["worst_hist"], gap_h)
        stats["hist_beats"] += int(gap_h > 1e-9)
        v_mem = best_memory_policy_values(P, R, gamma3)
        stats["mem_gap"] = max(stats["mem_gap"], np.abs(v_mem - upper).max())
    print(f"\n[Part 3] {n_mdps} random MDPs, all {len(all_pols)} deterministic policies enumerated:")
    print(f"  a deterministic policy attaining max_pi v_pi(s) in EVERY state exists in "
          f"{stats['dominating_exists']}/{n_mdps} MDPs (and equals value iteration's v_* in {stats['vi_match']})")
    counts = {int(c): int(m) for c, m in zip(*np.unique(stats["n_optimal"], return_counts=True))}
    print(f"  number of such optimal deterministic policies -> number of MDPs: {counts}")
    inc = np.array(stats["incomparable"])
    print(f"  fraction of policy pairs that are incomparable: mean {inc.mean():.3f}, "
          f"range [{inc.min():.3f}, {inc.max():.3f}]")
    print(f"  random stochastic policies beating v_* somewhere: {stats['stoch_beats']} MDPs "
          f"(largest max_s v_pi(s) - v_*(s) = {stats['worst_stoch']:.2e})")
    print(f"  random history-dependent policies pi(a|s_prev,s) beating v_*: {stats['hist_beats']} MDPs "
          f"(largest gap = {stats['worst_hist']:.2e})")
    print(f"  BEST policy that may use the previous state (value iteration on (s_prev, s)): "
          f"max |v - v_*| = {stats['mem_gap']:.1e} -> memory does not help in an MDP")

    if not args.quick:
        import matplotlib.pyplot as plt
        from plotting import BLUE, INK, INK2, ORANGE, setup_style
        setup_style()
        figdir = Path(__file__).parent / "figures"
        figdir.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
        for ax, g in zip(axes, (0.5, 0.8)):
            V, vs = polytopes[g]
            ax.scatter(V[:, G], V[:, W], s=3, color=BLUE, alpha=0.25, lw=0,
                       label=r"$v_\pi$, stochastic $\pi$ (grid over $\pi$(run|$\cdot$))")
            xmin, ymin = V.min(axis=0)
            ax.fill_between([xmin - 1, vs[G]], ymin - 1, vs[W], color=ORANGE, alpha=0.07, lw=0,
                            label=r"region $\{v \leq v_\ast\}$")
            for acts in itertools.product(range(2), repeat=2):
                lab = "".join("rf"[a] for a in acts)
                v = evaluate_policy_exact(mdp, deterministic_policy(acts, 2), g)
                ax.plot(*v, "o", color=INK, ms=6)
                ax.annotate(f"$\\pi_{{{lab}}}$", v, textcoords="offset points", xytext=(6, -12),
                            fontsize=10, color=INK)
            ax.plot(*vs, "*", color=ORANGE, ms=15, label=r"$v_\ast$ (dominates every point)")
            ax.set_xlim(xmin - 0.4, vs[G] + 0.9)
            ax.set_ylim(ymin - 0.4, vs[W] + 0.9)
            ax.set_xlabel("v(Good)")
            ax.set_ylabel("v(Worn)", color=INK2)
            ax.set_title(f"Value polytope of the maintenance MDP, $\\gamma$ = {g}")
            ax.legend(loc="upper left", fontsize=8, markerscale=1.0, frameon=True, facecolor="white",
                      edgecolor="none", framealpha=0.9)
            ax.get_legend().legend_handles[0].set_sizes([25])
            ax.get_legend().legend_handles[0].set_alpha(0.8)
        fig.tight_layout()
        fig.savefig(figdir / "value_polytope.png")
        plt.close(fig)
        print(f"\nsaved figure to {figdir / 'value_polytope.png'}")

    print(f"\nSUMMARY: maintenance MDP v_* = {v_star} (pi_rf optimal at gamma=0.8, switch at gamma=4/7); "
          f"dominating deterministic policy found in {stats['dominating_exists']}/{n_mdps} random MDPs; "
          f"no stochastic or history-dependent policy beat it")
    print(f"done in {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
