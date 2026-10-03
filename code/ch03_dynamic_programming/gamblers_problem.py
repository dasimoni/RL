"""The gambler's problem (Sutton & Barto Example 4.3) solved by value iteration.

* In-place value iteration (Algorithm 3.3) for ph = 0.4; value estimates after sweeps 1, 2, 3, 32
  and at convergence; the full set of optimal stakes (there are many ties).
* ph = 0.25 and ph = 0.55 for comparison (S&B Exercise 4.9).
* A gamma = 1 pitfall: if a stake of 0 is allowed, the Bellman optimality equation has more than one
  solution, value iteration started from the wrong place converges to the wrong one, and the greedy
  policy w.r.t. v_* can be "never bet", which never terminates.

Run:  python code/ch03_dynamic_programming/gamblers_problem.py [--quick]
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from dp import bellman_optimality, evaluate_exact, greedy, q_from_v, value_iteration  # noqa: E402
from mdps import gamblers_problem  # noqa: E402

SEED = 0          # deterministic; printed for uniformity
THETA = 1e-13
SNAPSHOTS = [1, 2, 3, 32]


def solve(ph, inplace=True, record=False, allow_zero=False, V0=None):
    mdp = gamblers_problem(ph, allow_zero_stake=allow_zero)
    t0 = time.perf_counter()
    V, sweeps, deltas, hist = value_iteration(mdp, theta=THETA, inplace=inplace, record=record, V0=V0)
    return mdp, V, sweeps, hist, time.perf_counter() - t0


def polish(mdp, V):
    """One exact evaluation of the greedy policy (a policy-iteration step) to get v_* to ~1e-15."""
    pol = greedy(mdp, V)
    return evaluate_exact(mdp, pol)


def optimal_stakes(mdp, v, tol=1e-10):
    Q = q_from_v(mdp, v)
    best = Q.max(axis=1, keepdims=True)
    return [np.flatnonzero(Q[s] >= best[s] - tol) for s in range(mdp.S)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(f"gamblers_problem.py  seed={SEED}  quick={args.quick}  goal=100, gamma=1, theta={THETA}, "
          f"stakes 1..min(s, 100-s) unless stated")
    t_start = time.perf_counter()
    phs = [0.4] if args.quick else [0.25, 0.4, 0.55]
    sol = {}
    for ph in phs:
        mdp, V, sweeps, _, dt = solve(ph, inplace=True)
        # synchronous sweeps are recorded for the figure (S&B's Figure 4.3 shows sweeps 1, 2, 3, 32)
        _, _, sweeps_sync, hist, dt_sync = solve(ph, inplace=False, record=(ph == 0.4))
        v = polish(mdp, V)
        stakes = optimal_stakes(mdp, v)
        n_ties = sum(len(stakes[s]) > 1 for s in range(1, 100))
        print(f"\nph = {ph}: in-place VI {sweeps} sweeps ({dt * 1e3:.0f} ms), synchronous VI {sweeps_sync} sweeps "
              f"({dt_sync * 1e3:.0f} ms); max|V_VI - v_*| = {np.max(np.abs(V - v)):.1e}")
        print(f"  v_*(s) for s = 1, 25, 50, 75, 99: " + ", ".join(f"{v[s]:.6f}" for s in (1, 25, 50, 75, 99)))
        print(f"  states with more than one optimal stake: {n_ties} of 99")
        smallest = [int(stakes[s][0]) for s in range(1, 100)]
        print(f"  smallest optimal stake for s = 1..99:\n    {smallest}")
        sol[ph] = dict(mdp=mdp, v=v, hist=hist, stakes=stakes, V_vi=V)

    # Bold play: stake min(s, 100 - s).  Known to be optimal when ph <= 0.5 (Dubins & Savage).
    mdp = sol[0.4]["mdp"]
    bold = np.array([0] + [min(s, 100 - s) for s in range(1, 100)] + [0])
    v_bold = evaluate_exact(mdp, bold)
    print(f"\nBold play (stake everything you can) at ph = 0.4: max |v_bold - v_*| = "
          f"{np.max(np.abs(v_bold - sol[0.4]['v'])):.1e}")
    timid = np.array([0] + [1] * 99 + [0])
    if 0.55 in sol:
        # Independent checks (v_* above is polish(V_VI), i.e. the exact value of VI's greedy policy, which is
        # timid play here, so comparing v_timid with it would be circular):
        #  (1) VI itself, which knows nothing about timid play, against timid play's exact value;
        #  (2) timid play's value against the gambler's-ruin closed form (1 - (q/p)^s) / (1 - (q/p)^100).
        v_t = evaluate_exact(sol[0.55]["mdp"], timid)
        p_, q_ = 0.55, 0.45
        closed = (1 - (q_ / p_) ** np.arange(101)) / (1 - (q_ / p_) ** 100)
        print(f"Timid play (always stake 1) at ph = 0.55: max |V_VI - v_timid| = "
              f"{np.max(np.abs(sol[0.55]['V_vi'] - v_t)):.1e} (VI vs exact value of timid play);  "
              f"max |v_timid - closed form| = {np.max(np.abs(v_t[1:100] - closed[1:100])):.1e};  "
              f"VI's greedy policy is timid play in {sum(sol[0.55]['stakes'][x][0] == 1 for x in range(1, 100))}"
              f" of 99 states (ties: {sum(len(sol[0.55]['stakes'][x]) > 1 for x in range(1, 100))})")

    # ---- The zero-stake pitfall (gamma = 1, improper policies) ------------------------------
    print("\nZero stake allowed (S&B's action set {0, 1, ..., min(s, 100-s)}), ph = 0.4:")
    v_star = sol[0.4]["v"]
    for name, V0 in (("V_0 = 0", None), ("V_0 = 1 on non-terminal states", np.r_[0, np.ones(99), 0])):
        for allow in (False, True):
            m2, V, sweeps, _, _ = solve(0.4, inplace=False, allow_zero=allow, V0=V0)
            fp = np.max(np.abs(bellman_optimality(m2, V) - V))
            print(f"  {name:32s} stake 0 {'allowed' if allow else 'excluded'}: VI stops after {sweeps:4d} sweeps;"
                  f" ||T*V - V|| = {fp:.1e};  max|V - v_*| = {np.max(np.abs(V - v_star)):.3f}")
    m2 = gamblers_problem(0.4, allow_zero_stake=True)
    # q(s, 0) = v(s) for every v, so at v = v_* the zero stake is ALWAYS among the maximisers.
    st0 = optimal_stakes(m2, v_star, tol=1e-12)
    n0 = sum(st0[s][0] == 0 for s in range(1, 100))
    print(f"  stake 0 is an optimal (greedy) action w.r.t. v_* in {n0} of 99 states. The greedy policy that "
          f"breaks ties toward the smallest stake never bets, never terminates, and has value 0 everywhere "
          f"(v_*(50) = {v_star[50]:.3f})")

    if not args.quick:
        make_figures(sol)
    print(f"\nDone in {time.perf_counter() - t_start:.1f} s")


def make_figures(sol):
    from plotting import C, plt, save

    s = np.arange(1, 100)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    ax = axes[0, 0]
    hist = sol[0.4]["hist"]
    for i, k in enumerate([k for k in SNAPSHOTS if k < len(hist)]):
        ax.plot(s, hist[k][1:100], color=C[i], lw=1.2, label=f"sweep {k}")
    ax.plot(s, sol[0.4]["v"][1:100], color="k", lw=1.5, ls="--", label=f"final ({len(hist) - 1} sweeps)")
    ax.set_xlabel("capital s"); ax.set_ylabel("value estimate")
    ax.set_title(r"Value iteration sweeps (synchronous), $p_h=0.4$")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    stakes = sol[0.4]["stakes"]
    xs = [x for x in s for _ in stakes[x]]
    ys = [a for x in s for a in stakes[x]]
    ax.scatter(xs, ys, s=7, color="0.6", label="all optimal stakes (ties within 1e-10)")
    ax.step(s, [stakes[x][0] for x in s], where="mid", color=C[1], lw=1.4, label="smallest optimal stake")
    ax.set_xlabel("capital s"); ax.set_ylabel("stake")
    ax.set_title(r"Optimal policies, $p_h=0.4$ (not unique)")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1, 0]
    for i, ph in enumerate(sorted(sol)):
        ax.plot(s, sol[ph]["v"][1:100], color=C[i], lw=1.5, label=f"$p_h={ph}$")
    ax.set_xlabel("capital s"); ax.set_ylabel(r"$v_\ast(s)$ = P(reach 100)")
    ax.set_title("Optimal value for three coin biases")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    styles = {0.25: dict(lw=3.5, alpha=0.45, ls="-"), 0.4: dict(lw=1.2, ls="--"), 0.55: dict(lw=1.5, ls="-")}
    for i, ph in enumerate(sorted(sol)):
        st = sol[ph]["stakes"]
        ax.step(s, [st[x][0] for x in s], where="mid", color=C[i], label=f"$p_h={ph}$", **styles[ph])
    ax.set_xlabel("capital s"); ax.set_ylabel("smallest optimal stake")
    ax.set_title("Smallest optimal stake ($p_h=0.25$ and $0.4$ coincide)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "gambler.png")


if __name__ == "__main__":
    main()
