"""Policy iteration and value iteration on FrozenLake-v1 (4x4 and 8x8, slippery), model read from env.unwrapped.P.

* Policy iteration (Algorithm 3.2) from the equiprobable random policy: value of the start state and
  number of changed actions per iteration.
* Value iteration (Algorithm 3.3) from V_0 = 0: true error ||V_k - v_*||, the a-priori bound
  gamma^k ||V_0 - v_*|| (eq. 3.9), the a-posteriori bound gamma*Delta_k/(1-gamma) (eq. 3.10), the true
  loss of the greedy policy and its bound 2*gamma*Delta_k/(1-gamma) (eq. 3.21).
* Why VI beats the gamma^k bound here (Section 6.5): most state-action pairs cannot terminate in one
  step and an improper policy exists, so T* is no better than a gamma-contraction; but the OPTIMAL
  policy terminates, and 0 <= v* - V_k <= (gamma P_pi*)^k (v* - V_0), so the error shrinks at the rate
  gamma * rho(P_pi*) (spectral radius) instead of gamma.
* Q-value iteration (Section 2.6) reproduces VI's v*: max_a Q = v*.
* Checks the DP answer in the real Gymnasium environment: Monte Carlo success rate of the optimal
  policy, with Gymnasium's default TimeLimit and without it, against exact success probabilities,
  and compares with the policy that is optimal for gamma = 1 (maximal probability of ever reaching G).

Run:  python code/ch03_dynamic_programming/frozenlake_pi_vi.py [--quick]
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

from dp import (TabularMDP, bellman_optimality, evaluate_exact, evaluate_finite_horizon,  # noqa: E402
                greedy, greedy_actions_all, policy_iteration, policy_model, q_from_v, q_value_iteration,
                uniform_policy, value_iteration)
from finite_horizon import stationary_optimal  # noqa: E402
from mdps import FL_ARROWS, frozenlake_mdp  # noqa: E402

SEED = 0
GAMMA = 0.99


def run_pi(mdp):
    t0 = time.perf_counter()
    actions, v_star, hist = policy_iteration(mdp, uniform_policy(mdp))
    elapsed = time.perf_counter() - t0
    print(f"  Policy iteration from the uniform random policy: {len(hist)} evaluations "
          f"({len(hist) - 1} improvements), {elapsed * 1e3:.1f} ms")
    for i, h in enumerate(hist):
        err = np.max(np.abs(h["V"] - v_star))
        print(f"    iter {i}: v(start) = {h['V'][0]:.6f}   ||v_pi - v*|| = {err:.2e}   "
              f"actions changed by the following improvement: {h['n_changed'] if i + 1 < len(hist) else 0}")
    return actions, v_star, hist


def run_vi(mdp, v_star, theta, P_star):
    """Synchronous VI with full diagnostics at every sweep (diagnostics are not part of the algorithm).

    Also checks the policy-specific bound of Section 6.5: for V_0 = 0 <= v_* (rewards >= 0),
    0 <= v_* - V_k <= (gamma P_pi*)^k (v_* - V_0), where pi* is an optimal policy.
    """
    g = mdp.gamma
    V = np.zeros(mdp.S)
    e0 = np.max(np.abs(V - v_star))
    B = v_star - V                                      # (gamma P_pi*)^k (v* - V_0), k = 0
    bound_ok = True
    rows = []
    t_alg = 0.0
    first_opt = None
    for k in range(1, 100_000):
        t = time.perf_counter()
        V_new = bellman_optimality(mdp, V)
        delta = np.max(np.abs(V_new - V))
        t_alg += time.perf_counter() - t
        pi = greedy(mdp, V_new)
        loss = np.max(v_star - evaluate_exact(mdp, pi))
        if first_opt is None and loss < 1e-9:
            first_opt = k
        B = g * (P_star @ B)
        gap = v_star - V_new
        bound_ok &= bool(gap.min() >= -1e-12 and np.all(gap <= B + 1e-12))
        rows.append(dict(k=k, err=np.max(np.abs(V_new - v_star)), prior=g ** k * e0,
                         post=g * delta / (1 - g), delta=delta, loss=loss, loss_bound=2 * g * delta / (1 - g),
                         v0=V_new[0]))
        V = V_new
        if delta < theta:
            break
    return V, rows, first_opt, t_alg, bound_ok


def spectral_radius(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def exact_mean_length(mdp_undisc: TabularMDP, actions):
    """Expected episode length (no time limit) of a proper policy: ((I - P_pi)^{-1} 1)(start)."""
    _, P_pi = policy_model(mdp_undisc, actions)
    return np.linalg.solve(np.eye(mdp_undisc.S) - P_pi, np.ones(mdp_undisc.S))[0]


def exact_success(mdp_undisc: TabularMDP, actions, H=None):
    """P(reach goal) under a stationary policy: gamma = 1, reward 1 at the goal.  H = None -> no time limit."""
    if H is None:
        return evaluate_exact(mdp_undisc, actions)[0]
    return evaluate_finite_horizon(mdp_undisc, lambda t: actions, H)[0]


def simulate(map_name, actions, n_episodes, max_steps=None):
    """Run the policy in the real Gymnasium environment; returns success rate, its std. error, truncation rate."""
    kwargs = {} if max_steps is None else {"max_episode_steps": max_steps}
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=True, **kwargs)
    obs, _ = env.reset(seed=SEED)             # seed once; later resets continue the RNG stream
    wins, truncs, lengths = 0, 0, []
    for ep in range(n_episodes):
        if ep > 0:
            obs, _ = env.reset()
        for t in range(1, 10 ** 7):
            obs, r, terminated, truncated, _ = env.step(int(actions[obs]))
            if terminated or truncated:
                wins += r > 0
                truncs += truncated and not terminated
                lengths.append(t)
                break
    env.close()
    p = wins / n_episodes
    return p, np.sqrt(p * (1 - p) / n_episodes), truncs / n_episodes, np.mean(lengths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n_eps = 300 if args.quick else 5000
    eps_opt = 1e-6
    theta = eps_opt * (1 - GAMMA) / (2 * GAMMA)        # eq. (3.21): Delta < theta => greedy is eps-optimal
    print(f"frozenlake_pi_vi.py  seed={SEED}  quick={args.quick}  gamma={GAMMA}  slippery=True  "
          f"VI stops at Delta < theta = eps(1-gamma)/(2 gamma) = {theta:.3e} (eps = {eps_opt})  "
          f"MC episodes = {n_eps}")
    t0 = time.perf_counter()
    results = {}
    for map_name in ("4x4", "8x8"):
        mdp, desc = frozenlake_mdp(map_name, GAMMA)
        print(f"\n=== FrozenLake {map_name}: |S| = {mdp.S}, |A| = {mdp.A} ===")
        actions, v_star, pi_hist = run_pi(mdp)

        # --- termination structure (Sections 2.4 and 6.5) ---
        nt = ~mdp.terminal
        cont = mdp.P.sum(axis=2)                       # continuation probability of each (s, a)
        never = (cont >= 1 - 1e-12) & nt[:, None]
        all_cont = never.all(axis=1) & nt
        _, P_up = policy_model(mdp, np.full(mdp.S, 3))  # "always up"
        _, P_star = policy_model(mdp, actions)
        rho = spectral_radius(P_star)
        print(f"  Termination structure: {int(never.sum())} of {int(nt.sum()) * mdp.A} non-terminal (s, a) pairs "
              f"({100 * never.sum() / (nt.sum() * mdp.A):.0f}%) cannot terminate in one step; in {int(all_cont.sum())} "
              f"states NO action can, so the one-step max-norm modulus of T* is exactly gamma = {GAMMA}.")
        print(f"    'always up' is improper: rho(P_up) = {spectral_radius(P_up):.4f}.  The optimal policy terminates: "
              f"rho(P_pi*) = {rho:.4f}, gamma * rho = {GAMMA * rho:.5f}")

        V, rows, first_opt, t_alg, bound_ok = run_vi(mdp, v_star, theta, P_star)
        last = rows[-1]
        print(f"  Value iteration from V_0 = 0: stopped after {last['k']} sweeps ({t_alg * 1e3:.1f} ms of "
              f"backups); ||V - v*|| = {last['err']:.2e} (a-posteriori bound {last['post']:.2e})")
        print(f"    greedy policy first optimal after {first_opt} sweeps, when ||V_k - v*|| = "
              f"{rows[first_opt - 1]['err']:.3f};  the bound 2 gamma Delta/(1-gamma) first certifies "
              f"0.01-optimality at sweep {next(r['k'] for r in rows if r['loss_bound'] < 0.01)}")
        for k in (1, 10, 50, 100, 200, 500, 1000):
            if k <= len(rows):
                r = rows[k - 1]
                print(f"    sweep {k:5d}: ||V_k - v*|| = {r['err']:.2e}  gamma^k bound = {r['prior']:.2e}  "
                      f"a-posteriori = {r['post']:.2e}  greedy loss = {r['loss']:.2e}  "
                      f"loss bound = {r['loss_bound']:.2e}")
        assert all(r["err"] <= r["prior"] + 1e-12 and r["err"] <= r["post"] + 1e-12 for r in rows)
        assert all(r["loss"] <= r["loss_bound"] + 1e-12 for r in rows)
        k1, k2 = 250, len(rows)                         # well inside the asymptotic regime
        rate = (rows[k2 - 1]["err"] / rows[k1 - 1]["err"]) ** (1 / (k2 - k1))
        print(f"    measured error ratio per sweep between sweeps {k1} and {k2}: {rate:.5f}  "
              f"(gamma * rho(P_pi*) = {GAMMA * rho:.5f}, gamma = {GAMMA});  "
              f"bound 0 <= v* - V_k <= (gamma P_pi*)^k v* held at every sweep: {bound_ok}")
        vi_pol = greedy(mdp, V)
        print(f"    VI's greedy policy has the same value as PI's: "
              f"{np.allclose(evaluate_exact(mdp, vi_pol), v_star, atol=1e-9)}")
        Q, q_sweeps = q_value_iteration(mdp, theta=theta)
        print(f"  Q-value iteration Q <- T*Q (eq. 3.13b) from Q_0 = 0: {q_sweeps} sweeps; "
              f"max|max_a Q - v*| = {np.max(np.abs(Q.max(axis=1) - v_star)):.1e}, "
              f"max|Q - q_v*| = {np.max(np.abs(Q - q_from_v(mdp, v_star))):.1e}")

        # --- check against the simulator ---
        und, _ = frozenlake_mdp(map_name, 1.0)
        # FrozenLake-v1 keeps its 100-step TimeLimit with either map (FrozenLake8x8-v1 would use 200)
        H = gym.make("FrozenLake-v1", map_name=map_name).spec.max_episode_steps
        p_inf = exact_success(und, actions)
        p_H = exact_success(und, actions, H)
        p_sim, se, trunc, mlen = simulate(map_name, actions, n_eps)
        p_sim_inf, se_inf, _, mlen_inf = simulate(map_name, actions, n_eps, max_steps=100_000)
        print(f"  Optimal (gamma={GAMMA}) policy in the simulator: v*(start) = {v_star[0]:.4f} (discounted)")
        print(f"    no time limit:         exact P(success) = {p_inf:.4f}   simulated = {p_sim_inf:.4f} +- {se_inf:.4f}"
              f"   mean length {mlen_inf:.1f}")
        print(f"    default TimeLimit({H}): exact P(success) = {p_H:.4f}   simulated = {p_sim:.4f} +- {se:.4f}"
              f"   truncated {100 * trunc:.1f}% of episodes")
        a1 = stationary_optimal(map_name, 1.0)       # optimal for gamma = 1: max P(ever reaching G)
        v1_star = value_iteration(und, theta=1e-13)[0][0]
        print(f"  gamma is part of the problem: max P(ever reaching G) = v*_(gamma=1)(start) = {v1_star:.4f}; "
              f"the gamma=1-optimal policy reaches G w.p. {exact_success(und, a1):.4f} with exact mean episode "
              f"length {exact_mean_length(und, a1):.1f}, vs {p_inf:.4f} and {exact_mean_length(und, actions):.1f} "
              f"steps for the gamma={GAMMA}-optimal policy")
        results[map_name] = dict(mdp=mdp, desc=desc, v_star=v_star, rows=rows, pi_hist=pi_hist,
                                 first_opt=first_opt, rho=rho)
    if not args.quick:
        make_figures(results)
    print(f"\nDone in {time.perf_counter() - t0:.1f} s")


def make_figures(results):
    from plotting import C, grid_values, plt, save

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw=dict(width_ratios=[1, 1.6]))
    for ax, name in zip(axes, ("4x4", "8x8")):
        R = results[name]
        mdp, v = R["mdp"], R["v_star"]
        acts = greedy_actions_all(mdp, v, tol=1e-9)
        arrows = [None if mdp.terminal[s] else acts[s] for s in range(mdp.S)]
        grid_values(ax, v, mdp.shape, desc=R["desc"], arrows=arrows, arrow_chars=FL_ARROWS,
                    fmt="{:.2f}", fontsize=8 if name == "4x4" else 6.5, cmap="viridis")
        ax.set_title(f"FrozenLake {name}: $v_\\ast$ and optimal actions ($\\gamma={GAMMA}$)")
    save(fig, "frozenlake_values_policy.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    R = results["8x8"]
    rows = R["rows"]
    k = np.array([r["k"] for r in rows])
    ax = axes[0]
    ax.semilogy(k, [r["prior"] for r in rows], "--", color=C[6], lw=1, label=r"a-priori bound $\gamma^k\|V_0-v_\ast\|$")
    ax.semilogy(k, [r["post"] for r in rows], color=C[0], lw=1.2,
                label=r"a-posteriori bound $\frac{\gamma}{1-\gamma}\|V_k-V_{k-1}\|$")
    ax.semilogy(k, [r["err"] for r in rows], color=C[1], lw=2, label=r"true error $\|V_k-v_\ast\|$")
    ax.semilogy(k, [r["loss_bound"] for r in rows], ":", color=C[2], lw=1.5,
                label=r"loss bound $\frac{2\gamma}{1-\gamma}\|V_k-V_{k-1}\|$")
    loss = np.array([max(r["loss"], 1e-16) for r in rows])
    ax.semilogy(k, loss, color=C[3], lw=2, label=r"true loss $\|v_\ast-v_{\pi_k}\|$ of greedy $\pi_k$")
    kk = np.arange(100, len(rows) + 1)
    gr = GAMMA * R["rho"]
    ax.semilogy(kk, rows[199]["err"] * gr ** (kk - 200) * 100, "-.", color="0.35", lw=1,
                label=f"reference slope $(\\gamma\\rho(P_{{\\pi_\\ast}}))^k$, $\\gamma\\rho={gr:.3f}$ (shifted up)")
    ax.axvline(R["first_opt"], color="0.5", lw=0.8)
    ax.text(R["first_opt"] * 1.05, 1e-12, f"greedy policy optimal\nfrom sweep {R['first_opt']}", fontsize=8)
    ax.set_ylim(1e-14, 1e3)
    ax.set_xlabel("sweep k"); ax.set_ylabel("max-norm error")
    ax.set_title(f"Value iteration on FrozenLake 8x8 ($\\gamma={GAMMA}$)")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    ax = axes[1]
    for i, name in enumerate(("4x4", "8x8")):
        R = results[name]
        vs = R["v_star"]
        e_pi = [max(np.max(np.abs(h["V"] - vs)), 1e-16) for h in R["pi_hist"]]
        ax.semilogy(range(len(e_pi)), e_pi, "o-", color=C[i], label=f"policy iteration, {name}")
        e_vi = [r["err"] for r in R["rows"]]
        ax.semilogy(range(1, len(e_vi) + 1), e_vi, "-", color=C[i], alpha=0.5, lw=1,
                    label=f"value iteration, {name}")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("iteration (PI: evaluation+improvement; VI: one sweep)")
    ax.set_ylabel(r"$\|V - v_\ast\|_\infty$")
    ax.set_title("PI needs a handful of (expensive) iterations\n(exact zeros plotted at $10^{-16}$)", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, "frozenlake_convergence.png")


if __name__ == "__main__":
    main()
