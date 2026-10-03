"""Trajectory sampling: uniform vs on-policy distribution of updates.

Chapter 07, Section 7 (S&B Section 8.6, Figure 8.8).

Random episodic tasks: |S| states, 2 actions per state. Every (s, a) has b
successor states drawn uniformly at random from S (fixed when the task is
created); each transition goes to the terminal state with probability 0.1 and
otherwise to one of the b successors with equal probability. Each successor
transition (s, a, j) has its own reward r(s, a, j) ~ N(0, 1); termination
gives reward 0. There is no discounting; episodes start in state 0.

Both planners use the SAME expected update (Section 6)
    Q(s, a) <- 0.9 * (1/b) * sum_j [ r(s, a, j) + max_a' Q(s_j, a') ]
and differ only in WHICH (s, a) they update:
  * uniform:   cycle through all state-action pairs in a fixed order;
  * on-policy: simulate episodes from the start state with an eps-greedy
               policy (eps = 0.1) w.r.t. the current Q and update the pairs
               encountered (the simulation uses the model, so no real
               experience is consumed).
At ~40 evaluation points (dense early, see eval_schedule) we compute
v_pi(start) for the greedy policy pi w.r.t. Q by iterative policy evaluation
run to convergence (tolerance 1e-5).

All tasks are simulated in parallel as one batch of numpy arrays.

Run:  python code/ch07_planning_and_learning_tabular/trajectory_sampling.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
P_CONTINUE = 0.9   # 1 - termination probability
EPS = 0.1


def make_tasks(n_tasks, n_states, b, rng):
    nxt = rng.integers(0, n_states, size=(n_tasks, n_states, 2, b))
    rew = rng.standard_normal((n_tasks, n_states, 2, b))
    return nxt, rew


def expected_update(Q, nxt, rew, k_idx, s, a):
    """Expected updates of Q[k, s[k], a[k]] for every task k at once."""
    succ = nxt[k_idx, s, a]                                  # (K, b)
    best_next = Q[k_idx[:, None], succ].max(axis=2)          # (K, b)
    Q[k_idx, s, a] = P_CONTINUE * (rew[k_idx, s, a] + best_next).mean(axis=1)


def evaluate_greedy(Q, nxt, rew, v_init, tol=1e-5, max_iter=1000):
    """Value of the greedy policy in every task, by iterative policy
    evaluation  v(s) = 0.9 * mean_j [ r(s, pi(s), j) + v(s_j) ]  run until the
    largest change is below tol (values are O(1..10), so 1e-5 is ample).
    The operator is a 0.9-contraction, so this converges; we warm-start from
    the previous evaluation because the greedy policy changes slowly."""
    K, nS, _, b = nxt.shape
    k_idx = np.arange(K)[:, None]
    s_idx = np.arange(nS)[None, :]
    pi = Q.argmax(axis=2)                                    # (K, nS)
    # flat indices into v.ravel() make the gather a single fast np.take
    flat_succ = (nxt[k_idx, s_idx, pi] + (np.arange(K) * nS)[:, None, None]).reshape(K, -1)
    r_pi = rew[k_idx, s_idx, pi].mean(axis=2)                 # (K, nS)
    v = v_init.copy()
    for _ in range(max_iter):
        v_new = P_CONTINUE * (r_pi + np.take(v, flat_succ).reshape(K, nS, b).mean(axis=2))
        if np.max(np.abs(v_new - v)) < tol:
            return v_new
        v = v_new
    return v


def eval_schedule(n_updates):
    """Dense early (every 0.5% of the budget up to 10%), sparse later (every 5%)."""
    early = np.arange(0, n_updates // 10, n_updates // 200)
    late = np.arange(n_updates // 10, n_updates + 1, n_updates // 20)
    return np.unique(np.concatenate([early, late]))


def run(method, n_tasks, n_states, b, n_updates, seed):
    rng = np.random.default_rng(seed)
    nxt, rew = make_tasks(n_tasks, n_states, b, rng)        # same tasks for both methods
    sim_rng = np.random.default_rng(seed + 1)
    Q = np.zeros((n_tasks, n_states, 2))
    k_idx = np.arange(n_tasks)
    v = np.zeros((n_tasks, n_states))
    evals = set(eval_schedule(n_updates).tolist())
    values = [evaluate_greedy(Q, nxt, rew, v)[:, 0]]           # at 0 updates
    state = np.zeros(n_tasks, dtype=np.int64)                # on-policy: current simulated state
    for u in range(1, n_updates + 1):
        if method == "uniform":
            pair = (u - 1) % (2 * n_states)
            s = np.full(n_tasks, pair // 2)
            a = np.full(n_tasks, pair % 2)
            expected_update(Q, nxt, rew, k_idx, s, a)
        else:
            s = state
            greedy = Q[k_idx, s].argmax(axis=1)
            explore = sim_rng.random(n_tasks) < EPS
            a = np.where(explore, sim_rng.integers(0, 2, n_tasks), greedy)
            expected_update(Q, nxt, rew, k_idx, s, a)
            # simulate one step of the episode with the model
            j = sim_rng.integers(0, b, n_tasks)
            terminated = sim_rng.random(n_tasks) >= P_CONTINUE
            state = np.where(terminated, 0, nxt[k_idx, s, a, j])
        if u in evals:
            v = evaluate_greedy(Q, nxt, rew, v)
            values.append(v[:, 0])
    return np.array(values)                                  # (n_evals + 1, n_tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.quick:
        configs = [(1000, 1, 4000, 20), (1000, 10, 4000, 20)]
    else:
        # (n_states, b, n_updates, n_tasks)
        configs = [(1000, 1, 20_000, 60), (1000, 3, 20_000, 60),
                   (1000, 10, 20_000, 60), (10_000, 1, 200_000, 30)]
    print(f"Trajectory sampling (S&B Fig. 8.8) | seed={args.seed} eps={EPS} "
          f"termination prob={1 - P_CONTINUE:.1f} | configs (|S|, b, updates, tasks)={configs}")

    results = {}
    for n_states, b, n_updates, n_tasks in configs:
        sched = eval_schedule(n_updates)
        for method in ("on-policy", "uniform"):
            t0 = time.time()
            vals = run(method, n_tasks, n_states, b, n_updates,
                       seed=args.seed * 1000 + n_states + b)
            results[(n_states, b, method)] = vals
            mean = vals.mean(axis=1)
            se = vals.std(axis=1, ddof=1) / np.sqrt(n_tasks)
            marks = [n_updates // 20, n_updates // 10, n_updates // 4, n_updates // 2, n_updates]
            idx = [int(np.searchsorted(sched, m)) for m in marks]
            summary = " ".join(f"{sched[i]:,}:{mean[i]:5.2f}±{se[i]:.2f}" for i in idx)
            print(f"|S|={n_states:>5} b={b:>2} {method:<9} v_pi(start) after updates -> {summary} "
                  f"({time.time() - t0:.1f}s)")
        on, un = results[(n_states, b, "on-policy")].mean(1), results[(n_states, b, "uniform")].mean(1)
        # Both methods plan on the SAME random tasks, so compare them pairwise:
        # the SE of the per-task difference is smaller than the per-method SEs.
        diff = results[(n_states, b, "uniform")] - results[(n_states, b, "on-policy")]
        dse = diff.std(axis=1, ddof=1) / np.sqrt(n_tasks)
        print("      paired difference uniform - on-policy (mean ± SE over tasks): " +
              " ".join(f"{sched[i]:,}:{diff[i].mean():+.2f}±{dse[i]:.2f}" for i in idx))
        ahead = on > un
        # first evaluation from which uniform stays ahead at EVERY later evaluation
        stay = next((i for i in range(1, len(sched)) if not ahead[i:].any()), None)
        frac_on = ahead[1:].mean()
        print(f"      -> on-policy ahead at {100 * frac_on:.0f}% of evaluations; uniform ahead at every "
              f"evaluation from {'never' if stay is None else f'{sched[stay]:,} updates'} on")

    if args.quick:
        print("quick mode: no figures written")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    colors = {1: "tab:blue", 3: "tab:orange", 10: "tab:green"}
    for (n_states, b, n_updates, n_tasks) in configs:
        ax = axes[0] if n_states == 1000 else axes[1]
        x = eval_schedule(n_updates)
        for method, ls in (("on-policy", "-"), ("uniform", "--")):
            vals = results[(n_states, b, method)]
            ax.plot(x, vals.mean(axis=1), ls=ls, color=colors[b], label=f"b={b}, {method}")
        ax.set_title(f"{n_states:,} states (mean over {n_tasks} random tasks)", fontsize=10)
        ax.set_xlabel("computation time, in expected updates")
        ax.set_ylabel("value of start state under greedy policy")
        # Put the legend in the empty band between the b = 1 curves (values 6-8)
        # and the b = 3 curves (about 3), where it hides no curve.
        ax.legend(fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.6),
                  ncol=2 if n_states == 1000 else 1)
        ax.grid(alpha=0.3)
    fig.suptitle("Trajectory sampling: on-policy vs uniform distribution of expected updates", fontsize=11)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "trajectory_sampling.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
