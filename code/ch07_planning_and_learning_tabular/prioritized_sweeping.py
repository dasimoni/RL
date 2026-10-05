"""Prioritized sweeping vs Dyna-Q: updates until an optimal policy is found.

Chapter 07, Section 5 (in the spirit of S&B Example 8.4, which reports Peng &
Williams' experiment; the refinement scheme here is our own). The Dyna maze is
refined to resolutions 1x, 2x, ... (every cell becomes a factor x factor block)
and two planners are compared:

  * Dyna-Q: after each real step, n planning updates on (state, action) pairs
    drawn uniformly (state, then action) from previously observed ones;
  * Prioritized sweeping (S&B box, deterministic environment): a max-priority
    queue keyed by |TD error|; after each real step, up to n pairs are popped
    and updated, and every predecessor of an updated state is (re)inserted with
    its own |TD error| if that exceeds theta.

Both use the same epsilon-greedy behaviour policy, step size and n. After every
real step we follow the greedy policy from the start state; a run "solves" the
maze at the first step where that greedy path reaches the goal in the optimal
(shortest-path) number of moves. We report, up to that point:
  * the number of Q-value updates (real + simulated);
  * PS's total work = updates + priority computations (each priority is a
    TD-error evaluation, about as costly as an update; Dyna-Q has none);
  * the number of real environment steps;
and diagnose every PS run that never became optimal.

Run:  python code/ch07_planning_and_learning_tabular/prioritized_sweeping.py [--quick] [--alpha A] [--no-figures]
"""

from __future__ import annotations

import argparse
from collections import deque
import heapq
import os
import random
import time

import numpy as np

import maze_env
from maze_env import N_ACTIONS

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")


def eps_greedy(q, eps, rng):
    if rng.random() < eps:
        return rng.randrange(N_ACTIONS)
    m = max(q)
    best = [a for a in range(N_ACTIONS) if q[a] == m]
    return best[0] if len(best) == 1 else rng.choice(best)


def greedy_path_is_optimal(maze, Q, optimal_len):
    """Follow argmax Q from the start; True iff we hit the goal in exactly
    optimal_len moves. Stops early on a non-positive-valued state or a too-long
    path. Also returns the set of states it examined: the answer can only
    change after an update to one of THOSE states, so callers re-check only
    then (an exact speed-up, not an approximation)."""
    s = maze.start
    examined = {s}
    for _ in range(optimal_len):
        q = Q[s]
        m = max(q)
        if m <= 0.0:
            return False, examined
        s, _, term = maze.table[s][q.index(m)]
        if term:
            return True, examined
        examined.add(s)
    return False, examined


def dyna_q_until_optimal(maze, optimal_len, n, alpha, gamma, eps, rng, max_updates, max_steps, q0=0.0):
    """Tabular Dyna-Q (Section 2.1 box). Returns a dict with the number of
    Q-value updates and real steps until the greedy path was optimal (updates
    = None if never), and how many distinct (s, a) pairs had been tried."""
    nS = maze.n_states
    table = maze.table
    Q = [[q0] * N_ACTIONS for _ in range(nS)]
    model = [None] * (nS * N_ACTIONS)
    seen_states, seen_actions = [], [[] for _ in range(nS)]
    updates = steps = tried = 0
    examined, dirty = set(), True       # see greedy_path_is_optimal
    s = maze.start
    while updates < max_updates and steps < max_steps:
        a = eps_greedy(Q[s], eps, rng)
        s2, r, term = table[s][a]
        Q[s][a] += alpha * ((r if term else r + gamma * max(Q[s2])) - Q[s][a])
        dirty = dirty or s in examined
        updates += 1
        key = s * N_ACTIONS + a
        if model[key] is None:
            tried += 1
            if not seen_actions[s]:
                seen_states.append(s)
            seen_actions[s].append(a)
        model[key] = (r, s2, term)
        for _ in range(n):
            ps = seen_states[rng.randrange(len(seen_states))]
            acts = seen_actions[ps]
            pa = acts[rng.randrange(len(acts))]
            pr, ps2, pterm = model[ps * N_ACTIONS + pa]
            Q[ps][pa] += alpha * ((pr if pterm else pr + gamma * max(Q[ps2])) - Q[ps][pa])
            if ps in examined:
                dirty = True
        updates += n
        steps += 1
        if dirty:
            ok, examined = greedy_path_is_optimal(maze, Q, optimal_len)
            dirty = False
            if ok:
                return dict(updates=updates, steps=steps, tried=tried)
        s = maze.start if term else s2
    return dict(updates=None, steps=steps, tried=tried)


def prioritized_sweeping_until_optimal(maze, optimal_len, n, alpha, gamma, eps, theta, rng,
                                       max_updates, max_steps, q0=0.0, probe_step=None):
    """S&B's 'Prioritized sweeping for a deterministic environment' box.
    The PQueue is a binary heap with lazy deletion: `in_queue[key]` holds the
    priority currently valid for key; stale heap entries are skipped.

    Besides Q-value updates we count PRIORITY COMPUTATIONS (steps (e) and the
    predecessor loop of (g)): each is a TD-error evaluation, roughly as costly
    as an update, so "total work" = updates + priority computations is the
    fair measure of computation (Section 5.1). If probe_step is given, we also
    record how many distinct (s, a) pairs had been tried after that many real
    steps (used to compare exploration with Dyna-Q at equal real experience)."""
    nS = maze.n_states
    table = maze.table
    Q = [[q0] * N_ACTIONS for _ in range(nS)]
    model = [None] * (nS * N_ACTIONS)
    predecessors = [set() for _ in range(nS)]    # predecessors[s'] = {(s, a) keys}
    heap, in_queue, counter = [], {}, 0
    updates = steps = priority_evals = idle_steps = tried = 0
    tried_at_probe = None
    examined, dirty = set(), True

    def insert(key, p):
        nonlocal counter
        if in_queue.get(key, -1.0) >= p:      # already queued with >= priority
            return
        in_queue[key] = p
        counter += 1
        heapq.heappush(heap, (-p, counter, key))

    s = maze.start
    while updates < max_updates and steps < max_steps:
        a = eps_greedy(Q[s], eps, rng)
        s2, r, term = table[s][a]
        key = s * N_ACTIONS + a
        tried += model[key] is None
        model[key] = (r, s2, term)
        predecessors[s2].add(key)
        # (e)-(f): priority of the real transition; NOTE no direct Q update here
        p = abs((r if term else r + gamma * max(Q[s2])) - Q[s][a])
        priority_evals += 1
        if p > theta:
            insert(key, p)
        # (g) planning: up to n highest-priority updates
        done_updates = 0
        while done_updates < n and heap:
            negp, _, k = heapq.heappop(heap)
            if in_queue.get(k) != -negp:
                continue                       # stale entry
            del in_queue[k]
            ps, pa = divmod(k, N_ACTIONS)
            pr, ps2, pterm = model[k]
            Q[ps][pa] += alpha * ((pr if pterm else pr + gamma * max(Q[ps2])) - Q[ps][pa])
            done_updates += 1
            if ps in examined:
                dirty = True
            v_ps = max(Q[ps])
            # Every (bs, ba) predicted to lead to ps. ps is never the goal (we only
            # update non-terminal states), so these transitions are non-terminal
            # and their target is br + gamma * max_a Q(ps, a).
            for bk in predecessors[ps]:
                bs, ba = divmod(bk, N_ACTIONS)
                br = model[bk][0]
                pb = abs(br + gamma * v_ps - Q[bs][ba])
                priority_evals += 1
                if pb > theta:
                    insert(bk, pb)
        updates += done_updates
        idle_steps += (done_updates == 0)
        steps += 1
        if steps == probe_step:
            tried_at_probe = tried
        if dirty:
            ok, examined = greedy_path_is_optimal(maze, Q, optimal_len)
            dirty = False
            if ok:
                return dict(updates=updates, all_updates=updates, steps=steps, evals=priority_evals,
                            diag=None, tried_at_probe=tried_at_probe)
        s = maze.start if term else s2
    return dict(updates=None, all_updates=updates, steps=steps, evals=priority_evals,
                tried_at_probe=tried_at_probe,
                diag=stall_diagnostics(maze, Q, model, idle_steps / steps, optimal_len))


def stall_diagnostics(maze, Q, model, idle_frac, optimal_len):
    """Why did a run fail to become optimal? Compare the greedy path with the
    shortest path that exists INSIDE THE LEARNED MODEL (BFS over known
    transitions) and count never-tried (s, a) pairs. Two kinds of stall:
      * 'exploration': the greedy path IS the shortest path in the (incomplete)
        model, so planning is done and only new real experience can help;
      * 'residual': the model already contains an optimal path but the greedy
        path is longer, i.e. un-requeued residual errors (alpha < 1, Sec. 5)."""
    s, greedy_len = maze.start, None
    for t in range(1, 4 * maze.n_states):
        q = Q[s]
        s, _, term = maze.table[s][q.index(max(q))]
        if term:
            greedy_len = t
            break
    dist, frontier = {maze.start: 0}, deque([maze.start])
    while frontier:
        x = frontier.popleft()
        for a in range(N_ACTIONS):
            e = model[x * N_ACTIONS + a]
            if e is not None and e[1] not in dist:
                dist[e[1]] = dist[x] + 1
                frontier.append(e[1])
    untried = sum(model[x * N_ACTIONS + a] is None for x in maze.free_states() for a in range(N_ACTIONS))
    model_shortest = dist.get(maze.goal)
    if greedy_len is not None and greedy_len == model_shortest:
        kind = "exploration"
    elif model_shortest == optimal_len:
        kind = "residual"
    else:
        kind = "other"
    return dict(greedy_len=greedy_len, model_shortest=model_shortest, untried=untried,
                n_pairs=N_ACTIONS * len(maze.free_states()), idle_frac=idle_frac, kind=kind)


def _inf_if_none(values, flags):
    """Per-run measurements as floats, +inf where the run never became optimal."""
    return np.array([np.inf if f is None else v for v, f in zip(values, flags)], float)


def _ratio(a, b):
    return a / b if np.isfinite(a) and np.isfinite(b) else float("nan")


def run_condition(base, factors, n_runs, q0, hp, seed0, max_updates, max_steps):
    """All maze sizes for one initial value q0. Returns a list of per-size dicts."""
    rows = []
    for f in factors:
        maze = base.scaled(f)
        L = maze.shortest_path_length()
        n_states = len(maze.free_states()) + 1   # free cells incl. the goal
        dy, ps = [], []
        t_dyna = t_ps = 0.0
        for run in range(n_runs):
            seed = seed0 * 100_000 + 1000 * f + run
            t0 = time.time()
            dy.append(dyna_q_until_optimal(maze, L, hp["n"], hp["alpha"], hp["gamma"], hp["eps"],
                                           random.Random(seed), max_updates, max_steps, q0=q0))
            t_dyna += time.time() - t0
            t0 = time.time()
            # probe: how many pairs had PS tried after as many real steps as
            # Dyna-Q needed (only meaningful when Dyna-Q became optimal)
            probe = dy[-1]["steps"] if dy[-1]["updates"] is not None else None
            ps.append(prioritized_sweeping_until_optimal(maze, L, hp["n"], hp["alpha"], hp["gamma"],
                                                         hp["eps"], hp["theta"], random.Random(seed),
                                                         max_updates, max_steps, q0=q0, probe_step=probe))
            t_ps += time.time() - t0
        # A capped run (updates None) never found the optimal path: count it as
        # +inf so it pushes the median up honestly instead of disappearing.
        dflag, pflag = [r["updates"] for r in dy], [r["updates"] for r in ps]
        fails = {"dyna": sum(u is None for u in dflag), "ps": sum(u is None for u in pflag)}
        d = _inf_if_none(dflag, dflag)
        p = _inf_if_none(pflag, pflag)
        # Dyna-Q has no priority queue: its total work is its updates. PS's total
        # work adds one priority computation per real step and per predecessor visited.
        p_work = _inf_if_none([r["all_updates"] + r["evals"] for r in ps], pflag)
        ds = _inf_if_none([r["steps"] for r in dy], dflag)
        pst = _inf_if_none([r["steps"] for r in ps], pflag)
        upd_per_step = np.array([r["all_updates"] / r["steps"] for r in ps])
        evals_per_upd = np.array([r["evals"] / max(r["all_updates"], 1) for r in ps])
        cov = [(a["tried"], b["tried_at_probe"]) for a, b in zip(dy, ps)
               if a["updates"] is not None and b["tried_at_probe"] is not None]
        row = dict(n_states=n_states, L=L, dyna=d, ps=p, ps_work=p_work, dyna_steps=ds, ps_steps=pst,
                   ps_fails=fails["ps"], dyna_fails=fails["dyna"], upd_per_step=upd_per_step,
                   evals_per_upd=evals_per_upd, cov=cov,
                   diags=[r["diag"] for r in ps if r["diag"] is not None])
        rows.append(row)
        md, mp, mw = np.median(d), np.median(p), np.median(p_work)
        print(f"  factor {f}: {maze.n_rows}x{maze.n_cols} grid, {n_states:4d} states, shortest path {L:2d}"
              f" | runs not optimal within {max_steps:.0e} real steps: Dyna-Q {fails['dyna']}, PS {fails['ps']}\n"
              f"     value updates to optimal (median):      Dyna-Q {md:9.0f} | PS {mp:9.0f} "
              f"| ratio {_ratio(md, mp):5.1f} | PS fewer in {int((p < d).sum())}/{n_runs} runs\n"
              f"     total work = updates + priority computations (median): Dyna-Q {md:9.0f} | PS {mw:9.0f} "
              f"| ratio {_ratio(md, mw):5.2f} | PS less in {int((p_work < d).sum())}/{n_runs} runs\n"
              f"     real steps to optimal (median):         Dyna-Q {np.median(ds):9.0f} | PS {np.median(pst):9.0f} "
              f"| ratio {_ratio(np.median(ds), np.median(pst)):5.2f} | PS fewer in {int((pst < ds).sum())}/{n_runs} runs\n"
              f"     PS updates per real step: median {np.median(upd_per_step):.2f} "
              f"(range {upd_per_step.min():.2f}-{upd_per_step.max():.2f}, budget n={hp['n']}); "
              f"priority computations per update: median {np.median(evals_per_upd):.1f} "
              f"(range {evals_per_upd.min():.1f}-{evals_per_upd.max():.1f})")
        if cov:
            c = np.array(cov)
            print(f"     (s,a) pairs tried when Dyna-Q became optimal, vs PS after the same number of real steps: "
                  f"median Dyna-Q {np.median(c[:, 0]):.0f} | PS {np.median(c[:, 1]):.0f}"
                  f" | Dyna-Q tried more in {int((c[:, 0] > c[:, 1]).sum())}/{len(c)} runs"
                  f" (runs where PS was still running)")
        print(f"     wall-clock: Dyna-Q {t_dyna:.1f}s, PS {t_ps:.1f}s")
        for dg in row["diags"]:
            print(f"     stalled PS run [{dg['kind']}]: greedy path {dg['greedy_len']} moves; shortest path inside "
                  f"learned model {dg['model_shortest']} (true optimum {L}); untried pairs "
                  f"{dg['untried']}/{dg['n_pairs']}; queue empty on {100 * dg['idle_frac']:.2f}% of steps")
    return rows


def print_summary(results, n_runs):
    """Compact table of the medians, one block per initialisation."""
    print("\nSUMMARY (medians over %d runs; inf = median run never optimal)" % n_runs)
    for q0, rows in results.items():
        print(f"Q0 = {q0:g}")
        print(f"  {'states':>6} {'L':>3} | {'Dyna upd':>9} {'PS upd':>8} {'ratio':>6} | {'PS work':>9} "
              f"{'ratio':>6} | {'Dyna steps':>10} {'PS steps':>9} | {'PS fail':>7} | {'PS upd/step':>11}")
        for r in rows:
            md, mp, mw = np.median(r["dyna"]), np.median(r["ps"]), np.median(r["ps_work"])
            print(f"  {r['n_states']:>6} {r['L']:>3} | {md:>9.0f} {mp:>8.0f} {_ratio(md, mp):>6.1f} | {mw:>9.0f} "
                  f"{_ratio(md, mw):>6.2f} | {np.median(r['dyna_steps']):>10.0f} {np.median(r['ps_steps']):>9.0f} | "
                  f"{r['ps_fails']:>4}/{n_runs:<2} | {np.median(r['upd_per_step']):>11.2f}")
        kinds = [dg["kind"] for r in rows for dg in r["diags"]]
        if kinds:
            print("  stalled PS runs by kind: " + ", ".join(f"{k} {kinds.count(k)}" for k in sorted(set(kinds))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="step size for both methods (default 1.0: in a deterministic world a "
                             "sample update with alpha=1 is an exact expected update; with alpha<1 "
                             "the textbook PS box can stall, see the chapter's pitfalls section; "
                             "the full run with --alpha 0.5 takes about 4 minutes)")
    parser.add_argument("--no-figures", action="store_true",
                        help="full-size run without writing figures (e.g. for --alpha 0.5)")
    args = parser.parse_args()

    factors = [1, 2] if args.quick else [1, 2, 3, 4, 5, 6]
    n_runs = 3 if args.quick else 10
    hp = dict(n=5, alpha=args.alpha, gamma=0.95, eps=0.1, theta=1e-4)
    max_updates, max_steps = 20_000_000, 1_000_000
    print(f"Prioritized sweeping vs Dyna-Q | seed={args.seed} runs={n_runs} resolution factors={factors} "
          f"{hp} | cap: {max_steps:.0e} real steps")

    base = maze_env.dyna_maze()
    results = {}
    for q0, label in [(0.0, "Q0 = 0 (as in the S&B boxes)"), (1.0, "Q0 = 1 (optimistic: no return exceeds 1)")]:
        print(f"--- initial action values {label}")
        results[q0] = run_condition(base, factors, n_runs, q0, hp, args.seed, max_updates, max_steps)
    print_summary(results, n_runs)

    if args.quick or args.no_figures:
        print("no figures written")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    panels = [(axes[0], "", "computation until greedy path is optimal"),
              (axes[1], "_steps", "real environment steps until greedy path is optimal")]
    for ax, suffix, ylabel in panels:
        for q0, ls, marker in [(0.0, "-", "o"), (1.0, "--", "s")]:
            rows = results[q0]
            sizes = np.array([r["n_states"] for r in rows])
            series = [("dyna", "Dyna-Q", "tab:blue"), ("ps", "PS", "tab:red")]
            if suffix == "":
                # PS's priority computations cost about as much as updates: show both.
                series.append(("ps_work", "PS updates + priority comps", "tab:orange"))
            for method, label, color in series:
                key = method + suffix if method != "ps_work" else method
                med = np.array([np.median(r[key]) for r in rows])
                if suffix == "":
                    label = label + (" updates" if method != "ps_work" else "")
                ax.plot(sizes, med, ls=ls, marker=marker, color=color, ms=4, label=f"{label}, Q0={q0:g}")
        ax.set_yscale("log")
        ax.set_xlabel("number of states in the maze (resolution 1x ... 6x)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower right")      # the lower right is empty; upper left hides curves
    axes[0].set_title(f"Value updates and total work: median of {n_runs} runs (n={hp['n']}, alpha={hp['alpha']:g})",
                      fontsize=10)
    axes[1].legend(fontsize=7, loc="upper left")
    axes[1].set_title("Real experience, same runs (missing point = median run never optimal)", fontsize=10)
    stalls = ", ".join(f"{r['ps_fails']}" for r in results[0.0])
    other = sum(r["dyna_fails"] for q0 in results for r in results[q0]) + \
        sum(r["ps_fails"] for r in results[1.0])
    sizes_txt = ", ".join(f"{r['n_states']}" for r in results[0.0])
    axes[1].text(0.98, 0.03, f"runs never optimal within {max_steps:.0e} real steps\n"
                 f"PS, Q0=0: {stalls} of {n_runs}\n(maze sizes {sizes_txt})\nall other method/init pairs: {other} in total",
                 transform=axes[1].transAxes, ha="right", va="bottom", fontsize=8,
                 bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "prioritized_sweeping.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
