"""Real-time dynamic programming (RTDP) vs value iteration on a racetrack.

Chapter 07, Section 8 (in the spirit of S&B Example 8.6; the track below is
our own design, the dynamics follow S&B Exercise 5.12).

Racetrack MDP
  * state  = (row, col, vy, vx): position on the grid and velocity; vy >= 0 is
    the speed UPWARD (row decreases), vx >= 0 the speed to the RIGHT;
    0 <= vy, vx <= 4 and the velocity may be (0, 0) only on the start line;
  * action = velocity increments (ay, ax) in {-1, 0, +1}^2 (9 actions); an
    action is available only if the new velocity is legal;
  * noise  = with probability 0.1 the increments are both 0 whatever the action;
  * move   = new position = old position + new velocity; we check the cells
    along the straight path: entering a finish cell 'F' ends the episode,
    leaving the track first is a crash: the car goes back to a uniformly random
    start cell with velocity (0, 0) and the episode continues;
  * reward = -1 per step, no discounting; v*(s) = -(expected steps to finish).

Planners (both use the same expected update, Section 8):
    V(s) <- max_a [ -1 + 0.9 * W(s, v+a) + 0.1 * W(s, v) ]
  where W(outcome) = 0 at the finish, mean_{start s0} V(s0) after a crash,
  V(s') otherwise.
  * DP: in-place (Gauss-Seidel) value iteration over all reachable states;
  * RTDP: episodes from a random start state; at each visited state do the
    expected update, act greedily, sample the next state from the model.
    Initial values 0 >= v*, so the conditions of Barto, Bradtke & Singh (1995)
    hold and RTDP converges to an optimal policy on the relevant states.

Run:  python code/ch07_planning_and_learning_tabular/rtdp_racetrack.py [--quick]
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from collections import deque

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")

TRACK = [
    "####...................F",
    "###....................F",
    "##.....................F",
    "#......................F",
    "#......................F",
    "#......................F",
    "#..........#############",
    "#..........#############",
    "#..........#############",
    "#.........##############",
    "#.........##############",
    "#.........##############",
    "#.........##############",
    "..........##############",
    "..........##############",
    "..........##############",
    "..........##############",
    "..........##############",
    "..........##############",
    "..........##############",
    "..........##############",
    "#.........##############",
    "#.........##############",
    "#.........##############",
    "#.........##############",
    "#.........##############",
    "#.........##############",
    "##........##############",
    "##........##############",
    "##........##############",
    "##........##############",
    "##SSSSSSSS##############",
]
MAX_SPEED = 4
P_NOISE = 0.1
ACTIONS = [(ay, ax) for ay in (-1, 0, 1) for ax in (-1, 0, 1)]
FINISH, CRASH = -1, -2          # special outcome codes


class Racetrack:
    def __init__(self, track):
        self.grid = [list(row) for row in track]
        self.H, self.W = len(track), len(track[0])
        self.start_cells = [(r, c) for r in range(self.H) for c in range(self.W) if self.grid[r][c] == "S"]
        self._enumerate_states()

    def cell_type(self, r, c):
        if not (0 <= r < self.H and 0 <= c < self.W):
            return "#"
        return self.grid[r][c]

    def move(self, r, c, vy, vx):
        """Outcome of moving from (r, c) with velocity (vy, vx): FINISH, CRASH or
        the new (row, col). Samples points densely along the segment."""
        dr, dc = -vy, vx
        n = 4 * max(abs(dr), abs(dc), 1)
        for i in range(1, n + 1):
            rr = int(round(r + dr * i / n))
            cc = int(round(c + dc * i / n))
            t = self.cell_type(rr, cc)
            if t == "F":
                return FINISH
            if t == "#":
                return CRASH
        return (r + dr, c + dc)

    def legal_velocity(self, vy, vx):
        return 0 <= vy <= MAX_SPEED and 0 <= vx <= MAX_SPEED and (vy, vx) != (0, 0)

    def _enumerate_states(self):
        """BFS over every state reachable from the start states under ANY policy
        (all actions, with and without noise, including crashes). Builds the
        outcome tables used by every planner."""
        starts = [(r, c, 0, 0) for (r, c) in self.start_cells]
        index = {s: i for i, s in enumerate(starts)}
        states = list(starts)
        frontier = deque(starts)
        outcome = {}   # (state, new_velocity) -> FINISH / CRASH / state tuple
        while frontier:
            s = frontier.popleft()
            r, c, vy, vx = s
            vels = {(vy, vx)} if (vy, vx) != (0, 0) else set()   # noise outcome
            # at a start state with v = 0 the noise outcome "stay put" is legal
            if (vy, vx) == (0, 0):
                vels.add((0, 0))
            for ay, ax in ACTIONS:
                if self.legal_velocity(vy + ay, vx + ax):
                    vels.add((vy + ay, vx + ax))
            for (nvy, nvx) in vels:
                if (nvy, nvx) == (0, 0):
                    res = (r, c, 0, 0)                 # start state, noise: no movement
                else:
                    pos = self.move(r, c, nvy, nvx)
                    res = pos if pos in (FINISH, CRASH) else (pos[0], pos[1], nvy, nvx)
                outcome[(s, (nvy, nvx))] = res
                if res not in (FINISH, CRASH) and res not in index:
                    index[res] = len(states)
                    states.append(res)
                    frontier.append(res)
        self.states, self.index = states, index
        self.n = len(states)
        self.start_idx = [index[s] for s in starts]
        self.start_set = set(self.start_idx)
        # Outcome indices into an "extended" value vector Vext = [V, 0 (finish), mean V(start) (crash)]
        self.TERM, self.CRASHI = self.n, self.n + 1

        def code(res):
            if res == FINISH:
                return self.TERM
            if res == CRASH:
                return self.CRASHI
            return index[res]

        self.noise_out = np.empty(self.n, dtype=np.int64)
        self.act_out = np.full((self.n, len(ACTIONS)), -1, dtype=np.int64)  # -1 = illegal action
        for i, s in enumerate(states):
            r, c, vy, vx = s
            self.noise_out[i] = code(outcome[(s, (vy, vx))])
            for a, (ay, ax) in enumerate(ACTIONS):
                if self.legal_velocity(vy + ay, vx + ax):
                    self.act_out[i, a] = code(outcome[(s, (vy + ay, vx + ax))])
        self.legal = self.act_out >= 0
        # plain-Python copies for the fast scalar loops
        self.noise_list = self.noise_out.tolist()
        self.act_list = [[(a, o) for a, o in enumerate(row) if o >= 0] for row in self.act_out.tolist()]


# ------------------------------------------------------------------ planners
def backup(track, V, i, crash_value):
    """Expected update at state i; returns (new value, greedy action)."""
    def W(o):
        if o == track.TERM:
            return 0.0
        if o == track.CRASHI:
            return crash_value
        return V[o]
    w_noise = W(track.noise_list[i])
    best, best_a = -1e18, None
    for a, o in track.act_list[i]:
        q = -1.0 + (1 - P_NOISE) * W(o) + P_NOISE * w_noise
        if q > best:
            best, best_a = q, a
    return best, best_a


def greedy_policy(track, V):
    Vext = np.concatenate([V, [0.0, np.mean(V[track.start_idx])]])
    q = -1.0 + (1 - P_NOISE) * Vext[np.where(track.legal, track.act_out, 0)] + \
        P_NOISE * Vext[track.noise_out][:, None]
    q[~track.legal] = -np.inf
    return q.argmax(axis=1)


def evaluate_policy(track, pi):
    """Exact value of a deterministic policy, averaged over the start states
    (= minus the expected episode length). Returns -inf if pi is improper.

    1. Collect the states reachable from the start line under pi.
    2. pi is proper iff every one of them can reach the finish (a graph
       question: backward search from the finish over pi's transitions).
    3. If proper, solve the linear Bellman equation (I - P_pi) v = -1 on those
       states exactly (sparse LU), with a crash spreading its probability
       uniformly over the start states."""
    starts = track.start_idx
    o_main = track.act_out[np.arange(track.n), pi]

    def successors(i):
        out = []
        for o, p in ((int(o_main[i]), 1 - P_NOISE), (int(track.noise_out[i]), P_NOISE)):
            if o == track.CRASHI:
                out.extend((j, p / len(starts)) for j in starts)
            elif o != track.TERM:
                out.append((o, p))
        return out

    reach, frontier = set(starts), deque(starts)
    while frontier:
        i = frontier.popleft()
        for j, _ in successors(i):
            if j not in reach:
                reach.add(j)
                frontier.append(j)
    order = sorted(reach)
    pos = {i: k for k, i in enumerate(order)}
    # backward reachability of the finish
    preds = {i: [] for i in order}
    can_finish = deque()
    ok = set()
    for i in order:
        if int(o_main[i]) == track.TERM or int(track.noise_out[i]) == track.TERM:
            ok.add(i)
            can_finish.append(i)
        for j, _ in successors(i):
            preds[j].append(i)
    while can_finish:
        j = can_finish.popleft()
        for i in preds[j]:
            if i not in ok:
                ok.add(i)
                can_finish.append(i)
    if len(ok) < len(order):
        return -np.inf
    rows, cols, vals = [], [], []
    for i in order:
        for j, p in successors(i):
            rows.append(pos[i])
            cols.append(pos[j])
            vals.append(p)
    n = len(order)
    P = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    v = spsolve((sp.identity(n, format="csr") - P).tocsc(), -np.ones(n))
    return float(np.mean([v[pos[i]] for i in starts]))


def heuristic_values(track):
    """An ADMISSIBLE (optimistic) initial value function: each step moves the
    car at most MAX_SPEED cells along each axis, and the finish line is the
    right-most column of rows 0-5, so at least
        ceil(max(cols to the right edge, rows above row 5) / MAX_SPEED)
    steps remain. Hence H(s) = -that >= v*(s) (Section 9, heuristic search)."""
    return np.array([-math.ceil(max(track.W - 1 - c, r - 5, 0) / MAX_SPEED)
                     for (r, c, vy, vx) in track.states], dtype=float)


GAPS = [("10%", 0.10), ("1%", 0.01), ("0.1%", 0.001)]   # relative suboptimality thresholds


def first_hits(log, J_star):
    """From a log of (updates, J(greedy)) return, for each threshold, the first
    number of updates at which the greedy policy was that close to optimal."""
    out = {}
    for name, frac in GAPS + [("exact", None)]:
        tol = 1e-3 if frac is None else frac * abs(J_star)
        out[name] = next((u for u, J in log if J > J_star - tol), None)
    return out


def value_iteration(track, tol, order=None, V0=None, J_star=None, log=None):
    """In-place (Gauss-Seidel) value iteration: sweeps over ALL reachable
    states in the given order until the largest change in a sweep is < tol.
    If J_star is given, the greedy policy is evaluated exactly after each sweep."""
    order = list(range(track.n)) if order is None else order
    V = [0.0] * track.n if V0 is None else list(V0)
    sweeps = updates = 0
    while True:
        delta = 0.0
        crash_value = sum(V[j] for j in track.start_idx) / len(track.start_idx)
        for i in order:
            v_new, _ = backup(track, V, i, crash_value)
            delta = max(delta, abs(v_new - V[i]))
            V[i] = v_new
            if i in track.start_set:
                crash_value = sum(V[j] for j in track.start_idx) / len(track.start_idx)
        sweeps += 1
        updates += track.n
        if J_star is not None and log is not None:
            log.append((updates, evaluate_policy(track, greedy_policy(track, np.array(V)))))
        if delta < tol:
            return np.array(V), sweeps, updates


def rtdp(track, rng, V0, n_episodes, eval_every, J_star, snapshot_gap=0.01):
    """Trajectory-based RTDP (S&B Section 8.7): episodes from random start
    states; expected update at every visited state; greedy action; next state
    sampled from the model. Returns the evaluation log and the per-state update
    counts at the moment the greedy policy first came within `snapshot_gap`
    (relative) of optimal, plus the final counts and values."""
    V = list(V0)
    counts = [0] * track.n
    starts = track.start_idx
    updates = 0
    log, snapshot = [], None
    for ep in range(1, n_episodes + 1):
        i = starts[rng.randrange(len(starts))]
        while True:
            crash_value = sum(V[j] for j in starts) / len(starts)
            V[i], a = backup(track, V, i, crash_value)   # expected update at the current state
            counts[i] += 1
            updates += 1
            # sample the next state from the model: noise, then crash/finish
            o = track.noise_list[i] if rng.random() < P_NOISE else track.act_out[i, a]
            if o == track.TERM:
                break
            if o == track.CRASHI:
                o = starts[rng.randrange(len(starts))]
            i = int(o)
        if ep % eval_every == 0:
            J = evaluate_policy(track, greedy_policy(track, np.array(V)))
            log.append((updates, J))
            if snapshot is None and J > J_star - snapshot_gap * abs(J_star):
                snapshot = (ep, updates, np.array(counts))
    return log, snapshot, np.array(counts), np.array(V)


def fmt(u):
    return "   never" if u is None else f"{u:>8,}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    n_runs = 1 if args.quick else 5
    n_episodes = 1500 if args.quick else 10_000
    eval_every = 100
    track = Racetrack(TRACK)
    n_cells = sum(ch != "#" for row in TRACK for ch in row)
    print(f"RTDP vs value iteration on a racetrack | seed={args.seed} RTDP runs={n_runs} "
          f"episodes/run={n_episodes} noise={P_NOISE} eval every {eval_every} episodes")
    print("\n".join(TRACK))
    print(f"track cells: {n_cells}; states reachable from the start line under any policy: {track.n}")

    t0 = time.time()
    V_star, _, _ = value_iteration(track, tol=1e-10)
    J_star = V_star[track.start_idx].mean()
    pi_star = greedy_policy(track, V_star)
    H = heuristic_values(track)
    assert np.all(H >= V_star - 1e-9), "heuristic must be admissible"
    print(f"optimal expected steps from the start line: {-J_star:.4f} "
          f"(check by evaluating the greedy policy exactly: {-evaluate_policy(track, pi_star):.4f}); "
          f"admissible heuristic at the start line: {-H[track.start_idx].mean():.1f} steps "
          f"[{time.time() - t0:.1f}s]")

    # States visited with positive probability under ONE optimal policy, pi_star
    # (greedy w.r.t. V*, ties broken by the lowest action index). The relevant
    # set of Section 8 (union over ALL optimal policies) is at least this large.
    relevant, frontier = set(track.start_idx), deque(track.start_idx)
    while frontier:
        i = frontier.popleft()
        for o in (int(track.act_out[i, pi_star[i]]), int(track.noise_out[i])):
            nxt = track.start_idx if o == track.CRASHI else ([] if o == track.TERM else [o])
            for j in nxt:
                if j not in relevant:
                    relevant.add(j)
                    frontier.append(j)
    print(f"states visited with positive probability by one optimal policy (greedy w.r.t. V*, ties by index): "
          f"{len(relevant)} ({100 * len(relevant) / track.n:.1f}% of reachable states)")
    rel_idx = np.array(sorted(relevant))

    header = f"{'method':<34}" + "".join(f"{'within ' + n:>14}" for n, _ in GAPS) + f"{'exact':>10}"
    print(f"\nupdates until the greedy policy's expected time is within x% of optimal ({-J_star:.3f} steps)")
    print(header)
    curves = {}
    # ---- DP: two sweep orders. "forward" = breadth-first order from the start line
    # (the order the states were discovered); "backward" = the reverse.
    for name, order in [("DP sweeps, forward order", list(range(track.n))),
                        ("DP sweeps, backward order", list(range(track.n))[::-1])]:
        t0 = time.time()
        log = []
        _, sweeps, upd = value_iteration(track, 1e-4, order=order, J_star=J_star, log=log)
        curves[name] = [log]
        h = first_hits(log, J_star)
        print(f"{name:<34}" + "".join(f"{fmt(h[n]):>14}" for n, _ in GAPS) + f"{fmt(h['exact']):>10}"
              f"   | stop rule max change<1e-4: {sweeps} sweeps = {upd:,} updates [{time.time() - t0:.1f}s]")

    # ---- RTDP from V0 = 0 and from the admissible heuristic
    stats = {}
    for name, V0 in [("RTDP, V0 = 0", np.zeros(track.n)), ("RTDP, V0 = admissible heuristic", H)]:
        t0 = time.time()
        hits, snaps, logs, finals = [], [], [], []
        for run in range(n_runs):
            rng = random.Random(args.seed * 1000 + run)
            log, snap, counts, V = rtdp(track, rng, V0, n_episodes, eval_every, J_star)
            logs.append(log)
            hits.append(first_hits(log, J_star))
            snaps.append(snap)
            finals.append((counts, V))
        curves[name] = logs
        stats[name] = (snaps, counts, V)
        cols = []
        for n, _ in GAPS + [("exact", None)]:
            vals = [h[n] for h in hits]
            if any(v is None for v in vals):
                cols.append(f"{sum(v is not None for v in vals)}/{n_runs} runs")
            else:
                cols.append(f"{np.mean(vals):,.0f}")
        print(f"{name:<34}" + "".join(f"{c:>14}" for c in cols[:-1]) + f"{cols[-1]:>10}"
              f"   | mean over {n_runs} runs [{time.time() - t0:.1f}s]")
        good = [s for s in snaps if s is not None]
        if good:
            c = np.array([s[2] for s in good])
            print(f"{'':<34}at first reaching 1%: {np.mean([s[0] for s in good]):,.0f} episodes; states updated "
                  f"0 times {100 * np.mean(c == 0):.1f}%, <=10 times {100 * np.mean(c <= 10):.1f}%, "
                  f"<=100 times {100 * np.mean(c <= 100):.1f}% (of all {track.n} reachable states)")
        # Where is the remaining error after all episodes? Look at the states
        # pi_star visits (a subset of the relevant states).
        cr = np.array([c[rel_idx] for c, _ in finals])
        err = np.array([np.abs(v[rel_idx] - V_star[rel_idx]) for _, v in finals])
        print(f"{'':<34}after {n_episodes:,} episodes, on the {len(rel_idx)} states pi* visits: "
              f"median updates per state {np.median(cr):.0f}; updated <=10 times {100 * np.mean(cr <= 10):.1f}%, "
              f"never {100 * np.mean(cr == 0):.1f}%; |V - V*| > 0.1 on {100 * np.mean(err > 0.1):.1f}% "
              f"(mean over runs); total updates per run {np.mean([c.sum() for c, _ in finals]):,.0f}")

    if args.quick:
        print("quick mode: no figures written")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    styles = {"DP sweeps, forward order": ("tab:blue", "o-", 1.0),
              "DP sweeps, backward order": ("tab:cyan", "s-", 1.0),
              "RTDP, V0 = 0": ("tab:red", "-", 0.6),
              "RTDP, V0 = admissible heuristic": ("tab:orange", "-", 0.6)}
    floor = 1e-3
    for name, logs in curves.items():
        color, style, alpha = styles[name]
        for k, log in enumerate(logs):
            x = [u for u, _ in log]
            gap = [max(J_star - J, floor) if np.isfinite(J) else np.nan for _, J in log]
            ax.plot(x, gap, style, color=color, alpha=alpha, ms=3, lw=1.2,
                    label=name + (f" ({len(logs)} runs)" if len(logs) > 1 else "") if k == 0 else None)
    for n, frac in GAPS:
        ax.axhline(frac * abs(J_star), color="0.6", ls=":", lw=0.8)
        ax.text(2.1e3, frac * abs(J_star) * 1.1, f"{n} of optimal", fontsize=7, color="0.4")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2e3, None)
    ax.set_ylim(floor * 0.8, 50)
    ax.set_xlabel("number of expected updates (log scale)")
    ax.set_ylabel("suboptimality of greedy policy (expected extra steps)")
    ax.set_title(f"Greedy-policy quality vs computation (optimal: {-J_star:.2f} steps; "
                 f"gaps < {floor:g} drawn at {floor:g})", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    # Heat map: RTDP (V0 = 0) updates per track cell (summed over velocities), last run
    ax = axes[1]
    _, last_counts, last_V = stats["RTDP, V0 = 0"]
    heat = np.full((track.H, track.W), np.nan)
    for i, (r, c, vy, vx) in enumerate(track.states):
        heat[r, c] = (0 if np.isnan(heat[r, c]) else heat[r, c]) + last_counts[i]
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("#d9e6f2")        # track cells no state ever occupies
    im = ax.imshow(np.log10(1 + heat), cmap=cmap)
    for r in range(track.H):
        for c in range(track.W):
            if TRACK[r][c] == "#":
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="0.35"))
            elif TRACK[r][c] in "SF":
                ax.text(c, r, TRACK[r][c], ha="center", va="center", fontsize=7, color="tab:blue")
    # one greedy trajectory of the final RTDP policy, without noise
    i = track.start_idx[0]
    pi = greedy_policy(track, last_V)
    path = [track.states[i][:2]]
    for _ in range(100):
        o = int(track.act_out[i, pi[i]])
        if o in (track.TERM, track.CRASHI):
            break
        i = o
        path.append(track.states[i][:2])
    pr, pc = zip(*path)
    ax.plot(pc, pr, "o-", color="tab:cyan", ms=3, lw=1.5, label="greedy path, no noise")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"RTDP (V0 = 0) updates per cell after {n_episodes:,} episodes\n"
                 f"(log10 of 1 + count, summed over velocities)\n"
                 f"pale blue = cells no state occupies:\n"
                 f"unreachable cells and the finish line", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "rtdp_racetrack.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
