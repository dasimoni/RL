"""The Tiger POMDP: exact belief updates, exact alpha-vector value iteration, and the value of memory.

Chapter 15, Section 2 (and the frame-stacking discussion of Section 3.1).  Everything here is exact (linear algebra on small models) and is
double-checked by Monte Carlo simulation.

The problem (Cassandra, Kaelbling & Littman, 1994; Kaelbling, Littman & Cassandra, 1998).  A tiger is behind the left or the right
door (hidden state s in {TL, TR}).  Actions: LISTEN (reward -1, state unchanged, you hear the
tiger on the correct side with probability 0.85), OPEN-LEFT, OPEN-RIGHT (+10 if the tiger is
behind the other door, -100 if you open the tiger's door).  After a door is opened the tiger
is re-placed uniformly at random and the observation is pure noise (each growl 50/50).
Discount gamma = 0.95, infinite horizon (the usual 'tiger.95' benchmark setting).

What the script does
  1. Belief update, Eq. (15.3) of the chapter, checked two ways:
       (a) against brute-force Bayes: sum over every hidden-state path of the joint probability;
       (b) against Monte Carlo: empirical frequency of TL among simulated runs that produced
           the same action-observation history.
  2. Value iteration on the belief MDP with alpha vectors (Section 2.5, Algorithm 15.1b).  With two hidden states
     each alpha vector is a line over b = Pr(TL), so pruning is an exact upper-envelope
     computation.  We show how the EXACT finite-horizon sets grow, then solve to convergence
     with epsilon-pruning (drop vectors that raise the envelope by less than 1e-6 anywhere).
  3. Exact evaluation of finite-state controllers (action rule + memory update) by solving the
     linear system on (hidden state, controller node) pairs, each checked by Monte Carlo:
        - the best memoryless policy pi(a | last observation; a 'start' node before the first
          growl), found by exhaustive search over the 27 deterministic policies, a 0.1-step grid
          over stochastic ones (start action: listen), and a continuous local search (L-BFGS-B
          from random starts) over all stochastic ones, start action included;
        - "frame stacking": the policy sees only the last k (action, observation) pairs.  We
          evaluate (i) the naive rule "act greedily on the belief filtered from the window" and
          (ii) the best window policy found by coordinate ascent with exact evaluation;
        - the belief-optimal policy from step 2.

Run:  python code/ch15_beyond_mdps/tiger_pomdp.py [--quick]
Full mode writes figures/tiger_value_function.png, tiger_belief_check.png, tiger_policies.png.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one CPU thread: multithreaded BLAS on a
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # shared machine makes small solves ~1000x slower
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import itertools
import time

import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# ----------------------------------------------------------------------------------------------
# The model.  Index conventions: states 0=TL, 1=TR; actions 0=LISTEN, 1=OPEN-LEFT, 2=OPEN-RIGHT;
# observations 0=HEAR-LEFT (HL), 1=HEAR-RIGHT (HR).
# ----------------------------------------------------------------------------------------------
A_NAMES = ["listen", "open-left", "open-right"]
ACC = 0.85     # listening accuracy
GAMMA = 0.95

# P[a, s, s'] = p(s' | s, a)
P = np.zeros((3, 2, 2))
P[0] = np.eye(2)                    # listening does not move the tiger
P[1] = P[2] = 0.5                   # opening a door resets the problem
# Z[a, s', o] = O(o | s', a): the observation kernel depends on the action and the NEXT state
Z = np.zeros((3, 2, 2))
Z[0] = [[ACC, 1 - ACC], [1 - ACC, ACC]]
Z[1] = Z[2] = 0.5                   # after opening, the growl is uninformative
# R[s, a] = r(s, a)
R = np.array([[-1.0, -100.0, 10.0],     # tiger left: opening LEFT is fatal
              [-1.0, 10.0, -100.0]])    # tiger right: opening RIGHT is fatal
D0 = np.array([0.5, 0.5])


def belief_update(b, a, o):
    """Bayes filter, Eq. (15.3):  b'(s') = O(o|s',a) sum_s p(s'|s,a) b(s) / Pr(o | b, a).

    Returns (b', Pr(o | b, a)).  b is a length-|S| probability vector.
    """
    predicted = b @ P[a]                      # prediction: sum_s b(s) p(s'|s,a)
    unnorm = predicted * Z[a][:, o]           # correction: multiply by the likelihood O(o|s',a)
    pr_o = unnorm.sum()                       # normaliser = probability of observing o
    return unnorm / pr_o, pr_o


def brute_force_posterior(actions, observations):
    """Pr(S_T = s | a_0, o_1, ..., a_{T-1}, o_T) by summing over ALL hidden paths s_0..s_T.

    No recursion: just the definition of conditional probability applied to the joint
    distribution of the model.  Exponential in T, so only for short histories.
    """
    T = len(actions)
    joint = np.zeros(2)
    for path in itertools.product(range(2), repeat=T + 1):
        pr = D0[path[0]]
        for t in range(T):
            pr *= P[actions[t], path[t], path[t + 1]] * Z[actions[t], path[t + 1], observations[t]]
        joint[path[T]] += pr
    return joint / joint.sum()


def check_belief_update_exact(rng, n_histories=300, max_len=8):
    """Max |recursive belief - brute-force posterior| over random action-observation histories."""
    worst = 0.0
    for _ in range(n_histories):
        T = rng.integers(1, max_len + 1)
        acts = rng.choice(3, size=T, p=[0.7, 0.15, 0.15])
        obs = rng.integers(0, 2, size=T)
        b = D0.copy()
        for a, o in zip(acts, obs):
            b, _ = belief_update(b, a, o)
        worst = max(worst, np.abs(b - brute_force_posterior(acts, obs)).max())
    return worst


def check_belief_update_mc(rng, n_runs=400_000, T=4, min_count=2_000):
    """Simulate a random policy and compare empirical posteriors with the analytic belief.

    The policy listens w.p. 0.75 and otherwise opens a random door, so histories contain resets.
    For every history seen at least `min_count` times we compare the fraction of runs in which
    the tiger is actually on the left with b(TL).
    """
    s = rng.integers(0, 2, size=n_runs)
    keys = np.zeros(n_runs, dtype=np.int64)       # the (a, o) history as a base-6 integer
    for _ in range(T):
        a = np.where(rng.random(n_runs) < 0.75, 0, rng.integers(1, 3, size=n_runs))
        reset = a > 0
        s = np.where(reset, rng.integers(0, 2, size=n_runs), s)
        correct = rng.random(n_runs) < ACC
        o_listen = np.where(correct, s, 1 - s)    # growl on the tiger's side w.p. 0.85
        o = np.where(reset, rng.integers(0, 2, size=n_runs), o_listen)
        keys = keys * 6 + a * 2 + o
    rows = []
    for key in np.unique(keys):
        mask = keys == key
        n = int(mask.sum())
        if n < min_count:
            continue
        hist, k = [], int(key)
        for _ in range(T):
            hist.append(((k % 6) // 2, k % 2))   # (action, observation)
            k //= 6
        b = D0.copy()
        for a, o in hist[::-1]:
            b, _ = belief_update(b, a, o)
        emp = float((s[mask] == 0).mean())
        se = np.sqrt(max(b[0] * (1 - b[0]), 1e-12) / n)
        rows.append((b[0], emp, se, n))
    return rows


# ----------------------------------------------------------------------------------------------
# Value iteration with alpha vectors (Section 2.5)
# ----------------------------------------------------------------------------------------------
def _line_x(c, m, i, j):
    """b at which lines i and j (value = c + m*b) intersect."""
    return (c[i] - c[j]) / (m[j] - m[i])


def prune(G, acts, eps=0.0, tol=1e-10):
    """Keep only the alpha vectors that are maximal somewhere on the belief simplex.

    With two hidden states, alpha . b = c + m b with b = Pr(TL), c = alpha(TR) and
    m = alpha(TL) - alpha(TR): a line on [0, 1].  The upper envelope of lines is computed exactly
    by the 'convex hull trick' (sort by slope, pop lines that are never on top).  With eps > 0 we
    additionally drop, one at a time, any envelope line that raises the envelope by less than
    eps anywhere (epsilon-pruning): the represented V changes by at most eps per removal.
    """
    m = G[:, 0] - G[:, 1]
    c = G[:, 1]
    order = np.lexsort((-c, m))                  # slope ascending, then intercept descending
    hull = []
    for i in order:
        if hull and abs(m[hull[-1]] - m[i]) < tol:   # (numerically) parallel lines:
            if c[i] <= c[hull[-1]]:
                continue                             # ... keep only the higher one
            hull.pop()
        while len(hull) >= 2:
            l1, l2 = hull[-2], hull[-1]
            if _line_x(c, m, l1, i) <= _line_x(c, m, l1, l2) + tol:
                hull.pop()                       # l2 is below max(l1, i) everywhere
            else:
                break
        hull.append(i)
    # restrict to b in [0, 1]
    keep = []
    for j, i in enumerate(hull):
        lo = -np.inf if j == 0 else _line_x(c, m, hull[j - 1], i)
        hi = np.inf if j == len(hull) - 1 else _line_x(c, m, i, hull[j + 1])
        if hi > tol and lo < 1 - tol and hi - lo > tol:
            keep.append(i)
    if eps > 0:
        changed = True
        while changed and len(keep) > 1:
            changed = False
            for j, i in enumerate(keep):
                left = keep[j - 1] if j > 0 else None
                right = keep[j + 1] if j + 1 < len(keep) else None
                lo = 0.0 if left is None else max(0.0, _line_x(c, m, left, i))
                hi = 1.0 if right is None else min(1.0, _line_x(c, m, i, right))
                xs = [lo, hi]
                if left is not None and right is not None:
                    x = _line_x(c, m, left, right)
                    if lo < x < hi:
                        xs.append(x)
                gain = max(c[i] + m[i] * x - max(c[k] + m[k] * x for k in (left, right) if k is not None)
                           for x in xs)
                if gain < eps:                   # removing i lowers V by < eps everywhere
                    keep.pop(j)
                    changed = True
                    break
    keep = np.array(keep, dtype=int)
    return G[keep], acts[keep]


def alpha_backup(G, acts, eps=0.0):
    """One Bellman backup of a PWLC value function, Eqs. (15.9)-(15.10).

    For each action a and observation o, every old vector alpha' gives a back-projected vector
        u_{a,o}^{alpha'}(s) = sum_{s'} p(s'|s,a) O(o|s',a) alpha'(s')        (Eq. 15.9)
    The new vectors for action a are r(., a) + gamma * (one choice of u per observation), Eq.
    (15.10): a cross-sum,
    pruned incrementally (after each observation) to keep it small.  Returns the pruned set,
    its root-action labels, and the number of vectors an unpruned backup would have created
    (|A| |Gamma|^|O|).
    """
    L = len(G)
    new_G, new_a = [], []
    for a in range(3):
        cross = R[:, a][None, :].copy()
        for o in range(2):
            M = P[a] * Z[a][:, o][None, :]       # M[s, s'] = p(s'|s,a) O(o|s',a)
            u = GAMMA * G @ M.T                   # gamma * u_{a,o}^{alpha'} for every alpha', (L, |S|)
            cross = (cross[:, None, :] + u[None, :, :]).reshape(-1, 2)
            cross, _ = prune(cross, np.zeros(len(cross), dtype=int), eps=eps)
        new_G.append(cross)
        new_a.append(np.full(len(cross), a))
    G2, a2 = prune(np.vstack(new_G), np.concatenate(new_a), eps=eps)
    return G2, a2, 3 * L ** 2


def value_iteration(eps, tol, max_iter):
    G = np.zeros((1, 2))                          # V_0 = 0: horizon-0 value
    acts = np.zeros(1, dtype=int)
    grid = np.linspace(0, 1, 2001)
    B = np.stack([grid, 1 - grid], axis=1)
    V_old = (B @ G.T).max(axis=1)
    history = []      # (n, |Gamma_n|, vectors an unpruned backup would create, residual)
    for n in range(1, max_iter + 1):
        G, acts, n_gen = alpha_backup(G, acts, eps=eps)
        V = (B @ G.T).max(axis=1)
        res = np.abs(V - V_old).max()
        history.append((n, len(G), n_gen, res))
        V_old = V
        if res < tol:
            break
    return G, acts, history


def greedy_action(G, acts, b_tl):
    return int(acts[np.argmax(G @ np.array([b_tl, 1 - b_tl]))])


# ----------------------------------------------------------------------------------------------
# Finite-state controllers (Section 2.7).  A controller is a set of memory nodes, an action
# distribution pi(a | node) and a deterministic memory update node' = delta(node, a, o).
# ----------------------------------------------------------------------------------------------
def compile_controller(start, policy, transition, max_nodes=20_000):
    """Enumerate the nodes reachable from `start` and tabulate pi[N, 3] and nxt[N, 3, 2]."""
    nodes, index, frontier = [start], {start: 0}, [start]
    pis = {}
    while frontier:
        new = []
        for n in frontier:
            pis[n] = np.asarray(policy(n), dtype=float)
            for a in range(3):
                if pis[n][a] == 0:
                    continue
                for o in range(2):
                    m = transition(n, a, o)
                    if m not in index:
                        index[m] = len(nodes)
                        nodes.append(m)
                        new.append(m)
        if len(nodes) > max_nodes:
            raise RuntimeError("controller has too many nodes")
        frontier = new
    N = len(nodes)
    pi = np.stack([pis[n] for n in nodes])
    nxt = np.zeros((N, 3, 2), dtype=int)
    for j, n in enumerate(nodes):
        for a in range(3):
            for o in range(2):
                nxt[j, a, o] = index[transition(n, a, o)] if pi[j, a] > 0 else j
    return nodes, pi, nxt


def evaluate(pi, nxt):
    """Exact value from b0 = 0.5 at node 0 (Eq. 15.12): a linear solve on (node, state) pairs.

    V(n, s) = sum_a pi(a|n) [ r(s,a) + gamma sum_{s',o} p(s'|s,a) O(o|s',a) V(delta(n,a,o), s') ].
    """
    N = len(pi)
    M = np.eye(2 * N)
    rhs = (pi @ R.T).reshape(-1)                 # rhs[2n + s] = sum_a pi(a|n) r(s, a)
    rows_n = np.arange(N)
    for a in range(3):
        for s in range(2):
            for s2 in range(2):
                for o in range(2):
                    w = GAMMA * pi[:, a] * P[a, s, s2] * Z[a, s2, o]
                    np.add.at(M, (2 * rows_n + s, 2 * nxt[:, a, o] + s2), -w)
    V = np.linalg.solve(M, rhs)
    return 0.5 * (V[0] + V[1])


def simulate(rng, pi, nxt, n_episodes, horizon=300):
    """Monte Carlo estimate of the same value (vectorised; truncation error gamma^300 ~ 2e-7)."""
    s = rng.integers(0, 2, size=n_episodes)
    node = np.zeros(n_episodes, dtype=int)
    ret = np.zeros(n_episodes)
    total = np.zeros(n_episodes)
    disc = 1.0
    cum = pi.cumsum(axis=1)
    for _ in range(horizon):
        a = (rng.random(n_episodes)[:, None] > cum[node]).sum(axis=1)
        a = np.minimum(a, 2)
        r = R[s, a]
        ret += disc * r
        total += r
        disc *= GAMMA
        s2 = np.where(a == 0, s, rng.integers(0, 2, size=n_episodes))
        o = np.where(a == 0, np.where(rng.random(n_episodes) < ACC, s2, 1 - s2),
                     rng.integers(0, 2, size=n_episodes))
        node = nxt[node, a, o]
        s = s2
    return ret.mean(), ret.std(ddof=1) / np.sqrt(n_episodes), total.mean() / horizon


def opening_diagnostics(rng, pi, nxt, n_runs=20_000, horizon=100):
    """When a controller opens a door, how sure SHOULD it have been?

    Tracks, alongside the controller, the net growl count c since the last reset (#HL - #HR); the
    true belief is a function of c alone.  Returns the fraction of openings made with net count
    |c| <= 1 (premature: the opened door is safe w.p. <= 0.85), |c| = 2 (as the belief-optimal
    policy does: safe w.p. 0.9698) and |c| >= 3 (later than necessary), plus the fraction of
    openings that hit the tiger.
    """
    s = rng.integers(0, 2, size=n_runs)
    node = np.zeros(n_runs, dtype=int)
    c = np.zeros(n_runs, dtype=int)
    cum = pi.cumsum(axis=1)
    counts = np.zeros(3)
    wrong = 0
    for _ in range(horizon):
        a = np.minimum((rng.random(n_runs)[:, None] > cum[node]).sum(axis=1), 2)
        opened = a > 0
        # evidence in favour of the door being SAFE: open-right is safe if TL (c > 0)
        support = np.where(a == 2, c, -c)[opened]
        counts += [(support <= 1).sum(), (support == 2).sum(), (support >= 3).sum()]
        wrong += int(((a == 1) & (s == 0)).sum() + ((a == 2) & (s == 1)).sum())
        s2 = np.where(opened, rng.integers(0, 2, size=n_runs), s)
        o = np.where(opened, rng.integers(0, 2, size=n_runs),
                     np.where(rng.random(n_runs) < ACC, s2, 1 - s2))
        c = np.where(opened, 0, c + np.where(o == 0, 1, -1))
        node = nxt[node, a, o]
        s = s2
    total = counts.sum()
    return counts / max(total, 1), wrong / max(total, 1), total


def onehot(a):
    v = np.zeros(3)
    v[a] = 1.0
    return v


def belief_controller(G, acts):
    """The belief-optimal policy: the controller node IS the (rounded) belief."""
    def policy(n):
        return onehot(greedy_action(G, acts, n))

    def transition(n, a, o):
        b2, _ = belief_update(np.array([n, 1 - n]), a, o)
        return round(float(b2[0]), 12)
    return compile_controller(0.5, policy, transition)


def truncated_belief(window):
    b = D0.copy()
    for item in window:
        if item is not None:
            b, _ = belief_update(b, item[0], item[1])
    return b[0]


def window_controller(k, rule):
    """'Frame stacking': the node is the last k (action, observation) pairs (None = padding).

    rule(window) -> action.  Older information is forgotten.
    """
    def policy(n):
        return onehot(rule(n))

    def transition(n, a, o):
        return n[1:] + ((a, o),)
    return compile_controller((None,) * k, policy, transition)


def optimise_window(k, base_rule, max_sweeps=10):
    """Coordinate ascent over deterministic window-k policies with exact evaluation.

    Start from `base_rule`; repeatedly try every alternative action at every reachable window
    and keep any change that increases the exact value.  This finds a LOCAL optimum (the exact
    problem, choosing the best finite-memory policy, is NP-hard in general).
    """
    table = {}

    def rule(n):
        return table.get(n, base_rule(n))
    nodes, pi, nxt = window_controller(k, rule)
    best = evaluate(pi, nxt)
    for _ in range(max_sweeps):
        improved = False
        for n in list(nodes):
            current = rule(n)
            for a in range(3):
                if a == current:
                    continue
                table[n] = a
                cand_nodes, cpi, cnxt = window_controller(k, rule)
                v = evaluate(cpi, cnxt)
                if v > best + 1e-9:
                    best, nodes, current, improved = v, cand_nodes, a, True
                else:
                    table[n] = current
        if not improved:
            break
    return rule, best


def memoryless_controller(table):
    """pi(a | last observation); node 'start' before the first observation.

    After a door is opened the next observation is noise, exactly as in the model.
    """
    return compile_controller("start", lambda n: table[n], lambda n, a, o: o)


def best_memoryless(n_grid):
    """Exhaustive search over memoryless policies.

    Deterministic: all 3^3 maps {start, HL, HR} -> action.  Stochastic: every pair
    (pi(.|HL), pi(.|HR)) on a grid of the probability simplex (start action: listen).
    """
    best = (-np.inf, None)
    for choice in itertools.product(range(3), repeat=3):
        table = {key: onehot(c) for key, c in zip(["start", 0, 1], choice)}
        _, pi, nxt = memoryless_controller(table)
        v = evaluate(pi, nxt)
        if v > best[0]:
            best = (v, table)
    simplex = [np.array([i, j, n_grid - i - j]) / n_grid
               for i in range(n_grid + 1) for j in range(n_grid + 1 - i)]
    for p_hl in simplex:
        for p_hr in simplex:
            table = {"start": onehot(0), 0: p_hl, 1: p_hr}
            _, pi, nxt = memoryless_controller(table)
            v = evaluate(pi, nxt)
            if v > best[0] + 1e-12:
                best = (v, table)
    q_curve = []
    for q in np.linspace(0, 1, 21):
        table = {"start": onehot(0), 0: np.array([1 - q, 0, q]), 1: np.array([1 - q, q, 0])}
        _, pi, nxt = memoryless_controller(table)
        q_curve.append((q, evaluate(pi, nxt)))
    return best, q_curve


def memoryless_value(pi3):
    """Exact value (Eq. 15.12) of the memoryless controller with nodes 0 = start, 1 = HL, 2 = HR.

    pi3[n] = action distribution at node n; the node after observation o is 1 + o.  A direct
    6 x 6 solve (no reachability bookkeeping), fast enough for continuous optimisation.
    """
    K = np.einsum("ast,ato->asot", P, Z)              # K[a,s,o,s'] = p(s'|s,a) O(o|s',a)
    T = np.zeros((3, 2, 3, 2))
    T[:, :, 1:, :] = np.einsum("na,asot->nsot", pi3, K)
    M = np.eye(6) - GAMMA * T.reshape(6, 6)
    V = np.linalg.solve(M, (pi3 @ R.T).reshape(-1))
    return 0.5 * (V[0] + V[1])


def local_search_memoryless(rng, n_starts):
    """Continuous search over ALL memoryless policies, including a randomised start action.

    Each node's distribution is parameterised by (u, v) in [0,1]^2 ("stick breaking"):
    listen u, open-left (1-u) v, open-right (1-u)(1-v), so the boundary (pure actions) is
    reachable.  L-BFGS-B with numerical gradients from random starting points; returns the best
    value found and the corresponding table.
    """
    from scipy.optimize import minimize

    def to_pi(x):
        u, v = x[0::2], x[1::2]
        return np.stack([u, (1 - u) * v, (1 - u) * (1 - v)], axis=1)

    best_v, best_pi = -np.inf, None
    for _ in range(n_starts):
        res = minimize(lambda x: -memoryless_value(to_pi(x)), rng.random(6), method="L-BFGS-B",
                       bounds=[(0.0, 1.0)] * 6)
        if -res.fun > best_v:
            best_v, best_pi = -res.fun, to_pi(res.x)
    return best_v, best_pi


# ----------------------------------------------------------------------------------------------
def make_figures(G, acts, hist_exact, hist_eps, mc_rows, results, example):
    from plot_style import setup, C, GREY
    plt = setup()
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. value function and alpha vectors + growth of the vector sets
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    b = np.linspace(0, 1, 1001)
    B = np.stack([b, 1 - b], axis=1)
    vals = B @ G.T
    styles = {0: "-", 1: "--", 2: "-."}
    for k in range(len(G)):
        ax.plot(b, vals[:, k], color=C[acts[k]], lw=0.8, alpha=0.7, ls=styles[acts[k]])
    V = vals.max(axis=1)
    best = acts[vals.argmax(axis=1)]
    for a in range(3):
        ax.plot(b, np.where(best == a, V, np.nan), color=C[a], lw=3.2, ls=styles[a],
                label=f"{A_NAMES[a]} is optimal")
    ax.set_ylim(-5, 30)
    ax.set_xlabel("belief b(TL) = Pr(tiger behind the LEFT door)")
    ax.set_ylabel("V*(b)")
    ax.set_title(f"Optimal value: upper envelope of {len(G)} alpha vectors")
    ax.legend(loc="lower center")
    ax = axes[1]
    its = [h[0] for h in hist_exact]
    ax.semilogy(its, [h[2] for h in hist_exact], color=C[1], ls="--", marker=".",
                label="one unpruned backup creates 3|Γ|²")
    ax.semilogy(its, [h[1] for h in hist_exact], color=C[0], marker=".", label="|Γ_n|, exact pruning")
    its2 = [h[0] for h in hist_eps]
    ax.semilogy(its2, [h[1] for h in hist_eps], color=C[2], ls="-.", label="|Γ_n|, ε-pruning (ε = 1e-6)")
    ax.semilogy(its2, [h[3] for h in hist_eps], color=GREY, ls=":", label="residual max_b |V_n − V_n−1|")
    ax.set_xlabel("value-iteration step n (= planning horizon)")
    ax.set_title("Size of the alpha-vector sets")
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "tiger_value_function.png"))
    plt.close(fig)

    # 2. belief-update checks
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    ax = axes[0]
    beliefs, acts_ex, obs_ex, states_ex = example
    t = np.arange(len(beliefs))
    ax.step(t, beliefs, where="post", color=C[0], label="belief b_t(TL)")
    ax.plot(t[1:], [1.0 if s == 0 else 0.0 for s in states_ex], "o", ms=3.5, color=C[2],
            label="true state S_t (1 = TL)")
    for i, (a, o) in enumerate(zip(acts_ex, obs_ex)):
        if a == 0:
            ax.text(i + 1, 1.05, "L" if o == 0 else "R", ha="center", fontsize=8, color=C[1])
        else:
            ax.axvline(i + 1, color=GREY, lw=0.8, ls=":")
            ax.text(i + 1, -0.12, "OL" if a == 1 else "OR", ha="center", fontsize=8, color=GREY)
    ax.set_ylim(-0.17, 1.12)
    ax.set_xlim(-0.5, len(beliefs) + 13)
    ax.set_xlabel("time step t  (top: growl heard; bottom: door opened)")
    ax.set_ylabel("b_t(TL)")
    ax.set_title("Belief-optimal agent: belief vs hidden state")
    ax.legend(loc="center right", fontsize=8)
    ax = axes[1]
    bb = np.array([r[0] for r in mc_rows])
    ee = np.array([r[1] for r in mc_rows])
    se = np.array([r[2] for r in mc_rows])
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--", label="y = x")
    ax.errorbar(bb, ee, yerr=2 * se, fmt="o", ms=3, color=C[0], ecolor=C[0], elinewidth=0.8,
                label="one history (±2 s.e.)")
    ax.set_xlabel("analytic belief b(TL | history), Eq. (15.3)")
    ax.set_ylabel("Monte Carlo frequency of TL")
    ax.set_title(f"{len(mc_rows)} action-observation histories of length 4")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "tiger_belief_check.png"))
    plt.close(fig)

    # 3. value of memory
    fig, ax = plt.subplots(figsize=(10, 4.2))
    names = [r[0] for r in results]
    vals = np.array([r[1] for r in results])
    mc = np.array([r[3] for r in results])
    mcse = np.array([r[4] for r in results])
    x = np.arange(len(results))
    colors = [GREY if "memoryless" in n else (C[3] if "naive" in n else (C[0] if "window" in n else C[2]))
              for n in names]
    ax.bar(x, vals, color=colors, width=0.62)
    ax.errorbar(x, mc, yerr=2 * mcse, fmt="D", color=C[1], ms=4, label="Monte Carlo estimate (±2 s.e.)")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.8 if v >= 0 else 1.0, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(-22, 24)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("exact discounted return from b0 = 0.5")
    ax.set_title("How much memory does the Tiger need?")
    from matplotlib.patches import Patch
    handles = [Patch(color=GREY, label="memoryless"), Patch(color=C[3], label="window: naive rule"),
               Patch(color=C[0], label="window: best policy found"), Patch(color=C[2], label="belief-optimal")]
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + h, loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "tiger_policies.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figures")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    n_mc_runs = 40_000 if args.quick else 400_000
    n_eval_eps = 2_000 if args.quick else 40_000
    n_grid = 4 if args.quick else 10
    k_max = 2 if args.quick else 4
    eps = 1e-6
    print(f"Tiger POMDP | seed={args.seed} gamma={GAMMA} listen accuracy={ACC} "
          f"epsilon-pruning={eps} quick={args.quick}")

    # 1. belief update ------------------------------------------------------------------------
    worst = check_belief_update_exact(rng, n_histories=50 if args.quick else 300)
    print(f"[belief] max |recursive - brute-force posterior| over random histories: {worst:.2e}")
    rows = check_belief_update_mc(rng, n_runs=n_mc_runs, min_count=200 if args.quick else 2_000)
    z = np.array([abs(r[1] - r[0]) / r[2] for r in rows])
    err = np.array([abs(r[1] - r[0]) for r in rows])
    print(f"[belief] Monte Carlo ({n_mc_runs} runs, {len(rows)} histories seen often enough): "
          f"max |emp - b| = {err.max():.4f}, max |z| = {z.max():.2f}, "
          f"fraction |z| < 2 = {np.mean(z < 2):.2f}")
    b = D0.copy()
    trail = [b[0]]
    for o in [0, 0, 1, 0]:
        b, pr = belief_update(b, 0, o)
        trail.append(b[0])
    print("[belief] by hand: listen x4 hearing L, L, R, L gives b(TL) =",
          " -> ".join(f"{x:.4f}" for x in trail))

    # 2. value iteration ---------------------------------------------------------------------
    _, _, hist_exact = value_iteration(eps=0.0, tol=0.0, max_iter=12 if args.quick else 40)
    print("[VI exact] |Gamma_n| for n = 1..: " + ", ".join(str(h[1]) for h in hist_exact))
    print(f"[VI exact] at n = {hist_exact[-1][0]}: |Gamma| = {hist_exact[-1][1]}, an unpruned "
          f"backup would have created {hist_exact[-1][2]} vectors")
    G, acts, hist_eps = value_iteration(eps=eps, tol=1e-6 if args.quick else 1e-10, max_iter=2000)
    n_final, L_final, _, res = hist_eps[-1]
    print(f"[VI eps]   converged after {n_final} backups (residual {res:.1e}); |Gamma| = {L_final}; "
          f"max over n of |Gamma_n| = {max(h[1] for h in hist_eps)}")
    V05 = float((G @ np.array([0.5, 0.5])).max())
    grid = np.linspace(0, 1, 20001)
    pol = np.array([greedy_action(G, acts, x) for x in grid])
    listen = grid[pol == 0]
    print(f"[VI eps]   V*(b=0.5) = {V05:.4f}; listen is optimal for b(TL) in "
          f"[{listen.min():.4f}, {listen.max():.4f}]")
    for k in np.argsort(G[:, 0] - G[:, 1]):
        print(f"           alpha = ({G[k, 0]:9.4f}, {G[k, 1]:9.4f})  action = {A_NAMES[acts[k]]}")

    # 3. controllers -------------------------------------------------------------------------
    results = []

    def add(name, nodes, pi, nxt):
        v = evaluate(pi, nxt)
        m, s, per = simulate(rng, pi, nxt, n_eval_eps)
        frac, wrong, n_open = opening_diagnostics(rng, pi, nxt, n_runs=2_000 if args.quick else 20_000)
        results.append((name, v, len(nodes), m, s, per, frac, wrong))
        return v

    (best_v, table), q_curve = best_memoryless(n_grid)
    add("memoryless\n(best)", *memoryless_controller(table))
    desc = {k: np.round(v, 2).tolist() for k, v in table.items()}
    print(f"[memoryless] best: pi(.|start), pi(.|HL), pi(.|HR) = {desc}; V = {best_v:.4f}")
    print("[memoryless] symmetric family (open the door opposite the growl w.p. q): "
          + ", ".join(f"q={q:.2f}: {val:.2f}" for q, val in q_curve[:5]) + ", ...")
    # consistency of the fast evaluator with the general one, then a continuous local search
    rng_ls = np.random.default_rng(args.seed + 1)            # separate stream: rng is unchanged
    worst_diff = 0.0
    for _ in range(20):
        pi3 = rng_ls.dirichlet(np.ones(3), size=3)
        _, pi, nxt = memoryless_controller({"start": pi3[0], 0: pi3[1], 1: pi3[2]})
        worst_diff = max(worst_diff, abs(memoryless_value(pi3) - evaluate(pi, nxt)))
    n_starts = 10 if args.quick else 200
    v_ls, pi_ls = local_search_memoryless(rng_ls, n_starts)
    print(f"[memoryless] continuous local search over pi(.|start), pi(.|HL), pi(.|HR) (L-BFGS-B, "
          f"{n_starts} random starts; fast evaluator agrees with the general one to {worst_diff:.1e}): "
          f"best V = {v_ls:.4f} at P(listen | start, HL, HR) = "
          + ", ".join(f"{p:.3f}" for p in pi_ls[:, 0]))

    naive_cache = {}

    def naive_rule(n):                           # greedy action on the window's own belief
        if n not in naive_cache:
            naive_cache[n] = greedy_action(G, acts, truncated_belief(n))
        return naive_cache[n]

    prev_rule = None
    for k in range(1, k_max + 1):
        add(f"window k={k}\nnaive", *window_controller(k, naive_rule))
        # Coordinate ascent from two starts: the naive rule, and the best window-(k-1) policy
        # embedded in window-k (ignore the oldest pair).  The second start guarantees that the
        # best value found never decreases with k (window-k policies include window-(k-1) ones).
        starts = [naive_rule]
        if prev_rule is not None:
            starts.append((lambda r: (lambda n: r(n[1:])))(prev_rule))
        found = [optimise_window(k, base, max_sweeps=3 if args.quick else 10) for base in starts]
        rule, v_opt = max(found, key=lambda rv: rv[1])
        print(f"[window k={k}] coordinate ascent: " +
              ", ".join(f"start {i}: {v:.4f}" for i, (_, v) in enumerate(found)))
        add(f"window k={k}\nbest found", *window_controller(k, rule))
        prev_rule = rule
    nodes, pi, nxt = belief_controller(G, acts)
    v_bel = add("belief-\noptimal", nodes, pi, nxt)
    print(f"[belief-optimal] controller value {v_bel:.6f} vs alpha-vector V*(0.5) = {V05:.6f} "
          f"(|diff| = {abs(v_bel - V05):.1e}); reachable beliefs: "
          + ", ".join(f"{n:.4f}" for n in sorted(nodes)))

    print("\n  policy                    exact V(b0)   Monte Carlo (± s.e.)   avg reward/step   #nodes"
          "   openings at net count <=1 / =2 / >=3   tiger hit")
    for name, v, N, m, s, per, frac, wrong in results:
        opens = "        (never opens)" if frac.sum() == 0 else f"   {frac[0]:.3f} / {frac[1]:.3f} / {frac[2]:.3f}"
        print(f"  {name.replace(chr(10), ' '):<25} {v:10.3f}   {m:10.3f} ± {s:5.3f}       "
              f"{per:8.3f}     {N:5d}   {opens}        {wrong:.3f}")

    # one illustrative trajectory of the belief-optimal agent
    rng_ex = np.random.default_rng(3)
    s = rng_ex.integers(0, 2)
    bel = D0.copy()
    beliefs, ex_a, ex_o, ex_s = [bel[0]], [], [], []
    for _ in range(25):
        a = greedy_action(G, acts, bel[0])
        s2 = rng_ex.choice(2, p=P[a, s])
        o = rng_ex.choice(2, p=Z[a, s2])
        bel, _ = belief_update(bel, a, o)
        beliefs.append(bel[0])
        ex_a.append(a)
        ex_o.append(o)
        ex_s.append(s2)                          # b_{t+1} is a belief about S_{t+1}
        s = s2

    if not args.quick:
        make_figures(G, acts, hist_exact, hist_eps, rows, results, (beliefs, ex_a, ex_o, ex_s))
        print(f"figures written to {FIG_DIR}")
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
