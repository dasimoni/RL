"""Batch control on Mountain Car: fitted Q-iteration with an averager, least-squares FQI,
and LSPI (Chapter 08, Section 11.4).

All three methods learn from ONE fixed batch of transitions and never interact with the
environment while learning.

* Dataset. N transitions (s, a, r, s', terminated) with s drawn uniformly from the box
  position in [-1.2, 0.5), velocity in [-0.07, 0.07], and a uniform over the three actions.
  (Trajectories of a random policy from the standard start rarely reach the goal, so they
  would cover the state space poorly.) Each transition is one step of Gymnasium's
  MountainCar-v0 dynamics, re-implemented in vectorised NumPy and checked against
  Gymnasium below. Reward -1 per step; `terminated` = the goal was reached. One-step
  transitions are never truncated, so the only end-of-episode rule needed is
  "no bootstrap after termination". gamma = 0.99 (FQI's contraction argument needs
  gamma < 1).

* (a) K-NN averager FQI: Q_k(s,a) = mean of the targets of the K = 10 stored transitions
  with action a whose states are nearest to s (states rescaled to [0,1]^2). The weights
  depend only on states, never on targets: an averager (Gordon, 1995), so FQI is a
  gamma-contraction in the sup norm and must converge.
* (b) Least-squares FQI on tile coding (8 tilings of 8x8 tiles, 648 features per action,
  as in mountain_car_sarsa.py), ridge lambda: Q_{k+1} = argmin_w sum_i (w^T x(s_i,a_i) - y_i)^2
  + lambda |w|^2. Not an averager: nothing guarantees convergence.
* (c) LSPI (Lagoudakis & Parr, 2003) on the same features: LSTDQ evaluation of the current
  greedy policy from the same batch, then greedy improvement, until the greedy action at
  every stored next state stops changing (or 30 iterations).

Reported: the sup-norm change ||Q_{k+1} - Q_k|| at the points where targets read Q (every
non-terminal next state s'_i and all three actions), iteration counts, and the number of
steps the greedy policy needs to reach the goal from 100 held-out start states drawn from
MountainCar-v0's start distribution (cap 1,000 steps; reference: the hand-made
"push in the direction of motion" policy).

Run:  python code/ch08_function_approximation/fqi_lspi_mountaincar.py [--quick] [--seed 0]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread: the machine is shared
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree

from tiles import TileCoder

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
LOWS, HIGHS = np.array([-1.2, -0.07]), np.array([0.6, 0.07])
GAMMA = 0.99
N_ACTIONS = 3
CAP = 1000                    # step cap for greedy evaluation rollouts
DIVERGED = 1e6                # |Q| beyond this (true values lie in [-100, 0]) = diverged


# --------------------------------------------------------------------------- #
# Dynamics and data
# --------------------------------------------------------------------------- #
def mc_step(pos, vel, a):
    """Vectorised MountainCar-v0 step (same arithmetic as gymnasium's MountainCarEnv.step)."""
    vel = vel + (a - 1) * 0.001 + np.cos(3 * pos) * (-0.0025)
    vel = np.clip(vel, -0.07, 0.07)
    pos = np.clip(pos + vel, -1.2, 0.6)
    vel = np.where((pos == -1.2) & (vel < 0), 0.0, vel)
    terminated = (pos >= 0.5) & (vel >= 0)
    return pos, vel, terminated


def check_dynamics(n, rng):
    """Compare mc_step with gymnasium on n random (state, action) pairs."""
    import gymnasium as gym
    env = gym.make("MountainCar-v0").unwrapped
    env.reset(seed=0)
    S = np.c_[rng.uniform(-1.2, 0.6, n), rng.uniform(-0.07, 0.07, n)]
    S[: n // 20, 0] = -1.2 + 1e-4 * rng.random(n // 20)            # exercise the left wall
    A = rng.integers(0, N_ACTIONS, n)
    p2, v2, term = mc_step(S[:, 0], S[:, 1], A)
    err, mism = 0.0, 0
    for i in range(n):
        env.state = np.array(S[i])
        _, _, t, _, _ = env.step(int(A[i]))
        err = max(err, abs(env.state[0] - p2[i]), abs(env.state[1] - v2[i]))
        mism += int(t != term[i])
    env.close()
    return err, mism


def check_rollout_with_gym(starts, policy, cap=CAP):
    """Steps to goal under gymnasium itself (to confirm the vectorised rollouts)."""
    import gymnasium as gym
    env = gym.make("MountainCar-v0").unwrapped
    env.reset(seed=0)
    out = []
    for x0 in starts:
        env.state = np.array([x0, 0.0])
        n = 0
        while n < cap:
            _, _, t, _, _ = env.step(int(policy(np.array([env.state]))[0]))
            n += 1
            if t:
                break
        out.append(n)
    env.close()
    return np.array(out)


def make_dataset(n, rng):
    pos = rng.uniform(-1.2, 0.5, n)
    vel = rng.uniform(-0.07, 0.07, n)
    a = rng.integers(0, N_ACTIONS, n)
    p2, v2, term = mc_step(pos, vel, a)
    return dict(S=np.c_[pos, vel], a=a, r=-np.ones(n), S2=np.c_[p2, v2], term=term)


def greedy_steps(policy, starts, cap=CAP):
    """Vectorised greedy rollouts from (x0, 0); returns steps to goal (cap if never)."""
    pos, vel = starts.astype(float).copy(), np.zeros(len(starts))
    done = np.zeros(len(starts), bool)
    steps = np.full(len(starts), cap)
    for t in range(cap):
        act = policy(np.c_[pos, vel])
        p2, v2, term = mc_step(pos, vel, act)
        pos, vel = np.where(done, pos, p2), np.where(done, vel, v2)
        new = term & ~done
        steps[new] = t + 1
        done |= term
        if done.all():
            break
    return steps


def heuristic_policy(X):
    """Push in the direction of motion (right when at rest): a classic hand-made controller."""
    return np.where(X[:, 1] >= 0, 2, 0)


# --------------------------------------------------------------------------- #
# Function approximators used inside FQI
# --------------------------------------------------------------------------- #
def unit_box(S):
    return (S - LOWS) / (HIGHS - LOWS)


class KNNAverager:
    """Q(s,a) = mean of the stored targets of the K nearest samples that took action a."""

    def __init__(self, data, K=10):
        self.K = K
        self.rows = [np.flatnonzero(data["a"] == b) for b in range(N_ACTIONS)]
        self.trees = [cKDTree(unit_box(data["S"][r])) for r in self.rows]
        self.nt = ~data["term"]
        S2 = data["S2"][self.nt]
        # neighbour sets of the bootstrap points are fixed: precompute them once
        self.boot_nbrs = [self._nbrs(b, S2) for b in range(N_ACTIONS)]

    def _nbrs(self, b, S):
        _, j = self.trees[b].query(unit_box(S), k=self.K)
        return self.rows[b][j.reshape(len(S), self.K)]

    def fit(self, y):
        return y.copy()                                   # an averager just stores the targets

    def q_boot(self, y):
        return np.stack([y[n].mean(1) for n in self.boot_nbrs], 1)

    def q(self, y, S):
        return np.stack([y[self._nbrs(b, S)].mean(1) for b in range(N_ACTIONS)], 1)


def tile_matrix(tc, S):
    """Sparse 0/1 tile-feature matrix (len(S) x tc.n_grid_features); vectorised TileCoder.active."""
    u = (S - tc.lows) * tc.scale
    c = np.clip(np.floor(u[:, None, :] + tc.offsets[None]).astype(np.int64), 0, tc.dmax)
    idx = tc.base[None] + c @ tc.strides                  # (n, n_tilings)
    n, m = idx.shape
    return sp.csr_matrix((np.ones(n * m), idx.ravel(), np.arange(0, n * m + 1, m)),
                         shape=(n, tc.n_grid_features))


class TileLeastSquares:
    """Per-action ridge regression on tile features: w_a = (X_a^T X_a + lam I)^{-1} X_a^T y_a."""

    def __init__(self, data, lam=1e-3, n_tilings=8, tiles=8):
        self.lam = lam
        self.tc = TileCoder(LOWS, HIGHS, n_tilings=n_tilings, tiles_per_dim=tiles)
        self.d = self.tc.n_grid_features
        self.rows = [np.flatnonzero(data["a"] == b) for b in range(N_ACTIONS)]
        self.X = [tile_matrix(self.tc, data["S"][r]) for r in self.rows]
        self.chol = [cho_factor((X.T @ X).toarray() + lam * np.eye(self.d)) for X in self.X]
        self.nt = ~data["term"]
        self.X2 = tile_matrix(self.tc, data["S2"][self.nt])

    def fit(self, y):
        return np.stack([cho_solve(c, X.T @ y[r]) for c, X, r in zip(self.chol, self.X, self.rows)])

    def q_boot(self, W):
        return np.asarray(self.X2 @ W.T)

    def q(self, W, S):
        return np.asarray(tile_matrix(self.tc, S) @ W.T)

    def hat_inf_norm(self, n_rows, rng):
        """Lower bound on the sup-norm gain of the fit, max_i sum_j |H_ij|, over n_rows sampled
        bootstrap points (H maps the targets of action a to Q(s'_i, a)). An averager has at most 1."""
        rows = rng.choice(self.X2.shape[0], size=min(n_rows, self.X2.shape[0]), replace=False)
        best = 0.0
        for c, X in zip(self.chol, self.X):
            for chunk in np.array_split(rows, max(1, len(rows) // 250)):
                B = cho_solve(c, self.X2[chunk].toarray().T)     # d x m
                H = np.asarray(X @ B).T                          # m x n_a
                best = max(best, float(np.abs(H).sum(1).max()))
        return best


# --------------------------------------------------------------------------- #
# Algorithms
# --------------------------------------------------------------------------- #
def fqi(model, data, max_iter, tol=1e-6, eval_at=(), starts=None):
    """Fitted Q-iteration: y_{k+1} = r + gamma (1 - terminated) max_a' Q_k(s', a'),
    Q_{k+1} = fit(y_{k+1}), from Q_0 = 0. Returns a dict with the sup-norm changes."""
    r, nt = data["r"], ~data["term"]
    y = r.copy()                                       # targets from Q_0 = 0
    q_prev = np.zeros((nt.sum(), N_ACTIONS))           # Q_0 at the bootstrap points
    diffs, evals, status = [], [], "max_iter"
    for k in range(1, max_iter + 1):
        params = model.fit(y)                          # Q_k
        qb = model.q_boot(params)
        diffs.append(float(np.abs(qb - q_prev).max()))
        q_prev = qb
        if k in eval_at:
            evals.append((k, greedy_steps(lambda X: model.q(params, X).argmax(1), starts).mean()))
        if not np.isfinite(qb).all() or np.abs(qb).max() > DIVERGED:
            status = "diverged"
            break
        if k > 1 and diffs[-1] < tol:
            status = "converged"
            break
        y = r.copy()
        y[nt] += GAMMA * qb.max(1)
    return dict(params=params, diffs=np.array(diffs), iters=k, status=status, evals=evals,
                q_boot=qb)


def lspi(data, lam=1e-3, max_iter=30, starts=None, n_tilings=8, tiles=8):
    """LSPI with LSTDQ on stacked tile features x(s,a). pi_0 is the uniform-random policy
    (expected next features); afterwards pi_{m+1} is greedy w.r.t. Q_m. x(s', .) = 0 only
    when s' is terminal (one-step data are never truncated)."""
    tc = TileCoder(LOWS, HIGHS, n_tilings=n_tilings, tiles_per_dim=tiles)
    d, n = tc.n_grid_features, len(data["r"])
    nt = ~data["term"]

    def stacked(X, acts):
        X = X.tocoo()
        return sp.csr_matrix((X.data, (X.row, X.col + acts[X.row] * d)), shape=(X.shape[0], N_ACTIONS * d))

    Phi = stacked(tile_matrix(tc, data["S"]), data["a"])           # x(s_i, a_i)
    X2 = tile_matrix(tc, data["S2"])
    mask = sp.diags(nt.astype(float))
    X2a = [mask @ stacked(X2, np.full(n, b)) for b in range(N_ACTIONS)]   # x(s'_i, b), 0 if terminal
    PhiT = Phi.T.tocsr()
    b_vec = PhiT @ data["r"]
    nxt = (X2a[0] + X2a[1] + X2a[2]) / N_ACTIONS                    # pi_0 = uniform random
    pol_prev, q_prev, hist = None, None, []
    for m in range(1, max_iter + 1):
        A = (PhiT @ (Phi - GAMMA * nxt)).toarray() + lam * np.eye(N_ACTIONS * d)
        w = np.linalg.solve(A, b_vec)                              # LSTDQ
        qb = np.stack([X2a[b] @ w for b in range(N_ACTIONS)], 1)[nt]
        pol = qb.argmax(1)
        changed = None if pol_prev is None else int((pol != pol_prev).sum())
        dq = None if q_prev is None else float(np.abs(qb - q_prev).max())
        W = w.reshape(N_ACTIONS, d)
        steps = greedy_steps(lambda X: np.asarray(tile_matrix(tc, X) @ W.T).argmax(1), starts).mean()
        hist.append(dict(m=m, changed=changed, dq=dq, steps=float(steps), max_abs_q=float(np.abs(qb).max())))
        if changed == 0:
            break
        pol_prev, q_prev = pol, qb
        full = np.zeros(n, dtype=np.int64)
        full[nt] = pol
        nxt = mask @ stacked(X2, full)                             # x(s'_i, pi_{m+1}(s'_i))
    return dict(W=W, tc=tc, hist=hist, converged=hist[-1]["changed"] == 0, q_boot=qb)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figures")
    ap.add_argument("--seed", type=int, default=0, help="first dataset seed")
    args = ap.parse_args()
    t0 = time.time()

    if args.quick:
        sizes, n_seeds, max_fqi, max_lspi, n_starts, n_check = [3000], 1, 60, 3, 10, 200
        lams, trace_sizes, eval_at = [1e-3], [], ()
    else:
        sizes, n_seeds, max_fqi, max_lspi, n_starts, n_check = [5000, 20000, 50000], 5, 3000, 30, 100, 20000
        lams, trace_sizes = [1e-3, 1.0], [5000, 50000]
        eval_at = tuple([1] + list(range(10, 401, 10)) + list(range(450, 3001, 50)))
    K = 10
    print(f"seed={args.seed} (dataset seeds {args.seed}..{args.seed + n_seeds - 1}; start states: seed 12345)")
    print(f"gamma={GAMMA}, K={K} neighbours, tile coding 8 tilings x 8x8 (648 features/action), "
          f"ridge lambda in {lams}, FQI tolerance 1e-6, max {max_fqi} FQI / {max_lspi} LSPI iterations")

    err, mism = check_dynamics(n_check, np.random.default_rng(999))
    print(f"\n[0] vectorised dynamics vs gymnasium on {n_check} random transitions: "
          f"max |state difference| = {err:.1e}, terminated mismatches = {mism}")
    starts = np.random.default_rng(12345).uniform(-0.6, -0.4, n_starts)
    h = greedy_steps(heuristic_policy, starts)
    hg = check_rollout_with_gym(starts[:10], heuristic_policy)
    assert np.array_equal(h[:10], hg), (h[:10], hg)
    print(f"    reference policy 'push in the direction of motion': {h.mean():.1f} steps "
          f"(min {h.min()}, max {h.max()}) from {n_starts} held-out starts; "
          f"gymnasium rollouts agree on the first 10 starts")

    results, traces = [], {}
    for N in sizes:
        for seed in range(args.seed, args.seed + n_seeds):
            data = make_dataset(N, np.random.default_rng(seed))
            trace = (seed == args.seed and N in trace_sizes)
            ev = eval_at if trace else ()
            row = dict(N=N, seed=seed, n_boot=int((~data["term"]).sum()))
            # (a) K-NN averager FQI
            knn = KNNAverager(data, K=K)
            out = fqi(knn, data, max_fqi, eval_at=ev, starts=starts)
            st = greedy_steps(lambda X: knn.q(out["params"], X).argmax(1), starts)
            ratios = out["diffs"][1:] / np.maximum(out["diffs"][:-1], 1e-300)
            assert ratios.max() <= GAMMA + 1e-9, ratios.max()        # the averager contraction
            row["knn"] = dict(status=out["status"], iters=out["iters"], steps=st.mean(),
                              succ=(st <= 200).mean(), max_ratio=ratios.max())
            if trace:
                traces[("knn", N)] = out
            # (b) least-squares FQI on tiles
            for lam in lams:
                ls = TileLeastSquares(data, lam=lam)
                out = fqi(ls, data, max_fqi, eval_at=ev if lam == lams[0] else (), starts=starts)
                st = greedy_steps(lambda X: ls.q(out["params"], X).argmax(1), starts)
                ratios = out["diffs"][1:] / np.maximum(out["diffs"][:-1], 1e-300)
                row[("ls", lam)] = dict(status=out["status"], iters=out["iters"], steps=st.mean(),
                                        succ=(st <= 200).mean(), max_ratio=ratios.max(),
                                        q_boot=out["q_boot"])
                if trace and lam == lams[0]:
                    traces[("ls", N)] = out
                    row["hat"] = ls.hat_inf_norm(2000, np.random.default_rng(7))
            # (c) LSPI on the same features
            lo = lspi(data, lam=lams[0], max_iter=max_lspi, starts=starts)
            last = lo["hist"][-1]
            row["lspi"] = dict(converged=lo["converged"], iters=len(lo["hist"]), steps=last["steps"],
                               changed=last["changed"], hist=lo["hist"])
            Wl = lo["W"]
            st = greedy_steps(lambda X: np.asarray(tile_matrix(lo["tc"], X) @ Wl.T).argmax(1), starts)
            row["lspi"]["succ"] = (st <= 200).mean()
            ls_main = row[("ls", lams[0])]
            if lo["converged"] and ls_main["status"] == "converged":
                row["lspi"]["gap_to_lsfqi"] = float(np.abs(lo["q_boot"] - ls_main["q_boot"]).max())
            if trace:
                traces[("lspi", N)] = lo
            results.append(row)
            k_, l_, p_ = row["knn"], ls_main, row["lspi"]
            print(f"  N={N:6d} seed={seed}: kNN-FQI {k_['status']} in {k_['iters']:4d} it, "
                  f"{k_['steps']:6.1f} steps ({100 * k_['succ']:.0f}% <=200) | "
                  f"LS-FQI {l_['status']:9s} in {l_['iters']:4d} it, {l_['steps']:6.1f} steps "
                  f"({100 * l_['succ']:.0f}%) | LSPI {'converged' if p_['converged'] else 'not conv.'} "
                  f"after {p_['iters']:2d} it, {p_['steps']:6.1f} steps ({100 * p_['succ']:.0f}%)"
                  + (f" | max|Q_LSPI - Q_LSFQI| = {p_['gap_to_lsfqi']:.1e}" if "gap_to_lsfqi" in p_ else ""),
                  flush=True)

    # ---- summary table ----
    print(f"\n[1] Summary over {n_seeds} datasets per N (greedy steps: mean over seeds of the mean over "
          f"{n_starts} starts, cap {CAP}; '<=200' = fraction of all start/seed pairs reaching the goal "
          f"within Gymnasium's 200-step limit)")

    def line(name, rows, key, conv):
        ok = [conv(r[key]) for r in rows]
        its = [r[key]["iters"] for r in rows]
        steps = np.array([r[key]["steps"] for r in rows])
        succ = np.mean([r[key]["succ"] for r in rows])
        se = steps.std(ddof=1) / np.sqrt(len(steps)) if len(steps) > 1 else 0.0
        extra = ""
        if "max_ratio" in rows[0][key]:
            mr = [r[key]["max_ratio"] for r in rows if r[key]["status"] == "converged"]
            extra = f" | max step ratio (converged runs) {max(mr):.3f}" if mr else ""
        print(f"  {name:22s} converged {sum(ok)}/{len(rows)} | iterations {min(its)}-{max(its)} | "
              f"steps {steps.mean():6.1f} +- {se:5.1f} (per seed: {', '.join(f'{s:.0f}' for s in steps)}) "
              f"| <=200: {100 * succ:.0f}%{extra}")

    for N in sizes:
        rows = [r for r in results if r["N"] == N]
        print(f" N = {N}")
        line("K-NN averager FQI", rows, "knn", lambda d: d["status"] == "converged")
        for lam in lams:
            line(f"LS-FQI, lambda={lam:g}", rows, ("ls", lam), lambda d: d["status"] == "converged")
            div = sum(r[("ls", lam)]["status"] == "diverged" for r in rows)
            if div:
                its = [r[("ls", lam)]["iters"] for r in rows if r[("ls", lam)]["status"] == "diverged"]
                print(f"  {'':22s} ({div} diverged: |Q| > {DIVERGED:.0e} after {', '.join(map(str, its))} iterations)")
        line(f"LSPI, lambda={lams[0]:g}", rows, "lspi", lambda d: d["converged"])
        nc = [f"{r['lspi']['changed']}/{r['n_boot']}" for r in rows if not r["lspi"]["converged"]]
        if nc:
            print(f"  {'':22s} (non-converged LSPI runs still changed the greedy action at "
                  f"{', '.join(nc)} non-terminal next states in their last iteration)")
        gaps = [r["lspi"]["gap_to_lsfqi"] for r in rows if "gap_to_lsfqi" in r["lspi"]]
        if gaps:
            print(f"  {'':22s} where both converged ({len(gaps)} runs): max |Q_LSPI - Q_LSFQI| at the "
                  f"bootstrap points = {max(gaps):.1e}")

    if traces:
        print("\n[2] Traces (first dataset seed)")
        for N in trace_sizes:
            for name in ("knn", "ls"):
                tr = traces[(name, N)]
                d = tr["diffs"]
                pick = [i for i in (1, 2, 10, 50, 100, 200, 300, 500, 1000) if i <= len(d)]
                print(f"  {name:4s} N={N}: {tr['status']} after {tr['iters']} iterations; "
                      + ", ".join(f"||Q_{i} - Q_{i - 1}|| = {d[i - 1]:.3g}" for i in pick))
                ev = dict(tr["evals"])
                pk = [k for k in (1, 50, 100, 150, 200, 300) if k in ev]
                print(f"       greedy steps at iteration " + ", ".join(f"{k}: {ev[k]:.1f}" for k in pk))
            lh = traces[("lspi", N)]["hist"]
            print(f"  lspi N={N}: greedy-action changes per iteration "
                  f"{[h['changed'] for h in lh[1:]]}; steps {[round(h['steps'], 1) for h in lh]}")
        for N in trace_sizes:
            r0 = [r for r in results if r["N"] == N and r["seed"] == args.seed][0]
            print(f"  sup-norm gain of the LS tile fit (lambda={lams[0]:g}), N={N}: max_i sum_j |H_ij| >= "
                  f"{r0['hat']:.1f} over 2,000 sampled bootstrap points (an averager has at most 1)")

    if not args.quick:
        make_figure(traces, trace_sizes, h.mean())
    print(f"\ntotal time {time.time() - t0:.1f} s")


def make_figure(traces, trace_sizes, heuristic_steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    colors = {"knn": "tab:blue", "ls": "tab:orange", "lspi": "tab:green"}
    labels = {"knn": "K-NN averager FQI", "ls": "least-squares FQI (tiles)", "lspi": "LSPI (tiles)"}
    styles = {trace_sizes[0]: "--", trace_sizes[-1]: "-"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    for N in trace_sizes:
        for name in ("knn", "ls"):
            d = traces[(name, N)]["diffs"]
            ax.semilogy(np.arange(1, len(d) + 1), d, color=colors[name], ls=styles[N], lw=2,
                        label=f"{labels[name]}, N={N:,}")
    k = np.arange(1, 801)
    ax.semilogy(k, GAMMA ** (k - 1), color="gray", lw=1, ls=":", label=r"$\gamma^{k-1}$ (averager bound)")
    ax.set_xlim(0, 800)
    ax.set_ylim(1e-7, 1e7)
    ax.set_xlabel("FQI iteration k")
    ax.set_ylabel(r"$\|Q_k - Q_{k-1}\|_\infty$ at the bootstrap points")
    ax.set_title("Sup-norm change per FQI iteration")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    for N in trace_sizes:
        for name in ("knn", "ls"):
            ev = traces[(name, N)]["evals"]
            ax.plot([e[0] for e in ev], [e[1] for e in ev], color=colors[name], ls=styles[N], lw=2,
                    label=f"{labels[name]}, N={N:,}")
    ax.axhline(heuristic_steps, color="gray", lw=1, ls=":", label="push in direction of motion")
    ax.set_xlim(0, 600)
    ax.set_ylim(0, CAP + 20)
    ax.set_xlabel("FQI iteration k")
    ax.set_ylabel(f"greedy steps to goal (mean, 100 starts, cap {CAP})")
    ax.set_title("Greedy policy of $Q_k$")
    ax.legend(fontsize=8, loc="center right")

    ax = axes[2]
    for N in trace_sizes:
        lh = traces[("lspi", N)]["hist"]
        m = [hh["m"] for hh in lh[1:]]
        ch = [max(hh["changed"], 0.5) for hh in lh[1:]]
        small = N == trace_sizes[0]
        ax.semilogy(m, ch, color="tab:olive" if small else colors["lspi"], ls=styles[N], lw=1.5,
                    marker="s" if small else "o", ms=5, mfc="none" if small else None,
                    label=f"LSPI, N={N:,}")
    ax.set_xlabel("LSPI iteration m")
    ax.set_ylabel("next states whose greedy action changed")
    ax.set_title("LSPI policy changes (0 plotted as 0.5)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fqi_lspi_mountaincar.png"), dpi=110)
    plt.close(fig)
    print(f"figure written to {os.path.join(FIG_DIR, 'fqi_lspi_mountaincar.png')}")


if __name__ == "__main__":
    main()
