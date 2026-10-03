"""Multi-objective RL on Deep Sea Treasure: Pareto front, convex coverage set, SFs + GPI, mixtures.

Chapter 15, Section 6.7 (Eq. 15.29d). Deep Sea Treasure (DST; Vamplew, Yearwood, Dazeley & Berry,
2008; a standard benchmark since Vamplew et al., 2011): a submarine starts in the top-left cell
of an 11 x 10 grid and moves up/down/left/right (deterministic; moves into rock or off the grid
leave it in place). Reaching one of ten treasures ends the episode. The
reward is a VECTOR r = (treasure value, -1 per step), undiscounted (gamma = 1).

Part 1 - Pareto front.  Every policy that ends with treasure k needs at least the shortest-path time
    t_k to it, so the Pareto front of deterministic policies is {(value_k, -t_k)}; all ten points
    are Pareto optimal.
Part 2 - linear scalarization.  For w = (w1, 1 - w1), the scalarized MDP with reward w.r is solved
    exactly (value iteration) on a grid of 1001 weights: only the SUPPORTED points (those on the
    convex hull of the front) are ever optimal.  The same sweep with tabular Q-learning
    (101 weights, epsilon-greedy) finds only supported points too.
Part 3 - Optimistic Linear Support (OLS) + successor features + GPI.  OLS solves the scalarized
    problem only at "corner weights" and builds the convex coverage set (CCS); with
    phi = r (the vector reward) the SFs of a policy are its vector value, psi^pi(s, a) = q^pi(s, a)
    in R^2, and GPI over the library gives the optimal scalarized value for EVERY weight
    (checked on 1001 weights).
Part 4 - SER vs ESR.  Under the scalarized expected return (SER) u(E[G]), a stochastic mixture
    of two CCS policies (pick one at the start of the episode) dominates an unsupported point;
    under the expected scalarized return (ESR) E[u(G)] with a nonlinear utility, the unsupported
    deterministic policy can be strictly better.  Checked exactly and by simulation.

Run:  python code/ch15_beyond_mdps/deep_sea_treasure.py [--quick]
Full mode writes figures/deep_sea_treasure.png.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import time
from collections import deque

import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
ROWS, COLS = 11, 10
# sea-floor depth per column (rows 0..depth-1 are water, the treasure sits at row depth)
TREASURES = {(1, 0): 1, (2, 1): 2, (3, 2): 3, (4, 3): 5, (4, 4): 8, (4, 5): 16,
             (7, 6): 24, (7, 7): 50, (9, 8): 74, (10, 9): 124}
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]          # up, down, left, right
START = (0, 0)


class DST:
    def __init__(self):
        depth = {c: r for (r, c) in TREASURES}
        self.cells = [(r, c) for r in range(ROWS) for c in range(COLS) if r <= depth[c]]
        self.idx = {rc: i for i, rc in enumerate(self.cells)}
        self.N = len(self.cells)
        self.terminal = np.array([rc in TREASURES for rc in self.cells])
        self.nxt = np.zeros((self.N, 4), dtype=int)
        self.phi = np.zeros((self.N, 4, 2))           # vector reward r(s, a) = (treasure, -1)
        for i, (r, c) in enumerate(self.cells):
            for a, (dr, dc) in enumerate(MOVES):
                j = self.idx.get((r + dr, c + dc), i)
                self.nxt[i, a] = j
                self.phi[i, a] = (TREASURES.get(self.cells[j], 0.0), -1.0)
        self.s0 = self.idx[START]

    def shortest_times(self):
        dist = {self.s0: 0}
        q = deque([self.s0])
        while q:
            s = q.popleft()
            if self.terminal[s]:
                continue
            for a in range(4):
                j = self.nxt[s, a]
                if j not in dist:
                    dist[j] = dist[s] + 1
                    q.append(j)
        return {TREASURES[self.cells[s]]: d for s, d in dist.items() if self.terminal[s]}

    def solve(self, w, tol=1e-10):
        """Value iteration on the scalarized reward w.r (gamma = 1, terminal treasures)."""
        r = self.phi @ w
        v = np.zeros(self.N)
        for _ in range(10_000):
            q = r + np.where(self.terminal[self.nxt], 0.0, v[self.nxt])
            v_new = np.where(self.terminal, 0.0, q.max(axis=1))
            if np.abs(v_new - v).max() < tol:
                break
            v = v_new
        return q, v_new

    def rollout(self, pi, max_steps=200):
        """Vector return of a deterministic policy (array or callable) from the start."""
        s, G = self.s0, np.zeros(2)
        for _ in range(max_steps):
            a = pi(s) if callable(pi) else pi[s]
            G += self.phi[s, a]
            s = self.nxt[s, a]
            if self.terminal[s]:
                return G, True
        return G, False

    def sf(self, pi):
        """Successor features with phi = vector reward: psi(s, a) = phi(s, a) + psi(s', pi(s'))."""
        psi_v = np.zeros((self.N, 2))
        for _ in range(500):
            new = np.where(self.terminal[:, None], 0.0,
                           self.phi[np.arange(self.N), pi]
                           + np.where(self.terminal[self.nxt[np.arange(self.N), pi]][:, None], 0.0,
                                      psi_v[self.nxt[np.arange(self.N), pi]]))
            if np.abs(new - psi_v).max() < 1e-12:
                break
            psi_v = new
        return self.phi + np.where(self.terminal[self.nxt][..., None], 0.0, psi_v[self.nxt])


def weight(w1):
    """w = (w1, 1 - w1); a 1e-6 floor on the time weight breaks ties against wandering at w1 = 1."""
    return np.array([w1, max(1.0 - w1, 1e-6)])


def supported(points):
    """Indices of points on the upper convex hull that are optimal for some w >= 0 (2 objectives)."""
    W = np.linspace(0, 1, 100_001)
    vals = points @ np.stack([W, 1 - W])                # (n_points, n_weights)
    best = vals.max(axis=0)
    return sorted({int(i) for i in np.where(np.isclose(vals, best[None], atol=1e-9))[0]})


def q_learning_sweep(env, weights, episodes, rng, alpha=0.2, eps=0.2, max_steps=100, optimistic=True):
    """Tabular Q-learning on the scalarized reward, all weights in parallel (vectorized).

    Q starts optimistic, at w1 * max treasure (an upper bound on any scalarized return): with zero
    initialization epsilon-greedy settles on the nearest treasures and never finds the far ones
    (the deep-exploration problem of Chapter 14); pass optimistic=False to see it."""
    W = len(weights)
    w = np.stack([weight(w1) for w1 in weights])                       # (W, 2)
    R = np.einsum("sad,wd->wsa", env.phi, w)                            # (W, N, 4)
    Q = np.zeros((W, env.N, 4))
    if optimistic:
        Q += w[:, 0, None, None] * max(TREASURES.values())
    ar = np.arange(W)
    for _ in range(episodes):
        s = np.full(W, env.s0)
        alive = np.ones(W, dtype=bool)
        for _ in range(max_steps):
            greedy = Q[ar, s].argmax(axis=1)
            a = np.where(rng.random(W) < eps, rng.integers(4, size=W), greedy)
            s2 = env.nxt[s, a]
            boot = np.where(env.terminal[s2], 0.0, Q[ar, s2].max(axis=1))
            td = R[ar, s, a] + boot - Q[ar, s, a]
            Q[ar, s, a] += alpha * td * alive
            alive &= ~env.terminal[s2]
            s = np.where(alive, s2, s)
            if not alive.any():
                break
    return Q


INK_COLOR = "#0b0b0b"


def ols_full(env, tol=1e-9):
    """Optimistic Linear Support, two objectives: process a queue of corner weights; at each one
    solve the scalarized MDP and keep the solution if it beats the current CCS there.  (Full OLS
    processes the corners in order of their optimistic improvement bound and can stop early with an
    approximate CCS; with exact solves and two objectives the order does not change the result.)"""
    vals, lib = [], []
    queue, seen = [0.0, 1.0], set()
    solves = 0
    while queue:
        w1 = queue.pop(0)
        if round(w1, 10) in seen:
            continue
        seen.add(round(w1, 10))
        w = np.array([w1, 1 - w1])
        q, _ = env.solve(weight(w1))
        solves += 1
        pi = q.argmax(axis=1)
        V, _ = env.rollout(pi)
        best_known = max((v @ w for v in vals), default=-np.inf)
        if V @ w > best_known + tol and not any(np.allclose(V, v) for v in vals):
            vals.append(V)
            lib.append(pi)
            vs = sorted(vals, key=lambda v: v[0])
            for a, b in zip(vs[:-1], vs[1:]):            # corners between neighbours on the envelope
                den = (a[0] - a[1]) - (b[0] - b[1])
                if abs(den) > 1e-12:
                    c = (b[1] - a[1]) / den
                    if 0 < c < 1 and round(c, 10) not in seen:
                        queue.append(c)
    return lib, vals, solves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figures")
    ap.add_argument("--episodes", type=int, default=3000, help="Q-learning episodes per weight")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)
    episodes = 1500 if args.quick else args.episodes
    n_q = 21 if args.quick else 101
    n_sim = 2_000 if args.quick else 100_000
    print(f"Deep Sea Treasure | {ROWS}x{COLS} grid, gamma = 1, reward = (treasure, -1 per step) | seed=0 "
          f"Q-learning weights={n_q} episodes={episodes} mixture simulations={n_sim} quick={args.quick}")
    env = DST()

    # ---- Part 1: Pareto front --------------------------------------------------------------
    times = env.shortest_times()
    front = np.array(sorted([(v, -t) for v, t in times.items()]), dtype=float)
    dominated = [i for i in range(len(front))
                 if any(np.all(front[j] >= front[i]) and np.any(front[j] > front[i]) for j in range(len(front)))]
    print("[front] (treasure, -time): " + ", ".join(f"({int(v)}, {int(t)})" for v, t in front)
          + f"; Pareto-dominated points: {len(dominated)}")
    sup = supported(front)
    unsup = [i for i in range(len(front)) if i not in sup]
    print(f"[front] supported (on the convex hull; optimal for some linear w): "
          + ", ".join(f"({int(front[i, 0])}, {int(front[i, 1])})" for i in sup)
          + f"\n[front] unsupported ({len(unsup)} of {len(front)}): "
          + ", ".join(f"({int(front[i, 0])}, {int(front[i, 1])})" for i in unsup))

    # ---- Part 2: exact weight sweep and Q-learning sweep ------------------------------------
    grid = np.linspace(0, 1, 1001)
    found_exact = {}
    for w1 in grid:
        q, _ = env.solve(weight(w1))
        G, ok = env.rollout(q.argmax(axis=1))
        found_exact.setdefault((int(G[0]), int(G[1])), []).append(w1)
    print("[sweep] exact scalarized optimum over 1001 weights w = (w1, 1 - w1): outcome -> w1 range")
    for k in sorted(found_exact):
        ws = found_exact[k]
        print(f"        {k}: w1 in [{min(ws):.3f}, {max(ws):.3f}] ({len(ws)} weights)")
    q_weights = np.linspace(0, 1, n_q)
    unsup_set = {(int(front[i, 0]), int(front[i, 1])) for i in unsup}
    print(f"[sweep] unsupported points found by the exact sweep: {sorted(unsup_set & set(found_exact))}")
    for optimistic in (True, False):
        t1 = time.time()
        Q = q_learning_sweep(env, q_weights, episodes, np.random.default_rng(1), optimistic=optimistic)
        found = {}
        n_mismatch = 0
        for k, w1 in enumerate(q_weights):
            G, ok = env.rollout(Q[k].argmax(axis=1))
            key = (int(G[0]), int(G[1])) if ok else ("no treasure", int(G[1]))
            found.setdefault(key, []).append(w1)
            qx, _ = env.solve(weight(w1))
            Gx, _ = env.rollout(qx.argmax(axis=1))
            n_mismatch += not (ok and np.allclose(G, Gx))
        init = "optimistic Q0 = 124 w1" if optimistic else "Q0 = 0 (ablation)"
        print(f"[sweep] Q-learning, {init} ({n_q} weights, {episodes} episodes each, alpha=0.2, eps=0.2; "
              f"{time.time() - t1:.1f} s): greedy outcomes " + ", ".join(
                  f"{k} x{len(v)}" for k, v in sorted(found.items(), key=lambda kv: str(kv[0])))
              + f"; differs from the exact scalarized optimum for {n_mismatch} of {n_q} weights; unsupported "
              f"points reached: {sorted(unsup_set & set(k for k in found if isinstance(k[0], int)))}")
        if optimistic:
            found_q = found

    # ---- Part 3: OLS + SFs + GPI ---------------------------------------------------------------
    lib, vals, solves = ols_full(env)
    print(f"[OLS] {solves} scalarized solves (at w1 = 0, 1 and the corner weights) -> CCS of "
          f"{len(lib)} policies: " + ", ".join(f"({int(v[0])}, {int(v[1])})" for v in sorted(vals, key=lambda v: v[0])))
    psis = [env.sf(pi) for pi in lib]
    err_sf = max(np.abs(psi[env.s0, pi[env.s0]] - env.rollout(pi)[0]).max() for psi, pi in zip(psis, lib))
    worst = 0.0
    for w1 in grid:
        w = weight(w1)
        Qg = np.stack([psi @ w for psi in psis]).max(axis=0)          # GPI: max_i psi_i(s, a).w
        G, ok = env.rollout(Qg.argmax(axis=1))
        _, v_star = env.solve(w)
        worst = max(worst, abs(G @ w - v_star[env.s0]))
    print(f"[GPI] psi(s0, pi(s0)) equals the rolled-out vector return to {err_sf:.1e}; over 1001 weights, "
          f"|w.G(GPI) - V*_w(s0)| <= {worst:.1e}")

    # ---- Part 4: SER vs ESR --------------------------------------------------------------------
    u_target = (24, -13)
    i24 = [i for i in range(len(front)) if tuple(front[i].astype(int)) == u_target][0]
    best_mix = None
    sup_pts = [front[i] for i in sup]
    for a in range(len(sup_pts)):
        for b in range(a + 1, len(sup_pts)):
            for p in np.linspace(0, 1, 1001):
                m = (1 - p) * sup_pts[a] + p * sup_pts[b]
                if np.all(m >= front[i24]):
                    margin = (m - front[i24]).min()
                    if best_mix is None or margin > best_mix[0]:
                        best_mix = (margin, a, b, p, m)
    _, a, b, p, m = best_mix
    pa, pb = sup_pts[a], sup_pts[b]
    lo = (front[i24, 0] - pa[0]) / (pb[0] - pa[0])
    hi = (pa[1] - front[i24, 1]) / (pa[1] - pb[1])
    print(f"[SER] mixing CCS policies ({int(pa[0])}, {int(pa[1])}) and ({int(pb[0])}, {int(pb[1])}) with "
          f"probability p on the second dominates the unsupported ({u_target[0]}, {u_target[1]}) for "
          f"p in [{lo:.3f}, {hi:.3f}]; the most balanced mixture p = {p:.3f} has E[G] = ({m[0]:.2f}, {m[1]:.2f})")
    pol_a = lib[[k for k, v in enumerate(vals) if np.allclose(v, pa)][0]]
    pol_b = lib[[k for k, v in enumerate(vals) if np.allclose(v, pb)][0]]
    Ga, Gb = env.rollout(pol_a)[0], env.rollout(pol_b)[0]
    pick = rng.random(n_sim) < p
    G_sim = np.where(pick[:, None], Gb[None], Ga[None])
    u = lambda G: ((G[..., 0] >= 24) & (G[..., 1] >= -13)).astype(float)
    print(f"[SER] simulated mixture ({n_sim} episodes): mean return ({G_sim[:, 0].mean():.2f}, "
          f"{G_sim[:, 1].mean():.2f}) +- ({G_sim[:, 0].std() / np.sqrt(n_sim):.2f}, "
          f"{G_sim[:, 1].std() / np.sqrt(n_sim):.3f})")
    print(f"[ESR] utility u(G) = 1[treasure >= 24 and time <= 13]: deterministic ({u_target[0]}, {u_target[1]}) "
          f"policy E[u(G)] = {u(front[i24]):.3f}; mixture E[u(G)] = {u(G_sim).mean():.3f} (exact "
          f"{(1 - p) * u(Ga) + p * u(Gb):.3f}); u(E[G]) of the mixture = {u(G_sim.mean(axis=0)):.0f}")

    if not args.quick:
        from plot_style import setup, C, GREY
        plt = setup()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        ax = axes[0]
        hull = front[sup]
        ax.plot(-hull[:, 1], hull[:, 0], color=GREY, lw=1, ls="--", label="convex hull of the front")
        ax.scatter(-front[sup, 1], front[sup, 0], s=60, color=C[0], zorder=3,
                   label="supported (found by linear scalarization)")
        ax.scatter(-front[unsup, 1], front[unsup, 0], s=60, facecolors="none", edgecolors=C[1], lw=1.5,
                   zorder=3, label="unsupported Pareto points")
        qpts = np.array([k for k in found_q if isinstance(k[0], int)], dtype=float)
        ax.scatter(-qpts[:, 1], qpts[:, 0], marker="x", s=70, color=C[2], zorder=4,
                   label=f"Q-learning greedy outcomes ({n_q} weights)")
        ax.scatter([-m[1]], [m[0]], marker="D", s=50, color=C[3], zorder=4,
                   label=f"mixture, p = {p:.2f} (SER)")
        for v, t in front:
            ax.annotate(f"{int(v)}", (-t, v), textcoords="offset points", xytext=(6, -4), fontsize=8)
        ax.set_xlabel("time to treasure (steps)")
        ax.set_ylabel("treasure value")
        ax.set_title("Deep Sea Treasure: Pareto front")
        ax.legend(loc="upper left", fontsize=8)
        ax = axes[1]
        W = np.linspace(0, 0.3, 301)
        for k, v in enumerate(sorted(vals, key=lambda v: v[0])):
            ax.plot(W, v[0] * W + v[1] * (1 - W), color=C[k % 8], lw=1.2, label=f"({int(v[0])}, {int(v[1])})")
        for i in unsup:
            ax.plot(W, front[i, 0] * W + front[i, 1] * (1 - W), color=GREY, lw=0.8, ls=":")
        env_v = np.max([[v[0] * x + v[1] * (1 - x) for x in W] for v in vals], axis=0)
        ax.plot(W, env_v, color=INK_COLOR, lw=2.5, alpha=0.35, label="max over CCS = V*_w(s0)")
        corner = (vals[0][1] - vals[1][1]) / ((vals[1][0] - vals[1][1]) - (vals[0][0] - vals[0][1]))
        ax.axvline(corner, color=GREY, lw=0.8, ls="--")
        ax.text(corner + 0.004, 12, f"corner weight\nw1 = {corner:.3f}", fontsize=8, color=GREY)
        ax.set_xlabel("w1 (weight on treasure; time weight 1 - w1)")
        ax.set_ylabel("scalarized value w . V")
        ax.set_xlim(0, 0.3)
        ax.set_ylim(-20, 25)
        ax.set_title("Scalarized values: CCS lines (solid), unsupported (dotted)")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "deep_sea_treasure.png"))
        plt.close(fig)
        print(f"figures written to {FIG_DIR}")
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
