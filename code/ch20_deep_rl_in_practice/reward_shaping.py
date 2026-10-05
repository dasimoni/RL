"""Potential-based vs naive reward shaping on a zig-zag gridworld (Chapter 20, Section 2).

Grid (S start, G goal +1, P pit -1, # wall); both G and P are terminal:

        col 0 1 2 3 4 5 6
    row 0   . . . . . . G
    row 1   . # # # # # #
    row 2   . . . . . . .
    row 3   P # # # # # .
    row 4   S . . . . . .

The shortest path S -> G has 22 steps; the pit is one step from S.
Original reward: +1 for entering G, -1 for entering P, 0 otherwise; gamma = 0.99.

Part 1 (exact, value iteration) compares the optimal policies of nine rewards:
  R0  original sparse reward
  R1  potential-based shaping (PBRS), Phi = V* = gamma^(d(s)-1)   (d = true shortest-path distance)
  R2  PBRS, Phi = 1 - d(s)/22                     (a "progress" potential anchored near V*)
  R3  PBRS, Phi = -0.1 d(s)                       ("negative distance": the common choice)
  R4  PBRS, Phi = -0.1 d(s) + 20                  (same as R3 plus a constant)
  R5  PBRS, Phi = -0.1 * Manhattan distance       (a misleading heuristic: walls)
  B1  R4's potential, but the code forgets to set Phi(terminal) = 0   (a bug)
  N1  naive "progress" bonus +0.1 whenever d decreases (no penalty for increases)
  N2  naive dense penalty -0.05 * d(s') on every step
and checks Q*_shaped(s,a) = Q*(s,a) - Phi(s) (Eq. 20.4) to machine precision.

Part 2 (learning) trains Q-learning on R0-R4, B1, N1, N2 with 50 seeds and
reports the TRUE (original-reward) performance of the greedy policy.

Part 3 checks Wiewiora's (2003) equivalence: Q-learning with PBRS behaves
exactly like Q-learning on the original reward with Q initialised to Phi(s).

Run:  python code/ch20_deep_rl_in_practice/reward_shaping.py [--quick]
"""
import argparse
import time
from collections import deque

import numpy as np

import rl_stats as st

SEED = 0
GAMMA = 0.99
LAYOUT = ["......G",
          ".######",
          ".......",
          "P#####.",
          "S......"]
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]          # up, right, down, left
ACTION_NAMES = ["up", "right", "down", "left"]


# ---------------------------------------------------------------------------
# The gridworld as exact tables
# ---------------------------------------------------------------------------
class Grid:
    def __init__(self):
        self.rows, self.cols = len(LAYOUT), len(LAYOUT[0])
        self.cells = [(r, c) for r in range(self.rows) for c in range(self.cols) if LAYOUT[r][c] != "#"]
        self.index = {rc: i for i, rc in enumerate(self.cells)}
        self.S, self.A = len(self.cells), 4
        find = lambda ch: self.index[next((r, c) for r in range(self.rows) for c in range(self.cols) if LAYOUT[r][c] == ch)]
        self.start, self.goal, self.pit = find("S"), find("G"), find("P")
        self.terminal = np.zeros(self.S, dtype=bool)
        self.terminal[[self.goal, self.pit]] = True
        self.nxt = np.zeros((self.S, self.A), dtype=np.int64)
        for i, (r, c) in enumerate(self.cells):
            for a, (dr, dc) in enumerate(ACTIONS):
                rc = (r + dr, c + dc)
                self.nxt[i, a] = self.index.get(rc, i)       # walls and borders: stay put
        self.r_env = np.zeros((self.S, self.A))              # original reward of (s, a)
        self.r_env[self.nxt == self.goal] = 1.0
        self.r_env[self.nxt == self.pit] = -1.0
        self.term_sa = self.terminal[self.nxt]               # does (s, a) terminate?
        self.d = self._bfs_distance()                        # true shortest-path distance to G
        self.manhattan = np.array([abs(r - 0) + abs(c - 6) for (r, c) in self.cells], dtype=float)

    def _bfs_distance(self):
        """Shortest-path distance to the goal. Paths may not pass THROUGH the pit (it is
        terminal), but the pit itself gets a distance (9: one step from cell (2, 0))."""
        dist = np.full(self.S, np.inf)
        dist[self.goal] = 0
        q = deque([self.goal])
        while q:
            s = q.popleft()
            if s == self.pit:
                continue                                     # cannot continue a path from a terminal cell
            for s_prev in range(self.S):
                if np.any(self.nxt[s_prev] == s) and dist[s_prev] == np.inf and s_prev != s:
                    dist[s_prev] = dist[s] + 1
                    q.append(s_prev)
        return dist


# ---------------------------------------------------------------------------
# Reward variants. Each returns the shaped reward table R'(s, a) (deterministic env).
# ---------------------------------------------------------------------------
def pbrs_table(g, phi, zero_terminal=True):
    """F(s,a,s') = gamma * Phi(s') - Phi(s), with Phi(terminal) = 0 unless the bug is on (Eq. 20.3)."""
    phi_next = phi[g.nxt].copy()
    if zero_terminal:
        phi_next[g.term_sa] = 0.0                     # absorbing terminal state has potential 0
    return g.r_env + GAMMA * phi_next - phi[:, None]


def potentials(g):
    return {
        "R1 PBRS, Phi = V* (ideal)": np.where(g.terminal, 0.0, GAMMA ** (g.d - 1)),
        "R2 PBRS, Phi = 1 - d/22": 1.0 - g.d / g.d[g.start],
        "R3 PBRS, Phi = -0.1 d": -0.1 * g.d,
        "R4 PBRS, Phi = -0.1 d + 20": -0.1 * g.d + 20.0,
        "R5 PBRS, Phi = -0.1 Manhattan": -0.1 * g.manhattan,
    }


def reward_variants(g):
    """Name -> (shaped reward table R'(s,a), potential or None)."""
    progress = 0.1 * (g.d[g.nxt] < g.d[:, None])           # +0.1 if the move reduces d
    out = {"R0 original (sparse)": (g.r_env.copy(), None)}
    for name, phi in potentials(g).items():
        out[name] = (pbrs_table(g, phi), phi)
    out["B1 bug: R4 with Phi(terminal) not zeroed"] = (pbrs_table(g, -0.1 * g.d + 20.0, zero_terminal=False), None)
    out["N1 naive progress bonus +0.1"] = (g.r_env + progress, None)
    out["N2 naive penalty -0.05 d(s')"] = (g.r_env - 0.05 * np.where(g.term_sa, 0.0, g.d[g.nxt]), None)
    return out


def value_iteration(g, R, tol=1e-13):
    Q = np.zeros((g.S, g.A))
    for it in range(100_000):
        V = Q.max(1)
        V[g.terminal] = 0.0                          # terminal states are absorbing with value 0
        Qn = R + GAMMA * (~g.term_sa) * V[g.nxt]
        if np.abs(Qn - Q).max() < tol:
            return Qn
        Q = Qn
    raise RuntimeError("value iteration did not converge")


def optimal_sets(Q, tol=1e-9):
    return Q >= Q.max(1, keepdims=True) - tol


def greedy_outcome(g, Q, max_steps=200):
    s, path = g.start, [g.start]
    for t in range(1, max_steps + 1):
        s = g.nxt[s, int(np.argmax(Q[s]))]
        path.append(s)
        if s == g.goal:
            return f"reaches G in {t} steps", path
        if s == g.pit:
            return f"enters the pit at step {t}", path
    cyc = sorted(set(path[-6:]))
    return f"never terminates (cycles among cells {[g.cells[c] for c in cyc]})", path


# ---------------------------------------------------------------------------
# Vectorized Q-learning over seeds (Algorithm 20.1 with shaping)
# ---------------------------------------------------------------------------
def q_learning(g, R, n_seeds, steps, seed, alpha=0.5, eps=0.1, horizon=100, eval_every=500, Q0=None,
               return_actions=False):
    rng = np.random.default_rng(seed)
    N = n_seeds
    Q = np.zeros((N, g.S, g.A)) if Q0 is None else np.broadcast_to(Q0, (N, g.S, g.A)).copy()
    rows = np.arange(N)
    s = np.full(N, g.start)
    t_ep = np.zeros(N, dtype=np.int64)
    evals, actions_log = [], []
    for t in range(1, steps + 1):
        q = Q[rows, s]
        best = q >= q.max(1, keepdims=True) - 1e-9           # tolerant ties (needed for Part 3)
        a = np.argmax(rng.random(q.shape) * best, axis=1)
        explore = rng.random(N) < eps
        a[explore] = rng.integers(0, g.A, size=explore.sum())
        s2 = g.nxt[s, a]
        r = R[s, a]
        term = g.term_sa[s, a]
        t_ep += 1
        trunc = (t_ep >= horizon) & ~term
        # bootstrap through truncation, never through termination
        target = r + GAMMA * (~term) * Q[rows, s2].max(1)
        Q[rows, s, a] += alpha * (target - Q[rows, s, a])
        if return_actions:
            actions_log.append(a.copy())
        done = term | trunc
        s = np.where(done, g.start, s2)
        t_ep[done] = 0
        if t % eval_every == 0:
            evals.append(evaluate_greedy(g, Q, R, horizon))
    if return_actions:
        return Q, np.array(actions_log)
    return Q, evals


def evaluate_greedy(g, Q, R, horizon):
    """Deterministic greedy rollout from S: (success, true discounted return, shaped discounted return)."""
    N = Q.shape[0]
    rows = np.arange(N)
    s = np.full(N, g.start)
    alive = np.ones(N, dtype=bool)
    true_ret, shaped_ret, success = np.zeros(N), np.zeros(N), np.zeros(N, dtype=bool)
    for t in range(horizon):
        a = Q[rows, s].argmax(1)
        s2 = g.nxt[s, a]
        disc = GAMMA ** t
        true_ret += alive * disc * g.r_env[s, a]
        shaped_ret += alive * disc * R[s, a]
        success |= alive & (s2 == g.goal)
        alive &= ~g.term_sa[s, a]
        s = s2
    return success, true_ret, shaped_ret


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_seeds, steps, eval_every = (10, 4000, 500) if args.quick else (50, 150_000, 2500)
    print(f"seed={SEED} gamma={GAMMA}; Q-learning: alpha=0.5, eps=0.1, zero init, time limit 100 steps, "
          f"{n_seeds} seeds x {steps} steps, greedy policy evaluated every {eval_every} steps; quick={args.quick}")
    t0 = time.time()
    g = Grid()
    print(f"grid: {g.S} free cells, start {g.cells[g.start]}, goal {g.cells[g.goal]}, pit {g.cells[g.pit]}, "
          f"shortest path {int(g.d[g.start])} steps")

    # ---------------- Part 1: exact analysis ----------------
    print("\nPart 1: optimal policies of the shaped MDPs (value iteration)")
    variants = reward_variants(g)
    Q0 = value_iteration(g, variants["R0 original (sparse)"][0])
    nonterm = ~g.terminal
    exact = {}
    for name, (R, phi) in variants.items():
        Q = value_iteration(g, R)
        differs = (optimal_sets(Q) != optimal_sets(Q0))[nonterm].any(1).sum()
        outcome, _ = greedy_outcome(g, Q)
        line = f"  {name:<42s} states with a different optimal action set: {differs:2d}/{nonterm.sum()}; greedy from S {outcome}"
        if phi is not None:
            err = np.abs(Q - (Q0 - phi[:, None]))[nonterm].max()
            line += f"; max|Q* - (Q*_0 - Phi)| = {err:.1e}"
        print(line)
        exact[name] = (Q, differs, outcome)
    v_loop = 0.1 * GAMMA / (1 - GAMMA ** 2)
    print(f"  (N1 hand check: value of the two-step loop 'away, back' = 0.1*gamma/(1-gamma^2) = {v_loop:.3f}"
          f" > 1.1, the best a goal-entering step can earn)")

    # ---------------- Part 2: learning ----------------
    print("\nPart 2: Q-learning, true (original-reward) performance of the greedy policy")
    learn_names = ["R0 original (sparse)", "R1 PBRS, Phi = V* (ideal)", "R2 PBRS, Phi = 1 - d/22",
                   "R3 PBRS, Phi = -0.1 d", "R4 PBRS, Phi = -0.1 d + 20",
                   "B1 bug: R4 with Phi(terminal) not zeroed", "N1 naive progress bonus +0.1",
                   "N2 naive penalty -0.05 d(s')"]
    tables = {k: v[0] for k, v in variants.items()}
    curves = {}
    rng_boot = np.random.default_rng(SEED)
    for k, name in enumerate(learn_names):
        _, evals = q_learning(g, tables[name], n_seeds, steps, seed=SEED + k, eval_every=eval_every)
        succ = np.array([e[0] for e in evals])                 # (n_evals, seeds)
        true_ret = np.array([e[1] for e in evals])
        shaped_ret = np.array([e[2] for e in evals])
        curves[name] = (succ, true_ret, shaped_ret)
        ok = succ.mean(1) >= 0.9
        first = (np.argmax(ok) + 1) * eval_every if ok.any() else None
        fin = true_ret[-1][:, None]
        p, lo, hi = st.stratified_bootstrap_ci(fin, st.agg_iqm, reps=2000, rng=rng_boot)
        marks = [len(succ) // 6, len(succ) // 3, 2 * len(succ) // 3, len(succ)]      # 1/6, 1/3, 2/3, all of the budget
        at = ", ".join(f"{m * eval_every // 1000}k: {succ[m - 1].mean():.2f}" for m in marks)
        print(f"  {name:<42s} success after {at};"
              f" final true return IQM {p:+.3f} [{lo:+.3f}, {hi:+.3f}] (optimal {GAMMA ** 21:.3f});"
              f" >= 90% of seeds succeed from " + (f"{first:,d} steps" if first is not None else "never"))
    # reward hacking signature for N1: shaped return up, true return down
    sh = curves["N1 naive progress bonus +0.1"]
    print(f"  N1 greedy-policy returns at the first / last evaluation: shaped {sh[2][0].mean():.2f} -> {sh[2][-1].mean():.2f},"
          f" true {sh[1][0].mean():+.3f} -> {sh[1][-1].mean():+.3f}")

    # ---------------- Part 3: Wiewiora's equivalence ----------------
    phi = -0.1 * g.d
    phi_tab = np.where(g.terminal, 0.0, phi)
    n3, steps3 = (5, 2000) if args.quick else (20, 20_000)
    Qs, acts_s = q_learning(g, tables["R3 PBRS, Phi = -0.1 d"], n3, steps3, seed=123, return_actions=True)
    Qi, acts_i = q_learning(g, tables["R0 original (sparse)"], n3, steps3, seed=123,
                            Q0=np.repeat(phi_tab[:, None], g.A, axis=1), return_actions=True)
    gap = np.abs((Qs + phi_tab[None, :, None]) - Qi)[:, nonterm].max()
    print(f"\nPart 3: PBRS vs Q-initialization with Phi ({n3} seeds x {steps3} steps, same random numbers):"
          f" identical actions at every step: {bool(np.all(acts_s == acts_i))};"
          f" max |Q_shaped + Phi - Q_init| = {gap:.1e}")

    if not args.quick:
        make_figure(g, exact, curves, learn_names, eval_every)
    print(f"Total time {time.time() - t0:.1f} s")


def make_figure(g, exact, curves, learn_names, eval_every):
    import plot_style
    plt = plot_style.setup()
    C = plot_style.C
    fig_dir = __file__.rsplit("/", 1)[0] + "/figures/"
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [0.95, 1.2, 1.2]})

    # Panel 1: the grid with greedy paths of the optimal policies
    ax = axes[0]
    img = np.ones((g.rows, g.cols, 3))
    for r in range(g.rows):
        for c in range(g.cols):
            if LAYOUT[r][c] == "#":
                img[r, c] = (0.35, 0.35, 0.35)
    ax.imshow(img, origin="upper")
    for ch, col in (("S", plot_style.INK), ("G", C[2]), ("P", C[7])):
        r, c = next((r, c) for r in range(g.rows) for c in range(g.cols) if LAYOUT[r][c] == ch)
        ax.text(c, r, ch, ha="center", va="center", fontsize=14, fontweight="bold", color=col)
    styles = [("R0 original (sparse)", C[0], "-", "R0-R5 optimal: G in 22 steps"),
              ("N1 naive progress bonus +0.1", C[1], "--", "N1: dithers next to G forever"),
              ("N2 naive penalty -0.05 d(s')", C[7], ":", "N2: into the pit"),
              ("B1 bug: R4 with Phi(terminal) not zeroed", C[6], "-.", "B1 (bug): into the pit")]
    for k, (name, col, ls, lab) in enumerate(styles):
        _, path = greedy_outcome(g, exact[name][0], max_steps=40)
        pts = np.array([g.cells[s] for s in path], dtype=float)
        off = (k - 1.5) * 0.08
        ax.plot(pts[:, 1] + off, pts[:, 0] + off, color=col, ls=ls, lw=2, label=lab)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("Greedy paths of the optimal policies")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=8, ncol=2)

    x = np.arange(1, len(curves[learn_names[0]][0]) + 1) * eval_every
    # Panel 2: potential-based variants (same optimal policy)
    ax = axes[1]
    pb = learn_names[:5]
    for k, name in enumerate(pb):
        ax.plot(x, curves[name][0].mean(1), color=C[k], ls=["-", "--", "-.", ":", "-"][k],
                lw=2.0 if k != 4 else 1.4, label=name,
                marker="o" if k == 2 else None, ms=4, markevery=4)       # R2 marked: it coincides with R1
    ax.annotate("R1 and R2 coincide: 1.0 from\nthe first evaluation (2.5k steps)", xy=(x[6], 1.0),
                xytext=(x[6], 0.72), fontsize=8, arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax.set_xlabel("environment steps")
    ax.set_ylabel("fraction of seeds whose greedy policy reaches G")
    ax.set_title("Potential-based shaping: same optimum, very different speed")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8, loc="center right")
    plot_style.kfmt(ax)

    # Panel 3: non-potential shaping and the terminal bug
    ax = axes[2]
    bad = [learn_names[0], learn_names[5], learn_names[6], learn_names[7]]
    cols = [C[0], C[6], C[1], C[7]]
    for k, name in enumerate(bad):
        ax.plot(x, curves[name][1].mean(1), color=cols[k], ls=["-", "-.", "--", ":"][k], label=name,
                marker="s" if k == 3 else None, ms=4, markevery=5)       # N2 marked: it coincides with B1
    ax.axhline(GAMMA ** 21, color=plot_style.GREY, ls="--", lw=1)
    ax.text(x[-1], GAMMA ** 21 + 0.03, "optimal", ha="right", fontsize=8, color=plot_style.GREY)
    ax.text(x[len(x) // 2], -0.93, "B1 and N2 coincide at -1 (into the pit)", ha="center", fontsize=8)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("true discounted return of the greedy policy (mean)")
    ax.set_title("Non-potential shaping and the terminal bug: hacking")
    ax.set_ylim(-1.1, 1.0)
    ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.12))
    plot_style.kfmt(ax)
    # Inset: the signature of reward hacking for N1 -- shaped return up, true return flat at 0
    ins = ax.inset_axes([0.07, 0.61, 0.29, 0.23])           # empty area above R0's early, flat stretch
    n1 = curves["N1 naive progress bonus +0.1"]
    ins.plot(x, n1[2].mean(1), color=C[1], ls="--", label="shaped return")
    ins.plot(x, n1[1].mean(1), color=plot_style.INK, ls="-", label="true return")
    ins.set_title("N1, greedy policy", fontsize=8)
    ins.tick_params(labelsize=7)
    ins.set_xticks([])
    ins.set_ylim(-0.3, 4.6)
    ins.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(fig_dir + "reward_shaping.png")
    plt.close(fig)
    print("Figure written to", fig_dir + "reward_shaping.png")


if __name__ == "__main__":
    main()
