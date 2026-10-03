"""Robust value iteration on a slippery gridworld (Chapter 20, Section 11.6).

Grid (S start, G goal +1, L lava -1, # wall); G and L are terminal:

        col 0 1 2 3 4 5
    row 0   . . . . . .
    row 1   . # # # # .
    row 2   . # # # # .
    row 3   S . . . . G
    row 4   # L L L L #

Slip model with slip probability xi: the intended move happens with probability
1 - xi, and with probability xi/2 each the agent moves in one of the two
perpendicular directions. Moving into a wall or the border leaves the agent in
place. The reward is +1 for entering G and -1 for entering L; gamma = 0.9.
The short route along row 3 (5 steps) passes four lava cells: each of its last
four moves to the right falls into the lava with probability xi/2 (below S there
is a wall). The detour over row 0 (11 steps) is exposed to the lava only if its
first move up from S slips to the right, onto (3, 1); otherwise slips only delay it.

The nominal model is xi_hat = 0.05. Around each nominal row p_hat(.|s,a) we put
an (s,a)-rectangular uncertainty set, restricted to the support of p_hat:
  L1 ball:  { p : ||p - p_hat||_1 <= kappa }   (inner problem solved by sorting)
  KL ball:  { p : KL(p || p_hat) <= kappa }     (inner problem solved by exponential
                                                 tilting; checked against its dual)

Part 1: nominal value iteration vs robust value iteration for several radii;
        contraction factor of the robust operator; route; robust value.
Part 2: checks of the inner solvers (L1 sort vs a linear program; KL primal vs dual).
Part 3: exact evaluation of every policy for true slip probabilities xi in [0, 0.5];
        which xi each set covers; the robust value as a certified lower bound;
        where the gap between the robust value and the worst slip model comes from
        (a different symmetric slip in every state; a single slip model that drifts
        to one side of the heading); the cost of robustness at xi_hat and its benefit
        at large xi.

Run:  python code/ch20_deep_rl_in_practice/robust_vi.py [--quick]
"""
import argparse
import time

import numpy as np
from scipy.optimize import linprog, minimize_scalar

SEED = 0
GAMMA = 0.9
XI_HAT = 0.05
LAYOUT = ["......",
          ".####.",
          ".####.",
          "S....G",
          "#LLLL#"]
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]          # up, right, down, left
ARROWS = ["^", ">", "v", "<"]
L1_RADII = [0.02, 0.05, 0.1, 0.2, 0.3]
KL_RADII = [0.01, 0.02, 0.05, 0.0703, 0.1]
# The two robust policies of the figure. Both balls contain every slip model with xi in [0, 0.15]:
# the L1 distance of xi from xi_hat is at most 2|xi - xi_hat| = 0.2, and the KL divergence is at most
# KL(Bernoulli(0.15) || Bernoulli(0.05)) = 0.07025 (Part 3 checks the coverage on a grid of xi).
L1_SHOW, KL_SHOW = 0.2, 0.0703


# ---------------------------------------------------------------------------
# The gridworld as exact tables
# ---------------------------------------------------------------------------
class Grid:
    def __init__(self):
        R, C = len(LAYOUT), len(LAYOUT[0])
        self.cells = [(r, c) for r in range(R) for c in range(C) if LAYOUT[r][c] != "#"]
        idx = {rc: i for i, rc in enumerate(self.cells)}
        self.S, self.A = len(self.cells), 4
        kind = [LAYOUT[r][c] for r, c in self.cells]
        self.start = kind.index("S")
        self.terminal = np.array([k in "GL" for k in kind])
        self.nonterm = ~self.terminal
        self.r_in = np.array([1.0 if k == "G" else -1.0 if k == "L" else 0.0 for k in kind])  # reward for entering
        self.move = np.array([[idx.get((r + dr, c + dc), i) for (dr, dc) in ACTIONS]
                              for i, (r, c) in enumerate(self.cells)])

    def P(self, xi):
        """Transition tensor P[s, a, s'] for slip probability xi (terminal states absorb)."""
        P = np.zeros((self.S, self.A, self.S))
        for s in range(self.S):
            for a in range(self.A):
                if self.terminal[s]:
                    P[s, a, s] = 1.0
                    continue
                P[s, a, self.move[s, a]] += 1 - xi
                for d in ((a + 1) % 4, (a + 3) % 4):          # the two perpendicular directions
                    P[s, a, self.move[s, d]] += xi / 2
        return P

    def P_sides(self, left, right):
        """Slip model that may favour one side of the heading: the intended move with probability
        1 - left - right, a slip to the left of the heading with probability `left`, to the right
        with probability `right`. P_sides(xi/2, xi/2) is P(xi)."""
        P = np.zeros((self.S, self.A, self.S))
        rows = np.arange(self.S)
        for a in range(self.A):
            for d, w in ((a, 1 - left - right), ((a + 1) % 4, right), ((a + 3) % 4, left)):
                np.add.at(P[:, a, :], (rows, self.move[:, d]), w)
        t = np.flatnonzero(self.terminal)
        P[t] = 0.0
        P[t, :, t] = 1.0
        return P

    def landing_value(self, v):
        """u(s') = reward for entering s' + gamma * v(s') (zero continuation after termination)."""
        return self.r_in + GAMMA * v * self.nonterm


# ---------------------------------------------------------------------------
# Inner problems: min_{p in U} p . u for every row of a nominal kernel
# ---------------------------------------------------------------------------
def l1_inner(Pn, u, kappa):
    """Worst case over {p : ||p - p_hat||_1 <= kappa, supp p within supp p_hat}.

    Move kappa/2 of probability mass to the worst successor, taking it from the
    best successors first (sort by u). Returns (values, worst-case rows)."""
    rows = Pn.reshape(-1, Pn.shape[-1]).copy()
    for row in rows:
        sup = np.flatnonzero(row > 0)
        if len(sup) < 2:
            continue
        order = sup[np.argsort(u[sup], kind="stable")]
        worst = order[0]
        add = min(kappa / 2, 1.0 - row[worst])
        row[worst] += add
        rest = add
        for j in order[::-1]:                              # best successor first
            if j == worst or rest <= 0:
                continue
            take = min(rest, row[j])
            row[j] -= take
            rest -= take
    rows = rows.reshape(Pn.shape)
    return rows @ u, rows


def kl_inner(Pn, u, kappa, iters=200):
    """Worst case over {p : KL(p || p_hat) <= kappa}. The minimizer is the exponential tilt
    q_eta(s') proportional to p_hat(s') exp(-u(s')/eta), with eta > 0 chosen so that the
    KL equals kappa (bisection on log eta, vectorized over rows). If kappa >= -log p_hat(argmin u),
    the minimizer is p_hat restricted to the argmin set and the value is min u."""
    shape = Pn.shape[:-1]
    P = Pn.reshape(-1, Pn.shape[-1])
    sup = P > 0
    big = np.where(sup, u[None, :], np.inf)
    m = big.min(1, keepdims=True)                          # min of u over each row's support
    span = np.where(sup, u[None, :], -np.inf).max(1, keepdims=True) - m
    at_min = sup & (u[None, :] <= m + 1e-12)
    p_min = np.where(at_min, P, 0).sum(1)
    with np.errstate(divide="ignore"):
        kl_limit = -np.log(p_min)                          # KL of the tilt as eta -> 0
    shifted = np.where(sup, u[None, :] - m, 0.0)

    def tilt(eta):
        w = np.where(sup, P * np.exp(-shifted / eta[:, None]), 0.0)
        q = w / w.sum(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl = np.where(q > 0, q * np.log(q / np.where(sup, P, 1.0)), 0.0).sum(1)
        return q, kl

    scale = np.maximum(span[:, 0], 1e-12)
    lo, hi = np.log(scale * 1e-6), np.log(scale * 1e6)    # KL(eta) decreases in eta
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        _, kl = tilt(np.exp(mid))
        too_far = kl > kappa                               # eta too small: tilt too strong
        lo = np.where(too_far, mid, lo)
        hi = np.where(too_far, hi, mid)
    q, _ = tilt(np.exp(hi))
    corner = (kl_limit <= kappa) | (span[:, 0] <= 1e-12)
    q_corner = np.where(at_min, P, 0) / np.maximum(p_min, 1e-300)[:, None]
    q = np.where(corner[:, None], q_corner, q)
    return (q @ u).reshape(shape), q.reshape(Pn.shape)


def kl_dual(p_hat, u, kappa):
    """sup_{eta>0} { -eta log E_{p_hat}[exp(-u/eta)] - eta kappa }  (Exercise 20.17)."""
    sup = p_hat > 0
    pp, uu = p_hat[sup], u[sup]
    m = uu.min()

    def neg_g(log_eta):
        eta = np.exp(log_eta)
        return -(m - eta * np.log(np.sum(pp * np.exp(-(uu - m) / eta))) - eta * kappa)

    res = minimize_scalar(neg_g, bounds=(np.log(1e-8), np.log(1e4)), method="bounded",
                          options={"xatol": 1e-12})
    return max(-res.fun, m)                                # eta -> 0 gives min u


INNER = {"L1": l1_inner, "KL": kl_inner}


# ---------------------------------------------------------------------------
# Value iteration: nominal, robust, and robust policy evaluation
# ---------------------------------------------------------------------------
def nominal_vi(g, P, tol=1e-12):
    v = np.zeros(g.S)
    while True:
        q = P @ g.landing_value(v)
        q[g.terminal] = 0.0
        v_new = q.max(1)
        if np.abs(v_new - v).max() < tol:
            return q, v_new
        v = v_new


def robust_vi(g, Pn, kind, kappa, tol=1e-10, policy=None):
    """Robust VI with the (s,a)-rectangular set `kind` of radius kappa around Pn.
    With `policy` given, robust policy evaluation of that fixed policy instead.
    Returns q, v, the list of sup-norm changes per iteration."""
    v, deltas = np.zeros(g.S), []
    inner = INNER[kind]
    while True:
        u = g.landing_value(v)
        if policy is None:
            q, _ = inner(Pn, u, kappa)
            q[g.terminal] = 0.0
            v_new = q.max(1)
        else:
            q = None
            v_new, _ = inner(Pn[np.arange(g.S), policy], u, kappa)
            v_new = v_new * g.nonterm
        deltas.append(np.abs(v_new - v).max())
        v = v_new
        if deltas[-1] < tol:
            return q, v, deltas


def evaluate(g, P, pi):
    """Exact value of deterministic policy pi under kernel P (linear solve)."""
    Ppi = P[np.arange(g.S), pi]
    r = (Ppi @ g.r_in) * g.nonterm
    M = np.eye(g.S) - GAMMA * Ppi * g.nonterm[None, :] * g.nonterm[:, None]
    return np.linalg.solve(M, r)


def greedy(q, g):
    """Greedy policy with ties broken toward the lowest action index, after rounding."""
    return np.round(q, 10).argmax(1)


def route(g, pi):
    """Cells visited from S when every move succeeds (the policy's intended path)."""
    s, path, seen = g.start, [], set()
    while not g.terminal[s] and s not in seen:
        seen.add(s)
        path.append(s)
        s = g.move[s, pi[s]]
    path.append(s)
    return path


def route_name(g, pi):
    path = route(g, pi)
    r_end, c_end = g.cells[path[-1]]
    if LAYOUT[r_end][c_end] == "G":
        return "detour (row 0)" if 0 in {g.cells[s][0] for s in path} else "short (row 3)"
    return "does not reach G"


def row_dist(g, Pt, Pn, kind):
    """Largest L1 distance or KL divergence KL(Pt || Pn), over non-terminal (s, a), between the rows
    of a kernel Pt and of the nominal kernel Pn."""
    Pt, Ph = Pt[g.nonterm], Pn[g.nonterm]
    if kind == "L1":
        return np.abs(Pt - Ph).sum(-1).max()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(Pt > 0, Pt * np.log(Pt / Ph), 0.0).sum(-1).max()


def coverage(g, Pn, xis, kind, kappa):
    """Which true slip probabilities xi give a kernel inside the rectangular set (every row)?"""
    return np.array([row_dist(g, g.P(xi), Pn, kind) <= kappa + 1e-12 for xi in xis])


def worst_rect_slip(g, pi, lo, hi, tol=1e-12):
    """Worst-case value of policy pi when the adversary may choose a different *symmetric* slip
    probability xi(s) in [lo, hi] at every state (a rectangular set of slip models). The expected
    landing value is linear in xi, so the inner minimum is attained at lo or at hi."""
    P_lo, P_hi = g.P(lo)[np.arange(g.S), pi], g.P(hi)[np.arange(g.S), pi]
    v = np.zeros(g.S)
    while True:
        u = g.landing_value(v)
        v_new = np.minimum(P_lo @ u, P_hi @ u) * g.nonterm
        if np.abs(v_new - v).max() < tol:
            return v_new
        v = v_new


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    n_checks = 200 if args.quick else 2000
    xis = np.round(np.linspace(0, 0.5, 21 if args.quick else 101), 4)   # both grids contain xi_hat, 0.15 and 0.3
    side_grid = np.round(np.linspace(0, 0.2, 21 if args.quick else 81), 6)  # one-sided slip models (Part 3)
    print(f"seed={SEED} gamma={GAMMA} nominal slip xi_hat={XI_HAT}; L1 radii {L1_RADII}; KL radii {KL_RADII}; "
          f"{n_checks} random inner-problem checks; true-xi grid of {len(xis)} points in [0, 0.5]; quick={args.quick}")
    t0 = time.time()
    g = Grid()
    Pn = g.P(XI_HAT)

    # ---------------- Part 1 ----------------
    print("\nPart 1: nominal vs robust value iteration (value from S; route = intended path)")
    qn, vn = nominal_vi(g, Pn)
    pi_nom = greedy(qn, g)
    print(f"  nominal VI at xi_hat:                route {route_name(g, pi_nom):<15s} v(S) = {vn[g.start]:.4f}")
    policies = {"nominal": pi_nom}
    robust_values = {}
    for kind, radii in (("L1", L1_RADII), ("KL", KL_RADII)):
        for kappa in radii:
            q, v, deltas = robust_vi(g, Pn, kind, kappa)
            pi = greedy(q, g)
            ratios = np.array(deltas[1:]) / np.maximum(np.array(deltas[:-1]), 1e-300)
            ratios = ratios[np.array(deltas[:-1]) > 1e-8]
            v_nom_model = evaluate(g, Pn, pi)[g.start]
            _, v_nom_rob, _ = robust_vi(g, Pn, kind, kappa, policy=pi_nom)
            print(f"  robust VI, {kind} ball kappa={kappa:<6g}  route {route_name(g, pi):<15s} robust v(S) = {v[g.start]:7.4f}"
                  f"  its value at xi_hat {v_nom_model:.4f}; nominal policy's robust value {v_nom_rob[g.start]:7.4f}"
                  f"  ({len(deltas)} iterations, max contraction ratio {ratios.max():.4f} <= gamma)")
            policies[f"{kind} {kappa:g}"] = pi
            robust_values[(kind, kappa)] = v[g.start]
    # A worst-case row, to show what the adversary does on the short route
    s_edge = g.cells.index((3, 2))
    _, vr, _ = robust_vi(g, Pn, "L1", L1_SHOW)
    _, rows_l1 = l1_inner(Pn[s_edge, 1][None], g.landing_value(vr), L1_SHOW)
    _, vk, _ = robust_vi(g, Pn, "KL", KL_SHOW)
    _, rows_kl = kl_inner(Pn[s_edge, 1][None], g.landing_value(vk), KL_SHOW)
    lava = g.cells.index((4, 2))
    print(f"  moving right from (3, 2): nominal P(lava) = {Pn[s_edge, 1, lava]:.3f}; worst case in the L1 ball "
          f"(kappa={L1_SHOW}) {rows_l1[0, lava]:.3f}; in the KL ball (kappa={KL_SHOW}) {rows_kl[0, lava]:.3f}")

    # ---------------- Part 2 ----------------
    print(f"\nPart 2: inner-problem checks on {n_checks} random instances (5 successors, random support)")
    err_l1 = err_kl = 0.0
    for _ in range(n_checks):
        n = 5
        p = rng.dirichlet(np.ones(n)) * (rng.random(n) < 0.8)
        if p.sum() == 0:
            p[0] = 1.0
        p /= p.sum()
        u = rng.normal(size=n)
        kappa = rng.uniform(0.0, 1.0)
        val_sort, _ = l1_inner(p[None], u, kappa)
        sup = np.flatnonzero(p > 0)
        k = len(sup)                                        # LP over (q, t) with |q - p| <= t, sum t <= kappa
        c = np.concatenate([u[sup], np.zeros(k)])
        A_ub = np.block([[np.eye(k), -np.eye(k)], [-np.eye(k), -np.eye(k)], [np.zeros((1, k)), np.ones((1, k))]])
        b_ub = np.concatenate([p[sup], -p[sup], [kappa]])
        A_eq = np.concatenate([np.ones(k), np.zeros(k)])[None]
        lp = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * (2 * k), method="highs")
        err_l1 = max(err_l1, abs(val_sort[0] - lp.fun))
        kappa_kl = rng.uniform(0.0, 0.5)
        val_tilt, q = kl_inner(p[None], u, kappa_kl)
        err_kl = max(err_kl, abs(val_tilt[0] - kl_dual(p, u, kappa_kl)))
    print(f"  L1 ball: sorting solution vs linear program, max |difference| = {err_l1:.1e}")
    print(f"  KL ball: exponential-tilting primal vs dual sup_eta {{-eta log E[exp(-u/eta)] - eta kappa}}, "
          f"max |difference| = {err_kl:.1e}")

    # ---------------- Part 3 ----------------
    print("\nPart 3: exact value from S of each policy when the true slip probability is xi")
    show = {"nominal": pi_nom, f"L1 {L1_SHOW:g}": policies[f"L1 {L1_SHOW:g}"],
            f"KL {KL_SHOW:g}": policies[f"KL {KL_SHOW:g}"]}
    curves = {name: np.array([evaluate(g, g.P(xi), pi)[g.start] for xi in xis]) for name, pi in show.items()}
    curves["oracle"] = np.array([nominal_vi(g, g.P(xi))[1][g.start] for xi in xis])
    print("    xi    " + "".join(f"{n:>12s}" for n in curves))
    for i, xi in enumerate(xis):
        if args.quick or np.isclose(xi * 100 % 5, 0) or np.isclose(xi * 100 % 5, 5):
            print(f"  {xi:6.3f}  " + "".join(f"{curves[n][i]:12.4f}" for n in curves))
    diff = curves[f"L1 {L1_SHOW:g}"] - curves["nominal"]
    cross = xis[np.argmax(diff > 0)] if (diff > 0).any() else None
    print(f"  the robust (detour) policy beats the nominal policy for xi >= {cross} on this grid")
    # oracle switch point
    routes = [route_name(g, greedy(nominal_vi(g, g.P(xi))[0], g)) for xi in xis]
    sw = next((xis[i] for i in range(1, len(xis)) if routes[i] != routes[i - 1]), None)
    print(f"  the optimal policy for the true xi switches from the short route to the detour at xi = {sw}")
    for kind, kappa in (("L1", L1_SHOW), ("KL", KL_SHOW)):
        cov = coverage(g, Pn, xis, kind, kappa)
        name = f"{kind} {kappa:g}"
        lo, hi = xis[cov].min(), xis[cov].max()
        worst = curves[name][cov].min()
        print(f"  {kind} ball kappa={kappa:g}: contains every slip model with xi in [{lo:.3f}, {hi:.3f}]; "
              f"robust value {robust_values[(kind, kappa)]:.4f} <= worst true value over that range {worst:.4f} "
              f"(gap {worst - robust_values[(kind, kappa)]:.4f}; nominal policy's worst over the range "
              f"{curves['nominal'][cov].min():.4f})")
        # Where does the gap come from? (i) A different symmetric slip in every state, xi(s) in [lo, hi]:
        rect = worst_rect_slip(g, policies[name], lo, hi)[g.start]
        # (ii) One slip model for all states that may drift to one side of the heading, inside the ball:
        worst_side, arg_side = np.inf, None
        for left in side_grid:
            for right in side_grid:
                Pt = g.P_sides(left, right)
                if row_dist(g, Pt, Pn, kind) <= kappa + 1e-12:
                    val = evaluate(g, Pt, policies[name])[g.start]
                    if val < worst_side:
                        worst_side, arg_side = val, (left, right)
        print(f"    gap analysis for this policy: a different symmetric slip xi(s) in [{lo:.3f}, {hi:.3f}] in every "
              f"state: worst value {rect:.4f}; one slip model drifting to one side of the heading (inside the "
              f"ball; slip probabilities left/right on a grid of step {side_grid[1]:.4f}): worst value "
              f"{worst_side:.4f} at left {arg_side[0]:.4f}, right {arg_side[1]:.4f}; the full rectangular ball: "
              f"{robust_values[(kind, kappa)]:.4f}")
    i_hat = int(np.argmin(np.abs(xis - XI_HAT)))
    i_3 = int(np.argmin(np.abs(xis - 0.3)))
    print(f"  cost of robustness at xi_hat: {curves['nominal'][i_hat] - curves[f'L1 {L1_SHOW:g}'][i_hat]:.4f} "
          f"({curves['nominal'][i_hat]:.4f} vs {curves[f'L1 {L1_SHOW:g}'][i_hat]:.4f}); "
          f"benefit at xi = 0.3: {curves[f'L1 {L1_SHOW:g}'][i_3] - curves['nominal'][i_3]:.4f} "
          f"({curves[f'L1 {L1_SHOW:g}'][i_3]:.4f} vs {curves['nominal'][i_3]:.4f})")
    print(f"\nTotal time {time.time() - t0:.1f} s")
    if args.quick:
        return

    # ---------------- Figure ----------------
    import plot_style as ps
    plt = ps.setup()
    fig_dir = __file__.rsplit("/", 1)[0] + "/figures/"
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1, 1.45]})
    R, C = len(LAYOUT), len(LAYOUT[0])
    for r in range(R):
        for c in range(C):
            ch = LAYOUT[r][c]
            face = {"#": "#5f5e5a", "L": "#e34948", "G": "#1baf7a"}.get(ch, "#fcfcfb")
            ax0.add_patch(plt.Rectangle((c, R - 1 - r), 1, 1, facecolor=face, edgecolor="#b9b8b3", lw=0.8))
            if ch in "SGL":
                ax0.text(c + 0.08, R - 1 - r + 0.92, {"S": "S", "G": "G", "L": "lava"}[ch], ha="left", va="top",
                         fontsize=9, color=ps.INK if ch == "S" else "white")
    for name, pi, col, off, ls in (("nominal policy", pi_nom, ps.C[1], -0.12, "-"),
                                   (f"robust policy (L1, $\\kappa$={L1_SHOW:g})", policies[f"L1 {L1_SHOW:g}"], ps.C[0], 0.12, "--")):
        path = route(g, pi)
        xs = [g.cells[s][1] + 0.5 + off for s in path]
        ys = [R - 1 - g.cells[s][0] + 0.5 + off for s in path]
        ax0.plot(xs, ys, color=col, ls=ls, lw=2.2, marker="o", ms=4, label=name)
    ax0.set_xlim(0, C)
    ax0.set_ylim(-0.05, R)
    ax0.set_aspect("equal")
    ax0.axis("off")
    ax0.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1)
    ax0.set_title(f"Routes planned for slip $\\hat\\xi$ = {XI_HAT}")

    cov = coverage(g, Pn, xis, "L1", L1_SHOW)
    ax1.axvspan(xis[cov].min(), xis[cov].max(), color="#e4e3df", alpha=0.6, lw=0)
    ax1.text(xis[cov].max() - 0.004, -0.40, f"slips covered by the\nL1 ball, $\\kappa$={L1_SHOW:g}", ha="right",
             va="bottom", fontsize=8.5, color="#5f5e5a")
    ax1.plot(xis, curves["oracle"], color=ps.GREY, ls="-", lw=1.4, label="optimal for the true $\\xi$")
    ax1.plot(xis, curves["nominal"], color=ps.C[1], label=f"nominal policy (planned for $\\xi$={XI_HAT})")
    ax1.plot(xis, curves[f"L1 {L1_SHOW:g}"], color=ps.C[0], ls="--", label=f"robust, L1 ball $\\kappa$={L1_SHOW:g}")
    ax1.plot(xis, curves[f"KL {KL_SHOW:g}"], color=ps.C[2], ls=(0, (1, 2.5)), lw=2.4,
             label=f"robust, KL ball $\\kappa$={KL_SHOW:g}")
    # The robust value bounds the true value only for models inside the ball: solid there, faint beyond.
    ax1.hlines(robust_values[("L1", L1_SHOW)], xis[cov].min(), xis[cov].max(), color=ps.C[0], lw=1.2, ls="-.")
    ax1.hlines(robust_values[("L1", L1_SHOW)], xis[cov].max(), 0.5, color=ps.C[0], lw=0.8, ls=":", alpha=0.6)
    ax1.text(0.5, robust_values[("L1", L1_SHOW)] + 0.012, "robust value (L1): a certified lower bound inside the ball",
             ha="right", fontsize=8.5, color="#5f5e5a")
    ax1.axvline(XI_HAT, color=ps.INK, lw=0.8)
    ax1.text(XI_HAT + 0.004, 0.66, "$\\hat\\xi$", fontsize=10)
    ax1.set_xlabel("true slip probability $\\xi$")
    ax1.set_ylabel("value from S")
    ax1.set_xlim(0, 0.5)
    ax1.legend(loc="upper right")
    ax1.set_title("Each policy evaluated exactly under the true slip model")
    fig.tight_layout()
    fig.savefig(fig_dir + "robust_vi.png")
    print(f"wrote {fig_dir}robust_vi.png")


if __name__ == "__main__":
    main()
