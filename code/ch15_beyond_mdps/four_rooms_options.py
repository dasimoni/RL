"""Options in the four-rooms domain: planning and learning with temporally extended actions.

Chapter 15, Section 5 (Algorithms 15.4-15.5).  A re-implementation of the rooms example of Sutton,
Precup & Singh (1999), "Between MDPs and semi-MDPs".

Domain.  104 free cells in four rooms joined by four one-cell hallways.  Four primitive actions
(up, down, left, right); with probability 2/3 the agent moves in the intended direction and with
probability 1/9 in each of the other three directions; bumping into a wall leaves it in place.
Reward 0 everywhere except +1 on ENTERING the goal cell, which terminates the episode.
gamma = 0.9.

Options.  Eight "hallway options", two per room.  Option (room, target hallway):
   initiation set I  = the room's cells + the room's OTHER hallway
   policy pi_omega   = the fastest way (under the slippery dynamics) to the target hallway,
                       computed by value iteration inside the room
   termination beta  = 0 inside the room, 1 everywhere else (hallways included) and at the goal.

What the script does
  1. Exact multi-time option models R(s, omega), P(s' | s, omega) (Eqs. 15.19-15.20) by solving
     linear systems, checked against Monte Carlo rollouts of the option.
  2. Synchronous SMDP value iteration (Eq. 15.21) with primitives only (A), hallway options only
     (H) and both (A u H), for a goal in a hallway (G1) and a goal inside a room (G2): value maps
     after 1, 2, 3 sweeps, and sweeps needed to converge.
  3. Learning: Q-learning with primitives, SMDP Q-learning with H and with A u H, and intra-option
     Q-learning with A u H (Eq. 15.22).  Steps per episode, averaged over seeds.

Run:  python code/ch15_beyond_mdps/four_rooms_options.py [--quick]
Full mode writes figures/four_rooms_planning.png and figures/four_rooms_learning.png.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import time

import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

LAYOUT = [
    "wwwwwwwwwwwww",
    "w     w     w",
    "w     w     w",
    "w           w",
    "w     w     w",
    "w     w     w",
    "ww wwww     w",
    "w     www www",
    "w     w     w",
    "w     w     w",
    "w           w",
    "w     w     w",
    "wwwwwwwwwwwww",
]
GAMMA = 0.9
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]      # up, down, left, right
HALLWAYS = {"N": (3, 6), "W": (6, 2), "E": (7, 9), "S": (10, 6)}
# room name -> (row range, col range, its two hallways)
ROOMS = {
    "NW": ((1, 5), (1, 5), ("N", "W")),
    "NE": ((1, 6), (7, 11), ("N", "E")),
    "SW": ((7, 11), (1, 5), ("W", "S")),
    "SE": ((8, 11), (7, 11), ("E", "S")),
}
GOALS = {"G1 (hallway)": HALLWAYS["E"], "G2 (in a room)": (9, 9)}
START = (2, 2)                                    # near the top-left corner


class FourRooms:
    def __init__(self):
        self.cells = [(r, c) for r, row in enumerate(LAYOUT) for c, ch in enumerate(row) if ch != "w"]
        self.idx = {rc: i for i, rc in enumerate(self.cells)}
        self.N = len(self.cells)
        # P[a, s, s']: 2/3 intended direction, 1/9 each other direction, walls -> stay
        self.P = np.zeros((4, self.N, self.N))
        for i, (r, c) in enumerate(self.cells):
            for a in range(4):
                for d, (dr, dc) in enumerate(MOVES):
                    prob = 2 / 3 if d == a else 1 / 9
                    j = self.idx.get((r + dr, c + dc), i)
                    self.P[a, i, j] += prob
        self.cumP = self.P.cumsum(axis=2)
        self.room_of = {}
        for name, ((r0, r1), (c0, c1), _) in ROOMS.items():
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if (r, c) in self.idx:
                        self.room_of[self.idx[(r, c)]] = name

    def step(self, rng, s, a):
        return int(np.searchsorted(self.cumP[a, s], rng.random() * self.cumP[a, s, -1]))


class Option:
    """A Markov option (I, pi, beta) with a deterministic policy."""

    def __init__(self, name, init, policy, cont):
        self.name = name
        self.init = init          # bool[N]: initiation set I
        self.policy = policy      # int[N]: action pi(s) (-1 where undefined)
        self.cont = cont          # bool[N]: beta(s) = 0 (continue) iff cont[s]


def hallway_options(env):
    opts = []
    for room, (_, _, halls) in ROOMS.items():
        cells = np.array([env.room_of.get(i) == room for i in range(env.N)])
        for target, other in [(halls[0], halls[1]), (halls[1], halls[0])]:
            t, o = env.idx[HALLWAYS[target]], env.idx[HALLWAYS[other]]
            # option policy: value iteration on "reach t" inside the room (+1 on entering t;
            # leaving via the other hallway ends with 0).  Gives the fastest route under slips.
            domain = cells.copy()
            domain[o] = True
            V = np.zeros(env.N)
            for _ in range(500):
                Q = np.einsum("asn,n->sa", env.P, np.where(np.arange(env.N) == t, 1.0, GAMMA * V * cells))
                V_new = np.where(domain, Q.max(axis=1), 0.0)
                if np.abs(V_new - V).max() < 1e-12:
                    break
                V = V_new
            policy = np.where(domain, Q.argmax(axis=1), -1)
            init = domain.copy()
            opts.append(Option(f"{room}->{target}", init, policy, cells.copy()))
    return opts


def primitive_options(env):
    return [Option(f"prim-{a}", np.ones(env.N, bool), np.full(env.N, a), np.zeros(env.N, bool))
            for a in range(4)]


def option_model(env, opt, goal):
    """Multi-time model of an option (Eqs. 15.19-15.20), exactly, by a linear solve.

    R[s]     = E[ r_1 + gamma r_2 + ... + gamma^{k-1} r_k ]                 (k = duration)
    Pd[s,x]  = sum_k gamma^k Pr(option terminates in x after exactly k steps)
    For s in the continuation set D (beta = 0):  (I - gamma P_DD) R_D = r_D, etc.  The goal is
    terminal: entering it pays +1 and stops the option (and the episode).
    """
    N = env.N
    Ppi = env.P[np.where(opt.policy >= 0, opt.policy, 0), np.arange(N)]      # P_pi[s, s']
    r = Ppi[:, goal].copy()                                                   # Pr(enter goal)
    cont = opt.cont.copy()
    cont[goal] = False                                                        # goal: beta = 1
    D = np.where(cont)[0]
    stop = ~cont
    # values for states in D
    if len(D) > 0:
        M = np.eye(len(D)) - GAMMA * Ppi[np.ix_(D, D)]
        R_D = np.linalg.solve(M, r[D])
        T_D = np.linalg.solve(M, GAMMA * Ppi[np.ix_(D, np.where(stop)[0])])
    else:                          # a primitive action: always stops after one step
        R_D, T_D = np.zeros(0), np.zeros((0, int(stop.sum())))
    R = np.zeros(N)
    Pd = np.zeros((N, N))
    R[D] = R_D
    Pd[np.ix_(D, np.where(stop)[0])] = T_D
    # states outside D where the option may be initiated: one step, then continue from D
    for s in np.where(opt.init & ~cont)[0]:
        R[s] = r[s] + GAMMA * Ppi[s, D] @ R_D
        Pd[s, stop] = GAMMA * Ppi[s, stop] + GAMMA * Ppi[s, D] @ T_D
    Pd[:, goal] = 0.0             # V(goal) = 0: the +1 is already inside R
    R[~opt.init] = 0.0
    Pd[~opt.init] = 0.0
    return R, Pd


def run_option(env, rng, opt, s, goal):
    """Execute an option until it terminates: returns (s', discounted reward, k, primitive steps)."""
    G, disc, k = 0.0, 1.0, 0
    transitions = []
    while True:
        a = opt.policy[s]
        s2 = env.step(rng, s, a)
        r = 1.0 if s2 == goal else 0.0
        transitions.append((s, a, r, s2))
        G += disc * r
        disc *= GAMMA
        k += 1
        s = s2
        if s == goal or not opt.cont[s]:
            return s, G, k, transitions


def check_model_mc(env, opts, goal, rng, n_rollouts):
    """Monte Carlo check of the option models at a few random initiation states.

    Compares R(s, o) with the mean discounted reward of rollouts, and Pd(x | s, o) with the mean
    of gamma^k * 1[option ends in x] (x != goal).  Returns the two largest absolute errors.
    """
    worst_R, worst_P = 0.0, 0.0
    for opt in opts:
        R, Pd = option_model(env, opt, goal)
        starts = np.where(opt.init)[0]
        for s in rng.choice(starts, size=2, replace=False):
            if s == goal:
                continue
            Gs, mass = [], np.zeros(env.N)
            for _ in range(n_rollouts):
                s_end, G, k, _ = run_option(env, rng, opt, s, goal)
                Gs.append(G)
                if s_end != goal:
                    mass[s_end] += GAMMA ** k
            worst_R = max(worst_R, abs(np.mean(Gs) - R[s]))
            worst_P = max(worst_P, np.abs(mass / n_rollouts - Pd[s]).max())
    return worst_R, worst_P


def smdp_value_iteration(models, init_masks, N, goal, sweeps):
    """Synchronous SMDP value iteration (Eq. 15.21): V(s) = max_{o in O_s} R(s,o) + sum_x Pd(x|s,o) V(x)."""
    V = np.zeros(N)
    history = [V.copy()]
    for _ in range(sweeps):
        cand = np.full((len(models), N), -np.inf)
        for i, (R, Pd) in enumerate(models):
            cand[i] = np.where(init_masks[i], R + Pd @ V, -np.inf)
        V = cand.max(axis=0)
        V[goal] = 0.0
        V[np.isinf(V)] = 0.0
        history.append(V.copy())
    return history


# ----------------------------------------------------------------------------------------------
# Learning (Algorithm 15.4: SMDP Q-learning; Algorithm 15.5: intra-option Q-learning)
# ----------------------------------------------------------------------------------------------
def learn(env, rng, opts, goal, method, episodes, alpha=0.125, eps=0.1, max_steps=20_000):
    """Returns primitive steps per episode.

    method: 'smdp'  - SMDP Q-learning: one update per executed option, Eq. (15.18)
            'intra' - intra-option Q-learning: after EVERY primitive step, update every option
                      whose policy would have taken that action in that state, Eq. (15.22)
    Primitive actions are options that always terminate after one step, so with opts = A
    both methods reduce to ordinary Q-learning.
    """
    N, K = env.N, len(opts)
    Q = np.zeros((N, K))
    avail = np.stack([o.init for o in opts], axis=1)            # avail[s, o]: s in I_o
    cont = np.stack([o.cont for o in opts], axis=1)
    pol = np.stack([o.policy for o in opts], axis=1)
    steps_per_episode = []
    start = env.idx[START]
    for _ in range(episodes):
        s, steps = start, 0
        while s != goal and steps < max_steps:
            choices = np.where(avail[s])[0]
            if rng.random() < eps:
                o = rng.choice(choices)
            else:
                q = Q[s, choices]
                o = choices[rng.choice(np.flatnonzero(q == q.max()))]   # random tie-breaking
            s2, G, k, trans = run_option(env, rng, opts[o], s, goal)
            steps += k
            if method == "smdp":
                target = G if s2 == goal else G + GAMMA ** k * Q[s2, avail[s2]].max()
                Q[s, o] += alpha * (target - Q[s, o])
            else:
                for (x, a, r, x2) in trans:
                    # every option consistent with (x, a): x in its domain and pi_o(x) = a
                    consistent = np.where(avail[x] & (pol[x] == a))[0]
                    if x2 == goal:
                        U = np.zeros(len(consistent))
                    else:
                        best = Q[x2, avail[x2]].max()
                        # U(x2, o) = (1 - beta_o(x2)) Q(x2, o) + beta_o(x2) max_o' Q(x2, o')
                        U = np.where(cont[x2, consistent], Q[x2, consistent], best)
                    Q[x, consistent] += alpha * (r + GAMMA * U - Q[x, consistent])
            s = s2
        steps_per_episode.append(steps)
    return np.array(steps_per_episode), Q


def intra_option_offpolicy(env, rng, opts, goal, Q_true, n_steps, alpha=0.1, log_every=2_000):
    """Learn the values of the hallway options WITHOUT ever executing them (Section 5.4).

    Behaviour: uniformly random PRIMITIVE actions (episodes restart from a random cell after the
    goal).  After every step (x, a, r, x2), every option whose deterministic policy picks a in x is
    updated with Eq. (15.22).  No importance ratios are needed: for a deterministic option policy
    the ratio pi_o(a|x)/mu(a|x) is either 0 (option not updated) or a constant absorbed into alpha.
    SMDP Q-learning cannot do this at all: it only learns about options it executes.
    Returns (steps, mean |Q - Q_H*| over all (state, option) pairs with s in I_o).
    """
    K = len(opts)
    Q = np.zeros((env.N, K))
    avail = np.stack([o.init for o in opts], axis=1)
    avail[goal] = False
    cont = np.stack([o.cont for o in opts], axis=1)
    pol = np.stack([o.policy for o in opts], axis=1)
    non_goal = [i for i in range(env.N) if i != goal]
    x = rng.choice(non_goal)
    curve = []
    for t in range(1, n_steps + 1):
        a = rng.integers(4)
        x2 = env.step(rng, x, a)
        r = 1.0 if x2 == goal else 0.0
        consistent = np.where(avail[x] & (pol[x] == a))[0]
        if len(consistent):
            if x2 == goal:
                U = np.zeros(len(consistent))
            else:
                U = np.where(cont[x2, consistent], Q[x2, consistent], Q[x2, avail[x2]].max())
            Q[x, consistent] += alpha * (r + GAMMA * U - Q[x, consistent])
        x = rng.choice(non_goal) if x2 == goal else x2
        if t % log_every == 0:
            curve.append((t, np.abs(Q - Q_true)[avail].mean()))
    return np.array(curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figures")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=150)
    args = ap.parse_args()
    t0 = time.time()
    seeds = 3 if args.quick else args.seeds
    episodes = 20 if args.quick else args.episodes
    print(f"Four rooms with hallway options | gamma={GAMMA} seeds={seeds} episodes={episodes} "
          f"alpha=0.125 eps=0.1 start={START} quick={args.quick}")
    n_off_seeds = 2 if args.quick else 5
    print(f"rng seeds: option-model checks 0; learning runs 1000..{1000 + seeds - 1}; "
          f"off-policy intra-option runs 500..{500 + n_off_seeds - 1}")
    env = FourRooms()
    H = hallway_options(env)
    A = primitive_options(env)
    print(f"{env.N} free cells; {len(H)} hallway options: " + ", ".join(o.name for o in H))
    rng = np.random.default_rng(0)

    # ---- 1. option models, checked by Monte Carlo ------------------------------------------
    g1 = env.idx[GOALS["G1 (hallway)"]]
    wR, wP = check_model_mc(env, H, g1, rng, n_rollouts=300 if args.quick else 3000)
    print(f"[models] Monte Carlo check at 2 initiation states per option: max |R error| = {wR:.4f}, "
          f"max |P error| = {wP:.4f}")
    o = H[0]
    s = env.idx[(2, 2)]
    R, Pd = option_model(env, o, g1)
    k_mean = np.mean([run_option(env, rng, o, s, g1)[2] for _ in range(2000)])
    nz = np.flatnonzero(Pd[s] > 1e-9)
    print(f"[models] option {o.name} from cell (2,2): mean duration {k_mean:.2f} steps; "
          f"terminal distribution sum_k gamma^k Pr(...): "
          + ", ".join(f"{env.cells[x]}: {Pd[s, x]:.4f}" for x in nz)
          + f"  (sum = {Pd[s].sum():.4f})")

    # ---- 2. SMDP value iteration ------------------------------------------------------------
    plan = {}
    for gname, gcell in GOALS.items():
        goal = env.idx[gcell]
        sets = {"A (primitives)": A, "H (hallway options)": H, "A u H": A + H}
        for sname, opts in sets.items():
            models = [option_model(env, o, goal) for o in opts]
            masks = [o.init.copy() for o in opts]
            hist = smdp_value_iteration(models, masks, env.N, goal, sweeps=300)
            # contraction modulus of the SMDP Bellman operator: max over (s, option) of the row
            # sum of P(.|s, option) = E[gamma^K; the option ends outside the goal] (Exercise 8)
            modulus = max(Pd[m & (np.arange(env.N) != goal)].sum(axis=1).max()
                          for (_, Pd), m in zip(models, masks))
            Vinf = hist[-1]
            err = [np.abs(V - Vinf).max() for V in hist]
            n_conv = next(i for i, e in enumerate(err) if e < 1e-4)
            reach = [(V > 0).sum() for V in hist]
            n_reach = next((i for i, rr in enumerate(reach) if rr >= env.N - 1), None)
            plan[(gname, sname)] = hist
            print(f"[plan] {gname:<15} {sname:<20} V(start) = {Vinf[env.idx[START]]:.4f}; "
                  f"sweeps to |V_k - V_inf| < 1e-4: {n_conv:3d}; states with V > 0 after "
                  f"1, 2, 3 sweeps: {reach[1]}, {reach[2]}, {reach[3]}; sweeps until every "
                  f"state has V > 0: {n_reach}; contraction modulus max E[gamma^K] = {modulus:.4f}")

    # ---- 3. learning ------------------------------------------------------------------------
    learning = {}
    configs = [("Q-learning, A", A, "smdp"),
               ("SMDP Q-learning, H", H, "smdp"),
               ("SMDP Q-learning, A u H", A + H, "smdp"),
               ("intra-option Q-learning, A u H", A + H, "intra")]
    for gname, gcell in GOALS.items():
        goal = env.idx[gcell]
        for label, opts, method in configs:
            if "H" in label and "A" not in label and "room" in gname:
                continue    # H alone reaches G2 only by accident (V(start) = 0.0289): skip it
            t1 = time.time()
            runs = np.stack([learn(env, np.random.default_rng(1000 + sd), opts, goal, method, episodes)[0]
                             for sd in range(seeds)])
            learning[(gname, label)] = runs
            m = runs.mean(axis=0)
            print(f"[learn] {gname:<15} {label:<32} mean steps/episode: episodes 1-10 {m[:10].mean():7.1f}, "
                  f"11-50 {m[10:50].mean():6.1f}, last 20 {m[-20:].mean():6.1f}   [{time.time() - t1:.1f} s]")

    # ---- 4. intra-option learning about options that are never executed -----------------------
    goal = env.idx[GOALS["G1 (hallway)"]]
    models = [option_model(env, o, goal) for o in H]
    V_H = smdp_value_iteration(models, [o.init for o in H], env.N, goal, sweeps=300)[-1]
    Q_true = np.stack([R + Pd @ V_H for R, Pd in models], axis=1)
    avail = np.stack([o.init for o in H], axis=1)
    avail[goal] = False
    scale = np.abs(Q_true[avail]).mean()
    n_off = 20_000 if args.quick else 200_000
    off_curves = np.stack([intra_option_offpolicy(env, np.random.default_rng(500 + sd), H, goal, Q_true, n_off)
                           for sd in range(n_off_seeds)])
    m = off_curves[:, :, 1].mean(axis=0)
    print(f"[off-policy] intra-option Q-learning of the 8 hallway options from random primitive actions "
          f"(goal G1): mean |Q - Q_H*| = {m[0]:.4f} after {int(off_curves[0, 0, 0])} steps, "
          f"{m[len(m) // 2 - 1]:.4f} after {int(off_curves[0, len(m) // 2 - 1, 0])}, {m[-1]:.4f} after "
          f"{int(off_curves[0, -1, 0])} (mean |Q_H*| = {scale:.4f}); the options were never executed")

    if not args.quick:
        from plot_style import setup, C
        plt = setup()
        os.makedirs(FIG_DIR, exist_ok=True)
        rows = [("G1 (hallway)", "A (primitives)"), ("G1 (hallway)", "H (hallway options)"),
                ("G2 (in a room)", "A (primitives)"), ("G2 (in a room)", "A u H")]
        cols = [1, 2, 3, 300]
        fig, axes = plt.subplots(len(rows), len(cols), figsize=(9.5, 10))
        for i, key in enumerate(rows):
            for j, it in enumerate(cols):
                ax = axes[i, j]
                img = np.full((len(LAYOUT), len(LAYOUT[0])), np.nan)
                V = plan[key][it]
                for k, (r, c) in enumerate(env.cells):
                    img[r, c] = V[k]
                walls = np.array([[1.0 if ch == "w" else np.nan for ch in row] for row in LAYOUT])
                ax.imshow(walls, cmap="Greys", vmin=0, vmax=1.5)
                ax.imshow(img, cmap="viridis", vmin=0, vmax=1)
                gr, gc = GOALS[key[0]]
                ax.text(gc, gr, "G", ha="center", va="center", color="white", fontsize=9, weight="bold")
                ax.text(START[1], START[0], "S", ha="center", va="center", color="white", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                if i == 0:
                    ax.set_title("converged" if it == 300 else f"after {it} sweep{'s' if it > 1 else ''}")
                if j == 0:
                    ax.set_ylabel(["goal G1, A", "goal G1, H", "goal G2, A", "goal G2, A ∪ H"][i])
        fig.suptitle("SMDP value iteration in the four rooms (colour = V, 0 to 1)", y=0.995)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "four_rooms_planning.png"))
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1, 0.9]})
        styles = {"Q-learning, A": ("--", C[0]), "SMDP Q-learning, H": (":", C[3]),
                  "SMDP Q-learning, A u H": ("-", C[2]), "intra-option Q-learning, A u H": ("-.", C[1])}
        for ax, gname in zip(axes[:2], GOALS):
            for label, (ls, col) in styles.items():
                if (gname, label) not in learning:
                    continue
                runs = learning[(gname, label)]
                kernel = np.ones(5) / 5                    # 5-episode moving average for legibility
                sm = np.stack([np.convolve(r_, kernel, mode="valid") for r_ in runs.astype(float)])
                m = sm.mean(axis=0)
                se = sm.std(axis=0, ddof=1) / np.sqrt(len(sm))
                x = np.arange(5, len(m) + 5)
                ax.plot(x, m, ls=ls, color=col, label=label)
                ax.fill_between(x, m - 2 * se, m + 2 * se, color=col, alpha=0.15)
            ax.set_yscale("log")
            ax.set_xlabel("episode (5-episode moving average)")
            ax.set_title(f"goal {gname}")
        axes[0].set_ylabel(f"primitive steps per episode (mean of {seeds} seeds)")
        axes[1].set_ylim(axes[0].get_ylim())
        axes[0].legend()
        ax = axes[2]
        for curve in off_curves:
            ax.plot(curve[:, 0] / 1000, curve[:, 1] / scale, color=C[1], lw=1, alpha=0.5)
        ax.plot(off_curves[0, :, 0] / 1000, off_curves[:, :, 1].mean(axis=0) / scale, color=C[1], lw=2.5,
                label=f"mean over {len(off_curves)} seeds (thin: single seeds)")
        ax.set_yscale("log")
        ax.set_xlabel("thousands of primitive steps of a RANDOM behaviour policy")
        ax.set_ylabel("mean |Q(s,o) − Q*_H(s,o)| / mean |Q*_H|")
        ax.set_title("Intra-option learning of 8 options never executed")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "four_rooms_learning.png"))
        plt.close(fig)
        print(f"figures written to {FIG_DIR}")
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
