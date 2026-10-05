"""Numerical checks for the exercises of Chapter 15.

Each function prints the numbers quoted in the corresponding worked solution.

  Ex 2   closed-form value of the belief-optimal Tiger policy (net-count controller)
  Ex 3   horizon-2 alpha vectors of the Tiger problem, by hand and by the exact backup
  Ex 4   value of the memoryless policy "open the door opposite the last growl"
  Ex 7   upward bias of hindsight relabelling in a stochastic one-step environment
  Ex 9   multi-time model of an option in a corridor (Eqs. 15.19-15.20 vs closed form)
  Ex 10  GPI on a corridor with two rewarding ends
  Ex 11  successor representation of a two-state chain (closed form vs inverse)
  Ex 12  Bayes-optimal two-armed Bernoulli bandit for horizons 1, 2 and 3
  Ex 13  tabular Q-learning with frame stacking (window-k) on the Tiger problem
  Ex 14  interrupting hallway options in the four rooms (interruption theorem)
  Ex 15  MAML on a quadratic: Eq. (15.32) by finite differences; exact vs first-order MAML
  Ex 16  bisimulation metric (Eq. 15.29a) on small random MDPs and the bound |v*(s) - v*(t)| <= d(s, t)
  Ex 19  a two-stage reward machine: product MDP vs memoryless policies on the corridor
  Ex 20  an epistemic POMDP: two contexts, every memoryless policy is suboptimal
  Ex 21  the PLR replay distribution, by hand and from procgen_lite.PLRSampler

Run:  python code/ch15_beyond_mdps/exercise_solutions.py [--quick]   (no figures)
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import time

import numpy as np

import four_rooms_options as fr
import rl2_bandits as rb
import tiger_pomdp as tg


def ex2():
    g = tg.GAMMA
    b2 = 0.85 ** 2 / (0.85 ** 2 + 0.15 ** 2)
    r2 = 10 * b2 - 100 * (1 - b2)
    p_hl = 0.85 ** 2 + 0.15 ** 2
    A = np.array([[1, -g, 0], [-g * (1 - p_hl), 1, -g * p_hl], [-g, 0, 1]])
    V = np.linalg.solve(A, np.array([-1.0, -1.0, r2]))
    G, acts, _ = tg.value_iteration(eps=1e-6, tol=1e-10, max_iter=2000)
    v_star = float((G @ np.array([0.5, 0.5])).max())
    print(f"[Ex 2] b(c=2) = {b2:.6f}, expected reward of opening at c=2 = {r2:.6f}, Pr(HL | c=1) = {p_hl:.3f}")
    print(f"[Ex 2] V0, V1, V2 = {V[0]:.6f}, {V[1]:.6f}, {V[2]:.6f};  alpha-vector V*(0.5) = {v_star:.6f}")


def ex3():
    g = tg.GAMMA
    plan_open = (-1 + g * (0.85 * 10 + 0.15 * (-100)), -1 + g * (0.15 * (-100) + 0.85 * 10))
    plan_listen = (-1 + g * (-1), -1 + g * (-1))
    plan_mixed = (-1 + g * (0.85 * 10 + 0.15 * (-1)), -1 + g * (0.15 * (-100) + 0.85 * (-1)))
    print(f"[Ex 3] listen, then open opposite the growl: alpha = ({plan_open[0]:.4f}, {plan_open[1]:.4f})")
    print(f"[Ex 3] listen, then listen:                   alpha = ({plan_listen[0]:.4f}, {plan_listen[1]:.4f})")
    print(f"[Ex 3] listen, then OR after HL / listen after HR: alpha = ({plan_mixed[0]:.4f}, {plan_mixed[1]:.4f})")
    G = np.zeros((1, 2))
    acts = np.zeros(1, dtype=int)
    for _ in range(2):
        G, acts, _ = tg.alpha_backup(G, acts)
    order = np.argsort(G[:, 0] - G[:, 1])
    print("[Ex 3] exact pruned Gamma_2 (alpha(TL), alpha(TR), root action): "
          + "; ".join(f"({G[k, 0]:.4f}, {G[k, 1]:.4f}, {tg.A_NAMES[acts[k]]})" for k in order))


def ex4():
    table = {"start": tg.onehot(0), 0: np.array([0.0, 0.0, 1.0]), 1: np.array([0.0, 1.0, 0.0])}
    _, pi, nxt = tg.memoryless_controller(table)
    v = tg.evaluate(pi, nxt)
    g = tg.GAMMA
    closed = -1 + g * (-6.5) + g ** 2 * (-45) / (1 - g)
    print(f"[Ex 4] memoryless 'open opposite the last growl' (listen at start): exact V = {v:.3f}; "
          f"closed form -1 + 0.95(-6.5) + 0.95^2(-45)/0.05 = {closed:.3f}")


def ex7(rng, n):
    """One-step task: from s0 the only action reaches s1 or s2 w.p. 1/2; reward 0 iff s' = g."""
    s_next = rng.integers(1, 3, size=n)              # achieved state: 1 or 2
    g = rng.integers(1, 3, size=n)                   # original goal
    # stored tuples for goal 1: originals with g = 1, plus 'final' relabels with g' = s_next = 1
    orig = g == 1
    rew_orig = np.where(s_next[orig] == 1, 0.0, -1.0)
    relab = s_next == 1                              # relabelled goal equals the achieved state
    rew_relab = np.zeros(int(relab.sum()))
    est_no_her = rew_orig.mean()
    est_her = np.concatenate([rew_orig, rew_relab]).mean()
    print(f"[Ex 7] true Q(s0, a, g=s1) = -0.5; mean stored target without HER = {est_no_her:.4f}, "
          f"with 'final' relabelling = {est_her:.4f}  ({n} episodes)")


def ex9():
    gamma = 0.9
    # corridor 0-1-2-3, deterministic 'right', reward -1 per step, option from 0 terminates at 3
    N = 4
    P = np.zeros((N, N))
    for s in range(3):
        P[s, s + 1] = 1.0
    cont = np.array([True, True, True, False])       # beta = 0 on 0,1,2 ; beta(3) = 1
    D = np.where(cont)[0]
    r = np.where(cont, -1.0, 0.0)
    M = np.eye(len(D)) - gamma * P[np.ix_(D, D)] * cont[D][None, :]
    R = np.linalg.solve(M, r[D])                     # Eq. (15.19)
    Pterm = np.linalg.solve(M, gamma * P[np.ix_(D, [3])])   # Eq. (15.20), x = 3
    print(f"[Ex 9] r(0, w) = {R[0]:.4f} (closed form -(1 + 0.9 + 0.81) = {-(1 + 0.9 + 0.81):.4f}); "
          f"P(3 | 0, w) = {Pterm[0, 0]:.4f} (0.9^3 = {0.9 ** 3:.4f})")
    # slippery version: 'right' succeeds w.p. 0.8, otherwise stays
    P2 = np.zeros((N, N))
    for s in range(3):
        P2[s, s + 1] = 0.8
        P2[s, s] = 0.2
    M2 = np.eye(len(D)) - gamma * P2[np.ix_(D, D)]
    R2 = np.linalg.solve(M2, r[D])
    T2 = np.linalg.solve(M2, gamma * P2[np.ix_(D, [3])])
    q = 0.8 * gamma / (1 - 0.2 * gamma)
    print(f"[Ex 9] slippery (success 0.8): r(0, w) = {R2[0]:.4f}, P(3 | 0, w) = {T2[0, 0]:.4f} "
          f"(closed form E[gamma^K] = (0.8*0.9/(1-0.2*0.9))^3 = {q ** 3:.4f})")


def ex10():
    """Corridor cells 0..6; entering cell 0 pays w[0], entering cell 6 pays w[1]; both terminate."""
    gamma, L = 0.9, 7

    def value(policy_dir, w):                        # iterative policy evaluation (deterministic moves)
        v = np.zeros(L)
        for _ in range(500):
            v_new = v.copy()
            for s in range(1, L - 1):
                s2 = s + (-1 if policy_dir[s] == "L" else 1)
                reward = w[0] if s2 == 0 else (w[1] if s2 == L - 1 else 0.0)
                v_new[s] = reward + (0.0 if s2 in (0, L - 1) else gamma * v[s2])
            v = v_new
        return v

    for w in [(1.0, 1.0), (-1.0, -1.0)]:
        vL, vR = value(["L"] * L, w), value(["R"] * L, w)
        gpi = []                                     # GPI: argmax_a max_i [r + gamma v_i(s')]
        for s in range(L):
            best, best_a = -np.inf, "L"
            for a in "LR":
                s2 = s + (-1 if a == "L" else 1)
                if s2 < 0 or s2 >= L:
                    continue
                reward = w[0] if s2 == 0 else (w[1] if s2 == L - 1 else 0.0)
                for v in (vL, vR):
                    q = reward + (0.0 if s2 in (0, L - 1) else gamma * v[s2])
                    if q > best:
                        best, best_a = q, a
            gpi.append(best_a)
        vG = value(gpi, w)
        print(f"[Ex 10] cells 1..5, w = {w}, gamma = {gamma}")
        print("        v(pi_L) = " + ", ".join(f"{x:.3f}" for x in vL[1:-1]))
        print("        v(pi_R) = " + ", ".join(f"{x:.3f}" for x in vR[1:-1]))
        print("        v(GPI)  = " + ", ".join(f"{x:.3f}" for x in vG[1:-1]) + f"   (actions {''.join(gpi[1:-1])})")


def ex11():
    gamma, p = 0.9, 0.2
    P = np.array([[1 - p, p], [p, 1 - p]])
    M = np.linalg.inv(np.eye(2) - gamma * P)
    a = 1 / (1 - gamma)
    c = 1 / (1 - gamma * (1 - 2 * p))
    closed = 0.5 * np.array([[a + c, a - c], [a - c, a + c]])
    print(f"[Ex 11] gamma = {gamma}, p = {p}: M = [[{M[0, 0]:.4f}, {M[0, 1]:.4f}], [{M[1, 0]:.4f}, {M[1, 1]:.4f}]]; "
          f"closed form max error {np.abs(M - closed).max():.1e}; eigenvalues {a:.4f}, {c:.4f}")


def ex12():
    for H in (1, 2, 3):
        _, v = rb.bayes_optimal_independent(H)
        print(f"[Ex 12] H = {H}: Bayes-optimal expected reward = {v:.4f}; oracle = {H * 2 / 3:.4f}; "
              f"Bayes regret = {H * 2 / 3 - v:.4f}")


def ex13(steps, seeds, ep_len=50, eps=0.1, power=0.8):
    """Tabular Q-learning whose input is (a) the window of the last k (action, observation) pairs
    ("frame stacking"), or (b) the net growl count since the last reset (a Markov summary).

    The Tiger task is continuing; we cut experience into episodes of ep_len steps (fresh tiger,
    empty window) so that the padded start windows are visited often, and we BOOTSTRAP through
    these artificial truncations.  Step size 1/n(x, a)^power; epsilon-greedy behaviour.  The greedy
    policy is then evaluated EXACTLY from b0 = 0.5 with the controller evaluator of tiger_pomdp.py.
    """
    gamma = tg.GAMMA

    def run(kind, k, sd, power=power, ep_len=ep_len):
        r_ = np.random.default_rng(100 + sd)
        Q, N = {}, {}
        u = r_.random((steps, 4))                  # pre-drawn uniforms: explore?, action, s', o
        for t in range(steps):
            if t == 0 or (ep_len is not None and t % ep_len == 0):
                s = int(r_.integers(2))
                node = 0 if kind == "count" else (None,) * k
            q = Q.setdefault(node, np.zeros(3))
            n = N.setdefault(node, np.zeros(3))
            a = int(u[t, 1] * 3) if u[t, 0] < eps else int(np.argmax(q))
            rew = tg.R[s, a]
            s2 = s if a == 0 else int(u[t, 2] < 0.5)                 # opening re-places the tiger
            if a == 0:
                o = s2 if u[t, 3] < tg.ACC else 1 - s2               # growl on the tiger's side w.p. 0.85
            else:
                o = int(u[t, 3] < 0.5)                               # after opening: noise
            if kind == "count":
                node2 = 0 if a > 0 else max(-6, min(6, node + (1 if o == 0 else -1)))
            else:
                node2 = node[1:] + ((a, o),)
            q2 = Q.setdefault(node2, np.zeros(3))
            n[a] += 1
            step = 0.05 if power is None else 1.0 / n[a] ** power      # None: constant step size
            q[a] += step * (rew + gamma * q2.max() - q[a])
            s, node = s2, node2

        def greedy(m):
            return int(np.argmax(Q[m])) if m in Q else 0
        if kind == "count":
            _, pi, nxt = tg.compile_controller(
                0, lambda m: tg.onehot(greedy(m)),
                lambda m, a, o: 0 if a > 0 else max(-6, min(6, m + (1 if o == 0 else -1))))
        else:
            _, pi, nxt = tg.window_controller(k, greedy)
        return tg.evaluate(pi, nxt)

    for kind, k in [("count", 0), ("window", 1), ("window", 2), ("window", 3), ("window", 4)]:
        vals = [run(kind, k, sd) for sd in range(seeds)]
        label = "net-count input (Markov)" if kind == "count" else f"window k={k}"
        print(f"[Ex 13] Q-learning, {label:<25} ({steps} steps, {seeds} seeds): exact value of the greedy "
              f"policy = " + ", ".join(f"{v:.3f}" for v in vals))
    vals = [run("count", 0, sd, power=None) for sd in range(seeds)]
    print("[Ex 13] variant: net-count input, CONSTANT step size 0.05: " + ", ".join(f"{v:.3f}" for v in vals))
    vals = [run("window", 1, sd, ep_len=None) for sd in range(seeds)]
    print("[Ex 13] variant: window k=1 WITHOUT periodic resets (start window seen once): "
          + ", ".join(f"{v:.3f}" for v in vals))


def ex14():
    env = fr.FourRooms()
    H = fr.hallway_options(env)
    start = env.idx[fr.START]
    for gname, gcell in fr.GOALS.items():
        goal = env.idx[gcell]
        models = [fr.option_model(env, o, goal) for o in H]
        V = fr.smdp_value_iteration(models, [o.init for o in H], env.N, goal, sweeps=400)[-1]
        Q = np.stack([np.where(o.init, R + Pd @ V, -np.inf) for o, (R, Pd) in zip(H, models)], axis=1)
        K = len(H)
        # Interrupted execution as a Markov chain on (state, running option).  At (s, w): if the
        # option has terminated (beta = 1, i.e. s outside its room) or Q(s, w) < V(s), switch to
        # argmax_w' Q(s, w'); then take pi_w(s).  Without interruption: switch only on termination.
        for interrupt in (False, True):
            idx = lambda s, w: s * K + w
            A_ = np.eye(env.N * K)
            b = np.zeros(env.N * K)
            for s in range(env.N):
                if s == goal:
                    continue
                for w in range(K):
                    cur = w
                    must = (not H[w].cont[s]) or (not H[w].init[s])
                    if must or (interrupt and Q[s, w] < V[s] - 1e-12):
                        cur = int(np.argmax(Q[s]))
                    a = H[cur].policy[s]
                    for s2 in np.flatnonzero(env.P[a, s]):
                        p = env.P[a, s, s2]
                        if s2 == goal:
                            b[idx(s, w)] += p * 1.0
                        else:
                            A_[idx(s, w), idx(s2, cur)] -= fr.GAMMA * p
            Vsw = np.linalg.solve(A_, b)
            w0 = int(np.argmax(Q[start]))
            v_start = Vsw[idx(start, w0)]
            mean_v = np.mean([Vsw[idx(s, int(np.argmax(Q[s])))] for s in range(env.N) if s != goal])
            print(f"[Ex 14] {gname:<15} {'with' if interrupt else 'without'} interruption: "
                  f"V(start) = {v_start:.4f}; mean over states = {mean_v:.4f}")
        print(f"[Ex 14] {gname:<15} SMDP-optimal V_H(start) = {V[start]:.4f}")


def ex15(rng, c_bar=1.0, sigma=0.5, eta=0.05, steps=200, batch=32):
    """J_M(theta) = -(theta - c)^2, c ~ N(c_bar, sigma^2); one exact inner step of size alpha.

    theta' = theta - 2 alpha (theta - c).  Exact meta-gradient (Eq. 15.32): (1 - 2 alpha) * dJ/dtheta'
    = -2 (1 - 2 alpha)^2 (theta - c).  First-order MAML drops the factor (1 - 2 alpha) from the
    Jacobian: -2 (1 - 2 alpha) (theta - c).  We check Eq. (15.32) against a finite difference and run
    meta-gradient ASCENT with both, from theta = 0, on mini-batches of sampled tasks.
    """
    def meta_obj(theta, c, alpha):
        theta_p = theta - 2 * alpha * (theta - c)
        return -(theta_p - c) ** 2

    for alpha in (0.25, 0.75):
        c = rng.normal(c_bar, sigma, size=5)
        h = 1e-5
        fd = (meta_obj(0.3 + h, c, alpha) - meta_obj(0.3 - h, c, alpha)) / (2 * h)
        formula = (1 - 2 * alpha) * (-2 * ((0.3 - 2 * alpha * (0.3 - c)) - c))   # Jacobian x grad at theta'
        theta = {"exact": 0.0, "first-order": 0.0}
        for _ in range(steps):
            cs = rng.normal(c_bar, sigma, size=batch)
            for kind in theta:
                factor = (1 - 2 * alpha) ** 2 if kind == "exact" else (1 - 2 * alpha)
                theta[kind] += eta * np.mean(-2 * factor * (theta[kind] - cs))
        print(f"[Ex 15] alpha = {alpha}: Eq. (15.32) vs finite difference, max error "
              f"{np.abs(fd - formula).max():.1e}; after {steps} meta-steps (eta = {eta}) from theta = 0, "
              f"c_bar = {c_bar}: exact MAML theta = {theta['exact']:.4f}, first-order MAML theta = "
              f"{theta['first-order']:.4g}")


def w1_distance(p, q, D):
    """Wasserstein-1 distance between distributions p, q on n points with ground metric D (an LP)."""
    from scipy.optimize import linprog
    n = len(p)
    A_eq = np.zeros((2 * n, n * n))
    for i in range(n):
        A_eq[i, i * n:(i + 1) * n] = 1.0          # row sums = p
        A_eq[n + i, i::n] = 1.0                   # column sums = q
    res = linprog(D.reshape(-1), A_eq=A_eq, b_eq=np.concatenate([p, q]), bounds=(0, None), method="highs")
    return res.fun


def ex16(rng, n=5, n_act=2, gamma=0.9, n_mdps=3, iters=150):
    """Bisimulation metric (Eq. 15.29a) by fixed-point iteration; check |v*(s) - v*(t)| <= d(s, t)."""
    worst_ratio, worst_gap = 0.0, -np.inf
    for m in range(n_mdps):
        r = rng.random((n, n_act))
        P = rng.dirichlet(np.full(n, 0.5), size=(n, n_act))           # P[s, a, s']
        d = np.zeros((n, n))
        for _ in range(iters):
            d_new = np.zeros((n, n))
            for s_ in range(n):
                for t_ in range(s_ + 1, n):
                    d_new[s_, t_] = d_new[t_, s_] = max(
                        abs(r[s_, a] - r[t_, a]) + gamma * w1_distance(P[s_, a], P[t_, a], d) for a in range(n_act))
            delta = np.abs(d_new - d).max()
            d = d_new
            if delta < 1e-9:
                break
        v = np.zeros(n)
        for _ in range(2000):
            v = (r + gamma * P @ v).max(axis=1)
        dv = np.abs(v[:, None] - v[None, :])
        off = ~np.eye(n, dtype=bool)
        worst_ratio = max(worst_ratio, (dv[off] / d[off]).max())
        worst_gap = max(worst_gap, (dv - d)[off].max())
    print(f"[Ex 16] bisimulation metric on {n_mdps} random MDPs ({n} states, {n_act} actions, gamma = {gamma}): "
          f"max over state pairs of |v*(s) - v*(t)| / d(s, t) = {worst_ratio:.3f} (<= 1), "
          f"max of |v*(s) - v*(t)| - d(s, t) = {worst_gap:.3f}")


def ex19(gamma=0.9):
    """Reward machine 'visit cell 0, then cell 3' on a 4-cell corridor; product MDP vs memoryless policies."""
    from scipy.optimize import minimize
    n = 4                                           # cells 0..3, start 1; actions 0 = left, 1 = right

    def move(c, a):
        return max(0, c - 1) if a == 0 else min(n - 1, c + 1)

    # product states (c, u), u = 0 (0 not yet visited) or 1 (visited); reaching 3 with u = 1 pays 1 and ends
    def value(pR):                                  # pR[c, u] = probability of "right"
        idx = {(c, u): i for i, (c, u) in enumerate((c, u) for u in (0, 1) for c in range(n))}
        A = np.eye(2 * n)
        b = np.zeros(2 * n)
        for (c, u), i in idx.items():
            for a, pa in ((0, 1 - pR[c, u]), (1, pR[c, u])):
                c2 = move(c, a)
                u2 = 1 if (u == 1 or c2 == 0) else 0
                if u2 == 1 and c2 == n - 1:
                    b[i] += pa                      # reward 1, episode ends
                else:
                    A[i, idx[(c2, u2)]] -= gamma * pa
        return np.linalg.solve(A, b)[idx[(1, 0)]]

    import itertools
    best_prod = max(value(np.array(pol, dtype=float).reshape(n, 2)) for pol in itertools.product((0, 1), repeat=2 * n))
    best_det = max(value(np.repeat(np.array(pol, dtype=float)[:, None], 2, axis=1))
                   for pol in itertools.product((0, 1), repeat=n))
    from scipy.special import expit
    f = lambda x: -value(np.repeat(expit(x)[:, None], 2, axis=1))
    rng = np.random.default_rng(19)
    best_sto = max(-minimize(f, rng.normal(size=n) * 2, method="Nelder-Mead",
                             options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 20000}).fun for _ in range(20))
    uniform = value(np.full((n, 2), 0.5))
    print(f"[Ex 19] reward machine 'visit 0 then 3', start in cell 1, gamma = {gamma}: optimal value on the "
          f"product MDP {best_prod:.4f} (= gamma^3); best deterministic memoryless policy {best_det:.4f}; best "
          f"stochastic memoryless policy {best_sto:.4f}; uniform random policy {uniform:.4f}")


def ex20(gamma=0.9):
    """Epistemic POMDP: corridor -2..2, start 0, goal at -2 (context L) or +2 (context R), equally likely."""
    from scipy.optimize import minimize
    cells = [-2, -1, 0, 1, 2]

    def value_ctx(pR, goal):
        idx = {c: i for i, c in enumerate(cells)}
        A = np.eye(5)
        b = np.zeros(5)
        for c in cells:
            if c == goal:
                continue                            # absorbing (episode over), value 0
            for step, pa in ((-1, 1 - pR[idx[c]]), (1, pR[idx[c]])):
                c2 = min(2, max(-2, c + step))
                if c2 == goal:
                    b[idx[c]] += pa
                else:
                    A[idx[c], idx[c2]] -= gamma * pa
        return np.linalg.solve(A, b)[idx[0]]

    def value(pR):
        return 0.5 * value_ctx(pR, -2) + 0.5 * value_ctx(pR, 2)

    import itertools
    best_det = max(value(np.array(p_, dtype=float)) for p_ in itertools.product((0, 1), repeat=5))
    from scipy.special import expit
    f = lambda x: -value(expit(x))
    rng = np.random.default_rng(20)
    runs = [minimize(f, rng.normal(size=5) * 2, method="Nelder-Mead",
                     options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 40000}) for _ in range(30)]
    best = min(runs, key=lambda r_: r_.fun)
    p_best = expit(best.x)
    bayes = 0.5 * gamma + 0.5 * gamma ** 5
    print(f"[Ex 20] epistemic corridor (gamma = {gamma}): Bayes-optimal (history-dependent) value {bayes:.4f}; "
          f"per-context optimum {gamma:.4f}; best deterministic memoryless {best_det:.4f}; best stochastic "
          f"memoryless {-best.fun:.4f} with P(right | cell -2..2) = " + ", ".join(f"{x:.3f}" for x in p_best))


def ex21():
    """PLR replay distribution by hand vs procgen_lite.PLRSampler.probs()."""
    import procgen_lite as pg
    scores = [0.5, 0.2, 0.05, 0.3]
    last = [44, 35, 15, 41]                         # episode count when each level was last played
    for beta in (1.0, 0.1):
        smp = pg.PLRSampler(levels=[0, 1, 2, 3], rng=np.random.default_rng(0), beta=beta, lam=0.1)
        smp.score = {i: s_ for i, s_ in enumerate(scores)}
        smp.last = {i: l_ for i, l_ in enumerate(last)}
        smp.episodes = 45
        _, p = smp.probs()
        print(f"[Ex 21] PLR, scores {scores}, staleness {[45 - l_ for l_ in last]}, beta = {beta}, lambda = 0.1: "
              f"P_replay = " + ", ".join(f"{x:.3f}" for x in p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)
    print(f"Chapter 15 exercise checks | seed=0 quick={args.quick}")
    ex2()
    ex3()
    ex4()
    ex7(rng, 20_000 if args.quick else 1_000_000)
    ex9()
    ex10()
    ex11()
    ex12()
    ex13(20_000 if args.quick else 500_000, 1 if args.quick else 3)
    ex14()
    ex15(np.random.default_rng(15))
    ex16(np.random.default_rng(16), n_mdps=1 if args.quick else 3)
    ex19()
    ex20()
    ex21()
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
