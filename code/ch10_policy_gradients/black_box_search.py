"""Black-box policy search: finite differences, SPSA, evolution strategies, CEM and ARS.

Chapter 10, Section 15.  NumPy + Gymnasium.  (Part A also runs the PyTorch REINFORCE of
reinforce_cartpole.py, the network of Section 6.4, as a likelihood-ratio reference.)

Part A  CartPole-v1 with a deterministic LINEAR policy a = 1[theta^T s > 0] (d = 4 parameters).
        J(theta) is a black box: each evaluation is the return of one episode.  All black-box
        methods start at theta = 0, have a budget of 640 training episodes, and use 5 seeds:
          fd    central finite differences, 2d = 8 episodes per gradient estimate, Adam step
          spsa  simultaneous perturbation (Spall, 1992): 2 episodes per estimate, Adam step
          es    antithetic Gaussian ES with centred-rank fitness shaping (Algorithm 10.9),
                N = 8 pairs (16 episodes) per estimate, Adam step
          cem   cross-entropy method over theta: diagonal Gaussian, 16 samples, 4 elites, plus a
                constant extra variance so the search distribution cannot collapse (Szita & Lorincz)
          ars   Augmented Random Search V2 (Mania, Guy & Recht, 2018; Algorithm 10.10): running state
                normalization, top-b directions, step divided by the std of the kept returns
        Pairs of perturbed evaluations share an environment seed (common random numbers).
        Two likelihood-ratio references use STOCHASTIC policies and one episode per update:
          reinforce-linear  logistic policy pi(right|s) = sigmoid(theta^T s), the same 4 parameters,
                            reward-to-go with a learned linear baseline, Adam (lr 0.1, the best of
                            {0.01, 0.03, 0.1, 0.3} in a pilot run)
          reinforce-mlp     the 2x64 tanh MLP with a learned baseline of reinforce_cartpole.py
        Every 16 training episodes the current policy (the mean, for CEM) is evaluated on 10 fixed
        evaluation episodes, which do not count towards the budget.  Stochastic policies are
        evaluated stochastically.  Control: the fraction of random theta ~ N(0, I) that already
        average >= 475 on 5 fresh episodes.  The black-box settings were set once (perturbation scale
        0.1, Adam lr 0.05) and not tuned per method.

Part B  A family of linear-quadratic problems with d = n^2 parameters (n = 2, 4, 8, 16):
            S_{t+1} = F S_t + G A_t,  A_t = -K S_t,  R_{t+1} = -(S_t' C_s S_t + A_t' C_a A_t),
            gamma = 0.9,  S_0 ~ N(0, I/n),  F = 0.9 x (random orthogonal),  G = C_s = C_a = I,
        with theta = vec(K) and J(K) = -tr(P_K Sigma_0), where P_K solves the Lyapunov equation
        P = C_s + K'C_a K + gamma (F-GK)' P (F-GK).  J and its gradient
            grad_K J = -2 [ (C_a + gamma G'P_K G) K - gamma G'P_K F ] Sigma_K,
            Sigma_K = Sigma_0 + gamma (F-GK) Sigma_K (F-GK)'
        are EXACT (Lyapunov sums by Smith's doubling), so the only randomness is the search noise
        xi.  By the rotational symmetry of this family, exact gradient ascent with step c*n behaves
        identically for every n: any growth with d below comes from the estimator.
          B1  tr Cov(g_hat) / ||grad J||^2 at K = 0 for the plain, forward-difference and antithetic
              estimators, sigma = 0.01, against d (and against sigma at d = 64)
          B2  stochastic gradient ascent K <- K + alpha g_hat with 16 J-evaluations per iteration
              (antithetic: 8 pairs; forward: 15 perturbations + J(theta); plain: 16 perturbations),
              alpha = best of a factor-2 grid on 6 separate tuning seeds (3 for plain); iterations to reach 1%
              suboptimality, (J* - J)/(J* - J(0)) <= 0.01, against d.

Outputs (full mode): figures/black_box_cartpole.png, figures/black_box_lq.png
Run:  python code/ch10_policy_gradients/black_box_search.py [--quick]
"""
from __future__ import annotations

import argparse
import os
import time

import gymnasium as gym
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
SOLVED = 475.0
EVAL_EVERY = 16                       # a multiple of every method's episodes per iteration (2, 8, 16)
EVAL_SEEDS = [10_000_000 + k for k in range(10)]

# ----------------------------------------------------------------------------------------------
# Part A: CartPole-v1 with a linear policy
# ----------------------------------------------------------------------------------------------


class Adam:
    """Adam for gradient ASCENT on a NumPy vector (as OpenAI-ES uses)."""

    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = self.v = 0.0
        self.t = 0

    def step(self, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh, vh = self.m / (1 - self.b1 ** self.t), self.v / (1 - self.b2 ** self.t)
        return self.lr * mh / (np.sqrt(vh) + self.eps)


class CartPole:
    """One environment; counts TRAINING episodes and steps (evaluation calls pass count=False)."""

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.episodes = 0
        self.steps = 0

    def linear_return(self, theta, seed, mu=None, sd=None, collect=None, count=True):
        """Return of one episode of the deterministic policy a = 1[theta^T x > 0], x = (s - mu)/sd."""
        obs, _ = self.env.reset(seed=int(seed))
        G, T = 0.0, 0
        while True:
            if collect is not None:
                collect.append(obs)
            x = obs if mu is None else (obs - mu) / sd
            obs, r, term, trunc, _ = self.env.step(int(theta @ x > 0))
            G += r
            T += 1
            if term or trunc:
                break
        if count:
            self.episodes += 1
            self.steps += T
        return G

    def logistic_episode(self, theta, seed, rng, count=True):
        """One episode of the stochastic policy pi(right|s) = sigmoid(theta^T s)."""
        obs, _ = self.env.reset(seed=int(seed))
        O, A, R = [], [], []
        while True:
            p = 1.0 / (1.0 + np.exp(-theta @ obs))
            a = int(rng.random() < p)
            O.append(obs)
            A.append(a)
            obs, r, term, trunc, _ = self.env.step(a)
            R.append(r)
            if term or trunc:
                break
        if count:
            self.episodes += 1
            self.steps += len(R)
        return np.array(O), np.array(A), np.array(R)


def centred_ranks(x):
    """Fitness shaping: ranks scaled to [-0.5, 0.5] (they sum to zero, so they act as a baseline)."""
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


BB_METHODS = ["fd", "spsa", "es", "cem", "ars"]
HP = dict(fd=dict(h=0.1, lr=0.05), spsa=dict(c=0.1, lr=0.05), es=dict(N=8, sigma=0.1, lr=0.05),
          cem=dict(N=16, K=4, s0=1.0, extra_var=0.01), ars=dict(N=8, b=4, nu=0.1, alpha=0.05))


def run_black_box(method, seed, budget, cp):
    """Train one black-box method; return (episodes, env steps, eval return) at every checkpoint."""
    rng = np.random.default_rng(seed)
    hp = HP[method]
    D = 4
    theta = np.zeros(D)
    opt = Adam(hp.get("lr", 0.05))
    m, s = np.zeros(D), np.full(D, hp.get("s0", 1.0))             # CEM search distribution
    mu, sd, n_obs, M2 = np.zeros(D), np.ones(D), 0, np.zeros(D)     # ARS state statistics
    log = []

    def evaluate():
        if method == "cem":
            return np.mean([cp.linear_return(m, e, count=False) for e in EVAL_SEEDS])
        if method == "ars":
            return np.mean([cp.linear_return(theta, e, mu, sd, count=False) for e in EVAL_SEEDS])
        return np.mean([cp.linear_return(theta, e, count=False) for e in EVAL_SEEDS])

    cp.episodes = cp.steps = 0
    while True:
        if cp.episodes % EVAL_EVERY == 0:
            log.append((cp.episodes, cp.steps, evaluate()))
        if cp.episodes >= budget:
            break
        if method == "fd":                         # central differences, one coordinate at a time
            g = np.zeros(D)
            for i in range(D):
                e = np.zeros(D)
                e[i] = hp["h"]
                sd_i = rng.integers(2**31)             # same env seed for +h and -h
                g[i] = (cp.linear_return(theta + e, sd_i) - cp.linear_return(theta - e, sd_i)) / (2 * hp["h"])
            theta = theta + opt.step(g)
        elif method == "spsa":                     # all coordinates at once along Delta in {-1,+1}^d
            delta = rng.choice([-1.0, 1.0], D)
            sd_i = rng.integers(2**31)
            c = hp["c"]
            g = (cp.linear_return(theta + c * delta, sd_i) - cp.linear_return(theta - c * delta, sd_i)) / (2 * c) * delta
            theta = theta + opt.step(g)
        elif method == "es":                       # Algorithm 10.9
            N, sig = hp["N"], hp["sigma"]
            xi = rng.standard_normal((N, D))
            Fp, Fm = np.zeros(N), np.zeros(N)
            for k in range(N):
                sd_k = rng.integers(2**31)
                Fp[k] = cp.linear_return(theta + sig * xi[k], sd_k)
                Fm[k] = cp.linear_return(theta - sig * xi[k], sd_k)
            u = centred_ranks(np.concatenate([Fp, Fm]))
            g = ((u[:N] - u[N:])[:, None] * xi).sum(0) / (2 * N * sig)
            theta = theta + opt.step(g)
        elif method == "cem":                      # elite refit, as in Algorithm 13.2, over theta
            X = m + s * rng.standard_normal((hp["N"], D))
            F = np.array([cp.linear_return(x, rng.integers(2**31)) for x in X])
            E = X[np.argsort(-F, kind="stable")[:hp["K"]]]
            m, s = E.mean(0), np.sqrt(E.var(0) + hp["extra_var"])
        elif method == "ars":                      # Algorithm 10.10 (ARS V2-t)
            N, b, nu = hp["N"], hp["b"], hp["nu"]
            delta = rng.standard_normal((N, D))
            Fp, Fm, visited = np.zeros(N), np.zeros(N), []
            for k in range(N):
                sd_k = rng.integers(2**31)
                Fp[k] = cp.linear_return(theta + nu * delta[k], sd_k, mu, sd, visited)
                Fm[k] = cp.linear_return(theta - nu * delta[k], sd_k, mu, sd, visited)
            top = np.argsort(-np.maximum(Fp, Fm), kind="stable")[:b]
            sig_R = np.concatenate([Fp[top], Fm[top]]).std()
            if sig_R > 0:
                theta = theta + hp["alpha"] / (b * sig_R) * ((Fp[top] - Fm[top])[:, None] * delta[top]).sum(0)
            for o in visited:                      # Welford running mean / variance of the states
                n_obs += 1
                dlt = o - mu
                mu = mu + dlt / n_obs
                M2 = M2 + dlt * (o - mu)
            sd = np.sqrt(np.maximum(M2 / max(n_obs - 1, 1), 1e-8))
    return np.array(log)


def run_reinforce_linear(seed, budget, cp, lr=0.1, lr_v=0.05, gamma=0.99, t_max=500):
    """REINFORCE with reward-to-go and a learned linear baseline; logistic policy, 4 parameters."""
    rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(123_456 + seed)
    theta, w = np.zeros(4), np.zeros(5)
    opt, opt_v = Adam(lr), Adam(lr_v)
    log = []
    cp.episodes = cp.steps = 0
    while True:
        if cp.episodes % EVAL_EVERY == 0:
            ev = np.mean([cp.logistic_episode(theta, e, eval_rng, count=False)[2].sum() for e in EVAL_SEEDS])
            log.append((cp.episodes, cp.steps, ev))
        if cp.episodes >= budget:
            break
        O, A, R = cp.logistic_episode(theta, rng.integers(2**31), rng)
        G = np.zeros(len(R))
        run = 0.0
        for t in range(len(R) - 1, -1, -1):
            run = R[t] + gamma * run
            G[t] = run
        X = np.c_[O, np.ones(len(O))]
        adv = G - X @ w
        p = 1.0 / (1.0 + np.exp(-O @ theta))
        theta = theta + opt.step(((adv * (A - p))[:, None] * O).sum(0) / t_max)   # score (a - p) s
        w = w + opt_v.step((adv[:, None] * X).mean(0))
    return np.array(log)


def run_reinforce_mlp(seed, budget, cp):
    """The REINFORCE-with-baseline MLP of reinforce_cartpole.py (Section 6.4), evaluated at checkpoints."""
    import reinforce_cartpole as rc               # PyTorch; sets torch.set_num_threads(1)
    ckpts = set(range(0, budget + 1, EVAL_EVERY))
    returns, _, saved, _ = rc.train("baseline", seed, budget + 1, 0.99, 1e-3, 5e-3, checkpoints=ckpts)
    eval_rng = np.random.default_rng(654_321 + seed)
    policy = rc.mlp(4, 2)
    log = []
    for ep in sorted(saved):
        policy.load_state_dict(saved[ep][0])
        actor = rc.NumpyPolicy(policy)
        rets = []
        for e in EVAL_SEEDS:
            obs, _ = cp.env.reset(seed=e)
            G = 0.0
            while True:
                p = actor.probs(obs.astype(np.float64))
                obs, r, term, trunc, _ = cp.env.step(int(eval_rng.random() < p[1]))
                G += r
                if term or trunc:
                    break
            rets.append(G)
        log.append((ep, returns[:ep].sum(), np.mean(rets)))
    return np.array(log)


def summarize_a(name, logs):
    """Episodes and steps to the first checkpoint >= 475, stability afterwards, final return."""
    firsts, steps, stab, final = [], [], [], []
    for L in logs:
        ok = L[:, 2] >= SOLVED
        if ok.any():
            i = int(np.argmax(ok))
            firsts.append(int(L[i, 0]))
            steps.append(int(L[i, 1]))
            stab.append(ok[i:].mean())
        else:
            firsts.append(None)
            steps.append(None)
            stab.append(0.0)
        final.append(L[-5:, 2].mean())
    reached = [f for f in firsts if f is not None]
    med = f"{np.median(reached):5.0f}" if len(reached) == len(firsts) else "  n/a"
    meds = f"{np.median([s for s in steps if s is not None]) / 1e3:6.1f}k" if len(reached) == len(firsts) else "    n/a"
    print(f"  {name:17s} | {med} | {str(firsts):28s} | {meds} | {np.mean(stab):6.2f} | "
          f"{np.mean(final):6.1f}  {np.round(final).astype(int).tolist()}")
    return firsts


def part_a(quick, seed0):
    budget = 96 if quick else 640
    n_seeds = 2 if quick else 5
    n_random = 40 if quick else 500
    print(f"\n=== Part A: CartPole-v1, linear policy (d = 4), budget {budget} training episodes, "
          f"{n_seeds} seeds, evaluation every {EVAL_EVERY} episodes on {len(EVAL_SEEDS)} fixed episodes ===")
    print("  black-box settings:", {k: v for k, v in HP.items()})
    cp = CartPole()
    t0 = time.time()
    rng = np.random.default_rng(seed0 + 777)
    scores = []
    for _ in range(n_random):                      # each theta gets 5 fresh start states
        th = rng.standard_normal(4)
        scores.append(np.mean([cp.linear_return(th, rng.integers(2**31), count=False) for _ in range(5)]))
    scores = np.array(scores)
    print(f"  control: {np.mean(scores >= SOLVED):.3f} of {n_random} random theta ~ N(0, I) average >= {SOLVED:.0f} "
          f"on 5 episodes ({np.mean(scores >= 200):.3f} reach 200); theta = 0 scores "
          f"{np.mean([cp.linear_return(np.zeros(4), e, count=False) for e in EVAL_SEEDS]):.1f}")
    logs = {}
    for meth in BB_METHODS + ["reinforce-linear", "reinforce-mlp"]:
        t1 = time.time()
        if meth in BB_METHODS:
            logs[meth] = [run_black_box(meth, seed0 + i, budget, cp) for i in range(n_seeds)]
        elif meth == "reinforce-linear":
            logs[meth] = [run_reinforce_linear(seed0 + i, budget, cp) for i in range(n_seeds)]
        else:
            logs[meth] = [run_reinforce_mlp(seed0 + i, budget, cp) for i in range(n_seeds)]
        print(f"  {meth} done ({time.time() - t1:.0f}s)")
    print(f"\n  method            | median episodes to eval >= {SOLVED:.0f} | per seed | median env steps to it | "
          "frac. later checkpoints >= 475 | final eval return (mean of last 5 checkpoints; per seed)")
    for meth, L in logs.items():
        summarize_a(meth, L)
    print(f"  Part A time {time.time() - t0:.0f}s")
    return logs, scores


# ----------------------------------------------------------------------------------------------
# Part B: linear-quadratic problems with exact J
# ----------------------------------------------------------------------------------------------

GAMMA_LQ = 0.9


def make_lq(n, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    I = np.eye(n)
    return dict(n=n, F=0.9 * Q, G=I, Cs=I, Ca=I, S0=I / n)


def smith(L, Q, iters=20):
    """Batched P = sum_{t>=0} (L')^t Q L^t by doubling; converged where L^(2^iters) is ~0."""
    P, Lk = Q.copy(), L.copy()
    with np.errstate(all="ignore"):
        for _ in range(iters):
            P = P + np.swapaxes(Lk, -1, -2) @ P @ Lk
            Lk = Lk @ Lk
        ok = np.isfinite(P).all((-1, -2)) & (np.abs(np.nan_to_num(Lk, nan=1.0, posinf=1.0, neginf=1.0)).max((-1, -2)) < 1e-10)
    return P, ok


def lq_J(lq, K):
    """Exact J for a batch of gains K (B, n, n); -inf where the closed loop is unstable."""
    L = np.sqrt(GAMMA_LQ) * (lq["F"] - lq["G"] @ K)
    Q = lq["Cs"] + np.swapaxes(K, -1, -2) @ lq["Ca"] @ K
    P, ok = smith(L, Q)
    J = -np.einsum("bij,ji->b", np.nan_to_num(P), lq["S0"])
    J[~ok] = -np.inf
    return J


def lq_grad(lq, K):
    F, G, Ca, S0 = lq["F"], lq["G"], lq["Ca"], lq["S0"]
    L = np.sqrt(GAMMA_LQ) * (F - G @ K)
    P, _ = smith(L[None], (lq["Cs"] + K.T @ Ca @ K)[None])
    Sig, _ = smith(L.T[None], S0[None])
    P, Sig = P[0], Sig[0]
    return -2 * ((Ca + GAMMA_LQ * G.T @ P @ G) @ K - GAMMA_LQ * G.T @ P @ F) @ Sig


def lq_opt(lq):
    F, G, Cs, Ca, S0 = lq["F"], lq["G"], lq["Cs"], lq["Ca"], lq["S0"]
    P = Cs.copy()
    for _ in range(10_000):
        Pn = Cs + GAMMA_LQ * F.T @ P @ F - GAMMA_LQ**2 * F.T @ P @ G @ np.linalg.solve(Ca + GAMMA_LQ * G.T @ P @ G, G.T @ P @ F)
        if np.abs(Pn - P).max() < 1e-14:
            P = Pn
            break
        P = Pn
    K = GAMMA_LQ * np.linalg.solve(Ca + GAMMA_LQ * G.T @ P @ G, G.T @ P @ F)
    return -np.trace(P @ S0), K


def estimator_samples(lq, K, sig, M, rng, chunk=2000):
    """M one-sample (plain, forward) and one-pair (antithetic) gradient estimates, flattened."""
    n = lq["n"]
    J0 = lq_J(lq, K[None])[0]
    out = {"plain": [], "forward": [], "antithetic": []}
    n_bad = 0
    for i in range(0, M, chunk):
        xi = rng.standard_normal((min(chunk, M - i), n, n))
        Jp, Jm = lq_J(lq, K + sig * xi), lq_J(lq, K - sig * xi)
        n_bad += int((~np.isfinite(Jp)).sum() + (~np.isfinite(Jm)).sum())
        X = xi.reshape(len(xi), -1)
        out["plain"].append(Jp[:, None] * X / sig)
        out["forward"].append((Jp - J0)[:, None] * X / sig)
        out["antithetic"].append(((Jp - Jm) / (2 * sig))[:, None] * X)
    return {k: np.concatenate(v) for k, v in out.items()}, J0, n_bad


def es_ascent(lq, method, alpha, T, seed, sig=0.01, n_eval=16, stop_at=None):
    """Gradient ascent on J with an ES estimator; returns the suboptimality curve (inf = diverged)."""
    n = lq["n"]
    rng = np.random.default_rng(seed)
    K = np.zeros((n, n))
    J0 = lq_J(lq, K[None])[0]
    Jstar = lq["Jstar"]
    sub = np.full(T + 1, np.inf)
    for t in range(T + 1):
        Jk = lq_J(lq, K[None])[0]
        if not np.isfinite(Jk):
            return sub
        sub[t] = (Jstar - Jk) / (Jstar - J0)
        if t == T or (stop_at is not None and sub[t] <= stop_at):
            sub[t + 1:] = sub[t]
            return sub
        if method == "exact":
            g = lq_grad(lq, K)
        else:
            if method == "antithetic":
                xi = rng.standard_normal((n_eval // 2, n, n))
                w = (lq_J(lq, K + sig * xi) - lq_J(lq, K - sig * xi)) / (2 * sig)
            elif method == "forward":
                xi = rng.standard_normal((n_eval - 1, n, n))
                w = (lq_J(lq, K + sig * xi) - Jk) / sig
            else:                                            # plain
                xi = rng.standard_normal((n_eval, n, n))
                w = lq_J(lq, K + sig * xi) / sig
            if not np.all(np.isfinite(w)):
                return sub
            g = np.einsum("b,bij->ij", w, xi) / len(xi)
        K = K + alpha * g
    return sub


def iters_to(sub, tol=0.01):
    hit = np.flatnonzero(sub <= tol)
    return int(hit[0]) if len(hit) else None


def part_b(quick, seed0):
    ns = [2, 4] if quick else [2, 4, 8, 16]
    M = 2000 if quick else 20000
    T = 300 if quick else 3000
    n_eval_seeds = 2 if quick else 5
    sig = 0.01
    print(f"\n=== Part B: linear-quadratic family, d = n^2 for n in {ns}, gamma = {GAMMA_LQ}, sigma = {sig} ===")
    t0 = time.time()
    lqs = {}
    for n in ns:
        lq = make_lq(n, seed=seed0)
        lq["Jstar"], Kstar = lq_opt(lq)
        K0 = np.zeros((n, n))
        g = lq_grad(lq, K0)
        E = np.random.default_rng(1).standard_normal((n, n))
        h = 1e-6
        fd = (lq_J(lq, (K0 + h * E)[None])[0] - lq_J(lq, (K0 - h * E)[None])[0]) / (2 * h)
        rel = abs(fd - np.sum(g * E)) / abs(np.sum(g * E))
        assert rel < 1e-6, rel
        assert np.abs(lq_grad(lq, Kstar)).max() < 1e-10
        lqs[n] = lq
        print(f"  n={n:2d} d={n * n:3d}: J(0) = {lq_J(lq, K0[None])[0]:.4f}, J* = {lq['Jstar']:.4f}, "
              f"||grad J(0)||^2 = {np.sum(g * g):.4g}; exact gradient vs finite differences: rel. err {rel:.1e}; "
              f"||grad J(K*)||_max = {np.abs(lq_grad(lq, Kstar)).max():.1e}")

    # B1: variance against d
    print(f"\n  B1. tr Cov(g_hat)/||grad J||^2 at K = 0 from M = {M} samples (one evaluation per plain/forward "
          "sample, two per antithetic pair); theory: plain ~ d J^2/(sigma^2 ||grad J||^2) + (d+1), antithetic = d+1 "
          "on a quadratic")
    print("     d |      plain   (theory) |  forward | antithetic | d+1 | rel. error of the sample mean: plain / forward / antithetic | unstable")
    b1 = {}
    rng = np.random.default_rng(seed0 + 2)
    for n in ns:
        lq = lqs[n]
        d = n * n
        g = lq_grad(lq, np.zeros((n, n))).reshape(-1)
        S, J0, n_bad = estimator_samples(lq, np.zeros((n, n)), sig, M, rng)
        g2 = np.sum(g * g)
        row = {k: v.var(0, ddof=1).sum() / g2 for k, v in S.items()}
        err = {k: np.linalg.norm(v.mean(0) - g) / np.sqrt(g2) for k, v in S.items()}
        theory = d * J0**2 / (sig**2 * g2) + d + 1
        b1[d] = (row, theory)
        print(f"  {d:4d} | {row['plain']:10.4g} ({theory:8.3g}) | {row['forward']:8.4g} | {row['antithetic']:10.4g} | "
              f"{d + 1:3d} | {err['plain']:7.3f} / {err['forward']:.3f} / {err['antithetic']:.3f} | {n_bad}")
    if 8 in lqs:
        print("\n     sigma dependence at d = 64:   sigma |     plain |  forward | antithetic")
        for s in [0.001, 0.003, 0.01]:
            S, _, n_bad = estimator_samples(lqs[8], np.zeros((8, 8)), s, M // 4, rng)
            g2 = np.sum(lq_grad(lqs[8], np.zeros((8, 8))) ** 2)
            v = {k: x.var(0, ddof=1).sum() / g2 for k, x in S.items()}
            print(f"                                  {s:6.3f} | {v['plain']:9.4g} | {v['forward']:8.4g} | {v['antithetic']:10.4g}"
                  f"{'' if n_bad == 0 else f'  ({n_bad} unstable)'}")

    # B2: convergence against d
    grids = {"exact": [0.0125 * 2**k for k in range(6)],              # step alpha = c * n
             "antithetic": [2.5e-4 * 2**k for k in range(9)],
             "forward": [2.5e-4 * 2**k for k in range(9)],
             "plain": [1e-7 * 2**k for k in range(13)]}
    n_tune = {"exact": 1, "antithetic": 6, "forward": 6, "plain": 3}
    print(f"\n  B2. ascent with 16 J-evaluations per iteration, {T} iterations, step alpha = c * n, c tuned on seeds "
          f"{seed0 + 100}-{seed0 + 105} (plain: {seed0 + 100}-{seed0 + 102}) over a factor-2 grid (worst tuning seed decides), then {n_eval_seeds} evaluation seeds")
    print("     d | method     |        c | iterations to 1% (median; per seed)      | final suboptimality (median)")
    b2 = {}
    for n in ns:
        lq = lqs[n]
        for meth in ["exact", "antithetic", "forward", "plain"]:
            best = None
            Tm = min(T, 300) if meth == "exact" else T
            for c in grids[meth]:
                # tuning seeds 100.. (never used for evaluation); stop a run once it reaches 1%.  Six seeds
                # (three for plain, whose runs never stop early) so that the chosen step is not one that
                # diverges on a sizeable fraction of seeds.
                subs = [es_ascent(lq, meth, c * n, Tm, seed0 + 100 + j, sig=sig, stop_at=0.01)
                        for j in range(n_tune[meth])]
                its = [iters_to(s) for s in subs]
                fails = sum(i is None for i in its)
                if fails == 0:
                    key = (0, 0, max(its))                 # all reached 1%: the worst seed decides
                else:                                      # otherwise: fewest failures, no divergence, lowest median
                    key = (fails, sum(not np.isfinite(s[-1]) for s in subs), float(np.median([s[-1] for s in subs])))
                if best is None or key < best[0]:
                    best = (key, c)
            c = best[1]
            subs = np.array([es_ascent(lq, meth, c * n, Tm, seed0 + s, sig=sig) for s in range(n_eval_seeds)])
            its = [iters_to(s) for s in subs]
            med_it = np.median(its) if all(i is not None for i in its) else None
            b2[(n, meth)] = (c, subs, its)
            print(f"  {n * n:4d} | {meth:10s} | {c:8.3g} | {str(med_it):>6s}  {str(its):32s} | {np.median(subs[:, -1]):.3g}"
                  f"{'' if np.all(np.isfinite(subs[:, -1])) else '  (some runs diverged)'}")
    print(f"  Part B time {time.time() - t0:.0f}s")
    return lqs, b1, b2, T


# ----------------------------------------------------------------------------------------------


def make_figures(logs, b1, b2, ns, T):
    from plot_style import setup, C, GREY, INK
    plt = setup()
    os.makedirs(FIG_DIR, exist_ok=True)
    styles = {"fd": ("finite differences", C[0], "-", "o"), "spsa": ("SPSA", C[1], "-", "s"),
              "es": ("antithetic ES + ranks", C[2], "-", "^"), "cem": ("CEM", C[3], "-", "D"),
              "ars": ("ARS V2-t", C[4], "-", "v"), "reinforce-linear": ("REINFORCE, linear (stochastic)", C[6], "--", None),
              "reinforce-mlp": ("REINFORCE, 2x64 MLP (Sec. 6.4)", GREY, ":", None)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for meth, L in logs.items():
        lab, col, ls, mk = styles[meth]
        A = np.array(L)                                   # (seeds, checkpoints, 3)
        ax = axes[0]
        ax.plot(A[0, :, 0], A[:, :, 2].mean(0), color=col, ls=ls, marker=mk, ms=3.5, markevery=4, label=lab)
        ax.fill_between(A[0, :, 0], A[:, :, 2].min(0), A[:, :, 2].max(0), color=col, alpha=0.10)
        axes[1].plot(A[:, 1:, 1].mean(0) / 1e3, A[:, 1:, 2].mean(0), color=col, ls=ls, marker=mk,
                     ms=3.5, markevery=4, label=lab)        # from the first checkpoint after step 0
    axes[0].set_xlabel("training episodes")
    axes[0].set_ylabel(f"evaluation return ({len(EVAL_SEEDS)} episodes)")
    axes[0].set_title(f"CartPole-v1: mean over {len(logs['fd'])} seeds (band = min/max)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("training environment steps (thousands; seed mean)")
    axes[1].set_title("Same runs against environment steps")
    for ax in axes:
        ax.axhline(SOLVED, color=GREY, lw=0.8, ls="--")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(FIG_DIR, "black_box_cartpole.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ds = sorted(b1)
    ax = axes[0]
    for k, (lab, col, mk) in {"plain": ("plain $J(\\theta+\\sigma\\xi)\\xi/\\sigma$", C[1], "s"),
                             "forward": ("forward (baseline $J(\\theta)$)", C[3], "D"),
                             "antithetic": ("antithetic (10.38)", C[2], "o")}.items():
        ax.plot(ds, [b1[d][0][k] for d in ds], color=col, marker=mk, label=lab)
    ax.plot(ds, [b1[d][1] for d in ds], color=C[1], ls=":", lw=1.2, label="$dJ^2/(\\sigma^2\\|\\nabla J\\|^2)+d+1$")
    ax.plot(ds, [d + 1 for d in ds], color=C[2], ls=":", lw=1.2, label="$d+1$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("$d=\\dim\\theta$")
    ax.set_ylabel("tr Cov$(\\hat g)$ / $\\|\\nabla J\\|^2$")
    ax.set_title("Per-sample gradient variance ($\\sigma=0.01$)")
    ax.legend(fontsize=8)
    ax = axes[1]
    for j, n in enumerate(ns):
        c, subs, _ = b2[(n, "antithetic")]
        ax.plot(np.arange(subs.shape[1]), np.median(subs, 0), color=C[j], label=f"antithetic, d={n * n}")
        c, subs, _ = b2[(n, "plain")]
        sub_p = np.where(np.isfinite(subs), subs, np.nan)
        ax.plot(np.arange(sub_p.shape[1]), np.nanmedian(sub_p, 0), color=C[j], ls="--", lw=1.2,
                label=f"plain, d={n * n}")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(1e-4, 2)
    ax.set_xlabel("iteration (16 evaluations of $J$ each)")
    ax.set_ylabel("$(J^\\ast-J)/(J^\\ast-J(0))$, median over seeds")
    ax.set_title("Convergence, best step size for each")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax = axes[2]
    dd = [n * n for n in ns]
    for meth, col, mk in [("exact", INK, "x"), ("antithetic", C[2], "o"), ("forward", C[3], "D")]:
        its = [np.median(b2[(n, meth)][2]) if all(i is not None for i in b2[(n, meth)][2]) else np.nan for n in ns]
        ax.plot(dd, its, color=col, marker=mk, label=meth if meth != "exact" else "exact gradient")
    ref = np.median(b2[(ns[-1], "antithetic")][2])
    ax.plot(dd, [ref * d / dd[-1] for d in dd], color=GREY, ls=":", lw=1.2, label="slope 1")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("$d=\\dim\\theta$")
    ax.set_ylabel("iterations to 1% suboptimality (median)")
    ax.set_title("Cost of a gradient-free gradient grows with $d$\n(plain: median never reaches 1% in 3000 iterations)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "black_box_lq.png"))
    plt.close(fig)
    print(f"figures written to {FIG_DIR}/black_box_cartpole.png and black_box_lq.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    print(f"seed={args.seed}  mode={'quick' if args.quick else 'full'}")
    logs, _ = part_a(args.quick, args.seed)
    lqs, b1, b2, T = part_b(args.quick, args.seed)
    print(f"\nTotal time {time.time() - t0:.0f}s")
    if not args.quick:
        make_figures(logs, b1, b2, sorted(lqs), T)


if __name__ == "__main__":
    main()
