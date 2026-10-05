"""Shared bandit environments, agents and a vectorized experiment runner (Chapter 02).

Every agent here simulates MANY independent bandit runs at once: arrays have shape
[runs, k] and row r belongs to run r.  The per-row logic is exactly the sequential
pseudocode in the chapter; vectorizing over runs just makes 2000-run experiments
take seconds instead of minutes.  `simple_bandit_loop` below is the same algorithm
written as a plain single-run Python loop (Chapter 02, Algorithm 2.1), and the
self-test at the bottom checks that the two agree.

Agent interface (mirrors the pseudocode boxes):
    agent.reset(runs, rng)      # initialise Q, N, preferences, posteriors ...
    a = agent.act(t)            # choose A_t for every run (t = 1, 2, ...)
    agent.update(a, r)          # learn from (A_t, R_t)

Notation (Chapter 02 / NOTATION.md): k arms, q_star[a] true means, Q[a] estimates,
N[a] pull counts, R_t the reward that follows A_t (bandit convention, no state).

Run:  python code/ch02_multi_armed_bandits/bandits.py [--quick]   (self-test only)
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# Validated categorical palette (fixed order, never cycled) + line styles as a
# secondary encoding so that curves stay distinguishable without colour.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1)), (0, (1, 1)), (0, (8, 2))]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd7"


def setup_matplotlib():
    """Agg backend + a quiet, readable style shared by every figure of the chapter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 110, "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "legend.frameon": False,
    })
    return plt


def style(i: int) -> dict:
    """Colour + line style for series i (fixed order)."""
    return {"color": PALETTE[i % len(PALETTE)], "linestyle": LINESTYLES[i % len(LINESTYLES)]}


# ----------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------

def argmax_random_ties(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row-wise argmax with ties broken uniformly at random.

    Breaking ties at random matters: with Q_1 = 0 for every arm, np.argmax would
    always pick arm 0 and a greedy agent would never even look at the others.
    """
    is_max = X == X.max(axis=1, keepdims=True)
    return np.argmax(np.where(is_max, rng.random(X.shape), -1.0), axis=1)


def softmax(H: np.ndarray) -> np.ndarray:
    """Row-wise softmax pi(a) = exp(H_a) / sum_b exp(H_b) (Eq. 8.1), numerically stable."""
    Z = np.exp(H - H.max(axis=1, keepdims=True))
    return Z / Z.sum(axis=1, keepdims=True)


def sample_categorical(P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One draw per row from the categorical distribution in that row (inverse CDF)."""
    u = rng.random((P.shape[0], 1))
    return np.minimum((P.cumsum(axis=1) < u).sum(axis=1), P.shape[1] - 1)


def bernoulli_kl(p, q, eps: float = 1e-12):
    """kl(p, q) = p log(p/q) + (1-p) log((1-p)/(1-q)), the KL between Bernoulli(p) and Bernoulli(q)."""
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


# ----------------------------------------------------------------------------------
# Environments (vectorized over runs)
# ----------------------------------------------------------------------------------

class GaussianBandit:
    """k arms with R ~ Normal(q_star[a], reward_std^2).  The S&B 10-armed testbed uses
    q_star[a] ~ Normal(mean_offset, 1) per run.  If walk_std > 0 the means take
    independent random walks AFTER every pull (the nonstationary problem of §4)."""

    def __init__(self, q_star: np.ndarray, rng: np.random.Generator,
                 reward_std: float = 1.0, walk_std: float = 0.0):
        self.q_star = np.array(q_star, dtype=float)
        self.runs, self.k = self.q_star.shape
        self.rng, self.reward_std, self.walk_std = rng, reward_std, walk_std

    @classmethod
    def testbed(cls, runs: int, k: int, rng: np.random.Generator, mean_offset: float = 0.0, **kw):
        return cls(rng.normal(mean_offset, 1.0, size=(runs, k)), rng, **kw)

    def pull(self, a: np.ndarray) -> np.ndarray:
        r = self.q_star[np.arange(self.runs), a] + self.reward_std * self.rng.normal(size=self.runs)
        if self.walk_std > 0:
            self.q_star += self.walk_std * self.rng.normal(size=self.q_star.shape)
        return r


class BernoulliBandit:
    """k arms with R ~ Bernoulli(q_star[a]) in {0, 1}.  probs has shape [k] (same
    instance in every run) or [runs, k]."""

    def __init__(self, probs: np.ndarray, runs: int, rng: np.random.Generator):
        probs = np.asarray(probs, dtype=float)
        self.q_star = np.broadcast_to(probs, (runs, probs.shape[-1])).copy()
        self.runs, self.k = self.q_star.shape
        self.rng = rng

    def pull(self, a: np.ndarray) -> np.ndarray:
        return (self.rng.random(self.runs) < self.q_star[np.arange(self.runs), a]).astype(float)


# ----------------------------------------------------------------------------------
# Agents
# ----------------------------------------------------------------------------------

class Agent:
    name = "agent"

    def __init__(self, k: int):
        self.k = k

    def reset(self, runs: int, rng: np.random.Generator):
        self.runs, self.rng = runs, rng
        self.rows = np.arange(runs)


class ActionValueAgent(Agent):
    """Keeps Q (estimates) and N (counts) and updates them incrementally (§3-4):
        Q <- Q + step * (R - Q),  step = 1/N (sample average, Eq. 3.1) or alpha (constant, Eq. 4.1).
    With unbiased=True the step is beta_n = alpha / o_n, o_n = o_{n-1} + alpha (1 - o_{n-1}),
    the 'unbiased constant-step-size trick' (Eq. 4.4, §4.3)."""

    def __init__(self, k: int, alpha: float | None = None, q1: float = 0.0, unbiased: bool = False):
        super().__init__(k)
        self.alpha, self.q1, self.unbiased = alpha, q1, unbiased

    def reset(self, runs, rng):
        super().reset(runs, rng)
        self.Q = np.full((runs, self.k), self.q1, dtype=float)
        self.N = np.zeros((runs, self.k))
        self.o = np.zeros((runs, self.k))          # trace for the unbiased step size

    def update(self, a, r):
        rows = self.rows
        self.N[rows, a] += 1
        if self.alpha is None:                     # sample average: Eq. (3.1)
            step = 1.0 / self.N[rows, a]
        elif self.unbiased:                        # Eq. (4.4)
            self.o[rows, a] += self.alpha * (1.0 - self.o[rows, a])
            step = self.alpha / self.o[rows, a]
        else:                                      # constant step size: Eq. (4.1)
            step = self.alpha
        self.Q[rows, a] += step * (r - self.Q[rows, a])


class EpsilonGreedy(ActionValueAgent):
    """With probability 1-eps take argmax_a Q(a) (ties at random), else a uniform arm (§2.2)."""

    def __init__(self, k, eps=0.1, alpha=None, q1=0.0, unbiased=False, name=None):
        super().__init__(k, alpha, q1, unbiased)
        self.eps = eps
        self.name = name or (f"greedy" if eps == 0 else f"ε-greedy ε={eps:g}")

    def act(self, t):
        greedy = argmax_random_ties(self.Q, self.rng)
        if self.eps == 0:
            return greedy
        explore = self.rng.random(self.runs) < self.eps
        return np.where(explore, self.rng.integers(0, self.k, self.runs), greedy)


class EpsilonDecreasing(ActionValueAgent):
    """eps_t = min(1, c k / t): the 'eps_n-greedy' schedule of Auer et al. (2002), §11.3."""

    def __init__(self, k, c=5.0, name=None):
        super().__init__(k)
        self.c = c
        self.name = name or f"ε_t-greedy (ε_t=min(1,{c:g}k/t))"

    def act(self, t):
        eps = min(1.0, self.c * self.k / t)
        greedy = argmax_random_ties(self.Q, self.rng)
        explore = self.rng.random(self.runs) < eps
        return np.where(explore, self.rng.integers(0, self.k, self.runs), greedy)


class UCB(ActionValueAgent):
    """A_t = argmax_a [ Q_t(a) + c sqrt(ln t / N_t(a)) ], untried arms first (Eq. 6.5, §6.3).
    c = sqrt(2) with rewards in [0,1] is Auer et al.'s UCB1 (Eq. 6.4)."""

    def __init__(self, k, c=2.0, alpha=None, name=None):
        super().__init__(k, alpha)
        self.c = c
        self.name = name or f"UCB c={c:g}"

    def act(self, t):
        bonus = self.c * np.sqrt(np.log(t) / np.maximum(self.N, 1))
        index = np.where(self.N == 0, np.inf, self.Q + bonus)   # an untried arm is maximizing
        return argmax_random_ties(index, self.rng)


class KLUCB(ActionValueAgent):
    """KL-UCB for Bernoulli rewards (Garivier & Cappé 2011), Eq. (11.7), §11.4:
        index(a) = max{ q in [Q(a), 1] : N(a) kl(Q(a), q) <= ln t },
    found by bisection.  Its leading constant matches the Lai-Robbins bound."""

    name = "KL-UCB"

    def act(self, t):
        N = np.maximum(self.N, 1)
        level = np.log(max(t, 1)) / N
        lo, hi = self.Q.copy(), np.ones_like(self.Q)
        for _ in range(16):                                   # bisection (precision 2^-16): kl(Q, q) increases in q >= Q
            mid = 0.5 * (lo + hi)
            ok = bernoulli_kl(self.Q, mid) <= level
            lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
        index = np.where(self.N == 0, np.inf, lo)
        return argmax_random_ties(index, self.rng)


class Boltzmann(ActionValueAgent):
    """Softmax / Boltzmann exploration on action values: pi(a) ∝ exp(Q(a)/tau) (§9).
    tau is a TEMPERATURE here (NOTATION.md: tau is context dependent)."""

    def __init__(self, k, tau=0.1, alpha=None, q1=0.0, name=None):
        super().__init__(k, alpha, q1)
        self.tau = tau
        self.name = name or f"Boltzmann τ={tau:g}"

    def act(self, t):
        return sample_categorical(softmax(self.Q / self.tau), self.rng)


class ExploreThenCommit(ActionValueAgent):
    """Pull every arm m times in round robin (an 'A/B/n test'), then commit forever
    to the empirical best arm (Algorithm 11.1, §11.6).

    The committed arm A_hat is chosen ONCE, at step t = m k + 1, and never revisited:
    rewards collected after that still update Q (inherited update), but they cannot
    change the decision.  (Re-taking argmax Q at every step would be a different,
    self-correcting algorithm, 'explore-first-then-greedy'.)"""

    def __init__(self, k, m=50, name=None):
        super().__init__(k)
        self.m = m
        self.name = name or f"explore-then-commit m={m}"

    def reset(self, runs, rng):
        super().reset(runs, rng)
        self.committed = None                      # A_hat, one entry per run, fixed after exploration

    def act(self, t):
        if t <= self.m * self.k:                   # exploration phase: round robin
            return np.full(self.runs, (t - 1) % self.k)
        if self.committed is None:                 # first step after exploration: commit (ties at random)
            self.committed = argmax_random_ties(self.Q, self.rng)
        return self.committed


def etc_exact_regret_two_arms(p1, p2, m, T):
    """Exact E[Reg(T)] of explore-then-commit on two Bernoulli arms with p1 >= p2 (Exercise 2.9,
    §11.7).  Each arm is pulled m times; the commitment is wrong if S2 > S1, and a tie
    S1 = S2 is broken at random, so E[Reg] = m Delta + (T - 2m) Delta P(commit to arm 2)
    with S_i ~ Binomial(m, p_i).  No simulation needed."""
    from scipy import stats
    s = np.arange(m + 1)
    f1, f2 = stats.binom.pmf(s, m, p1), stats.binom.pmf(s, m, p2)
    F1 = np.cumsum(f1)                                   # P(S1 <= s)
    p_wrong = np.sum(f2[1:] * F1[:-1]) + 0.5 * np.sum(f1 * f2)   # P(S1 < S2) + P(tie)/2
    gap = p1 - p2
    return m * gap + (T - 2 * m) * gap * p_wrong


class GradientBandit(Agent):
    """Gradient bandit (§8): preferences H, policy pi = softmax(H), and
        H(a) <- H(a) + alpha (R_t - baseline_t) (1[a = A_t] - pi_t(a))     (Eq. 8.2)
    baseline_t = average of R_1..R_{t-1} (R_1 at t = 1), or 0 without baseline."""

    def __init__(self, k, alpha=0.1, baseline=True, name=None):
        super().__init__(k)
        self.alpha, self.baseline = alpha, baseline
        self.name = name or f"gradient α={alpha:g}" + ("" if baseline else " (no baseline)")

    def reset(self, runs, rng):
        super().reset(runs, rng)
        self.H = np.zeros((runs, self.k))
        self.Rbar = np.zeros(runs)
        self.t = 0

    def act(self, t):
        self.pi = softmax(self.H)                  # remember pi_t for the update
        return sample_categorical(self.pi, self.rng)

    def update(self, a, r):
        self.t += 1
        if self.baseline:
            # The baseline must not depend on A_t (§8.3), so it uses rewards BEFORE R_t.
            b = r if self.t == 1 else self.Rbar
        else:
            b = 0.0
        onehot = np.zeros_like(self.H)
        onehot[self.rows, a] = 1.0
        self.H += self.alpha * (r - b)[:, None] * (onehot - self.pi)
        self.Rbar += (r - self.Rbar) / self.t      # incremental mean, updated AFTER use


class ThompsonGaussian(Agent):
    """Thompson sampling with independent Normal(m0, s0^2) priors and known reward noise
    sigma (Algorithm 7.2, §7.4).  Posterior (Eq. 7.3) of arm a after n pulls with reward sum S:
        precision = 1/s0^2 + n/sigma^2,  mean = (m0/s0^2 + S/sigma^2) / precision."""

    def __init__(self, k, m0=0.0, s0=1.0, sigma=1.0, name=None):
        super().__init__(k)
        self.m0, self.s0, self.sigma = m0, s0, sigma
        self.name = name or f"Thompson (Gaussian, s0={s0:g})"

    def reset(self, runs, rng):
        super().reset(runs, rng)
        self.N = np.zeros((runs, self.k))
        self.S = np.zeros((runs, self.k))

    def posterior(self):
        prec = 1.0 / self.s0 ** 2 + self.N / self.sigma ** 2
        mean = (self.m0 / self.s0 ** 2 + self.S / self.sigma ** 2) / prec
        return mean, 1.0 / np.sqrt(prec)

    def act(self, t):
        mean, std = self.posterior()
        theta = mean + std * self.rng.normal(size=mean.shape)   # one sample per arm
        return np.argmax(theta, axis=1)                          # ties have probability 0

    def update(self, a, r):
        self.N[self.rows, a] += 1
        self.S[self.rows, a] += r


class ThompsonBeta(Agent):
    """Beta-Bernoulli Thompson sampling (Algorithm 7.1, §7.3): Beta(a0 + successes, b0 + failures)."""

    def __init__(self, k, a0=1.0, b0=1.0, name=None):
        super().__init__(k)
        self.a0, self.b0 = a0, b0
        self.name = name or "Thompson (Beta)"

    def reset(self, runs, rng):
        super().reset(runs, rng)
        self.S = np.zeros((runs, self.k))
        self.F = np.zeros((runs, self.k))

    def act(self, t):
        theta = self.rng.beta(self.a0 + self.S, self.b0 + self.F)
        return argmax_random_ties(theta, self.rng)

    def update(self, a, r):
        self.S[self.rows, a] += r
        self.F[self.rows, a] += 1.0 - r


class EXP3(Agent):
    """EXP3 for adversarial bandits with rewards in [0,1] (Algorithm 12.1, §12.2), loss-based form:
        P_t(a) ∝ exp(-eta * Lhat(a)),  Lhat(A_t) += (1 - R_t) / P_t(A_t)."""

    def __init__(self, k, horizon, eta=None, name=None):
        super().__init__(k)
        self.eta = eta if eta is not None else np.sqrt(2 * np.log(k) / (horizon * k))
        self.name = name or "EXP3"

    def reset(self, runs, rng):
        super().reset(runs, rng)
        self.Lhat = np.zeros((runs, self.k))

    def act(self, t):
        self.P = softmax(-self.eta * self.Lhat)
        return sample_categorical(self.P, self.rng)

    def update(self, a, r):
        self.Lhat[self.rows, a] += (1.0 - r) / self.P[self.rows, a]   # importance-weighted loss


# ----------------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------------

@dataclass
class Result:
    name: str
    avg_reward: np.ndarray       # [steps]  mean over runs of R_t
    pct_optimal: np.ndarray      # [steps]  fraction of runs with q*(A_t) = max_a q*(a)
    regret_mean: np.ndarray      # [steps]  mean over runs of cumulative pseudo-regret
    regret_se: np.ndarray        # [steps]  standard error of that mean
    counts: np.ndarray           # [runs, k] final N_T(a) (in the environment's arm order)
    final_regret: np.ndarray     # [runs]   per-run pseudo-regret at the horizon
    run_avg_reward: np.ndarray   # [runs]   per-run average of R_1..R_T
    seconds: float


def run(env, agent: Agent, steps: int, rng: np.random.Generator) -> Result:
    """Interact for `steps` steps with all runs in parallel and record learning curves.
    Pseudo-regret uses the true means (Eq. 11.2): sum_t [max_a q*(a) - q*(A_t)]."""
    t0 = time.time()
    runs, k = env.q_star.shape
    rows = np.arange(runs)
    agent.reset(runs, rng)
    avg_r, opt, reg_m, reg_se = (np.zeros(steps) for _ in range(4))
    cum = np.zeros(runs)
    rsum = np.zeros(runs)
    counts = np.zeros((runs, k))
    for t in range(1, steps + 1):
        a = agent.act(t)
        q_a = env.q_star[rows, a]                     # before any random-walk drift
        best = env.q_star.max(axis=1)
        r = env.pull(a)
        agent.update(a, r)
        counts[rows, a] += 1
        cum += best - q_a
        rsum += r
        avg_r[t - 1] = r.mean()
        opt[t - 1] = np.mean(q_a == best)
        reg_m[t - 1] = cum.mean()
        reg_se[t - 1] = cum.std(ddof=1) / np.sqrt(runs) if runs > 1 else 0.0
    return Result(agent.name, avg_r, opt, reg_m, reg_se, counts, cum.copy(), rsum / steps, time.time() - t0)


# ----------------------------------------------------------------------------------
# A literal, single-run implementation of Algorithm 2.1 (for reading and for testing)
# ----------------------------------------------------------------------------------

def simple_bandit_loop(q_star, steps, eps, rng):
    """Algorithm 2.1 'A simple bandit algorithm', exactly as in the pseudocode box (Chapter 02, §3)."""
    k = len(q_star)
    Q = [0.0] * k
    N = [0] * k
    rewards = []
    for _ in range(steps):
        if rng.random() < eps:
            A = int(rng.integers(k))                          # explore
        else:
            m = max(Q)
            A = int(rng.choice([a for a in range(k) if Q[a] == m]))   # greedy, random ties
        R = q_star[A] + rng.normal()                          # bandit(A)
        N[A] += 1
        Q[A] += (R - Q[A]) / N[A]                             # Eq. (3.1)
        rewards.append(R)
    return np.array(rewards)


def _self_test(quick: bool):
    seed = 0
    runs, steps = (100, 200) if quick else (400, 500)
    print(f"bandits.py self-test  (seed={seed}, runs={runs}, steps={steps}, quick={quick})")
    rng = np.random.default_rng(seed)

    # 1. The incremental update (3.2) reproduces the batch sample mean exactly.
    x = rng.normal(size=1000)
    Q = 0.0
    for n, r in enumerate(x, start=1):
        Q += (r - Q) / n
    print(f"  incremental mean vs np.mean: |diff| = {abs(Q - x.mean()):.2e}")
    assert abs(Q - x.mean()) < 1e-12

    # 2. Vectorized EpsilonGreedy agrees (statistically) with the literal loop.
    q_stars = np.random.default_rng(1).normal(size=(runs, 10))
    loop = np.mean([simple_bandit_loop(q, steps, 0.1, np.random.default_rng(100 + i)).mean()
                    for i, q in enumerate(q_stars)])
    env = GaussianBandit(q_stars.copy(), np.random.default_rng(2))
    vec = run(env, EpsilonGreedy(10, 0.1), steps, np.random.default_rng(3)).avg_reward.mean()
    print(f"  ε=0.1 average reward over {steps} steps: loop {loop:.3f}, vectorized {vec:.3f}")
    assert abs(loop - vec) < 0.06, "vectorized agent disagrees with the reference loop"

    # 3. Softmax derivative d pi(x)/dH(a) = pi(x)(1[a=x] - pi(a)) (Eq. 8.4) by finite differences.
    H = rng.normal(size=(1, 5))
    pi = softmax(H)[0]
    analytic = np.diag(pi) - np.outer(pi, pi)          # [x, a]
    h = 1e-6
    numeric = np.stack([(softmax(H + h * np.eye(5)[a])[0] - softmax(H - h * np.eye(5)[a])[0]) / (2 * h)
                        for a in range(5)], axis=1)
    print(f"  softmax Jacobian max |analytic - numeric| = {np.abs(analytic - numeric).max():.2e}")
    assert np.abs(analytic - numeric).max() < 1e-8

    # 4. Every agent runs (KL-UCB, Beta-Thompson and EXP3 need rewards in [0, 1]).
    for env_kind in ("gaussian", "bernoulli"):
        agents = [EpsilonGreedy(5, 0.1), EpsilonDecreasing(5), UCB(5, 2.0), Boltzmann(5, 0.2),
                  ExploreThenCommit(5, 5), GradientBandit(5, 0.1), ThompsonGaussian(5)]
        if env_kind == "bernoulli":
            agents += [KLUCB(5), ThompsonBeta(5), EXP3(5, 100)]
        for ag in agents:
            env = (GaussianBandit.testbed(20, 5, np.random.default_rng(4)) if env_kind == "gaussian"
                   else BernoulliBandit([0.2, 0.4, 0.5, 0.6, 0.7], 20, np.random.default_rng(4)))
            res = run(env, ag, 100, np.random.default_rng(5))
            assert np.isfinite(res.regret_mean).all()
    print("  all agents ran (Gaussian and Bernoulli bandits): OK")

    # 5. Explore-then-commit really commits: after exploration every run pulls ONE arm only.
    env = BernoulliBandit([0.55, 0.45], 400, np.random.default_rng(6))
    etc = ExploreThenCommit(2, m=5)
    res = run(env, etc, 300, np.random.default_rng(7))
    assert np.all(res.counts.min(axis=1) == 5), "ETC pulled an arm after committing to the other one"
    exact = etc_exact_regret_two_arms(0.55, 0.45, 5, 300)
    print(f"  ETC (m=5, T=300, two arms): every run commits; simulated E[Reg] {res.regret_mean[-1]:.2f} "
          f"± {res.regret_se[-1]:.2f} vs exact {exact:.2f}")
    assert abs(res.regret_mean[-1] - exact) < 4 * res.regret_se[-1] + 0.1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    _self_test(p.parse_args().quick)
