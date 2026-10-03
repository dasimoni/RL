"""Contextual bandits: LinUCB vs context-free learners (Chapter 02, Section 13).

Problem (one per run): k = 5 arms, contexts x_t in R^d with d = 6 (a constant feature 1 plus
5 features drawn uniformly on the unit sphere), and a separate unknown weight vector per arm
("disjoint" linear model):
        R_t = x_t^T theta_{A_t} + noise,   noise ~ N(0, 0.5^2),   theta_a ~ N(0, I_d / d).
The best arm depends on the context, so the benchmark is the context-dependent oracle
max_a x_t^T theta_a, and regret is sum_t [max_a x_t^T theta_a - x_t^T theta_{A_t}] (Eq. 13.1).

Learners:
  * LinUCB (Li et al. 2010, disjoint): ridge estimate theta_hat_a = A_a^{-1} b_a and index
        x^T theta_hat_a + alpha * sqrt(x^T A_a^{-1} x)                      (Eq. 13.5)
  * linear Thompson sampling: theta~_a ~ N(theta_hat_a, v^2 A_a^{-1})
  * linear eps-greedy (eps = 0.1) and pure greedy on the same ridge estimates
  * context-free UCB1-style learner that ignores x (it can at best find the single arm that
    is best ON AVERAGE, so its regret against the contextual oracle grows linearly)
A_a^{-1} is maintained with the Sherman-Morrison formula (Eq. 13.6).  All runs are simulated
in parallel: arrays have a leading [runs] axis.

Run:  python code/ch02_multi_armed_bandits/linucb_contextual.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from bandits import FIG_DIR, UCB, argmax_random_ties, setup_matplotlib, style


class LinearContextualBandit:
    def __init__(self, runs, k, d, noise, rng):
        self.runs, self.k, self.d, self.noise, self.rng = runs, k, d, noise, rng
        self.theta = rng.normal(0, 1 / np.sqrt(d), size=(runs, k, d))

    def context(self):
        z = self.rng.normal(size=(self.runs, self.d - 1))
        z /= np.linalg.norm(z, axis=1, keepdims=True)              # uniform on the unit sphere
        return np.concatenate([np.ones((self.runs, 1)), z], axis=1)

    def means(self, x):
        return np.einsum("rkd,rd->rk", self.theta, x)              # x^T theta_a for every arm

    def pull(self, x, a):
        m = self.means(x)[np.arange(self.runs), a]
        return m + self.noise * self.rng.normal(size=self.runs)


class RidgeArms:
    """Per-arm ridge regression with A_a = lambda I + sum x x^T kept as an inverse."""

    def __init__(self, runs, k, d, lam=1.0):
        self.A_inv = np.broadcast_to(np.eye(d) / lam, (runs, k, d, d)).copy()
        self.b = np.zeros((runs, k, d))
        self.rows = np.arange(runs)

    def theta_hat(self):
        return np.einsum("rkij,rkj->rki", self.A_inv, self.b)

    def width(self, x):
        """sqrt(x^T A_a^{-1} x) for every arm: the standard-deviation factor of x^T theta_hat_a."""
        return np.sqrt(np.einsum("ri,rkij,rj->rk", x, self.A_inv, x))

    def update(self, x, a, r):
        Ai = self.A_inv[self.rows, a]                              # [runs, d, d]
        Ax = np.einsum("rij,rj->ri", Ai, x)
        denom = 1.0 + np.einsum("ri,ri->r", x, Ax)
        self.A_inv[self.rows, a] = Ai - np.einsum("ri,rj->rij", Ax, Ax) / denom[:, None, None]  # Sherman-Morrison
        self.b[self.rows, a] += r[:, None] * x


def run_contextual(policy, runs, k, d, T, noise, seed, agent_seed, **kw):
    env = LinearContextualBandit(runs, k, d, noise, np.random.default_rng(seed))   # same problems
    rng = np.random.default_rng(agent_seed)
    rows = np.arange(runs)
    ridge = RidgeArms(runs, k, d)
    ctx_free = None
    if policy == "context-free UCB":
        ctx_free = UCB(k, c=kw.get("c", 1.0))
        ctx_free.reset(runs, rng)
    regret = np.zeros((runs, T))
    for t in range(1, T + 1):
        x = env.context()
        mu = env.means(x)
        if policy == "LinUCB":
            index = np.einsum("rki,ri->rk", ridge.theta_hat(), x) + kw["alpha"] * ridge.width(x)
            a = argmax_random_ties(index, rng)
        elif policy == "linear Thompson":
            L = np.linalg.cholesky(ridge.A_inv)                      # A^{-1} = L L^T
            theta = ridge.theta_hat() + kw["v"] * np.einsum("rkij,rkj->rki", L, rng.normal(size=(runs, k, d)))
            a = np.argmax(np.einsum("rki,ri->rk", theta, x), axis=1)
        elif policy == "linear ε-greedy":
            greedy = argmax_random_ties(np.einsum("rki,ri->rk", ridge.theta_hat(), x), rng)
            a = np.where(rng.random(runs) < kw["eps"], rng.integers(0, k, runs), greedy)
        elif policy == "context-free UCB":
            a = ctx_free.act(t)
        else:
            raise ValueError(policy)
        r = env.pull(x, a)
        if ctx_free is not None:
            ctx_free.update(a, r)
        else:
            ridge.update(x, a, r)
        regret[:, t - 1] = mu.max(axis=1) - mu[rows, a]
    # Reference: per-step regret of the best FIXED arm (the best a context-free learner can do).
    # E[x] = (1, 0, ..., 0) by symmetry of the sphere, so that arm maximizes theta_a[0].
    best_arm_on_avg = env.theta[:, :, 0].argmax(axis=1)
    gaps = []
    for _ in range(200):
        mu = env.means(env.context())
        gaps.append((mu.max(axis=1) - mu[rows, best_arm_on_avg]).mean())
    return np.cumsum(regret, axis=1), float(np.mean(gaps))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    seed = 0
    k, d, noise = 5, 6, 0.5
    runs, T = (20, 1000) if args.quick else (100, 5000)
    print(f"Seed {seed}; k={k} arms, d={d} features, noise sd {noise}; {runs} runs x T={T}; ridge λ=1")
    t0 = time.time()
    configs = [
        ("LinUCB α=0 (greedy ridge)", "LinUCB", dict(alpha=0.0)),
        ("LinUCB α=0.5", "LinUCB", dict(alpha=0.5)),
        ("LinUCB α=1", "LinUCB", dict(alpha=1.0)),
        ("LinUCB α=2", "LinUCB", dict(alpha=2.0)),
        ("linear Thompson v=0.5", "linear Thompson", dict(v=0.5)),
        ("linear ε-greedy ε=0.1", "linear ε-greedy", dict(eps=0.1)),
        ("context-free UCB (c=1)", "context-free UCB", dict(c=1.0)),
    ]
    curves = {}
    for i, (name, policy, kw) in enumerate(configs):
        t1 = time.time()
        cum, fixed_gap = run_contextual(policy, runs, k, d, T, noise, seed, agent_seed=seed + 1 + i, **kw)
        curves[name] = cum
        fin = cum[:, -1]
        late = (cum[:, -1] - cum[:, T // 2 - 1]) / (T - T // 2)
        print(f"  {name:>27}: Reg(T) = {fin.mean():7.1f} ± {fin.std(ddof=1) / np.sqrt(runs):5.1f}; "
              f"regret/step in 2nd half = {late.mean():.4f}   ({time.time() - t1:.1f}s)")
    print(f"  reference: the best single (context-free) arm loses {fixed_gap:.4f} per step "
          f"against the contextual oracle -> {fixed_gap * T:.0f} over T")

    if not args.quick:
        plt = setup_matplotlib()
        fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
        t = np.arange(1, T + 1)
        for i, (name, cum) in enumerate(curves.items()):
            m = cum.mean(axis=0)
            se = cum.std(axis=0, ddof=1) / np.sqrt(runs)
            ax[0].plot(t, m, label=name, **style(i), lw=1.5)
            ax[0].fill_between(t, m - 2 * se, m + 2 * se, color=style(i)["color"], alpha=0.12, lw=0)
        ax[0].plot(t, fixed_gap * t, color="#8a8984", lw=1, ls=(0, (6, 3)), label="best fixed arm (oracle)")
        ax[0].set(xlabel="t", ylabel="cumulative regret vs contextual oracle",
                  title=f"(a) Linear contextual bandit, k={k}, d={d} ({runs} runs, ±2 s.e.)")
        ax[0].legend(loc="upper left", fontsize=7.5)
        for i, (name, cum) in enumerate(curves.items()):
            if "context-free" in name:
                continue
            ax[1].plot(t, cum.mean(axis=0), label=name, **style(i), lw=1.5)
        ax[1].set_xscale("log")
        ax[1].set(xlabel="t (log scale)", ylabel="cumulative regret",
                  title="(b) Contextual learners only, log time axis")
        ax[1].legend(loc="upper left", fontsize=7.5)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "linucb_contextual.png")
        fig.savefig(out)
        print(f"\nsaved {out}")
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
