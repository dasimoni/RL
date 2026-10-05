"""Reference solutions for the coding exercises of Chapter 10 (and numeric checks of two derivations).

Exercise 10.3  (two-armed bandit) zero-variance optimal baseline: Monte Carlo check of the
               closed forms for b* = (1-p) r1 + p r2 and of the variance with b = J.
Exercise 10.11 REINFORCE with a learned baseline on CartPole-v1, with and without the gamma^t
               factor of Eq. (10.15) (gamma = 0.99, 3 seeds, 600 episodes each).
Exercise 10.12 One-step actor-critic (Algorithm 10.3) on the aliased short corridor, where the
               critic is aliased too: the closed form E[update] = -w (1 - p) of the expected
               actor update, and learning runs against REINFORCE with the same baseline.

Outputs (full mode): figures/exercise_corridor_ac.png
Run:  python code/ch10_policy_gradients/exercise_solutions.py [--quick]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ---------------------------------------------------------------- Exercise 10.3
def ex_bandit(rng, n=200_000):
    print("=== Exercise 10.3: two-armed bandit, softmax policy, r = (1, 2) ===")
    r = np.array([1.0, 2.0])
    for p in (0.5, 0.2):
        pi = np.array([p, 1 - p])
        A = (rng.random(n) > p).astype(int)               # action 0 w.p. p
        psi = np.eye(2)[A] - pi                           # score of the tabular softmax
        J = pi @ r
        b_star = (1 - p) * r[0] + p * r[1]
        out = []
        for b in (0.0, J, b_star):
            g = (r[A] - b)[:, None] * psi
            out.append(g.var(0).sum())
        grad = p * (1 - p) * (r[0] - r[1]) * np.array([1.0, -1.0])
        print(f"  p={p}: grad J = {grad}, J = {J:.2f}, b* = {b_star:.2f};  tr Cov with b=0: {out[0]:.4f}, "
              f"b=J: {out[1]:.4f}, b=b*: {out[2]:.2e}")


# ---------------------------------------------------------------- Exercise 10.11
def ex_gamma_t(quick):
    import reinforce_cartpole as rc
    print("\n=== Exercise 10.11: REINFORCE + baseline on CartPole, with vs without gamma^t ===")
    n_seeds, n_eps = (1, 40) if quick else (3, 600)
    gamma = 0.99
    res = {}
    for keep in (False, True):
        finals = []
        for seed in range(n_seeds):
            finals.append(train_gamma_t(rc, seed, n_eps, gamma, keep))
        res[keep] = np.array(finals)
        print(f"  gamma^t {'kept   ' if keep else 'dropped'}: mean return over episodes "
              f"[1-100, 201-300, last 100] per seed: " +
              "; ".join(f"{f[0]:.0f}/{f[1]:.0f}/{f[2]:.0f}" for f in res[keep]))
    return res


def train_gamma_t(rc, seed, n_eps, gamma, keep_gamma_t, lr_pi=1e-3, lr_v=5e-3):
    """REINFORCE with a learned baseline (as in reinforce_cartpole.py), optionally weighting
    step t by gamma^t (Eq. 10.15) instead of dropping it."""
    import gymnasium as gym
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")
    policy, value = rc.mlp(4, 2), rc.mlp(4, 1)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=lr_pi)
    opt_v = torch.optim.Adam(value.parameters(), lr=lr_v)
    rets = np.zeros(n_eps)
    for ep in range(n_eps):
        O, A, R = rc.run_episode(env, policy, rng, seed=seed * 100_000 + ep)
        rets[ep] = R.sum()
        obs_t = torch.as_tensor(O)
        G = torch.as_tensor(rc.rewards_to_go(R, gamma))
        with torch.no_grad():
            w = G - value(obs_t).squeeze(-1)
        if keep_gamma_t:
            w = w * gamma ** torch.arange(len(R), dtype=torch.float32)
        logp = torch.distributions.Categorical(logits=policy(obs_t)).log_prob(torch.as_tensor(A))
        loss = -(w * logp).sum() / rc.T_MAX         # constant normalizer, as in reinforce_cartpole.py
        opt_pi.zero_grad()
        loss.backward()
        opt_pi.step()
        loss_v = 0.5 * ((value(obs_t).squeeze(-1) - G) ** 2).mean()
        opt_v.zero_grad()
        loss_v.backward()
        opt_v.step()
    env.close()
    q = max(1, n_eps // 6)
    return rets[:min(100, n_eps)].mean(), rets[200:300].mean() if n_eps > 300 else np.nan, rets[-q:].mean()


# ---------------------------------------------------------------- Exercise 10.12
def one_step_ac_corridor(n_episodes, alpha_theta, alpha_w, p0, seed, max_steps=1000):
    """Algorithm 10.3 on the aliased short corridor: one shared actor parameter AND one shared
    critic weight w (the critic cannot tell the states apart either).  gamma = 1."""
    import math
    import random
    import short_corridor as sc
    rng = random.Random(seed)
    theta, w = math.log(p0 / (1 - p0)), 0.0
    rets, ps = np.empty(n_episodes), np.empty(n_episodes)
    for ep in range(n_episodes):
        s, T = 0, 0
        ps[ep] = 1 / (1 + math.exp(-theta))
        while s != 3 and T < max_steps:
            p = 1 / (1 + math.exp(-theta))
            a = sc.RIGHT if rng.random() < p else sc.LEFT
            s2 = sc.step(s, a)
            T += 1
            delta = -1.0 + (0.0 if s2 == 3 else w) - w     # = -1 except on the final step
            psi = (1 - p) if a == sc.RIGHT else -p
            w += alpha_w * delta
            theta = max(-30.0, min(30.0, theta + alpha_theta * delta * psi))
            s = s2
        rets[ep] = -T
    return rets, ps


def ex_corridor_ac(quick):
    import random
    import short_corridor as sc
    print("\n=== Exercise 10.12: one-step actor-critic on the aliased short corridor ===")
    # (a) closed form of the expected actor update for a fixed critic weight w:
    #     E[sum_t delta_t psi_t] = -w (1 - p)   (the episode always ends with 'right' from state 2)
    rng = random.Random(0)
    n = 2_000 if quick else 20_000
    for p in (0.3, 0.586, 0.9):
        w = sc.J_exact(p)                               # the aliased critic's TD fixed point, w = J(p)
        tot = 0.0
        for _ in range(n):
            s = 0
            while s != 3:
                a = sc.RIGHT if rng.random() < p else sc.LEFT
                s2 = sc.step(s, a)
                delta = -1.0 + (0.0 if s2 == 3 else w) - w
                tot += delta * ((1 - p) if a == sc.RIGHT else -p)
                s = s2
        print(f"  p={p}: w=J(p)={w:7.2f};  simulated E[update] = {tot / n:6.3f},  closed form -w(1-p) = "
              f"{-w * (1 - p):6.3f};  true dJ/dtheta = {p * (1 - p) * (sc.J_exact(p + 1e-6) - sc.J_exact(p - 1e-6)) / 2e-6:7.3f}")
    # (b) learning runs: one-step AC vs REINFORCE with the same aliased baseline
    n_runs, n_eps = (5, 200) if quick else (30, 1000)
    out = {}
    for name, fn in (("one-step AC", lambda i: one_step_ac_corridor(n_eps, 2 ** -9, 2 ** -6, 0.05, i)),
                     ("REINFORCE + baseline", lambda i: sc.reinforce(n_eps, 2 ** -9, 2 ** -6, 0.05, i, True))):
        R, P = zip(*[fn(1000 + i) for i in range(n_runs)])
        R, P = np.array(R), np.array(P)
        out[name] = (R, P)
        marks = [n_eps // 3, 2 * n_eps // 3, n_eps - 1]
        print(f"  {name:21s} (alpha_theta=2^-9, alpha_w=2^-6): mean p at episodes {marks}: "
              + ", ".join(f"{P[:, m].mean():.3f}" for m in marks)
              + f";  mean return over the last {n_eps // 10} episodes {R[:, -(n_eps // 10):].mean():.1f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(1)
    print(f"seed={args.seed}  quick={args.quick}")
    t0 = time.time()
    ex_bandit(np.random.default_rng(args.seed))
    ex_gamma_t(args.quick)
    out = ex_corridor_ac(args.quick)
    print(f"Time {time.time() - t0:.0f}s")
    if not args.quick:
        from plot_style import setup, C
        plt = setup()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.6, 4.0))
        for j, (k, (R, P)) in enumerate(out.items()):
            x = np.arange(1, P.shape[1] + 1)
            ax.plot(x, P.mean(0), color=C[j], ls=["-", "--"][j], label=k + f" (mean of {P.shape[0]} runs)")
            ax.fill_between(x, np.quantile(P, 0.1, 0), np.quantile(P, 0.9, 0), color=C[j], alpha=0.15)
        ax.axhline(2 - np.sqrt(2), color="#8a8985", ls=":", lw=1, label="$p^\\ast=0.586$")
        ax.set_xlabel("episode")
        ax.set_ylabel("$p=\\pi(\\mathrm{right})$ (band: 10-90% of runs)")
        ax.set_title("Aliased corridor: a bootstrapping critic misleads the actor")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "exercise_corridor_ac.png"))
        plt.close(fig)
        print(f"figure written to {FIG_DIR}/exercise_corridor_ac.png")


if __name__ == "__main__":
    main()
