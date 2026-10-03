"""Numerical checks for the Chapter 00 exercise solutions (Exercises 4, 5, 10, 12, 13, 14, 15).

  Ex. 4   TD targets with terminated / truncated flags (Eq. 8.1)
  Ex. 5   importance sampling under partial coverage: OIS stays unbiased, WIS does not converge
  Ex. 10  optimal constant baseline for the score-function estimator of d/dmu E[x^2]
  Ex. 12  gradient w.r.t. sigma: score function vs reparameterization
  Ex. 13  mixing of the 'two rooms' chain as the door probability shrinks
  Ex. 14  vectorised CartPole with SAME_STEP autoreset and info["final_obs"]
  Ex. 15  the four silent bugs in a semi-gradient Q-learning update, and the fixed version

Run:  python code/ch00_math_toolkit/exercise_solutions.py [--quick]
(no figures are produced in either mode; Ex. 13 reuses functions from markov_chain_stationary.py)
"""
import argparse
import time
import warnings

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.vector import AutoresetMode

from markov_chain_stationary import (linear_solve_method, power_iteration,
                                     second_eigenvalue_modulus, two_rooms_chain)


def ex04_flags():
    gamma = 0.99
    print("\nEx. 4: TD targets y = r + gamma (1 - terminated) V(s'), gamma = 0.99")
    for r, term, trunc, v_next in [(1.0, False, True, 50.0), (1.0, True, True, 50.0)]:
        y = r + gamma * (1 - term) * v_next
        y_bug = r + gamma * (1 - (term or trunc)) * v_next     # the 'done' mask bug
        print(f"  r={r}, terminated={term!s:5s}, truncated={trunc!s:5s}, V(s')={v_next}: y = {y:.2f} "
              f"(with done = terminated or truncated: {y_bug:.2f})")


def ex05_partial_coverage(rng, n):
    pi, b, f = np.array([0.5, 0.5]), np.array([1.0, 0.0]), np.array([1.0, 0.0])
    x = rng.choice(2, size=n, p=b)
    rho = pi[x] / b[x]
    print(f"\nEx. 5(c): pi={pi}, b={b}, f={f}, true E_pi[f] = {pi @ f}; {n} samples from b")
    print(f"  OIS = {(rho * f[x]).mean():.4f}   WIS = {(rho * f[x]).sum() / rho.sum():.4f}   "
          f"mean ratio (-> pi(supp b) = 0.5, not 1) = {rho.mean():.4f}")


def ex10_optimal_baseline(rng, M):
    mu, sigma = 1.0, 1.0
    xi = rng.standard_normal(M)
    x = mu + sigma * xi
    print("\nEx. 10: score-function estimator of d/dmu E[x^2] with constant baselines (mu = sigma = 1)")
    for name, b, theory in [("no baseline", 0.0, 30.0), ("b = E[f] = mu^2 + sigma^2", mu ** 2 + sigma ** 2, 18.0),
                            ("b* = mu^2 + 3 sigma^2", mu ** 2 + 3 * sigma ** 2, 14.0)]:
        g = (x ** 2 - b) * xi / sigma
        print(f"  {name:28s} mean {g.mean():.4f}  variance {g.var():7.3f}  (theory {theory})")


def ex12_sigma_gradient(rng, M):
    mu, sigma = 1.0, 1.0
    xi = rng.standard_normal(M)
    x = mu + sigma * xi
    g_rp = 2 * x * xi                      # d/dsigma f(mu + sigma*xi) = f'(x) * xi
    g_sf = x ** 2 * (xi ** 2 - 1) / sigma  # f(x) * d/dsigma log N(x; mu, sigma^2)
    print("\nEx. 12: d/dsigma E[x^2] = 2 sigma, mu = sigma = 1")
    print(f"  reparameterization: mean {g_rp.mean():.4f}, variance {g_rp.var():7.2f} (theory 4mu^2 + 8sigma^2 = 12)")
    print(f"  score function    : mean {g_sf.mean():.4f}, variance {g_sf.var():7.2f} "
          f"(theory 2mu^4/sigma^2 + 60mu^2 + 74sigma^2 = 136)")


def ex13_two_rooms(rng):
    print("\nEx. 13: two-rooms chain (2 x 10 states), power iteration from a one-hot start")
    for door in [0.1, 0.03, 0.01, 0.003]:
        P = two_rooms_chain(10, door, rng)
        d = linear_solve_method(P)
        lam2 = second_eigenvalue_modulus(P)
        d0 = np.zeros(P.shape[0]); d0[0] = 1.0
        _, err = power_iteration(P, d0, 20_000, d)
        k = int(np.argmax(err < 1e-6))
        print(f"  door={door:6.3f}  |lambda2|={lam2:.4f} (1-2*door={1 - 2 * door:.4f})  "
              f"iterations to L1 error 1e-6: {k:5d}   log(1e-6)/log|lambda2| = {np.log(1e-6) / np.log(lam2):6.0f}")


def ex14_same_step(steps):
    envs = gym.make_vec("CartPole-v1", num_envs=4, vectorization_mode="sync", max_episode_steps=30,
                        vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP})
    envs.action_space.seed(0)
    obs, _ = envs.reset(seed=0)
    n_trans = n_term = n_trunc = 0
    for _ in range(steps):
        next_obs, r, term, trunc, info = envs.step(envs.action_space.sample())
        # SAME_STEP: for finished sub-envs next_obs is ALREADY the first obs of the new episode;
        # the true last observation is in info["final_obs"] where info["_final_obs"] is True.
        s_next = next_obs.copy()
        if "final_obs" in info:
            for i in np.where(info["_final_obs"])[0]:
                s_next[i] = info["final_obs"][i]
        # (obs, a, r, s_next, term) is now a valid transition for every sub-env; TD target:
        #   r + gamma * (1 - term) * V(s_next)
        n_trans += len(r); n_term += int(term.sum()); n_trunc += int((trunc & ~term).sum())
        obs = next_obs
    envs.close()
    print(f"\nEx. 14: SAME_STEP autoreset, {steps} vector steps x 4 envs (limit 30): {n_trans} transitions, "
          f"{n_term} terminal, {n_trunc} truncated-only (bootstrap from final_obs)")


# ---------------------------------------------------------------- Ex. 15
def td_loss_buggy(q_net, s, a, r, s_next, terminated, gamma=0.99):
    """The update as printed in Exercise 15 (loss only; the optimizer step is omitted here)."""
    q_sa = q_net(s)[:, a]
    target = r + gamma * (1.0 - terminated) * q_net(s_next).max(dim=1, keepdim=True).values
    loss = ((q_sa - target) ** 2).mean()
    return q_sa, target, loss


def td_loss_fixed(q_net, s, a, r, s_next, terminated, gamma=0.99):
    """The corrected version from the solution of Exercise 15."""
    q_sa = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)                    # (B,)
    with torch.no_grad():                                                     # semi-gradient target
        target = r + gamma * (1.0 - terminated) * q_net(s_next).max(dim=1).values   # (B,)
    assert q_sa.shape == target.shape == r.shape, (q_sa.shape, target.shape, r.shape)
    loss = ((q_sa - target) ** 2).mean()
    return q_sa, target, loss


def ex15_td_bugs(seed):
    torch.manual_seed(seed)
    B = 5
    q_net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))
    s, s_next = torch.randn(B, 4), torch.randn(B, 4)
    a = torch.randint(0, 2, (B,))
    r = torch.ones(B)
    term = np.array([False, False, True, False, True])
    trunc = np.array([False, True, False, False, True])
    done_bug = torch.as_tensor(term | trunc, dtype=torch.float32)        # bug 4: the caller's mask
    term_ok = torch.as_tensor(term, dtype=torch.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        q_b, t_b, l_b = td_loss_buggy(q_net, s, a, r, s_next, done_bug)
    q_f, t_f, l_f = td_loss_fixed(q_net, s, a, r, s_next, term_ok)
    print("\nEx. 15: semi-gradient Q-learning update, batch B = 5")
    print(f"  buggy : q_sa shape {tuple(q_b.shape)}, target shape {tuple(t_b.shape)}, "
          f"target.requires_grad={t_b.requires_grad}, loss {l_b.item():.4f}  "
          f"(exceptions: 0, warnings: {len(caught)})")
    print(f"  fixed : q_sa shape {tuple(q_f.shape)}, target shape {tuple(t_f.shape)}, "
          f"target.requires_grad={t_f.requires_grad}, loss {l_f.item():.4f}")
    print(f"  bug 4 : mask differs on {int((done_bug != term_ok).sum())} truncated-only transition(s) "
          f"(their bootstrap term is dropped)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke test")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    M = 200_000 if args.quick else 2_000_000
    print(f"[exercise_solutions] seed={args.seed} samples={M} quick={args.quick}")
    t0 = time.time()
    ex04_flags()
    ex05_partial_coverage(np.random.default_rng(args.seed), 10_000)
    ex10_optimal_baseline(np.random.default_rng(args.seed), M)
    ex12_sigma_gradient(np.random.default_rng(args.seed), M)
    ex13_two_rooms(np.random.default_rng(args.seed))
    ex14_same_step(100 if args.quick else 1000)
    ex15_td_bugs(args.seed)
    print(f"\nruntime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
