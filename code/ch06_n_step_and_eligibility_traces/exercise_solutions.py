"""Reference solutions for the coding exercises of Chapter 06.

Exercise 14 -- Watkins's Q(lambda) vs "naive" Q(lambda) vs SARSA(lambda) as exploration grows.
    On the gridworld of sarsa_lambda_gridworld.py (lambda = 0.9, replacing traces),
    compare the three methods for epsilon in {0.1, 0.3}. Watkins's Q(lambda) cuts its
    trace after every exploratory action; naive Q(lambda) never cuts. Naive Q(lambda) is
    Q*(lambda) of Harutyunyan et al. (2016): q_* is still a fixed point of its expected
    update, but that update is a contraction only for
    lambda < (1 - gamma) / (gamma * max_s ||pi_greedy - b||_1), and lambda = 0.9 is far
    outside that range. SARSA(lambda) is on-policy and never needs to cut.
    Reported: mean steps per episode over 50 episodes (best alpha), the length of the
    greedy path after learning, and the largest |Q| reached.

Exercise 15 -- How long must the truncation horizon of TTD(lambda) be?
    On the 19-state random walk, lambda = 0.9, run TTD(lambda) for K in {1, 2, 4, 8, 16, 32}
    and compare its best RMS error (first 10 episodes, best alpha) with the offline
    lambda-return algorithm and true online TD(lambda).

Run:  python code/ch06_n_step_and_eligibility_traces/exercise_solutions.py [--quick]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import lambda_methods_random_walk as lm
import random_walk19 as rw
import sarsa_lambda_gridworld as gw


def naive_q_lambda(alpha, lam, n_episodes, rng):
    """Q(lambda) with max-based TD errors but traces that are never cut ("naive" Q(lambda),
    which is Q*(lambda) of Harutyunyan et al. 2016: trace coefficient c = lambda, greedy target).

    Nothing keeps these values bounded: if they overflow we stop the run, mark it as
    diverged and charge MAX_STEPS for every remaining episode."""
    Q = np.zeros((gw.N_S, gw.N_A))
    steps = []
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(n_episodes):
            z = np.zeros((gw.N_S, gw.N_A))
            s, a = gw.S0, gw.eps_greedy(Q, gw.S0, rng)
            for t in range(gw.MAX_STEPS):
                s1, r, term = gw.step(s, a)
                z[s, a] = 1.0
                if term:
                    Q += alpha * (r - Q[s, a]) * z
                    break
                delta = r + gw.GAMMA * Q[s1].max() - Q[s, a]
                Q += alpha * delta * z
                if not np.isfinite(Q).all():
                    steps += [gw.MAX_STEPS] * (n_episodes - len(steps))
                    return steps, Q, True
                a1 = gw.eps_greedy(Q, s1, rng)
                z *= gw.GAMMA * lam                   # never cut
                s, a = s1, a1
            steps.append(t + 1)
    return steps, Q, False


def exercise_14(quick, seed):
    alphas = [0.2, 0.6] if quick else [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    n_runs, n_eps = (2, 10) if quick else (20, 50)
    methods = {
        "SARSA(λ) replacing": lambda al, E, rng: gw.sarsa_lambda(al, 0.9, E, rng, "replacing"),
        "Watkins's Q(λ)": lambda al, E, rng: gw.watkins_q_lambda(al, 0.9, E, rng),
        "naive Q(λ)": lambda al, E, rng: naive_q_lambda(al, 0.9, E, rng),
    }
    print(f"Exercise 14: gridworld, lambda=0.9, runs={n_runs}, episodes={n_eps}, alpha in {alphas}")
    print("  epsilon  method                 best alpha   mean steps/episode   greedy path after learning"
          "   max|Q| (finite runs)")
    for eps in (0.1, 0.3):
        gw.EPS = eps                                   # eps_greedy reads the module-level EPS
        for name, fn in methods.items():
            means, greedy, n_div, q_max = [], [], [], 0.0
            for al in alphas:
                st, gl, dv = [], [], 0
                for run in range(n_runs):
                    rng = np.random.default_rng(10_000 * seed + run)
                    steps, Q, diverged = fn(al, n_eps, rng)
                    st.append(np.mean(steps))
                    if diverged is True:
                        dv += 1
                    else:
                        gl.append(gw.greedy_path_length(Q, rng))
                        q_max = max(q_max, float(np.abs(Q).max()))   # over all alphas and runs
                means.append(np.mean(st)); greedy.append(np.mean(gl) if gl else np.nan); n_div.append(dv)
            j = int(np.argmin(means))
            div = ", ".join(f"α={a:g}: {d}/{n_runs}" for a, d in zip(alphas, n_div) if d)
            print(f"  {eps:6.1f}   {name:22s} {alphas[j]:8.1f}   {means[j]:14.1f}       {greedy[j]:14.1f}"
                  f"      {q_max:10.3g}" + (f"     diverged runs: {div}" if div else ""))
    gw.EPS = 0.1
    p = 0.3 * 3 / 4
    print(f"  (with epsilon = 0.3 an exploratory, non-greedy action occurs with prob. {p:.3f} per step,\n"
          f"   so Watkins's traces survive {1 / p:.1f} steps on average; with epsilon = 0.1, {1 / 0.075:.1f})")
    # Harutyunyan et al. (2016): Q^pi(lambda) with c = lambda (naive Q(lambda) when pi is greedy)
    # contracts if lambda < (1 - gamma) / (gamma * eps_pib), eps_pib = max_s ||pi(.|s) - b(.|s)||_1.
    # For a greedy pi and epsilon-greedy b over 4 actions, ||pi - b||_1 = 2 * (3/4) epsilon = 1.5 epsilon.
    for eps in (0.1, 0.3):
        dist = 2 * eps * (gw.N_A - 1) / gw.N_A
        print(f"  naive Q(lambda) = Q*(lambda): contraction guaranteed only for lambda < "
              f"{(1 - gw.GAMMA) / (gw.GAMMA * dist):.3f} at epsilon = {eps} (we use lambda = 0.9)")


def exercise_15(quick, seed):
    n_runs = 10 if quick else 50
    alphas = np.array([0.2, 0.5]) if quick else np.round(np.arange(0.05, 1.0001, 0.05), 2)
    Ks = [1, 2, 4] if quick else [1, 2, 4, 8, 16, 32]
    lam = np.array([[0.9]])
    al = alphas[None, :]
    episodes = rw.generate_runs(n_runs, 10, seed)

    def best(fn):
        err = np.zeros((1, len(alphas)))
        for run_eps in episodes:
            V = np.zeros((1, len(alphas), rw.N_TOTAL))
            for S, R in run_eps:
                fn(V, S, R)
                err += rw.rms_error(V)
        err /= n_runs * 10
        j = int(np.argmin(err[0]))
        return err[0, j], alphas[j]

    print(f"\nExercise 15: TTD(lambda=0.9) horizon K on the 19-state random walk ({n_runs} runs)")
    print("  method                     best RMS   best alpha")
    for K in Ks:
        e, a = best(lambda V, S, R: lm.ttd_lambda_episode(V, S, R, lam, al, K))
        print(f"  TTD(λ), K = {K:<3d}             {e:.4f}      {a:.2f}")
    e, a = best(lambda V, S, R: lm.offline_lambda_return_episode(V, S, R, lam, al))
    print(f"  offline λ-return            {e:.4f}      {a:.2f}")
    e, a = best(lambda V, S, R: lm.true_online_td_lambda_episode(V, S, R, lam, al))
    print(f"  true online TD(λ)           {e:.4f}      {a:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smoke test (this script writes no figures)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(f"Chapter 06 exercise solutions | seed={args.seed} quick={args.quick}")
    t0 = time.time()
    exercise_14(args.quick, args.seed)
    exercise_15(args.quick, args.seed)
    print(f"\nelapsed {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
