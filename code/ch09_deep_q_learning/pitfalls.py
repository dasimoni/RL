"""Two classic DQN bugs and one design choice, measured: terminal handling and the loss.

Chapter 09, Section 14 (debugging DQN).  5 seeds each, 50k CartPole-v1 steps, dqn.Config defaults.

  correct                          store `terminated`; bootstrap through time-limit truncation
  BUG: ignore terminal states      store False always: the target bootstraps through the fall,
                                   y = 1 + gamma max_a Q(S_fallen, a), so falling costs nothing
  BUG: truncation = termination    store `terminated or truncated`: the state at the 500-step
                                   limit is treated as worth 0, although nothing bad happened
  MSE instead of Huber             squared loss 0.5 delta^2 instead of the Huber loss (Eq. 9.5)

Output (full mode): figures/pitfalls.png
Run:  python code/ch09_deep_q_learning/pitfalls.py [--quick]
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import time

from dqn import Config, FIG_DIR
from dqn_batched import plot_suite, print_suite_summary, run_suite

CONFIGS = {
    "correct": dict(),
    "BUG: ignore terminal states": dict(terminal_bug="ignore_terminal"),
    "BUG: truncation = termination": dict(terminal_bug="truncation_is_terminal"),
    "MSE loss instead of Huber": dict(loss="mse"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    base, K = Config(seed=0, eval_every=5_000), 5
    if args.quick:
        base = dataclasses.replace(base, total_steps=2_000, eval_every=1_000, learning_starts=500,
                                   eps_decay_steps=1_000)
        K = 2
    print(f"Seeds {base.seed}..{base.seed + K - 1}; base config: {dataclasses.asdict(base)}")
    t0 = time.perf_counter()
    results = run_suite(CONFIGS, base, K)
    print_suite_summary(results)
    print(f"(final = mean over the last 2 evaluations, i.e. the last 10k steps)\nTotal runtime "
          f"{time.perf_counter() - t0:.0f}s")
    if not args.quick:
        os.makedirs(FIG_DIR, exist_ok=True)
        plot_suite(results, os.path.join(FIG_DIR, "pitfalls.png"), K,
                   "Terminal handling and the loss function (CartPole-v1)", q_log=True)


if __name__ == "__main__":
    main()
