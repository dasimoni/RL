"""Why naive neural Q-learning is unstable: ablating experience replay and the target network.

Chapter 09, Sections 1.3, 2.3 and 2.9 (and the target-period discussion of Section 14).

Five configurations, 5 seeds each, 50k CartPole-v1 steps, otherwise the dqn.Config defaults:
  DQN (replay + target)           Algorithm 9.1: replay of 50k transitions, target copied every 250 steps
  no target network               bootstrap from the online network itself (target period 1)
  no replay (online, batch 1)     update on the newest transition only, once per step (but keep
                                  the target network, period 250)
  neither (naive)                 online updates AND no target network: semi-gradient Q-learning,
                                  Eq. (9.1), with a neural network, Adam and the Huber loss
  target period 1000              replay + target network, but the target is copied 4x less often

Output (full mode): figures/ablate_stabilizers.png
Run:  python code/ch09_deep_q_learning/ablate_stabilizers.py [--quick]
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import time

from dqn import Config, FIG_DIR
from dqn_batched import plot_suite, print_suite_summary, run_suite

CONFIGS = {
    "DQN (replay + target, period 250)": dict(),
    "no target network": dict(use_target=False),
    "no replay (online, batch 1)": dict(buffer_size=1, batch_size=1),
    "neither (naive neural Q-learning)": dict(buffer_size=1, batch_size=1, use_target=False),
    "target period 1000": dict(target_update=1000),
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
        plot_suite(results, os.path.join(FIG_DIR, "ablate_stabilizers.png"), K,
                   "Removing DQN's stabilisers on CartPole-v1", q_log=True)


if __name__ == "__main__":
    main()
