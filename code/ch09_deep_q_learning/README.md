# Chapter 09 code: Deep Q-Networks and value-based deep RL

Companion code for [Chapter 09 — Deep Q-Networks and Value-Based Deep RL](../../chapters/09-deep-q-learning.md).

Run everything from the repository root with `python code/ch09_deep_q_learning/<script>.py`. Every script prints its seed(s) and settings and accepts `--quick`, a smoke test that writes no figures. In full mode it saves figures to `figures/`. All PyTorch code runs on one CPU thread (`torch.set_num_threads(1)`), and all results are deterministic given the seeds (re-running a script reproduces its numbers exactly).

**About the runtimes.** They were measured on one thread of a shared 4-CPU machine while other jobs were running (load average between 1.5 and 5.6). The multi-seed scripts took between 4.5 and 8.1 minutes under that load. On an idle core they should be faster, but we did not measure that.

| Script | What it demonstrates | Quick | Full | Headline results (full mode) |
|---|---|---|---|---|
| `dqn.py` | Library and demo. DQN from scratch (Algorithm 9.1): plain, dueling and noisy Q-networks; uniform replay; sum-tree prioritized replay (Algorithm 9.2); $n$-step accumulator (Algorithm 9.3); Double DQN target; Polyak or hard target updates; greedy evaluation that compares predicted $Q$ with Monte Carlo returns. The demo trains one agent on CartPole-v1 for 50k steps. Flags: `--double --dueling --noisy --n-step 3 --prioritized --polyak 0.005 --no-use-target --seed 1 ...` (every boolean has a `--no-` form) | 6 s | 60 s (`--noisy`: 85–98 s) | seed 0: greedy score 500 at 25k–30k and 45k steps, collapses in between; final $Q$ 98.3 vs Monte Carlo 37.7 after a collapse. `--noisy`, seeds 0–4: final score 281.1 ± 196.8; 4 of 5 seeds reach 500 at some point; on 2 (seeds 1 and 2) the values exceed 100 (183.6, 113.1) and the policy collapses, seed 2 after scoring 500. `--noisy --noisy-bug` (our first, broken version: no random warm-up, noise drawn only at gradient steps): score 9.2 / 9.4 at every evaluation on seeds 0 / 1 |
| `dqn_batched.py` | $K$ independent DQN agents with stacked weights, trained in lock-step: equivalent in distribution to $K$ independent runs of `dqn.py` (same algorithm and hyperparameters, independent weights and data), but not the same random streams as `dqn.py --seed k`. Used by all multi-seed experiments. Also records $\max_a Q$ at recent "fallen" next states (environment `terminated=True`) | 7 s | 6 s | after 200 identical updates, \|Q_batched − Q_single\| ≤ 3.6e-7 (0 for the plain net); one update: 0.86 ms (1 `dqn.py` agent) vs 0.94 ms (5 batched agents) vs 1.30 ms (10) |
| `compare_variants.py` | DQN, Double, Dueling, Double+Dueling on CartPole, 5 seeds × 50k steps | 13–17 s | 484 s | final score 386.5 ± 195.7 / 380.5 ± 95.0 / 315.1 ± 137.9 / 343.7 ± 149.8 (sample SD): no significant differences; final mean $Q$ 90.8 / 80.5 / 86.1 / 76.2. The dueling net has ~8.8k parameters vs ~4.6k (not capacity-matched) |
| `overestimation.py` | (A) the tight lower bound $\sqrt{\sigma^2/(m-1)}$ (printed check) and the max-estimator vs double-estimator bias vs $m$ (figure); (B) DQN vs Double DQN with 2 or 8 (redundant) actions: predicted $Q$ vs Monte Carlo returns, fraction of impossible values $Q>100$ | 12 s | 384 s | states with $Q>100$ in the last 10k steps: DQN 19.7% (2 actions), 39.5% (8 actions); Double DQN 0.0% and 8.5%. Scores not improved |
| `ablate_stabilizers.py` | No target network, no replay (target network kept), neither (= Eq. 9.1 with Adam and Huber), target period 1000 | 12 s | 483 s | without a target network $Q$ explodes to ~1e8 and the score is 9.2; no replay: 133.8 with $Q$=119 (impossible); period 1000: stable but slow; mean $Q$ follows $(1-\gamma^k)/(1-\gamma)$ to within ~11% over the whole run |
| `pitfalls.py` | Ignoring terminal states; treating truncation as termination; MSE vs Huber; $\max_a Q$ at the never-trained "fallen" states | 11 s | 425 s | ignore-terminal: score 9.2, $Q\approx3\times10^6$ at 50k, fallen states valued above visited ones at every evaluation; truncation bug: no measurable effect here (395.2 vs 396.6); MSE at least as good as Huber (mean over run 279.1 vs 213.5), and it values fallen states at 2.5–5.5 vs 21–34 for Huber |
| `prioritized_replay.py` | (A) Blind Cliffwalk, uniform vs proportional PER (no IS weights: all targets are deterministic); (B) fixed point of PER with and without IS weights vs Eq. (9.12) | 8 s | 81 s | $n=14$ (32,766 transitions): 433k updates (uniform) vs 65k ($\alpha$=0.6) and 52k ($\alpha$=1), i.e. 6.7× and 8.4× fewer; PER without IS: Q = 2.52 (predicted 2.50, true 1.0) |
| `distributional.py` | C51 (Algorithm 9.4) and QR-DQN (Algorithm 9.5); learned return distributions vs Monte Carlo ground truth on FrozenLake (slippery, 20k steps per agent), and at probe states on CartPole (40k steps per agent) | 17–19 s | 373 s at load ≈2–4.5; 497 s at load ≈5 | FrozenLake $W_1$ to the Monte Carlo distribution (start state / next to the goal): C51 0.060 / 0.013; QR-DQN 0.244 / 0.146 with $\kappa$=1 (it learns expectiles), 0.082 / 0.028 with $\kappa$=0.01; CartPole: QR quantiles extrapolate to impossible values (−117 to +65) at a rare state, C51 stays within its support (0 to 4) |
| `exercise_solutions.py` | Numbers used in the exercise solutions, including 3-step DQN (Ex 9.13) and Polyak averaging (Ex 9.14), 5 seeds each | 2 s | 270 s | 3-step DQN: final score 498.7 ± 2.9 (vs 396.6 ± 193.2 for 1-step), mean $Q$ follows $(1-\gamma^{3k})/(1-\gamma)$ early; Polyak $\tau$=0.008: values grow ~2× faster than with hard copies (54.7 vs 32.6 at 10k steps) and overshoot to 109.7; final score 257.7 ± 199.9 |
| `plot_style.py` | Shared matplotlib style (library, no `__main__`) | — | — | — |

(± is always the sample standard deviation over seeds, `ddof=1`; shaded bands in the learning-curve figures are ±1 standard error.)

## Figures (written in full mode)

| File | Script | Chapter section |
|---|---|---|
| `dqn_demo.png` | `dqn.py` (default flags; other flags overwrite it) | In code |
| `compare_variants.png` | `compare_variants.py` | 5.3 |
| `overestimation.png` | `overestimation.py` | 4.4 |
| `ablate_stabilizers.png` | `ablate_stabilizers.py` | 2.9 |
| `pitfalls.png` | `pitfalls.py` | 14 |
| `prioritized_replay.png` | `prioritized_replay.py` | 6.4 |
| `distributional_frozenlake.png`, `distributional_cartpole.png` | `distributional.py` | 9.6 |

## Notes

* **Termination vs truncation.** Every agent stores `terminated` (not `terminated or truncated`) as the terminal flag, and resets on either. Truncated transitions keep the real last observation as $s'$. The `terminal_bug` option in `Config` exists only to reproduce the two bugs in `pitfalls.py`.
* **Noisy Nets.** A noisy agent has no $\varepsilon$, so `train()` acts uniformly at random until learning starts, and `DQNAgent.act` draws fresh weight noise before every action (the update draws independent noise for the online and target networks). The `noisy_bug` option exists only to reproduce the broken first version discussed in Section 8. The $\varepsilon$-greedy agents need no special warm-up: $\varepsilon\ge0.9$ during the first 1,000 steps.
* **Measuring overestimation.** Agents bootstrap through the 500-step limit, so $Q$ estimates values of the task without a time limit. Evaluation episodes therefore run to 1,000 steps, and the Monte Carlo returns are computed only for states before step 500 (ignored tail ≤ 0.66).
* **Why batched agents?** On a CPU, tiny networks are dominated by per-call overhead. `dqn_batched.py` stacks $K$ networks into `(K, n_in, n_out)` weight tensors. The loss is the sum of the members' mean losses and Adam is elementwise, so the members are independent, and `--quick` checks the update numerically against `dqn.py`. PER and Noisy Nets are only implemented in the single-agent `dqn.py`.
* **Default hyperparameters** (`dqn.Config`): 4-64-64-2 ReLU MLP, Adam with step size 5e-4, batch 64, replay 50k, learning starts at 1k, one update per step, hard target copy every 250 steps, $\varepsilon$ from 1 to 0.05 over 10k steps, Huber loss, $\gamma=0.99$, 50k environment steps. They were chosen once and not tuned per variant.
