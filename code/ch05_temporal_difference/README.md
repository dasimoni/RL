# Chapter 05 code — Temporal-Difference Learning

Companion code for [Chapter 05: Temporal-Difference Learning: SARSA, Q-Learning and Friends](../../chapters/05-temporal-difference.md).

Every script runs from the repository root, uses only numpy / matplotlib / gymnasium (toy_text), fixes its seeds, prints its settings and a results summary, and accepts `--quick` (a smoke test that writes **no** figures). Full mode writes figures to `figures/`.

```bash
python code/ch05_temporal_difference/random_walk_td_vs_mc.py      # add --quick for a smoke test
python code/ch05_temporal_difference/batch_td_vs_mc.py
python code/ch05_temporal_difference/cliff_walking.py
python code/ch05_temporal_difference/cliff_alpha_sweep.py
python code/ch05_temporal_difference/maximization_bias.py
python code/ch05_temporal_difference/taxi_q_learning.py
python code/ch05_temporal_difference/q_learning_step_sizes.py
python code/ch05_temporal_difference/conditioning_td.py
python code/ch05_temporal_difference/exercise_solutions.py
```

| Script | Demonstrates (chapter section) | Quick | Full | Headline results (full mode, seed 0) |
|---|---|---|---|---|
| `random_walk_td_vs_mc.py` | TD(0) vs constant-α MC on the 5-state random walk, S&B Example 6.2 (§2–3); effect of initial values on the TD learning curve (Ex. 5.9) | 0.6 s | 9 s | RMS error after 100 episodes (500 runs): best TD(0) 0.035 (α=0.05), best MC 0.082 (α=0.02) |
| `batch_td_vs_mc.py` | "You are the predictor" (S&B Ex. 6.4), certainty equivalence, batch TD vs batch MC learning curves (S&B Fig. 6.2) (§4) | 0.8 s | 43 s | V(A): batch TD 0.75, batch MC 0.00; batch TD = closed-form CE solution to 2e-9; batch TD lower RMS at 99/99 batch sizes (0.039 vs 0.056 at 100 episodes) |
| `cliff_walking.py` | SARSA vs Q-learning vs Expected SARSA on cliff walking, S&B Ex. 6.6 (§10); exact ε-soft value iteration (the common fixed point of SARSA and Expected SARSA); stuck greedy read-outs; SARSA's greedy route vs step size | 0.5 s | 22 s | online reward, last 100 episodes: SARSA −27.8, Q-learning −51.1, Expected SARSA −21.1; greedy path: Q-learning optimal (−13) in 100/100 runs, Expected SARSA row 1 (−15) in 100/100, SARSA mostly row 0 (16/100 read-outs stuck: 9 wall bumps, 7 two-cell cycles); best ε-greedy policy = row 1, exact return −20.71 (row 0: −21.61, edge: −45.71; ε-greedy around q*: −50.80); SARSA's route is row 1 in 10/20 seeds at α=0.1 (5,000 ep.) and 18/20 at α=0.05 (10,000 ep.) |
| `cliff_alpha_sweep.py` | interim and long-run performance vs α (in the spirit of S&B Fig. 6.3) (§10.3) | 0.7 s | 34 s | Expected SARSA best at every α (long-run ≈ −21); SARSA long-run collapses to −660 at α=1; Q-learning long-run ≈ −51 for all α |
| `maximization_bias.py` | maximization bias MDP, S&B Ex. 6.7 (10 actions in B; S&B say only "many"); Q-learning vs Double Q-learning (+ SARSA, Expected SARSA); update counts and bootstrap targets at episode 300; 20,000-episode runs under ε-greedy and uniformly random behaviour in A and/or B (§11.4–11.5) | 2 s | 110 s | peak % left: Q-learning 94.0%, SARSA 87.0%, Expected SARSA 67.2%, Double Q 50.9% (initial tie); by episode 300 Q-learning updated Q(A,left) 113 times (Double Q: 17.5 per table, and only 1.7 per Q_i(B,b)); Q(A,left) at episode 20,000 (true −0.1): ε-greedy behaviour Q-learning −0.132, Double Q −0.336 (both below the truth); uniform behaviour Q-learning +0.257 (pure max bias), Double Q −0.102 (unbiased) |
| `taxi_q_learning.py` | Gymnasium Taxi-v4 loop with terminated/truncated handling; optimal values by value iteration on `env.unwrapped.P`; greedy policy evaluated every 100 episodes; truncation bug on Taxi and on a toy `TimeLimit` env (§13) | 8 s | 52 s | greedy return 7.93 = optimal (99.9% of start states exactly optimal); behaviour plateaus at 2.45 from ~episode 1,500 while the greedy policy goes −130.6 (ep. 1,000) → −20.4 (2,000) → 5.75 (3,000) → 7.93 (5,000); bug harmless on Taxi, but flips the toy's policy (Q(stay) 5.94 vs correct 10.00) |
| `q_learning_step_sizes.py` | Robbins–Monro step sizes vs constant α, error measured against exact q* (§8.4) | 3 s | 65 s | ‖Q−q*‖∞ after 4·10⁵ steps: 1/n^0.6 → 0.10, 1/n^0.8 → 0.14, 1/n → 1.53, const 0.02 → 0.35, const 0.1 → 0.82 |
| `conditioning_td.py` | psychology and neuroscience links (§14): blocking under Rescorla–Wagner and the CSC TD model; TD errors over acquisition and on US omission; second-order conditioning (Ex. 5.16); the two-step task with model-free, model-based and hybrid agents (stay probabilities) | 0.2 s | 2 s | blocking: w_X = 0.000 (RW) and 0.000 (TD) vs control 0.500 (RW) and 0.387 = half of 0.774 (TD); TD error at the cue 0.735 = γ^6 and 0.000 at the US after 200 trials, −1.000 at the expected US time on omission; second-order: TD V(B) peaks at 0.685 after 30 phase-2 trials, RW w_B → −0.5 (never positive); two-step (2,000 agents × 201 trials) reward effect / interaction: MF +0.331 / +0.019, MB −0.001 / +0.420, hybrid +0.240 / +0.184 |
| `exercise_solutions.py` | coding exercises 5.12 (SARSA on cliff walking with per-episode ε decay vs per-state GLIE ε = 1/√n(s) with α = 1/n^0.6) and 5.13 (α = 1 on deterministic Taxi) | 9 s | 142 s | no schedule finds the 13-step path in 5,000 or 50,000 episodes; per-episode decay stops exploring the cliff edge (Q((2,1), right) stays ≈ −19.5 vs q* = −11); α=1: Q-learning 7.70, Expected SARSA 7.73, SARSA −430 |

Runtimes were measured on one CPU core of a shared machine (Python 3.11, numpy 2.4, gymnasium 1.3); expect them to vary by up to about ±50%.

Figures produced in full mode:

- `figures/random_walk_values.png`, `figures/random_walk_rms.png`
- `figures/batch_td_vs_mc.png`
- `figures/cliff_rewards.png`, `figures/cliff_paths.png`, `figures/cliff_alpha_sweep.png`
- `figures/maximization_bias.png`, `figures/maximization_bias_long.png`
- `figures/taxi_learning_curve.png`
- `figures/q_learning_step_sizes.png`
- `figures/conditioning_td.png`, `figures/two_step_task.png`

Notes

- `cliff_alpha_sweep.py` and `exercise_solutions.py` import the environment and agents from `cliff_walking.py` (and `taxi_q_learning.py`), so run them from the repository root as shown (Python puts the script's folder on `sys.path`). They set `sys.dont_write_bytecode = True` so that these imports leave no `__pycache__/` behind.
- Greedy read-outs used for evaluation (`greedy_path` in `cliff_walking.py`, `greedy_rollout_return` in `taxi_q_learning.py`) are deterministic and break ties by the lowest action index; the learning agents break ties at random.
- `batch_td_vs_mc.py` iterates batch updates with a per-state step size η/n(s); the fixed point is independent of the step size, and the script checks it against the closed-form certainty-equivalence solution.
- `conditioning_td.py` implements Rescorla–Wagner at the trial level and the TD model of conditioning in real time (20 steps per trial, CS onset at step 6, US 6 steps later, γ = 0.95, α = 0.1, one CSC feature per stimulus and time since onset). Parts A, B and D are deterministic; Part C (two-step task) uses `--seed`. Its model-free agent uses λ = 1 (stage-1 value updated towards the reward), the model-based agent the known 0.7/0.3 transition model, and reward probabilities drift with s.d. 0.025, reflected into [0.25, 0.75].
- `maximization_bias.py` simulates 10,000 independent learners in parallel with numpy (one row per run). Its `behaviour` option changes only how actions are *chosen* in A and B (ε-greedy or uniformly random); the update rules stay the same. Uniform behaviour separates the estimators' own bias from the effect of value-dependent revisiting with a constant step size (§11.5).
