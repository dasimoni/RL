# Chapter 20 code: deep RL in practice

Companion code for [Chapter 20 — Deep RL in Practice: Engineering, Debugging, Evaluation and Safety](../../chapters/20-deep-rl-in-practice.md).

Run everything from the repository root. Every script prints its seed and settings, accepts `--quick` (a smoke test that writes **no** figures), and in full mode writes figures to `figures/`. PyTorch scripts call `torch.set_num_threads(1)`. Runtimes are wall-clock times on one CPU thread of a shared 4-CPU machine with other jobs running, so expect some variation.

```bash
python code/ch20_deep_rl_in_practice/reward_shaping.py      # add --quick for a smoke test
python code/ch20_deep_rl_in_practice/rl_stats.py
python code/ch20_deep_rl_in_practice/evaluate_agents.py
python code/ch20_deep_rl_in_practice/probe_envs.py           # --long-p6 for the 30k-step P6 diagnostic
python code/ch20_deep_rl_in_practice/seed_variance.py
python code/ch20_deep_rl_in_practice/cartpole_a2c.py
python code/ch20_deep_rl_in_practice/robust_vi.py
```

| Script | What it demonstrates (chapter section) | Quick | Full | Headline results (full mode) |
|---|---|---|---|---|
| `reward_shaping.py` | Potential-based vs naive shaping on a 22-step zig-zag gridworld with a pit next to the start: exact optimal policies by value iteration, Q-learning with 50 seeds × 150k steps, and Wiewiora's shaping = Q-initialization equivalence (2.3–2.5) | 2 s | 63–77 s | all five potentials leave the optimal policy unchanged (max \|q'* − (q* − Φ)\| = 7.1e-15); progress bonus: agent dithers next to the goal forever (its greedy policy's shaped return rises 3.26 → 4.09 while the true return stays 0); dense distance penalty and the "Φ(terminal) not zeroed" bug: agent jumps into the pit. Learning: Φ ≈ v* → 100% success by 2,500 steps; sparse → 90% of seeds by 117,500; Φ = −0.1d → 2% by 150k; Φ = −0.1d + 20 → 100% by 60k. PBRS and Q-init with Φ take identical actions (max gap 3.3e-14) |
| `rl_stats.py` | From-scratch IQM, median of task means (rliable's definition) and pooled median, optimality gap, stratified bootstrap CIs, performance profiles, probability of improvement; self-tests against scipy and the worked example (6.2–6.6) | 1 s | 2 s | worked example reproduced exactly (median of task means 0.6125, pooled median 0.5, IQM 0.525, P(X>Y) 0.5625); IQM = `scipy.stats.trim_mean(x, 0.25)`; rank-based P(X>Y) = brute force = Mann–Whitney U/(nk); percentile-CI coverage for a Gaussian mean: 84.8% with 3 runs/task, 92.0% with 10 (1,000 trials, Monte Carlo SE ≈ 1 point) |
| `evaluate_agents.py` (+ `tabular_suite.py`) | Q-learning, Double Q, Expected SARSA, SARSA × 5 toy-text tasks × 100 runs (exact greedy-policy scores, normalized with lo = best trivial policy, hi = optimal), reported with the Section 6 toolkit; the "statistical precipice" by subsampling (6.7) | 6 s | 170–175 s | with lo = the random policy, trivial policies would already score 0.733 (Taxi) and 0.964 (slippery Cliff Walking), hence the trivial-policy reference; 10-run IQM [95% CI]: Q-learning 0.646 [0.577, 0.704] (100 runs: 0.580), Double Q −0.215, Expected SARSA 0.547, SARSA 0.559; P(Q > Expected SARSA) = 0.616 [0.510, 0.720]; IQM sd across 3-run subsets 0.069 vs 0.121 (median of task means) and 0.127 (pooled median); with 3 runs/task the IQM CI covers 80% and Q-learning vs SARSA is ranked wrongly 19% of the time (20 runs: 2%) |
| `probe_envs.py` | Seven probe environments (constant, obs-dependent, discounting, action-dependent, action+obs, delayed credit, time limit) unit-testing an A2C and a DQN, a correct version and ten planted bugs, plus a static gradient-isolation check that runs the agents' own loss code (7.2, 8). `--long-p6` adds a 30k-step P6 diagnostic | 8 s | 166 s (177 s with `--long-p6`) | correct agents pass all probes; missing done mask → V(0) = 10.0 on P1; truncation-as-termination caught only by P7 (V = 1.90 / 2.56 instead of 10); done off-by-one passes P1, fails P3 (V(0) = 0.00); shape broadcast fails P2; target from the online net with gradients passes every probe, caught only by the gradient check; advantage-not-detached fails P6 marginally (V(1) = 1.13 vs 1.0 at 6k steps, 0.999 at 30k) and the gradient check |
| `seed_variance.py` (+ `cartpole_a2c.py`) | A2C on CartPole, 80k steps, two summaries per run (final greedy-policy evaluation on a separate env; learning performance = step-weighted mean training return): two groups of 5 seeds; bit-for-bit determinism; varying one random stream at a time; learning-rate sweep; diagnostics dashboard (5.4, 7.4) | 8 s | 233–239 s | final performance: 9 of 10 seeds at 500 (seed 7: 402.6); learning performance: 158.4–334.8, seeds 0–4 vs 5–9: 291.7 vs 253.8 (Welch p = 0.34; 4 of 126 splits reach p < 0.05); rerun identical bit for bit; learning-performance sd 27.3 (init only), 35.3 (env only), 43.9 (actions only), 57.6 (all); lr 3e-4/1e-3/3e-3/1e-2 → final means 215.1/500/500/269.3, learning means 198.1/291.7/383.0/311.9; the lr 1e-2 seed-3 run: final 20.8, last-10k training return 104.5 episode-weighted vs 390.8 step-weighted; critics overshoot 1/(1−γ) = 100 (V(s0) up to 106) |
| `cartpole_a2c.py` | The instrumented A2C used above; run directly for one training run with printed diagnostics and a final-policy evaluation | 3 s | 10–13 s | seed 0: mean return of the last 20 training episodes 219.0 (its curve oscillates); final policy: greedy 500.0, stochastic 480.8 |
| `robust_vi.py` | Robust MDPs on a 5×6 slippery gridworld (short route along four lava cells vs an 11-step detour over the top, exposed to the lava only if its first step slips; slip ξ to each perpendicular direction with probability ξ/2; nominal ξ̂ = 0.05, γ = 0.9): nominal VI vs robust VI with (s,a)-rectangular L1 and KL balls (restricted to the nominal support), contraction check, inner solvers checked against an LP and the KL dual, exact evaluation of every policy for true ξ ∈ [0, 0.5], and a breakdown of the gap between the robust value and the worst slip model (11.6; Exercises 20.16–20.17) | 4–5 s | 16–17 s | nominal policy takes the lava route, v(S) = 0.505; robust policies take the detour for L1 κ ≥ 0.1 and KL κ ≥ 0.05; worst-case lava probability at an edge cell 0.025 → 0.125 (L1, κ = 0.2) / 0.102 (KL, κ = 0.0703); contraction ratio ≤ 0.898 < γ; sort vs LP 5.6e-16, KL primal vs dual 1.6e-15 (2,000 instances); both balls cover ξ ∈ [0, 0.15]; robust value 0.255 (L1) / 0.270 (KL) ≤ worst true value 0.272 / 0.281 over that range (nominal policy: 0.226); a different symmetric slip ξ(s) ∈ [0, 0.15] in every state gives the same 0.272 / 0.281, the worst single slip model drifting to one side of the heading inside the ball 0.270 / 0.274; robust policy ahead for ξ ≥ 0.13 (where the optimal route also switches); cost at ξ̂ 0.179 (0.505 vs 0.326), benefit at ξ = 0.3 0.291 (0.169 vs −0.123) |
| `tabular_suite.py`, `plot_style.py` | Libraries (no main block): vectorized tabular tasks and agents with exact evaluation; shared figure style | — | — | — |

## Figures (written in full mode)

| File | Script | Chapter section |
|---|---|---|
| `reward_shaping.png` | `reward_shaping.py` | 2.5 |
| `seed_variance.png` | `seed_variance.py` | 5.4 |
| `eval_aggregates.png`, `eval_profiles_poi.png`, `eval_precipice.png`, `eval_curves.png` | `evaluate_agents.py` | 6.7 |
| `probe_results.png` | `probe_envs.py` | 7.2 |
| `diagnostics_dashboard.png` | `seed_variance.py` | 7.4 |
| `robust_vi.png` | `robust_vi.py` | 11.6 |

## Notes

* `tabular_suite.py` reads each toy-text task's exact transition tables (`env.unwrapped.P`) and steps all runs of an agent at once with numpy, so 100 runs cost about as much as one. Scores are exact expected H-step returns of the greedy policy (finite-horizon policy evaluation), so they contain no evaluation noise.
* Every agent bootstraps through time-limit truncation and not through termination. The probe suite and the shaping script show what happens when that is done wrong.
* The diagnostics dashboard deliberately shows the *worst* of five seeds at the largest learning rate next to a run at the best learning rate (ties in final performance broken by learning performance); the selection is stated in the figure title.
* `evaluate_agents.py` also prints, for Q-learning's 100-run pool, the per-task 5–95% ranges, the bounds of the middle half that the IQM averages, and the mean of each estimator over subsamples (its bias), which Section 6.7 quotes.
* All results are deterministic given the seeds on this machine (CPU, one thread); a different PyTorch or BLAS build may change low-order digits and hence individual runs.
* `robust_vi.py` solves the L1 inner problem by sorting and the KL inner problem by exponential tilting with a vectorized bisection on the dual variable; Part 2 checks both against `scipy.optimize.linprog` and a direct maximization of the dual (20.21c). The uncertainty sets are restricted to the support of the nominal kernel, so the adversary can change slip probabilities and directions but cannot move the agent to a cell the slip model never reaches.
