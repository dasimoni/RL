# Chapter 02 code — Multi-Armed Bandits

Companion code for [Chapter 02: Multi-Armed Bandits: Exploration vs Exploitation](../../chapters/02-multi-armed-bandits.md).

Every script runs from the repository root, uses only numpy / scipy / matplotlib, fixes its seeds, prints its settings and a results summary, and accepts `--quick` (a smoke test that writes **no** figures). Full mode writes figures to `figures/`. All agents simulate many independent runs in parallel (arrays of shape `[runs, k]`, one row per run); `bandits.py` checks the vectorized ε-greedy agent against a literal single-run loop.

```bash
python code/ch02_multi_armed_bandits/bandits.py                    # library self-test
python code/ch02_multi_armed_bandits/testbed_greedy_vs_epsilon.py
python code/ch02_multi_armed_bandits/nonstationary.py
python code/ch02_multi_armed_bandits/testbed_optimistic_ucb.py
python code/ch02_multi_armed_bandits/concentration.py
python code/ch02_multi_armed_bandits/thompson_sampling.py
python code/ch02_multi_armed_bandits/gradient_bandit.py
python code/ch02_multi_armed_bandits/parameter_study.py
python code/ch02_multi_armed_bandits/regret_bernoulli.py
python code/ch02_multi_armed_bandits/regret_vs_gap.py
python code/ch02_multi_armed_bandits/exp3_adversarial.py
python code/ch02_multi_armed_bandits/linucb_contextual.py
python code/ch02_multi_armed_bandits/best_arm_identification.py
python code/ch02_multi_armed_bandits/exercise_solutions.py
# add --quick to any of them for a smoke test
```

| Script | Demonstrates (chapter section) | Quick | Full | Headline results (full mode, seed 0) |
|---|---|---|---|---|
| `bandits.py` | environments (Gaussian, Bernoulli, random-walk), agents (ε-greedy, ε_t-greedy, UCB/UCB1, KL-UCB, Boltzmann, explore-then-commit, gradient bandit, Gaussian and Beta Thompson sampling, EXP3), exact two-arm ETC regret, vectorized runner; self-test (§1.5, §3) | 1.3 s | 2.9 s | incremental mean = `np.mean` to 1e-17; loop vs vectorized ε-greedy 1.255 vs 1.258; softmax Jacobian error 3e-11; ETC commits in every run, simulated regret 11.38 ± 0.70 vs exact 11.48 |
| `testbed_greedy_vs_epsilon.py` | greedy vs ε-greedy, S&B Fig. 2.2 style, plus a 10,000-step run (§2.3) | 1.0 s | 12 s | % optimal (steps 901–1000): greedy 35.0%, ε=0.01 58.8%, ε=0.1 79.1%; ε=0.01 overtakes ε=0.1 at ≈ step 6,665 (500-step moving average); E[max q*] = 1.535 (short-run problems), 1.568 (long-run problems) |
| `nonstationary.py` | random-walk q*, sample average vs constant α vs unbiased step, optimistic, stationary TS (§4.4); Part 2: what the unbiased step is for | 1.6 s | 54 s | regret/step (2nd half): sample avg 0.403, constant α 0.166, unbiased 0.165, optimistic 0.252, stationary TS 0.397; with q* starting at N(4,1): % optimal (steps 901–1000) 36.4% (constant α) vs 76.7% (unbiased) |
| `testbed_optimistic_ucb.py` | optimistic initial values (S&B Fig. 2.3), UCB (Fig. 2.4), the step-11 spike (§5, §6.4) | 0.7 s | 4.5 s | optimistic 85.9% vs realistic 76.2% optimal; UCB avg reward 1.387 vs ε-greedy 1.304; spike 42.6% (independent check 41.8%) |
| `concentration.py` | Hoeffding vs exact binomial tails and Chernoff–KL; coverage of the UCB radius (§6.2–6.3) | 1.0 s | 1.8 s | Hoeffding loose ×3.2–10.3 at p=0.5, up to ×8.4e6 at p=0.05; worst P(UCB < p)/δ = 0.18 |
| `thompson_sampling.py` | Beta posteriors evolving under TS; all methods on the testbed (§7.3, §7.5) | 1.9 s | 7.7 s | Gaussian TS (true prior): avg reward 1.466, 92.1% optimal vs UCB 1.387 / 85.8% |
| `gradient_bandit.py` | Monte Carlo check of the gradient derivation; baselines incl. action-dependent ones; S&B Fig. 2.5 (§8) | 1.9 s | 14 s | action-independent baselines unbiased; variance 15.1 (none) vs 1.35 (baseline); b = q*(A) leaves 0.001 of the gradient, noisy Q(A) has cosine −0.05 with it; +4 testbed α=0.1: 84.6% vs 49.6% optimal without baseline |
| `parameter_study.py` | S&B Fig. 2.6 extended with Boltzmann and Thompson sampling; paired s.e. between methods (§10) | 1.9 s | 35 s | best avg reward: UCB 1.476 (c=1, tied with c=1/2), TS 1.467 (s0=1), optimistic 1.453, Boltzmann 1.399, gradient 1.398, ε-greedy 1.333; paired differences UCB−TS 0.0094 ± 0.0009, TS−optimistic 0.0143 ± 0.0024 |
| `regret_bernoulli.py` | linear vs log regret, Lai–Robbins, UCB1 bound, regret decomposition, ETC expectation (§11) | 2.6 s | 84 s | Reg(20,000): TS 50.1, KL-UCB 75.8, ε_t 172.9, ETC (m=100) 248.2 ± 16.6 (expectation 237, 7.5% of runs commit wrongly), UCB1 281.5, ε=0.1 434.9, greedy 3230; Lai–Robbins 9.46 ln T = 93.7 |
| `regret_vs_gap.py` | regret vs gap, worst case ∝ √T; exact ETC regret curve (§11.7) | 2.9 s | 50 s | worst-case regret/√T: TS 0.43, 0.42, 0.42; UCB1 0.85 → 1.00; ETC m=50 (exact) 25.0 / 70.0 / 273.0, interior peak ×3.62, ×3.91 per ×4 in T |
| `exp3_adversarial.py` | adversary built against deterministic UCB1; regime switch; EXP3 on stochastic arms (§12) | 1.2 s | 29 s | UCB1 regret 0.667T vs EXP3 10.3; EXP3 regret ×1.97, ×1.94 per ×4 in T (√T) vs TS ×1.28, ×1.24 |
| `linucb_contextual.py` | disjoint LinUCB, linear TS, linear ε-greedy vs context-free UCB (§13) | 0.9 s | 8 s | Reg(5000): LinUCB α=1 73.9, linear TS 104.3, greedy ridge 186.5, linear ε-greedy 380.7, context-free 1016.4 |
| `best_arm_identification.py` | fixed-budget BAI (uniform, sequential halving, UCB1/TS recommendations), successive elimination (§15) | 0.7 s | 48 s | P(error) at n=5000: uniform 6.8%, halving 3.0%, UCB1 0.9%, TS 1.7%; successive elimination (δ=0.05): 0 errors, ≈58,000 samples |
| `exercise_solutions.py` | Exercises 2.9 (exact ETC regret), 2.13 (nonstationary parameter study), 2.14 (discounted TS) | 9.1 s | 108 s | ETC optimum m=276 (regret 36.1); closed-form bound-optimal m≈783 (78.3), exact bound minimiser m=759; best nonstationary: ε-greedy α=0.1 1.823; discounted TS γ_f=0.999 1.251 vs tuned ε-greedy (ε=1/32) 1.242, paired +0.009 ± 0.003 |

Runtimes were measured on one CPU core of a shared machine (Python 3.11, numpy 2.4, scipy 1.17, matplotlib 3.11).

Figures produced in full mode (`figures/`):

- `testbed_greedy_vs_epsilon.png`, `testbed_optimistic_ucb.png`, `testbed_all_methods.png`, `parameter_study.png`
- `nonstationary.png`, `concentration.png`, `thompson_posteriors.png`, `gradient_bandit.png`
- `regret_bernoulli.png`, `regret_vs_gap.png`, `exp3_adversarial.png`
- `linucb_contextual.png`, `best_arm_identification.png`, `exercise_solutions.png`

Notes

- All other scripts import `bandits.py`, so run them from the repository root as shown (Python puts the script's folder on `sys.path`).
- Comparisons use common random numbers: every agent in a comparison faces the same sampled problems.
- `ThompsonBeta`, `KLUCB` and `EXP3` assume rewards in [0, 1] (`ThompsonBeta` needs {0, 1}); use `ThompsonGaussian` for Gaussian rewards.
- `KLUCB` computes its index by 16 bisection steps (precision 2^-16) with exploration level ln t (c' = 0 in Eq. 11.7); it is the slowest agent (~45 s of `regret_bernoulli.py`).
- `ExploreThenCommit` fixes its arm once, at step m k + 1 (Algorithm 11.1). Re-taking argmax Q after exploration would be a different, self-correcting algorithm.
- Each script uses its own agent random stream, so the same configuration can differ by about one percentage point between scripts (Monte Carlo noise).
