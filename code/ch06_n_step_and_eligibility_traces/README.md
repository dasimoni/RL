# Chapter 06 code — n-Step Bootstrapping and Eligibility Traces

Companion code for [Chapter 06: n-Step Bootstrapping and Eligibility Traces](../../chapters/06-n-step-and-eligibility-traces.md).

Every script runs from the repository root, uses only NumPy and Matplotlib, fixes its seeds, prints its settings and a results summary, and accepts `--quick` (a smoke test that writes **no** figures). Full mode writes figures to `figures/`.

```bash
python code/ch06_n_step_and_eligibility_traces/backup_diagrams.py              # add --quick for a smoke test
python code/ch06_n_step_and_eligibility_traces/n_step_td_random_walk.py
python code/ch06_n_step_and_eligibility_traces/return_bias_variance.py
python code/ch06_n_step_and_eligibility_traces/lambda_methods_random_walk.py
python code/ch06_n_step_and_eligibility_traces/equivalence_checks.py
python code/ch06_n_step_and_eligibility_traces/trace_visualization.py
python code/ch06_n_step_and_eligibility_traces/sarsa_lambda_gridworld.py
python code/ch06_n_step_and_eligibility_traces/off_policy_traces.py
python code/ch06_n_step_and_eligibility_traces/vtrace_fixed_point.py
python code/ch06_n_step_and_eligibility_traces/exercise_solutions.py
```

| Script | Demonstrates (chapter section) | Quick | Full | Headline results (full mode, seed 0) |
|---|---|---|---|---|
| `backup_diagrams.py` | schematic Figure 6.2: backup diagrams of $n$-step TD, 3-step SARSA / Expected SARSA / Tree Backup, and the λ-return weights (§2.1, 3, 6.1, 8.1); nothing is learned | 0.2 s | 1 s | weights for λ = 0.8, $T - t = 10$: 0.200, 0.160, …, 0.034 on $G_{t:t+1}, \dots, G_{t:t+9}$ and 0.134 on $G_t$ (sum checked to be 1) |
| `n_step_td_random_walk.py` | $n$-step TD on the 19-state random walk, S&B Figure 7.2 (§2.5) | 0.7 s | 5 s | RMS error over first 10 episodes, 100 runs: best $n = 4$ (α = 0.4) 0.262; TD(0) 0.344; $n = 512$ 0.485; no learning 0.548 |
| `return_bias_variance.py` | exact error-reduction property (§2.3); bias², variance and MSE of $n$-step and λ-return targets (§15.1) | 0.8 s | 9 s | all error ratios at or below the survival bound, which is exactly 1 for $n \leq 8$ (e.g. 0.889 at $n = 1$, 0.029 at $n = 64$ for $V = 0$); min-MSE target: $n = 16$ / λ = 0.95 for $V = 0$, $n = 1$ / λ = 0.4 for a nearly correct $V$ |
| `lambda_methods_random_walk.py` | offline λ-return, TD(λ) accumulating and replacing, true online TD(λ), TTD(λ) with $K = 8$, S&B Figures 12.3/12.6/12.8 (§8.3, 9.4, 11, 12.4) | 0.6 s | 12 s | all best at λ = 0.8: replacing 0.245, true online 0.252, TTD 0.253, offline λ-return 0.258, accumulating 0.261; accumulating unusable above α = 0.1 at λ = 0.99 |
| `equivalence_checks.py` | machine-precision checks of the chapter's identities: $n$-step and λ-return errors as sums of TD errors, offline forward/backward equivalence, true online TD(λ) = online λ-return (tabular and linear), true online TD(1) = MC, off-policy trace recursion (§2.1, 8.2, 10, 12, 14.1) | 0.2 s | 2 s | all exact identities hold to < 4e-15; online TD(λ) differs from the online λ-return by 10.5 α² |
| `trace_visualization.py` | accumulating, replacing and dutch traces over one 30-step episode (§9.2) | 0.2 s | 1 s | max trace 3.43 / 1.00 / 2.52 (bounds 10 / 1 / 3.57) |
| `sarsa_lambda_gridworld.py` | first-episode credit assignment (S&B Figure 7.4 style); SARSA, 4- and 16-step SARSA, SARSA(λ) accumulating/replacing, true online SARSA(λ), Watkins's Q(λ) on a 10×7 sparse-reward gridworld (§1.1, 13.3) | 0.4 s | 31 s | first episode: 1 / 7 / 73 action values changed (1-step / 10-step / SARSA(λ)); mean steps per episode over 50 episodes: 60.5, 30.8, 28.1, 26.8, 26.1, 25.7, 24.4; one-step SARSA's 30 runs are step-for-step identical for every α ≤ 0.2 (28/30 at α = 0.3) |
| `off_policy_traces.py` | off-policy evaluation of $q_\pi$ on a slippery 15-state chain: naive, IS, $c = 1$ (Q-corrected, no ratios), control variates, Tree Backup, $Q(\sigma)$ ($n$-step); $Q^\pi(\lambda)$, IS(λ), TB(λ), Retrace(λ) (traces) (§7.3, 14.3) | 1.5 s | 70–80 s | naive bias grows with $n$ (final error 0.09 → 0.48) while $c = 1$ stays unbiased (best final 0.08–0.11); IS breaks down at $n \geq 8$ and λ ≥ 0.9; TB and $Q(\sigma)$ stable (best 0.181); Retrace best at λ ≥ 0.8 (0.175 at λ = 1) |
| `vtrace_fixed_point.py` | V-trace converges to $v_{\pi_{\bar\rho}}$ (§14.4) | 0.3 s | 12 s | $\bar\rho = 1$: RMS 0.003 to $v_{\pi_{\bar\rho}}$ vs 0.063 to $v_\pi$; $\bar\rho = 2$: 0.0015 to $v_\pi$ |
| `exercise_solutions.py` | coding exercises 14 (naive vs Watkins's Q(λ)) and 15 (TTD horizon) | 0.9 s | 70 s | naive Q(λ) $= Q^\ast(\lambda)$ reaches \|Q\| = 5.5 at ε = 0.1 and diverges in up to 15/20 runs at ε = 0.3 (λ = 0.9 vs contraction bound 0.35 / 0.12); TTD(λ = 0.9) best at $K = 4$ (0.253) vs true online 0.268 |

Shared modules (no `__main__`): `random_walk19.py` (the 19-state walk, exact values, transition matrix), `chain_mdp.py` (slippery chain, policies, exact $q_\pi$ / $v_\pi$), `plot_style.py` (colours and Matplotlib settings).

Runtimes were measured on one CPU core of a shared machine (Python 3.11, NumPy 2.4); they vary by roughly ±10% with load. Scripts that use matrix products set `OMP_NUM_THREADS=1`. `return_bias_variance.py` stores its 4,000 sampled walks per state as int8/int16 arrays (drawn in blocks that reproduce exactly the same random numbers), so it peaks at about 180 MB of memory.

Figures produced in full mode:

- `figures/backup_diagrams.png`
- `figures/n_step_td_random_walk.png`
- `figures/error_reduction.png`, `figures/return_bias_variance.png`
- `figures/lambda_methods_random_walk.png`
- `figures/online_equivalence.png`
- `figures/eligibility_traces.png`
- `figures/gridworld_first_episode.png`, `figures/gridworld_learning.png`
- `figures/off_policy_traces.png`
- `figures/vtrace_fixed_point.png`

Notes

- Prediction episodes do not depend on the value estimates, so the prediction scripts generate each run's episodes once and evaluate all step sizes (and λ values) in parallel as NumPy arrays. The order of updates is exactly that of the online algorithms. `n_step_td_random_walk.py`, `lambda_methods_random_walk.py` and `exercise_solutions.py` share the same episodes (seed 0), which is why TTD(λ = 1, $K = 8$) reproduces the 8-step TD result (0.2745) exactly.
- The offline λ-return algorithm in `lambda_methods_random_walk.py` follows S&B (2018) Eq. 12.4: targets from the values frozen during the episode, applied as a sequence of updates. `equivalence_checks.py` also implements the summed-increment version used in the forward–backward equivalence theorem, and shows the two differ by O(α²).
- `exercise_solutions.py` and the gridworld script import from their sibling modules, so run them from the repository root as shown (Python puts the script's folder on `sys.path`).
