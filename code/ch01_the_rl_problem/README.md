# Chapter 01 code: the RL problem and Markov decision processes

Companion code for [Chapter 01](../../chapters/01-the-rl-problem.md). Everything is NumPy (plus SciPy's LP solver and Gymnasium's FrozenLake) and written from scratch. Run every command from the repository root. `--quick` runs a smoke test that writes no figures. Full mode saves figures to `figures/`.

| Script | What it demonstrates | Command | Runtime quick / full | Headline result (full mode, seed 0) |
|---|---|---|---|---|
| `mdp.py` | Library: `FiniteMDP` built on the four-argument dynamics p(s', r \| s, a), derived quantities (1.7)–(1.9), induced chain (P_pi, r_pi), exact policy evaluation, v↔q, value iteration, greedy policies. Self-test on the maintenance MDP. | `python code/ch01_the_rl_problem/mdp.py` | 0.2 s / 0.2 s | v of "run when Good, fix when Worn" = (7.5, 5) at gamma = 0.8 |
| `gridworld.py` | Sutton & Barto Example 3.5 as a model (`to_mdp()`) and as a Gymnasium-style environment (`reset`/`step`); a few steps of the agent–environment loop; layout figure. | `python code/ch01_the_rl_problem/gridworld.py` | 0.2 s / 0.7 s | p tensor of shape (25, 4, 25, 4) over R = {-1, 0, 5, 10}; every p(.,.\|s,a) sums to 1 |
| `returns_and_discounting.py` | Backward computation of returns, absorbing-state padding, constant rewards, effective horizon, gamma vs interest rate. | `python code/ch01_the_rl_problem/returns_and_discounting.py` | 0.2 s / 1.0 s | G_0..G_4 = 6.67, 6.3, 7, 10, 0; at least 1 - 1/e (about 63%) of the discount weight lies in the first 1/(1-gamma) steps (exactly 1 - gamma^(1/(1-gamma))) |
| `policy_evaluation_exact.py` | v_pi of the equiprobable random policy from (I - gamma P_pi) v = r_pi; Bellman checks by hand; q_pi and advantages; invertibility (spectrum, Neumann series, discounted occupancy, infinity-norm and 2-norm condition numbers); gamma sweep (number of positive values, (1-gamma) v -> average reward). | `python code/ch01_the_rl_problem/policy_evaluation_exact.py` | 0.1 s / 1.1 s | v(A) = 8.789, v(B) = 5.322; all 25 values equal S&B Fig. 3.2 after rounding (max unrounded difference 0.050); kappa_inf = (1+gamma)/(1-gamma) exactly (19 at gamma = 0.9); average reward -0.0112 per step, all 25 values negative at gamma = 0.999 |
| `monte_carlo_returns.py` | Monte Carlo averages of sampled returns converge to the exact v_pi; distribution of returns; discounting as a survival probability. | `python code/ch01_the_rl_problem/monte_carlo_returns.py` | 0.3 s / 3.0 s | MC v(A) = 8.785 +- 0.008 (exact 8.789); exact value inside the 95% CI for 25/25 states; survival estimator 8.7865 +- 0.0135 |
| `optimal_values.py` | `v_*` by value iteration, by linear programming (HiGHS) and in closed form; optimal (tied) actions; greedy improvement of the random policy; cost of brute force; how gamma changes the optimal policy, and which states change behaviour at the switch. | `python code/ch01_the_rl_problem/optimal_values.py` | 0.5 s / 1.9 s | `v_*(A)` = 24.419 (= 10/(1-0.9^5)); VI and LP agree to 2e-10; 262,144 optimal deterministic policies; at gamma = 0.8484 eight right-hand states (B' among them) switch from B to A, five of them changing their optimal-action sets |
| `policy_ordering.py` | Maintenance MDP by the numbers (four policies, `q_*`, the gamma = 4/7 switch); the value polytope; brute-force enumeration on 200 random MDPs (5 states, 3 actions) against random stochastic and history-dependent policies. | `python code/ch01_the_rl_problem/policy_ordering.py` | 0.6 s / 4.5 s | A deterministic policy dominating all 243 others in every state exists in 200/200 MDPs; 17.6% of policy pairs incomparable on average; best previous-state-memory policy equals `v_*` to 9e-12 |
| `frozenlake_exact.py` | FrozenLake 4x4 (slippery) read from `env.unwrapped.P`; exact evaluation with gamma = 1 on non-terminal states; simulation through the Gymnasium API; greedy-w.r.t.-`v_*` pitfall at gamma = 1; effect of the 100-step TimeLimit, and the best time-aware policy (backward induction over steps left). | `python code/ch01_the_rl_problem/frozenlake_exact.py` | 0.7 s / 5.1 s | Random policy: success 0.0139 exact vs 0.0151 +- 0.0017 simulated. Policy optimal without a limit: 0.8235 without a limit, 0.7402 within 100 steps (exact), 0.7398 +- 0.0086 simulated; a policy that sees the steps left: 0.7442 within 100 steps |
| `exercise_slippery_gridworld.py` | Solution to Exercise 13: the gridworld with slippery moves (0.8 intended, 0.1 each perpendicular), written as four-argument dynamics; random-policy values, `v_*`, the B' switch point found by bisection, reward rates of the two bonus cycles. | `python code/ch01_the_rl_problem/exercise_slippery_gridworld.py` | 1.5 s / 1.9 s | Random-policy values unchanged (v(A) = 8.7893); `v_*(A)` drops from 24.42 to 19.84; the B' switch moves from gamma = 0.8484 to 0.9820 |

Runtimes were measured on one core of a shared 4-CPU machine and vary somewhat between runs. In particular, the brute-force timing printed by `optimal_values.py` is of the order of 10–60 µs per policy evaluation depending on machine load (several hundred to a few thousand years for all 4^25 policies). The script limits BLAS to one thread before importing NumPy, so the figure is a single-core one.

## Figures (full mode)

| File | Produced by | Shows |
|---|---|---|
| `figures/gridworld_layout.png` | `gridworld.py` | The 5x5 gridworld with the A→A' (+10) and B→B' (+5) teleports |
| `figures/discount_weights.png` | `returns_and_discounting.py` | gamma^k weights and the share of weight in the first m rewards |
| `figures/random_policy_values.png` | `policy_evaluation_exact.py` | v_pi of the random policy; discounted occupancy from A |
| `figures/neumann_series_convergence.png` | `policy_evaluation_exact.py` | Geometric convergence of the Neumann series |
| `figures/monte_carlo_returns.png` | `monte_carlo_returns.py` | Running MC estimates, return distribution, survival vs discounting |
| `figures/optimal_values_policy.png` | `optimal_values.py` | `v_*` and all optimal actions at gamma = 0.9 |
| `figures/optimal_policy_vs_gamma.png` | `optimal_values.py` | Optimal policies at gamma = 0.8 and 0.9; the switch at gamma = 0.8484 |
| `figures/value_polytope.png` | `policy_ordering.py` | Value functions of all stochastic policies of the maintenance MDP |
| `figures/frozenlake_values.png` | `frozenlake_exact.py` | `v_*`, greedy actions and success probability vs step limit (stationary optimal policy and the best time-aware policy) |

## Notes

* `mdp.py`, `gridworld.py` and `plotting.py` are imported by the other scripts. Python finds them because each script's own folder is on `sys.path` when it is run as `python code/ch01_the_rl_problem/<script>.py`.
* Terminal states follow the absorbing-state convention (they loop to themselves with reward 0). With gamma = 1, `evaluate_policy_exact` solves only on non-terminal states, which requires a policy that terminates with probability 1. The chapter writes that restricted matrix as P~_pi (P_pi restricted to non-terminal states); the scripts print it under the same name.
* `+-` in the script output and in the headline results above is a 95% confidence half-width (1.96 standard errors), as in the chapter. `frozenlake_exact.py` also prints one standard error (0.0009 and 0.0044 for its two simulations), which is the convention Chapter 03 uses.
* The Gymnasium loop in `frozenlake_exact.py` seeds only the first `reset` and lets the environment's RNG stream continue, and it distinguishes `terminated` from `truncated`.
