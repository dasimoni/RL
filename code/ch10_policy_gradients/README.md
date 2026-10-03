# Chapter 10 code — Policy Gradient Methods: REINFORCE to Actor-Critic

Companion code for [Chapter 10: Policy Gradient Methods](../../chapters/10-policy-gradients.md).

Every script runs from the repository root, fixes its seeds, prints its settings and a results summary, and accepts `--quick` (a smoke test that writes **no** figures). Full mode writes figures to `figures/`. The PyTorch scripts call `torch.set_num_threads(1)`.

```bash
python code/ch10_policy_gradients/pg_theorem_check.py       # add --quick for a smoke test
python code/ch10_policy_gradients/short_corridor.py
python code/ch10_policy_gradients/reinforce_cartpole.py      # --mean-loss: the per-episode-mean bug (Pitfall 7)
python code/ch10_policy_gradients/actor_critic_traces.py
python code/ch10_policy_gradients/a2c_gae_cartpole.py
python code/ch10_policy_gradients/a2c_parallel_actors.py
python code/ch10_policy_gradients/gaussian_policy.py
python code/ch10_policy_gradients/discount_bias.py
python code/ch10_policy_gradients/vtrace_tabular.py
python code/ch10_policy_gradients/exercise_solutions.py
```

| Script | Demonstrates (chapter section) | Quick | Full | Headline results (full mode, seed 0) |
|---|---|---|---|---|
| `pg_theorem_check.py` | policy gradient theorem, estimator bias/variance, compatible FA (4, 6, 12, 13) | 2 s | 11 s | theorem = finite differences to $4\times10^{-10}$; one-episode variance 71.5 (total return) → 24.0 (reward-to-go) → 12.6 (+ baseline) → 2.3 (true advantage); no-$\gamma^t$ bias 0.465; compatible critic exact to $10^{-15}$ |
| `short_corridor.py` | stochastic optimal policy; REINFORCE with/without baseline; per-step vs summed updates (1, 5, 6, pitfall 7) | 1 s | 29 s | REINFORCE ($2^{-12}$) last-100 return $-12.5$; with baseline $-11.8$ and $p=0.578\pm0.059$ ($p^\ast=0.586$); no baseline at $2^{-9}$: 51% of runs end near-deterministic; baseline with per-episode summed updates: 9/100 runs diverge |
| `reinforce_cartpole.py` | REINFORCE variants on CartPole, measured gradient variance (5, 6) | 3 s | 114 s (91 s with `--mean-loss`) | last-100 return: total return 77, reward-to-go 254, learned baseline 421 (20-episode average reaches 475 on 5/5 seeds by episode 253; one seed collapses late); baseline cuts gradient variance 3.6–11× vs reward-to-go. With `--mean-loss` (per-episode `.mean()`): total return collapses to 11 (fails faster than random) |
| `actor_critic_traces.py` | one-step AC vs AC(lambda), parameter study on FrozenLake (7) | 0.4 s | 111 s | success rate over the first 200 episodes at the best $\alpha$: $\lambda=0$ 0.22, $\lambda=0.5$ 0.36, $\lambda=0.95$ 0.12 |
| `a2c_gae_cartpole.py` | A2C + GAE: lambda, entropy and clipping ablations (8, 9, 10) | 4 s | 285 s | final return $\lambda=0$: 169, $\lambda=0.95$: 450, $\lambda=1$: 405; $c_{\mathcal H}=0.05$: 253; joint clipping: 386 (critic gradient 34× the actor's) |
| `a2c_parallel_actors.py` | how a batch is collected: parallel vs consecutive segments (9.5) | 5 s | 214 s | final return 8 envs × 16: 450; 1 env, 8 consecutive segments: 477; 1 env × 128: 496; 32 envs × 4: 338 |
| `gaussian_policy.py` | Gaussian/tanh scores, continuous bandit, SF vs deterministic variance (2, 11) | 4 s | 31 s | $\sigma\to0.023$ and gradient norm 0.11→0.37 without entropy bonus; noiseless-reward control: $J=-0.0008$, gradient norm 0.009; at $\sigma=0.01$ score-function variance 4061 (no baseline), 3.9 (exact baseline, noise-free $q$), 52 (exact baseline, noisy reward) vs 1.08 deterministic |
| `discount_bias.py` | dropping gamma^t drives REINFORCE to the worst policy (12) | 0.4 s | 3 s | with $\gamma^t$: all 100 runs reach $p\ge0.989$ (optimal); without: 94/100 runs end at $p<0.5$ (pessimal side) |
| `vtrace_tabular.py` | V-trace fixed point v_{pi_rho_bar} (14) | 1 s | 24 s | V-trace matches $v_{\pi_{\bar\rho}}$ to 0.006–0.025 in all 5 settings; at $\bar\rho=1$ it is 0.43 away from $v_\pi$ |
| `exercise_solutions.py` | coding exercises 10.3, 10.11, 10.12 | 3 s | 42 s | zero-variance optimal baseline; keeping $\gamma^t$ slows CartPole learning in episodes 201–300 (3/3 seeds); aliased one-step actor-critic drifts to $p=0.986$ |

Runtimes were measured on one core of a shared 4-core machine (Python 3.11, numpy 2.4, torch 2.14 CPU, gymnasium 1.3). Expect some variation.

Figures produced in full mode:

- `figures/pg_estimator_variance.png`, `figures/pg_mc_convergence.png` (pg_theorem_check.py)
- `figures/short_corridor.png`
- `figures/reinforce_cartpole.png`
- `figures/actor_critic_traces.png`
- `figures/a2c_gae_lambda.png`, `figures/a2c_ablations.png` (a2c_gae_cartpole.py)
- `figures/a2c_parallel_actors.png`
- `figures/gaussian_policy.png`
- `figures/discount_bias.png`
- `figures/vtrace_tabular.png`
- `figures/exercise_corridor_ac.png` (exercise_solutions.py)

Helper modules (no `__main__`, not run by the smoke tests):

- `tabular_pg.py`: exact machinery for small episodic MDPs. It provides $v_\pi$, $q_\pi$, discounted visit counts $\eta_\gamma$, the policy-gradient theorem's gradient, finite-difference gradients, vectorized episode sampling, the linear/tabular softmax policy, and the short corridor. It is imported by `pg_theorem_check.py`, `discount_bias.py` and `vtrace_tabular.py`.
- `plot_style.py`: figure style (copied from Chapter 00).

Notes

- `exercise_solutions.py` imports `reinforce_cartpole.py` and `short_corridor.py`, and `a2c_parallel_actors.py` imports `a2c_gae_cartpole.py`. Run it from the repository root as shown, so that Python puts the script's folder on `sys.path`.
- The REINFORCE and A2C scripts use a frozen NumPy copy of the policy network to choose actions during rollouts, which is much faster than a per-step PyTorch call. Gradients are always computed by PyTorch on the whole batch, with the same weights.
- `a2c_gae_cartpole.py` uses Gymnasium's `SAME_STEP` autoreset mode and bootstraps from `info["final_obs"]` on truncation (not on termination).
- `reinforce_cartpole.py` sums each episode's per-step policy-gradient terms and divides by the **constant** 500 (the time limit). Dividing by the episode's own length (`.mean()`) is not a rescaling: it reweights episodes by $1/T$ and, for the total-return variant, turns the update into the gradient of $\mathbb E[G_0/T]$, which rewards short episodes (chapter Section 5.4 and Pitfall 7). `--mean-loss` reproduces that bug for comparison and writes no figure.
- As in nearly all implementations, `reinforce_cartpole.py` and `a2c_gae_cartpole.py` drop the $\gamma^t$ factor of the exact discounted policy gradient (chapter Section 12). `discount_bias.py` and Exercise 10.11 show what that changes.
