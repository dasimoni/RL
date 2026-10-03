# Chapter 16 code: Imitation Learning, Inverse RL and Offline RL

Companion code for [Chapter 16](../../chapters/16-offline-rl-and-imitation.md). Every script runs from the repository root, fixes its seeds, prints its settings and a results summary, and accepts `--quick` (a smoke test that writes no figures). Full runs write figures to `figures/`.

The tabular demos use NumPy and compute expected costs and values exactly, by propagating distributions or solving linear systems. The CartPole experiments use small PyTorch MLPs on one CPU thread (`torch.set_num_threads(1)`).

| Script | What it demonstrates | Command | Quick | Full |
|---|---|---|---|---|
| `dagger_vs_bc.py` | Compounding errors on an unstable "tightrope" (Section 2.6): BC vs DAgger with the same 1,000 expert labels, excess cost vs horizon, and label efficiency | `python code/ch16_offline_rl_and_imitation/dagger_vs_bc.py` | 2–3 s | 112–141 s |
| `irl_gridworld.py` | BC, apprenticeship learning (projection method) and MaxEnt IRL (Sections 3–4) on a terrain gridworld; transfer to a changed world | `python code/ch16_offline_rl_and_imitation/irl_gridworld.py` | 1 s | 17–23 s |
| `offline_cartpole.py` | Offline RL on CartPole-v1 (Sections 6–9): expert / medium / random datasets of 20k transitions; BC, naive offline DQN, discrete BCQ (with a diagnostic of its action filter), CQL(H), IQL; 5 seeds | `python code/ch16_offline_rl_and_imitation/offline_cartpole.py` | 12–15 s | 300–372 s |
| `offline_sensitivity.py` | CQL's alpha and IQL's (tau, beta) swept on the medium and random datasets, with diagnostics of IQL's failure, plus an expectile probe on the expert data (Sections 8.6, 9.4) | `python code/ch16_offline_rl_and_imitation/offline_sensitivity.py` | 5–7 s | 415 s |
| `cql_tabular.py` | The CQL lower-bound theorems checked in a table, exactly and with finite data (Section 8.5) | `python code/ch16_offline_rl_and_imitation/cql_tabular.py` | <1 s | 3–4 s |
| `sequence_vs_dp.py` | Return-conditioned supervised learning (Decision-Transformer style) vs in-sample DP: stitching and luck (Section 11) | `python code/ch16_offline_rl_and_imitation/sequence_vs_dp.py` | <1 s | <1 s |
| `ope_tabular.py` | Off-policy evaluation (Section 12): IS, WIS, PDIS, WPDIS, FQE, DR, WDR, plus FQE and DR with a misspecified model, vs data size, horizon and policy mismatch | `python code/ch16_offline_rl_and_imitation/ope_tabular.py` | <1 s | 58–61 s |
| `exercise_solutions.py` | Numerical checks for Exercises 16.2, 16.9, 16.10, 16.13 (tabular GAIL, with a stable and a too-large discriminator step) and 16.14 (offline policy selection) | `python code/ch16_offline_rl_and_imitation/exercise_solutions.py` | 2 s | 15–16 s |
| `offline_lib.py` | Shared CartPole pieces: scripted expert (LQR) and medium policies, dataset collection, `EnsembleMLP` (one ensemble member per seed), lockstep evaluation | (imported) | | |
| `plot_style.py` | Shared matplotlib style | (imported) | | |

Times are wall-clock seconds on a shared 4-CPU machine with one thread per script; ranges span runs at different machine loads (load average 3–5). Add `--quick` to any command for the smoke test.

## Headline results (full runs)

**`dagger_vs_bc.py`** (300 draws of the training data, exact expected costs). With 1,000 expert labels, BC's excess cost grows with fitted log–log slope 1.44 (10 ≤ T ≤ 100) and 1.54 (100 ≤ T ≤ 1000); DAgger's with 1.17 and 1.32. Excess cost per step from T = 10 to T = 3000: BC 0.024 → 0.274, DAgger 0.022 → 0.081. At T = 1000, DAgger reaches an excess cost of 63 with 1,000 labels; BC needs about 10,000 labels to get to 74. BC's error on the expert's own distribution, averaged over the first T steps (ε_T of Eq. 16.3), is 0.097 at every T from 10 to 3000, so the bound ε_T T² is very loose here (9.7 vs an actual 0.24 at T = 10). Both learners are stationary policies, so their per-step excess saturates (BC ≈ 0.27, DAgger ≈ 0.08 at T = 3000).

**`irl_gridworld.py`** (true returns over H = 30 steps):

| world | optimal | soft expert | BC | projection | MaxEnt IRL | random |
|---|---|---|---|---|---|---|
| training | 50.00 | 49.38 | 48.01 | 49.44 | 48.99 | −59.62 |
| transfer | 56.67 | 55.85 | −42.16 | 43.33 | 55.93 | −52.69 |

MAP MaxEnt (ℓ2 = 0.03) recovers reward differences to road of (−1.33, −3.69, +5.15) for grass, mud and goal (true: −2, −6, +3). Plain maximum likelihood does not converge, because the empirical μ_E lies outside the set of achievable feature expectations: the 30 demonstrations spent 16.80 steps at the goal, but no policy can average more than 16.667 (the script prints the achievable range of every feature; grass and mud are inside theirs). Stopped at 2,000 steps it still transfers as well (56.34). The projection method's gap ‖μ_E − μ̄‖ stalls at 0.154 from iteration 5 on, the same floor as unregularised MaxEnt. Both end at μ = (13.211, 0.044, 0.078, 16.667); the residual (−0.044, −0.044, −0.045, +0.133) is the unreachable goal excess.

**`offline_cartpole.py`** (final greedy return, mean (sd) over 5 seeds):

| dataset | behaviour | BC | naive DQN | BCQ | CQL | IQL |
|---|---|---|---|---|---|---|
| expert | 500.0 | 500.0 (0) | 9.3 (0) | 500.0 (0) | 500.0 (0) | 500.0 (0) |
| medium | 176.7 | 171.4 (0) | 9.9 (0) | 180.0 (1) | 166.9 (40) | 10.2 (1) |
| random | 22.5 | 25.6 (10) | 500.0 (0) | 441.1 (109) | 497.1 (6) | 422.4 (56) |

Naive DQN's mean max-Q on dataset states reaches 1.5 × 10⁵ (expert) and 2.1 × 10⁵ (medium); no true value exceeds 100. The sd is over training seeds on one dataset per behaviour. BCQ's filter (threshold 0.3) lets both actions through in 2.9% (expert), 2.4% (medium) and 100% (random) of dataset states, so BCQ is near-BC on narrow data and near-double-DQN on random data.

**`offline_sensitivity.py`** (3 seeds per setting): CQL with α ∈ {0, 0.1} diverges on medium data (return 9.9), α = 1 gives 183.8 and α = 10 gives 172.2; on random data α ≤ 1 gives 500 but α = 10 only 199.8. So the safe range shifts with the data (medium needs α ≥ 1, random α ≤ 1); α = 1 happens to lie in both. IQL on medium data: (τ, β) = (0.5, 1) gives 174.9, (0.5, 10) 9.9, (0.7, 1) 141.7, (0.7, 3) 9.9, (0.9, 1) 9.6 with values of 501. The critic's per-state preference Q(s, other) − Q(s, controller) is −0.03 ± 0.77 at τ = 0.5 (positive in 50% of states) and +0.39 ± 1.23 at τ = 0.7. On random data every IQL setting scores 409–491, and τ = 0.9 is best (491) although its values reach 162 (> 100). Expectile probe on the expert data (one action per state): mean max-Q after 8,000 steps 79.8 at τ = 0.5 and 99.5 (still rising) at τ = 0.9.

**`cql_tabular.py`**: the closed form matches the theory to ~1e-14. With finite data, P(V̂ ≤ V^π at all states) is 0.41 at α = 0 and rises to 0.91 (20 transitions/state) or 1.00 (200/state) at α = 0.03. Datasets that miss some (s, a) are re-drawn (the tabular penalty would be infinite): 83% of draws at 20/state (4,881 re-draws for 1,000 datasets), 2 at 200/state, so the 20/state results are conditional on full coverage.

**`sequence_vs_dp.py`**: stitching problem, return from A: offline DP 10, deterministic BC 0 (ties broken by first occurrence), BC sampling the empirical action frequencies 4 in expectation, RCSL at most 3 for any conditioning value. Luck problem: RCSL conditioned on 10 gambles (expected return 3); DP plays safe (5).

**`ope_tabular.py`** (RMSE, true value 5.10, H = 10):

| n | IS | WIS | PDIS | WPDIS | FQE | FQE-agg | DR | DR-agg | WDR |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 5.52 | 0.66 | 2.19 | 0.39 | 0.19 | 0.44 | 0.63 | 0.73 | 0.39 |
| 5000 | 1.18 | 0.24 | 0.31 | 0.11 | 0.02 | 0.41 | 0.12 | 0.12 | 0.11 |

**`exercise_solutions.py`**: tabular GAIL with a stable discriminator step (lr_d = 0.2) reaches JS divergence 0.0007 to the empirical expert occupancy (return 48.7 vs expert 49.4), with D at the goal 0.50–0.51 and 0.41–0.62 elsewhere on the expert's support; in the transfer world the replayed policy scores 17.43 and the re-optimised discriminator reward 50.00 (optimum 56.67; it crosses a mud cell no demonstration entered). With lr_d = 1 the discriminator update is unstable at the high-occupancy goal (linearised factor 5.89 > 2): D there flips between ≈0.01 and ≈0.94 from one iteration to the next, and JS stalls near 0.03. Offline selection among 7 candidate policies: FQE picks the best one in 100% of 300 datasets, DR 76%, WDR 68%, WPDIS 1%, WIS 0%.
