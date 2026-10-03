# Chapter 08 code: value function approximation

Companion code for [Chapter 08 — Value Function Approximation: Linear Methods, Features, and the Deadly Triad](../../chapters/08-function-approximation.md).

Run everything from the repository root. Every script prints its seed and settings, accepts `--quick` (a smoke test that writes no figures), and in full mode saves figures to `figures/`. Scripts pin numpy/BLAS (and torch) to one thread. Runtimes were measured on one CPU thread of a shared 4-CPU machine.

| Script | What it demonstrates | Command | Quick | Full | Headline results (full mode) |
|---|---|---|---|---|---|
| `rw1000.py` | Helper module: the 1000-state random walk, exact $v_\pi$, on-policy distribution $\mu$, feature matrices, exact projection and TD fixed point | `python code/ch08_function_approximation/rw1000.py` | 0.7 s | — | expected episode length 83.15; $\mu(500)=0.0137$ vs $\mu(499)=\mu(501)=0.0017$ |
| `random_walk_aggregation.py` | Gradient MC vs semi-gradient TD(0) with state aggregation (S&B Figs 9.1, 9.2); $n$-step sweep; positive definiteness of $A$ | `python code/ch08_function_approximation/random_walk_aggregation.py` | 0.6 s | 12 s | MC $\overline{VE}=0.00300$ (min 0.00296); TD 0.01348 (exact $w_{TD}$: 0.01363); best $n$-step: $n=4$, $\alpha=0.4$, RMS 0.292 |
| `basis_comparison.py` | Polynomial vs Fourier vs tile coding, gradient MC, 30 runs × 5000 episodes (S&B Figs 9.5, 9.10); exact "noise-free" error of the expected update after the same number of steps | `python code/ch08_function_approximation/basis_comparison.py` | 1.3 s | 81–100 s | final $\sqrt{\overline{VE}}$: Fourier 0.057–0.059 (noise-free 0.012–0.016), polynomial 0.122–0.132 (noise-free 0.085–0.100), 50 tilings 0.068 (0.056), 1 tiling 0.112 |
| `feature_gallery.py` | Basis functions, 2-D Fourier features, uniform vs asymmetric tile offsets, coarse-coding width (S&B Fig 9.8), step-size rule | `python code/ch08_function_approximation/feature_gallery.py` | 0.1 s | 2.6 s | coarse coding RMS after 10,240 samples: 0.078 / 0.078 / 0.082 (narrow / medium / broad) |
| `tiles.py` | Tile coder from scratch (asymmetric offsets, optional hashing: stateless mix hash or index hash table); self-test | `python code/ch08_function_approximation/tiles.py` | 0.2 s | — | self-test passes |
| `mountain_car_sarsa.py` | Semi-gradient SARSA + tile coding on Gymnasium `MountainCar-v0`; learning curves, cost-to-go surfaces, greedy policy, truncation ablation (with truncation statistics); optional hashing demo (mix hash and index hash table) | `python code/ch08_function_approximation/mountain_car_sarsa.py [--hashing-demo]` | 4.0 s (2.0 s with `--hashing-demo`) | 120 s (`--hashing-demo`: 98 s) | steps/episode (last 50 of 300): 147 / 137 / 117 for $\alpha$ = 0.1/8, 0.2/8, 0.5/8; truncation bug had no measurable effect here (44–52 of 300 episodes truncated per run); IHT 1024 and mix-hash 4096 match no hashing (140 / 143 vs 140 steps), 128 or 32 slots break learning |
| `nstep_sarsa_mountain_car.py` | Exercise 12: $n$-step semi-gradient SARSA on `MountainCar-v0` (default 200-step limit), $n \in \lbrace 1,4,8\rbrace$ × 5 step sizes, correct truncation handling | `python code/ch08_function_approximation/nstep_sarsa_mountain_car.py` | 0.7 s | 130 s | mean steps/episode over the first 100 episodes at each $n$'s best $\alpha$: $n=4$ 156.7 ± 1.3 ($\alpha=0.7/8$), $n=8$ 161.9 ± 3.1 (0.4/8), $n=1$ 166.4 ± 0.8 (1.0/8) |
| `fqi_lspi_mountaincar.py` | Batch control on `MountainCar-v0` dynamics (Section 11.4): fitted Q-iteration with a $K$-NN averager ($K=10$) and with ridge least squares on tile coding (8 tilings, 648 features/action), and LSPI (LSTDQ + greedy improvement) on the same features; batches of $N \in \lbrace 5000, 20000, 50000\rbrace$ uniformly random states and actions, 5 batches each, $\gamma = 0.99$; greedy policies evaluated from 100 held-out starts | `python code/ch08_function_approximation/fqi_lspi_mountaincar.py` | 1.8 s | 212–217 s | $K$-NN averager FQI converged on 15/15 batches (259–381 iterations; step ratio never above $\gamma$), 121.8 ± 4.4 greedy steps at $N=50000$; least-squares FQI (ridge $\lambda=10^{-3}$) diverged on 5/5 batches at $N=5000$, converged on 10/10 larger ones (161.8 ± 31.8 steps at $N=50000$); LSPI converged in 7–9 iterations to the least-squares FQI solution (max gap $2.9\times10^{-5}$) and never settled at $N=5000$; reference policy 118.8 steps |
| `linear_td_theory.py` | Exact checks: TD error bound, positive definiteness on/off-policy (split by whether expected TD converges; non-expansion and feature-subspace gain of $P$), asymmetry of $A$, futility of discounting, unlearnability of the Bellman error, VE/BE/PBE geometry figure | `python code/ch08_function_approximation/linear_td_theory.py [--dense]` | 1.4 s | 10 s (`--dense`: 9 s) | 0 bound violations in 1500 on-policy MRPs; off-policy: $A$ not PD in up to 137/300, divergence in up to 32/300, worst ratio among convergent instances $3.1\times10^4$ |
| `lstd_vs_td.py` | Exact limits of VE/TD/BE/TDE objectives; exact TD($\lambda$) fixed points; LSTD vs TD(0) vs TD($\lambda$) learning curves; cost per update vs $d$ | `python code/ch08_function_approximation/lstd_vs_td.py` | 2.8 s | 31 s | polynomial features: LSTD $\sqrt{\overline{VE}}=0.043$ vs TD(0) $\ge 0.175$ after 300 episodes; TD($\lambda$) fixed point (10 groups) 0.117 → 0.054 as $\lambda$: 0 → 1; LSTD roughly 600× slower per step at $d=800$ (590× and 646× in two runs; machine-dependent) |
| `nonlinear_and_kernel.py` | PyTorch MLP trained with gradient MC, semi-gradient TD(0), and the full-gradient TD loss; kernel (Nadaraya–Watson) regression | `python code/ch08_function_approximation/nonlinear_and_kernel.py` | 4.9 s | 165–200 s | final $\sqrt{\overline{VE}}$ per seed: MC 0.039/0.053/0.028, TD 0.016/0.090/0.011, full-gradient 0.246/0.254/0.246; mean over the last 20% of training: 0.042 / 0.043 / 0.248 |
| `access_control_differential_sarsa.py` | Differential semi-gradient SARSA on the access-control queuing task (S&B Example 10.2), checked against relative value iteration | `python code/ch08_function_approximation/access_control_differential_sarsa.py` | 1.5 s | 7 s | learned greedy policy: average reward 2.733 vs optimal 2.748; $\bar R$ (second half) 2.632 vs exact 2.630 |
| `baird_counterexample.py` | Baird's counterexample: semi-gradient TD/DP divergence, TDC, GTD2, emphatic TD, the deadly triad | `python code/ch08_function_approximation/baird_counterexample.py` | 0.2 s | 3.0 s | $\lVert w\rVert$: 10.3 → 298 (TD, 1000 steps), → 437 (DP); TDC $\sqrt{\overline{PBE}}\to 0.0074$ with $\sqrt{\overline{VE}}\approx1.93$; expected ETD $\sqrt{\overline{VE}}\to 0$ |

## Figures (written in full mode)

| File | Script |
|---|---|
| `rw_aggregation_mc_vs_td.png`, `rw_nstep_sweep.png` | `random_walk_aggregation.py` |
| `basis_comparison.png` | `basis_comparison.py` |
| `feature_gallery.png`, `coarse_coding_width.png` | `feature_gallery.py` |
| `mountain_car_learning_curves.png`, `mountain_car_cost_to_go.png`, `mountain_car_policy.png` | `mountain_car_sarsa.py` |
| `nstep_sarsa_mountain_car.png` | `nstep_sarsa_mountain_car.py` |
| `fqi_lspi_mountaincar.png` | `fqi_lspi_mountaincar.py` |
| `td_fixed_point_bound.png`, `bellman_geometry.png` | `linear_td_theory.py` |
| `lstd_vs_td.png` | `lstd_vs_td.py` |
| `nonlinear_and_kernel.png` | `nonlinear_and_kernel.py` |
| `access_control.png` | `access_control_differential_sarsa.py` |
| `baird_divergence.png`, `baird_tdc_etd.png`, `baird_triad.png` | `baird_counterexample.py` |

## Notes

* Prediction experiments on the random walk generate trajectories first and then update the weights. The policy is fixed, so trajectories do not depend on the weights, and `basis_comparison.py` and `lstd_vs_td.py` update all runs in lock-step with numpy.
* `mountain_car_sarsa.py` bootstraps on `truncated` and not on `terminated`. Its experiment (c) deliberately implements the wrong variant for comparison.
* `fqi_lspi_mountaincar.py` simulates `MountainCar-v0` with a vectorised NumPy copy of Gymnasium's dynamics; at start-up it checks the copy against Gymnasium on random transitions (20,000 in full mode) and checks the reference policy's rollouts from the first 10 held-out start states against Gymnasium itself. It uses `scipy.spatial.cKDTree` for the nearest-neighbour averager.
* `nonlinear_and_kernel.py` is the only script using PyTorch (`torch.set_num_threads(1)`).
* Wall-clock timings (runtimes above, and the cost-per-update panel of `lstd_vs_td.png`) depend on machine load; that panel is the only figure that does not regenerate bit-identically.
* In `linear_td_theory.py` the off-policy weighting $d\sim$ Dirichlet(0.1) is an arbitrary stress test: the random MRPs have no actions, so $d$ need not be the state distribution of any behaviour policy.
