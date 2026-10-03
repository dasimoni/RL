# Chapter 11 code: natural gradients, trust regions, TRPO and PPO

Companion code for [Chapter 11 — Natural Gradients, Trust Regions, TRPO and PPO](../../chapters/11-trust-regions-and-ppo.md).

Run everything from the repository root. Every script prints its seed and settings, accepts `--quick` (a smoke test that writes no figures), and in full mode saves figures to `figures/`. All scripts pin PyTorch and numpy/BLAS to one thread (the thread environment variables are set before numpy is imported). Runtimes are wall-clock times measured on one CPU thread of a shared 4-CPU machine with other jobs running, so expect some variation.

| Script | What it demonstrates | Command | Quick | Full | Headline results (full mode) |
|---|---|---|---|---|---|
| `natural_gradient_geometry.py` | Same logit step, different KL (Bernoulli); Fisher; the worked bandit example (vanilla vs natural gradient); reparameterization invariance of natural gradient ascent; how long vanilla ascent stays on the second-best arm; exact vanilla PG vs NPG on a 6-state chain | `python code/ch11_trust_regions_and_ppo/natural_gradient_geometry.py` | 0.5 s | 15 s | KL of a 0.5 step: 0.0309 at θ=0 vs 0.0019 at θ=4; after 400 steps the natural paths agree to 1e-15 under θ = c⊙φ (both end at π=(0.831, 0.168, 0.000)), vanilla ends at (0.989, 0.009, 0.002) in θ and (0.001, 0.998, 0.001) in φ, and is still at π₂ = 0.99999 after 100,000 steps; chain: NPG reaches 90% of J* in 4–6 iterations, vanilla PG stays at J = 1.000 after 20,000 |
| `surrogate_bound.py` | Exact checks on a random 6-state MDP: performance difference lemma, first-order match of the surrogate, the TRPO bound and its KL/average-TV versions, the surrogate's optimism for large steps (shared features), CPI (with a zoom on the tiny region where its bound is positive), MM vs trust region vs policy iteration | `python code/ch11_trust_regions_and_ppo/surrogate_bound.py` | 1.8 s | 21 s | PDL error 1.2e-14; 0 violations in 3000 pairs; avg-TV penalty 193× the true gap; true gain −0.05 where the surrogate promises +0.54; CPI certifies κ* = 0.016; MM reaches J = 7.39 after 400 iterations vs J* = 7.916 in ~20 with a KL ≤ 0.01 trust region |
| `trpo_cartpole.py` | Fisher-vector products (double backprop) vs an explicit Fisher; conjugate gradient; KL of the TRPO step vs its quadratic model; VPG (3 step sizes) vs NPG vs TRPO (δ = 0.01, 0.1) on CartPole, 3 seeds | `python code/ch11_trust_regions_and_ppo/trpo_cartpole.py` | 11 s | 5.6 min | FVP error 4.6e-15; 4 of 386 Fisher eigenvalues above 1% of the largest; CG error < 1e-3 by iteration 5; NPG/TRPO δ = 0.01: 500/500/500, KL per update ≈ 0.008; VPG: KL per update from 5.5e-5 to 3.2 |
| `ppo.py` | Single-file PPO (CleanRL style) for discrete (CartPole-v1) and continuous (Pendulum-v1, Gaussian with state-independent log-std) actions, with all implementation details switchable and the diagnostics of Section 8.5 | `python code/ch11_trust_regions_and_ppo/ppo.py` | 7 s | 5.5 min | CartPole (150k steps): 500 on 3/3 seeds, training and deterministic evaluation. Pendulum (200k steps): training return −168 / −171 / −145, deterministic evaluation −122 / −132 / −110. Prints the realized KL per update, the k1/k3 estimates, clip fraction, entropy, explained variance and value-target spread |
| `ppo_clip_ablation.py` | Clip vs no clip vs no clip + KL early stopping in an aggressive regime (K = 10, lr 1e-3), CartPole, 5 seeds; value clipping on in all three | `python code/ch11_trust_regions_and_ppo/ppo_clip_ablation.py` | 10 s | 6.2 min | Mean return over training: clip 426 ± 10 (no collapse; median realized KL 0.0035 per update); no clip 348 ± 60 (2/5 collapses; median KL 0.10, max 3.5); KL early stop 354 ± 48 (2/5 collapses; median KL 0.030, max 0.21) |
| `ppo_details_ablation.py` | One implementation detail switched off at a time (advantage norm, reward scaling, gradient clipping, both, orthogonal init), CartPole, 5 seeds × 70k steps; `--extra` runs the preset and "no reward scaling, no value clipping" | `python code/ch11_trust_regions_and_ppo/ppo_details_ablation.py [--extra]` | 5 s (`--extra`: 6 s) | 6.1 min (`--extra`: 2.1 min) | Mean return over training: full 304 ± 23; no reward scaling 195 ± 23 (−7.6 SE of the difference; critic explained variance ≈ 0, realized KL per update ~60× smaller, critic gradient norm 34 vs actor 0.24); no grad clip 257; no orthogonal init 270; no advantage norm 284 (−1.4 SE); `--extra`: no reward scaling, no value clipping 203 ± 28 |
| `plot_style.py` | Shared matplotlib styling (no `__main__`) | — | — | — | — |

`ppo.py` also accepts `--env`, `--seeds`, `--total-steps`, `--no-figures` and `--set key=value ...` to override any field of `PPOConfig`, for example

```bash
# Exercise 11: PPO with the adaptive KL penalty (Algorithm 11.6) in the aggressive regime (144 s)
python code/ch11_trust_regions_and_ppo/ppo.py --env CartPole-v1 --seeds 5 --no-figures \
    --set update_epochs=10 lr=1e-3 total_steps=80000 clip_coef=inf kl_penalty_target=0.01
# Exercise 12: the truncation-as-termination bug on Pendulum (228 s)
python code/ch11_trust_regions_and_ppo/ppo.py --env Pendulum-v1 --seeds 3 --no-figures --set bootstrap_truncation=False
```

Exercise 11 result: all 5 seeds reach 500 (after 25,200–35,200 steps), no collapse, median realized KL 0.0087 → 0.0027 (max 0.26). Exercise 12 result: training returns −1153 / −982 / −1147, critic explained variance 0.47 → 0.33 (median, first → last quarter).

## Figures (written in full mode)

| File | Script | Chapter section |
|---|---|---|
| `natural_gradient_geometry.png` | `natural_gradient_geometry.py` | 5.5 |
| `surrogate_bound.png` | `surrogate_bound.py` | 3.6 |
| `trpo_fisher_cg.png`, `trpo_vs_vpg.png` | `trpo_cartpole.py` | 6.7 |
| `ppo_cartpole.png`, `ppo_pendulum.png` | `ppo.py` | 8.5, 8.6 |
| `ppo_clip_ablation.png` | `ppo_clip_ablation.py` | 8.4 |
| `ppo_details_ablation.png` | `ppo_details_ablation.py` | 8.3 |

## What the PPO diagnostics measure

* **Realized KL per update** (`kl_post`, the "KL per update" of Sections 7.4 and 8.3–8.5): the exact D_KL(π_old ‖ π_new), averaged over the whole batch, of the policy the update produced (computed after the K epochs). This is the same quantity TRPO's line search controls (Section 6.7), so the two are comparable.
* **k1 and k3 estimates** (`approx_kl_k1`, `approx_kl_k3`): Schulman's per-sample estimators, averaged over the minibatches of the last epoch, each computed *before* that minibatch's gradient step. This is what CleanRL logs and what the early-stopping option (`target_kl`) uses. They lag the final policy and therefore underestimate the realized KL slightly.
* **Clip fraction**: fraction of samples with |ρ − 1| > ε over all minibatches of the update (pre-step). The clip ablation also reports the fraction and max_t |ρ_t − 1| *after* the update (`clipfrac_post`, `max_ratio_dev_post`).
* **Entropy**: batch-mean entropy of the policy after the update.
* **Gradient norms**: pre-clipping norms of the actor's and the critic's gradients, averaged over each update's minibatches; the ablation reports the median over updates.

## Why the critic cannot fit raw CartPole returns in 70,000 steps (Section 8.3, mechanism 1)

Without reward scaling the value targets approach Σ_t 0.99^t ≈ 100. The critic's output is w·h + b with h ∈ [−1, 1]^64 (tanh features) and an orthogonally initialized output row w of Euclidean norm 1, so initially |w·h| ≤ ‖w‖₁ ≤ √64 · ‖w‖₂ = 8. A run of 70,000 steps has 136 updates × 16 minibatch steps ≈ 2,200 Adam steps; Adam moves each weight by at most about the learning rate per step, and the learning rate is annealed from 2.5e-4 to 0 (mean 1.25e-4), so each output weight and the bias move by at most about 0.3 in total. The output therefore stays below roughly 8 + 65 × 0.3 ≈ 27, far from 100: no amount of fitting within this budget can make the critic accurate, and the explained variance stays near 0 (0.05 in the ablation).

## Notes

* Environments are stepped in a plain Python list and reset by hand, so the handling of `terminated` vs `truncated` is explicit: on truncation the reward of the last step is augmented with γ·V(s_final), and the GAE recursion is cut in both cases.
* Value clipping has its own range (`vclip_coef`, default 0.2), separate from the policy clip `clip_coef`. Setting `clip_coef=inf` removes only the policy clip.
* `ppo.py` turns off `torch.distributions` argument validation, which only checks inputs and never changes a result; it saves about 15% of the run time.
* `trpo_cartpole.py` Part A uses float64 and a small 16-16 policy so that the 386 × 386 Fisher matrix can be built from per-sample scores and compared with the double-backprop Fisher-vector products. Part B uses 64-64 networks in float32. Eigenvalues ≤ 1e-16 (zero up to round-off) are clamped to 1e-16 in the spectrum plot.
* `ppo_clip_ablation.py` and `ppo_details_ablation.py` import `train` from `ppo.py`, so every variant shares exactly the same code apart from the switched option. Standard deviations over seeds use n − 1 in the denominator.
* All results are deterministic given the seeds on this machine (PyTorch CPU, one thread); a different PyTorch build or BLAS may change the low-order digits and therefore individual runs.
