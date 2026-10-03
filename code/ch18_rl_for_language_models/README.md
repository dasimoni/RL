# Chapter 18 code: RL for language models

Code for [Chapter 18 — RL for Language Models: RLHF, DPO, GRPO and Verifiable Rewards](../../chapters/18-rl-for-language-models.md).

Everything runs on CPU with one thread (PyTorch, NumPy, SciPy, Matplotlib). Run from the repository root. Every script prints its seed and settings, and `--quick` runs a smoke test that writes no figures. Runtimes were measured on one core of a shared 4-CPU machine.

| Script | What it demonstrates | Command | Quick | Full |
|---|---|---|---|---|
| `kl_bandit.py` | The KL-regularized optimum $\pi^\ast_\beta\propto\pi_{\mathrm{ref}}e^{r/\beta}$ on an 8-armed bandit, checked against SLSQP, exact gradient ascent and sampled REINFORCE-with-RLOO; population DPO recovers $\pi^\ast_\beta$; deterministic preferences make DPO collapse for every $\beta$ while IPO keeps a finite solution; exact best-of-$n$ KL vs $\log n-(n-1)/n$; best-of-$n$ vs the optimal reward–KL frontier | `python code/ch18_rl_for_language_models/kl_bandit.py` | 2 s | 12 s |
| `toy_lm.py` | Library (nothing to run): conditional GRU language model, masked sampling, per-token log-probs, exact per-token KL, scalar-head GRU for reward and value models | — | — | — |
| `rlhf_toy.py` | The full pipeline on a toy LM: SFT on demonstrations → preferences from a hidden true reward → Bradley–Terry reward models (2k and 8k pairs) → PPO with per-token KL, value model and GAE ($\beta$ sweep), RLOO, DPO ($\beta$ sweep), IPO, best-of-$n$; over-optimization curves | `python code/ch18_rl_for_language_models/rlhf_toy.py` | 11 s | 5.1 min |
| `grpo_rlvr.py` | RL with a verifier on running-sum arithmetic with a systematic "carry bug" in the base model: REINFORCE, RLOO, GRPO, Dr. GRPO, DAPO-lite, GRPO with process rewards (3 seeds each); pass@$k$ and majority voting for base vs GRPO | `python code/ch18_rl_for_language_models/grpo_rlvr.py` | 10 s | 5.1 min |
| `exercise_solutions.py` | Numerical checks for Exercises 1–10; adaptive KL control (Ex. 11), SimPO vs DPO (Ex. 12), one round of iterated RLHF (Ex. 13) | `python code/ch18_rl_for_language_models/exercise_solutions.py` | 13 s | 2.2 min |

## Headline results (full runs)

**`kl_bandit.py`** (seed 0)
* Closed form vs solvers, max |difference| in probability: SLSQP $\le 2.8\times10^{-7}$; exact gradient $2.1\times10^{-4}$ ($\beta=0.1$) to $4.5\times10^{-16}$ ($\beta=2$); sampled REINFORCE with a leave-one-out baseline $7.8\times10^{-4}$ to $5.6\times10^{-17}$ (zero-variance gradient at the optimum).
* Population DPO (Bradley–Terry preferences, L-BFGS) matches $\pi^\ast_\beta$ to $\le 2.1\times10^{-8}$.
* Deterministic preferences: DPO goes to the point mass on the best arm (KL 2.483) for $\beta=0.1, 0.5, 2$; IPO converges to $\pi_{\mathrm{ref}}e^{p(y\succ\pi_{\mathrm{ref}})/\tau}/Z$ (error $\le 10^{-9}$) with KL 1.281 / 0.149 / 0.010 for $\tau=0.1/0.5/2$.
* Best-of-$n$ KL equals $\log n-(n-1)/n$ for $10^4$ equiprobable responses, but saturates at 2.483 on the 8-arm bandit. Best-of-$n$ reaches 92–96% of the optimal frontier's reward at the same KL ($n=4$ to 256).

**`rlhf_toy.py`** (seed 0; true reward of $\pi_{\mathrm{ref}}$ = 0.818, best achievable 4.039)

| method | $\beta$ | true reward | KL | repeats |
|---|---|---|---|---|
| PPO (RM-2k) | 0.01 / 0.03 / 0.1 / 0.3 / 1 | 1.830 / 2.508 / **2.963** / 2.375 / 1.578 | 24.9 / 13.4 / 3.96 / 1.46 / 0.27 | 100% / 87% / 3% / 1% / 2% |
| PPO (RM-8k) | 0.01 / 0.1 | 2.020 / **3.482** | 29.9 / 5.07 | 100% / 5% |
| RLOO (RM-2k) | 0.03 / 0.1 | 2.701 / 3.052 | 11.0 / 3.92 | 51% / 2% |
| DPO (2k pairs) | 0.03 / 0.1 / 0.3 / 1 | 2.757 / 2.897 / 2.884 / 2.302 | 5.70 / 5.47 / 4.55 / 2.00 | 12% / 9% / 9% / 7% |
| IPO | $\tau=0.1$ | 2.539 | 2.87 | 11% |
| best-of-512 (RM-2k / RM-8k) | — | 3.101 / 3.488 | $\le 5.24$ | — |

At $\beta=0.01$ the true reward peaks at 3.08 (KL 7.4) and falls to 1.83, while the reward model's score keeps rising (the reward model never saw repetition). DPO's chosen-response log-probability falls by 11.7 nats at $\beta=0.1$ (rejected: 17.0).

**`grpo_rlvr.py`** (3 seeds; base model: sampled accuracy 0.360, greedy 0.218)

| method | accuracy (mean, range) | greedy | KL | responses sampled |
|---|---|---|---|---|
| REINFORCE | 0.631 (0.609–0.643) | 0.743 | 2.77 | 51,200 |
| RLOO | 0.616 (0.578–0.635) | 0.731 | 2.85 | 51,200 |
| GRPO | 0.700 (0.694–0.707) | 0.851 | 1.62 | 51,200 |
| Dr. GRPO | 0.670 (0.663–0.684) | 0.833 | 1.49 | 51,200 |
| DAPO-lite | 0.739 (0.732–0.745) | 0.856 | 1.84 | 102,912 (0.646 at 51,200) |
| GRPO + process reward | 0.773 (0.749–0.788) | 0.905 | 1.49 | 51,200 |

Test-time compute: base pass@64 = 0.998, but base maj@$k$ *falls* from 0.364 ($k=1$) to 0.270 ($k=64$) because the carry bug is systematic; after GRPO, maj@$k$ rises from 0.706 to 0.847.

**`exercise_solutions.py`**: adaptive KL control rescues a $\beta=0.01$ run (true reward 0.81 → 3.36) but reacts too slowly to keep the KL near its target; SimPO ≈ DPO (2.90 vs 3.05 true reward); one round of iterated RLHF raises the true reward at $\beta=0.01$ from 0.63 to 3.11 without eliminating repetition.

## Figures

`figures/kl_bandit.png`, `figures/bon_kl.png`, `figures/rlhf_overoptimization.png`, `figures/rlhf_beta_sweep.png`, `figures/dpo_dynamics.png`, `figures/grpo_rlvr_curves.png`, `figures/grpo_rlvr_tts.png` (all written by full runs only).
