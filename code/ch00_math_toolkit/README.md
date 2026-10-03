# Chapter 00 code: the mathematical and ML toolkit

Companion code for [Chapter 00: The Mathematical and ML Toolkit for RL](../../chapters/00-math-toolkit.md).
Each script only imports the tiny `plot_style.py` helper (`exercise_solutions.py` also reuses functions from
`markov_chain_stationary.py`), uses fixed seeds, prints its settings, and supports `--quick` (a smoke test that
writes no figures). Run everything from the repository root.

| Script | What it demonstrates | Chapter section |
|---|---|---|
| `markov_chain_stationary.py` | Stationary distribution by power iteration vs eigenvector vs linear solve; geometric convergence at rate $\lvert\lambda_2\rvert$; a slow-mixing chain and why a small step is not a small error (Algorithm 1.1's stopping rule); periodicity; the ergodic theorem | 1.6 |
| `estimation_step_sizes.py` | Sample mean vs constant-$\alpha$ exponential recency-weighted averages (bias, variance, MSE against closed forms); tracking a drifting mean; Robbins–Monro step-size schedules | 2.3, 2.4, 3.4 |
| `importance_sampling.py` | Ordinary vs weighted importance sampling; MSE vs sample size; variance blow-up and effective sample size vs horizon | 2.6 |
| `contraction_fixed_point.py` | Banach fixed-point iteration for $T(\mathbf v)=\mathbf r+\gamma\mathbf P\mathbf v$; a-priori and a-posteriori bounds; max-norm vs 2-norm vs the 2-norm weighted by the stationary distribution; cobweb plots; conditioning as $\gamma\to1$ | 4.2–4.4 |
| `kl_divergence.py` | Discrete KL asymmetry; fitting a Gaussian to a bimodal mixture by forward KL (mode covering) and reverse KL (mode seeking, three local optima, cost $\approx-\log w$ of locking onto a mode of weight $w$) | 5.3, 5.4 |
| `gradient_estimators.py` | Score-function vs reparameterization gradients: closed-form variances, dimension scaling, a case where reparameterization loses, softmax policies and baselines, PyTorch versions | 6 |
| `optimization_demo.py` | GD, heavy-ball momentum and Adam on an ill-conditioned quadratic; SGD noise floor vs decaying step size | 3.2–3.5 |
| `pytorch_regression.py` | A minimal PyTorch training loop (SGD, momentum, Adam); autograd checked against the hand-computed derivative of §3.1; live demos of eight common bugs (shape broadcasting, zero_grad, detach, no_grad, dtype, gather, boolean flags in arithmetic, in-place modification) | 7 |
| `gymnasium_basics.py` | Gymnasium spaces, reset/step, terminated vs truncated (random agent on CartPole), why truncation must bootstrap, seeding, wrappers, vector envs and autoreset | 8 |
| `exercise_solutions.py` | Numerical checks quoted in the solutions of Exercises 4, 5(c), 10, 12, 13, 14 and 15 (no figures) | Exercises |
| `plot_style.py` | Shared matplotlib style (colour order and line styles) | |

## Commands and measured runtimes

Measured wall-clock times on the course machine (one CPU thread, including Python start-up):

| Command | `--quick` | full |
|---|---|---|
| `python code/ch00_math_toolkit/markov_chain_stationary.py` | 0.7 s | 3.5 s |
| `python code/ch00_math_toolkit/estimation_step_sizes.py` | 0.6 s | 11.4 s |
| `python code/ch00_math_toolkit/importance_sampling.py` | 0.8 s | 5.0 s |
| `python code/ch00_math_toolkit/contraction_fixed_point.py` | 0.7 s | 1.4 s |
| `python code/ch00_math_toolkit/kl_divergence.py` | 1.6 s | 4.5 s |
| `python code/ch00_math_toolkit/gradient_estimators.py` | 2.9 s | 7.7 s |
| `python code/ch00_math_toolkit/optimization_demo.py` | 0.7 s | 2.4 s |
| `python code/ch00_math_toolkit/pytorch_regression.py` | 4.1 s | 10.4 s |
| `python code/ch00_math_toolkit/gymnasium_basics.py` | 0.8 s | 2.2 s |
| `python code/ch00_math_toolkit/exercise_solutions.py` | 2.9 s | 3.4 s |

Full mode writes the figures to `code/ch00_math_toolkit/figures/`.

## Headline results (full mode, seed 0)

* **Markov chains.** For the 3-state weather chain, power iteration, the eigenvector of $\mathbf P^\top$ and a linear solve all give $(7,6,5)/18$ to within $1.7\times10^{-16}$; the error shrinks by exactly $\lvert\lambda_2\rvert = 0.4$ per iteration. A 20-state "two rooms" chain with $\lvert\lambda_2\rvert=0.98$ needs 684 iterations (vs 16) to reach $L_1$ error $10^{-6}$; stopping when the *step* falls below $10^{-6}$ (Algorithm 1.1) quits at iteration 492 with a true error of $4.8\times10^{-5}$, 48 times the tolerance. Visit frequencies along one trajectory of $10^6$ steps are within $1.3\times10^{-3}$ ($L_1$) of the stationary distribution.
* **Step sizes.** Constant $\alpha=0.1$ plateaus at variance $0.0533$ (theory $\alpha\sigma^2/(2-\alpha)=0.0526$); the sample mean keeps improving as $\sigma^2/n$. On a drifting target, $\alpha=0.01$ tracks with error $0.010$ while the sample mean's error grows to $0.318$. A schedule with $\sum\alpha<\infty$ ($1/(n+1)^2$) stalls at $2.50$ when the target is $5$, exactly as the theory predicts.
* **Importance sampling.** One step, $\pi=(0.9,0.1)$, $b=(0.5,0.5)$: ordinary IS has MSE $0.81/n$; weighted IS is biased at small $n$ (mean $0.51$ at $n=1$ instead of $0.9$) but its MSE at $n=1024$ is $25\times$ lower. With episodes of length $H$, $\mathbb E_b[\rho^2]=1.64^H$: at $H=64$ the median effective sample size is $1.5$ out of $1000$ episodes, and the *measured* error of ordinary IS badly understates the true one (the rare huge weights are never sampled).
* **Contraction.** $T(\mathbf v)=\mathbf r+0.9\,\mathbf P\mathbf v$ never expands max-norm distances (largest observed ratio $0.876$), but expands some Euclidean distances by up to $1.418$. In the 2-norm weighted by the stationary distribution of $\mathbf P$ it is again a $0.9$-contraction (exact Lipschitz constant $0.9000$; largest ratio over random pairs $0.8935$). Iterations to reach $10^{-8}$ grow from 26 ($\gamma=0.5$) to 23,682 ($\gamma=0.999$).
* **KL.** Forward-KL fit to a bimodal mixture is $\mathcal N(-0.20, 2.31^2)$ (straddles both modes); reverse KL has three local optima, the best $\mathcal N(-2.00,0.60^2)$ sits on one mode.
* **Gradient estimators.** For $f(x)=x^2$, $\mu=\sigma=1$: measured variances 30.1 / 18.1 / 4.00 for score function / with baseline / reparameterization (theory 30 / 18 / 4). In $D=256$ dimensions the score-function variance is about $66{,}000\times$ the reparameterization variance; for $f(x)=\cos(\omega x)$ the reparameterization estimator is worse once $\omega\gtrsim0.9$ (variance $449.7$ at $\omega=30$, closed form $450.0$). At $\sigma=0.1$ the no-baseline score-function variance has risen to $113$, while with the baseline $\mathbb E f$ it stays at $8.0$ (it tends to $8\mu^2$ as $\sigma\to0$).
* **Optimisation.** GD with $\alpha=1/L$ on a quadratic with $\kappa=25$ shrinks the loss by $0.9216$ per step (theory $(1-1/\kappa)^2=0.9216$). Minibatch SGD with constant $\alpha$ stalls at an excess loss of $1.5\times10^{-2}$ ($\alpha=0.1$) or $1.6\times10^{-3}$ ($\alpha=0.01$); the decaying schedule reaches $4.1\times10^{-4}$ and is still improving.
* **PyTorch.** Autograd returns $-0.200249$ for the tanh example of §3.1, matching the hand computation. Adam reaches test MSE $0.0016$ against the clean function (SGD $0.025$, momentum $0.013$). The $(B,1)$-vs-$(B,)$ broadcasting bug silently trains the network to predict the mean (test MSE $0.475$).
* **Gymnasium.** A random CartPole agent lasts 22.1 steps on average and *every* episode terminates; with a 30-step limit, 18.6% of episodes are truncated only. A hand-written controller always reaches the 500-step limit, and the state it is truncated in is still worth $\approx 100 = 1/(1-\gamma)$ for $\gamma=0.99$.
