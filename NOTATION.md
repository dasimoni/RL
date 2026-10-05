# Notation and Conventions

This course follows the notation of Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018) wherever possible, and the dominant convention of the original papers in the deep-RL chapters. When a chapter departs from this table (usually to match a well-known paper), it says so explicitly at the point of departure.

## General

| Symbol | Meaning |
|---|---|
| $X$ (capital) vs. $x$ (lowercase) | random variable vs. a particular value it takes |
| $\mathbf{x}$, $\mathbf{w}$ (bold lowercase) | vectors (column vectors by default) |
| $\mathbf{A}$, $\mathbf{P}$ (bold uppercase) | matrices |
| $\mathbb{E}[X]$, $\mathbb{E}_\pi[X]$ | expectation; subscript says which policy/distribution generates the randomness |
| $\Pr\{X = x\}$ | probability |
| $\doteq$ | "is defined as" |
| $\leftarrow$ | assignment / update |
| $\mathbb{1}[\cdot]$ | indicator function |
| $\mathcal{H}(p)$ | entropy $-\sum_x p(x)\log p(x)$ |
| $D_{\mathrm{KL}}(p \,\Vert\, q)$ | KL divergence $\sum_x p(x)\log\frac{p(x)}{q(x)}$ |
| $\nabla_{\boldsymbol\theta} f$ | gradient of $f$ with respect to $\boldsymbol\theta$ |
| $\lVert \cdot \rVert_\infty$ | max norm |

## Markov decision processes

| Symbol | Meaning |
|---|---|
| $\mathcal{S}$, $\mathcal{A}$, $\mathcal{R}$ | state, action, reward sets ($\mathcal{A}(s)$ = actions available in $s$; $\mathcal{S}^+$ = states including terminal) |
| $t$ | discrete time step |
| $S_t, A_t, R_{t+1}$ | state, action at time $t$, and the reward that **results** from them (S&B convention: reward index is $t+1$) |
| $T$ | final time step of an episode |
| $p(s', r \mid s, a)$ | dynamics: $\Pr\{S_{t+1}=s', R_{t+1}=r \mid S_t=s, A_t=a\}$ |
| $p(s' \mid s, a)$ | state-transition probabilities |
| $r(s, a)$ | expected immediate reward $\mathbb{E}[R_{t+1} \mid S_t=s, A_t=a]$ |
| $d_0$ | initial-state distribution |
| $\gamma \in [0, 1]$ | discount factor |
| $G_t$ | return: $G_t \doteq \sum_{k=0}^{\infty}\gamma^k R_{t+k+1}$ (or the finite sum up to $T$) |
| $G_{t:t+n}$ | $n$-step return (bootstrapped from step $t+n$) |
| $G_t^\lambda$ | $\lambda$-return |

## Policies and value functions

| Symbol | Meaning |
|---|---|
| $\pi(a \mid s)$ | (stochastic) policy: probability of taking $a$ in $s$ |
| $\pi(s)$ | action taken in $s$ by a deterministic policy (tabular chapters) |
| $\mu_{\boldsymbol\theta}(s)$ | deterministic parameterized policy (DPG / DDPG / TD3 chapters) |
| $b(a \mid s)$ | behavior policy (off-policy learning); $\pi$ is then the target policy |
| $\rho_{t:h} \doteq \prod_{k=t}^{h}\frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)}$ | importance-sampling ratio |
| $v_\pi(s)$, $q_\pi(s,a)$ | true state- and action-value functions of $\pi$ |
| $v_\ast$, $q_\ast$, $\pi_\ast$ | optimal value functions and an optimal policy (written `v_\ast` in source to keep GitHub's renderer happy) |
| $A_\pi(s,a) \doteq q_\pi(s,a) - v_\pi(s)$ | advantage function |
| $V$, $Q$ (or $V_t$, $Q_t$) | tabular estimates of $v$, $q$ |
| $\hat v(s, \mathbf{w})$, $\hat q(s, a, \mathbf{w})$ | parameterized (approximate) value functions |
| $\mathbf{x}(s)$, $\mathbf{x}(s,a)$ | feature vectors (linear function approximation) |
| $\mathbf{w}$ | value-function / critic weights; $\mathbf{w}^-$ or $\bar{\mathbf{w}}$ = target-network weights |
| $\boldsymbol\theta$ | policy (actor) parameters |
| $\boldsymbol\psi$ | parameters of a learned model (dynamics, reward) |
| $d^\pi(s)$ | (normalized discounted or on-policy) state distribution under $\pi$; each chapter states which |
| $\mathcal{T}^\pi$, $\mathcal{T}^\ast$ | Bellman expectation and optimality operators |
| $\Pi$ | projection onto the representable function class |

## Learning

| Symbol | Meaning |
|---|---|
| $\alpha$ | step size (learning rate); $\alpha^{\mathbf{w}}$, $\alpha^{\boldsymbol\theta}$ when critic and actor differ |
| $\varepsilon$ | exploration rate in $\varepsilon$-greedy |
| $\delta_t$ | TD error, e.g. $\delta_t \doteq R_{t+1} + \gamma \hat v(S_{t+1},\mathbf{w}) - \hat v(S_t,\mathbf{w})$ |
| $\lambda$ | trace-decay parameter (TD($\lambda$)) and the GAE parameter |
| $\mathbf{z}_t$ | eligibility trace vector |
| $n$ | number of steps in $n$-step methods |
| $\tau$ | **context-dependent**: softmax/Boltzmann temperature (bandits, tabular) or Polyak averaging coefficient $\bar{\mathbf{w}} \leftarrow \tau\mathbf{w} + (1-\tau)\bar{\mathbf{w}}$ (deep RL). Each chapter states which. |
| $\alpha$ (SAC only) | entropy temperature in maximum-entropy RL — flagged where used, because it collides with the step size |
| $\epsilon$ (PPO only) | clipping range in PPO |

## Bandits

| Symbol | Meaning |
|---|---|
| $k$ | number of arms |
| $q_\ast(a)$ | true mean reward of arm $a$ |
| $Q_t(a)$, $N_t(a)$ | estimated value and pull count of arm $a$ before step $t$ |
| $\mathrm{Reg}(T)$ | cumulative (pseudo-)regret after $T$ pulls |

## Extensions

| Symbol | Meaning |
|---|---|
| $O_t$, $\mathcal{O}(o \mid s', a)$ | observation and observation kernel (POMDPs) |
| $b_t(s)$ | belief state (POMDP chapter only — not to be confused with the behavior policy $b$) |
| $g$ | goal (goal-conditioned RL) |
| $\omega$, $\beta_\omega$ | option and its termination function (hierarchical RL) |
| $i \in \{1,\dots,N\}$, $\mathbf{a} = (a^1,\dots,a^N)$, $a^{-i}$ | agent index, joint action, actions of all agents but $i$ (multi-agent) |
| $x$, $y$ | prompt and response (RL for language models) |
| $\pi_{\mathrm{ref}}$ | reference (usually supervised fine-tuned) policy |
| $r_\phi(x, y)$ | learned reward model |
| $\beta$ | KL-penalty coefficient (RLHF, DPO) |

## Code conventions

* Gymnasium API: `obs, info = env.reset(seed=...)`; `obs, reward, terminated, truncated, info = env.step(action)`.
  **Bootstrap through truncation, not through termination**: the TD target is $R_{t+1}$ alone only when `terminated` is true; when an episode is cut off by a time limit (`truncated`), the next state still has value.
* Random seeds are fixed so results are reproducible; every script prints its seed and settings.
* Every script supports `--quick` (a smoke test that finishes in about 30 seconds and writes no figures).
