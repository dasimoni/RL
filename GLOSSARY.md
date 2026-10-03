# Glossary

This glossary defines the terms, algorithms, theorems and acronyms used in the course, in alphabetical order. Each definition is short and follows the chapter that teaches the term, and the symbols are those of [NOTATION.md](NOTATION.md) ($v_\ast$, $q_\ast$, $\mathcal{T}^\pi$, $\rho_{t:h}$, $\pi_{\mathrm{ref}}$ and so on). Every entry ends with an arrow to the section where the term is taught, followed by up to two other places where it is used or developed further; the § numbers are the numbered headings inside each chapter. Terms that begin with a symbol are alphabetized by its spelled-out name ($\varepsilon$-greedy under E, $\lambda$-return under L, $n$-step return under N), closely related terms are often defined together in one entry (search the page for a term that has no headword of its own), and an acronym with two meanings in the course, such as IQL, is listed under both expansions. The chapter list is in [README.md](README.md).

[A](#a) · [B](#b) · [C](#c) · [D](#d) · [E](#e) · [F](#f) · [G](#g) · [H](#h) · [I](#i) · [J](#j) · [K](#k) · [L](#l) · [M](#m) · [N](#n) · [O](#o) · [P](#p) · [Q](#q) · [R](#r) · [S](#s) · [T](#t) · [U](#u) · [V](#v) · [W](#w) · [Z](#z)

## A

**A2C** (advantage actor-critic): The synchronous version of A3C. Several environments step in lock-step, and each rollout yields one batched update of an actor loss, a critic loss and an entropy bonus, usually with GAE advantages. → [Ch 10 §9.2](chapters/10-policy-gradients.md); also [Ch 10 §9.5](chapters/10-policy-gradients.md)

**A3C** (asynchronous advantage actor-critic): An actor-critic in which many actor-learners, each with its own copy of the environment, compute $n$-step advantage gradients and apply them asynchronously and without locks to shared parameters. → [Ch 10 §9.1](chapters/10-policy-gradients.md)

**Absorbing state**: A state that transitions only to itself with reward zero forever. Treating termination as entry into an absorbing state lets one return formula cover both episodic and continuing tasks. → [Ch 01 §4.4](chapters/01-the-rl-problem.md); also [Ch 00 §1.5](chapters/00-math-toolkit.md)

**A/B test** (A/B/n test): Pull each arm a fixed number of times $m$, then deploy the empirical winner forever. As a bandit algorithm it is explore-then-commit. It is best for inference about which arm is better; a bandit algorithm is better for earning while learning. → [Ch 02 §11.6](chapters/02-multi-armed-bandits.md); also [Ch 02 §14](chapters/02-multi-armed-bandits.md)

**ACKTR**: An actor-critic that approximates the natural gradient by a Kronecker-factored approximation of each layer's Fisher matrix (K-FAC), which is cheap to invert, combined with a KL-based trust region. → [Ch 11 §9](chapters/11-trust-regions-and-ppo.md)

**Action chunking**: Predicting a chunk of the next $H_c$ actions $a_{t:t+H_c-1}$ and executing several of them before querying the policy again, as Diffusion Policy and ACT (Action Chunking with Transformers) do. A chunk commits a multimodal policy to one mode for $H_c$ steps, cuts the number of decisions over which errors compound and makes pauses easier to imitate. The price is open-loop control inside a chunk, which ACT's temporal ensembling (averaging what the overlapping chunks predict for the current step) partly recovers. → [Ch 16 §2.8](chapters/16-offline-rl-and-imitation.md)

**Action gap**: The difference $\hat q(s,a_{(1)}) - \hat q(s,a_{(2)})$ between the best and second-best estimated action values. Small gaps make greedy decisions fragile; the gap is worth logging when debugging a value-based agent. → [Ch 09 §14](chapters/09-deep-q-learning.md); also [Ch 04 §4.3](chapters/04-monte-carlo.md)

**Action masking**: Removing forbidden actions from a discrete policy's support by setting their logits to $-\infty$; the simplest kind of shield. → [Ch 20 §11.4](chapters/20-deep-rl-in-practice.md)

**Action repeat** (frame skip): Repeating each chosen action for $k$ environment steps. The agent then sees a reward $\tilde R = \sum_{j=0}^{k-1}\gamma^j R_{t+j+1}$ per decision and an effective discount $\gamma^k$. DQN's Atari pipeline uses $k = 4$. → [Ch 20 §3.2](chapters/20-deep-rl-in-practice.md); also [Ch 09 §2.7](chapters/09-deep-q-learning.md)

**Action-value function** ($q_\pi$): The expected return from taking action $a$ in state $s$ and following $\pi$ afterwards, $q_\pi(s,a) \doteq \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$. Greedy improvement from action values needs no model, which is why model-free control learns them. → [Ch 01 §8](chapters/01-the-rl-problem.md); also [Ch 04 §3.1](chapters/04-monte-carlo.md)

**Action-value methods**: Bandit methods that estimate each action's mean reward $Q_t(a)$, for example by sample averages, and choose actions from those estimates (greedy, $\varepsilon$-greedy, UCB). → [Ch 02 §2](chapters/02-multi-armed-bandits.md)

**Actor-critic**: A method that learns both a policy (the actor) and a value function (the critic). The critic's bootstrapped estimate, for example the TD error, replaces the Monte Carlo return in the policy gradient, trading variance for bias. → [Ch 10 §7.3](chapters/10-policy-gradients.md); also [Ch 10 §7.1](chapters/10-policy-gradients.md), [Ch 12 §3](chapters/12-continuous-control-actor-critic.md)

**Adam**: An optimizer that keeps exponential moving averages of the gradient and of its elementwise square, divides one by the square root of the other, and corrects both for their initialization bias. It is the default optimizer of the deep-RL chapters; momentum (a decaying average of past gradients) is its first ingredient. → [Ch 00 §3.5](chapters/00-math-toolkit.md)

**Adaptive KL penalty** (PPO-penalty): The PPO variant that maximizes $\hat{\mathbb{E}}_t[\rho_t\hat A_t - \beta D_{\mathrm{KL}}(\pi_{\text{old}}\Vert\pi_{\boldsymbol\theta})]$ and halves or doubles $\beta$ to track a target KL. It is the ancestor of the KL penalty in RLHF. → [Ch 11 §7.5](chapters/11-trust-regions-and-ppo.md)

**Ad hoc teamwork**: Collaborating with teammates never met in training, possibly adapting to them during the interaction. The classic approach is type-based reasoning: hypothesize partner models ("types"), update a posterior over them from the partner's actions and best-respond to it, which is the Bayes-adaptive view with the partner's type as the hidden variable. The Hanabi challenge has an ad hoc team version. → [Ch 17 §9.6](chapters/17-multi-agent-rl.md); also [Ch 15 §7.2](chapters/15-beyond-mdps.md)

**Advantage function** ($A_\pi$): $A_\pi(s,a) \doteq q_\pi(s,a) - v_\pi(s)$, how much better action $a$ is than the policy's average behaviour in $s$; $\sum_a \pi(a \mid s) A_\pi(s,a) = 0$. With an exact critic, the TD error is an unbiased estimate of it. → [Ch 01 §8](chapters/01-the-rl-problem.md); also [Ch 10 §7.2](chapters/10-policy-gradients.md)

**Advantage normalization** (advantage whitening): Standardizing advantage estimates to zero mean and unit variance within each batch or minibatch. It is one of the implementation details of PPO and of RLHF. → [Ch 11 §8.2](chapters/11-trust-regions-and-ppo.md); also [Ch 18 §4.2](chapters/18-rl-for-language-models.md), [Ch 20 §3.4](chapters/20-deep-rl-in-practice.md)

**Advantage-weighted regression** (AWR): Policy extraction by weighted behaviour cloning, $\max_{\boldsymbol\theta}\mathbb{E}_{\mathcal D}[e^{\beta A}\log\pi_{\boldsymbol\theta}(a \mid s)]$, which fits the KL-constrained improved policy $\pi^\ast \propto \hat b\, e^{\beta A}$ using only logged actions. IQL extracts its policy this way; AWAC and CRR use the same update with a critic trained off-policy. → [Ch 16 §9.3](chapters/16-offline-rl-and-imitation.md); also [Ch 16 §6.3](chapters/16-offline-rl-and-imitation.md)

**Adversarial bandit**: A bandit whose rewards are set by an adversary instead of being drawn i.i.d. Performance is measured against the best fixed arm in hindsight, and a good algorithm must randomize (EXP3). → [Ch 02 §12](chapters/02-multi-armed-bandits.md)

**Afterstate**: The position after the agent's known, deterministic part of a transition but before the environment's random response (for example a board after our move, before the opponent's). Learning afterstate values shares one value among all state–action pairs that lead to the same afterstate. → [Ch 05 §12](chapters/05-temporal-difference.md); also [Ch 13 §10.5](chapters/13-model-based-rl.md)

**Agent57**: A distributed agent that extends Never Give Up with separate networks for intrinsic and extrinsic values and a bandit meta-controller that chooses among a family of exploration–exploitation trade-offs. It was the first agent to exceed the human benchmark on all 57 Atari games. MEME later reached the same milestone with roughly 200 times less experience. → [Ch 14 §8](chapters/14-exploration.md); also [Ch 09 §11](chapters/09-deep-q-learning.md)

**Agent–environment interface**: The formal loop of RL: in state $S_t$ the agent (the learner and decision maker) chooses $A_t$; the environment returns a reward $R_{t+1}$ and a next state $S_{t+1}$. The boundary is drawn around what the agent controls, not around what it knows. → [Ch 01 §2](chapters/01-the-rl-problem.md)

**Agentic RL** (multi-turn RL): RL for language models that act over many turns, calling tools such as search or a code interpreter and reading their outputs until a task is done or a budget runs out. Only the tokens the policy wrote enter the gradient (observation masks); credit across turns comes from trajectory-level group baselines or turn-level critics, and asynchronous samplers call for truncated importance weights. → [Ch 18 §13](chapters/18-rl-for-language-models.md); also [Ch 18 §13.1](chapters/18-rl-for-language-models.md), [Ch 18 §13.2](chapters/18-rl-for-language-models.md)

**AIRL** (adversarial inverse reinforcement learning): Adversarial imitation whose discriminator has a reward inside, $D = e^{f}/(e^{f} + \pi(a \mid s))$ with $f = g(s) + \gamma h(s') - h(s)$, so that training returns a disentangled reward that can transfer to new dynamics. → [Ch 16 §5.3](chapters/16-offline-rl-and-imitation.md)

**Aleatoric uncertainty**: Randomness in the environment itself (a dice roll, sensor noise) that does not shrink with more data. A planner should average over it; in a probabilistic ensemble it is the members' predicted variance. → [Ch 13 §3.2](chapters/13-model-based-rl.md); also [Ch 14 §7.2](chapters/14-exploration.md)

**Algorithm Distillation** (AD): An in-context RL method that trains a causal transformer by supervised learning on whole learning histories of a source RL algorithm, so that the transformer improves its policy in context without weight updates. → [Ch 15 §8](chapters/15-beyond-mdps.md)

**Aliasing**: Different hidden states that produce the same observation must be treated identically by a memoryless policy. Aliasing can make the best memoryless policy stochastic. → [Ch 15 §2.2](chapters/15-beyond-mdps.md); also [Ch 10 §1.1](chapters/10-policy-gradients.md)

**AlphaGo**: The 2016 Go program that combined a supervised policy network, a reinforcement-learning policy network, a value network and a fast rollout policy with Monte Carlo Tree Search. It beat Fan Hui and Lee Sedol. → [Ch 13 §9.1](chapters/13-model-based-rl.md)

**AlphaGo Zero**: AlphaGo without human data, rollouts or separate networks: one residual network with policy and value heads, trained from random play purely by self-play towards the outputs of its own search. → [Ch 13 §9.2](chapters/13-model-based-rl.md)

**AlphaStar**: The StarCraft II agent that reached Grandmaster level by combining supervised learning from human replays with multi-agent RL in a league of main agents, main exploiters and league exploiters. → [Ch 17 §9.4](chapters/17-multi-agent-rl.md)

**Alpha vector**: In POMDP planning, a vector $\boldsymbol\alpha$ over hidden states that gives the value of one conditional plan ("do $a$; then, depending on the observation, continue with another plan"). The optimal finite-horizon value is $V(b) = \max_{\boldsymbol\alpha\in\Gamma}\boldsymbol\alpha^\top b$, hence piecewise linear and convex (PWLC) in the belief. → [Ch 15 §2.5](chapters/15-beyond-mdps.md)

**AlphaZero**: The AlphaGo Zero algorithm applied to chess, shogi and Go. PUCT search guided by a policy/value network generates self-play games, and the network is trained towards the search's visit-count policy and the game outcome; search acts as a policy-improvement operator. → [Ch 13 §9.3](chapters/13-model-based-rl.md); also [Ch 13 §9.4](chapters/13-model-based-rl.md)

**Amortization** (of the argmax): Training an actor network to output a good action in one forward pass, instead of solving $\max_a Q(s,a)$ afresh for every state and every target. It is how off-policy actor-critics handle continuous actions. → [Ch 12 §1.2](chapters/12-continuous-control-actor-critic.md)

**Antithetic sampling** (antithetic pairs): Evaluating each random parameter perturbation in both directions and using the difference, $\nabla J_\sigma(\boldsymbol\theta) = \frac{1}{2\sigma}\mathbb{E}[(J(\boldsymbol\theta+\sigma\boldsymbol\xi) - J(\boldsymbol\theta-\sigma\boldsymbol\xi))\boldsymbol\xi]$. The difference cancels $J(\boldsymbol\theta)$, which would otherwise add variance $dJ(\boldsymbol\theta)^2/\sigma^2$, and every even-order Taylor term; on a quadratic the total variance is $(d+1)\lVert\nabla J\rVert^2$. Running both members of a pair on the same environment seed also cancels most of the episode noise. → [Ch 10 §15.2](chapters/10-policy-gradients.md); also [Ch 10 §15.6](chapters/10-policy-gradients.md)

**Ape-X**: A distributed DQN with hundreds of CPU actors and one GPU learner, simplifying the earlier Gorila design (many actors and learners around a parameter server). Each actor explores with its own $\varepsilon$ and computes initial priorities for its transitions; the learner is a double, dueling, $n$-step DQN with prioritized replay. → [Ch 09 §11](chapters/09-deep-q-learning.md)

**Apprenticeship learning**: Inverse RL that seeks a policy whose feature expectations match the expert's, so that it performs as well as the expert for every reward linear in the features (with bounded weights). → [Ch 16 §3.3](chapters/16-offline-rl-and-imitation.md)

**Approximate KL** (KL estimators $k_1$, $k_2$, $k_3$): Per-sample estimates of $D_{\mathrm{KL}}(\pi_{\text{old}}\Vert\pi_{\boldsymbol\theta})$ from actions sampled by $\pi_{\text{old}}$, with $\rho = \pi_{\boldsymbol\theta}/\pi_{\text{old}}$: $k_1 = -\log\rho$, $k_2 = \frac12(\log\rho)^2$, $k_3 = (\rho - 1) - \log\rho$. $k_3$ is unbiased and never negative, and is the usual logged `approx_kl`. → [Ch 11 §8.5](chapters/11-trust-regions-and-ppo.md); also [Ch 18 §9.2](chapters/18-rl-for-language-models.md), [Ch 20 §7.4](chapters/20-deep-rl-in-practice.md)

**Approximate value iteration**: Value iteration in which each backup is computed with error up to $\epsilon$; the errors accumulate to at most $\epsilon/(1-\gamma)$ in the limit. Fitted Q-iteration is the standard example. → [Ch 19 §2.2](chapters/19-rl-theory.md); also [Ch 08 §11.4](chapters/08-function-approximation.md)

**Arcade Learning Environment** (ALE): The platform that made Atari 2600 games a standard RL benchmark. Its reporting conventions shape how value-based results are compared: sticky actions (repeat the previous action with probability 0.25, so the deterministic emulator cannot be memorized), frame budgets and human-normalized scores. → [Ch 09 §13](chapters/09-deep-q-learning.md)

**ARS** (augmented random search): Antithetic random search over policy parameters, usually those of a static linear policy, with three cheap additions: the step is divided by the standard deviation of the returns it uses, states are normalized by running statistics ("V2"), and only the best $b$ of the $N$ directions are kept ("-t"). On the MuJoCo locomotion benchmarks it matched the sample efficiency of the best deep RL methods of its time with at least 15 times less computation than the fastest competing model-free methods, a warning about what those benchmarks measure. → [Ch 10 §15.4](chapters/10-policy-gradients.md)

**Asymmetric actor-critic**: Giving the critic privileged information available only in simulation (exact friction, true object positions) while the actor sees only what it will see at deployment. It is used in sim-to-real transfer. → [Ch 20 §3.1](chapters/20-deep-rl-in-practice.md); also [Ch 15 §3.5](chapters/15-beyond-mdps.md)

**Asynchronous dynamic programming**: DP that backs up one state at a time, in any order, using whatever values are currently stored. It converges as long as every state continues to be updated. → [Ch 03 §8](chapters/03-dynamic-programming.md)

**Atari 100k**: A benchmark that allows 100,000 agent steps (400,000 frames, about two hours of play) on each of 26 Atari games; the testbed for data-efficient RL. → [Ch 09 §12](chapters/09-deep-q-learning.md); also [Ch 13 §10.2](chapters/13-model-based-rl.md)

**Automatic temperature tuning**: SAC's dual gradient descent on the entropy temperature $\alpha$, $J(\alpha) = \mathbb{E}[-\alpha\log\pi(a \mid s) - \alpha\bar{\mathcal H}]$, which raises $\alpha$ when the policy's entropy falls below the target $\bar{\mathcal H}$ and lowers it otherwise. → [Ch 12 §6.5](chapters/12-continuous-control-actor-critic.md)

**Autoreset**: Vectorized Gymnasium environments reset finished sub-environments automatically. Under next-step autoreset (the Gymnasium 1.x default) the step after an episode ends returns the reset observation; it is not a real transition and must not be learned from. Vectorized environments step several copies in lock-step and batch their observations. → [Ch 00 §8.5](chapters/00-math-toolkit.md); also [Ch 20 §9.2](chapters/20-deep-rl-in-practice.md)

**Averager**: A fitting method whose fitted values are weighted averages of the targets, $\hat f(s,a) = \sum_i\kappa_i(s,a)\,y_i$ with $\kappa_i \ge 0$ and $\sum_i\kappa_i \le 1$, where the weights do not depend on the targets (Gordon, 1995). An averager is a sup-norm non-expansion, so fitted Q-iteration with one converges to a unique fixed point. $K$-nearest neighbours, kernel regression, state aggregation, grid interpolation and trees with target-independent splits are averagers; least squares with generalizing features is not, and can diverge. → [Ch 08 §11.4](chapters/08-function-approximation.md)

**Average reward** ($r(\pi)$): The long-run expected reward per step of a policy in a continuing task. With function approximation it is the principled objective for continuing tasks; values are then defined through the differential return. → [Ch 08 §12.1](chapters/08-function-approximation.md); also [Ch 10 §4.5](chapters/10-policy-gradients.md)

## B

**Backtracking line search**: The last stage of a TRPO update: shrink the proposed natural-gradient step geometrically until the surrogate objective improves and the sampled KL satisfies the constraint. → [Ch 11 §6.5](chapters/11-trust-regions-and-ppo.md)

**Backup diagram**: A picture of an update that shows which successor actions, rewards and states it averages over (all of them for an expected update, one sampled path for a sample update) and how far ahead it looks. → [Ch 01 §9.3](chapters/01-the-rl-problem.md); also [Ch 04 §2.6](chapters/04-monte-carlo.md), [Ch 05 §9.1](chapters/05-temporal-difference.md)

**Backward induction**: Exact solution of a finite-horizon problem from the end: $V^\ast_H = h$ (the terminal values) and $V^\ast_t = \mathcal{T}^\ast V^\ast_{t+1}$, which is exact after $H$ sweeps. Optimal policies then depend on the time left. → [Ch 03 §11](chapters/03-dynamic-programming.md)

**Backward view**: The incremental view of TD($\lambda$), in which each TD error is sent back to previously visited states in proportion to their eligibility traces. Offline it reproduces the forward view (the $\lambda$-return algorithm) exactly. → [Ch 06 §9](chapters/06-n-step-and-eligibility-traces.md); also [Ch 06 §10](chapters/06-n-step-and-eligibility-traces.md)

**Baird's counterexample**: A seven-state MDP with eight linear weights, a behaviour policy that differs from the target policy, and zero rewards, on which semi-gradient off-policy TD, and even expected DP-style updates, diverge although the true values are representable. → [Ch 08 §13.3](chapters/08-function-approximation.md); also [Ch 08 §14](chapters/08-function-approximation.md)

**Banach fixed-point theorem**: A $\gamma$-contraction on a complete metric space has a unique fixed point, and repeated application converges to it from any start at rate $\gamma^k$. It underlies the convergence of policy evaluation and value iteration. → [Ch 00 §4.3](chapters/00-math-toolkit.md); also [Ch 03 §2.4](chapters/03-dynamic-programming.md)

**Baseline** ($b(s)$): A function of the state, not the action, subtracted from the return in a score-function gradient estimate. It leaves the gradient unbiased and can greatly reduce its variance; a learned $\hat v(s,\mathbf{w})$ is the usual choice. → [Ch 10 §6](chapters/10-policy-gradients.md); also [Ch 00 §6.3](chapters/00-math-toolkit.md), [Ch 02 §8.3](chapters/02-multi-armed-bandits.md)

**Batch updating**: Presenting a fixed, finite dataset over and over and changing the estimates only once per sweep, by the sum of all increments, until they converge. Batch Monte Carlo then converges to per-state sample means, and batch TD(0) to the certainty-equivalence estimate. → [Ch 05 §4.1](chapters/05-temporal-difference.md); also [Ch 05 §4.3](chapters/05-temporal-difference.md)

**Bayes-adaptive MDP** (BAMDP): The belief MDP on the hyper-state $(s, b)$, where $b$ is the posterior over the unknown task. Its optimal policy is the Bayes-optimal policy for the prior. The epistemic POMDP of generalization and type-based ad hoc teamwork use the same construction. → [Ch 15 §7.2](chapters/15-beyond-mdps.md); also [Ch 15 §10.3](chapters/15-beyond-mdps.md), [Ch 17 §9.6](chapters/17-multi-agent-rl.md)

**Bayes filter** (belief update): The recursive posterior update of a POMDP, $b'(s') \propto \mathcal{O}(o \mid s', a)\sum_s p(s' \mid s, a)\, b(s)$, after taking action $a$ and observing $o$. → [Ch 15 §2.3](chapters/15-beyond-mdps.md)

**Bayesian regret**: Regret averaged over a prior distribution on environments. PSRL's guarantees are of this kind, which is weaker than a worst-case regret bound. → [Ch 14 §4.1](chapters/14-exploration.md)

**Bayes-optimal policy**: The policy that maximizes expected return averaged over a prior on the unknown environment. It explores exactly as much as information is worth over the remaining interaction; meta-RL methods can be read as approximations of it. → [Ch 15 §7.2](chapters/15-beyond-mdps.md); also [Ch 02 §7.1](chapters/02-multi-armed-bandits.md)

**BBF** (Bigger, Better, Faster): A data-efficient value-based agent for Atari 100k. It starts from SPR (Self-Predictive Representations, an auxiliary loss that predicts the agent's own future latent representations) with resets, scales up the network, uses a replay ratio of 8, anneals the update horizon and discount after each reset, and adds weight decay. → [Ch 09 §12](chapters/09-deep-q-learning.md)

**BCQ** (batch-constrained Q-learning): An offline RL policy-constraint method that only considers actions that are likely under the data. The discrete version allows only actions with $\hat b(a \mid s)/\max_{a'}\hat b(a' \mid s) > \tau_{\text{BCQ}}$; the continuous version samples candidates from a conditional VAE and perturbs them slightly. → [Ch 16 §7.2](chapters/16-offline-rl-and-imitation.md)

**BEAR** (bootstrapping error accumulation reduction): An offline RL method that keeps the policy approximately within the data's support by bounding the maximum mean discrepancy between samples of $\pi(\cdot \mid s)$ and $\hat b(\cdot \mid s)$, and takes a minimum over an ensemble of critics. → [Ch 16 §7.3](chapters/16-offline-rl-and-imitation.md)

**Behaviour cloning** (BC; also spelled behavior cloning): Imitation as supervised learning: maximize the likelihood of the expert's actions in the expert's states. Errors compound on the learner's own states, so its excess cost can grow as $\epsilon T^2$ with the horizon. Multimodal demonstrations call for expressive policy classes (mode averaging). → [Ch 16 §2](chapters/16-offline-rl-and-imitation.md); also [Ch 16 §14](chapters/16-offline-rl-and-imitation.md), [Ch 16 §2.8](chapters/16-offline-rl-and-imitation.md)

**Behaviour policy** ($b$; also spelled behavior policy): The policy that generates the data in off-policy learning, as opposed to the target policy $\pi$ being evaluated or improved. Importance sampling requires coverage: $b(a \mid s) > 0$ wherever $\pi(a \mid s) > 0$. → [Ch 04 §6.1](chapters/04-monte-carlo.md); also [Ch 01 §14](chapters/01-the-rl-problem.md)

**Belief MDP**: The MDP whose states are POMDP beliefs, with reward $\bar r(b,a) = \sum_s b(s) r(s,a)$ and transitions induced by the Bayes filter. Solving it solves the POMDP. → [Ch 15 §2.4](chapters/15-beyond-mdps.md)

**Belief state** ($b_t$): The posterior distribution over hidden states given the history of actions and observations. In a POMDP it is a sufficient statistic of the history. (Not to be confused with the behaviour policy $b$.) → [Ch 15 §2.3](chapters/15-beyond-mdps.md)

**Bellman equation** (Bellman expectation equation): The consistency condition $v_\pi(s) = \sum_a \pi(a \mid s)\sum_{s',r} p(s',r \mid s,a)[r + \gamma v_\pi(s')]$ and its action-value analogue. It is linear; in matrix form $\mathbf{v}_\pi = \mathbf{r}_\pi + \gamma\mathbf{P}_\pi\mathbf{v}_\pi$, with unique solution $(\mathbf{I} - \gamma\mathbf{P}_\pi)^{-1}\mathbf{r}_\pi$ when $\gamma < 1$. → [Ch 01 §9](chapters/01-the-rl-problem.md); also [Ch 01 §10.1](chapters/01-the-rl-problem.md)

**Bellman error** ($\overline{BE}$): The $\mu$-weighted squared norm of the Bellman error vector $\bar\delta_{\mathbf w} = \mathcal{T}^\pi v_{\mathbf w} - v_{\mathbf w}$. It is not learnable from data, and minimizing it by sampling needs two independent next states (double sampling). → [Ch 08 §15.1](chapters/08-function-approximation.md); also [Ch 08 §15.2](chapters/08-function-approximation.md)

**Bellman operator**: The expectation operator $\mathcal{T}^\pi V = \mathbf{r}_\pi + \gamma\mathbf{P}_\pi V$ and the optimality operator $(\mathcal{T}^\ast V)(s) = \max_a [r(s,a) + \gamma\sum_{s'} p(s' \mid s,a) V(s')]$. Both are monotone $\gamma$-contractions in the max norm, with fixed points $v_\pi$ and $v_\ast$. → [Ch 03 §2.2](chapters/03-dynamic-programming.md); also [Ch 03 §2.4](chapters/03-dynamic-programming.md), [Ch 03 §2.6](chapters/03-dynamic-programming.md)

**Bellman optimality equation**: $v_\ast(s) = \max_a \sum_{s',r} p(s',r \mid s,a)[r + \gamma v_\ast(s')]$, and $q_\ast(s,a) = \sum_{s',r} p(s',r \mid s,a)[r + \gamma\max_{a'} q_\ast(s',a')]$. It is nonlinear, and $v_\ast$ is its unique solution. → [Ch 01 §12.3](chapters/01-the-rl-problem.md); also [Ch 03 §2.5](chapters/03-dynamic-programming.md)

**Bellman rank**: A structural complexity measure: the rank of the matrix of average Bellman errors of candidate functions on the state distributions induced by other candidates' greedy policies. Problems with low Bellman rank are learnable with general function approximation (by the OLIVE algorithm, assuming realizability). Bilinear classes, witness rank and the decision–estimation coefficient are later complexity measures of the same kind. → [Ch 19 §9.3](chapters/19-rl-theory.md)

**Bernstein's inequality**: A concentration bound that uses the variance as well as the range of the variables. Combined with the law of total variance, it removes a factor of the horizon from sample-complexity bounds. → [Ch 19 §5.1](chapters/19-rl-theory.md); also [Ch 19 §5.3](chapters/19-rl-theory.md)

**Best-arm identification** (BAI, pure exploration): A bandit problem in which only the final recommendation matters. It is posed with a fixed budget (minimize the error probability or simple regret) or a fixed confidence (minimize the number of pulls). → [Ch 02 §15](chapters/02-multi-armed-bandits.md)

**Best-of-$n$ sampling** (BoN): Draw $n$ responses from the reference policy and return the one with the highest reward. It is a strong, KL-efficient baseline whose KL from the reference is at most $\log n - (n-1)/n$. → [Ch 18 §8.3](chapters/18-rl-for-language-models.md); also [Ch 18 §12](chapters/18-rl-for-language-models.md)

**Best response**: A policy that is optimal for one agent when the other agents' policies are held fixed, that is, an optimal policy of the MDP those policies induce. → [Ch 17 §3.1](chapters/17-multi-agent-rl.md); also [Ch 17 §2.3](chapters/17-multi-agent-rl.md)

**Bias–variance decomposition**: The bias of an estimator is $\mathbb{E}[\hat\theta] - \theta$; its mean-squared error is bias squared plus variance. The trade-off runs through the course: Monte Carlo targets are unbiased but noisy, bootstrapped targets are biased but less noisy. → [Ch 00 §2.1](chapters/00-math-toolkit.md)

**Bias–variance trade-off of $n$ and $\lambda$**: Longer multi-step returns (larger $n$ or $\lambda$) reduce the bias of bootstrapping from wrong estimates but increase the variance of long sampled returns. Intermediate values usually win; Monte Carlo and TD(0) are the two ends. → [Ch 06 §15](chapters/06-n-step-and-eligibility-traces.md); also [Ch 10 §8.3](chapters/10-policy-gradients.md)

**Bisimulation metric**: Two states are bisimilar if they earn the same rewards and move with the same probabilities into equivalent states. The bisimulation metric relaxes this to the fixed point of $d(s,t) = \max_a[\lvert r(s,a) - r(t,a)\rvert + \gamma W_1(p(\cdot \mid s,a), p(\cdot \mid t,a); d)]$, which bounds $\lvert v_\ast(s) - v_\ast(t)\rvert$. Representations built on this idea (DeepMDP, deep bisimulation for control) discard detail that affects neither rewards nor dynamics, but they are tailored to one reward. → [Ch 15 §6.6](chapters/15-beyond-mdps.md)

**Black-box policy search**: Treating the expected episodic return $J(\boldsymbol\theta)$ as a black-box function of the policy parameters: perturb the parameters once per episode, run whole episodes and compare their returns (finite differences, SPSA, evolution strategies, CEM, ARS). It needs no score function, backpropagation, value function or Markov property, allows deterministic and non-differentiable policies and parallelizes trivially, but its gradient variance grows with $d = \dim\boldsymbol\theta$ and it does no per-step credit assignment. → [Ch 10 §15](chapters/10-policy-gradients.md); also [Ch 10 §15.7](chapters/10-policy-gradients.md)

**Blackwell's sufficient conditions**: Any operator that is monotone and satisfies the constant-shift property $\mathcal{T}(u + c\mathbf{1}) \le \mathcal{T}u + \gamma c\mathbf{1}$ is a $\gamma$-contraction in the max norm. → [Ch 03 §2.4](chapters/03-dynamic-programming.md); also [Ch 03 §2.3](chapters/03-dynamic-programming.md)

**Blocking**: In conditioning, first training a stimulus A to predict the US prevents a stimulus X, later presented in compound with A, from acquiring associative strength, although X is paired with the US as often as in a control group (Kamin, 1969). Rescorla–Wagner and the TD model explain it because all the stimuli present share one prediction error: pairing is not enough for learning, the US must be surprising. → [Ch 05 §14.2](chapters/05-temporal-difference.md)

**Boltzmann exploration** (softmax exploration): Choosing action $a$ with probability proportional to $e^{Q(a)/\tau}$. It grades exploration by estimated value but is sensitive to the scale of the values; it maximizes $\sum_a \pi(a) Q(a) + \tau\mathcal{H}(\pi)$. → [Ch 02 §9](chapters/02-multi-armed-bandits.md); also [Ch 14 §2](chapters/14-exploration.md)

**Bootstrapped DQN**: An ensemble of Q-heads trained on different bootstrap resamples of the data; at the start of each episode one head is sampled and followed greedily, approximating posterior sampling. Adding randomized prior functions makes it explore deeply. → [Ch 14 §4.3](chapters/14-exploration.md)

**Bootstrapping**: Updating an estimate from other current estimates, as the TD target $R_{t+1} + \gamma V(S_{t+1})$ does, instead of waiting for a complete return. It lowers variance but introduces bias, and is one ingredient of the deadly triad. → [Ch 05 §1.2](chapters/05-temporal-difference.md); also [Ch 01 §14](chapters/01-the-rl-problem.md), [Ch 08 §14](chapters/08-function-approximation.md)

**BRAC** (behaviour-regularized actor-critic): A systematic study of offline policy regularization (penalties on the policy objective, the value target or both, with several divergences). It found that simple choices such as a well-tuned KL penalty match more elaborate methods. → [Ch 16 §7.3](chapters/16-offline-rl-and-imitation.md)

**Bradley–Terry model**: The preference model $\Pr\lbrace y_1 \succ y_2 \mid x\rbrace = \sigma(r(x,y_1) - r(x,y_2))$, which follows from a random-utility model with Gumbel noise. Its negative log-likelihood is the standard reward-model loss; applied repeatedly to rankings it becomes the Plackett–Luce model. → [Ch 18 §2.2](chapters/18-rl-for-language-models.md); also [Ch 18 §2.3](chapters/18-rl-for-language-models.md)

**Branching factor** ($b$ in Chapter 07): The number of possible next states of a state–action pair. After $t$ sample updates the error of a value estimate is about $\sqrt{(b-1)/(bt)}$ times the outcome standard deviation, so sample updates win when $b$ is large. → [Ch 07 §6.1](chapters/07-planning-and-learning-tabular.md)

**BRO** (Bigger, Regularized, Optimistic): A continuous-control agent that scales the critic to millions of parameters using layer-normalized residual blocks, together with resets, distributional critics, a high update-to-data ratio and optimistic exploration. → [Ch 12 §8](chapters/12-continuous-control-actor-critic.md)

**Burn-in**: R2D2's fix for stale recurrent state in replay: start a replayed sequence from the stored hidden state, run the current network over a prefix without computing losses to refresh it, and train only on the remaining steps. → [Ch 09 §11](chapters/09-deep-q-learning.md); also [Ch 15 §3.3](chapters/15-beyond-mdps.md)

## C

**C51** (categorical DQN): Distributional RL with a categorical return distribution on 51 fixed atoms. The Bellman target distribution is projected back onto the atoms and fitted by cross-entropy. → [Ch 09 §9.3](chapters/09-deep-q-learning.md)

**Catastrophic interference**: The degradation of values elsewhere in the state space when a neural network is trained on a stream of correlated, local updates, because all states share the parameters. → [Ch 09 §1.3](chapters/09-deep-q-learning.md)

**Causal confusion**: A failure of behaviour cloning in which extra information lets the learner predict the expert's action from a consequence of it (a cloned driver brakes "because" the brake light is on). → [Ch 16 §2.7](chapters/16-offline-rl-and-imitation.md)

**Causality** (reward-to-go): In REINFORCE, each score term $\nabla\log\pi(A_t \mid S_t)$ is multiplied only by the rewards that follow it, $G_t$, since earlier rewards contribute zero in expectation. The estimate stays unbiased and has lower variance. → [Ch 10 §4.3](chapters/10-policy-gradients.md); also [Ch 10 §6.4](chapters/10-policy-gradients.md)

**CEM** (cross-entropy method): An iterative sampling optimizer: sample action sequences from a diagonal Gaussian, keep the best $K$ (the elites), refit the Gaussian to them, and repeat. It is the standard planner inside MPC methods such as PETS and PlaNet. Applied to policy parameters once per batch of episodes, it is also a black-box policy search method whose elite variance tends to shrink too fast (adding noise to the variance at every refit fixes this); it is the hard-threshold relative of REPS and RWR. → [Ch 13 §4.3](chapters/13-model-based-rl.md); also [Ch 10 §15.3](chapters/10-policy-gradients.md), [Ch 12 §9.1](chapters/12-continuous-control-actor-critic.md)

**Certainty-equivalence estimate**: The value function of the maximum-likelihood Markov model fitted to the data, treated as if it were exact. Batch TD(0) converges to it, while batch Monte Carlo converges to the minimum-training-error estimate. → [Ch 05 §4.3](chapters/05-temporal-difference.md)

**CFR** (counterfactual regret minimization): Runs a regret minimizer (usually regret matching) on the counterfactual regrets at every information set of an extensive-form game. In two-player zero-sum games the reach-weighted average strategy converges to a Nash equilibrium. → [Ch 17 §10.3](chapters/17-multi-agent-rl.md)

**CFR+**: CFR with regret matching+, alternating updates and averaging weighted by iteration number. It converges much faster in practice and was used to essentially solve heads-up limit Texas hold'em. Linear and Discounted CFR also down-weight early iterations, and Deep CFR replaces the regret tables by neural networks trained on MCCFR samples. → [Ch 17 §10.4](chapters/17-multi-agent-rl.md)

**Chance node**: A node in a game or search tree at which chance (the environment) moves with fixed probabilities, such as a card deal or the outcome of a stochastic transition. → [Ch 17 §10.1](chapters/17-multi-agent-rl.md); also [Ch 07 §11.1](chapters/07-planning-and-learning-tabular.md), [Ch 13 §10.5](chapters/13-model-based-rl.md)

**Chernoff–KL bound**: The concentration bound $\Pr\lbrace\bar X_n - p \ge u\rbrace \le e^{-n\,\mathrm{kl}(p+u,p)}$ for Bernoulli rewards. It is much tighter than Hoeffding's inequality for small $p$ and leads to KL-UCB. → [Ch 02 §6.2](chapters/02-multi-armed-bandits.md); also [Ch 02 §11.4](chapters/02-multi-armed-bandits.md)

**Classical conditioning** (Pavlovian conditioning): A neutral conditioned stimulus (CS, such as a tone) repeatedly followed by an unconditioned stimulus (US, such as food) comes to evoke a response that anticipates the US. Nothing the animal does changes what happens, so it is a prediction problem, modelled by Rescorla–Wagner and the TD model; in instrumental conditioning, actions matter. → [Ch 05 §14.2](chapters/05-temporal-difference.md); also [Ch 05 §14.3](chapters/05-temporal-difference.md)

**Cliff walking**: A gridworld in which stepping into the cliff along one edge costs $-100$ and sends the agent back to the start. Under fixed $\varepsilon$-greedy exploration, Q-learning learns the optimal edge path but earns less while exploring than SARSA and Expected SARSA, which learn a safer path: online performance differs from the learned policy. → [Ch 05 §10](chapters/05-temporal-difference.md)

**Clip-higher**: DAPO's asymmetric ratio clip, $\mathrm{clip}(\rho, 1-\epsilon_{\mathrm{low}}, 1+\epsilon_{\mathrm{high}})$ with $\epsilon_{\mathrm{high}} > \epsilon_{\mathrm{low}}$, which lets unlikely tokens with positive advantage grow and counteracts entropy collapse. → [Ch 18 §9.3](chapters/18-rl-for-language-models.md)

**Clipped double-Q** (CDQ, clipped double Q-learning): The target used by TD3 and SAC, $y = r + \gamma(1 - \text{term})\min_{j=1,2} Q_{\bar{\mathbf w}_j}(s', \tilde a')$, which takes the smaller of two target critics to counter overestimation. → [Ch 12 §4.3](chapters/12-continuous-control-actor-critic.md); also [Ch 12 §6.4](chapters/12-continuous-control-actor-critic.md)

**Clipped surrogate objective** ($L^{CLIP}$): PPO's objective $\hat{\mathbb{E}}_t[\min(\rho_t\hat A_t, \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat A_t)]$. A sample's gradient is switched off only once its ratio has moved past $1 \pm \epsilon$ in the direction its advantage favours; it is a pessimistic surrogate, not a hard trust region. The clip fraction (share of samples with $\lvert\rho-1\rvert > \epsilon$) is a diagnostic, not the share of samples with zero gradient. → [Ch 11 §7.1](chapters/11-trust-regions-and-ppo.md); also [Ch 11 §7.2](chapters/11-trust-regions-and-ppo.md), [Ch 11 §8.5](chapters/11-trust-regions-and-ppo.md)

**CMA-ES** (covariance matrix adaptation evolution strategy): The heavily engineered practical relative of natural evolution strategies, with rank-based weights, a full covariance matrix updated from successful steps and from an evolution path of recent mean shifts, and separate step-size control. Its $O(d^2)$ covariance limits it to at most a few thousand parameters, a range in which it is among the most reliable black-box optimizers; it trained the controller of World Models. → [Ch 10 §15.3](chapters/10-policy-gradients.md); also [Ch 13 §7.2](chapters/13-model-based-rl.md)

**Coarse coding**: Linear features that are binary indicators of overlapping receptive fields; a state activates every field that contains it, and the fields' size and shape govern generalization. → [Ch 08 §6.4](chapters/08-function-approximation.md)

**Coarse correlated equilibrium** (CCE): A distribution over joint actions such that no agent gains by committing in advance to a fixed action instead of following the recommendation. The empirical play of no-regret learners converges to the set of CCEs. → [Ch 17 §3.3](chapters/17-multi-agent-rl.md); also [Ch 17 §5.4](chapters/17-multi-agent-rl.md)

**COMA** (counterfactual multi-agent policy gradients): A multi-agent actor-critic with a centralised critic and a counterfactual baseline that marginalizes out only agent $i$'s action: $Q(s,\mathbf a) - \sum_{a'^i}\pi^i(a'^i \mid \tau^i)Q(s,(a'^i, a^{-i}))$. → [Ch 17 §8.3](chapters/17-multi-agent-rl.md)

**Combination lock**: A chain in which one action per state moves forward and any other resets the agent to the start. Dithering exploration needs a number of steps exponential in the length (a random walk needs $2^{N+1}-2$ on average), which is why exploration must be deep. → [Ch 14 §1.2](chapters/14-exploration.md); also [Ch 19 §6.2](chapters/19-rl-theory.md)

**Compatible function approximation**: A critic $f_{\mathbf w}$ that is linear in the score features, $\nabla_{\mathbf w} f_{\mathbf w} = \nabla\log\pi$, and fitted by least squares gives the exact policy gradient; its weight vector is the natural gradient. → [Ch 10 §13](chapters/10-policy-gradients.md); also [Ch 11 §5.4](chapters/11-trust-regions-and-ppo.md), [Ch 12 §2.5](chapters/12-continuous-control-actor-critic.md)

**Completeness** (Bellman completeness): A function class is complete if it is closed under Bellman backups, $\mathcal{T}^\ast f \in \mathcal{F}$ for every $f \in \mathcal{F}$. It is much stronger than realizability, and fitted Q-iteration relies on it. → [Ch 19 §9.1](chapters/19-rl-theory.md)

**Compounding error**: The growth of error over a multi-step rollout of a learned model (for a Lipschitz model, $e_k \le \varepsilon_f(L^k-1)/(L-1)$), and, in imitation, the growth of a cloned policy's cost as its mistakes take it to states the expert never visited. → [Ch 13 §2.1](chapters/13-model-based-rl.md); also [Ch 16 §2.3](chapters/16-offline-rl-and-imitation.md)

**Concentrability coefficient**: A bound on how much the state–action distribution of a policy can exceed the data distribution, for example $C_{\mathrm{all}} = \max_\pi\lVert d^\pi_\rho/\nu_D\rVert_\infty$ (all-policy, with $\nu_D$ the data distribution) or $C^\ast = \lVert d^{\pi_\ast}_\rho/\nu_D\rVert_\infty$ for the comparator policy only (single-policy). Offline RL guarantees scale with it. → [Ch 19 §10.1](chapters/19-rl-theory.md); also [Ch 16 §6.3](chapters/16-offline-rl-and-imitation.md)

**Concentration inequality**: A bound on the probability that an empirical average deviates from its mean by more than a given amount; Hoeffding, Bernstein, Azuma–Hoeffding and the Chernoff–KL bound are the ones used in the course. → [Ch 02 §6.2](chapters/02-multi-armed-bandits.md); also [Ch 19 §5.1](chapters/19-rl-theory.md)

**Conjugate gradient** (CG): An iterative method that solves $\mathbf{F}\mathbf{x} = \mathbf{g}$ for a symmetric positive (semi-)definite $\mathbf F$ using only matrix–vector products. TRPO uses it, with Fisher-vector products, to compute the natural-gradient direction. → [Ch 11 §6.3](chapters/11-trust-regions-and-ppo.md)

**Conservative policy iteration** (CPI): A policy update to a mixture $(1-\kappa)\pi + \kappa\pi'$ with small $\kappa$, chosen so that a lower bound on the improvement is positive; the precursor of TRPO's monotonic-improvement argument. → [Ch 11 §3.4](chapters/11-trust-regions-and-ppo.md)

**Conservative Q-learning** (CQL): An offline RL method whose loss adds to the TD loss a term that pushes Q-values down under a chosen action distribution and up on the dataset's actions, so the learned values lower-bound the policy's true values. The discrete CQL(H) penalty is $\alpha\,\mathbb{E}_s[\log\sum_a e^{Q(s,a)} - \mathbb{E}_{a\sim\mathcal D}Q(s,a)]$. → [Ch 16 §8](chapters/16-offline-rl-and-imitation.md); also [Ch 16 §8.4](chapters/16-offline-rl-and-imitation.md)

**Consistent**: An estimator is consistent if it converges in probability to the true value as the sample size grows; biased estimators can be consistent. A bandit algorithm is called consistent if its regret grows more slowly than any power of $T$ on every instance of a class, the premise of the Lai–Robbins bound. → [Ch 00 §2.1](chapters/00-math-toolkit.md); also [Ch 02 §11.4](chapters/02-multi-armed-bandits.md)

**Constant per-step reward**: A constant $c$ added to every reward of an episodic task whose length the agent controls changes behaviour: an alive bonus rewards survival and a step penalty rewards ending the episode. Termination is part of the reward. → [Ch 20 §2.2](chapters/20-deep-rl-in-practice.md)

**Constant step size**: A fixed $\alpha$ in $Q \leftarrow Q + \alpha(\text{target} - Q)$. It produces an exponential recency-weighted average that tracks nonstationary targets but never converges, leaving a noise floor proportional to $\alpha$. → [Ch 00 §2.4](chapters/00-math-toolkit.md); also [Ch 02 §4](chapters/02-multi-armed-bandits.md)

**Constitutional AI**: Training a harmless assistant with no human harmlessness labels. A model critiques and revises its own responses against a short list of written principles (the constitution) for supervised fine-tuning, then AI-generated preference labels train a preference model for RL. → [Ch 18 §7](chapters/18-rl-for-language-models.md)

**Constrained MDP** (CMDP): An MDP with additional cost signals whose expected discounted totals must stay below budgets $d_i$ while the reward return is maximized; the standard formalism of safe RL. → [Ch 20 §11.1](chapters/20-deep-rl-in-practice.md)

**Constrained Policy Optimization** (CPO): TRPO's trust-region machinery applied to constrained MDPs: each update maximizes a linearized reward objective subject to a linearized cost constraint and a quadratic (Fisher) model of the KL. → [Ch 20 §11.3](chapters/20-deep-rl-in-practice.md)

**Contextual bandit**: A bandit in which the best action depends on an observed context, but actions do not influence future contexts. It sits between bandits and full RL; language generation viewed as one response per prompt is a contextual bandit. Sutton & Barto call the problem associative search. → [Ch 02 §13](chapters/02-multi-armed-bandits.md); also [Ch 18 §1.2](chapters/18-rl-for-language-models.md)

**Contextual MDP**: A family $\lbrace\mathcal{M}_c\rbrace_{c \in \mathcal{C}}$ of MDPs with common states and actions, in which a context $c \sim p(c)$ (a level seed, physical parameters, a layout) fixes the dynamics, the reward and the initial-state distribution. The objective averages over $p(c)$, but training sees only finitely many contexts, which opens a generalization gap; the agent usually does not observe $c$. It is not a constrained MDP (CMDP). → [Ch 15 §10.1](chapters/15-beyond-mdps.md)

**Continual RL**: RL in a non-stationary world or task sequence. The agent must balance catastrophic forgetting (training on the current task overwrites earlier ones) against loss of plasticity (it can no longer learn new ones). → [Ch 15 §9](chapters/15-beyond-mdps.md)

**Continuing task**: A task that goes on without end, so returns must be discounted (or replaced by the average reward) to be finite. → [Ch 01 §4.2](chapters/01-the-rl-problem.md); also [Ch 08 §12](chapters/08-function-approximation.md)

**Contraction**: A map $T$ with $\lVert T\mathbf{x} - T\mathbf{y}\rVert \le \gamma\lVert\mathbf{x} - \mathbf{y}\rVert$ for some $\gamma < 1$. The Bellman operators are $\gamma$-contractions in the max norm, which is why DP converges. → [Ch 00 §4.3](chapters/00-math-toolkit.md); also [Ch 03 §2.4](chapters/03-dynamic-programming.md)

**Contrastive RL**: Learns a goal-conditioned critic $f(s,a,g) = \boldsymbol\phi(s,a)^\top\boldsymbol\psi(g)$ by classifying states visited later in the same trajectory (positives) against states drawn from the data (negatives). The optimal critic is $\log[p^\pi_\gamma(g \mid s,a)/p(g)]$, where $p^\pi_\gamma$ is the normalized discounted occupancy, so it is a goal-reaching Q-function learned without ever writing down a reward. → [Ch 15 §6.6](chapters/15-beyond-mdps.md)

**Control**: The problem of finding a good or optimal policy, as opposed to prediction (evaluating a fixed policy). Most control methods alternate prediction and improvement (generalized policy iteration). → [Ch 01 §14](chapters/01-the-rl-problem.md)

**Control as inference** (RL as inference): Attach to every step a binary optimality variable with $p(\mathcal{O}_t = 1 \mid s_t, a_t) \propto \exp(r(s_t,a_t)/\alpha)$ and ask for the posterior over trajectories given success at every step. The backward messages are soft action values and the posterior policy is a Boltzmann policy, but exact inference also conditions the transitions on success, so it is optimistic about luck. Restricting the approximate posterior to the true dynamics turns the evidence lower bound into the maximum-entropy RL objective. → [Ch 12 §5.7](chapters/12-continuous-control-actor-critic.md); also [Ch 12 §9.1](chapters/12-continuous-control-actor-critic.md)

**Control variate**: A variance-reduction device: subtract from an estimator a correlated quantity with known mean, $X - c(Y - \mathbb{E}Y)$. Baselines, per-decision control variates in off-policy returns and the doubly robust estimator are all control variates. → [Ch 00 §6.3](chapters/00-math-toolkit.md); also [Ch 06 §5](chapters/06-n-step-and-eligibility-traces.md), [Ch 16 §12.4](chapters/16-offline-rl-and-imitation.md)

**Convex coverage set** (CCS): In multi-objective RL with a linear utility $\mathbf{w}^\top\mathbf{V}$ and unknown weights $\mathbf{w} \ge 0$, a set of policies that contains a maximizer of $\mathbf{w}^\top\mathbf{V}^\pi$ for every $\mathbf{w}$. It reaches only the supported points of the Pareto front, those on the boundary of its convex hull; under SER, random mixtures of CCS policies attain every point between them. → [Ch 15 §6.7](chapters/15-beyond-mdps.md)

**Correlated equilibrium** (CE): A distribution over joint actions such that no agent gains by deviating from its recommended action after seeing it. The set of CEs is a polytope computable by linear programming that contains every Nash equilibrium; the empirical play of no-swap-regret learners converges to it. → [Ch 17 §3.3](chapters/17-multi-agent-rl.md)

**Count-based exploration**: Adding an intrinsic bonus that decreases with a visit count, typically $\beta/\sqrt{n(s)}$; pseudo-counts and hashing extend it to large state spaces. → [Ch 14 §6.1](chapters/14-exploration.md)

**Counterfactual regret** (and counterfactual value): The counterfactual value $v_i^\sigma(I)$ of an information set is player $i$'s expected utility there, weighted by the probability that the other players and chance reach it. Counterfactual regret is regret measured in these values; keeping it small at every information set bounds overall regret (CFR). → [Ch 17 §10.3](chapters/17-multi-agent-rl.md)

**Covariate shift**: The mismatch between the expert's state distribution, on which behaviour cloning is trained, and the learner's own state distribution, on which it is tested. → [Ch 16 §2.2](chapters/16-offline-rl-and-imitation.md)

**Coverage**: The requirement that the data contain the actions (and states) a target policy uses. In importance sampling it means $b(a \mid s) > 0$ wherever $\pi(a \mid s) > 0$; in offline RL it decides whether learning is possible at all. → [Ch 04 §6.2](chapters/04-monte-carlo.md); also [Ch 16 §6.3](chapters/16-offline-rl-and-imitation.md), [Ch 19 §10.1](chapters/19-rl-theory.md)

**Credit assignment**: Deciding which earlier actions deserve credit or blame for an outcome. Temporally, it is the problem multi-step returns and traces address; in multi-agent RL, it is the problem of separating each agent's contribution to a team reward. → [Ch 01 §1.2](chapters/01-the-rl-problem.md); also [Ch 06 §1.1](chapters/06-n-step-and-eligibility-traces.md), [Ch 17 §6.2](chapters/17-multi-agent-rl.md)

**Cross-entropy**: $\mathcal{H}(p,q) = -\sum_x p(x)\log q(x) = \mathcal{H}(p) + D_{\mathrm{KL}}(p\Vert q)$; minimizing it in $q$ is forward-KL minimization. → [Ch 00 §5.2](chapters/00-math-toolkit.md)

**CrossQ**: A continuous-control agent that removes target networks and uses batch renormalization in the critic, passing current and next state–action batches jointly; it reaches high sample efficiency at an update-to-data ratio of 1. → [Ch 12 §8](chapters/12-continuous-control-actor-critic.md)

**CTDE** (centralised training with decentralised execution): The multi-agent paradigm in which training may use the global state, all observations and all actions, but each agent must act at test time on its own action–observation history. → [Ch 17 §7.1](chapters/17-multi-agent-rl.md)

**Curiosity** (prediction-error curiosity): An intrinsic reward equal to the error of a learned forward model, which drives the agent to where it cannot yet predict; it is trapped by unpredictable noise (the noisy-TV problem). → [Ch 14 §7.1](chapters/14-exploration.md); also [Ch 14 §7.2](chapters/14-exploration.md)

**Curse of dimensionality**: The number of states grows exponentially with the number of state variables, so exact methods that are polynomial in $\lvert\mathcal S\rvert$ become infeasible. Sampling, focusing and generalization are the ways around it. The linear-quadratic regulator is the exception: its value functions stay in a finite-dimensional quadratic family. → [Ch 03 §12.3](chapters/03-dynamic-programming.md); also [Ch 03 §11.4](chapters/03-dynamic-programming.md)

**Curse of horizon**: The second moment of trajectory importance weights grows exponentially with the horizon: if every state has the same per-step $\chi^2$-divergence between $\pi$ and $b$, then $\mathbb{E}_b[\rho^2_{0:H-1}] = (1 + \chi^2)^H$. It makes trajectory-level IS unusable for long episodes. → [Ch 16 §12.2](chapters/16-offline-rl-and-imitation.md); also [Ch 00 §2.6](chapters/00-math-toolkit.md)

**CVaR** (conditional value at risk): The expected return in the worst $\alpha$-fraction of outcomes, $\mathrm{CVaR}_\alpha(Z) = \max_\nu\lbrace\nu - \frac1\alpha\mathbb{E}[(\nu - Z)^+]\rbrace$; a risk-sensitive objective for safety. It can also be read as robustness to modelling errors, since it is the smallest expectation over reweightings whose density ratio is at most $1/\alpha$. → [Ch 20 §11.5](chapters/20-deep-rl-in-practice.md); also [Ch 20 §11.6](chapters/20-deep-rl-in-practice.md)

## D

**D4RL**: The standard offline RL benchmark: fixed datasets for MuJoCo locomotion at several quality levels (random, medium, medium-replay, medium-expert, expert) plus AntMaze and other tasks, scored by normalized return. The same section covers offline-to-online fine-tuning, in which an offline-trained agent continues learning online. → [Ch 16 §13](chapters/16-offline-rl-and-imitation.md)

**DAgger** (dataset aggregation): Interactive imitation learning that runs the learner's policy, asks the expert to label the states the learner actually visits, aggregates them into the dataset and retrains. It reduces behaviour cloning's quadratic compounding error to $uT\epsilon$ for recoverable tasks. → [Ch 16 §2.5](chapters/16-offline-rl-and-imitation.md)

**DAPO** (Decoupled Clip and Dynamic sAmpling Policy Optimization): A recipe for long chain-of-thought RL that modifies GRPO with clip-higher, dynamic sampling (discarding groups whose rewards are all equal), a token-level loss and overlong-response shaping. → [Ch 18 §9.3](chapters/18-rl-for-language-models.md)

**DDP** (differential dynamic programming): Trajectory optimization like iLQR, but keeping the second derivatives of the dynamics in the backward pass; it costs more per iteration and converges quadratically near the optimum. It expands the value function to second order around one trajectory, between the HJB equation (global) and Pontryagin's maximum principle (conditions along one trajectory). Guided policy search uses DDP solutions to train a neural-network policy. → [Ch 03 §11.5](chapters/03-dynamic-programming.md)

**DDPG** (deep deterministic policy gradient): An off-policy actor-critic for continuous actions: a deterministic actor trained by the deterministic policy gradient, a Q-critic trained from replay with target networks updated by Polyak averaging, and additive exploration noise. Sample-efficient but brittle, mainly because the actor exploits critic overestimation. → [Ch 12 §3](chapters/12-continuous-control-actor-critic.md)

**Deadly triad**: The combination of function approximation, bootstrapping and off-policy training, which can make value estimates diverge even with exact expected updates. Removing any one of the three restores stability in the linear case. → [Ch 08 §14](chapters/08-function-approximation.md); also [Ch 09 §1.3](chapters/09-deep-q-learning.md)

**Decision-time planning**: Planning that starts from the current state and whose only output is the current action, after which most of the computation is discarded (heuristic search, rollouts, MCTS). Contrast background planning. → [Ch 07 §9](chapters/07-planning-and-learning-tabular.md); also [Ch 13 §9](chapters/13-model-based-rl.md)

**Decision Transformer**: Offline RL as sequence modelling: a causal transformer over (return-to-go, state, action) tokens is trained by supervised learning and, at test time, conditioned on a high desired return. Like other return-conditioned supervised learning (RCSL, "upside-down RL"), it cannot stitch sub-trajectories and confuses luck with skill in stochastic environments. The Trajectory Transformer instead models whole trajectories and plans with beam search. → [Ch 16 §11.1](chapters/16-offline-rl-and-imitation.md); also [Ch 16 §11.2](chapters/16-offline-rl-and-imitation.md)

**Dec-POMDP** (decentralised POMDP): A cooperative multi-agent model with a common reward in which each agent receives only a private observation and acts on its own action–observation history. Solving it optimally is NEXP-complete. With individual rewards the model is a partially observable stochastic game (POSG). → [Ch 17 §2.4](chapters/17-multi-agent-rl.md)

**Deep exploration**: Choosing actions for the information they may yield several steps later, that is, committing to multi-step journeys whose payoff is only informational. Optimism with planning and posterior sampling achieve it; dithering does not. → [Ch 14 §1.1](chapters/14-exploration.md); also [Ch 14 §5](chapters/14-exploration.md)

**Deep neuroevolution** (deep GA): Evolving the weights of deep networks with a simple genetic algorithm (Gaussian mutations and truncation selection, no crossover and no gradient estimate). It evolved Atari networks with over four million parameters, beating DQN, A3C or ES on some games and losing on others, and stored each individual as a list of random seeds. → [Ch 10 §15.5](chapters/10-policy-gradients.md)

**DeepSea**: A combination lock with a built-in temptation: an $N \times N$ grid in which every step moves one row down and one column left or right, moving right costs $0.01/N$, and only moving right from the bottom-right cell pays the treasure $+1$ (optimal return $0.99$). A uniform policy finds it with probability $2^{-N}$, and $\varepsilon$-greedy does worse, so it separates deep from dithering exploration. → [Ch 14 §1.3](chapters/14-exploration.md); also [Ch 14 §5.1](chapters/14-exploration.md)

**Deep Sea Treasure** (DST): A multi-objective benchmark in which a submarine on an $11 \times 10$ grid can reach one of ten treasures, worth 1 to 124, that lie deeper and further away, with reward (treasure, $-1$ per step). Its Pareto front has ten points, but it is so concave that only the two extremes are supported, so linear scalarization finds only those. It is unrelated to the DeepSea exploration benchmark. → [Ch 15 §6.7](chapters/15-beyond-mdps.md)

**DeepSeek-R1**: A reasoning model trained at scale with RL on verifiable rewards using GRPO. Its precursor DeepSeek-R1-Zero applied GRPO directly to a pretrained base model with rule-based accuracy and format rewards and no supervised fine-tuning. Unlike OpenAI's o1, whose algorithm, reward design and data were not published, it was documented openly. → [Ch 18 §10.3](chapters/18-rl-for-language-models.md); also [Ch 18 §10.2](chapters/18-rl-for-language-models.md)

**Delayed policy updates**: TD3's practice of updating the actor and the target networks less often than the critics (every second critic step), so the actor follows a less noisy critic. → [Ch 12 §4.4](chapters/12-continuous-control-actor-critic.md)

**DER** (data-efficient Rainbow): Rainbow with hyperparameters retuned for the 100k-step regime (more updates per environment step, $n = 20$, more frequent target refreshes), which matched contemporary model-based agents on Atari 100k. → [Ch 09 §12](chapters/09-deep-q-learning.md)

**Detachment and derailment**: Two failure modes of bonus-driven exploration identified by Go-Explore. Detachment: the agent consumes the bonus near one frontier, wanders off and never returns. Derailment: exploration noise on the way back knocks it off course before it reaches the promising state. → [Ch 14 §9](chapters/14-exploration.md)

**Deterministic policy gradient** (DPG): For a deterministic policy $\mu_{\boldsymbol\theta}$, $\nabla_{\boldsymbol\theta}J = \frac{1}{1-\gamma}\mathbb{E}_{S\sim d^\mu}[\nabla_{\boldsymbol\theta}\mu_{\boldsymbol\theta}(S)\nabla_a q_\mu(S,a)\vert_{a=\mu_{\boldsymbol\theta}(S)}]$, the chain rule through the critic. It integrates over states only, so off-policy use needs no importance ratios, and it is the zero-noise limit of the stochastic policy gradient. → [Ch 12 §2.2](chapters/12-continuous-control-actor-critic.md); also [Ch 12 §2.4](chapters/12-continuous-control-actor-critic.md), [Ch 10 §11.3](chapters/10-policy-gradients.md)

**DIAYN** (Diversity Is All You Need): Reward-free skill discovery that maximizes the mutual information between a skill variable and the states it visits (plus policy entropy), using a learned skill discriminator $q_\phi(z \mid s)$ to form the intrinsic reward. → [Ch 14 §12.2](chapters/14-exploration.md)

**Differential return**: In the average-reward setting, the sum of rewards measured relative to the average reward, $\sum_k (R_{t+k+1} - r(\pi))$. Differential value functions and the differential TD error $\delta_t = R_{t+1} - \bar R_t + \hat q(S_{t+1},A_{t+1},\mathbf{w}) - \hat q(S_t,A_t,\mathbf{w})$ are built on it. → [Ch 08 §12.1](chapters/08-function-approximation.md); also [Ch 08 §12.2](chapters/08-function-approximation.md)

**Diffuser**: Offline planning by sampling from a diffusion model of whole trajectory segments, states and actions together. Classifier guidance along the gradient of a learned return model tilts the samples towards high return, start states and goals are imposed by inpainting, and the agent executes the first action of the plan and plans again. Decision Diffuser conditions on the return instead (classifier-free guidance). Both can favour trajectories that were merely lucky. → [Ch 16 §11.3](chapters/16-offline-rl-and-imitation.md)

**Diffusion policy**: A behaviour-cloning policy that generates an action, or a chunk of actions, by iteratively denoising Gaussian noise with a state-conditioned noise predictor $\boldsymbol\epsilon_{\boldsymbol\theta}(a^k,k,s)$, trained to recover the noise added to expert actions (DDPM). The loss approximates maximum likelihood (it is denoising score matching), so the policy can represent multimodal demonstrations that MSE or single-Gaussian BC would average, at the cost of $K$ network evaluations per action. Flow-matching policies instead integrate a learned velocity field from noise to action. → [Ch 16 §2.8](chapters/16-offline-rl-and-imitation.md); also [Ch 16 §11.3](chapters/16-offline-rl-and-imitation.md)

**Diffusion-QL and IDQL**: Two ways to keep a diffusion model as the policy class in offline RL. Diffusion-QL adds to the diffusion BC loss a term that maximizes $Q$ at the sampled actions, back-propagating through the denoising chain (TD3+BC with a diffusion loss). IDQL keeps IQL's critic, samples candidate actions from a diffusion model of the behaviour policy and resamples them with critic-based weights, so the critic is rarely asked about actions the data do not support. → [Ch 16 §11.3](chapters/16-offline-rl-and-imitation.md)

**Direct RL**: In Dyna, improving values or the policy directly from real experience, as opposed to indirect RL, which improves them by planning with a learned model. → [Ch 07 §2](chapters/07-planning-and-learning-tabular.md)

**Discounted state distribution** ($d^\pi$): The normalized discounted visitation $d^\pi(s) = (1-\gamma)\sum_t \gamma^t\Pr\lbrace S_t = s\rbrace$. It weights the policy gradient theorem, the performance difference lemma and the surrogate objective. → [Ch 11 §1.1](chapters/11-trust-regions-and-ppo.md); also [Ch 10 §3.2](chapters/10-policy-gradients.md), [Ch 01 §10.2](chapters/01-the-rl-problem.md)

**Discount factor** ($\gamma$): The weight $\gamma \in [0,1]$ by which each later reward is multiplied per step. It can be justified by mathematics (finite returns), by uncertainty (survival probability) and by preference; $1/(1-\gamma)$ is the effective horizon. $\gamma$ is part of the problem, not just a hyperparameter. → [Ch 01 §4.2](chapters/01-the-rl-problem.md); also [Ch 01 §4.3](chapters/01-the-rl-problem.md), [Ch 10 §12](chapters/10-policy-gradients.md)

**Discounting-aware importance sampling**: Off-policy Monte Carlo that treats discounting as a probability of termination and weights each flat partial return only by the ratios of the actions it depends on, reducing variance when $\gamma < 1$. → [Ch 04 §9.1](chapters/04-monte-carlo.md)

**Distributional Bellman operator**: The operator on return distributions given by $Z^\pi(s,a) \overset{D}{=} R + \gamma Z^\pi(S',A')$. It is a $\gamma$-contraction in the maximal Wasserstein distance, which justifies learning return distributions by bootstrapping. → [Ch 09 §9.1](chapters/09-deep-q-learning.md); also [Ch 09 §9.2](chapters/09-deep-q-learning.md)

**Distributional RL**: Learning the whole distribution of the return $Z(s,a)$ rather than only its mean, as in C51, QR-DQN and IQN. → [Ch 09 §9](chapters/09-deep-q-learning.md)

**Distribution mismatch coefficient** ($D_\infty$): $\lVert d^{\pi_\ast}_\rho/\mu\rVert_\infty$, how much more the optimal policy's discounted state distribution weights some state than the distribution $\mu$ under which the gradient is computed. It enters the gradient-domination inequality and the convergence rates of policy gradient methods. → [Ch 19 §8.2](chapters/19-rl-theory.md)

**Distribution model**: A model that gives the full distribution of next states and rewards, $p(s',r \mid s,a)$; it supports expected updates. Contrast a sample model. → [Ch 07 §1.1](chapters/07-planning-and-learning-tabular.md); also [Ch 04 §1.1](chapters/04-monte-carlo.md)

**Distribution shift**: The mismatch between the state–action distribution a learned policy visits and the one in the data. It is the common enemy of imitation learning and offline RL; there the estimates are guesses and the errors compound. → [Ch 16 §6.3](chapters/16-offline-rl-and-imitation.md); also [Ch 16 §1](chapters/16-offline-rl-and-imitation.md)

**Dithering**: Undirected exploration that injects local randomness (for example $\varepsilon$-greedy, Boltzmann, entropy bonuses, action noise) without regard to what is unknown. It needs time exponential in the horizon on combination locks. → [Ch 14 §2](chapters/14-exploration.md); also [Ch 14 §1.2](chapters/14-exploration.md)

**Domain randomization**: Training on a distribution of simulator parameters, $\max_{\boldsymbol\theta}\mathbb{E}_{\xi\sim P_\xi}[J(\pi_{\boldsymbol\theta};\xi)]$, so that reality looks like one more sample and the policy transfers. It optimizes the average over models: a robust MDP optimizes the worst case, EPOpt a CVaR in between, and unsupervised environment design adapts the distribution of training contexts to the agent. → [Ch 20 §12](chapters/20-deep-rl-in-practice.md); also [Ch 15 §10.4](chapters/15-beyond-mdps.md), [Ch 20 §11.6](chapters/20-deep-rl-in-practice.md)

**Double DQN**: DQN with the target $y = r + \gamma(1-\mathrm{term})\,\hat q(s', \arg\max_{a'}\hat q(s',a',\mathbf{w}), \mathbf{w}^-)$: the online network selects the action and the target network evaluates it, reducing overestimation. → [Ch 09 §4.3](chapters/09-deep-q-learning.md)

**Double Q-learning**: Keeps two independent action-value tables; one selects the maximizing action and the other evaluates it, which removes maximization bias. → [Ch 05 §11.3](chapters/05-temporal-difference.md)

**Doubly robust estimator** (DR): An off-policy evaluation estimator that uses a learned model $\hat Q$ as a control variate inside per-decision importance sampling. It is unbiased whatever the model (if the model is fitted on independent data) and has low variance when the model is good; weighted DR (WDR) self-normalizes the weights. → [Ch 16 §12.4](chapters/16-offline-rl-and-imitation.md)

**DPO** (Direct Preference Optimization): Fits a policy directly to preference pairs with the loss $-\mathbb{E}[\log\sigma(\beta\log\frac{\pi_{\boldsymbol\theta}(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta\log\frac{\pi_{\boldsymbol\theta}(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)})]$, obtained by substituting the closed-form KL-regularized optimal policy into the Bradley–Terry likelihood, where the partition function cancels. → [Ch 18 §5.1](chapters/18-rl-for-language-models.md); also [Ch 18 §5.4](chapters/18-rl-for-language-models.md)

**DQN** (deep Q-network): Semi-gradient Q-learning with a convolutional network, made workable by experience replay, a target network, the Huber loss, frame stacking, reward clipping and an $\varepsilon$ schedule. → [Ch 09 §2](chapters/09-deep-q-learning.md)

**Dreamer** (DreamerV1–V3): Agents that learn a recurrent state-space world model from pixels and train an actor-critic entirely on imagined latent trajectories, using $\lambda$-returns. DreamerV2 added discrete latents and KL balancing; DreamerV3 made one fixed configuration work across domains (symlog, two-hot, percentile return scaling, free bits, unimix). → [Ch 13 §7.4](chapters/13-model-based-rl.md); also [Ch 13 §7.5](chapters/13-model-based-rl.md)

**Dr. GRPO**: "GRPO done right": removes GRPO's per-response length normalization and per-prompt standard-deviation normalization, which bias the objective, so its first-step gradient is the REINFORCE gradient with a group-mean baseline. → [Ch 18 §9.3](chapters/18-rl-for-language-models.md)

**DroQ**: A high update-to-data continuous-control agent that replaces REDQ's large critic ensemble by two critics with dropout and layer normalization, at much lower compute. → [Ch 12 §8](chapters/12-continuous-control-actor-critic.md)

**DRQN** (deep recurrent Q-network): DQN with a recurrent layer, so the agent can integrate observations over time under partial observability. → [Ch 15 §3.2](chapters/15-beyond-mdps.md)

**Dueling network**: A Q-network head that splits $\hat q(s,a) = V(s) + A(s,a) - \frac{1}{\lvert\mathcal A\rvert}\sum_{a'}A(s,a')$; the mean subtraction makes the decomposition identifiable. → [Ch 09 §5](chapters/09-deep-q-learning.md)

**Dutch trace**: The eligibility trace of true online TD($\lambda$); in the tabular case $z_t = \gamma\lambda z_{t-1} + (1 - \alpha\gamma\lambda z_{t-1}(S_t))\mathbf{e}_{S_t}$. It falls out of deriving the online $\lambda$-return algorithm incrementally. → [Ch 06 §9.2](chapters/06-n-step-and-eligibility-traces.md); also [Ch 06 §12.3](chapters/06-n-step-and-eligibility-traces.md)

**Dyna** (Dyna-Q): An architecture that interleaves acting, direct RL from real transitions, model learning, and $n$ planning updates per real step from simulated transitions. Planning and learning are the same update applied to different experience. → [Ch 07 §2.1](chapters/07-planning-and-learning-tabular.md); also [Ch 07 §3](chapters/07-planning-and-learning-tabular.md)

**Dynamical movement primitives** (DMPs): Stable attractor dynamics whose shape is set by a few dozen weights per joint. As the policy class of episodic robot learning (PoWER, PI², REPS) they keep the parameter dimension small, which makes parameter-space perturbation efficient. → [Ch 10 §15.3](chapters/10-policy-gradients.md); also [Ch 12 §9.2](chapters/12-continuous-control-actor-critic.md)

**Dynamic programming** (DP): Computing values and optimal policies with a known model by turning the Bellman equations into update rules and applying them repeatedly (policy evaluation, policy and value iteration). It is the exact version of almost every other method in the course. → [Ch 03 §1.1](chapters/03-dynamic-programming.md)

**Dynamics** ($p(s',r \mid s,a)$): The four-argument function that gives the probability of each next state and reward given a state and action. Under the Markov property it fully specifies a finite MDP; $p(s' \mid s,a)$ and $r(s,a)$ are derived from it. → [Ch 01 §6.1](chapters/01-the-rl-problem.md)

**Dyna-Q+**: Dyna-Q with an exploration bonus in planning, $\tilde r = r + \kappa\sqrt{\tau(s,a)}$ where $\tau$ is the time since $(s,a)$ was last tried, so the agent plans to re-test neglected transitions and discovers when the world has changed. → [Ch 07 §4.1](chapters/07-planning-and-learning-tabular.md)

## E

**E3** (Explicit Explore or Exploit): The first algorithm with polynomial guarantees for general MDPs. In known states it computes an exploitation and an exploration policy in the known-state model and exploits if that is near-optimal, otherwise it explores; in unknown states it tries the least-tried action. → [Ch 14 §3.2](chapters/14-exploration.md)

**Effective horizon**: $1/(1-\gamma)$, the number of steps over which discounted rewards effectively count; the first $1/(1-\gamma)$ rewards carry at least about 63% of the total weight. → [Ch 01 §4.3](chapters/01-the-rl-problem.md)

**Effective sample size** (ESS): Kish's diagnostic $n_{\text{eff}} = (\sum_i\rho_i)^2/\sum_i\rho_i^2$ for a set of importance weights; it collapses when a few weights dominate. → [Ch 00 §2.6](chapters/00-math-toolkit.md)

**EfficientZero**: MuZero adapted to Atari 100k with a self-supervised temporal-consistency loss on latent states, a value-prefix reward head and a model-based off-policy correction of value targets; the first method to exceed human performance on both mean and median Atari 100k scores. → [Ch 13 §10.2](chapters/13-model-based-rl.md)

**Eligibility trace** ($\mathbf{z}_t$): A short-term memory vector that marks recently visited states (or weights) as eligible for learning; with accumulating traces $\mathbf{z}_t = \gamma\lambda\mathbf{z}_{t-1} + \nabla\hat v(S_t,\mathbf{w})$ and each TD error updates $\mathbf{w} \leftarrow \mathbf{w} + \alpha\delta_t\mathbf{z}_t$. Variants are accumulating, replacing and dutch traces. → [Ch 06 §9.1](chapters/06-n-step-and-eligibility-traces.md); also [Ch 06 §9.2](chapters/06-n-step-and-eligibility-traces.md), [Ch 08 §3.5](chapters/08-function-approximation.md)

**Eluder dimension**: Roughly, the length of the longest sequence of inputs each of which can be "surprising" given the earlier ones, for some pair of functions in the class that agree on those earlier inputs. It controls optimistic exploration with general function classes; the Bellman–eluder dimension applies it to Bellman residuals and contains low Bellman rank as a special case (the GOLF algorithm attains regret governed by it). → [Ch 19 §9.3](chapters/19-rl-theory.md)

**Emergent communication**: Learned signalling between agents that share a reward, studied in the Lewis signalling game and in deep MARL (RIAL, DIAL, CommNet). Independent learners can get trapped in partial-pooling equilibria. → [Ch 17 §11](chapters/17-multi-agent-rl.md)

**Emphatic TD** (ETD): Off-policy TD that reweights each update by an emphasis $M_t = \gamma\rho_{t-1}M_{t-1} + i_t$, restoring the positive definiteness that guarantees stability. Its expected update is stable, but the one-step sample version can have very high variance. → [Ch 08 §17](chapters/08-function-approximation.md)

**Empowerment**: An intrinsic objective equal to the channel capacity from an agent's next $n$ actions to the state they lead to, $\max_\omega I(A_{0:n-1}; S_n \mid s)$; it rewards being in states with many reachable futures. → [Ch 14 §10.2](chapters/14-exploration.md)

**Entropy** ($\mathcal H(p)$): $-\sum_x p(x)\log p(x)$, the uncertainty of a distribution; zero for a deterministic distribution and $\log\lvert\mathcal X\rvert$ for the uniform one. → [Ch 00 §5.1](chapters/00-math-toolkit.md)

**Entropy regularization** (entropy bonus): Adding $c_{\mathcal H}\mathcal H(\pi(\cdot \mid s))$ to the policy objective to keep the policy stochastic. It changes the objective, so the coefficient must be small; maximum-entropy RL makes it part of the reward. → [Ch 10 §10](chapters/10-policy-gradients.md); also [Ch 11 §7.6](chapters/11-trust-regions-and-ppo.md)

**Episode**: One run of an episodic task from a start state to a terminal state (or a time limit). Episodic tasks have a final time step $T$; continuing tasks do not. → [Ch 01 §4.1](chapters/01-the-rl-problem.md)

**Epistemic POMDP**: After training on finitely many contexts, an agent does not know which context it faces at test time, even if each one is fully observed, so the test problem is a POMDP whose hidden state includes the context. Its Bayes-optimal policy is in general history-dependent and acts to find out which context it is in; memoryless policies that are greedy with respect to one point estimate are the most fragile. LEEP approximates it with an ensemble of policies. → [Ch 15 §10.3](chapters/15-beyond-mdps.md)

**Epistemic uncertainty**: Uncertainty due to lack of data, which shrinks as data accumulate. A planner should not trust a model there: avoid such regions (pessimism) or visit them deliberately (optimism); ensemble disagreement measures it. → [Ch 13 §3.2](chapters/13-model-based-rl.md); also [Ch 14 §7.4](chapters/14-exploration.md)

**EPOpt**: Robust policy optimization that draws a simulator for each trajectory and trains only on the worst-performing fraction of the trajectories, which approximately maximizes a CVaR of the return over models. It sits between domain randomization (the average over models) and a robust MDP (the worst case). → [Ch 20 §11.6](chapters/20-deep-rl-in-practice.md)

**$\varepsilon$-greedy**: Choose a greedy action with probability $1-\varepsilon$ and a uniformly random action otherwise, so the greedy action has probability $1-\varepsilon+\varepsilon/\lvert\mathcal A(s)\rvert$. It is the simplest dithering exploration. → [Ch 02 §2.2](chapters/02-multi-armed-bandits.md); also [Ch 04 §5.1](chapters/04-monte-carlo.md)

**$\varepsilon$-soft policy**: A policy with $\pi(a \mid s) \ge \varepsilon/\lvert\mathcal A(s)\rvert$ for every $s$ and $a$. On-policy Monte Carlo control converges to the best $\varepsilon$-soft policy, which is not $\pi_\ast$ plus noise. → [Ch 04 §5.1](chapters/04-monte-carlo.md); also [Ch 04 §5.3](chapters/04-monte-carlo.md)

**Ergodic** (Markov chain): A finite chain that is irreducible and aperiodic. It has a unique stationary distribution to which it converges from any start, and time averages along one run converge to stationary averages. → [Ch 00 §1.6](chapters/00-math-toolkit.md); also [Ch 00 §1.5](chapters/00-math-toolkit.md)

**Error-reduction property**: The expected $n$-step return is closer to $v_\pi$ than the current estimate by a factor $\gamma^n$ in the worst case: $\max_s\lvert\mathbb{E}_\pi[G_{t:t+n} \mid S_t = s] - v_\pi(s)\rvert \le \gamma^n\max_s\lvert V(s) - v_\pi(s)\rvert$. → [Ch 06 §2.3](chapters/06-n-step-and-eligibility-traces.md)

**Every-visit MC**: Monte Carlo prediction that averages the returns following every visit to a state in an episode. It is a ratio estimator: biased but consistent. → [Ch 04 §2.1](chapters/04-monte-carlo.md); also [Ch 04 §2.3](chapters/04-monte-carlo.md)

**Evolution strategies** (ES): Black-box policy search with a Gaussian search distribution $\mathcal{N}(\boldsymbol\theta, \sigma^2\mathbf{I})$ over the parameters, following the gradient of the smoothed objective $J_\sigma(\boldsymbol\theta) = \mathbb{E}[J(\boldsymbol\theta + \sigma\boldsymbol\xi)]$, $\nabla J_\sigma = \frac{1}{\sigma}\mathbb{E}[J(\boldsymbol\theta + \sigma\boldsymbol\xi)\,\boldsymbol\xi]$: REINFORCE with one "action", the parameter vector, per episode. It handles non-differentiable policies and delayed rewards but gives every action the same credit; antithetic pairs and fitness shaping are standard. → [Ch 10 §15.2](chapters/10-policy-gradients.md); also [Ch 10 §15.3](chapters/10-policy-gradients.md), [Ch 10 §15.5](chapters/10-policy-gradients.md)

**EXP3**: An adversarial-bandit algorithm: exponential weights over arms, with each arm's loss estimated by importance weighting the observed loss by its selection probability. Its regret against the best fixed arm is at most $\sqrt{2Tk\ln k}$. → [Ch 02 §12.2](chapters/02-multi-armed-bandits.md)

**Expected SARSA**: TD control with target $R_{t+1} + \gamma\sum_a\pi(a \mid S_{t+1})Q(S_{t+1},a)$. Averaging over the next action lowers variance; with a greedy target policy it is Q-learning. → [Ch 05 §9](chapters/05-temporal-difference.md)

**Expected update**: An update that averages over all possible next states and rewards using a distribution model, as DP does; a sample update uses one sampled outcome instead. An expected update has no sampling error but costs time proportional to the branching factor $b$; after $t$ sample updates the RMS error is $\sqrt{(b-1)/(bt)}$ times the spread of the successor values, so with limited computation sampling wins for large $b$. Update width (sample vs expected) and update depth (bootstrap vs full return) are the two main dimensions of RL methods. → [Ch 07 §6](chapters/07-planning-and-learning-tabular.md); also [Ch 07 §13](chapters/07-planning-and-learning-tabular.md)

**Expectile regression**: Fitting the $\tau$-expectile ($\tau$ here is the expectile level, as in the IQL paper, not a temperature or Polyak coefficient) with the asymmetric squared loss $L^\tau_2(u) = \lvert\tau - \mathbb{1}[u<0]\rvert u^2$. As $\tau \to 1$ the expectile of $Q(s,a)$ over dataset actions approaches a maximum over the data's support, which IQL uses as an in-sample $\max$. → [Ch 16 §9.2](chapters/16-offline-rl-and-imitation.md)

**Experience replay**: Storing transitions in a buffer and training on minibatches sampled from it. It decorrelates consecutive samples, reuses data and smooths the training distribution; a replay buffer is also a nonparametric model. → [Ch 09 §2.2](chapters/09-deep-q-learning.md); also [Ch 09 §12](chapters/09-deep-q-learning.md)

**Expert iteration** (ExIt; filtered SFT): Sample several responses per prompt, keep those a verifier accepts, fine-tune on them by maximum likelihood and repeat. The name comes from ExIt, in which a tree-search expert trains an apprentice network, as in AlphaZero. Averaging the kept samples within each prompt is expectation–maximization on $\mathbb{E}_x[\log p_{\boldsymbol\theta}(x)]$, the log-probability of success, which weights hard but solved prompts by $1/p$; plain SFT on all kept samples is REINFORCE with a 0/1 reward and no baseline. STaR, rejection-sampling fine-tuning, RAFT, ReST and ReST$^{EM}$ are variants. → [Ch 18 §9.5](chapters/18-rl-for-language-models.md); also [Ch 13 §9.4](chapters/13-model-based-rl.md)

**Explained variance** (EV): $1 - \operatorname{Var}[G^{\text{target}} - \hat v]/\operatorname{Var}[G^{\text{target}}]$, a diagnostic of how much of the variation in value targets the critic captures. → [Ch 11 §8.5](chapters/11-trust-regions-and-ppo.md); also [Ch 20 §7.4](chapters/20-deep-rl-in-practice.md)

**Exploitability**: How much worse than the game value a strategy does against an opponent who best-responds to it. In two-player zero-sum games, NashConv sums both players' best-response gains, and the exploitability of a profile is usually reported as NashConv/2. → [Ch 17 §3.5](chapters/17-multi-agent-rl.md)

**Exploration–exploitation dilemma**: To find the best actions an agent must try actions it believes are worse (explore), at the cost of the reward it could earn by acting on its current beliefs (exploit). Feedback is evaluative, so it cannot be avoided. → [Ch 02 §1.4](chapters/02-multi-armed-bandits.md); also [Ch 14 §1.1](chapters/14-exploration.md)

**Explore-then-commit** (ETC): Explore every arm a fixed number of times, then play the empirically best arm forever; an A/B test is an instance. Its regret can be logarithmic only if the gap and horizon are known in advance. → [Ch 02 §11.6](chapters/02-multi-armed-bandits.md)

**Exploring starts** (ES): Starting each episode from a randomly chosen state–action pair, every pair having positive probability, so that every pair is visited infinitely often; convenient in simulation, usually impossible in the real world. → [Ch 04 §3.2](chapters/04-monte-carlo.md); also [Ch 04 §4.2](chapters/04-monte-carlo.md)

**Exponential recency-weighted average**: The estimate produced by a constant step size, $Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^n\alpha(1-\alpha)^{n-i}R_i$, which weights recent targets most and tracks nonstationary targets. → [Ch 00 §2.4](chapters/00-math-toolkit.md); also [Ch 02 §4.1](chapters/02-multi-armed-bandits.md)

**Extensive-form game**: A game tree whose nodes are histories, with a player function, chance moves with known probabilities, utilities at terminal histories, and information sets that model hidden information (poker). → [Ch 17 §10.1](chapters/17-multi-agent-rl.md); also [Ch 17 §2.5](chapters/17-multi-agent-rl.md)

**Extrapolation error**: Offline, the Bellman max queries actions the data never contain; their values are whatever the function approximator extrapolates, the max selects the overestimated ones, and bootstrapping copies the errors into actions that are in the data. → [Ch 16 §6.2](chapters/16-offline-rl-and-imitation.md)

## F

**Feature vector** ($\mathbf{x}(s)$): A representation of a state as a vector of numbers; a linear value function is $\hat v(s,\mathbf{w}) = \mathbf{w}^\top\mathbf{x}(s)$. The choice of features carries the domain knowledge: state aggregation, polynomials, the Fourier basis, coarse coding, tile coding and radial basis functions are the classical options. → [Ch 08 §4.1](chapters/08-function-approximation.md); also [Ch 08 §6](chapters/08-function-approximation.md)

**FeUdal Networks** (FuN): A hierarchical agent in which a Manager, running at a slower time scale in a learned latent space, emits goal directions, and a Worker acting every step is rewarded intrinsically for moving the latent state in the commanded direction. → [Ch 15 §5.8](chapters/15-beyond-mdps.md)

**Fictitious play**: Each player best-responds to the empirical average of the opponents' past play. The time-averaged strategies converge to equilibrium in two-player zero-sum and potential games. → [Ch 17 §5.1](chapters/17-multi-agent-rl.md)

**Fictitious self-play** (FSP, NFSP): Fictitious play in extensive-form games with learned approximate best responses; neural fictitious self-play gives each agent a DQN best response to the opponents' average behaviour and a supervised network that learns its own average strategy. → [Ch 17 §9.1](chapters/17-multi-agent-rl.md)

**Finite differences** (central finite differences): Estimating each partial derivative of the return from episodes at $\boldsymbol\theta \pm h\mathbf{e}_i$, $\partial J/\partial\theta_i \approx \frac{\hat J(\boldsymbol\theta + h\mathbf{e}_i) - \hat J(\boldsymbol\theta - h\mathbf{e}_i)}{2h}$; inside stochastic approximation this is the Kiefer–Wolfowitz scheme. One gradient costs $2d$ episodes, and running both members of each pair on the same environment seed (common random numbers) cancels most of the return noise. → [Ch 10 §15.1](chapters/10-policy-gradients.md)

**Finite-state controller** (FSC): A policy with finite memory for a POMDP: a set of nodes, an action rule $\pi(a \mid n)$ and a memory update $n' = \delta(n,a,o)$. Memoryless policies, frame windows and some belief-optimal policies are all FSCs. → [Ch 15 §2.7](chapters/15-beyond-mdps.md)

**First-visit MC**: Monte Carlo prediction that averages the returns following only the first visit to a state in each episode. It is unbiased, with variance $\sigma^2/n$. → [Ch 04 §2.1](chapters/04-monte-carlo.md); also [Ch 04 §2.3](chapters/04-monte-carlo.md)

**Fisher information matrix** ($\mathbf F$): $\mathbf{F} = \mathbb{E}[\boldsymbol\psi\boldsymbol\psi^\top]$ with $\boldsymbol\psi = \nabla_{\boldsymbol\theta}\log\pi_{\boldsymbol\theta}(a \mid s)$ the score (a local use of $\boldsymbol\psi$ in Chapters 10–11, where NOTATION.md reserves it for model parameters). It is the local metric of the KL divergence, $D_{\mathrm{KL}}(\pi_{\boldsymbol\theta}\Vert\pi_{\boldsymbol\theta+\boldsymbol\Delta}) \approx \frac12\boldsymbol\Delta^\top\mathbf{F}\boldsymbol\Delta$. → [Ch 11 §4.2](chapters/11-trust-regions-and-ppo.md); also [Ch 11 §4.1](chapters/11-trust-regions-and-ppo.md)

**Fisher-vector product**: Computing $\mathbf{F}\mathbf{v} = \nabla_{\boldsymbol\theta}[(\nabla_{\boldsymbol\theta}\bar D_{\mathrm{KL}})^\top\mathbf{v}]$ by double back-propagation, without ever forming $\mathbf F$; it is what makes TRPO's conjugate gradient feasible. → [Ch 11 §6.4](chapters/11-trust-regions-and-ppo.md)

**Fitness shaping** (centred ranks): Replacing the returns of an evolution-strategies batch by their centred ranks, evenly spaced values in $[-\tfrac12, \tfrac12]$, before forming the update. The step becomes invariant to any increasing transformation of the returns, a single outlier episode cannot dominate it, and the ranks act as a baseline; the price is that it follows a rank-based utility instead of the exact smoothed gradient. → [Ch 10 §15.2](chapters/10-policy-gradients.md)

**Fitted Q evaluation** (FQE): Off-policy evaluation by fitted Q-iteration with the max replaced by an expectation under the target policy; low variance but biased under misspecification. → [Ch 16 §12.3](chapters/16-offline-rl-and-imitation.md)

**Fitted Q-iteration** (FQI; neural fitted Q-iteration, NFQ): Value iteration on action values with each backup replaced by a regression onto targets $r_i + \gamma(1 - \text{term}_i)\max_{a'}Q_k(s'_i,a')$ computed from a fixed batch; NFQ retrains a neural network on the whole batch at every iteration. It is approximate value iteration: it converges when the regressor is an averager, while least squares can diverge. DQN with a target network behaves approximately like it, one Bellman backup per target refresh. → [Ch 08 §11.4](chapters/08-function-approximation.md); also [Ch 09 §2.3](chapters/09-deep-q-learning.md), [Ch 19 §9.1](chapters/19-rl-theory.md)

**Forward and reverse KL**: Minimizing forward KL $D_{\mathrm{KL}}(p\Vert q)$ over $q$ is mode covering (it must put mass wherever $p$ has mass; for a Gaussian $q$ it matches moments). Minimizing reverse KL $D_{\mathrm{KL}}(q\Vert p)$ is mode seeking. → [Ch 00 §5.4](chapters/00-math-toolkit.md)

**Forward–backward equivalence**: Offline, accumulating TD($\lambda$) makes exactly the same total updates as the offline $\lambda$-return algorithm; online they differ by $O(\alpha^2)$, and true online TD($\lambda$) removes the difference. → [Ch 06 §10](chapters/06-n-step-and-eligibility-traces.md)

**Forward-backward representation** (FB): Learns from reward-free data a backward embedding $B(s')$ and, for every task vector $\mathbf{z}$, a forward embedding $F(s,a,\mathbf{z})$ such that the successor measure of the policy $\pi_{\mathbf z}(s) = \arg\max_a F(s,a,\mathbf{z})^\top\mathbf{z}$ is approximately $F(s,a,\mathbf{z})^\top B(s')\,\nu(\mathrm{d}s')$. Given a reward, it computes $\mathbf{z}_r = \mathbb{E}_{s\sim\nu}[B(s)\,r(s)]$ and acts with $\pi_{\mathbf{z}_r}$, which is optimal if the factorization is exact: zero-shot RL. It generalizes successor features by learning both the features and a continuum of policies. (Unrelated to the forward–backward equivalence of TD($\lambda$).) → [Ch 15 §6.6](chapters/15-beyond-mdps.md)

**Forward view**: The view of multi-step methods in which each state is updated towards a target that looks ahead in time, such as the $\lambda$-return; it is not directly implementable online because the target depends on the future. → [Ch 06 §8](chapters/06-n-step-and-eligibility-traces.md)

**Fourier basis** (Fourier cosine basis): Linear features $x_i(\mathbf{s}) = \cos(\pi\,\mathbf{s}^\top\mathbf{c}^i)$ with $\mathbf{c}^i \in \lbrace 0,\dots,n\rbrace^k$ for states scaled to $[0,1]^k$; a strong default for low-dimensional smooth problems. → [Ch 08 §6.3](chapters/08-function-approximation.md)

**Frame stacking**: Using the last $k$ observations as the agent's input so that short-range missing information (such as velocity) becomes available; DQN stacks four frames. A naive longer window can be worse. → [Ch 15 §3.1](chapters/15-beyond-mdps.md); also [Ch 09 §2.7](chapters/09-deep-q-learning.md)

**Function approximation**: Representing a value function or policy by a parameterized function with far fewer parameters than states. Its point is generalization; its price is interference and the loss of tabular guarantees. Besides parametric (linear or neural) approximators, memory-based and kernel methods store examples and interpolate between them. → [Ch 08 §1](chapters/08-function-approximation.md); also [Ch 08 §10](chapters/08-function-approximation.md)

## G

**GAE** (generalized advantage estimation): The advantage estimator $\hat A_t = \sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l} = G^\lambda_t - V(S_t)$, an exponentially weighted average of $n$-step advantage estimators. $\lambda$ trades critic-induced bias ($\lambda = 0$, TD) against return variance ($\lambda = 1$, Monte Carlo). → [Ch 10 §8.2](chapters/10-policy-gradients.md); also [Ch 10 §8.4](chapters/10-policy-gradients.md)

**GAIL** (generative adversarial imitation learning): Imitation by adversarial training: a discriminator separates expert from policy state–action pairs, and the policy is trained by RL to fool it. At the optimum it minimizes the Jensen–Shannon divergence between the expert's and the policy's occupancy measures. → [Ch 16 §5.2](chapters/16-offline-rl-and-imitation.md)

**Gambler's problem**: An episodic MDP in which a gambler stakes integer amounts on a biased coin and wins on reaching 100; with $\gamma = 1$, $v_\ast(s)$ is the maximal probability of winning. A classic value-iteration example whose optimal policies are far from unique. → [Ch 03 §6.6](chapters/03-dynamic-programming.md)

**Gap** ($\Delta_a$): In a bandit, $\Delta_a \doteq v_\ast - q_\ast(a)$, how much worse arm $a$ is than the best arm. Regret is $\sum_a\Delta_a\mathbb{E}[N_T(a)]$. → [Ch 02 §1.2](chapters/02-multi-armed-bandits.md); also [Ch 02 §11.2](chapters/02-multi-armed-bandits.md)

**Gaussian policy**: A continuous-action policy $\pi(a \mid s) = \mathcal{N}(\mu_{\boldsymbol\theta}(s), \sigma^2)$, whose score $(a-\mu)/\sigma^2$ is of order $1/\sigma$, so baseline errors and return noise are amplified as $\sigma$ shrinks; its zero-noise limit leads to the deterministic policy gradient. → [Ch 10 §2.2](chapters/10-policy-gradients.md); also [Ch 10 §11.1](chapters/10-policy-gradients.md)

**Generalist policy**: A large policy trained mainly by behaviour cloning on demonstrations pooled across many tasks, scenes and sometimes robots, such as Gato, RT-1, RT-2, Octo, OpenVLA and $\pi_0$ (Open X-Embodiment pooled data from 22 robot types). As with a pretrained language model, BC is the first stage and RL fine-tuning, when used, the second; compounding errors and multimodal demonstrations still apply. → [Ch 16 §2.8](chapters/16-offline-rl-and-imitation.md)

**Generalization**: The effect of an update at one state on the values of other states through shared parameters. It is the point of function approximation; its downside is interference. → [Ch 08 §1](chapters/08-function-approximation.md); also [Ch 08 §8](chapters/08-function-approximation.md)

**Generalization gap**: $\Delta(\hat\pi) = J_{C_{\text{train}}}(\hat\pi) - J(\hat\pi)$, a policy's average return on its $N$ training contexts minus its expected return over the whole context distribution: the RL counterpart of the train/test gap. Deep RL agents overfit surprisingly large sets of procedurally generated levels (CoinRun, Procgen); more training levels, data augmentation, regularization and separate policy and value networks narrow the gap. → [Ch 15 §10.1](chapters/15-beyond-mdps.md); also [Ch 15 §10.2](chapters/15-beyond-mdps.md)

**Generalized policy improvement** (GPI, for transfer): Given a library of policies with known action values on a task, act by $\pi_{\text{GPI}}(s) \in \arg\max_a\max_i q^{\pi_i}(s,a)$. The result is at least as good as every policy in the library; with successor features it gives zero-shot transfer and builds the convex coverage set of a multi-objective problem from few policies. → [Ch 15 §6.3](chapters/15-beyond-mdps.md); also [Ch 15 §6.7](chapters/15-beyond-mdps.md)

**Generalized policy iteration** (GPI): The interplay of policy evaluation and policy improvement, at any granularity, that drives value and policy towards mutual consistency and optimality. It is the template for almost every control method in the course. → [Ch 03 §9](chapters/03-dynamic-programming.md); also [Ch 01 §14](chapters/01-the-rl-problem.md)

**Generative model**: Access to a simulator that returns a sampled next state and reward for any queried $(s,a)$; the setting of the cleanest sample-complexity results. It is one of the three access models of RL theory, with online interaction and offline data. → [Ch 19 §1.1](chapters/19-rl-theory.md); also [Ch 19 §5](chapters/19-rl-theory.md)

**Gibbs variational principle**: $\mathbb{E}_p[f - \alpha\log p] = \alpha\log Z - \alpha D_{\mathrm{KL}}(p\Vert e^{f/\alpha}/Z)$, so the maximizer of expected value plus $\alpha$ times entropy is the Boltzmann distribution $\propto e^{f/\alpha}$. It gives soft policy improvement and the closed-form RLHF policy; with a reference distribution in place of the uniform one it is the template $q^\ast \propto p_0 e^{f/\eta}$ of KL-regularized policy search. → [Ch 12 §5.4](chapters/12-continuous-control-actor-critic.md); also [Ch 18 §3.2](chapters/18-rl-for-language-models.md), [Ch 12 §9.1](chapters/12-continuous-control-actor-critic.md)

**Gittins index**: For discounted, infinite-horizon bandits with independent arms, the Bayes-optimal policy plays the arm with the largest index computed from that arm's own posterior alone. → [Ch 02 §7.1](chapters/02-multi-armed-bandits.md)

**GLIE** (greedy in the limit with infinite exploration): A condition on a sequence of behaviour policies: every state–action pair is tried infinitely often, and the policy converges to a greedy one. With Robbins–Monro step sizes, SARSA converges to $q_\ast$ under GLIE. → [Ch 05 §7.3](chapters/05-temporal-difference.md)

**Goal-conditioned RL**: RL with a family of rewards $r(s,a,s',g)$ indexed by a goal $g$, usually learned with a universal value function $Q(s,a,g)$. → [Ch 15 §4.1](chapters/15-beyond-mdps.md); also [Ch 15 §4.2](chapters/15-beyond-mdps.md)

**Goal-directed and habitual control**: In animal learning, a goal-directed action is chosen for the current value of its outcome, so it adapts at once when the outcome is devalued; a habit, formed by extended training, runs on cached values and does not (outcome devaluation tells the two apart). They correspond to model-based and model-free control, and one proposal is that the brain arbitrates between them according to the uncertainty of their value estimates. The two-step task separates them in humans. → [Ch 05 §14.3](chapters/05-temporal-difference.md)

**Goal misgeneralization**: Generalizing capabilities while pursuing the wrong goal out of distribution, because the intended goal and a proxy were indistinguishable in training: CoinRun agents trained with the coin always at the right end of the level learned "go right" and competently ran past a relocated coin. The remedy is training distributions diverse enough that only the intended goal explains success, and held-out evaluation that checks the goal, not just the return. → [Ch 15 §10.5](chapters/15-beyond-mdps.md); also [Ch 20 §2.1](chapters/20-deep-rl-in-practice.md)

**Go-Explore**: An exploration algorithm that keeps an archive of cells (coarse state representations) and how to reach them, first returns to a promising cell (by restoring the simulator state, replaying actions or a goal-conditioned policy) and then explores from it, avoiding detachment and derailment; a robustification phase then makes the trajectories robust to noise. Its archive of cells echoes quality-diversity search (MAP-Elites). → [Ch 14 §9](chapters/14-exploration.md); also [Ch 10 §15.5](chapters/10-policy-gradients.md)

**Goodhart's law**: "When a measure becomes a target, it ceases to be a good measure" (in Strathern's phrasing). Every reward is a measure of what we want, and optimization turns it into a target; it explains reward hacking and reward-model over-optimization. Regressional Goodhart (selecting the top of a noisy proxy also selects the noise) and extremal Goodhart (optimization leaves the region where proxy and truth agree) are two of its mechanisms. → [Ch 20 §2.1](chapters/20-deep-rl-in-practice.md); also [Ch 18 §8.1](chapters/18-rl-for-language-models.md)

**Gradient bandit algorithm**: A bandit method that learns action preferences $H_t(a)$ and acts by their softmax, updating $H_{t+1}(a) = H_t(a) + \alpha(R_t - \bar R_t)(\mathbb{1}[a = A_t] - \pi_t(a))$; it is stochastic gradient ascent on expected reward with a baseline. → [Ch 02 §8](chapters/02-multi-armed-bandits.md)

**Gradient domination**: An inequality bounding the suboptimality of a policy by the best first-order improvement available at it, scaled by the distribution mismatch coefficient. It is why local policy-gradient search can find global optima despite non-concavity, as in the linear-quadratic regulator, whose cost is non-convex in the gain. → [Ch 19 §8.2](chapters/19-rl-theory.md); also [Ch 03 §11.4](chapters/03-dynamic-programming.md)

**Gradient Monte Carlo**: SGD on the value error using the unbiased Monte Carlo target $G_t$, $\mathbf{w} \leftarrow \mathbf{w} + \alpha[G_t - \hat v(S_t,\mathbf{w})]\nabla\hat v(S_t,\mathbf{w})$; it converges to a local optimum of $\overline{VE}$. → [Ch 08 §3.1](chapters/08-function-approximation.md)

**Gradient-TD methods** (GTD2, TDC): True stochastic-gradient methods on the projected Bellman error that keep a second weight vector and cost $O(d)$ per step. They are stable off-policy but converge to the same off-policy TD fixed point, so they fix stability, not solution quality. → [Ch 08 §16](chapters/08-function-approximation.md)

**Greedy policy**: A policy that, in every state, picks an action maximizing the current value estimate (one-step lookahead on $V$, or $\arg\max_a Q(s,a)$). Acting greedily on $q_\ast$ is optimal and needs no model. → [Ch 01 §12.4](chapters/01-the-rl-problem.md); also [Ch 03 §4.2](chapters/03-dynamic-programming.md)

**GRPO** (Group Relative Policy Optimization): A critic-free policy optimizer for language models: sample a group of responses per prompt, standardize their rewards within the group as advantages, and maximize a PPO-style clipped objective with a $k_3$ KL penalty to the reference policy in the loss. → [Ch 18 §9.2](chapters/18-rl-for-language-models.md); also [Ch 18 §9.1](chapters/18-rl-for-language-models.md)

**Gumbel AlphaZero / Gumbel MuZero**: Search variants that sample root actions without replacement by the Gumbel-top-$m$ trick and allocate simulations by Sequential Halving, guaranteeing policy improvement even with very few simulations; the policy target uses completed Q-values. → [Ch 13 §10.3](chapters/13-model-based-rl.md)

**Gymnasium**: The standard Python API for RL environments: `reset` returns an observation and info, and `step` returns `obs, reward, terminated, truncated, info`. The terminated/truncated distinction decides whether to bootstrap. → [Ch 00 §8.1](chapters/00-math-toolkit.md); also [Ch 00 §8.2](chapters/00-math-toolkit.md)

## H

**Hamilton–Jacobi–Bellman equation** (HJB equation): The continuous-time Bellman optimality equation, $\eta V(s) = \max_a[r(s,a) + \nabla V(s)^\top f(s,a)]$ for dynamics $\dot s = f(s,a)$, reward rate $r$ and discount rate $\eta$; it is the limit of the discrete equation with $\gamma = e^{-\eta\Delta t}$ as $\Delta t \to 0$. Fine time steps make the effective horizon grow like $1/\Delta t$ and the action values collapse onto one another ($q - v = O(\Delta t)$). Value functions are often not differentiable, and the equation then holds for viscosity solutions; Pontryagin's maximum principle is its counterpart along one trajectory. → [Ch 03 §11.5](chapters/03-dynamic-programming.md)

**Hashing** (for counts and tiles): Mapping states or tiles pseudo-randomly into a fixed-size table. In tile coding it bounds memory; in count-based exploration, counting hash codes of states (for example SimHash, the signs of random projections) gives counts in large spaces. → [Ch 14 §6.3](chapters/14-exploration.md); also [Ch 08 §6.5](chapters/08-function-approximation.md)

**Hedge** (exponential weights): The full-information no-regret algorithm that plays each action with probability proportional to $\exp(\eta\times\text{its cumulative payoff})$; its regret is $O(\sqrt{T\ln k})$. EXP3 is its bandit version. → [Ch 17 §5.2](chapters/17-multi-agent-rl.md); also [Ch 02 §12.2](chapters/02-multi-armed-bandits.md)

**HER** (Hindsight Experience Replay): For goal-conditioned tasks, stores each transition again with goals the trajectory actually achieved, recomputing the reward, so failed episodes become successes for other goals. It requires an off-policy learner and one-step (or corrected) targets. → [Ch 15 §4.3](chapters/15-beyond-mdps.md); also [Ch 15 §4.4](chapters/15-beyond-mdps.md)

**Heuristic search**: Decision-time planning that expands a search tree from the current state, backs up values from a heuristic evaluation at the leaves, and picks the best root action. Deeper search shrinks the effect of evaluation errors by a factor $\gamma$ per level. → [Ch 07 §9.1](chapters/07-planning-and-learning-tabular.md)

**Hewer's algorithm**: Policy iteration for the linear-quadratic regulator: evaluate the linear policy $a = -K_ks$ by solving the linear Lyapunov equation $P_k = C_s + K_k^\top C_aK_k + \gamma(F - GK_k)^\top P_k(F - GK_k)$, then take the greedy gain at $P_k$. Started from a stabilizing gain, it is Newton's method on the Riccati equation and converges quadratically: 5–6 evaluations on the chapter's random systems, against 13 to 1,078 sweeps of value iteration. → [Ch 03 §11.4](chapters/03-dynamic-programming.md); also [Ch 03 §5](chapters/03-dynamic-programming.md)

**Hierarchical RL**: RL with temporally extended actions organized in levels, such as options, feudal managers and workers, or MAXQ task hierarchies. → [Ch 15 §5](chapters/15-beyond-mdps.md); also [Ch 15 §5.1](chapters/15-beyond-mdps.md)

**HIRO**: A two-level hierarchical agent in which the high level outputs desired state changes as goals every $c$ steps and a goal-conditioned low level reaches them; both are trained off-policy, with a relabelling correction for the non-stationarity of the lower level. → [Ch 15 §5.8](chapters/15-beyond-mdps.md)

**History** ($H_t$): The sequence of states, actions and rewards up to the current time, $(S_0, A_0, R_1, \dots, S_t)$, or of observations in a POMDP. A state is Markov if it summarizes everything in the history that matters for the future. → [Ch 01 §5](chapters/01-the-rl-problem.md); also [Ch 15 §2.2](chapters/15-beyond-mdps.md)

**Hoeffding's inequality**: For i.i.d. variables in $[0,1]$, $\Pr\lbrace\bar X_n - \mu \ge u\rbrace \le e^{-2nu^2}$. It gives the UCB1 bonus and, with a union bound and the simulation lemma, simple sample-complexity proofs. → [Ch 02 §6.2](chapters/02-multi-armed-bandits.md); also [Ch 19 §5.1](chapters/19-rl-theory.md)

**Huber loss**: $\ell_\kappa(\delta) = \frac12\delta^2$ for $\lvert\delta\rvert \le \kappa$ and $\kappa(\lvert\delta\rvert - \frac12\kappa)$ otherwise; its gradient is the TD error clipped to $[-\kappa, \kappa]$ ("error clipping" in DQN). → [Ch 09 §2.4](chapters/09-deep-q-learning.md); also [Ch 00 §7.2](chapters/00-math-toolkit.md)

**Human-normalized score** (HNS): $(\text{agent} - \text{random})/(\text{human} - \text{random})$ on each Atari game, the usual unit for aggregating results across games. → [Ch 09 §13](chapters/09-deep-q-learning.md)

**Hyperparameter search**: Tuning with a stated budget: random search (better than grid search for the same budget), successive halving and Hyperband (stop poor configurations early), Bayesian optimization and population-based training; tuning and evaluation should use different seeds. → [Ch 20 §5.2](chapters/20-deep-rl-in-practice.md)

## I

**ICM** (Intrinsic Curiosity Module): Prediction-error curiosity in a learned feature space: an inverse model (predict the action from consecutive features) shapes the features, and the error of a forward model in that space is the intrinsic reward. → [Ch 14 §7.1](chapters/14-exploration.md)

**IGM** (Individual-Global-Max): The condition that each agent's greedy action on its own utility $Q_i$ together form a greedy joint action of the team value $Q_{tot}$, so decentralised greedy execution is jointly greedy. VDN and QMIX satisfy it by construction. → [Ch 17 §7.2](chapters/17-multi-agent-rl.md)

**iLQR** (iterative linear-quadratic regulator): Trajectory optimization for known nonlinear deterministic dynamics. Around a nominal trajectory, linearize the dynamics and expand the reward to second order, solve the resulting LQ problem with a Riccati-like backward pass, roll the improved controls through the true dynamics with a line search, and repeat. It returns a locally optimal trajectory with a time-varying feedback law valid near it, not a global policy, and is usually re-solved at every step inside model-predictive control. → [Ch 03 §11.5](chapters/03-dynamic-programming.md); also [Ch 13 §4.1](chapters/13-model-based-rl.md)

**Imitation learning**: Learning a policy from demonstrations instead of a reward: behaviour cloning, interactive methods such as DAgger, inverse RL and adversarial imitation. → [Ch 16 §1](chapters/16-offline-rl-and-imitation.md); also [Ch 16 §2](chapters/16-offline-rl-and-imitation.md)

**IMPALA**: A distributed actor-critic that decouples acting from learning: many actors generate trajectories with slightly stale policies and a central learner updates from them, correcting the resulting policy lag with V-trace. → [Ch 10 §14](chapters/10-policy-gradients.md); also [Ch 06 §14.4](chapters/06-n-step-and-eligibility-traces.md)

**Implicit Q-learning** (IQL): Offline RL that never queries actions outside the data. It fits $V$ as an upper expectile of $Q(s,a)$ over dataset actions, fits $Q \leftarrow r + \gamma V(s')$, and extracts a policy by advantage-weighted regression. (Not to be confused with independent Q-learning, also abbreviated IQL.) → [Ch 16 §9](chapters/16-offline-rl-and-imitation.md)

**Implicit reward**: In DPO, $\hat r_{\boldsymbol\theta}(x,y) = \beta\log\frac{\pi_{\boldsymbol\theta}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}$, the reward under which the current policy would be the KL-regularized optimum; DPO is reward-model training on this reward. → [Ch 18 §5.1](chapters/18-rl-for-language-models.md)

**Importance sampling** (IS): Estimating an expectation under $\pi$ from samples drawn under $b$ by reweighting with $\rho = \pi/b$, using $\mathbb{E}_\pi[f] = \mathbb{E}_b[\rho f]$. Ordinary IS averages the weighted values and is unbiased; weighted IS normalizes by the sum of weights and is biased but consistent and far less variable. → [Ch 00 §2.6](chapters/00-math-toolkit.md); also [Ch 04 §6.3](chapters/04-monte-carlo.md), [Ch 16 §12.2](chapters/16-offline-rl-and-imitation.md)

**Importance-sampling ratio** ($\rho_{t:h}$): $\rho_{t:h} \doteq \prod_{k=t}^{h}\pi(A_k \mid S_k)/b(A_k \mid S_k)$. The dynamics cancel, so it can be computed without a model; its variance grows exponentially with the horizon. → [Ch 04 §6.2](chapters/04-monte-carlo.md); also [Ch 06 §4.1](chapters/06-n-step-and-eligibility-traces.md)

**In-context RL**: A sequence model that improves its behaviour by conditioning on more of its own experience, with frozen weights, as in RL² and Algorithm Distillation. → [Ch 15 §8](chapters/15-beyond-mdps.md)

**Incremental update** (the universal update): $Q \leftarrow Q + \alpha(\text{target} - Q)$. With $\alpha_n = 1/n$ it computes the sample average exactly; with a constant $\alpha$ it is an exponential recency-weighted average. Almost every learning rule in the course has this form. → [Ch 00 §2.3](chapters/00-math-toolkit.md); also [Ch 02 §3](chapters/02-multi-armed-bandits.md)

**Independent Q-learning** (IQL): Each agent runs its own Q-learner on its own observations and actions, treating the other agents as part of the environment. It is simple and often strong, but the environment it sees is nonstationary and it can fall into shadowed equilibria. → [Ch 17 §6.5](chapters/17-multi-agent-rl.md)

**Information gain**: An intrinsic reward equal to the information a transition provides about the unknown model parameters, the expected KL divergence between the posterior after and before observing it; VIME implements it with a Bayesian neural network. It rewards learning rather than surprise. Information-directed sampling trades expected regret against information gained about the optimal action. → [Ch 14 §10.1](chapters/14-exploration.md)

**Information set**: In an extensive-form game, a set of histories that a player cannot tell apart when it must act (for example, all deals consistent with its own cards); a behavioural strategy $\sigma_i(a \mid I)$ acts on information sets, not histories. The course assumes perfect recall: players never forget their own past actions or observations. → [Ch 17 §10.1](chapters/17-multi-agent-rl.md)

**In-place updates** (Gauss–Seidel): Overwriting $V(s)$ as soon as it is recomputed, so later states in the same sweep use new values; it also converges and is usually faster than the two-array (Jacobi) version. → [Ch 03 §3.2](chapters/03-dynamic-programming.md)

**Intra-option learning**: Learning about every option consistent with each primitive step taken, using the target $r + \gamma[(1-\beta_\omega(s'))Q(s',\omega) + \beta_\omega(s')\max_{\omega'}Q(s',\omega')]$; it learns about options it never executes. → [Ch 15 §5.4](chapters/15-beyond-mdps.md)

**Intrinsic reward** ($r^i$): A reward the agent generates for itself, such as a count bonus, a prediction error or a novelty signal, added to the extrinsic reward $r^e$ to drive exploration. → [Ch 14 §6.1](chapters/14-exploration.md); also [Ch 14 §7](chapters/14-exploration.md)

**Inverse reinforcement learning** (IRL): Inferring a reward function that explains observed behaviour. The problem is ill-posed (constants, scaling, shaping and degenerate rewards all explain the same behaviour), so every method adds a selection principle such as margins, feature matching or maximum entropy. → [Ch 16 §3](chapters/16-offline-rl-and-imitation.md); also [Ch 16 §3.2](chapters/16-offline-rl-and-imitation.md)

**IPO** (Identity Preference Optimization): A direct preference loss that regresses the margin of the log-ratios $h_{\boldsymbol\theta}(x,y) = \log\frac{\pi_{\boldsymbol\theta}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}$ towards a finite target, $(h_{\boldsymbol\theta}(x,y_w) - h_{\boldsymbol\theta}(x,y_l) - \frac{1}{2\tau})^2$, so that deterministic preferences do not drive the margin to infinity as in DPO. → [Ch 18 §6](chapters/18-rl-for-language-models.md)

**IQM** (interquartile mean): Pool all run-by-task scores, discard the bottom and top 25%, and average the rest. It is robust to outliers like the median but averages half the data like the mean, so it usually has much smaller variance than the median; it is the recommended aggregate for deep-RL results. The optimality gap, the average shortfall below a target score, complements it. → [Ch 20 §6.2](chapters/20-deep-rl-in-practice.md); also [Ch 09 §13](chapters/09-deep-q-learning.md)

**IQN** (implicit quantile networks): Distributional RL that takes the quantile level $\tau \sim U(0,1)$ as an input (via a cosine embedding) and outputs the corresponding return quantile, so it learns the whole quantile function. → [Ch 09 §9.5](chapters/09-deep-q-learning.md)

## J

**Jensen's inequality**: For convex $f$, $f(\mathbb{E}[X]) \le \mathbb{E}[f(X)]$. It proves $D_{\mathrm{KL}} \ge 0$ and explains maximization bias, $\mathbb{E}[\max_a Q(a)] \ge \max_a\mathbb{E}[Q(a)]$. → [Ch 00 §5.3](chapters/00-math-toolkit.md); also [Ch 05 §11.1](chapters/05-temporal-difference.md)

## K

**KL divergence** ($D_{\mathrm{KL}}(p\Vert q)$): $\sum_x p(x)\log\frac{p(x)}{q(x)}$; non-negative, zero only when $p = q$, and asymmetric. It measures policy change in TRPO and PPO and the distance from the reference model in RLHF. → [Ch 00 §5.3](chapters/00-math-toolkit.md); also [Ch 00 §5.4](chapters/00-math-toolkit.md)

**KL-regularized objective**: $J_\beta = \mathbb{E}_x[\mathbb{E}_{y\sim\pi}[r(x,y)] - \beta D_{\mathrm{KL}}(\pi(\cdot \mid x)\Vert\pi_{\mathrm{ref}}(\cdot \mid x))]$, the objective of RLHF. Its unique optimum is $\pi^\ast_\beta \propto \pi_{\mathrm{ref}}e^{r/\beta}$ with value $\beta\log Z_\beta$, which implies a support constraint and a reward–KL frontier. The same reference-times-exponentiated-score form underlies REPS, MPO and control as inference. → [Ch 18 §1.3](chapters/18-rl-for-language-models.md); also [Ch 18 §3.2](chapters/18-rl-for-language-models.md), [Ch 12 §9.1](chapters/12-continuous-control-actor-critic.md)

**KL-UCB**: A bandit index policy that replaces UCB1's Hoeffding radius by a KL confidence ball suggested by the Chernoff–KL bound. It is asymptotically optimal, attaining the Lai–Robbins constant. → [Ch 02 §11.4](chapters/02-multi-armed-bandits.md)

**KTO** (Kahneman–Tversky Optimization): A direct alignment loss that needs only a thumbs-up or thumbs-down per response, with a prospect-theory value function that judges each response's implicit reward relative to a reference point. → [Ch 18 §6](chapters/18-rl-for-language-models.md)

**Kuhn poker**: A three-card poker game, the "hello world" of imperfect-information games, with 6 information sets per player; Kuhn solved it by hand, and it is small enough that the exploitability of any strategy can be computed exactly with a best response. → [Ch 17 §10.2](chapters/17-multi-agent-rl.md); also [Ch 17 §10.5](chapters/17-multi-agent-rl.md)

## L

**Lagrangian methods**: Solving a constrained MDP by maximizing $J_R(\pi) - \sum_i\lambda_i(J_{C_i}(\pi) - d_i)$ over the policy while raising each multiplier $\lambda_i \ge 0$ when its constraint is violated; the algorithm finds the price of each constraint. → [Ch 20 §11.2](chapters/20-deep-rl-in-practice.md)

**Lai–Robbins lower bound**: No consistent bandit algorithm can have asymptotic regret below $\sum_{a:\Delta_a>0}\Delta_a\ln T/\mathrm{kl}(q_\ast(a), v_\ast)$; logarithmic regret is the best possible on every instance. → [Ch 02 §11.4](chapters/02-multi-armed-bandits.md)

**$\lambda$-return** ($G^\lambda_t$): The geometric average of $n$-step returns, $G^\lambda_t = (1-\lambda)\sum_{n\ge1}\lambda^{n-1}G_{t:t+n}$, which satisfies $G^\lambda_t = R_{t+1} + \gamma[(1-\lambda)V(S_{t+1}) + \lambda G^\lambda_{t+1}]$. $\lambda = 0$ gives TD(0) and $\lambda = 1$ Monte Carlo. The offline $\lambda$-return algorithm updates every state of an episode towards it at the episode's end. → [Ch 06 §8.1](chapters/06-n-step-and-eligibility-traces.md); also [Ch 06 §8.2](chapters/06-n-step-and-eligibility-traces.md), [Ch 06 §8.3](chapters/06-n-step-and-eligibility-traces.md)

**Latent world model**: A model that learns an encoder from observations to a compact latent state and predicts dynamics, rewards and values in that latent space instead of in pixel space, as in World Models, PlaNet, Dreamer and TD-MPC. → [Ch 13 §7.1](chapters/13-model-based-rl.md); also [Ch 13 §7](chapters/13-model-based-rl.md)

**Law of total variance**: $\operatorname{Var}X = \mathbb{E}[\operatorname{Var}(X \mid Y)] + \operatorname{Var}(\mathbb{E}[X \mid Y])$: averaging out randomness never increases variance. Applied along an episode it bounds the total variance of the return and sharpens sample-complexity bounds. → [Ch 00 §1.3](chapters/00-math-toolkit.md); also [Ch 19 §5.3](chapters/19-rl-theory.md)

**League training**: AlphaStar's population of main agents, main exploiters (which attack the current main agents) and league exploiters (which find strategies no one handles), with opponents chosen by prioritized fictitious self-play (PFSP). OpenAI Five (Dota 2) used a simpler recipe: PPO self-play at very large scale, partly against past versions. → [Ch 17 §9.4](chapters/17-multi-agent-rl.md); also [Ch 17 §9.5](chapters/17-multi-agent-rl.md)

**Libratus, DeepStack and Pluribus**: Superhuman poker systems that scaled CFR with abstraction into a blueprint strategy, safe subgame solving, depth-limited search with learned counterfactual values, and self-improvement. Pluribus reached superhuman play in six-player no-limit hold'em, where the two-player zero-sum guarantees no longer apply. → [Ch 17 §10.6](chapters/17-multi-agent-rl.md)

**Likelihood displacement**: A DPO failure mode in which the log-probability of the chosen responses falls during training along with that of the rejected ones, moving mass to responses in neither set. → [Ch 18 §5.4](chapters/18-rl-for-language-models.md)

**Linear function approximation**: $\hat v(s,\mathbf{w}) = \mathbf{w}^\top\mathbf{x}(s)$. Its gradient is the feature vector, and on-policy linear semi-gradient TD converges to a unique TD fixed point. → [Ch 08 §4.1](chapters/08-function-approximation.md); also [Ch 08 §5](chapters/08-function-approximation.md)

**Linear MDP**: An MDP whose transitions and rewards are linear in a known $d$-dimensional feature map, $P_h(s' \mid s,a) = \langle\boldsymbol\phi(s,a), \boldsymbol\mu_h(s')\rangle$; it is learnable with regret $\tilde O(\sqrt{d^3H^3T})$ by LSVI-UCB regardless of the number of states. → [Ch 19 §9.2](chapters/19-rl-theory.md)

**Linear programming formulation** (of an MDP): $v_\ast$ is the smallest $v$ with $v \ge \mathcal{T}^\ast v$ (the primal LP); the dual's variables are discounted state–action occupancy measures, and its vertices are deterministic policies. → [Ch 03 §10](chapters/03-dynamic-programming.md); also [Ch 03 §10.2](chapters/03-dynamic-programming.md)

**LinUCB**: A contextual-bandit algorithm that fits a ridge regression per arm and adds an exploration bonus $\alpha\lVert\mathbf{x}_t\rVert_{\mathbf{A}_a^{-1}}$, choosing the arm with the largest upper confidence bound. OFUL chooses the width so that the confidence ellipsoids hold uniformly over time. → [Ch 02 §13.3](chapters/02-multi-armed-bandits.md)

**Loss of plasticity**: The gradual loss of a network's ability to fit new targets after long training on a nonstationary stream; in deep RL it appears as the primacy bias (overfitting the earliest data). Periodic resets, reinitializing dormant units, shrink-and-perturb and normalization layers help. → [Ch 20 §4](chapters/20-deep-rl-in-practice.md); also [Ch 15 §9](chapters/15-beyond-mdps.md), [Ch 12 §8](chapters/12-continuous-control-actor-critic.md)

**LQG** (linear-quadratic-Gaussian control): The LQR problem observed through noisy linear measurements $O_t = MS_t + N_t$ with Gaussian noise, a POMDP whose belief stays Gaussian. The Kalman filter, the linear-Gaussian Bayes filter, computes the belief mean $\hat s_t$, and the optimal policy is $a_t = -K_t\hat s_t$ with the LQR gains. This separation principle (estimate as if there were no control, control as if the estimate were the state) fails for general nonlinear problems. → [Ch 03 §11.4](chapters/03-dynamic-programming.md); also [Ch 15 §2.3](chapters/15-beyond-mdps.md)

**LQR** (linear-quadratic regulator): Linear dynamics $S_{t+1} = FS_t + GA_t + W_t$ with reward $-(S_t^\top C_sS_t + A_t^\top C_aA_t)$. Backward induction keeps every optimal value function quadratic, $V^\ast_t(s) = -s^\top P_ts - c_t$, through the Riccati recursion, and every optimal decision rule linear, $a = -K_ts$. Additive noise lowers the value by a constant without changing the gains (certainty equivalence). LQR is the exception to the curse of dimensionality and the standard testbed of RL theory for continuous control. → [Ch 03 §11.4](chapters/03-dynamic-programming.md); also [Ch 12 §2.6](chapters/12-continuous-control-actor-critic.md), [Ch 10 §15.6](chapters/10-policy-gradients.md)

**LSPI** (least-squares policy iteration): Policy iteration on a fixed batch: alternate LSTDQ evaluation of the current policy with greedy improvement $\pi_{m+1}(s) = \arg\max_a\mathbf{x}(s,a)^\top\mathbf{w}_{\pi_m}$, reusing the same data at every iteration. It has no general convergence guarantee, since the policies can oscillate, but if every evaluation is within $\epsilon$ of $q_{\pi_m}$ in the sup norm, then $\limsup_m\lVert q_{\pi_m} - q_\ast\rVert_\infty \le 2\gamma\epsilon/(1-\gamma)^2$. When it converges it solves the fixed-point equation of least-squares fitted Q-iteration with the same ridge. → [Ch 08 §11.4](chapters/08-function-approximation.md); also [Ch 08 §9](chapters/08-function-approximation.md)

**LSTD** (least-squares TD): Computes the linear TD fixed point $\mathbf{w}_{TD} = \hat{\mathbf{A}}^{-1}\hat{\mathbf{b}}$ directly from data, with incremental Sherman–Morrison updates. It needs no step size but costs $O(d^2)$ per step; LSPI (least-squares policy iteration), built on its action-value version LSTDQ, is its control version. → [Ch 08 §9](chapters/08-function-approximation.md); also [Ch 08 §11.4](chapters/08-function-approximation.md)

**LSTDQ**: LSTD for the action values of any deterministic policy $\pi$ from a batch collected by any behaviour: $\mathbf{w}_\pi = \hat{\mathbf{A}}^{-1}\hat{\mathbf{b}}$ with $\hat{\mathbf{A}} = \sum_i\mathbf{x}(s_i,a_i)(\mathbf{x}(s_i,a_i) - \gamma\mathbf{x}(s'_i,\pi(s'_i)))^\top$ and $\hat{\mathbf{b}} = \sum_i r_i\mathbf{x}(s_i,a_i)$, with $\mathbf{x}(s'_i,\cdot) = \mathbf{0}$ only at terminal states. The next action is chosen by $\pi$, not read from the log, so no importance sampling is needed; it solves the sample version of the projected Bellman equation for $q_\pi$. → [Ch 08 §11.4](chapters/08-function-approximation.md)

**LSVI-UCB**: Least-squares value iteration with UCB bonuses for linear MDPs; optimistic exploration whose regret depends on the feature dimension rather than the number of states. Its analysis replaces the tabular pigeonhole step by the elliptical potential lemma. → [Ch 19 §9.2](chapters/19-rl-theory.md)

## M

**MADDPG** (multi-agent DDPG): DDPG for $N$ agents with deterministic decentralised actors and, for each agent, its own centralised critic on the joint state and actions, so it handles cooperative, competitive and mixed games. → [Ch 17 §8.2](chapters/17-multi-agent-rl.md)

**Majority voting** (self-consistency, maj@$k$): Sample $k$ answers and return the most common one; maj@$k$ is the resulting accuracy. It needs no verifier but fails when the model's errors are systematic. → [Ch 18 §12](chapters/18-rl-for-language-models.md)

**MAML** (model-agnostic meta-learning): Meta-learns an initialization from which one or a few policy-gradient steps on a new task give good performance; the meta-gradient differentiates through the inner adaptation step. → [Ch 15 §7.4](chapters/15-beyond-mdps.md)

**MAPPO** (multi-agent PPO): PPO with parameter sharing, decentralised actors and a centralised value function computing a shared team advantage; with careful tuning it, and its independent-critic variant IPPO, are very strong baselines on cooperative benchmarks. → [Ch 17 §8.4](chapters/17-multi-agent-rl.md)

**Markov chain**: A sequence of states in which the next state depends only on the current one, through a row-stochastic transition matrix $\mathbf{P}$; a stationary policy turns an MDP into one, with $\mathbf{P}_\pi(s,s') = \sum_a\pi(a \mid s)p(s' \mid s,a)$. Together with the expected rewards $\mathbf{r}_\pi$ it forms a Markov reward process (MRP). → [Ch 00 §1.4](chapters/00-math-toolkit.md); also [Ch 01 §7](chapters/01-the-rl-problem.md)

**Markov decision process** (MDP): A model of sequential decision making given by states, actions, rewards and dynamics $p(s',r \mid s,a)$ (plus an initial distribution and a discount). A finite MDP has finite sets and is fully specified by its four-argument dynamics. → [Ch 01 §6.1](chapters/01-the-rl-problem.md)

**Markov game** (stochastic game): The multi-agent generalization of an MDP: agents choose actions simultaneously, the state transition depends on the joint action, and each agent has its own reward. → [Ch 17 §2.3](chapters/17-multi-agent-rl.md)

**Markov property**: A state signal is Markov if the next state and reward depend on the history only through the current state and action. It is a property of the state representation, not of the world. → [Ch 01 §5](chapters/01-the-rl-problem.md)

**Maximization bias**: The upward bias of using the maximum of noisy estimates as an estimate of the maximum, $\mathbb{E}[\max_a Q(a)] \ge \max_a q(a)$. It makes Q-learning and DQN overestimate; double estimators remove it. → [Ch 05 §11.1](chapters/05-temporal-difference.md); also [Ch 09 §4.1](chapters/09-deep-q-learning.md)

**Maximum-entropy IRL** (MaxEnt IRL): IRL that models the demonstrator as Boltzmann-rational, $P_{\boldsymbol\omega}(\zeta) \propto e^{\boldsymbol\omega^\top\boldsymbol\phi(\zeta)}$, and fits the reward by maximum likelihood; the gradient is expert feature counts minus model feature counts, computed by soft value iteration (maximum causal entropy for stochastic dynamics). → [Ch 16 §4](chapters/16-offline-rl-and-imitation.md)

**Maximum-entropy RL** (MaxEnt RL): RL with an entropy bonus in every reward, $J_\alpha(\pi) = \mathbb{E}_\pi[\sum_t\gamma^t(R_{t+1} + \alpha\mathcal{H}(\pi(\cdot \mid S_t)))]$, where $\alpha$ is the temperature. Its optimal policy is a Boltzmann policy over soft action values; SAC is built on it. Its objective is the evidence lower bound of control as inference. → [Ch 12 §5.1](chapters/12-continuous-control-actor-critic.md); also [Ch 12 §5.7](chapters/12-continuous-control-actor-critic.md)

**MAXQ**: A hierarchical decomposition that writes the value of a task as the value of a child subtask plus a completion function for finishing the task afterwards. It allows state abstraction and reuse, but its learning converges to a recursively optimal policy, which can be worse than the hierarchically optimal one. → [Ch 15 §5.9](chapters/15-beyond-mdps.md)

**MBIE-EB**: Model-based interval estimation with an exploration bonus: solve the empirical Bellman equation with a bonus $\beta/\sqrt{n(s,a)}$ added to each state–action value, so optimism shrinks gradually with the count. → [Ch 14 §3.3](chapters/14-exploration.md)

**MBPO** (Model-Based Policy Optimization): Trains a model-free learner on short model rollouts branched from real states sampled from the replay buffer, so the model is never trusted for long. → [Ch 13 §6.2](chapters/13-model-based-rl.md)

**MCCFR** (Monte Carlo CFR): CFR that samples part of the game tree each iteration and corrects counterfactual values with importance weights. → [Ch 17 §10.4](chapters/17-multi-agent-rl.md)

**MCTS** (Monte Carlo Tree Search): Decision-time planning that grows a search tree by repeating selection (a tree policy such as UCT), expansion, simulation (a rollout or a value estimate) and backup of the result. With learned priors and values it is the search of AlphaZero and MuZero. Classical improvements include the RAVE (all-moves-as-first) heuristic and progressive widening. → [Ch 07 §11](chapters/07-planning-and-learning-tabular.md); also [Ch 07 §11.1](chapters/07-planning-and-learning-tabular.md), [Ch 13 §9.3](chapters/13-model-based-rl.md)

**MDPO** (mirror descent policy optimization): Reads the exponentiated-advantage NPG update as mirror descent with the KL divergence, and takes several SGD steps on $\mathbb{E}[\rho A] - \frac1\eta\mathrm{KL}$ instead of clipping. → [Ch 11 §9](chapters/11-trust-regions-and-ppo.md)

**Mean squared value error** ($\overline{VE}$): $\overline{VE}(\mathbf{w}) = \sum_s\mu(s)[v_\pi(s) - \hat v(s,\mathbf{w})]^2$, the prediction objective of function approximation, weighted by a state distribution $\mu$ (usually the on-policy distribution). → [Ch 08 §2.1](chapters/08-function-approximation.md)

**Meta-RL**: Learning to learn from a distribution of tasks: optimize the return of a whole adaptation trial on a task drawn from $p(\mathcal M)$. Its ideal is the Bayes-optimal policy; RL² (memory), MAML (gradients) and PEARL/VariBAD (inference) approximate it. → [Ch 15 §7.1](chapters/15-beyond-mdps.md); also [Ch 15 §7.6](chapters/15-beyond-mdps.md)

**Minimax-Q**: Q-learning for zero-sum Markov games that replaces the max by the value of the stage game, $Q(s,a,o) \leftarrow Q(s,a,o) + \alpha[r + \gamma\,\mathrm{val}\,Q(s',\cdot,\cdot) - Q(s,a,o)]$. → [Ch 17 §4.2](chapters/17-multi-agent-rl.md)

**Minimax regret**: The worst case of expected regret over all instances of a problem class. For $k$-armed bandits it is $\Theta(\sqrt{kT})$, which the logarithmic instance-dependent bounds hide; UCB1 attains it up to $\sqrt{\ln T}$ and MOSS exactly. → [Ch 02 §11.7](chapters/02-multi-armed-bandits.md); also [Ch 19 §7.5](chapters/19-rl-theory.md)

**Minimax theorem**: In two-player zero-sum games, $\max_{\mathbf x}\min_{\mathbf y}\mathbf{x}^\top\mathbf{A}\mathbf{y} = \min_{\mathbf y}\max_{\mathbf x}\mathbf{x}^\top\mathbf{A}\mathbf{y}$: the game has a unique value, and equilibrium strategies are interchangeable. It is proved both by LP duality and by no-regret learning. → [Ch 17 §3.2](chapters/17-multi-agent-rl.md); also [Ch 17 §5.5](chapters/17-multi-agent-rl.md)

**Minorize-maximize** (MM): Maximizing a lower bound (minorizer) that touches the objective at the current iterate; each maximization then cannot decrease the objective. It gives TRPO's monotonic-improvement guarantee. → [Ch 11 §3.5](chapters/11-trust-regions-and-ppo.md)

**Mixed strategy**: A probability distribution over a player's actions in a one-shot game (a pure strategy is a single action). Nash equilibria may require mixed strategies. → [Ch 17 §2.1](chapters/17-multi-agent-rl.md)

**Mode averaging**: The failure of behaviour cloning with a unimodal model on multimodal demonstrations. MSE regression learns the mean of the expert's actions (steering straight between "left" and "right"), and a maximum-likelihood Gaussian sits between the modes, wide enough to cover both. Expressive classes avoid it: mixture density networks, discretized (tokenized) actions, implicit (energy-based) policies, and diffusion and flow-matching policies. → [Ch 16 §2.8](chapters/16-offline-rl-and-imitation.md)

**Model-based RL**: Methods that use a model of the environment (a distribution model, a sample model or a learned latent model), given or learned, to plan, generate synthetic experience or build targets (DP, Dyna, MCTS, PETS, Dreamer, MuZero). Model-free methods learn values or policies directly from experience. Chapter 13 classifies model-based methods by what is learned (state-space dynamics, a latent world model, a decoder-free latent model, or a value-equivalent model such as the Predictron, VPN or MuZero) and by how it is used: background planning (Dyna, ME-TRPO, MBPO), decision-time planning, gradients through the model (PILCO, SVG, Dreamer V1) or value expansion. → [Ch 01 §14](chapters/01-the-rl-problem.md); also [Ch 13 §1.3](chapters/13-model-based-rl.md)

**Model bias**: Systematic error of a learned model wherever data are scarce, the function class is inadequate or the state is not Markov. Planning does not average it away, and optimizers seek it out. → [Ch 13 §1.2](chapters/13-model-based-rl.md); also [Ch 13 §5.3](chapters/13-model-based-rl.md)

**Model-predictive control** (MPC, receding-horizon control): At every step, optimize an action sequence over a short horizon with the model, execute only the first action, and re-plan from the next state; re-planning corrects model errors by feedback. With known smooth dynamics, iLQR warm-started from the previous solution is a common inner solver. → [Ch 13 §4.1](chapters/13-model-based-rl.md); also [Ch 03 §11.5](chapters/03-dynamic-programming.md)

**Modified policy iteration** (MPI, truncated policy iteration): Policy iteration with only $m$ evaluation sweeps per improvement; $m = 1$ is value iteration and $m \to \infty$ is policy iteration. → [Ch 03 §7.1](chapters/03-dynamic-programming.md)

**Monte Carlo ES** (exploring starts): Monte Carlo control that alternates evaluation and greedy improvement episode by episode and relies on exploring starts for exploration. → [Ch 04 §4.2](chapters/04-monte-carlo.md)

**Monte Carlo methods**: Learning values and policies from complete sample episodes by averaging returns, with no model and no bootstrapping. They need episodic tasks and update only at episode ends. Off-policy Monte Carlo control learns a greedy target policy from a soft behaviour policy with weighted importance sampling, but only from the tails of episodes. → [Ch 04 §1.2](chapters/04-monte-carlo.md); also [Ch 04 §8](chapters/04-monte-carlo.md), [Ch 00 §2.5](chapters/00-math-toolkit.md)

**MOPO and MOReL**: Model-based offline RL with pessimism. MOPO penalizes the reward of short model rollouts by an uncertainty estimate, $\tilde r = \hat r - \lambda u(s,a)$; MOReL sends the agent to an absorbing, heavily penalized halt state wherever the ensemble disagrees; COMBO applies CQL's penalty to model-generated data. → [Ch 16 §10.2](chapters/16-offline-rl-and-imitation.md)

**Mountain Car**: An underpowered car must rock back and forth to climb out of a valley; the standard test of semi-gradient SARSA with tile coding, which solves it with $\varepsilon = 0$ thanks to optimistic initialization. → [Ch 08 §11.3](chapters/08-function-approximation.md)

**MPO** (maximum a posteriori policy optimization): REPS made step-based and off-policy, with a learned critic. Its E-step computes a non-parametric improved policy $q(a \mid s) \propto \pi_{\text{old}}(a \mid s)\exp(Q(s,a)/\eta)$, with the temperature $\eta$ from the dual of a KL bound; its M-step fits the parametric policy to $q$ by weighted maximum likelihood under a second KL trust region (split into mean and covariance bounds for Gaussian policies). The critic is trained off-policy from replay with Retrace. → [Ch 12 §9.4](chapters/12-continuous-control-actor-critic.md); also [Ch 11 §9](chapters/11-trust-regions-and-ppo.md), [Ch 12 §9.1](chapters/12-continuous-control-actor-critic.md)

**MPPI** (model-predictive path integral control): A relative of CEM that replaces the hard elite cut by exponential weights on the sampled action sequences' returns; TD-MPC plans with an MPPI-style sampler. → [Ch 13 §4.3](chapters/13-model-based-rl.md); also [Ch 13 §7.6](chapters/13-model-based-rl.md), [Ch 12 §9.1](chapters/12-continuous-control-actor-critic.md)

**Multi-armed bandit** ($k$-armed bandit): Repeated choice among $k$ actions with noisy rewards drawn from unknown fixed distributions, where actions do not affect the future; RL with a single state. → [Ch 02 §1.2](chapters/02-multi-armed-bandits.md); also [Ch 02 §1.3](chapters/02-multi-armed-bandits.md)

**Multi-objective RL** (multi-objective MDP): RL with a vector reward $\mathbf{r}(s,a) \in \mathbb{R}^m$, so that policies have vector values and are only partially ordered, by Pareto dominance. The goal is a Pareto front or, for linear utilities, a convex coverage set; for a nonlinear utility it matters whether it applies to the expected return (SER) or to each episode's return (ESR). Outer-loop methods such as Optimistic Linear Support solve scalarized problems at corner weights; inner-loop methods (Pareto Q-learning, envelope Q-learning, successor features with GPI) learn many policies at once. → [Ch 15 §6.7](chapters/15-beyond-mdps.md)

**MuZero**: AlphaZero-style search with a learned model: a representation function, a dynamics function in latent space and a prediction head, trained by unrolling $K$ steps along real actions to match rewards, values and search policies. The model is value-equivalent; EfficientZero, Gumbel MuZero, Sampled MuZero and Stochastic MuZero extend it. → [Ch 13 §10.1](chapters/13-model-based-rl.md); also [Ch 13 §10.4](chapters/13-model-based-rl.md), [Ch 13 §10.5](chapters/13-model-based-rl.md)

**MVE and STEVE** (model-based value expansion): Build critic targets from a few model steps followed by a bootstrapped value, $R_{t+1} + \sum_{j=1}^{H-1}\gamma^j\hat r_{t+j} + \gamma^H\hat q(\hat s_{t+H},\cdot)$; STEVE weights the different horizons by the inverse variance of their ensemble predictions. → [Ch 13 §6.3](chapters/13-model-based-rl.md)

## N

**Nash equilibrium**: A strategy profile in which every agent's strategy is a best response to the others', $u^i(\mathbf x) \ge u^i(a'^i, \mathbf x^{-i})$ for all $i$ and $a'^i$; an $\epsilon$-Nash equilibrium allows gains up to $\epsilon$. One always exists in mixed strategies, but it need not be unique and is PPAD-complete to compute in general. → [Ch 17 §3.1](chapters/17-multi-agent-rl.md)

**Nash-Q**: The general-sum extension of minimax-Q that keeps one Q-table per agent and backs up each agent's payoff in a Nash equilibrium of the stage game; its convergence conditions are very restrictive. Friend-or-foe Q (max over the joint action for friends, minimax for foes) and correlated-Q (a correlated equilibrium of the stage game) are its relatives. → [Ch 17 §4.3](chapters/17-multi-agent-rl.md)

**Natural policy gradient** (NPG; natural gradient): The steepest-ascent direction under a KL budget, $\mathbf{F}^{-1}\nabla J$, where $\mathbf F$ is the Fisher information matrix. It is invariant to reparameterization; for the tabular softmax it is the exponentiated-advantage update $\pi_{k+1} \propto \pi_k\exp(\eta A_{\pi_k}/(1-\gamma))$, which converges at a dimension-free rate. A natural actor-critic estimates it as the weights of a compatible critic; with KL-normalized steps but no line search it is the truncated natural policy gradient (TNPG). → [Ch 11 §5.1](chapters/11-trust-regions-and-ppo.md); also [Ch 11 §5.6](chapters/11-trust-regions-and-ppo.md), [Ch 19 §8.4](chapters/19-rl-theory.md)

**Negamax**: The minimax backup for two-player zero-sum games written from the perspective of the player to move, $v(s) = \max_a(-v(\text{child}(s,a)))$, so every node's value is stored for its own mover. → [Ch 07 §12.1](chapters/07-planning-and-learning-tabular.md)

**NES** (natural evolution strategies): Treat the mean and covariance of the search distribution as parameters $\boldsymbol\phi$, estimate $\nabla_{\boldsymbol\phi}\mathbb{E}_{\tilde{\boldsymbol\theta}\sim p_{\boldsymbol\phi}}[J(\tilde{\boldsymbol\theta})]$ with the score function, and follow the natural gradient $\mathbf{F}_{\boldsymbol\phi}^{-1}\nabla_{\boldsymbol\phi}$. With the covariance frozen at $\sigma^2\mathbf{I}$ it is plain evolution strategies; once the covariance adapts, the natural gradient makes the update independent of how it is parameterized. → [Ch 10 §15.3](chapters/10-policy-gradients.md); also [Ch 11 §5](chapters/11-trust-regions-and-ppo.md)

**Never Give Up** (NGU): An exploration bonus that multiplies an episodic novelty term (a pseudo-count from nearest neighbours in a learned controllable-state embedding, over the current episode only) by a life-long novelty term from RND, and trains a family of policies with different exploration weights and discounts. → [Ch 14 §8](chapters/14-exploration.md); also [Ch 09 §11](chapters/09-deep-q-learning.md)

**Noisy Nets**: Exploration by learned parameter noise: linear layers with weights $\boldsymbol\mu + \boldsymbol\sigma\odot\boldsymbol\varepsilon$ whose noise scales are trained with the other weights, replacing $\varepsilon$-greedy. → [Ch 09 §8](chapters/09-deep-q-learning.md); also [Ch 14 §2](chapters/14-exploration.md)

**Noisy TV problem**: A source of unpredictable observations (a screen of random static) keeps a prediction-error agent's forward-model error high forever, so the agent watches it instead of exploring. Novelty methods such as RND and counts avoid this trap for stochastic transitions among a bounded set of observations. → [Ch 14 §7.2](chapters/14-exploration.md)

**Nonstationarity** (in multi-agent RL): From each learner's point of view, the other agents are part of the environment, and because they are learning, the MDP it faces changes over time. → [Ch 17 §6.1](chapters/17-multi-agent-rl.md); also [Ch 17 §2.3](chapters/17-multi-agent-rl.md)

**No-regret learning**: Online learning whose average regret against every fixed action tends to zero, such as Hedge or regret matching with $O(\sqrt T)$ regret. In self-play the empirical joint play converges to the set of coarse correlated equilibria, and in two-player zero-sum games the time averages converge to Nash. The iterates themselves can cycle (the continuous-time limit of Hedge is the replicator dynamics); optimistic variants such as optimistic Hedge (OMWU) fix this when the equilibrium is unique. → [Ch 17 §5.2](chapters/17-multi-agent-rl.md); also [Ch 17 §5.4](chapters/17-multi-agent-rl.md), [Ch 17 §5.6](chapters/17-multi-agent-rl.md)

**Normal-form game** (matrix game): A one-shot game in which every agent simultaneously chooses an action and receives a payoff $u^i(\mathbf a)$ that depends on the joint action. Games are fully cooperative (common payoff), competitive (zero-sum) or mixed (general-sum). → [Ch 17 §2.1](chapters/17-multi-agent-rl.md)

**Novelty search**: Rewarding behaviour unlike anything in an archive of past behaviours, ignoring the objective entirely. It solves deceptive mazes in which climbing the return leads into a dead end; like MAP-Elites, it searches for diversity rather than return alone. → [Ch 10 §15.5](chapters/10-policy-gradients.md)

**$n$-step return** ($G_{t:t+n}$): $n$ real rewards followed by a bootstrap, $G_{t:t+n} = \sum_{j=0}^{n-1}\gamma^jR_{t+j+1} + \gamma^nV(S_{t+n})$ (or $G_t$ if the episode ends first). $n$-step TD and $n$-step SARSA update towards it; $n = 1$ is one-step TD and $n \ge T-t$ is Monte Carlo. Deep agents use it as the multi-step target (3-step returns in Rainbow). → [Ch 06 §2.1](chapters/06-n-step-and-eligibility-traces.md); also [Ch 06 §3.1](chapters/06-n-step-and-eligibility-traces.md), [Ch 09 §7](chapters/09-deep-q-learning.md)

## O

**Objective mismatch**: A learned model is trained for one-step likelihood, but what matters is the return of the policy it produces; lower model loss does not always mean better control. → [Ch 13 §11](chapters/13-model-based-rl.md)

**Observation and reward normalization**: Feeding the network running-standardized observations (whose statistics are part of the model and must be saved and frozen at evaluation) and dividing rewards, without shifting them, by a running estimate of the standard deviation of the discounted return. PopArt normalizes value targets adaptively. → [Ch 20 §3.4](chapters/20-deep-rl-in-practice.md); also [Ch 11 §8.2](chapters/11-trust-regions-and-ppo.md)

**Observation mask** (loss masking of observations): In multi-turn RL for language models, the indicator $m_t$ that token $t$ was chosen by the policy ($m_t = 1$) rather than written by the environment, such as a tool output or a retrieved page ($m_t = 0$). The policy gradient contains only the policy's own tokens, so losses, KL terms and their normalizers must all use $m_t$. Leaving observations in adds an advantage-weighted gradient of the likelihood of text the policy never chose, which broke training in the chapter's calculator toy. → [Ch 18 §13.1](chapters/18-rl-for-language-models.md); also [Ch 18 §13.5](chapters/18-rl-for-language-models.md)

**Occupancy measure**: The discounted distribution of state–action pairs visited by a policy, $\sum_t\gamma^t\Pr_\pi\lbrace S_t = s, A_t = a\rbrace$ (normalized by $1-\gamma$ in some chapters). The dual LP of an MDP optimizes over occupancy measures, and GAIL matches them. → [Ch 03 §10.2](chapters/03-dynamic-programming.md); also [Ch 16 §5.1](chapters/16-offline-rl-and-imitation.md)

**Offline RL** (batch RL): Learning a policy from a fixed dataset with no further interaction. Naive off-policy algorithms fail through extrapolation error; the cure is some form of pessimism (policy constraints, conservative values, in-sample learning, uncertainty penalties). → [Ch 16 §6.1](chapters/16-offline-rl-and-imitation.md); also [Ch 19 §10](chapters/19-rl-theory.md)

**Off-policy evaluation** (OPE): Estimating the value of a target policy from data logged by a different policy, before deploying it. The main estimators are importance sampling (ordinary, weighted, per-decision), fitted Q evaluation and doubly robust; they trade bias against variance. → [Ch 16 §12](chapters/16-offline-rl-and-imitation.md); also [Ch 16 §12.2](chapters/16-offline-rl-and-imitation.md)

**Off-policy learning**: Learning about a target policy $\pi$ from data generated by a different behaviour policy $b$ (Q-learning, off-policy Monte Carlo, DQN with replay, SAC). It allows data reuse but needs corrections or care for stability. → [Ch 01 §14](chapters/01-the-rl-problem.md); also [Ch 04 §6.1](chapters/04-monte-carlo.md)

**One-step lookahead** ($q_V$): For any value vector $V$, $q_V(s,a) = r(s,a) + \gamma\sum_{s'}p(s' \mid s,a)V(s')$; the building block of Bellman operators and greedy policies. → [Ch 03 §1.3](chapters/03-dynamic-programming.md)

**On-policy distribution** ($\mu$): The fraction of time spent in each state while following $\pi$. Weighting the value error by it makes semi-gradient TD stable, because then the matrix $\mathbf A$ is positive definite. → [Ch 08 §2.2](chapters/08-function-approximation.md); also [Ch 08 §5.2](chapters/08-function-approximation.md)

**On-policy learning**: Learning about the same policy that generates the data (SARSA, REINFORCE, PPO); data must be discarded or used only briefly after each policy change. → [Ch 01 §14](chapters/01-the-rl-problem.md); also [Ch 04 §5](chapters/04-monte-carlo.md)

**OpenAI-ES**: Evolution strategies with antithetic pairs, centred-rank fitness shaping and Adam (plus weight decay), scaled to deep networks and more than a thousand CPU cores by sharing random seeds, so that workers exchange only scalar returns, never gradient vectors. With 1,440 workers it solved the MuJoCo 3D humanoid in about 10 minutes; on Atari, one hour on 720 cores, about the computation of a one-day A3C run, beat A3C on 23 games and lost on 28. → [Ch 10 §15.5](chapters/10-policy-gradients.md); also [Ch 10 §15.2](chapters/10-policy-gradients.md)

**Optimal policy** ($\pi_\ast$) and optimal value functions ($v_\ast$, $q_\ast$): $v_\ast(s) = \sup_\pi v_\pi(s)$ and $q_\ast(s,a) = \sup_\pi q_\pi(s,a)$. Some deterministic stationary policy is optimal in every state at once; $v_\ast$ and $q_\ast$ are unique but optimal policies need not be, and acting greedily on $q_\ast$ is optimal. → [Ch 01 §12.2](chapters/01-the-rl-problem.md); also [Ch 01 §12.4](chapters/01-the-rl-problem.md), [Ch 03 §2.5](chapters/03-dynamic-programming.md)

**Optimism in the face of uncertainty** (OFU): Act greedily with respect to an optimistic (upper-confidence) estimate of value, so that uncertain actions get tried until the data rule them out. In MDPs, planning carries the optimism to distant uncertain states (UCB, R-MAX, UCRL2, UCBVI). → [Ch 02 §6.1](chapters/02-multi-armed-bandits.md); also [Ch 14 §3.1](chapters/14-exploration.md), [Ch 19 §7.2](chapters/19-rl-theory.md)

**Optimistic initial values**: Initializing estimates above any plausible value so that every action looks attractive until tried; exploration that wears off by itself, unsuitable for nonstationary problems. → [Ch 02 §5](chapters/02-multi-armed-bandits.md)

**Option**: A temporally extended action $\omega = (\mathcal{I}_\omega, \pi_\omega, \beta_\omega)$ with an initiation set, an intra-option policy and a termination function giving the probability of stopping in each state. Options turn an MDP into a semi-MDP. By the interruption theorem, switching to a better option whenever the running option's value falls below the value of choosing afresh never makes things worse. → [Ch 15 §5.3](chapters/15-beyond-mdps.md); also [Ch 15 §5.6](chapters/15-beyond-mdps.md)

**Option-critic**: Learns intra-option policies and termination functions end to end by gradient ascent (an intra-option policy gradient and a termination gradient). Its main weakness is degeneracy: options shrink to single steps or one option takes over. → [Ch 15 §5.7](chapters/15-beyond-mdps.md)

**Ordinary importance sampling** (OIS): The unweighted average of importance-weighted returns, $\sum_t\rho_{t:T(t)-1}G_t/\lvert\mathcal{T}(s)\rvert$; unbiased but with possibly huge or infinite variance. → [Ch 04 §6.3](chapters/04-monte-carlo.md); also [Ch 04 §6.6](chapters/04-monte-carlo.md)

**Ornstein–Uhlenbeck noise** (OU noise): Temporally correlated exploration noise, $x_{t+1} = x_t - \kappa x_t\Delta t + \sigma_{\mathrm{OU}}\sqrt{\Delta t}\,\xi_t$, used by DDPG; plain Gaussian action noise usually works as well. → [Ch 12 §3.4](chapters/12-continuous-control-actor-critic.md)

**ORPO** (odds-ratio preference optimization): Folds preference alignment into supervised fine-tuning by adding to the likelihood of the chosen response a term that raises its odds relative to the rejected one; it needs no reference model. → [Ch 18 §6](chapters/18-rl-for-language-models.md)

**Other-play** (OP): A training objective for zero-shot coordination that maximizes the return against a partner relabelled by a random symmetry $\phi$ of the game, $\max_{\pi^1,\pi^2}\mathbb{E}_{\phi\sim\mathcal{U}(\Phi)}[J(\pi^1,\phi(\pi^2))]$, so conventions built on arbitrary labels earn nothing. In the lever game it favours the one lever that every symmetry leaves in place. It fixes the objective, not the optimization (independent gradient learners can still miss that lever), and it needs the symmetries to be known; off-belief learning does not. → [Ch 17 §9.6](chapters/17-multi-agent-rl.md)

**Outcome reward model** (ORM): A reward model or verifier that scores only the final answer of a solution; cheaper and harder to game than a process reward model, but it gives every step the same credit. → [Ch 18 §11](chapters/18-rl-for-language-models.md)

**Overestimation**: Systematic upward bias of learned action values caused by maximizing over noisy estimates and propagated by bootstrapping. In actor-critics the actor actively seeks the overestimated actions; Double DQN and clipped double-Q counter it. → [Ch 09 §4](chapters/09-deep-q-learning.md); also [Ch 12 §4.2](chapters/12-continuous-control-actor-critic.md)

**Over-optimization** (reward-model over-optimization): Optimizing a learned proxy reward eventually lowers the true reward: as KL from the reference grows, the true score rises, peaks and falls (Goodhart's law). The KL penalty acts much like early stopping. → [Ch 18 §8.2](chapters/18-rl-for-language-models.md); also [Ch 18 §8.5](chapters/18-rl-for-language-models.md)

## P

**PAC-MDP**: An exploration algorithm is PAC-MDP if, with probability at least $1-\delta$, the number of time steps on which its policy is not $\varepsilon$-optimal from the current state is polynomial in the problem size, $1/\varepsilon$, $\log(1/\delta)$ and $1/(1-\gamma)$. R-MAX is PAC-MDP; $\varepsilon$-greedy is not. → [Ch 19 §6.1](chapters/19-rl-theory.md); also [Ch 14 §3.2](chapters/14-exploration.md)

**PAIRED** (Protagonist Antagonist Induced Regret Environment Design): Unsupervised environment design that estimates regret with a second, antagonist agent. An adversary proposes levels to maximize the antagonist's return minus the protagonist's, and all three learn. At equilibrium the protagonist is minimax-regret optimal, and the levels grow in complexity as the agents improve. → [Ch 15 §10.4](chapters/15-beyond-mdps.md)

**Parameter sharing**: In multi-agent RL, letting all (homogeneous) agents use one network, usually with the agent index as an input, which speeds learning and scales to many agents. → [Ch 17 §7.1](chapters/17-multi-agent-rl.md); also [Ch 17 §8.4](chapters/17-multi-agent-rl.md)

**Parameter-space noise**: Exploration by perturbing the weights of the policy or Q-network (once per episode, or with learned noise as in Noisy Nets) instead of the actions, giving consistent, state-dependent exploration. Evolution strategies explore only this way. → [Ch 14 §2](chapters/14-exploration.md); also [Ch 14 §11](chapters/14-exploration.md), [Ch 10 §15.2](chapters/10-policy-gradients.md)

**Pareto front** (Pareto dominance): In multi-objective RL, $\pi$ Pareto-dominates $\pi'$ if $\mathbf{V}^\pi \ge \mathbf{V}^{\pi'}$ in every component and $>$ in at least one; the Pareto front is the set of undominated value vectors at the start state. Supported points, on the boundary of the front's convex hull, are optimal for some linear scalarization; unsupported points in its "dents" are optimal for none. → [Ch 15 §6.7](chapters/15-beyond-mdps.md); also [Ch 17 §3.4](chapters/17-multi-agent-rl.md)

**Pareto optimality**: An outcome is Pareto optimal if no other outcome is at least as good for every agent and strictly better for one. Equilibria need not be Pareto optimal (the prisoner's dilemma); the price of anarchy measures the loss. → [Ch 17 §3.4](chapters/17-multi-agent-rl.md)

**pass@$k$**: The probability that at least one of $k$ sampled answers is correct, estimated without bias from $N \ge k$ samples with $c$ correct as $1 - \binom{N-c}{k}/\binom{N}{k}$; it measures a model's reach given a perfect verifier. → [Ch 18 §12](chapters/18-rl-for-language-models.md); also [Ch 18 §10.4](chapters/18-rl-for-language-models.md)

**PEARL and VariBAD**: Inference-based meta-RL: infer a posterior over a latent task variable from the context collected so far and condition the policy on it. PEARL samples the latent per episode (posterior sampling) and trains off-policy with SAC; VariBAD conditions on the full approximate belief, approximating the Bayes-adaptive policy itself. → [Ch 15 §7.5](chapters/15-beyond-mdps.md)

**Peng's Q($\lambda$)**: A trace method that mixes backups along the actions actually taken with max-based bootstraps and does not cut traces; its fixed point is in general not $q_\ast$ because it blends in the behaviour policy's values. → [Ch 06 §13.2](chapters/06-n-step-and-eligibility-traces.md)

**Per-decision importance sampling** (PDIS): Weights each reward only by the ratios of the actions taken before it, $\tilde G_t = \sum_k\gamma^k\rho_{t:t+k}R_{t+k+1}$, reducing variance; it is the form used by off-policy $n$-step and trace methods and by OPE, where its self-normalized version is weighted PDIS (WPDIS). → [Ch 04 §9.2](chapters/04-monte-carlo.md); also [Ch 16 §12.2](chapters/16-offline-rl-and-imitation.md)

**Performance difference lemma** (PDL): $J(\pi') - J(\pi) = \frac{1}{1-\gamma}\mathbb{E}_{s\sim d^{\pi'}, a\sim\pi'}[A_\pi(s,a)]$: the improvement equals the old policy's advantage accumulated along the new policy's trajectories. It underlies TRPO, CPI and policy-gradient theory. → [Ch 11 §2](chapters/11-trust-regions-and-ppo.md); also [Ch 19 §4.3](chapters/19-rl-theory.md)

**Performance profile**: For each score threshold $\tau$, the fraction of runs (averaged over tasks) that score above it; if one algorithm's profile lies above another's everywhere, it stochastically dominates. → [Ch 20 §6.5](chapters/20-deep-rl-in-practice.md)

**Pessimism** (in the face of uncertainty): Acting as if poorly covered state–actions are bad, for example by subtracting an uncertainty bonus. It is the cure for offline RL's distribution shift and needs only coverage of the comparator policy rather than of all policies; pessimistic value iteration with lower-confidence-bound penalties (VI-LCB) is the tabular example. → [Ch 16 §6.3](chapters/16-offline-rl-and-imitation.md); also [Ch 19 §10.3](chapters/19-rl-theory.md), [Ch 20 §11.6](chapters/20-deep-rl-in-practice.md)

**PETS** (probabilistic ensembles with trajectory sampling): Model-based control that plans with CEM-based MPC through an ensemble of probabilistic neural dynamics models, propagating particles by trajectory sampling. → [Ch 13 §5.1](chapters/13-model-based-rl.md)

**PGPE** (parameter-exploring policy gradients): A likelihood-ratio gradient with respect to the mean and per-parameter standard deviations of a Gaussian over policy parameters, with symmetric sampling and a baseline. It reached the evolution-strategies estimator from the RL side, motivated by the variance of per-step action noise. → [Ch 10 §15.3](chapters/10-policy-gradients.md)

**Phasic Policy Gradient** (PPG): Alternates PPO policy phases with auxiliary phases that train the value function harder and distill value features into the policy network, while a KL term keeps the policy itself nearly unchanged. → [Ch 11 §9](chapters/11-trust-regions-and-ppo.md)

**PI²** (policy improvement with path integrals): Episodic policy search that weights sampled parameter perturbations by their exponentiated cost, $w_i \propto \exp(-\text{cost}_i/\kappa)$, an update derived from path-integral stochastic optimal control and applied to dynamical movement primitives. It belongs to the reward-weighted family with RWR and PoWER; MPPI is the action-sequence counterpart of these updates. → [Ch 10 §15.3](chapters/10-policy-gradients.md); also [Ch 12 §9.3](chapters/12-continuous-control-actor-critic.md)

**Plan2Explore**: Uses the disagreement of an ensemble of one-step latent predictors on top of a Dreamer world model as an intrinsic reward and learns, in imagination, to seek out what the model does not yet know, before any task reward is given. → [Ch 13 §7.5](chapters/13-model-based-rl.md); also [Ch 14 §7.4](chapters/14-exploration.md)

**PlaNet**: A latent-dynamics agent that learned a recurrent state-space model from pixels and planned in its latent space with CEM-based MPC; its model is the backbone of Dreamer. → [Ch 13 §7.3](chapters/13-model-based-rl.md)

**Planning**: Any computation that uses a model to produce or improve a policy or value function. Planning updates are learning updates fed with simulated instead of real experience. → [Ch 07 §1.2](chapters/07-planning-and-learning-tabular.md)

**Point-based value iteration** (PBVI): Approximate POMDP planning that backs up alpha vectors only at a finite set of reachable beliefs, keeping their number bounded. → [Ch 15 §2.6](chapters/15-beyond-mdps.md)

**Policy** ($\pi$): A rule for choosing actions: $\pi(a \mid s)$ for a stochastic policy, $\pi(s)$ for a deterministic one. It may be stationary or time-dependent, and Markov (depending only on the current state) or history-dependent. → [Ch 01 §7](chapters/01-the-rl-problem.md)

**Policy evaluation** (prediction): Computing $v_\pi$ or $q_\pi$ for a fixed policy, as opposed to control; with a model, by iterating $V \leftarrow \mathcal{T}^\pi V$ (iterative policy evaluation), which converges geometrically. → [Ch 03 §3.1](chapters/03-dynamic-programming.md); also [Ch 01 §10](chapters/01-the-rl-problem.md)

**Policy gradient theorem**: $\nabla J(\boldsymbol\theta) = \sum_s\eta_\gamma(s)\sum_a q_\pi(s,a)\nabla\pi(a \mid s)$, equivalently $\mathbb{E}[G_0\sum_t\nabla\log\pi(A_t \mid S_t)]$. The derivative of the state distribution never appears, so the gradient can be estimated from experience without a model. → [Ch 10 §4.1](chapters/10-policy-gradients.md); also [Ch 10 §4.2](chapters/10-policy-gradients.md), [Ch 10 §4.3](chapters/10-policy-gradients.md)

**Policy improvement theorem**: If $q_\pi(s,\pi'(s)) \ge v_\pi(s)$ for all $s$ (equivalently $\mathcal{T}^{\pi'}v_\pi \ge v_\pi$), then $v_{\pi'} \ge v_\pi$; acting greedily on a policy's own values never makes things worse, and strictly helps unless the policy is already optimal. → [Ch 03 §4.1](chapters/03-dynamic-programming.md); also [Ch 03 §4.2](chapters/03-dynamic-programming.md)

**Policy iteration** (PI): Alternate complete policy evaluation with greedy policy improvement; it terminates with an optimal policy after finitely many (usually very few) iterations and is Newton's method on the Bellman optimality equation. For the linear-quadratic regulator it is Hewer's algorithm. → [Ch 03 §5](chapters/03-dynamic-programming.md); also [Ch 03 §11.4](chapters/03-dynamic-programming.md)

**Policy lag**: The gap between the policy that generated data (an actor's stale copy) and the policy being updated (the learner's) in distributed agents; IMPALA corrects it with V-trace. → [Ch 10 §14.1](chapters/10-policy-gradients.md); also [Ch 20 §9.2](chapters/20-deep-rl-in-practice.md)

**Polyak averaging** (soft target update): Updating target-network weights slowly towards the online weights, $\bar{\mathbf{w}} \leftarrow \tau\mathbf{w} + (1-\tau)\bar{\mathbf{w}}$, instead of copying them periodically. → [Ch 12 §3.3](chapters/12-continuous-control-actor-critic.md); also [Ch 09 §2.3](chapters/09-deep-q-learning.md)

**POMDP** (partially observable MDP): An MDP in which the agent receives observations drawn from an observation kernel $\mathcal{O}(o \mid s', a)$ instead of the state. The belief is a sufficient statistic, and the problem becomes an MDP over beliefs; exact planning is PSPACE-complete for finite horizons. → [Ch 15 §2.1](chapters/15-beyond-mdps.md); also [Ch 15 §2.6](chapters/15-beyond-mdps.md)

**Population-based training** (PBT): Trains a population in parallel and periodically copies the weights of better members over worse ones while perturbing their hyperparameters; in multi-agent RL the population also provides diverse training partners. → [Ch 17 §9.3](chapters/17-multi-agent-rl.md); also [Ch 20 §5.2](chapters/20-deep-rl-in-practice.md)

**Potential-based shaping**: Adding $F(s,a,s') = \gamma\Phi(s') - \Phi(s)$, with $\Phi(\text{terminal}) = 0$, to the reward. It shifts every policy's values by $-\Phi(s)$, so it cannot change which policy is optimal (Ng, Harada & Russell's theorem); it is equivalent to initializing Q with $\Phi$. → [Ch 20 §2.3](chapters/20-deep-rl-in-practice.md)

**PoWER** (policy learning by weighting exploration with the returns): Carries the expectation–maximization view of reward-weighted regression over to episodic search over policy parameters. It learned ball-in-a-cup on a real robot arm. → [Ch 10 §15.3](chapters/10-policy-gradients.md); also [Ch 12 §9.3](chapters/12-continuous-control-actor-critic.md)

**PPO** (Proximal Policy Optimization): Collects a batch on-policy, computes GAE advantages, and takes several epochs of minibatch SGD on the clipped surrogate objective plus a value loss and an entropy bonus. Simple, first-order and robust, it became the default policy optimizer in deep RL and the original optimizer of RLHF. Its implementation details (normalization, initialization, learning-rate annealing, gradient-norm clipping, truncation handling) matter as much as the clip. → [Ch 11 §7](chapters/11-trust-regions-and-ppo.md); also [Ch 11 §8](chapters/11-trust-regions-and-ppo.md), [Ch 11 §10](chapters/11-trust-regions-and-ppo.md)

**Principle of optimality**: Bellman's principle that an optimal policy's remaining decisions must be optimal from whatever state results from the first decision; it is why one policy that acts greedily on $v_\ast$ is optimal from every start state. → [Ch 01 §12.3](chapters/01-the-rl-problem.md)

**Prioritized experience replay** (PER): Samples transitions with probability $P(i) \propto p_i^\alpha$, with priority $p_i = \lvert\delta_i\rvert + \epsilon_p$, using a sum tree for $O(\log N)$ sampling, and corrects the resulting shift of the fixed point with importance weights $(NP(i))^{-\beta}$. → [Ch 09 §6](chapters/09-deep-q-learning.md); also [Ch 09 §6.2](chapters/09-deep-q-learning.md), [Ch 09 §6.3](chapters/09-deep-q-learning.md)

**Prioritized level replay** (PLR): A curriculum over training levels that replays the levels where the agent can still learn most, scored by the average magnitude of the GAE advantages in the level's last episode (the L1 value loss), mixed with a staleness term so that old scores are refreshed. Robust PLR scores by the positive value loss, as an estimate of regret, and updates the policy only on replayed levels, which makes it a minimax-regret method; ACCEL adds small edits of high-regret levels. → [Ch 15 §10.4](chapters/15-beyond-mdps.md)

**Prioritized sweeping**: Planning that keeps a priority queue of state–action pairs keyed by the size of their expected value change and works backwards through the predecessors of states whose values have just changed (backward focusing). → [Ch 07 §5](chapters/07-planning-and-learning-tabular.md)

**Probabilistic ensemble**: An ensemble of $B$ neural dynamics models, each with a Gaussian output head trained by negative log-likelihood. The mean of the members' variances models aleatoric noise and the variance of their means measures epistemic uncertainty. → [Ch 13 §3.1](chapters/13-model-based-rl.md); also [Ch 13 §3.2](chapters/13-model-based-rl.md)

**Probability of improvement**: The probability that a random run of algorithm $X$ beats a random run of $Y$ on a random task (a Mann–Whitney statistic averaged over tasks). It is invariant to monotone transformations of the scores, but it is a pairwise comparison, not a ranking. → [Ch 20 §6.6](chapters/20-deep-rl-in-practice.md)

**Probe environments**: Tiny MDPs with known answers, each exercising one capability of an agent (value learning, discounting, policy updates, truncation handling), used to localize bugs. → [Ch 20 §7.2](chapters/20-deep-rl-in-practice.md)

**Process reward model** (PRM): A reward model that scores each intermediate step of a solution, giving step-level credit; hard to build and open to hacking. → [Ch 18 §11](chapters/18-rl-for-language-models.md)

**Projected Bellman equation**: $\mathbf{X}\mathbf{w} = \Pi\mathcal{T}^\pi\mathbf{X}\mathbf{w}$, whose solution is the linear TD fixed point. The projected Bellman error $\overline{PBE}(\mathbf{w}) = \lVert\Pi\bar\delta_{\mathbf w}\rVert^2_\mu$ is learnable from data and is zero there; gradient-TD methods minimize it. → [Ch 08 §4.3](chapters/08-function-approximation.md); also [Ch 08 §15.1](chapters/08-function-approximation.md)

**Proper policy**: In an episodic task with $\gamma = 1$, a policy that reaches termination with probability 1 from every state. Policy evaluation converges for proper policies, and stochastic shortest-path (SSP) theory gives conditions under which value iteration and policy iteration still work. → [Ch 03 §2.8](chapters/03-dynamic-programming.md); also [Ch 01 §10.3](chapters/01-the-rl-problem.md)

**Pseudo-count**: A count $\hat N = \rho(1-\rho')/(\rho'-\rho)$ derived from a density model's probability $\rho$ of an observation before and after one more update on it (the recoding probability $\rho'$); it extends count-based bonuses to large spaces. → [Ch 14 §6.2](chapters/14-exploration.md)

**PSRL** (posterior sampling for RL): Thompson sampling for MDPs: at the start of each episode, sample one MDP from the posterior, compute its optimal policy, and follow it for the whole episode. Its Bayesian regret is $\tilde O(HS\sqrt{AT})$. → [Ch 14 §4.1](chapters/14-exploration.md)

**PSRO** (Policy-Space Response Oracles): Maintains a population of policies per player, estimates the meta-game whose actions are those policies, solves it with a meta-solver, and adds a best response to the resulting mixture; with a Nash meta-solver it is the double oracle algorithm. → [Ch 17 §9.2](chapters/17-multi-agent-rl.md)

**PUCT**: The tree policy of AlphaZero-style search, $\arg\max_a[Q(s,a) + c\,P(s,a)\sqrt{N(s)}/(1+N(s,a))]$, which adds to UCB-style exploration a prior $P(s,a)$ from a policy network. → [Ch 13 §9.3](chapters/13-model-based-rl.md); also [Ch 07 §11.4](chapters/07-planning-and-learning-tabular.md)

## Q

**Q-learning**: Off-policy TD control with update $Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[R_{t+1} + \gamma\max_a Q(S_{t+1},a) - Q(S_t,A_t)]$. It learns $q_\ast$ from any sufficiently exploratory behaviour, with no importance sampling because its target contains no action sampled from the behaviour policy. → [Ch 05 §8.1](chapters/05-temporal-difference.md); also [Ch 05 §8.2](chapters/05-temporal-difference.md), [Ch 05 §8.4](chapters/05-temporal-difference.md)

**QMIX**: Value decomposition for cooperative MARL that mixes per-agent utilities with a monotonic network, $Q_{tot} = f_{mix}(Q_1, \dots, Q_N; s)$ with $\partial Q_{tot}/\partial Q_i \ge 0$, whose weights come from hypernetworks of the global state. It satisfies IGM but cannot represent payoffs where one agent's best action depends on another's. → [Ch 17 §7.4](chapters/17-multi-agent-rl.md)

**QR-DQN** (quantile-regression DQN): Distributional RL that learns $N$ quantile locations of the return at the midpoints $\hat\tau_i = (2i-1)/(2N)$, trained with the quantile Huber loss $\rho^\kappa_\tau(u) = \lvert\tau - \mathbb{1}[u<0]\rvert\ell_\kappa(u)/\kappa$; with a Huber threshold that is large relative to the returns it learns expectiles instead of quantiles. → [Ch 09 §9.4](chapters/09-deep-q-learning.md)

**$Q(\sigma)$**: A multi-step off-policy method that chooses, per step, between sampling the next action ($\sigma = 1$, as in SARSA with importance sampling) and taking the expectation over it ($\sigma = 0$, as in Tree Backup). It is the member of the per-decision family with trace coefficient $c = \sigma\rho + (1-\sigma)\pi(A \mid S)$. → [Ch 06 §7.1](chapters/06-n-step-and-eligibility-traces.md); also [Ch 06 §6.2](chapters/06-n-step-and-eligibility-traces.md)

**QTRAN, QPLEX and Weighted QMIX**: Value-decomposition methods that go beyond QMIX's monotonicity: QTRAN imposes IGM through soft penalties on an unrestricted joint $Q$, QPLEX writes IGM in a dueling advantage form, and Weighted QMIX keeps the monotonic mixer but weights its errors towards the best joint actions. → [Ch 17 §7.5](chapters/17-multi-agent-rl.md)

**Quality-diversity**: Population-based search for a collection of solutions that are both good and diverse, rather than for return alone. MAP-Elites keeps an archive with the best solution found in each cell of a grid over behaviour descriptors; the archive-of-cells idea reappears in Go-Explore. → [Ch 10 §15.5](chapters/10-policy-gradients.md); also [Ch 14 §9](chapters/14-exploration.md)

## R

**R2D2** (Recurrent Replay Distributed DQN): A distributed DQN with an LSTM, trained on replayed sequences that start from the stored recurrent state with a burn-in prefix; it also replaces reward clipping by the invertible value rescaling $h(x) = \mathrm{sign}(x)(\sqrt{\lvert x\rvert + 1} - 1) + \epsilon x$. → [Ch 09 §11](chapters/09-deep-q-learning.md); also [Ch 15 §3.3](chapters/15-beyond-mdps.md)

**Rainbow**: DQN combined with six extensions: double Q-learning, prioritized replay, dueling heads, multi-step returns, distributional (C51) learning and Noisy Nets. Its ablations found prioritized replay and multi-step returns mattered most. → [Ch 09 §10](chapters/09-deep-q-learning.md)

**Randomized least-squares value iteration** (RLSVI): Posterior sampling on value functions instead of models: treat each Bellman backup as a Bayesian linear regression and sample the value function from its posterior, then act greedily for the episode. → [Ch 14 §4.2](chapters/14-exploration.md)

**Randomized prior functions**: Ensemble members $Q_j = f_{\boldsymbol\theta_j} + \beta_p\,p_j$ that each add a fixed, untrained random network $p_j$ to a trainable one, so the ensemble keeps prior uncertainty where there are no data; with per-episode commitment to one member they give deep exploration. → [Ch 14 §4.3](chapters/14-exploration.md)

**Random Network Distillation** (RND): An exploration bonus equal to the error of a predictor network trained to match a fixed, randomly initialized target network on visited observations. Because the target is a deterministic function, the error measures novelty and does not stay high for stochastic transitions. → [Ch 14 §7.3](chapters/14-exploration.md)

**Random shooting**: The simplest MPC planner: sample many random action sequences, evaluate each with the model, and execute the first action of the best. → [Ch 13 §4.2](chapters/13-model-based-rl.md)

**RARL** (robust adversarial reinforcement learning): Adversarial training on the dynamics: a second agent learns to apply disturbance forces to the robot, and the two are trained alternately on the resulting zero-sum game. The adversary's force budget plays the role of the radius of a robust MDP's uncertainty set. → [Ch 20 §11.6](chapters/20-deep-rl-in-practice.md)

**Realizability**: The assumption that the target function lies in the function class (for example $q_\ast \in \mathcal F$). It is much weaker than completeness, and linear realizability of $q_\ast$ alone does not make a problem learnable with polynomially many samples. → [Ch 19 §9.1](chapters/19-rl-theory.md); also [Ch 19 §9.4](chapters/19-rl-theory.md)

**Real-time dynamic programming** (RTDP): Value-iteration backups applied only to states visited along greedy trajectories from the start states. With an admissible (optimistic) initial value function it converges to an optimal policy on the relevant states without sweeping the rest. → [Ch 07 §8](chapters/07-planning-and-learning-tabular.md)

**Rectangularity** ($(s,a)$-rectangular uncertainty sets): A set of transition models is $(s,a)$-rectangular if it is the product of per-pair sets $\mathcal{U}_{s,a}$, so the adversary's choice at one state–action pair does not constrain its choice anywhere else. It makes the robust Bellman operator a $\gamma$-contraction with a deterministic stationary optimal policy. With $s$-rectangular sets robust value iteration still works but optimal policies may randomize, and for general coupled sets computing an optimal robust policy is strongly NP-hard. Rectangular sets are usually conservative outer approximations. → [Ch 20 §11.6](chapters/20-deep-rl-in-practice.md)

**Recurrent state-space model** (RSSM): The latent dynamics model of PlaNet and Dreamer, whose latent state has a deterministic recurrent part $h_t$ and a stochastic part $z_t$, trained with an evidence lower bound (ELBO: reconstruction plus a KL between posterior and prior). PlaNet planned in it with CEM. → [Ch 13 §7.3](chapters/13-model-based-rl.md)

**REDQ** (randomized ensembled double Q-learning): An ensemble of ten critics whose targets take the minimum over a random subset of two, allowing a high update-to-data ratio with controlled overestimation. → [Ch 12 §8](chapters/12-continuous-control-actor-critic.md)

**Regret** ($\mathrm{Reg}(T)$): The reward lost relative to always acting optimally. In bandits, $\mathbb{E}[\mathrm{Reg}(T)] = \sum_a\Delta_a\mathbb{E}[N_T(a)]$ (the regret decomposition); the pseudo-regret uses true means instead of realized rewards. In episodic MDPs it is $\sum_k(V^\ast_0(s^k_0) - V^{\pi_k}_0(s^k_0))$, and in games regret is measured against the best fixed action in hindsight (external regret). → [Ch 02 §11.1](chapters/02-multi-armed-bandits.md); also [Ch 02 §11.2](chapters/02-multi-armed-bandits.md), [Ch 19 §7.1](chapters/19-rl-theory.md)

**Regret matching**: Plays each action with probability proportional to its positive cumulative regret, $x_{t+1}(a) \propto [\mathrm{Reg}_t(a)]^+$, with regret at most $L\sqrt{\lvert\mathcal A\rvert T}$. Regret matching+ keeps a regret vector clipped at zero at every step, so it never accumulates negative regret, and is used in CFR+. → [Ch 17 §5.3](chapters/17-multi-agent-rl.md); also [Ch 17 §10.4](chapters/17-multi-agent-rl.md)

**REINFORCE**: The Monte Carlo policy-gradient algorithm: after each episode, $\boldsymbol\theta \leftarrow \boldsymbol\theta + \alpha\gamma^tG_t\nabla\log\pi_{\boldsymbol\theta}(A_t \mid S_t)$ for every step. It is unbiased but high-variance; reward-to-go and a baseline reduce the variance. Its batched form with a fixed step size $\boldsymbol\theta \leftarrow \boldsymbol\theta + \alpha\hat{\mathbf g}$ is called vanilla policy gradient (VPG). → [Ch 10 §5.2](chapters/10-policy-gradients.md); also [Ch 00 §6.2](chapters/00-math-toolkit.md), [Ch 11 §1.2](chapters/11-trust-regions-and-ppo.md)

**Reinforcement learning** (RL): Learning to act from evaluative, delayed feedback in a loop where the agent's own choices determine its data, with the goal of maximizing expected cumulative reward. Exploration and credit assignment are therefore central. → [Ch 01 §1](chapters/01-the-rl-problem.md)

**Rejection-sampling fine-tuning** (RFT): Sample several responses per prompt, keep those a verifier or reward model accepts and fine-tune on them by maximum likelihood, as in Llama 2's post-training; repeated over rounds it is expert iteration. With a 0/1 reward and plain SFT on the kept samples its gradient is REINFORCE with no baseline, and failed samples get zero weight. DeepSeekMath found online RFT better than RFT on SFT samples, and GRPO better than both. → [Ch 18 §9.5](chapters/18-rl-for-language-models.md); also [Ch 18 §8.3](chapters/18-rl-for-language-models.md)

**Relative overgeneralisation** (shadowed equilibria): In cooperative games, learners are drawn to a suboptimal equilibrium because, while the partner is still exploring, the optimal action looks bad on average. → [Ch 17 §6.3](chapters/17-multi-agent-rl.md)

**Reparameterization trick** (pathwise gradient): Writing a sample as a differentiable function of parameters and fixed noise, $x = g_{\boldsymbol\theta}(\xi)$, so the gradient of an expectation can flow through the sample. It is usually lower-variance than the score function but needs a differentiable integrand; SAC's actor uses it. → [Ch 00 §6.4](chapters/00-math-toolkit.md); also [Ch 12 §6.2](chapters/12-continuous-control-actor-critic.md)

**REPS** (relative entropy policy search): Maximize the expected return over a new search distribution subject to $D_{\mathrm{KL}}(q\Vert p_{\text{old}}) \le \epsilon$. The solution is $q^\ast \propto p_{\text{old}}\,e^{R/\eta}$, with the temperature $\eta$ found by minimizing a convex one-dimensional dual, so it is set in reward units by the data rather than by hand. A weighted-maximum-likelihood M-step then fits a parametric (for example Gaussian) distribution to the weighted samples, which tends to shrink it too fast. Episodic REPS over movement-primitive parameters is the robot-learning form. → [Ch 12 §9.2](chapters/12-continuous-control-actor-critic.md); also [Ch 12 §9.1](chapters/12-continuous-control-actor-critic.md)

**Rescorla–Wagner model**: A model of classical conditioning in which, on a trial with the set $\mathcal{P}$ of stimuli present and US magnitude $R$, every present stimulus is updated by $w_i \leftarrow w_i + \alpha_i(R - \sum_{j\in\mathcal{P}}w_j)$: all stimuli share one prediction error. It is the least-mean-squares rule of Widrow and Hoff, gradient Monte Carlo with the US as target, and it explains blocking. Working with whole trials, it cannot say when the US is expected or produce second-order conditioning; the TD model can. → [Ch 05 §14.2](chapters/05-temporal-difference.md)

**Residual-gradient algorithm**: True SGD on the squared TD error, differentiating through the target as well. The naive version converges to the minimizer of the mean squared TD error, which gives wrong values in stochastic tasks; minimizing the Bellman error instead needs double sampling. → [Ch 08 §3.3](chapters/08-function-approximation.md); also [Ch 08 §15.1](chapters/08-function-approximation.md)

**Retrace($\lambda$)**: Off-policy multi-step learning with truncated trace coefficients $c_t = \lambda\min(1, \rho_t)$. Its operator is a $\gamma$-contraction to $q_\pi$ for any behaviour policy (safe), it does not cut traces needlessly near on-policy (efficient), and it never amplifies them (low variance). → [Ch 06 §14.2](chapters/06-n-step-and-eligibility-traces.md); also [Ch 06 §14.1](chapters/06-n-step-and-eligibility-traces.md)

**Return** ($G_t$): The cumulative discounted reward from time $t$, $G_t \doteq \sum_{k\ge0}\gamma^kR_{t+k+1}$, which satisfies $G_t = R_{t+1} + \gamma G_{t+1}$. RL maximizes its expectation. → [Ch 01 §4](chapters/01-the-rl-problem.md)

**Reward** ($R_{t+1}$): The scalar signal the environment sends after each action; it says what the agent should achieve, not how. By the course's convention, the reward that results from $S_t, A_t$ is indexed $t+1$. → [Ch 01 §3.1](chapters/01-the-rl-problem.md); also [Ch 01 §2](chapters/01-the-rl-problem.md)

**Reward-free RL**: Exploring an environment with no reward so that afterwards a near-optimal policy can be computed for any reward function; tabular methods need about $\tilde O(S^2A\,\mathrm{poly}(H)/\varepsilon^2)$ episodes. → [Ch 14 §12.1](chapters/14-exploration.md)

**Reward hacking** (specification gaming): Behaviour that achieves high reward through a loophole in its specification rather than by doing the intended task; with learned reward models it appears as over-optimization. → [Ch 20 §2.1](chapters/20-deep-rl-in-practice.md); also [Ch 18 §8](chapters/18-rl-for-language-models.md)

**Reward hypothesis**: The working assumption that all goals can be expressed as maximizing the expected cumulative sum of a scalar reward. It is powerful but limited by multiple objectives, risk, expressivity and misspecification. → [Ch 01 §3.2](chapters/01-the-rl-problem.md); also [Ch 01 §3.3](chapters/01-the-rl-problem.md)

**Reward machine**: A finite automaton over high-level events $L(s,a,s')$ whose state $u$ records progress through a task and whose transitions emit rewards. The product MDP on $(s,u)$ is Markov again, so ordinary RL applies to non-Markov task specifications such as "fetch the coffee, then bring it to the office"; specifications in linear temporal logic can be compiled into such automata, and one environment transition can update the values of every machine state at once. → [Ch 15 §6.7](chapters/15-beyond-mdps.md)

**Reward model** ($r_\phi$): A network trained on human (or AI) comparisons, usually with the Bradley–Terry loss, to score responses; it is identified only up to a per-prompt constant and is optimized against as a proxy for the true reward in RLHF. → [Ch 18 §2.3](chapters/18-rl-for-language-models.md); also [Ch 18 §2.5](chapters/18-rl-for-language-models.md)

**Reward-prediction-error hypothesis** (of dopamine): The phasic firing of midbrain dopamine neurons resembles the TD error: it responds to unexpected rewards, moves to the earliest predictive cue with learning, and dips when a predicted reward is omitted. A TD model whose states encode the time since the cue reproduces all three signatures. It is a hypothesis, not a settled fact: dopamine also responds to novelty and salience, and different neurons may encode a distribution of prediction errors. → [Ch 05 §14.1](chapters/05-temporal-difference.md); also [Ch 05 §14.2](chapters/05-temporal-difference.md), [Ch 05 §14.4](chapters/05-temporal-difference.md)

**Reward shaping**: Adding a shaping term $F$ to the reward to speed up learning; it can change the optimal policy unless it is potential-based. Its ancestor is Skinner's shaping of animal behaviour by rewarding successive approximations. → [Ch 20 §2.3](chapters/20-deep-rl-in-practice.md); also [Ch 05 §14.3](chapters/05-temporal-difference.md)

**Reward-weighted regression** (RWR): Policy search by expectation–maximization with exponentiated-reward weights: the E-step reweights samples from the old policy by $e^{R/\eta}$, and the M-step is weighted maximum likelihood (originally a weighted regression of actions on state features, for immediate rewards). In parameter space it is the REPS step with a hand-set temperature, which must track the spread of the returns as it shrinks during learning: too small a temperature collapses the search distribution, too large a one barely moves it. → [Ch 12 §9.3](chapters/12-continuous-control-actor-critic.md); also [Ch 10 §15.3](chapters/10-policy-gradients.md)

**Riccati equation** (Riccati recursion, DARE): The recursion $P_t = C_s + \gamma F^\top P_{t+1}F - \gamma^2F^\top P_{t+1}G(C_a + \gamma G^\top P_{t+1}G)^{-1}G^\top P_{t+1}F$ for the matrix of LQR's quadratic value function; it is backward induction (value iteration) on quadratic functions, at $O(d_s^3 + d_a^3)$ per stage. Its fixed-point equation is the discrete algebraic Riccati equation (DARE), whose solution gives the optimal stationary gain. The discounted DARE is the undiscounted one for $(\sqrt\gamma F, \sqrt\gamma G)$, a scaling that standard solvers need. → [Ch 03 §11.4](chapters/03-dynamic-programming.md)

**RiverSwim**: A stochastic chain whose large reward lies upstream, where moving right often fails, while moving left always works and pays a little. It shows that an exploration bonus must persist beyond the first visit and shrink with the counts. → [Ch 14 §5.3](chapters/14-exploration.md)

**RL²**: Meta-RL with a recurrent policy that receives the previous action, reward and termination flag and keeps its hidden state across the episodes of a trial; trained by ordinary RL on the trial return, its weights encode a learning algorithm and its hidden state plays the role of the belief. → [Ch 15 §7.3](chapters/15-beyond-mdps.md)

**RLAIF** (RL from AI feedback): RL with preference labels produced by an AI model instead of humans, as in the RL stage of Constitutional AI; LLM judges are now pervasive. → [Ch 18 §7](chapters/18-rl-for-language-models.md)

**RLHF** (RL from human feedback): The pipeline that turns a pretrained language model into an assistant: supervised fine-tuning, a reward model trained on human comparisons, and RL (originally PPO) against the reward model with a KL penalty to the reference policy. → [Ch 18 §1.4](chapters/18-rl-for-language-models.md); also [Ch 18 §4](chapters/18-rl-for-language-models.md), [Ch 16 §14](chapters/16-offline-rl-and-imitation.md)

**RLOO** (REINFORCE leave-one-out): Critic-free RLHF that samples $k$ responses per prompt and uses the average reward of the other $k-1$ as each response's baseline; folding the KL into the reward gives an exact gradient. → [Ch 18 §4.3](chapters/18-rl-for-language-models.md)

**RLVR** (RL with verifiable rewards): RL in which a program checks the answer (unit tests, a math checker, format checks) instead of a learned reward model; the training method behind reasoning models such as DeepSeek-R1. → [Ch 18 §10.1](chapters/18-rl-for-language-models.md); also [Ch 18 §10.4](chapters/18-rl-for-language-models.md)

**R-MAX**: Optimistic model-based exploration that treats state–action pairs visited fewer than $m$ times as unknown and plans in a model where unknown pairs lead to a fictitious state paying the maximum reward forever. It is PAC-MDP. → [Ch 14 §3.2](chapters/14-exploration.md); also [Ch 19 §6.3](chapters/19-rl-theory.md)

**Robbins–Monro conditions**: Step sizes with $\sum_n\alpha_n = \infty$ and $\sum_n\alpha_n^2 < \infty$, sufficient (not necessary) for stochastic-approximation iterates such as TD(0) and Q-learning to converge. Satisfying them does not make a schedule fast. → [Ch 00 §3.4](chapters/00-math-toolkit.md); also [Ch 05 §5](chapters/05-temporal-difference.md), [Ch 19 §3.5](chapters/19-rl-theory.md)

**Robust MDP**: An MDP whose transition kernel is known only to lie in an uncertainty set around a nominal kernel. The agent maximizes the value it would obtain if an adversary chose the transitions, through the robust Bellman operator $(\mathcal{T}_{\mathcal U}v)(s) = \max_a\min_{p\in\mathcal{U}_{s,a}}\sum_{s'}p(s')[r(s,a,s') + \gamma v(s')]$; for rectangular sets robust value iteration converges and a deterministic stationary policy is optimal. $L_1$ balls connect it to pessimism in offline RL and KL balls to exponential-utility risk aversion; domain randomization averages over models instead. → [Ch 20 §11.6](chapters/20-deep-rl-in-practice.md); also [Ch 20 §12](chapters/20-deep-rl-in-practice.md)

**Rollout algorithm**: Decision-time planning that estimates each action's value at the current state by Monte Carlo simulations of a fixed rollout policy and picks the best; one step of policy improvement over the rollout policy, not an optimal method. → [Ch 07 §10](chapters/07-planning-and-learning-tabular.md)

## S

**SAC** (Soft Actor-Critic): Off-policy maximum-entropy actor-critic: twin soft Q critics with soft targets, a reparameterized tanh-Gaussian actor, and automatic temperature tuning towards a target entropy. It became the default off-policy algorithm for continuous control. → [Ch 12 §6](chapters/12-continuous-control-actor-critic.md); also [Ch 12 §6.6](chapters/12-continuous-control-actor-critic.md)

**Sample complexity**: The number of samples (or of episodes, or of non-optimal steps) an algorithm needs to return an $\varepsilon$-optimal policy with probability at least $1-\delta$. With a generative model the minimax rate is $\tilde\Theta(SA/((1-\gamma)^3\varepsilon^2))$. → [Ch 19 §1.1](chapters/19-rl-theory.md); also [Ch 19 §5.2](chapters/19-rl-theory.md), [Ch 19 §5.3](chapters/19-rl-theory.md)

**Sample model**: A model that produces one sampled next state and reward for a given state and action, rather than their full distribution; it supports sample updates. → [Ch 07 §1.1](chapters/07-planning-and-learning-tabular.md)

**Sampler–trainer mismatch**: In distributed RL for language models, the data come from a sampler policy $\pi_{\text{samp}}$ that differs from the trainer's $\pi_{\boldsymbol\theta_{\text{old}}}$, because asynchronous samplers run a few updates behind and because inference engines and training code compute slightly different probabilities from the same weights. As in IMPALA, each token's surrogate is multiplied by a truncated importance weight $\min(\pi_{\boldsymbol\theta_{\text{old}}}/\pi_{\text{samp}}, C)$, while PPO's clip stays anchored at $\pi_{\boldsymbol\theta_{\text{old}}}$. → [Ch 18 §13.3](chapters/18-rl-for-language-models.md); also [Ch 10 §14](chapters/10-policy-gradients.md)

**SARSA**: On-policy TD control with update $Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]$, named after the quintuple $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$. It converges to $q_\ast$ under GLIE exploration and Robbins–Monro step sizes. → [Ch 05 §7.2](chapters/05-temporal-difference.md); also [Ch 05 §7.3](chapters/05-temporal-difference.md)

**SARSA($\lambda$)**: SARSA with eligibility traces over state–action pairs, which spreads credit along the whole recent path at once. → [Ch 06 §13.1](chapters/06-n-step-and-eligibility-traces.md)

**Scalarization** (SER and ESR): Turning a vector return into a scalar with a utility $U$. Linear scalarization $\mathbf{w}^\top\mathbf{V}$ makes multi-objective RL the successor-features setting but finds only supported points of the Pareto front. With a nonlinear utility, the scalarized expected return (SER), $U(\mathbb{E}_\pi[\mathbf{G}])$, suits a user who cares about averages over many episodes, and the expected scalarized return (ESR), $\mathbb{E}_\pi[U(\mathbf{G})]$, one who lives with each episode's outcome; under ESR the optimal policy may have to be non-stationary. → [Ch 15 §6.7](chapters/15-beyond-mdps.md)

**Score function** (log-derivative trick): $\nabla_{\boldsymbol\theta}\mathbb{E}_{p_{\boldsymbol\theta}}[f] = \mathbb{E}_{p_{\boldsymbol\theta}}[f\nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}]$, using $\nabla p = p\nabla\log p$. It works for any $f$ and for discrete variables but is noisy; it is the basis of REINFORCE and the policy gradient theorem. → [Ch 00 §6.2](chapters/00-math-toolkit.md); also [Ch 10 §4.3](chapters/10-policy-gradients.md)

**Self-play**: Training an agent against copies of itself. It drives AlphaZero; in intransitive games naive self-play can cycle, which fictitious self-play, PSRO and league training address by playing against mixtures and populations. Self-play teams in cooperative games can settle on arbitrary conventions that fail with new partners (zero-shot coordination). → [Ch 17 §9.1](chapters/17-multi-agent-rl.md); also [Ch 13 §9.3](chapters/13-model-based-rl.md), [Ch 17 §9.6](chapters/17-multi-agent-rl.md)

**Self-predictive representation**: An encoder trained, together with a latent transition model, so that the prediction made from $(s_t, a_t)$ matches the embedding of $s_{t+1}$ computed by a slowly moving copy of the encoder with the gradient stopped (SPR, TD-MPC). In an idealized linear analysis the stop-gradient and a fast predictor prevent collapse. An encoder that predicts the reward and its own next latent is sufficient to represent the optimal values, and applied to histories it yields a Markov, belief-like state. → [Ch 15 §6.6](chapters/15-beyond-mdps.md); also [Ch 09 §12](chapters/09-deep-q-learning.md)

**Semi-gradient method**: An update that uses a bootstrapped target but differentiates only the prediction, treating the target as a constant, e.g. semi-gradient TD $\mathbf{w} \leftarrow \mathbf{w} + \alpha[R_{t+1} + \gamma\hat v(S_{t+1},\mathbf{w}) - \hat v(S_t,\mathbf{w})]\nabla\hat v(S_t,\mathbf{w})$. It is not a gradient method: its expected update is in general not the gradient of any function. In code the target is computed under a stop-gradient (`detach()`). → [Ch 08 §3.2](chapters/08-function-approximation.md); also [Ch 08 §3.3](chapters/08-function-approximation.md), [Ch 00 §7.6](chapters/00-math-toolkit.md)

**Semi-MDP** (SMDP): A decision process in which actions take a variable, random amount of time; options turn an MDP into one, with multi-time models that fold the duration into discounting. SMDP Q-learning backs up over whole option executions. → [Ch 15 §5.2](chapters/15-beyond-mdps.md); also [Ch 15 §5.4](chapters/15-beyond-mdps.md)

**SFT** (supervised fine-tuning): Fine-tuning a pretrained language model on demonstration responses by maximum likelihood; the first stage of RLHF and the reference policy $\pi_{\mathrm{ref}}$ of later stages. It is behaviour cloning. → [Ch 18 §1.4](chapters/18-rl-for-language-models.md); also [Ch 16 §14](chapters/16-offline-rl-and-imitation.md)

**Shapley's value iteration**: Value iteration for two-player zero-sum Markov games that replaces the max by the value of the stage matrix game at each state; the operator is a $\gamma$-contraction. → [Ch 17 §4.1](chapters/17-multi-agent-rl.md)

**Shielding**: A safety layer, synthesized from a formal specification and a model of the safety-relevant dynamics, that replaces any action that could lead to a violation by a safe one. Action masking, safety layers that project actions onto a safe set, and control barrier functions are related tools. → [Ch 20 §11.4](chapters/20-deep-rl-in-practice.md)

**SimBa**: A network architecture for deep RL (running observation normalization, residual MLP blocks and layer normalization) that lets parameters scale up; plugged into SAC it matched or beat the state of the art at modest compute. → [Ch 12 §8](chapters/12-continuous-control-actor-critic.md); also [Ch 20 §3.5](chapters/20-deep-rl-in-practice.md)

**SimPO**: A reference-free direct preference loss that uses the length-normalized log-probability of a response as the implicit reward, plus a target margin. → [Ch 18 §6](chapters/18-rl-for-language-models.md)

**Sim-to-real transfer**: Training in simulation and deploying on a physical system despite the reality gap, using system identification, domain randomization, memory or adaptation modules and privileged teachers. → [Ch 20 §12](chapters/20-deep-rl-in-practice.md)

**Simulation lemma**: Bounds the value difference of a fixed policy in two MDPs by their reward and transition errors, $\lVert\hat V^\pi - V^\pi\rVert_\infty \le \frac{\varepsilon_r}{1-\gamma} + \frac{\gamma\varepsilon_P R_{\max}}{2(1-\gamma)^2}$; a policy optimal in the wrong model loses at most twice the value error. → [Ch 13 §2.2](chapters/13-model-based-rl.md); also [Ch 19 §4.2](chapters/19-rl-theory.md)

**Social dilemma**: A game in which individually rational choices lead to an outcome worse for everyone, such as the prisoner's dilemma or sequential versions in Markov games; repetition, reciprocity and opponent shaping (for example LOLA, which differentiates through the opponent's anticipated learning step) can sustain cooperation. → [Ch 17 §12](chapters/17-multi-agent-rl.md); also [Ch 17 §3.4](chapters/17-multi-agent-rl.md)

**Soft Bellman equation**: The Bellman equation of maximum-entropy RL, with the max replaced by a log-sum-exp: $v^\ast(s) = \alpha\log\sum_a e^{q^\ast(s,a)/\alpha}$ and $\pi^\ast(a \mid s) = e^{(q^\ast(s,a) - v^\ast(s))/\alpha}$. Iterating it is soft value iteration; the same equation appears in maximum-entropy IRL and at the token level in RLHF. → [Ch 12 §5.2](chapters/12-continuous-control-actor-critic.md); also [Ch 16 §4.3](chapters/16-offline-rl-and-imitation.md), [Ch 18 §3.5](chapters/18-rl-for-language-models.md)

**Softmax policy**: $\pi(a \mid s) = e^{h(s,a,\boldsymbol\theta)}/\sum_b e^{h(s,b,\boldsymbol\theta)}$ over action preferences $h$, the standard discrete policy parameterization; with one parameter per pair it is the tabular softmax. → [Ch 10 §2.1](chapters/10-policy-gradients.md); also [Ch 02 §8.1](chapters/02-multi-armed-bandits.md)

**Soft policy iteration**: Alternates soft policy evaluation (a $\gamma$-contraction) with soft policy improvement (projecting onto the Boltzmann policy of the soft action values); it converges to the soft-optimal policy. SAC is its function-approximation version. → [Ch 12 §5.5](chapters/12-continuous-control-actor-critic.md); also [Ch 12 §5.4](chapters/12-continuous-control-actor-critic.md)

**Sparse reward**: A reward that is nonzero only on rare events, such as reaching a goal. It specifies the task precisely but may never be seen under random exploration; dense rewards guide learning but are easier to game. → [Ch 20 §2.2](chapters/20-deep-rl-in-practice.md); also [Ch 14 §6](chapters/14-exploration.md)

**SPSA** (simultaneous perturbation stochastic approximation): Estimates the whole gradient from two episodes along a random direction $\boldsymbol\Delta \in \lbrace -1, +1\rbrace^d$ of independent fair signs, $\hat{\mathbf{g}} = \frac{\hat J(\boldsymbol\theta + c\boldsymbol\Delta) - \hat J(\boldsymbol\theta - c\boldsymbol\Delta)}{2c}\boldsymbol\Delta$, whose mean is $\nabla J + O(c^2)$ whatever $d$ is. The gradient's components along the other directions become noise: with exact evaluations the total variance is $(d-1)\lVert\nabla J\rVert^2$. → [Ch 10 §15.1](chapters/10-policy-gradients.md)

**Squashed Gaussian** (tanh-Gaussian policy): A Gaussian sample passed through tanh to respect action bounds; its log-density needs the change-of-variables correction $-\sum_i\log(1-\tanh^2u_i)$. → [Ch 12 §6.3](chapters/12-continuous-control-actor-critic.md); also [Ch 10 §2.3](chapters/10-policy-gradients.md)

**STaR** (Self-Taught Reasoner): Expert iteration for reasoning: generate rationales, keep those that reach the correct answer and fine-tune the base model on them afresh in every round. Its rationalization step shows a failed problem's answer as a hint and keeps the rationales that reach it as if unhinted, a biased E-step that buys coverage of hard problems, possibly with rationales written backwards from the answer. → [Ch 18 §9.5](chapters/18-rl-for-language-models.md)

**State** ($S_t$): The information the agent uses to choose actions; it should be Markov, summarizing everything in the history that matters for the future. → [Ch 01 §2](chapters/01-the-rl-problem.md); also [Ch 01 §5](chapters/01-the-rl-problem.md)

**State-adversarial MDP** (SA-MDP): An adversary perturbs what the agent observes within a set around the true state, while the true state evolves normally; it formalizes attacks such as imperceptible gradient-based perturbations of Atari frames. Under an optimal adversary a stationary Markov optimal policy may not exist, and a regularizer on how much the action distribution can change within the perturbation set bounds the loss of value. Adversarial policies attack through another agent instead: opponents trained against frozen victims win with uncoordinated behaviour that makes the victims' observations adversarial. → [Ch 20 §11.6](chapters/20-deep-rl-in-practice.md)

**State aggregation**: The simplest function approximator: partition the states into groups that share one value, so each update touches one weight. → [Ch 08 §3.4](chapters/08-function-approximation.md); also [Ch 08 §6.1](chapters/08-function-approximation.md)

**Stationary distribution**: A distribution $\mathbf d$ over the states of a Markov chain with $\mathbf{d}^\top = \mathbf{d}^\top\mathbf{P}$. An irreducible finite chain has a unique one; the on-policy distribution of a continuing task is the stationary distribution of the policy's chain. → [Ch 00 §1.6](chapters/00-math-toolkit.md); also [Ch 08 §2.2](chapters/08-function-approximation.md)

**Stochastic approximation**: The theory of iterations $\boldsymbol\theta_{n+1} = \boldsymbol\theta_n + \alpha_n[h(\boldsymbol\theta_n) + \text{noise}]$; under Robbins–Monro step sizes the iterates track the ODE $\dot{\boldsymbol\theta} = h(\boldsymbol\theta)$ (the ODE method), which explains why TD and Q-learning converge. → [Ch 00 §3.4](chapters/00-math-toolkit.md); also [Ch 19 §3.1](chapters/19-rl-theory.md), [Ch 19 §3.2](chapters/19-rl-theory.md)

**Stochastic gradient descent** (SGD): $\mathbf{w}_{k+1} = \mathbf{w}_k - \alpha_k\mathbf{g}_k$ with an unbiased gradient estimate $\mathbf{g}_k$; noise enters the expected descent only at second order in $\alpha$, and a constant step size leaves a noise floor. → [Ch 00 §3.3](chapters/00-math-toolkit.md)

**Stratified bootstrap**: Confidence intervals for an aggregate score obtained by resampling runs within each task, independently across tasks, and recomputing the aggregate. → [Ch 20 §6.4](chapters/20-deep-rl-in-practice.md)

**Successor features** (SF): The expected discounted sum of features $\boldsymbol\phi(s,a,s')$ under a policy, $\boldsymbol\psi^\pi(s,a)$ (Chapter 15's local use of $\boldsymbol\psi$). For any reward linear in the features, $r_{\mathbf w} = \boldsymbol\phi^\top\mathbf{w}$, the action values are $q^\pi_{\mathbf w}(s,a) = \boldsymbol\psi^\pi(s,a)^\top\mathbf{w}$, which with GPI enables zero-shot transfer to new tasks. With $\boldsymbol\phi$ equal to a vector reward they evaluate every linear scalarization of a multi-objective problem; forward-backward representations learn the features too. → [Ch 15 §6.2](chapters/15-beyond-mdps.md); also [Ch 15 §6.6](chapters/15-beyond-mdps.md), [Ch 15 §6.7](chapters/15-beyond-mdps.md)

**Successor measure**: $M^\pi(s,a,X) = \sum_{t\ge0}\gamma^t\Pr\lbrace S_{t+1}\in X \mid S_0 = s, A_0 = a, \pi\rbrace$, the successor representation for large state spaces: $q^\pi_r(s,a) = \int M^\pi(s,a,\mathrm{d}s')\,r(s')$ for every reward that depends on the next state. Contrastive RL and forward-backward representations learn it from reward-free data. → [Ch 15 §6.6](chapters/15-beyond-mdps.md)

**Successor representation** (SR): $\mathbf{M}^\pi = (\mathbf{I} - \gamma\mathbf{P}_\pi)^{-1}$, the expected discounted future occupancy of each state from each state, so that $\mathbf{v}_\pi = \mathbf{M}^\pi\mathbf{r}_\pi$ separates dynamics from reward. → [Ch 15 §6.1](chapters/15-beyond-mdps.md)

**Surrogate objective** ($L_\pi$): $L_\pi(\pi') = J(\pi) + \frac{1}{1-\gamma}\mathbb{E}_{s\sim d^\pi, a\sim\pi}[\frac{\pi'(a \mid s)}{\pi(a \mid s)}A_\pi(s,a)]$, the performance difference with the new policy's state distribution replaced by the old one; it matches $J$ to first order and can be estimated with importance ratios. In policy-gradient code, a surrogate loss is any loss whose gradient equals the gradient estimate. → [Ch 11 §3.1](chapters/11-trust-regions-and-ppo.md); also [Ch 10 §5.3](chapters/10-policy-gradients.md)

## T

**Tabular methods**: Methods that store one number per state or state–action pair. They come with clean convergence guarantees but do not generalize, which limits them to small problems. → [Ch 01 §14](chapters/01-the-rl-problem.md); also [Ch 08 §1](chapters/08-function-approximation.md)

**Target network** ($\mathbf{w}^-$, $\bar{\mathbf w}$): A lagged copy of the value network used to compute bootstrapped targets, refreshed every $C$ steps or by Polyak averaging, so that the regression target stays fixed between refreshes. → [Ch 09 §2.3](chapters/09-deep-q-learning.md); also [Ch 12 §3.3](chapters/12-continuous-control-actor-critic.md)

**Target policy**: The policy being evaluated or improved in off-policy learning, as opposed to the behaviour policy that generates the data. → [Ch 04 §6.1](chapters/04-monte-carlo.md); also [Ch 01 §14](chapters/01-the-rl-problem.md)

**Target policy smoothing**: TD3's addition of clipped Gaussian noise to the target action, $\tilde a' = \mathrm{clip}(\mu_{\bar{\boldsymbol\theta}}(s') + \mathrm{clip}(\tilde\sigma\xi, -c, c), -1, 1)$, so the critic cannot exploit narrow peaks in its own estimate. → [Ch 12 §4.5](chapters/12-continuous-control-actor-critic.md)

**TD3** (twin delayed DDPG): DDPG with three changes against overestimation: clipped double-Q targets, delayed policy (and target) updates, and target policy smoothing. → [Ch 12 §4](chapters/12-continuous-control-actor-critic.md); also [Ch 12 §4.7](chapters/12-continuous-control-actor-critic.md)

**TD3+BC**: Offline RL by adding a behaviour-cloning term to TD3's actor loss, $\max_\pi\mathbb{E}_{\mathcal D}[\lambda Q(s,\pi(s)) - \lVert\pi(s) - a\rVert^2]$ with $\lambda$ normalized by the average $\lvert Q\rvert$. → [Ch 16 §7.3](chapters/16-offline-rl-and-imitation.md)

**TD7**: TD3 plus learned state and state–action embeddings, loss-adjusted prioritized replay and policy checkpoints; it works both online and offline. → [Ch 12 §8](chapters/12-continuous-control-actor-critic.md)

**TD error** ($\delta_t$): $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$, the difference between the TD target $R_{t+1} + \gamma V(S_{t+1})$ and the current estimate. With $V$ fixed, the Monte Carlo error is the discounted sum of TD errors; with an exact critic it is an unbiased advantage estimate. → [Ch 05 §1.2](chapters/05-temporal-difference.md); also [Ch 05 §1.3](chapters/05-temporal-difference.md), [Ch 10 §7.2](chapters/10-policy-gradients.md)

**TD fixed point** ($\mathbf{w}_{TD}$): The weights $\mathbf{w}_{TD} = \mathbf{A}^{-1}\mathbf{b}$ at which the expected linear semi-gradient TD update is zero; they solve the projected Bellman equation. On-policy, $\overline{VE}(\mathbf{w}_{TD}) \le \frac{1}{1-\gamma^2}\min_{\mathbf w}\overline{VE}(\mathbf{w}) \le \frac{1}{1-\gamma}\min_{\mathbf w}\overline{VE}(\mathbf{w})$. For TD($\lambda$) the same Pythagorean argument gives the norm constant $1/\sqrt{1-\gamma_\lambda^2}$, $\gamma_\lambda = \gamma(1-\lambda)/(1-\gamma\lambda)$, sharper than Tsitsiklis and Van Roy's $(1-\gamma\lambda)/(1-\gamma)$. → [Ch 08 §4.2](chapters/08-function-approximation.md); also [Ch 08 §5.3](chapters/08-function-approximation.md), [Ch 19 §3.6](chapters/19-rl-theory.md)

**TD-Gammon**: Tesauro's backgammon program, which combined TD($\lambda$) with a neural network and self-play to reach master level, evaluating the positions reachable by its candidate moves (afterstates); an early landmark of RL with function approximation. → [Ch 05 §12](chapters/05-temporal-difference.md); also [Ch 17 §9.1](chapters/17-multi-agent-rl.md)

**TD($\lambda$)**: TD learning with eligibility traces, $\mathbf{w} \leftarrow \mathbf{w} + \alpha\delta_t\mathbf{z}_t$, the backward view of the $\lambda$-return; $\lambda$ interpolates between TD(0) and Monte Carlo. → [Ch 06 §9](chapters/06-n-step-and-eligibility-traces.md); also [Ch 08 §3.5](chapters/08-function-approximation.md)

**TD model of classical conditioning**: Sutton and Barto's real-time model: linear TD learning within each trial, with value $V_t = \mathbf{w}^\top\mathbf{x}_t$ over a stimulus representation such as the complete serial compound (CSC), one feature per time step since each stimulus came on. On single-step trials it reduces to Rescorla–Wagner, so it inherits blocking, and it adds timing and second-order conditioning. Its TD error moves from the US to the cue with training and dips when an expected US is withheld, as dopamine does. → [Ch 05 §14.2](chapters/05-temporal-difference.md); also [Ch 05 §14.1](chapters/05-temporal-difference.md)

**TD-MPC and TD-MPC2**: Model-based agents with a decoder-free latent model trained by latent consistency, reward and TD losses, which plan with an MPPI-style sampler over a short horizon plus a learned terminal value and a policy prior. TD-MPC2 made it robust across many tasks with one set of hyperparameters. → [Ch 13 §7.6](chapters/13-model-based-rl.md)

**Temperature**: A parameter that controls how random a softmax-type policy is: $\tau$ in Boltzmann exploration, the entropy temperature $\alpha$ in maximum-entropy RL (SAC), and $\tau$ in AlphaZero's visit-count move selection $\pi(a \mid s_0) \propto N(s_0,a)^{1/\tau}$. → [Ch 02 §9](chapters/02-multi-armed-bandits.md); also [Ch 12 §5.1](chapters/12-continuous-control-actor-critic.md), [Ch 13 §9.3](chapters/13-model-based-rl.md)

**Temporal-difference learning** (TD learning, TD(0)): Updating a guess from a later guess: $V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$ after every step, without a model (sampling) and without waiting for the final outcome (bootstrapping). Batch TD(0) converges to the certainty-equivalence estimate. → [Ch 05 §2.1](chapters/05-temporal-difference.md); also [Ch 05 §1.2](chapters/05-temporal-difference.md), [Ch 05 §5](chapters/05-temporal-difference.md)

**Terminal state**: The state that ends an episode, with value zero by definition, so the TD target of a transition into it is $R_{t+1}$ alone. It can be modelled as an absorbing state. → [Ch 01 §4.1](chapters/01-the-rl-problem.md); also [Ch 01 §4.4](chapters/01-the-rl-problem.md)

**Termination vs truncation**: An episode is terminated when it reaches a terminal state (no bootstrap: the target is $R_{t+1}$) and truncated when it is cut off by a time limit (bootstrap from the next state, which still has value). Gymnasium reports them as separate flags; confusing them is one of the most common RL bugs. → [Ch 01 §4.5](chapters/01-the-rl-problem.md); also [Ch 00 §8.2](chapters/00-math-toolkit.md), [Ch 05 §13.1](chapters/05-temporal-difference.md)

**Test-time compute**: Spending more inference computation per problem to raise accuracy: best-of-$n$ with a verifier or reward model, majority voting, search over partial solutions, or longer reasoning. → [Ch 18 §12](chapters/18-rl-for-language-models.md)

**Thompson sampling**: Sample a model (for bandits, one mean per arm) from the posterior and act optimally for the sample; it selects each action with the posterior probability that it is optimal (probability matching). Beta–Bernoulli Thompson sampling is asymptotically optimal. → [Ch 02 §7.2](chapters/02-multi-armed-bandits.md); also [Ch 02 §7.3](chapters/02-multi-armed-bandits.md)

**Tiger problem**: A two-state POMDP: a tiger hides behind one of two doors, listening gives a noisy growl, and opening the wrong door is costly. Memoryless agents can do no better than $-20$, while the belief-optimal agent earns about $19.4$. → [Ch 15 §2.1](chapters/15-beyond-mdps.md); also [Ch 15 §2.7](chapters/15-beyond-mdps.md)

**Tile coding**: Coarse coding with several offset grids (tilings), so each state activates exactly one tile per tiling and the features are sparse binary vectors; fast, easy to set step sizes for, and scalable with hashing. → [Ch 08 §6.5](chapters/08-function-approximation.md)

**Token-level MDP**: Language generation viewed as an MDP whose state is the prompt plus the tokens generated so far, whose action is the next token, and whose transitions are deterministic; the reward usually arrives at the end (with per-token KL penalties in RLHF). → [Ch 18 §1.2](chapters/18-rl-for-language-models.md)

**Tower rule** (law of total expectation): $\mathbb{E}[\mathbb{E}[X \mid Y]] = \mathbb{E}[X]$; "condition on the first step" arguments behind every Bellman equation rest on it. → [Ch 00 §1.3](chapters/00-math-toolkit.md)

**Trajectory sampling**: Distributing planning updates by simulating trajectories under the current policy (the on-policy distribution of updates) instead of sweeping uniformly; it helps early and hurts late. In PETS, trajectory sampling instead means propagating particles through ensemble members (TS-1, TS-$\infty$). → [Ch 07 §7](chapters/07-planning-and-learning-tabular.md); also [Ch 13 §5.1](chapters/13-model-based-rl.md)

**Transposition table**: In game-tree search, a table that shares statistics between nodes reached by different move orders leading to the same position. → [Ch 07 §12.2](chapters/07-planning-and-learning-tabular.md)

**Tree Backup** (TB): Multi-step off-policy learning without importance ratios: at each step the actions not taken contribute their estimated values weighted by $\pi$, and the path continues weighted by $\pi(A \mid S)$ (trace coefficient $c = \pi(A \mid S)$). → [Ch 06 §6.1](chapters/06-n-step-and-eligibility-traces.md); also [Ch 06 §6.2](chapters/06-n-step-and-eligibility-traces.md)

**TRPO** (Trust Region Policy Optimization): Maximizes the importance-weighted surrogate subject to an average-KL constraint $\delta$: it computes the natural-gradient direction by conjugate gradient with Fisher-vector products, scales the step to the KL budget, and finishes with a backtracking line search. It is motivated by the bound $J(\pi') \ge L_\pi(\pi') - \frac{4\epsilon\gamma}{(1-\gamma)^2}(D^{\max}_{\mathrm{TV}})^2$, where $D_{\mathrm{TV}}$ is the total variation distance. → [Ch 11 §6](chapters/11-trust-regions-and-ppo.md); also [Ch 11 §3.3](chapters/11-trust-regions-and-ppo.md)

**True online TD($\lambda$)**: An $O(d)$ algorithm that reproduces exactly the online $\lambda$-return algorithm (which redoes all updates of the episode with the truncated $\lambda$-returns available so far); the dutch trace falls out of its derivation. → [Ch 06 §12.2](chapters/06-n-step-and-eligibility-traces.md); also [Ch 06 §12.1](chapters/06-n-step-and-eligibility-traces.md)

**Truncated $\lambda$-return** ($G^\lambda_{t:h}$): The $\lambda$-return cut off at a horizon $h$, with all remaining weight on the longest available $n$-step return; it makes the forward view implementable with a delay (TTD($\lambda$)). → [Ch 06 §11](chapters/06-n-step-and-eligibility-traces.md)

**Trust region**: A bound on how far one policy update may move the policy, measured in distribution space (usually KL), inside which a surrogate objective can be trusted. TRPO enforces one; PPO's clip only approximates it. → [Ch 11 §1.4](chapters/11-trust-regions-and-ppo.md); also [Ch 11 §6.2](chapters/11-trust-regions-and-ppo.md)

**Two-step task**: A two-stage choice task in which the first choice leads to one of two second-stage states with probability 0.7 (common) or 0.3 (rare), and a second choice there pays off with a slowly drifting probability. After a reward that followed a rare transition, a model-free learner repeats its first choice while a model-based learner switches, so model-free choice shows a reward effect, model-based choice a reward × transition interaction, and people show both. → [Ch 05 §14.3](chapters/05-temporal-difference.md)

**Two time scales**: The step-size condition of actor-critic convergence theory, $\alpha^{\boldsymbol\theta}_k/\alpha^{\mathbf w}_k \to 0$: the critic converges "infinitely faster" than the actor, which therefore sees an essentially converged critic. → [Ch 10 §7.5](chapters/10-policy-gradients.md)

## U

**UCB** (upper confidence bound, UCB1): Choose $\arg\max_a[Q_t(a) + c\sqrt{\ln t/N_t(a)}]$, an optimistic index derived from Hoeffding's inequality; UCB1 ($c = \sqrt2$) has logarithmic regret. → [Ch 02 §6.3](chapters/02-multi-armed-bandits.md); also [Ch 02 §11.5](chapters/02-multi-armed-bandits.md)

**UCB-Q-learning** (UCB-H, UCB-Hoeffding Q-learning): Model-free optimistic Q-learning for episodic MDPs with step size $\alpha_t = (H+1)/(H+t)$ and a count bonus; its regret is $\tilde O(\sqrt{H^4SAT})$. → [Ch 14 §3.4](chapters/14-exploration.md); also [Ch 19 §7.4](chapters/19-rl-theory.md)

**UCBVI**: Optimistic value iteration with count-based bonuses for episodic MDPs; with Bernstein–Freedman bonuses (UCBVI-BF) its regret $\tilde O(\sqrt{HSAT})$ matches the lower bound up to logarithmic factors once $T$ is past a burn-in. → [Ch 14 §3.3](chapters/14-exploration.md); also [Ch 19 §7.3](chapters/19-rl-theory.md)

**UCRL2**: Optimistic exploration in the average-reward setting: plan in the best MDP inside confidence sets around the empirical transitions and rewards (extended value iteration), recomputing when a count doubles; with probability at least $1-\delta$ its regret is at most $34DS\sqrt{AT\ln(T/\delta)}$ for an MDP of diameter $D$. → [Ch 14 §3.3](chapters/14-exploration.md); also [Ch 19 §7.5](chapters/19-rl-theory.md)

**UCT**: Monte Carlo Tree Search with UCB1 as the tree policy at every node, $\arg\max_a[W(s,a)/N(s,a) + c\sqrt{\ln N(s)/N(s,a)}]$. → [Ch 07 §11.2](chapters/07-planning-and-learning-tabular.md)

**Universal value function approximator** (UVFA): A value function $Q(s,a,g)$ that takes the goal as an input, so it generalizes across goals. → [Ch 15 §4.2](chapters/15-beyond-mdps.md)

**Unsupervised environment design** (UED): Adapting the distribution of training contexts (levels) to the agent instead of sampling it uniformly, as domain randomization does. A principled target is the minimax-regret policy, $\arg\min_\pi\max_c[J_c(\pi^\ast_c) - J_c(\pi)]$, which ignores impossible levels (their regret is zero) and concentrates on levels that are solvable but not yet solved. PAIRED, prioritized level replay and its robust and ACCEL variants are examples; POET co-evolves environments and agents. → [Ch 15 §10.4](chapters/15-beyond-mdps.md)

**Update-to-data ratio** (UTD, replay ratio): The number of gradient updates per environment step. Raising it improves sample efficiency up to a point, beyond which overfitting, overestimation and loss of plasticity make performance fall (REDQ, DroQ, CrossQ and resets address this). → [Ch 20 §4](chapters/20-deep-rl-in-practice.md); also [Ch 12 §8](chapters/12-continuous-control-actor-critic.md), [Ch 09 §2.2](chapters/09-deep-q-learning.md)

## V

**Value equivalence**: A model is value-equivalent to the environment with respect to a set of policies and functions if its Bellman operators agree with the true ones on them; such a model can be wrong about everything that does not matter for value, as MuZero's is. → [Ch 13 §8](chapters/13-model-based-rl.md); also [Ch 13 §10.1](chapters/13-model-based-rl.md)

**Value function**: The expected return as a function of the state ($v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$, the state-value function) or of the state and action ($q_\pi$). Values are linked by $v_\pi = \sum_a\pi q_\pi$ and $q_\pi = r + \gamma\sum p\,v_\pi$. → [Ch 01 §8](chapters/01-the-rl-problem.md)

**Value iteration** (VI): Iterating the Bellman optimality operator, $V_{k+1} = \mathcal{T}^\ast V_k$. It converges to $v_\ast$ at rate $\gamma^k$; stopping when $\lVert V_{k+1} - V_k\rVert_\infty < \epsilon(1-\gamma)/(2\gamma)$ guarantees an $\epsilon$-optimal greedy policy. Q-value iteration, $Q_{k+1} = \mathcal{T}^\ast Q_k$, is the same algorithm on action values. → [Ch 03 §6.1](chapters/03-dynamic-programming.md); also [Ch 03 §6.3](chapters/03-dynamic-programming.md), [Ch 03 §2.6](chapters/03-dynamic-programming.md)

**Variance-minimizing baseline**: The baseline $b^\ast(s) = \sum_a\pi\lVert\boldsymbol\psi\rVert^2q_\pi/\sum_a\pi\lVert\boldsymbol\psi\rVert^2$, where $\boldsymbol\psi = \nabla\log\pi(a \mid s)$ is the score: a score-weighted average of action values; in practice $\hat v(s,\mathbf{w})$ is used instead. → [Ch 10 §6.3](chapters/10-policy-gradients.md)

**VDN** (value-decomposition networks): Cooperative MARL with $Q_{tot} = \sum_iQ_i$, the simplest decomposition satisfying IGM. → [Ch 17 §7.3](chapters/17-multi-agent-rl.md)

**Vision-language-action model** (VLA): A generalist robot policy built from a pretrained vision-language model and trained to output actions, either as discretized action tokens (RT-2, which coined the term, and OpenVLA) or through a flow-matching action expert that generates action chunks ($\pi_0$). RT-2 showed semantic generalization that the robot data alone did not teach. → [Ch 16 §2.8](chapters/16-offline-rl-and-imitation.md)

**V-MPO**: The on-policy version of MPO. It learns a state value and uses $n$-step advantages in place of $Q$, applies the exponential weights only to the top half of the advantages in each batch and keeps both KL bounds; without importance weighting, entropy bonuses or population-based tuning it set new multi-task scores on Atari-57 and DMLab-30. → [Ch 12 §9.4](chapters/12-continuous-control-actor-critic.md)

**V-trace**: IMPALA's off-policy correction for state values, using truncated ratios $\rho_t = \min(\bar\rho, \pi/b)$ and trace coefficients $c_t = \min(\bar c, \pi/b)$; its fixed point is the value of a policy $\pi_{\bar\rho} \propto \min(\bar\rho b, \pi)$ that lies between $b$ and $\pi$. → [Ch 06 §14.4](chapters/06-n-step-and-eligibility-traces.md); also [Ch 10 §14.2](chapters/10-policy-gradients.md)

## W

**Wasserstein distance**: A distance between distributions computed from their quantile functions; the distributional Bellman operator is a $\gamma$-contraction in its maximal form, which justifies distributional RL. → [Ch 09 §9.2](chapters/09-deep-q-learning.md)

**Watkins's Q($\lambda$)**: Q-learning with eligibility traces that are cut to zero after every exploratory (non-greedy) action, since later rewards no longer follow the greedy target policy. Without cutting (naive Q($\lambda$)), $q_\ast$ is still the fixed point but the update need not contract and can diverge. → [Ch 06 §13.2](chapters/06-n-step-and-eligibility-traces.md)

**Weighted importance sampling** (WIS): Normalizes importance-weighted returns by the sum of the weights, $\sum\rho_iG_i/\sum\rho_i$; biased but consistent and bounded, usually far better than ordinary IS on small samples. It can be computed incrementally with step size $W/C$. → [Ch 04 §6.3](chapters/04-monte-carlo.md); also [Ch 04 §7](chapters/04-monte-carlo.md), [Ch 00 §2.6](chapters/00-math-toolkit.md)

**World model**: A learned model of the environment's dynamics, often in a latent space, used to plan or to train a policy in imagination, from Ha and Schmidhuber's World Models to Dreamer and video "foundation" world models. → [Ch 13 §7.2](chapters/13-model-based-rl.md); also [Ch 13 §7.4](chapters/13-model-based-rl.md), [Ch 13 §12](chapters/13-model-based-rl.md)

## Z

**Zero-shot coordination** (ZSC): The problem of agents trained independently with the same algorithm and paired at test time with no chance to adapt. Self-play teams often succeed through arbitrary conventions (which of several equally good actions to take, what a signal means), which shows as a gap between the diagonal (self-play) and off-diagonal (cross-play) entries of the cross-play matrix over training runs. Other-play, off-belief learning and partner populations such as Fictitious Co-Play address it. → [Ch 17 §9.6](chapters/17-multi-agent-rl.md)

**Zero-shot RL**: Pre-training on reward-free data so that, once a reward is revealed (for example through samples of $r(s)$), a near-optimal policy follows with no planning and no fine-tuning. Forward-backward representations do this by computing a task vector $\mathbf{z}_r = \mathbb{E}_{s\sim\nu}[B(s)\,r(s)]$ and acting with $\pi_{\mathbf{z}_r}$; successor features with GPI do it for rewards linear in hand-picked features. → [Ch 15 §6.6](chapters/15-beyond-mdps.md); also [Ch 15 §6.3](chapters/15-beyond-mdps.md)

**Zero-shot transfer**: Acting well on a new task without further learning, for example by combining successor features of earlier policies with GPI when the new reward is a linear function of known features. Zero-shot RL learns the features as well, from reward-free data. → [Ch 15 §6.3](chapters/15-beyond-mdps.md); also [Ch 15 §6.4](chapters/15-beyond-mdps.md), [Ch 15 §6.6](chapters/15-beyond-mdps.md)

**Zero-sum game**: A two-player game in which one player's gain is the other's loss. Its value is unique, equilibrium strategies are interchangeable and guarantee the value, and it can be computed by linear programming. → [Ch 17 §2.2](chapters/17-multi-agent-rl.md); also [Ch 17 §3.2](chapters/17-multi-agent-rl.md)
