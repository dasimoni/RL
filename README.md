# Reinforcement Learning, From First Principles to the Frontier

A self-contained course on reinforcement learning (RL). It starts with the math you need and ends at current research: RLHF for language models, world models, offline RL, multi-agent learning and the theory behind all of it. Every chapter has:

* **Text**: intuition first, then full derivations of the central results, pseudocode for every algorithm, and worked examples.
* **Runnable code**: algorithms written from scratch in NumPy or PyTorch. Every number and figure in the text comes from running this code.
* **Exercises with full solutions**: concept checks (★), derivations (★★) and coding or open-ended problems (★★★).
* **History and further reading**: where each idea came from and what to read next.

In numbers: 21 chapters, 191 runnable scripts, 215 figures, 323 exercises with worked solutions, a 600-term glossary and a reading list of about 300 papers.

Notation is fixed course-wide in [NOTATION.md](NOTATION.md). It follows Sutton & Barto (2nd ed.).

## Chapters

| # | Chapter | What you will be able to do |
|---|---|---|
| 00 | [The Mathematical and ML Toolkit for RL](chapters/00-math-toolkit.md) | Use the probability, estimation, optimization, information-theory and PyTorch/Gymnasium tools that the rest of the course relies on |
| **Part I** | **Foundations and tabular methods** | |
| 01 | [The RL Problem and Markov Decision Processes](chapters/01-the-rl-problem.md) | Formalize a problem as an MDP; derive and solve the Bellman equations |
| 02 | [Multi-Armed Bandits](chapters/02-multi-armed-bandits.md) | Trade off exploration and exploitation with ε-greedy, UCB and Thompson sampling, and reason about regret |
| 03 | [Dynamic Programming](chapters/03-dynamic-programming.md) | Plan optimally with a known model; prove why it works (contractions, policy improvement); extend it to continuous control with LQR and iLQR |
| 04 | [Monte Carlo Methods](chapters/04-monte-carlo.md) | Learn from complete episodes; evaluate one policy from another's data with importance sampling |
| 05 | [Temporal-Difference Learning](chapters/05-temporal-difference.md) | Use TD(0), SARSA, Q-learning, Expected SARSA and Double Q-learning; connect TD errors to dopamine and animal learning |
| 06 | [n-Step Bootstrapping and Eligibility Traces](chapters/06-n-step-and-eligibility-traces.md) | Move along the spectrum between MC and TD with n-step returns, λ-returns and traces |
| 07 | [Planning and Learning with Tabular Models](chapters/07-planning-and-learning-tabular.md) | Combine learning and planning with Dyna and prioritized sweeping; search with MCTS |
| 08 | [Value Function Approximation](chapters/08-function-approximation.md) | Generalize with linear features and tile coding; run batch methods (LSTD, fitted Q-iteration, LSPI); understand the deadly triad |
| **Part II** | **Deep reinforcement learning** | |
| 09 | [Deep Q-Networks and Value-Based Deep RL](chapters/09-deep-q-learning.md) | Build DQN, Double, Dueling and distributional variants and know why each was introduced |
| 10 | [Policy Gradient Methods](chapters/10-policy-gradients.md) | Derive the policy gradient theorem; implement REINFORCE, actor-critic and GAE; compare with evolution strategies and random search |
| 11 | [Natural Gradients, Trust Regions, TRPO and PPO](chapters/11-trust-regions-and-ppo.md) | Implement PPO and know which implementation details matter |
| 12 | [Off-Policy Actor-Critic: DDPG, TD3, SAC](chapters/12-continuous-control-actor-critic.md) | Solve continuous control with deterministic and maximum-entropy actor-critics; see RL as inference (REPS, MPO) |
| 13 | [Model-Based Deep RL, World Models, AlphaZero/MuZero](chapters/13-model-based-rl.md) | Learn and plan with models: PETS, MBPO, Dreamer, AlphaZero, MuZero |
| **Part III** | **Advanced topics** | |
| 14 | [Exploration](chapters/14-exploration.md) | Explore deeply with optimism, posterior sampling, counts, curiosity and RND |
| 15 | [Beyond the Standard MDP](chapters/15-beyond-mdps.md) | Handle partial observability, goals (HER), hierarchy (options), meta-RL, multiple objectives, and generalization to unseen environments |
| 16 | [Imitation Learning, Inverse RL and Offline RL](chapters/16-offline-rl-and-imitation.md) | Learn from demonstrations and fixed datasets: BC, DAgger, GAIL, CQL, IQL, diffusion policies and off-policy evaluation |
| 17 | [Multi-Agent RL and Games](chapters/17-multi-agent-rl.md) | Reason about equilibria; use self-play, CTDE, QMIX and CFR |
| 18 | [RL for Language Models](chapters/18-rl-for-language-models.md) | Understand and implement RLHF, DPO, GRPO, expert iteration, RL with verifiable rewards and multi-turn agentic RL |
| 19 | [Theory of RL](chapters/19-rl-theory.md) | Read and state convergence, sample-complexity and regret results precisely |
| 20 | [Deep RL in Practice](chapters/20-deep-rl-in-practice.md) | Design rewards, debug agents, evaluate with statistical rigor, and handle safety constraints and robustness |

Reference material: [NOTATION.md](NOTATION.md) (symbols), [GLOSSARY.md](GLOSSARY.md) (terms), [CHEATSHEET.md](CHEATSHEET.md) (algorithms at a glance) and [READING_LIST.md](READING_LIST.md) (books, courses, papers).

## Learning paths

Chapter 00 is a reference. Skim it first, then come back to sections as later chapters point to them.

* **The full course (recommended)**: 01 → 20 in order. Each chapter assumes the earlier ones.
* **Foundations only**: 01 → 08. This is the classical core, roughly Sutton & Barto Part I and II.
* **Practitioner's fast track**: 01 → 05 → 09 → 10 → 11 → 12 → 20. This gets you to working deep-RL agents and the skills to debug them.
* **RL for LLMs track**: 01 → 02 → 05 → 10 → 11 → 16 (§ on off-policy evaluation and offline RL) → 18.
* **Theory track**: 01 → 03 → 05 → 08 → 10 → 11 → 19, with 02 for regret.

## Running the code

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: much smaller CPU-only PyTorch
pip install -r requirements.txt            # no GPU, MuJoCo or Atari needed

python code/ch05_temporal_difference/cliff_walking.py           # full run: reproduces the chapter's numbers and figures
python code/ch05_temporal_difference/cliff_walking.py --quick   # smoke test (seconds), writes no figures
pytest tests/                              # runs every script in --quick mode
```

Each `code/chNN_*/README.md` lists that chapter's scripts, what they show, their runtimes and their headline results. Most full runs take seconds to a few minutes on one CPU core; the heaviest deep-RL experiments take up to about ten minutes.

## How to study with this course

1. **Read actively.** Stop at each derivation and redo it on paper before reading the next line.
2. **Run, then break, the code.** Change a hyperparameter, remove a "trick" (a target network, a baseline, clipping), and predict the result before you run it.
3. **Do the exercises before opening the solutions.** The ★★★ exercises are where most of the learning happens.
4. **Re-implement.** After a chapter, close the code and write the core algorithm again from the pseudocode.
