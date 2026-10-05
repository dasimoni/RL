# Chapter 17 code — Multi-Agent RL and Games

Companion code for [Chapter 17: Multi-Agent RL and Games](../../chapters/17-multi-agent-rl.md).

Every script runs from the repository root, fixes its seeds, prints its settings and a results summary, and accepts `--quick` (a smoke test that writes **no** figures). Full mode writes figures to `figures/`. Everything is NumPy/SciPy except `value_decomposition.py` and `two_step_game.py`, which use tiny PyTorch networks (`torch.set_num_threads(1)`). Algorithms are written from scratch: no game-theory or RL libraries.

```bash
python code/ch17_multi_agent_rl/games.py                     # library + self-test: solution concepts (§2-3)
python code/ch17_multi_agent_rl/no_regret_dynamics.py        # FP, RM, RM+, Hedge, OMWU in self-play (§5)
python code/ch17_multi_agent_rl/markov_soccer.py             # Shapley VI, minimax-Q, IQL on grid soccer (§4)
python code/ch17_multi_agent_rl/cooperative_matrix_games.py  # IQL vs centralised learners, climbing/penalty games (§6)
python code/ch17_multi_agent_rl/value_decomposition.py       # VDN / QMIX / OW-QMIX fits vs the exact best monotonic fit (§7)
python code/ch17_multi_agent_rl/two_step_game.py             # IQL / VDN / QMIX with TD targets on the QMIX two-step game (§7.5)
python code/ch17_multi_agent_rl/credit_assignment_pg.py      # COMA counterfactual baseline vs N agents (§8)
python code/ch17_multi_agent_rl/psro_kuhn.py                 # self-play vs fictitious play vs PSRO-Nash on Kuhn poker (§9)
python code/ch17_multi_agent_rl/lever_game.py                # zero-shot coordination: self-play vs other-play, cross-play matrices (§9.6)
python code/ch17_multi_agent_rl/kuhn.py                      # Kuhn poker library + sequence-form LP self-test (§10)
python code/ch17_multi_agent_rl/cfr_kuhn.py                  # CFR, CFR+, chance-sampled MCCFR on Kuhn poker (§10)
python code/ch17_multi_agent_rl/lewis_signaling.py           # emergent communication (§11)
python code/ch17_multi_agent_rl/exercise_solutions.py        # numerical checks for Exercises 17.6-17.15
# add --quick to any of them for a smoke test
```

| Script | Demonstrates (chapter section) | Quick | Full | Headline results (full mode) |
|---|---|---|---|---|
| `games.py` | game catalogue, best responses, NashConv, zero-sum LP + fast tableau-simplex solver, support enumeration, CE/CCE LPs, Pareto (§2–3) | 0.9 s | 3.9 s | pivot solver = LP to 2e-15 on 500 random games (0.29 vs 4.8 ms/game); weighted RPS Nash (1/4, 1/2, 1/4); chicken: Nash payoffs (2,7), (7,2), (4.67,4.67), max-welfare CE (5.25, 5.25); RPS off-diagonal uniform: CCE gap 0, CE gap 0.5 |
| `no_regret_dynamics.py` | self-play with full-information feedback, T = 10^5 (§5.7); FP counts its starting beliefs once and plays a pure best response from round 1 (Alg. 5.1) | 2.6 s | 120 s | NashConv of averages ∝ t^-0.5 (FP, RM, RM+, Hedge) and t^-1.0 (OMWU); last iterates: FP 2.0, RM 1.7–2.0, RM+ ≈ 1, OMWU < 2e-15; Shapley's game: CCE gap 8.5e-5 (RM), 9.0e-5 (FP), fitted slope −0.88/−0.89 over t ∈ [10^3, 10^5], but NashConv of marginals 0.28–0.29; payoffs 0.44–0.56 per player vs Nash 1/3 |
| `markov_soccer.py` | 2×4 grid soccer (112 states): Shapley VI, minimax-Q (step 1/n(s,a,o)^0.6, one shared table), IQL (step 1/n(s,a^i)^0.6 from each agent's own counts), exact best-response evaluation (§4.4) | 1.7 s | 115 s | VI converges in 52 sweeps; kick-off stage game mixes (A stays 0.372, B blocks 0.378), value +0.2316 with the ball; worst-case value after 300k steps: minimax-Q −0.046/−0.032/−0.053, IQL −0.182/−0.182/−0.600 (minimax value 0, uniform random −0.560) |
| `cooperative_matrix_games.py` | central, JAL, IQL-ε, IQL-Boltzmann, hysteretic; 1000 runs × 5000 steps (§6.6) | 2.6 s | 25 s | P(optimal), climbing / penalty / stochastic climbing: central 1/1/1; JAL 0/0/0; IQL-ε 0/0/0; IQL-Boltz 0.13/0.89/0.15; hysteretic 0.89/0.94/0.62 |
| `value_decomposition.py` | fit Q_tot on uniform data, 6 seeds, plus the exact best monotonic least-squares fit (36 orderings, NNLS projection) (§7.5) | 11.5 s | 58 s | climbing: VDN → cc (6/6), QMIX → bc (5/6) or cc (MSE 155.7), best monotonic fit → cc (MSE 110.8), OW-QMIX → aa (6/6); non-monotonic game: QMIX never picks AA (MSE 35.6 = optimum of the ordering it learns), but the best monotonic fit does (MSE 32.0); OW-QMIX always does |
| `two_step_game.py` | two-step game of the QMIX paper with TD targets, replay, target networks; 10 seeds × 1500 episodes, uniform vs ε-greedy data (§7.5) | 10.8 s | 125–136 s | seeds reaching the optimal return 8, uniform / ε-greedy: central 10/10 / 10/10, IQL 0/10 / 1/10, VDN 0/10 / 10/10, QMIX 10/10 / 10/10 |
| `credit_assignment_pg.py` | REINFORCE / value baseline / central critic / COMA, N = 2…64 (§8.3) | 1.2 s | 9.5 s | variance at N = 64: 260.2 / 4.19 / 3.94 / 0 (matches Eq. 8.5); updates to 90% (N = 2 → 64): 44→never / 32→128 / 28→109 / 28→28 |
| `psro_kuhn.py` | PSRO with meta-solvers last / uniform / Nash, exact oracle (§9.2) | 0.8 s | 6.1 s | Nash (double oracle): exploitability 0 at iteration 6, meta value −1/18; naive self-play cycles (0.50, 0.17, 0.67, 1.17); uniform 0.031 after 60 iterations |
| `lever_game.py` | lever game (nine 1.0 levers, one 0.9 lever): independent REINFORCE pair and central Q-learner over joint actions, each trained by self-play and by other-play (fresh random relabelling of the 1.0 levers every game); 1000 runs × 5000 games; cross-play matrices; exact OP values by enumerating all 9! relabellings (§9.6, Exercise 17.16) | 0.8 s | 10.6 s | self-play / cross-play score: REINFORCE SP 0.993 / 0.101, central SP 0.995 / 0.103, REINFORCE OP 0.264 / 0.108, central OP 0.900 / 0.900; runs on the 0.9 lever 6.2% / 5.1% / 20.2% / 100%; OP value of a 1.0 lever 1/9 exactly, of the 0.9 lever 0.9; vs a uniform partner 0.100 vs 0.090 (threshold q* = 0.890) |
| `kuhn.py` | Kuhn poker tree, exact BR/exploitability, mixture → behavioural, sequence-form LP (§10.1–10.2) | 0.8 s | 1.4 s | LP value −0.055556 = −1/18; α-family verified for 7 values of α; uniform pair exploitability 0.458 |
| `cfr_kuhn.py` | vanilla CFR, CFR+, chance-sampled MCCFR, 20k iterations (§10.5) | 1.8 s | 64 s | exploitability of average: CFR 1.6e-3 (slope −0.51), CFR+ 6.7e-6 (slope −0.89); CFR current strategy 0.33; value of CFR+ average −0.055556; α = 0.203 (CFR), 0.225 (CFR+) |
| `lewis_signaling.py` | REINFORCE sender/receiver, 1000 runs × 10,000 games (§11) | 2.7 s | 58 s | perfect protocols: 100% (N=M=2), 98.6% (3), 95.1% (4), 55.1% (8); with M = 2N: 100%, 100%, 99.1% |
| `exercise_solutions.py` | Exercises 17.6–17.8, 17.10, 17.12–17.15 (and the CFR worked example of §10.3) | 3.2 s | 16 s | FP regret/T = 1.000 vs RM 0.0032; GDA radii 204.7 / 1.43 / 8.9e-3; Kuhn deal values → −1/18; distributed Q 100/100/0%; IPD: vs TFT 100% cooperative, two learners 21.4%; two-state Shapley VI: v*(1) = 0.869823, error ratio → 0.323; first CFR iteration: regrets K: (−0.125, 0.125), K:pb (−0.25, 0.25), Q:b (−0.0833, 0.0833), J:p (−0.0417, 0.0417) |

Runtimes are wall-clock times measured on one core of a shared 4-core machine (Python 3.11, NumPy 2.4, SciPy 1.17, PyTorch 2.x CPU); expect variation. Rows whose scripts were not changed in the latest revision keep their earlier measurements.

Figures produced in full mode (`figures/`):

- `nashconv_zero_sum.png`, `simplex_weighted_rps.png`, `shapley_cce.png` (no_regret_dynamics.py)
- `markov_soccer.png`
- `cooperative_matrix_games.png`
- `value_decomposition.png`
- `two_step_game.png`
- `credit_assignment.png`
- `psro_kuhn.png`
- `lever_game.png`
- `cfr_kuhn.png`
- `lewis_signaling.png`

Notes

- `games.py` and `kuhn.py` are libraries imported by the other scripts (run from the repository root as shown; Python puts the script's folder on `sys.path`). They also have a self-test `main`.
- `plot_style.py` is the shared figure style (copied from Chapter 00 so the folder is self-contained).
- Exploitability in the Kuhn scripts is (BR value of player 1 + BR value of player 2) / 2 in chips per hand, i.e. NashConv / 2; `no_regret_dynamics.py` reports NashConv itself.
- `markov_soccer.py` is our own small variant of Littman's (1994) soccer game, not a reproduction of his 4×5 field.
