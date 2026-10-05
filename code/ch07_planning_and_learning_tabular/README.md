# Chapter 07 code: Planning and Learning with Tabular Models

Code for [Chapter 07](../../chapters/07-planning-and-learning-tabular.md). Everything is pure Python 3.11 + NumPy (SciPy for one sparse solve, Matplotlib for figures), single-threaded, with fixed seeds. Run every script from the repository root. `--quick` is a smoke test that writes no figures; a full run writes PNGs to `figures/`.

| script | demonstrates (chapter section) | command | quick | full |
|---|---|---|---|---|
| `maze_env.py` | shared module: Dyna maze, blocking and shortcut mazes, maze refinement (not run directly) | — | — | — |
| `dyna_maze.py` | Dyna-Q with n = 0, 5, 50 planning steps (Sec. 3) | `python code/ch07_planning_and_learning_tabular/dyna_maze.py` | 0.4 s | 3.4 s |
| `changing_mazes.py` | blocking and shortcut mazes; Dyna-Q, Dyna-Q+, action-bonus variant; exact eps-greedy reference rates (Sec. 4); options `--n-planning`, `--alpha`, `--no-figures` | `python code/ch07_planning_and_learning_tabular/changing_mazes.py` | 0.5 s | 9–14 s (n = 50: 50 s; each `--alpha` variant: 11 s) |
| `prioritized_sweeping.py` | prioritized sweeping vs Dyna-Q on refined mazes, with Q0 = 0 and Q0 = 1: value updates, total work (updates + priority computations), real steps, coverage at equal real steps, stall diagnosis (Sec. 5); options `--alpha`, `--no-figures` | `python code/ch07_planning_and_learning_tabular/prioritized_sweeping.py` | 2.3 s | 104–111 s (`--alpha 0.5 --no-figures`: 238 s) |
| `expected_vs_sample.py` | RMS error of sample vs expected updates, theory check (Sec. 6) | `python code/ch07_planning_and_learning_tabular/expected_vs_sample.py` | 0.2 s | 1.8 s |
| `trajectory_sampling.py` | on-policy vs uniform update distributions on random MDPs, with paired differences (Sec. 7) | `python code/ch07_planning_and_learning_tabular/trajectory_sampling.py` | 5.2 s | 66–77 s |
| `rtdp_racetrack.py` | RTDP vs value iteration (two sweep orders) on a racetrack; admissible heuristic; update counts on the optimal policy's states (Sec. 8) | `python code/ch07_planning_and_learning_tabular/rtdp_racetrack.py` | 3.1 s | 30 s |
| `mcts_tictactoe.py` | exact negamax solver, flat Monte Carlo, UCT, UCT + transpositions, verified games (Secs. 10–12) | `python code/ch07_planning_and_learning_tabular/mcts_tictactoe.py` | 2.1 s | 91 s |
| `exercise_solutions.py` | numerical checks and reference code for Exercises 3, 5, 12, 13 (symmetric transpositions, PUCT) | `python code/ch07_planning_and_learning_tabular/exercise_solutions.py` | 1.6 s | 8 s |

Runtimes were measured on one core of a shared 4-CPU machine while other jobs were running, so expect some variation (the same full run of `changing_mazes.py` took 9 s and 14 s on two occasions).

## Headline results (full runs, default seeds)

- **Dyna maze** (30 runs): episodes until within 10% of the final plateau: 27 for n = 0, 7 for n = 5, 3 for n = 50. Episode 1 is a uniform random walk for every n: the exact expected length is 868.7 steps, and the pooled empirical mean over 90 runs is 869.4.
- **Changing mazes** (n = 10, alpha = 0.5, 30 runs):
  - Only Dyna-Q+ found the shortcut: 81.4 goals in the final 1000 steps vs 54.8 for Dyna-Q and 54.0 for the action-bonus variant. The exact references for an eps-greedy agent (eps = 0.1) with an optimal greedy policy are 88.8 on the shortcut and 55.8 on the old route.
  - In the blocking maze, Dyna-Q had no goal at all in the final 1000 steps in 15/30 runs (Dyna-Q+: 0/30; action bonus: 20/30). With `--n-planning 50`, Dyna-Q was stuck in 25/30 runs.
  - Sensitivity to alpha: with `--alpha 0.1`, Dyna-Q+ reached the goal only 11.1 times before the blocking change (Dyna-Q: 42.5) and was stuck in 10/30 blocking runs; with `--alpha 1.0`, Dyna-Q was stuck in 15/30, Dyna-Q+ in 0/30, the action-bonus variant in 2/30.
  - If the agent stands on a cell that becomes a wall when the maze changes, it is moved back to the start.
- **Prioritized sweeping** (n = 5, alpha = 1, mazes of 47–1,692 states, 10 runs each; ratios are Dyna-Q median / PS median):
  - Q0 = 0: PS needed 36–79x fewer Q-value updates (37, 48, 37, –, 36, 79). Counting its priority computations as well, its total work was 1.7–4.9x lower where the median run was solved, but 2.4x higher at 423 states. 10 of 60 PS runs never became optimal within 10^6 real steps; in every one the greedy path was the shortest path *inside the learned model*, two moves longer than optimal, and the queue was empty on more than 99% of steps. In real steps PS was worse at the four smaller sizes and slightly better at the two largest. At equal real experience, Dyna-Q had tried more state-action pairs than PS in 31 of 35 comparable runs.
  - Q0 = 1: every run was solved. PS made 2.8–3.1x fewer updates (3.8–4.8 per real step out of a budget of 5) but, with about 4 priority computations per update, did 1.65–1.85x *more* total work. It needed 1.9–2.2x fewer real steps.
  - `--alpha 0.5` (Q0 = 0): 5 of 60 PS runs stalled, 1 because of the un-requeued (1 - alpha) residual and 4 because of incomplete models; the update ratios fell to 1.8–7.8.
- **Expected vs sample**: the simulated RMS error matches sqrt((b-1)/(bt)), e.g. 0.099 vs 0.100 for b = 1000 after t = 100 sample updates.
- **Trajectory sampling**: on-policy sampling was ahead early; uniform sweeps stayed ahead from 4,000 / 1,300 / 1,000 updates for b = 1 / 3 / 10 with 1,000 states, and from 70,000 updates with 10,000 states. The paired differences are precise for b = 3, 10 (SE at most 0.04) but not for b = 1 (SE 0.06–0.48), so the b = 1 crossover points are uncertain.
- **RTDP** (racetrack, 3,877 reachable states, optimal expected time 13.331 steps; one optimal policy visits 1,946 states, a lower bound on the relevant set):
  - RTDP from V0 = 0 came within 1% of optimal after 73.5k updates (mean of 5 runs). Value iteration met its stopping rule after 155k updates in forward sweep order but only 58k in backward order, and its greedy policy was exactly optimal after 58k (forward) or 35k (backward) updates.
  - Starting RTDP from an admissible heuristic halved the updates needed to get within 10% of optimal, but only 1/5 runs got within 0.1% in 10,000 trials (V0 = 0: 5/5). After 10,000 trials, 8.2% of the optimal policy's states had never been updated from the heuristic start (0.0% from V0 = 0).
- **MCTS** (tic-tac-toe):
  - The solver reproduces 5,478 positions and 255,168 games, and gives the empty board the value draw.
  - UCT reached 100% minimax-optimal moves on 1,000 non-trivial positions at 3,000 simulations (1,000 with transpositions). Flat Monte Carlo plateaued near its exact infinite-budget limit of 95.0%.
  - With 3,000 simulations per move, UCT lost 0 of 500 games against random and perfect players, drew all 50 self-play games, and played 2,192 of 2,192 moves minimax-optimally.
