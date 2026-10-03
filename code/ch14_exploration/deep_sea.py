"""Deep exploration on DeepSea: dithering fails exponentially, directed exploration scales.

Chapter 14, Sections 1-5 (results reported in Sections 5.1 and 5.2).

DeepSea(N) (Osband et al., 2019) is an N x N grid.  The agent starts in the top-left cell
(row 0, column 0).  Every step moves it one row DOWN and one column LEFT or RIGHT, so an
episode lasts exactly N steps and the row index is the time step h.  Moving right costs
0.01/N; moving right from the bottom-right cell (row N-1, column N-1) pays +1 (the
"treasure").  To get it the agent must choose "right" N times in a row; every other policy
earns <= 0.  The optimal return is 1 - 0.01 = 0.99, and the best "safe" return is 0.

Two details make it a hard exploration problem:
  * the small cost makes every *local* signal say "go left": an agent that learns from what
    it has seen, without seeking out the unknown, learns to avoid the treasure;
  * the meaning of the two action indices is shuffled independently in every cell (a fixed
    random mask), so no fixed tie-breaking rule or action bias can solve it by luck.
A uniformly random policy reaches the treasure with probability 2^-N per episode.

Agents (Chapter 14 section in brackets):
  uniform random        eps = 1; only the first-success time is recorded          [Sec. 1.3]
  eps-greedy Q          Q-learning, Q0 = 0, eps = 0.1                              [Sec. 2]
  UCB-Q                 optimistic Q-learning with bonus beta/sqrt(n), step size
                        (H+1)/(H+n) as in Jin et al. (2018)                         [Sec. 3.4]
  PSRL                  posterior sampling: Gaussian rewards, Dirichlet transitions [Sec. 4.1]
  RLSVI                 randomized least-squares value iteration, tabular          [Sec. 4.2]
  Ensemble+prior        K Q-tables, random prior per table, every table trained on
                        every transition, one table followed per episode          [Sec. 4.3]
Ablations of the ensemble:
  Bootstrapped+prior (mask 0.5)   each transition reaches each table with prob. 0.5
  Bootstrapped, no prior (mask 0.5)   Bootstrapped DQN without priors: tables start at 0,
                        diversity comes only from the bootstrap masks (the real test of
                        Osband et al.'s "no data, no diversity" point)
  Ensemble, no prior (mask 1)   tables start at 0 and see identical data, so all members
                        are bit-identical: this is just greedy zero-initialised Q-learning
  Ensemble+prior, per-step   a new table is drawn at EVERY step (no commitment)
A final sweep varies the UCB-Q bonus scale beta at N = 20 (Sec. 5.2).

All Q-learning agents store the episode and apply their updates at the end of the episode
in reverse time order.  In DeepSea every state is visited at most once per episode, so this
is an ordinary sequence of Q-learning updates; reversing the order only lets information
travel the whole path in one episode instead of one step per episode.

For every N and seed we record T_first (first episode that reaches the treasure) and
T_solve (first episode at which >= 10 of the last 20 episodes reached it).  Runs stop at
T_solve or at a cap of `--cap` episodes; capped runs are reported as censored.

Run:  python code/ch14_exploration/deep_sea.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ---------------------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------------------
class DeepSea:
    """DeepSea(N) with a randomized action mapping (bsuite-style).

    States are indexed s = row * N + col.  `step` returns (next_state, reward, terminated);
    next_state is -1 after the last row.  There is no time-limit truncation: episodes end
    by *termination* after exactly N steps, so no bootstrapping happens at the end.
    """

    def __init__(self, N: int, rng: np.random.Generator, move_cost: float = 0.01):
        self.N = N
        self.n_states = N * N
        # right_action[row, col] in {0,1} is the action index that means "right" there.
        self.right_action = rng.integers(0, 2, size=(N, N))
        self.cost = move_cost / N
        self.row = self.col = 0

    def reset(self) -> int:
        self.row, self.col = 0, 0
        return 0

    def step(self, a: int):
        N, row, col = self.N, self.row, self.col
        reward = 0.0
        if a == self.right_action[row, col]:
            reward -= self.cost
            if row == N - 1 and col == N - 1:
                reward += 1.0                      # the treasure
            col = min(col + 1, N - 1)
        else:
            col = max(col - 1, 0)
        row += 1
        self.row, self.col = row, col
        terminated = row == N
        return (-1 if terminated else row * N + col), reward, terminated


def argmax_random(q: np.ndarray, rng: np.random.Generator) -> int:
    """argmax with uniformly random tie-breaking (a fixed tie rule biases exploration)."""
    best = np.flatnonzero(q == q.max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))


# ---------------------------------------------------------------------------------------
# Agents.  Interface: begin_episode(rng); act(s, rng) -> a; end_episode(traj, rng)
# traj is a list of (s, a, r, s_next, terminated).
# ---------------------------------------------------------------------------------------
class EpsGreedyQ:
    """Q-learning with eps-greedy behaviour (undirected exploration, Section 2)."""

    name = "eps-greedy Q"

    def __init__(self, N, eps=0.1):
        self.N, self.H, self.eps = N, N, eps
        self.Q = np.zeros((N * N, 2))
        self.n = np.zeros((N * N, 2))

    def begin_episode(self, rng):
        pass

    def act(self, s, rng):
        if rng.random() < self.eps:
            return int(rng.integers(2))
        return argmax_random(self.Q[s], rng)

    def end_episode(self, traj, rng):
        H = self.H
        for s, a, r, s2, term in reversed(traj):
            self.n[s, a] += 1
            alpha = (H + 1) / (H + self.n[s, a])           # Jin et al. step size (Sec. 3.4)
            target = r if term else r + self.Q[s2].max()   # no bootstrapping at termination
            self.Q[s, a] += alpha * (target - self.Q[s, a])


class UCBQ:
    """Optimistic Q-learning with a count bonus (Section 3.4).

    Q0 = v_max (an upper bound on any return), alpha_n = (H+1)/(H+n), and the target
    r + beta/sqrt(n) + min(v_max, max_a' Q(s', a')).  Jin et al. (2018) use the bonus
    c*sqrt(H^3 * iota / n); we keep the 1/sqrt(n) shape and fold the constants into beta.
    """

    name = "UCB-Q (count bonus)"

    def __init__(self, N, beta=0.1, v_max=1.0):
        self.N, self.H, self.beta, self.v_max = N, N, beta, v_max
        self.Q = np.full((N * N, 2), v_max)                # optimistic initialisation
        self.n = np.zeros((N * N, 2))

    def begin_episode(self, rng):
        pass

    def act(self, s, rng):
        return argmax_random(self.Q[s], rng)              # greedy w.r.t. optimistic Q

    def end_episode(self, traj, rng):
        H = self.H
        for s, a, r, s2, term in reversed(traj):
            self.n[s, a] += 1
            t = self.n[s, a]
            alpha = (H + 1) / (H + t)
            bonus = self.beta / np.sqrt(t)
            v_next = 0.0 if term else min(self.v_max, self.Q[s2].max())
            self.Q[s, a] += alpha * (r + bonus + v_next - self.Q[s, a])


class PSRL:
    """Posterior sampling for RL (Section 4.1), exploiting only the layered structure.

    The agent knows that row h leads to row h+1 (it is a finite-horizon problem) but not
    *which* column, nor the rewards.  Prior: mean reward of each (s, a) ~ N(0, sigma0^2),
    observations ~ N(mean, sigma_r^2); next column ~ Categorical(p) with p ~ Dirichlet(a0).
    Each episode: sample one MDP from the posterior, solve it exactly by backward
    induction, and follow its optimal policy for the whole episode.
    """

    name = "PSRL"

    def __init__(self, N, sigma0=1.0, sigma_r=0.1, dir_prior=None):
        self.N = N
        self.sigma0, self.sigma_r = sigma0, sigma_r
        self.a0 = 1.0 / N if dir_prior is None else dir_prior   # total prior mass 1
        self.n = np.zeros((N, N, 2))                    # visit counts per (row, col, a)
        self.r_sum = np.zeros((N, N, 2))
        self.trans = np.zeros((N - 1, N, 2, N))         # next-column counts (rows 0..N-2)
        self.Q = None

    def begin_episode(self, rng):
        N = self.N
        prec = 1.0 / self.sigma0 ** 2 + self.n / self.sigma_r ** 2      # posterior precision
        mean = (self.r_sum / self.sigma_r ** 2) / prec
        R = mean + rng.standard_normal((N, N, 2)) / np.sqrt(prec)
        G = rng.gamma(self.a0 + self.trans)            # Dirichlet sample = normalised gammas
        P = G / G.sum(axis=-1, keepdims=True)
        Q = np.empty((N, N, 2))
        Q[N - 1] = R[N - 1]                             # last row: next state is terminal
        for h in range(N - 2, -1, -1):
            V_next = Q[h + 1].max(axis=1)               # (N,)
            Q[h] = R[h] + P[h] @ V_next
        self.Q = Q

    def act(self, s, rng):
        row, col = divmod(s, self.N)
        return argmax_random(self.Q[row, col], rng)

    def end_episode(self, traj, rng):
        N = self.N
        for s, a, r, s2, term in traj:
            row, col = divmod(s, N)
            self.n[row, col, a] += 1
            self.r_sum[row, col, a] += r
            if not term:
                self.trans[row, col, a, s2 % N] += 1


class RLSVI:
    """Randomized least-squares value iteration, tabular version (Section 4.2).

    With one-hot features, Bayesian linear regression of the target
    y = r + max_a' Q~_{h+1}(s', a') onto Q_h(s, a), prior N(0, sigma_p^2) and noise sigma^2,
    has posterior N(mu, v) with v = 1/(1/sigma_p^2 + n/sigma^2), mu = v * n * ybar / sigma^2.
    RLSVI samples Q~_h from it, backwards in h, once per episode; ybar uses the empirical
    reward mean and empirical next-state frequencies.
    """

    name = "RLSVI"

    def __init__(self, N, sigma_p=1.0, sigma=0.1):
        self.N, self.sigma_p, self.sigma = N, sigma_p, sigma
        self.n = np.zeros((N, N, 2))
        self.r_sum = np.zeros((N, N, 2))
        self.trans = np.zeros((N - 1, N, 2, N))
        self.Q = None

    def begin_episode(self, rng):
        N = self.N
        v = 1.0 / (1.0 / self.sigma_p ** 2 + self.n / self.sigma ** 2)
        noise = rng.standard_normal((N, N, 2)) * np.sqrt(v)
        Q = np.empty((N, N, 2))
        for h in range(N - 1, -1, -1):
            ysum = self.r_sum[h].copy()                 # sum over visits of r + V~(s')
            if h < N - 1:
                ysum += self.trans[h] @ Q[h + 1].max(axis=1)
            Q[h] = v[h] * ysum / self.sigma ** 2 + noise[h]
        self.Q = Q

    act = PSRL.act

    def end_episode(self, traj, rng):
        PSRL.end_episode(self, traj, rng)


class EnsembleQ:
    """Ensemble Q-learning with randomized prior functions, tabular (Section 4.3).

    K Q-tables.  Table k is Q_k = f_k + p_k with a fixed random prior p_k ~ N(0, prior^2)
    and a learned part f_k (initialised to 0), i.e. Q_k starts at the prior.  Updates use
    the same step size (H+1)/(H+n) as the other Q-learners, so alpha = 1 on the first
    visit: at visited pairs Q_k fits the data (as a flexible network would), and the prior
    survives only where table k has no data.  That is exactly the role of the prior in
    Osband et al. (2018).  A transition reaches table k with probability `mask_p`
    (bootstrap mask; 1.0 = plain ensemble).  One table is drawn per episode
    (commitment); `per_step=True` redraws it at every step (the ablation).
    """

    def __init__(self, N, K=10, prior_scale=1.0, mask_p=1.0, per_step=False):
        self.N, self.H, self.K, self.mask_p, self.per_step = N, N, K, mask_p, per_step
        self.prior_scale = prior_scale
        self.Q = None
        self.n = np.zeros((K, N * N, 2))
        self.k = 0
        if prior_scale == 0:
            self.name = ("Bootstrapped, no prior (mask 0.5)" if mask_p < 1
                         else "Ensemble, no prior (mask 1)")
        elif per_step:
            self.name = "Ensemble+prior, per-step"
        elif mask_p < 1:
            self.name = "Bootstrapped+prior (mask 0.5)"
        else:
            self.name = "Ensemble+prior"

    def _init(self, rng):
        self.Q = self.prior_scale * rng.standard_normal((self.K, self.N * self.N, 2))

    def begin_episode(self, rng):
        if self.Q is None:
            self._init(rng)
        self.k = int(rng.integers(self.K))

    def act(self, s, rng):
        if self.per_step:
            self.k = int(rng.integers(self.K))
        return argmax_random(self.Q[self.k, s], rng)

    def end_episode(self, traj, rng):
        ks = np.arange(self.K)
        for s, a, r, s2, term in reversed(traj):
            mask = rng.random(self.K) < self.mask_p
            if not mask.any():
                continue
            target = np.full(self.K, r) if term else r + self.Q[:, s2, :].max(axis=1)
            km = ks[mask]
            self.n[km, s, a] += 1
            alpha = (self.H + 1) / (self.H + self.n[km, s, a])
            self.Q[km, s, a] += alpha * (target[km] - self.Q[km, s, a])


class UniformRandom(EpsGreedyQ):
    """eps = 1: the best case for any dithering scheme.  T_first ~ Geometric(2^-N)."""

    name = "uniform random"
    stop_at_first = True          # it never learns, so only T_first is meaningful

    def __init__(self, N):
        super().__init__(N, eps=1.0)

    def end_episode(self, traj, rng):
        pass


AGENTS = {
    "uniform random": lambda N: UniformRandom(N),
    "eps-greedy Q": lambda N: EpsGreedyQ(N),
    "UCB-Q (count bonus)": lambda N: UCBQ(N),
    "PSRL": lambda N: PSRL(N),
    "RLSVI": lambda N: RLSVI(N),
    "Ensemble+prior": lambda N: EnsembleQ(N),
    "Bootstrapped+prior (mask 0.5)": lambda N: EnsembleQ(N, mask_p=0.5),
    "Bootstrapped, no prior (mask 0.5)": lambda N: EnsembleQ(N, prior_scale=0.0, mask_p=0.5),
    "Ensemble, no prior (mask 1)": lambda N: EnsembleQ(N, prior_scale=0.0),
    "Ensemble+prior, per-step": lambda N: EnsembleQ(N, per_step=True),
}
# PSRL samples N^3 Dirichlet variables per episode; beyond N = 30 a full run would exceed
# the 3-minute budget, so it is not scaled further.
MAX_N = {"PSRL": 30}


# ---------------------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------------------
def run_episode(env, agent, rng):
    """One episode; returns True if the treasure was reached."""
    agent.begin_episode(rng)
    s = env.reset()
    traj, found = [], False
    while True:
        a = agent.act(s, rng)
        s2, r, term = env.step(a)
        traj.append((s, a, r, s2, term))
        found |= r > 0.5
        if term:
            break
        s = s2
    agent.end_episode(traj, rng)
    return found


def run_one(make_agent, N, seed, cap, window=20, need=10):
    """Run one agent on DeepSea(N) until T_solve or `cap` episodes.

    T_first = first episode that reaches the treasure; T_solve = first episode at which
    at least `need` of the last `window` episodes reached it.  None = censored at `cap`.
    """
    rng = np.random.default_rng(seed)
    env = DeepSea(N, np.random.default_rng(10_000 + seed))   # env mask depends on seed only
    agent = make_agent(N)
    successes = np.zeros(cap, dtype=bool)
    t_first = t_solve = None
    for ep in range(cap):
        successes[ep] = run_episode(env, agent, rng)
        if successes[ep] and t_first is None:
            t_first = ep + 1
            if getattr(agent, "stop_at_first", False):
                break
        if ep + 1 >= window and successes[ep + 1 - window: ep + 1].sum() >= need:
            t_solve = ep + 1
            break
    return {"t_first": t_first, "t_solve": t_solve}


def summarize(vals):
    """Median over seeds, treating censored runs (None) as +inf; and the censored count."""
    arr = np.array([v if v is not None else np.inf for v in vals], dtype=float)
    return np.median(arr), int(np.sum(np.isinf(arr)))


def learning_curves(N, seeds, episodes, names):
    """Cumulative number of treasure episodes, per agent and seed, at a fixed N."""
    out = {}
    for name in names:
        curves = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            env = DeepSea(N, np.random.default_rng(10_000 + seed))
            agent = AGENTS[name](N)
            succ = [run_episode(env, agent, rng) for _ in range(episodes)]
            curves.append(np.cumsum(succ))
        out[name] = np.array(curves)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--cap", type=int, default=4000, help="max episodes per run")
    args = parser.parse_args()

    if args.quick:
        sizes, seeds, cap = [4, 8, 12], list(range(2)), 300
        curve_N, curve_eps = 10, 150
        betas, beta_N = [0.0, 0.1, 1.0], 8
    else:
        sizes = [4, 6, 8, 10, 12, 16, 20, 25, 30, 40]
        seeds, cap = list(range(args.seeds)), args.cap
        curve_N, curve_eps = 20, 600
        betas, beta_N = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0], 20

    names = list(AGENTS)
    print("DeepSea exploration experiment (Chapter 14)")
    print(f"sizes N = {sizes}; seeds = {seeds}; episode cap = {cap}")
    print("T_first = first episode reaching the treasure; "
          "T_solve = first episode with >= 10 treasure hits among the last 20 episodes")
    print("hyperparameters: eps=0.1 | UCB-Q beta=0.1, v_max=1 | PSRL sigma0=1, sigma_r=0.1, "
          "Dirichlet a0=1/N | RLSVI sigma_p=1, sigma=0.1 | Ensembles K=10, prior sd 1, "
          "mask 1.0 (0.5 for 'Bootstrapped')")
    print("all Q-learners: step size (H+1)/(H+n), H=N; updates at episode end, reverse order")
    t0 = time.time()

    results = {name: {} for name in names}
    for name in names:
        t_name = time.time()
        for N in sizes:
            if N > MAX_N.get(name, np.inf):
                continue
            runs = [run_one(AGENTS[name], N, seed, cap) for seed in seeds]
            results[name][N] = runs
            med_first, cens_first = summarize([r["t_first"] for r in runs])
            med_solve, cens_solve = summarize([r["t_solve"] for r in runs])
            print(f"  {name:34s} N={N:3d}  median T_first={med_first:7.0f} "
                  f"(censored {cens_first}/{len(seeds)})  median T_solve={med_solve:7.0f} "
                  f"(censored {cens_solve}/{len(seeds)})")
            if not np.isfinite(med_first):
                # The median run never found the treasure within the cap: larger N will
                # not be better, so stop scaling this agent to save compute.
                print(f"  {name:34s} stopped scaling: median run never found the "
                      f"treasure at N={N}")
                break
        print(f"  [{name}: {time.time() - t_name:.1f} s]")

    # Sanity check for the "no prior, mask 1" ablation: with zero initialisation and no
    # masks every member receives identical updates, so the ensemble collapses to one table.
    chk_N = 6
    rng_chk = np.random.default_rng(0)
    env_chk = DeepSea(chk_N, np.random.default_rng(10_000))
    ens = EnsembleQ(chk_N, prior_scale=0.0)
    for _ in range(200):
        run_episode(env_chk, ens, rng_chk)
    print(f"\nCheck: 'Ensemble, no prior (mask 1)' after 200 episodes at N={chk_N}: "
          f"max_k |Q_k - Q_0| = {np.abs(ens.Q - ens.Q[0]).max():.1e} (identical members = "
          f"greedy Q-learning)")

    print(f"\nUCB-Q bonus-scale sweep at N={beta_N} ({len(seeds)} seeds, cap {cap})")
    beta_res = {}
    for beta in betas:
        runs = [run_one(lambda N, b=beta: UCBQ(N, beta=b), beta_N, seed, cap) for seed in seeds]
        beta_res[beta] = runs
        med, cens = summarize([r["t_solve"] for r in runs])
        print(f"  beta={beta:5.2f}  median T_solve={med:7.0f}  (censored {cens}/{len(seeds)})")

    print(f"\nLearning curves at N={curve_N}, {curve_eps} episodes, {len(seeds)} seeds")
    curve_names = ["eps-greedy Q", "UCB-Q (count bonus)", "PSRL", "RLSVI", "Ensemble+prior",
                   "Ensemble+prior, per-step"]
    curves = learning_curves(curve_N, seeds, curve_eps, curve_names)
    for name in curve_names:
        tot = curves[name][:, -1]
        print(f"  {name:34s} treasure episodes in {curve_eps}: mean {tot.mean():6.1f} "
              f"(min {tot.min():.0f}, max {tot.max():.0f})")

    print("\nFitted growth exponent p in T_solve ~ c N^p (least squares on log-log, "
          "N >= 10, finite medians only)")
    for name in names:
        pts = [(N, summarize([r["t_solve"] for r in runs])[0])
               for N, runs in results[name].items() if N >= 10]
        pts = [(N, t) for N, t in pts if np.isfinite(t)]
        if len(pts) >= 3:
            x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
            slope = np.polyfit(x, y, 1)[0]
            print(f"  {name:34s} p = {slope:.2f}  (N = {pts[0][0]}..{pts[-1][0]})")

    print("\nSummary: largest N with a finite median T_solve (cap "
          f"{cap}), and median T_solve there")
    for name in names:
        solved = [(N, summarize([r["t_solve"] for r in runs])[0])
                  for N, runs in results[name].items()
                  if np.isfinite(summarize([r["t_solve"] for r in runs])[0])]
        msg = f"N={solved[-1][0]} (median T_solve {solved[-1][1]:.0f})" if solved else "none"
        print(f"  {name:34s} {msg}")
    print(f"Total time {time.time() - t0:.1f} s")

    if args.quick:
        return
    make_figures(results, sizes, seeds, cap, curves, curve_names, curve_N, curve_eps,
                 beta_res, beta_N)


STYLES = {
    "uniform random": ("k", "*", "--"),
    "eps-greedy Q": ("tab:red", "o", "-"),
    "UCB-Q (count bonus)": ("tab:blue", "s", "-"),
    "PSRL": ("tab:green", "^", "-"),
    "RLSVI": ("tab:olive", "v", "-"),
    "Ensemble+prior": ("tab:purple", "D", "-"),
    "Bootstrapped+prior (mask 0.5)": ("tab:pink", "d", "--"),
    "Bootstrapped, no prior (mask 0.5)": ("tab:brown", "X", "--"),
    "Ensemble, no prior (mask 1)": ("tab:gray", "x", "--"),
    "Ensemble+prior, per-step": ("tab:orange", "P", "--"),
}


def make_figures(results, sizes, seeds, cap, curves, curve_names, curve_N, curve_eps,
                 beta_res, beta_N):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    Ns = np.arange(sizes[0], sizes[-1] + 1)
    panels = [
        ("Main methods", ["uniform random", "eps-greedy Q", "UCB-Q (count bonus)", "PSRL",
                          "RLSVI", "Ensemble+prior"]),
        ("Ensemble ablations", ["Ensemble+prior", "Bootstrapped+prior (mask 0.5)",
                                "Ensemble+prior, per-step", "Bootstrapped, no prior (mask 0.5)",
                                "Ensemble, no prior (mask 1)"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
    for ax, (title, members) in zip(axes, panels):
        # Uniformly random policy: T_first ~ Geometric(2^-N), median ~ ln(2) 2^N.
        ax.plot(Ns, np.log(2) * 2.0 ** Ns, color="k", lw=1, ls=":",
                label=r"theory: uniform random, median $\approx 0.69\cdot 2^N$")
        for name in members:
            col, mk, ls = STYLES[name]
            key = "t_first" if name == "uniform random" else "t_solve"
            xs, med, lo, hi, cens_x = [], [], [], [], []
            for N, runs in results[name].items():
                vals = np.array([r[key] if r[key] is not None else np.inf for r in runs])
                if np.isinf(np.median(vals)):
                    cens_x.append(N)
                    continue
                xs.append(N)
                med.append(np.median(vals))
                lo.append(vals.min())
                hi.append(min(vals.max(), cap))
            label = name + (" (first find)" if key == "t_first" else "")
            if xs:
                med, lo, hi = map(np.array, (med, lo, hi))
                ax.errorbar(xs, med, yerr=[med - lo, hi - med], color=col, marker=mk, ls=ls,
                            capsize=2, ms=5, label=label)
            if cens_x:
                ax.scatter(cens_x, [cap * 1.6] * len(cens_x), color=col, marker=mk, s=30,
                           label=None if xs else label)
        ax.axhline(cap, color="k", lw=0.8)
        ax.text(sizes[0] + 0.5, cap * 2.4,
                f"above the line: median run censored at {cap} episodes", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(1, cap * 4)
        ax.set_xlabel("DeepSea size N (= episode length)")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7.5, loc="lower right")
    axes[0].set_ylabel(f"episodes until solved (median of {len(seeds)} seeds; bars = min-max)")
    fig.suptitle(r"DeepSea: dithering needs $\sim 2^N$ episodes, deep exploration needs "
                 r"poly($N$)")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "deep_sea_scaling.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"saved {path}")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ep_axis = np.arange(1, curve_eps + 1)
    for name in curve_names:
        col, _, ls = STYLES[name]
        c = curves[name]
        ax.plot(ep_axis, c.mean(0), color=col, ls=ls, label=name)
        ax.fill_between(ep_axis, c.min(0), c.max(0), color=col, alpha=0.12)
    ax.set_xlabel("episode")
    ax.set_ylabel("cumulative episodes reaching the treasure")
    ax.set_title(f"DeepSea N={curve_N}: mean over {len(seeds)} seeds (band = min-max)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    bs = list(beta_res)
    for i, b in enumerate(bs):
        vals = [r["t_solve"] if r["t_solve"] is not None else cap * 1.6 for r in beta_res[b]]
        ax.scatter([i] * len(vals), vals, color="tab:blue", alpha=0.6)
        ax.plot([i - 0.25, i + 0.25], [np.median(vals)] * 2, color="k")
    ax.axhline(cap, color="k", lw=0.8)
    ax.set_xticks(range(len(bs)))
    ax.set_xticklabels([str(b) for b in bs])
    ax.set_yscale("log")
    ax.set_xlabel(r"bonus scale $\beta$ in $\beta/\sqrt{n}$")
    ax.set_ylabel("episodes until solved")
    ax.set_title(f"UCB-Q at N={beta_N}: bigger bonus = slower\n(points above line: censored)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "deep_sea_curves_and_bonus.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
