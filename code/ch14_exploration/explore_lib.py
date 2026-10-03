"""Shared code for the intrinsic-motivation experiments of Chapter 14 (Sections 6-7).

Contents
  GridWorld          a deterministic gridworld built from an ASCII map, with an optional
                     "noisy TV" cell and a 5th action ("press the remote")
  CountBonus         r_int = 1/sqrt(n(s'))                                   (Section 6.1)
  RND                Random Network Distillation novelty bonus (small MLPs)   (Section 7.3)
  ForwardCuriosity   prediction-error curiosity with a tabular forward model  (Sections 7.1-7.2)
  TwoStreamQ         tabular Q-learning with separate extrinsic and intrinsic value tables,
                     acting greedily on Q_E + beta * Q_I (as RND's two value heads)
  train              the training loop shared by rnd_gridworld.py and noisy_tv.py
                     (Algorithm "two-stream Q-learning with an intrinsic bonus", Sec. 7.3)
  greedy_episode     one evaluation episode, greedy on Q_E alone (beta = 0, eps = 0)

This file is a library (it has no main block) and is not run directly.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

FOUR_ROOMS = [
    "#####################",
    "#S........#.........#",
    "#.........#.........#",
    "#.........#.........#",
    "#.........#.........#",
    "#...................#",
    "#.........#.........#",
    "#.........#.........#",
    "#.........#.........#",
    "#.........#.........#",
    "####.###########.####",
    "#.........#.........#",
    "#.........#.........#",
    "#.........#.........#",
    "#.........#.........#",
    "#...................#",
    "#.........#.........#",
    "#.........#.........#",
    "#.........#.........#",
    "#.........#........G#",
    "#####################",
]

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]          # up, down, left, right


class GridWorld:
    """Deterministic gridworld from an ASCII map.

    'S' start, 'G' goal (+1 and termination), 'T' a noisy TV, '#' wall.
    Tabular states: one per free cell, plus (if there is a TV) K "screen" states: while the
    agent stands on the TV cell it observes which of K channels is showing.  Action 4
    ("press the remote") does nothing anywhere except on the TV cell, where it switches to
    a uniformly random channel: the only stochastic transition in the world.
    Episodes are truncated after `horizon` steps (time limit, so bootstrap through it).
    """

    def __init__(self, layout, horizon=100, n_channels=0, rng=None):
        self.layout = [list(row) for row in layout]
        self.h, self.w = len(layout), len(layout[0])
        self.horizon = horizon
        self.free = [(r, c) for r in range(self.h) for c in range(self.w)
                     if self.layout[r][c] != "#"]
        self.cell_id = {rc: i for i, rc in enumerate(self.free)}
        self.n_cells = len(self.free)
        find = lambda ch: next(((r, c) for r, c in self.free if self.layout[r][c] == ch), None)
        self.start, self.goal, self.tv = find("S"), find("G"), find("T")
        self.K = n_channels if self.tv is not None else 0
        self.n_states = self.n_cells + self.K
        self.n_actions = 5 if self.K else 4
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.pos, self.channel, self.t = self.start, 0, 0

    # state index: cell id, or n_cells + channel when on the TV cell
    def state(self):
        if self.K and self.pos == self.tv:
            return self.n_cells + self.channel
        return self.cell_id[self.pos]

    def cell_of_state(self, s):
        return self.free[s] if s < self.n_cells else self.tv

    def features(self, s):
        """Observation vector for the RND networks: one-hot(row) + one-hot(col) + channel."""
        r, c = self.cell_of_state(s)
        x = np.zeros(self.h + self.w + self.K, dtype=np.float32)
        x[r] = 1.0
        x[self.h + c] = 1.0
        if s >= self.n_cells:
            x[self.h + self.w + (s - self.n_cells)] = 1.0
        return x

    def reset(self):
        self.pos, self.t = self.start, 0
        return self.state()

    def step(self, a):
        self.t += 1
        if a < 4:
            r, c = self.pos[0] + MOVES[a][0], self.pos[1] + MOVES[a][1]
            if self.layout[r][c] != "#":
                self.pos = (r, c)
        elif self.K and self.pos == self.tv:
            self.channel = int(self.rng.integers(self.K))     # the noisy TV
        terminated = self.goal is not None and self.pos == self.goal
        reward = 1.0 if terminated else 0.0
        truncated = (not terminated) and self.t >= self.horizon
        return self.state(), reward, terminated, truncated


# ---------------------------------------------------------------------------------------
# Intrinsic reward modules.  Interface:
#   reward(s, a, s2) -> float       (called online, once per transition)
#   end_episode(states)             (train on the episode's next-states, if learned)
# ---------------------------------------------------------------------------------------
class CountBonus:
    """MBIE-EB-style count bonus on next states: r_int = 1/sqrt(n(s')) (Section 6.1)."""

    name = "count bonus"

    def __init__(self, env):
        self.n = np.zeros(env.n_states)

    def reward(self, s, a, s2):
        self.n[s2] += 1
        return 1.0 / np.sqrt(self.n[s2])

    def end_episode(self, states):
        pass


class ForwardCuriosity:
    """Prediction-error curiosity with a tabular forward model (Sections 7.1-7.2).

    The model predicts the one-hot next state from (s, a): p_hat(.|s,a) is an exponential
    moving average (step lr = 0.1) of one-hot(s').  It tracks the minimiser of the expected
    squared error, the true distribution p(.|s,a), but keeps fluctuating around it.  The
    intrinsic reward is the squared prediction error ||onehot(s') - p_hat(.|s,a)||^2
    BEFORE the update.  For a deterministic transition it decays to 0 (like 0.9^(2n));
    for a transition to one of C equally likely screens its expectation tends to
    (1 - 1/C)(1 + lr/(2 - lr)): 0.947 for C = 10, slightly above the minimum 1 - 1/C = 0.9
    that an exact running mean would reach.  Either way it never decays (aleatoric error).
    Errors of "press the remote" transitions on the TV are logged in `tv_errors`.
    """

    name = "forward-model curiosity"

    def __init__(self, env, lr=0.1):
        self.p = np.zeros((env.n_states, env.n_actions, env.n_states))
        self.lr = lr
        self.n_cells = env.n_cells
        self.tv_errors = []

    def reward(self, s, a, s2):
        pred = self.p[s, a]
        err = pred @ pred - 2 * pred[s2] + 1.0          # ||e_{s2} - pred||^2
        target = np.zeros_like(pred)
        target[s2] = 1.0
        self.p[s, a] += self.lr * (target - pred)
        if a == 4 and s >= self.n_cells:                # "press" while watching the TV
            self.tv_errors.append(float(err))
        return float(err)

    def end_episode(self, states):
        pass


class RND:
    """Random Network Distillation (Burda et al., 2019), Section 7.3.

    A fixed, randomly initialised target network f maps an observation to R^k; a predictor
    f_hat is trained by SGD to match it on visited observations.  The prediction error
    ||f_hat(x) - f(x)||^2 is large on observations unlike those trained on, and shrinks as
    an observation is visited: a learned, generalising novelty measure.

    Normalisation (ours, simpler than the paper's): inputs are fixed one-hot codes, so no
    observation whitening; the reward is the raw error divided by a running standard
    deviation of the raw per-step errors (Welford), then clipped to [0, r_clip].  The paper
    instead whitens observations and divides by a running std of the intrinsic RETURN.

    The running statistics are WARM-STARTED from the errors of all states under the
    initial predictor.  Starting them from zero is a real bug: if the first two observed
    errors are equal (bump into a wall, land on the same cell), the running variance is 0,
    the 1e-8 floor kicks in and the "normalised" reward is ~1e6.  One such spike poisons the
    running mean used for centring in `train` for the whole run (Chapter 14, Pitfalls).
    """

    name = "RND"

    def __init__(self, env, seed=0, out_dim=32, hidden=128, lr=1e-3, batch=32,
                 train_every=4, pool=256, r_clip=10.0):
        torch.manual_seed(seed)
        d = len(env.features(0))
        self.env = env
        self.target = nn.Sequential(nn.Linear(d, 64), nn.LeakyReLU(), nn.Linear(64, out_dim))
        self.pred = nn.Sequential(nn.Linear(d, hidden), nn.LeakyReLU(),
                                  nn.Linear(hidden, hidden), nn.LeakyReLU(),
                                  nn.Linear(hidden, out_dim))
        for p in self.target.parameters():
            p.requires_grad_(False)                       # the target is never trained
        self.opt = torch.optim.Adam(self.pred.parameters(), lr=lr)
        self.batch, self.train_every, self.pool = batch, train_every, pool
        self.r_clip = r_clip
        self.recent = []                                  # most recent observations
        self.rng = np.random.default_rng(seed)
        self.feat = torch.tensor(np.stack([env.features(s) for s in range(env.n_states)]))
        with torch.no_grad():
            self.y = self.target(self.feat)               # targets for every state, fixed
        self.err_cache = self._errors()                   # refreshed after each SGD step
        # Welford running mean/variance of the raw errors, warm-started from the initial
        # errors of all states (NOT from zero; see the class docstring).
        e0 = self.err_cache.astype(np.float64)
        self.count, self.mean, self.m2 = len(e0), float(e0.mean()), float(e0.var() * len(e0))
        self.steps = 0                                    # environment steps seen
        self.n_clipped = 0

    def _errors(self):
        with torch.no_grad():
            return ((self.pred(self.feat) - self.y) ** 2).mean(dim=1).numpy()

    def errors(self):
        return self.err_cache.copy()

    def reward(self, s, a, s2):
        e = float(self.err_cache[s2])                      # novelty of the NEW observation
        self.count += 1                                    # running std of raw errors
        self.steps += 1
        delta = e - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (e - self.mean)
        std = np.sqrt(self.m2 / self.count)
        # Train the predictor on recently visited observations (as RND trains on the
        # current rollouts), so the error of whatever the agent keeps visiting shrinks.
        self.recent.append(s2)
        if len(self.recent) > self.pool:
            self.recent.pop(0)
        if self.steps % self.train_every == 0:
            b = torch.tensor(self.rng.choice(self.recent, size=min(self.batch, len(self.recent))))
            loss = ((self.pred(self.feat[b]) - self.y[b]) ** 2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.err_cache = self._errors()
        r = e / max(std, 1e-8)
        if r > self.r_clip:                                # one outlier must not dominate
            self.n_clipped += 1
        return min(r, self.r_clip)

    def end_episode(self, states):
        pass


# ---------------------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------------------
class TwoStreamQ:
    """Tabular Q-learning with two value tables (Section 7.3, "two value heads").

    Q_E: extrinsic, episodic (no bootstrap at termination; bootstrap through truncation).
    Q_I: intrinsic.  intrinsic_at_goal="continue" (default) makes it NON-episodic, as in
         Burda et al. (2019): the intrinsic return does not stop when the episode
         terminates; we bootstrap from the restart state, the next state of the continuing
         stream.  intrinsic_at_goal="stop" ends the intrinsic return at termination too
         (target r_i alone), like the extrinsic stream.
    Truncation (time limit) is handled the same way in BOTH streams: partial-episode
    bootstrapping from s2.  The time limit is not part of the state, so treating a timeout
    as a jump to the start would make the stream non-Markov in s; a termination, by
    contrast, happens exactly at the goal cell, so "goal -> restart" is a Markov transition.
    Behaviour: eps-greedy on Q_E + beta * Q_I with random tie-breaking.  With beta = 0 and
    Q_E = 0 (no reward found yet) this is a uniform random walk.
    """

    def __init__(self, n_states, n_actions, beta=1.0, eps=0.05, alpha=0.2,
                 gamma_e=0.99, gamma_i=0.99, intrinsic_at_goal="continue"):
        assert intrinsic_at_goal in ("continue", "stop")
        self.QE = np.zeros((n_states, n_actions))
        self.QI = np.zeros((n_states, n_actions))
        self.beta, self.eps, self.alpha = beta, eps, alpha
        self.gamma_e, self.gamma_i = gamma_e, gamma_i
        self.intrinsic_at_goal = intrinsic_at_goal

    def act(self, s, rng):
        if rng.random() < self.eps:
            return int(rng.integers(self.QE.shape[1]))
        q = self.QE[s] + self.beta * self.QI[s]
        best = np.flatnonzero(q >= q.max() - 1e-12)
        return int(best[0]) if len(best) == 1 else int(rng.choice(best))

    def update(self, s, a, r_e, r_i, s2, terminated, s_restart):
        target_e = r_e if terminated else r_e + self.gamma_e * self.QE[s2].max()
        self.QE[s, a] += self.alpha * (target_e - self.QE[s, a])
        if terminated and self.intrinsic_at_goal == "stop":
            target_i = r_i                                 # episodic intrinsic stream
        else:
            s_next_i = s_restart if terminated else s2     # non-episodic: continue from restart
            target_i = r_i + self.gamma_i * self.QI[s_next_i].max()
        self.QI[s, a] += self.alpha * (target_i - self.QI[s, a])


def greedy_episode(env, agent, rng):
    """One evaluation episode, greedy on Q_E alone (beta = 0, eps = 0), no learning.

    This measures what the agent has learned to EXPLOIT, separately from what its
    exploratory behaviour policy does.  Ties (e.g. Q_E = 0 everywhere off the learned
    path) are broken at random with the separate `rng`, so evaluation does not perturb the
    training run.  Returns True if the goal was reached.  `env` must be a separate copy.
    """
    s = env.reset()
    while True:
        q = agent.QE[s]
        best = np.flatnonzero(q >= q.max() - 1e-12)
        a = int(best[0]) if len(best) == 1 else int(rng.choice(best))
        s, _, term, trunc = env.step(a)
        if term or trunc:
            return bool(term)


def train(env, agent, bonus, episodes, rng, centre=True, eval_env=None, eval_every=5,
          eval_rng=None):
    """Run `episodes` episodes of online (per-step) two-stream Q-learning (Sec. 7.3 box).

    Updates are online on purpose: a greedy policy frozen for a whole episode in a
    deterministic world quickly falls into a cycle; updating after every step makes the
    bonus of a state fall as soon as it is visited, which pushes the agent onwards.
    centre=False switches off the centring of the intrinsic reward (the ablation of
    Sec. 7.3).  If `eval_env` is given, every `eval_every` episodes one greedy episode on
    Q_E alone is run in it (see greedy_episode).

    Returns a dict with visit counts per state, distinct cells visited after each episode,
    goal reached per episode, the fraction of each episode's steps spent on the TV, and
    the greedy-evaluation results.
    """
    visits = np.zeros(env.n_states, dtype=np.int64)
    seen_cells = np.zeros(env.n_cells, dtype=bool)
    n_int, mean_int = 0, 0.0            # running mean of the intrinsic reward (centring)
    coverage, goals, tv_frac, steps, evals = [], [], [], [], []
    total_steps = 0
    s_restart = env.reset()
    for ep in range(episodes):
        s = env.reset()
        visits[s] += 1
        traj, tv_steps = [], 0
        while True:
            a = agent.act(s, rng)
            s2, r_e, term, trunc = env.step(a)
            r_i = 0.0
            if bonus is not None:
                # Centre the bonus: r_i - (running mean).  In the non-episodic intrinsic
                # stream a constant shift c changes every value by -c/(1-gamma_I) and not
                # the greedy policy; what it changes is the meaning of Q_I = 0 for an
                # untried action: "as good as an average step", i.e. optimistic relative
                # to well-known actions.  Without it, any tried action (positive bonus)
                # looks better than an untried one (Q_I = 0) and the agent circles.
                raw = bonus.reward(s, a, s2)
                n_int += 1
                mean_int += (raw - mean_int) / n_int
                r_i = raw - mean_int if centre else raw
            agent.update(s, a, r_e, r_i, s2, term, s_restart)
            traj.append((s, a, r_e, r_i, s2, term))
            visits[s2] += 1
            tv_steps += int(env.K > 0 and s2 >= env.n_cells)
            if term or trunc:
                break
            s = s2
        # Replay the episode once in reverse order (a tiny experience replay): it carries a
        # newly found reward, and the intrinsic values, back along the whole path at once.
        for (s, a, r_e, r_i, s2, term) in reversed(traj):
            agent.update(s, a, r_e, r_i, s2, term, s_restart)
        if bonus is not None:
            bonus.end_episode([t[4] for t in traj])
        total_steps += len(traj)
        for (_, _, _, _, s2, _) in traj:
            seen_cells[env.cell_id[env.cell_of_state(s2)]] = True
        seen_cells[env.cell_id[env.start]] = True
        coverage.append(seen_cells.sum())
        goals.append(bool(traj[-1][5]))
        tv_frac.append(tv_steps / len(traj))
        steps.append(total_steps)
        if eval_env is not None and (ep + 1) % eval_every == 0:
            evals.append(greedy_episode(eval_env, agent, eval_rng))
    return {"visits": visits, "coverage": np.array(coverage), "goals": np.array(goals),
            "tv_frac": np.array(tv_frac), "steps": np.array(steps),
            "evals": np.array(evals, dtype=bool), "eval_every": eval_every,
            "mean_int": mean_int}
