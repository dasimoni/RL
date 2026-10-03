"""A small benchmark suite of tabular tasks and four TD agents, vectorized over seeds.

Used by evaluate_agents.py (Chapter 20, Section 6). It has no main block: this is
a library module.

Design choices (and why):

* The environments are the Gymnasium toy-text tasks, but we read their exact
  transition tables (env.unwrapped.P) once and then step all N independent runs
  at once with numpy. N runs of one agent cost roughly the same wall-clock
  time as one run, which is what makes 100 runs x 5 tasks x 4 agents feasible
  on one CPU core. Each run has its own state, Q-table and episode clock.
* Every task has an explicit time limit H. Hitting it is a *truncation*: the
  TD target still bootstraps from the next state (Section 8, bug 1).
* Agents are scored by the EXACT expected undiscounted return of their greedy
  policy over H steps from the initial-state distribution (finite-horizon
  policy evaluation on the known tables), so the scores carry no evaluation
  noise: all the variability in Section 6 comes from training.
* Scores are normalized per task (Eq. 20.9): 1 = optimal finite-horizon
  policy, 0 = the best TRIVIAL policy, i.e. the better of the uniform-random
  policy and the best constant-action policy, all computed exactly by dynamic
  programming. Using the random policy alone as the lower reference would be
  misleading on tasks with large penalties: on Taxi, a policy that only moves
  (and never delivers) would score 0.733, and on slippery Cliff Walking,
  "always left" would score 0.964 (Section 6.7).
"""
import gymnasium as gym
import numpy as np

TASKS = {
    # name: (gymnasium id, kwargs, horizon H, training steps per run)
    "FrozenLake4x4": ("FrozenLake-v1", {"map_name": "4x4", "is_slippery": True}, 100, 40_000),
    "FrozenLake8x8": ("FrozenLake-v1", {"map_name": "8x8", "is_slippery": True}, 200, 150_000),
    "Taxi": ("Taxi-v4", {}, 200, 40_000),
    "CliffWalking": ("CliffWalking-v1", {}, 100, 10_000),
    "CliffSlippery": ("CliffWalkingSlippery-v1", {}, 100, 40_000),
}
AGENTS = ["Q-learning", "Double Q", "Expected SARSA", "SARSA"]


class TabularTask:
    """Exact tables of a toy-text task: outcome k of (s, a) has probability
    prob[s,a,k], next state nxt[s,a,k], reward rew[s,a,k], terminal flag term[s,a,k]."""

    def __init__(self, name):
        env_id, kwargs, self.H, self.train_steps = TASKS[name]
        self.name = name
        env = gym.make(env_id, **kwargs)
        u = env.unwrapped
        P = u.P
        self.S, self.A = u.observation_space.n, u.action_space.n
        K = max(len(P[s][a]) for s in P for a in P[s])
        self.prob = np.zeros((self.S, self.A, K))
        self.nxt = np.zeros((self.S, self.A, K), dtype=np.int64)
        self.rew = np.zeros((self.S, self.A, K))
        self.term = np.zeros((self.S, self.A, K))
        for s in range(self.S):
            for a in range(self.A):
                for k, (p, s2, r, d) in enumerate(P[s][a]):
                    self.prob[s, a, k], self.nxt[s, a, k] = p, int(s2)
                    self.rew[s, a, k], self.term[s, a, k] = r, float(d)
        self.cum = np.cumsum(self.prob, axis=-1)
        self.cum[..., -1] = 1.0 + 1e-12          # guard against round-off in sampling
        self.d0 = np.asarray(u.initial_state_distrib, dtype=float)
        self.d0_cum = np.cumsum(self.d0)
        self.d0_cum[-1] = 1.0 + 1e-12
        env.close()
        self.random_score = self.policy_value(np.full((1, self.S, self.A), 1.0 / self.A))[0]
        # exact H-step return of each constant-action policy "always take action a"
        self.constant_scores = self.deterministic_value(np.tile(np.arange(self.A)[:, None], (1, self.S)))
        self.trivial_score = max(self.random_score, float(self.constant_scores.max()))
        self.optimal_score = self.optimal_value()

    # ---------------- sampling (all runs at once) ----------------
    def reset(self, rng, n):
        return np.searchsorted(self.d0_cum, rng.random(n), side="right")

    def step(self, rng, s, a):
        u = rng.random(s.shape[0])
        k = (self.cum[s, a] < u[:, None]).sum(axis=1)
        return self.nxt[s, a, k], self.rew[s, a, k], self.term[s, a, k].astype(bool)

    # ---------------- exact finite-horizon evaluation ----------------
    def policy_value(self, pi):
        """Expected undiscounted H-step return from d0 for stationary policies.

        pi: (N, S, A) action probabilities, one policy per run. Returns (N,)."""
        r_sa = (self.prob * self.rew).sum(-1)                         # (S, A)
        V = np.zeros((pi.shape[0], self.S))
        for _ in range(self.H):
            # Q_h(s,a) = r(s,a) + sum_k p_k (1 - term_k) V_{h-1}(next_k)
            cont = (self.prob * (1 - self.term))[None] * V[:, self.nxt]   # (N, S, A, K)
            Qh = r_sa[None] + cont.sum(-1)
            V = (pi * Qh).sum(-1)
        return V @ self.d0

    def deterministic_value(self, act):
        """Same as policy_value for deterministic policies act: (N, S) action indices.

        Gathers only the chosen action's row, A times cheaper than policy_value."""
        rows = np.arange(self.S)[None, :]
        p = (self.prob * (1 - self.term))[rows, act]                  # (N, S, K)
        nxt = self.nxt[rows, act]                                     # (N, S, K)
        r = (self.prob * self.rew).sum(-1)[rows, act]                 # (N, S)
        n_idx = np.arange(act.shape[0])[:, None, None]
        V = np.zeros(act.shape, dtype=float)
        for _ in range(self.H):
            V = r + (p * V[n_idx, nxt]).sum(-1)
        return V @ self.d0

    def optimal_value(self):
        r_sa = (self.prob * self.rew).sum(-1)
        V = np.zeros(self.S)
        for _ in range(self.H):                                      # non-stationary optimum
            V = (r_sa + (self.prob * (1 - self.term) * V[self.nxt]).sum(-1)).max(-1)
        return float(V @ self.d0)

    def normalized(self, raw, lo=None):
        """Eq. (20.9) with hi = optimal and lo = best trivial policy (default)."""
        lo = self.trivial_score if lo is None else lo
        return (raw - lo) / (self.optimal_score - lo)

    def greedy_score(self, Q):
        """Normalized exact score of the greedy policy of each run's Q-table (N, S, A)."""
        return self.normalized(self.deterministic_value(Q.argmax(-1)))   # deterministic tie-break


def _greedy_random_ties(rng, q):
    """argmax over the last axis with uniformly random tie-breaking. q: (N, A)."""
    best = q == q.max(axis=1, keepdims=True)
    return np.argmax(rng.random(q.shape) * best, axis=1)


def _eps_greedy(rng, q, eps):
    n, A = q.shape
    a = _greedy_random_ties(rng, q)
    explore = rng.random(n) < eps
    a[explore] = rng.integers(0, A, size=explore.sum())
    return a


def train(task, agent, n_runs, seed, steps=None, alpha=0.1, eps=0.1, gamma=0.99, n_checkpoints=10):
    """Train n_runs independent copies of `agent` on `task` for `steps` steps each.

    Returns (final normalized scores (n_runs,), checkpoint steps, scores at checkpoints (C, n_runs))."""
    rng = np.random.default_rng(seed)
    steps = task.train_steps if steps is None else steps
    N, S, A = n_runs, task.S, task.A
    QA = np.zeros((N, S, A))
    QB = np.zeros((N, S, A)) if agent == "Double Q" else None
    rows = np.arange(N)
    s = task.reset(rng, N)
    t_ep = np.zeros(N, dtype=np.int64)

    def behaviour(idx, st):                    # the Q-values the eps-greedy behaviour policy uses
        return QA[idx, st] + QB[idx, st] if QB is not None else QA[idx, st]

    a = _eps_greedy(rng, behaviour(rows, s), eps)
    checkpoints = np.linspace(steps / n_checkpoints, steps, n_checkpoints).astype(int)
    checkpoint_set = set(int(c) for c in checkpoints)       # O(1) membership test in the loop
    curve = []
    for t in range(1, steps + 1):
        s2, r, term = task.step(rng, s, a)
        t_ep += 1
        trunc = (t_ep >= task.H) & ~term
        cont = (~term).astype(float)           # bootstrap unless TERMINATED (truncation still bootstraps)
        a2 = _eps_greedy(rng, behaviour(rows, s2), eps)    # next action (also needed by SARSA)
        if agent == "Q-learning":
            target = r + gamma * cont * QA[rows, s2].max(1)
            QA[rows, s, a] += alpha * (target - QA[rows, s, a])
        elif agent == "SARSA":
            target = r + gamma * cont * QA[rows, s2, a2]
            QA[rows, s, a] += alpha * (target - QA[rows, s, a])
        elif agent == "Expected SARSA":
            q2 = QA[rows, s2]
            best = q2 == q2.max(1, keepdims=True)
            pi2 = eps / A + (1 - eps) * best / best.sum(1, keepdims=True)   # eps-greedy probabilities
            target = r + gamma * cont * (pi2 * q2).sum(1)
            QA[rows, s, a] += alpha * (target - QA[rows, s, a])
        elif agent == "Double Q":
            upd_a = rng.random(N) < 0.5         # each run flips its own coin
            for Q1, Q2, m in ((QA, QB, upd_a), (QB, QA, ~upd_a)):
                idx = rows[m]
                a_star = _greedy_random_ties(rng, Q1[idx, s2[m]])
                target = r[m] + gamma * cont[m] * Q2[idx, s2[m], a_star]
                Q1[idx, s[m], a[m]] += alpha * (target - Q1[idx, s[m], a[m]])
        else:
            raise ValueError(agent)
        # episode boundaries: terminated or truncated runs restart from d0
        done = term | trunc
        if done.any():
            s2 = s2.copy()
            s2[done] = task.reset(rng, int(done.sum()))
            t_ep[done] = 0
            a2[done] = _eps_greedy(rng, behaviour(rows[done], s2[done]), eps)
        s, a = s2, a2
        if t in checkpoint_set:
            Qeval = QA if QB is None else QA + QB
            curve.append(task.greedy_score(Qeval))
    return curve[-1], checkpoints, np.array(curve)
