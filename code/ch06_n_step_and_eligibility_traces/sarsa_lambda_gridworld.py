"""Faster credit assignment with n-step SARSA and SARSA(lambda) on a sparse-reward gridworld.

Chapter 06, Sections 1.1, 3 and 13. A 10 x 7 open gridworld: start S at column 0,
row 3; goal G at column 9, row 3 (a shortest path takes 9 steps). Actions
up/right/down/left; moving into a wall leaves the agent in place. The reward is
+1 on reaching G and 0 otherwise; gamma = 0.95; epsilon-greedy behaviour with
epsilon = 0.1 and random tie-breaking; Q is initialised to 0.

Part 1 (in the style of S&B Figure 7.4): one episode of the uniformly random walk
that every method follows while Q = 0, and the action values each method changes
when it finally reaches G: one-step SARSA, 10-step SARSA, SARSA(lambda = 0.9).

Part 2: learning curves (steps per episode) for
    one-step SARSA, n-step SARSA (n = 4, 16), SARSA(lambda) with accumulating and
    replacing traces, true online SARSA(lambda), and Watkins's Q(lambda) (lambda = 0.9),
each with a sweep over the step size alpha; reported as the mean number of steps
per episode over the first 50 episodes (lower is better). The script also checks for
which step sizes one-step SARSA's runs are step-for-step identical (Section 13.3).

Outputs (full mode): figures/gridworld_first_episode.png, figures/gridworld_learning.png

Run:  python code/ch06_n_step_and_eligibility_traces/sarsa_lambda_gridworld.py [--quick]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
W, H = 10, 7
START, GOAL = (0, 3), (9, 3)
GAMMA, EPS = 0.95, 0.1
MAX_STEPS = 5000                      # time limit per episode (a truncation, never hit in practice)
MOVES = [(0, 1), (1, 0), (0, -1), (-1, 0)]   # up, right, down, left
N_S, N_A = W * H, 4


def idx(x, y):
    return y * W + x


NEXT = np.zeros((N_S, N_A), dtype=np.int64)
for x in range(W):
    for y in range(H):
        for a, (dx, dy) in enumerate(MOVES):
            NEXT[idx(x, y), a] = idx(min(max(x + dx, 0), W - 1), min(max(y + dy, 0), H - 1))
S0, SG = idx(*START), idx(*GOAL)


def step(s, a):
    """Returns (next state, reward, terminated)."""
    s1 = int(NEXT[s, a])
    return s1, (1.0 if s1 == SG else 0.0), s1 == SG


def eps_greedy(Q, s, rng):
    if rng.random() < EPS:
        return int(rng.integers(N_A))
    q = Q[s]
    best = np.flatnonzero(q == q.max())          # exact ties broken uniformly at random
    return int(best[0]) if len(best) == 1 else int(best[rng.integers(len(best))])


# ------------------------------------------------------------------------------ control algorithms
def n_step_sarsa(alpha, n, n_episodes, rng):
    """n-step SARSA (S&B 2018, Sec. 7.2 box). Returns steps per episode."""
    Q = np.zeros((N_S, N_A))
    visited = np.zeros((N_S, N_A), dtype=bool)
    steps = []
    disc = GAMMA ** np.arange(n)
    for _ in range(n_episodes):
        S, A, R = [S0], [eps_greedy(Q, S0, rng)], [0.0]      # R[0] is a dummy so R[t] = R_t
        T, truncated, t = np.inf, False, 0
        while True:
            if t < T:
                s1, r, term = step(S[t], A[t])
                visited[S[t], A[t]] = True
                S.append(s1); R.append(r)
                if term:
                    T = t + 1
                else:
                    A.append(eps_greedy(Q, s1, rng))
                    if t + 1 >= MAX_STEPS:                    # time limit: bootstrap from Q(S_T, A_T)
                        T, truncated = t + 1, True
            tau = t - n + 1
            if tau >= 0:
                end = int(min(tau + n, T))
                G = float(disc[: end - tau] @ np.asarray(R[tau + 1: end + 1]))
                if tau + n < T:
                    G += GAMMA ** n * Q[S[tau + n], A[tau + n]]
                elif truncated:
                    G += GAMMA ** (T - tau) * Q[S[T], A[T]]
                Q[S[tau], A[tau]] += alpha * (G - Q[S[tau], A[tau]])
            if tau == T - 1:
                break
            t += 1
        steps.append(int(T))
    return steps, Q, visited


def sarsa_lambda(alpha, lam, n_episodes, rng, trace="replacing"):
    """Tabular SARSA(lambda), backward view, accumulating or replacing traces."""
    Q = np.zeros((N_S, N_A))
    visited = np.zeros((N_S, N_A), dtype=bool)
    steps = []
    for _ in range(n_episodes):
        z = np.zeros((N_S, N_A))
        s, a = S0, eps_greedy(Q, S0, rng)
        for t in range(MAX_STEPS):
            s1, r, term = step(s, a)
            visited[s, a] = True
            if trace == "accumulating":
                z[s, a] += 1.0
            else:
                z[s, a] = 1.0
            if term:
                Q += alpha * (r - Q[s, a]) * z
                break
            a1 = eps_greedy(Q, s1, rng)
            delta = r + GAMMA * Q[s1, a1] - Q[s, a]
            Q += alpha * delta * z
            z *= GAMMA * lam
            s, a = s1, a1
        steps.append(t + 1)
    return steps, Q, visited


def true_online_sarsa_lambda(alpha, lam, n_episodes, rng):
    """True online SARSA(lambda) (S&B 2018, Sec. 12.7 box) with one-hot features."""
    Q = np.zeros((N_S, N_A))
    visited = np.zeros((N_S, N_A), dtype=bool)
    steps = []
    gl = GAMMA * lam
    for _ in range(n_episodes):
        z = np.zeros((N_S, N_A))
        q_old = 0.0
        s, a = S0, eps_greedy(Q, S0, rng)
        for t in range(MAX_STEPS):
            s1, r, term = step(s, a)
            visited[s, a] = True
            q = Q[s, a]
            if term:
                q1 = 0.0
            else:
                a1 = eps_greedy(Q, s1, rng)
                q1 = Q[s1, a1]
            delta = r + GAMMA * q1 - q
            zsa = z[s, a]
            z *= gl
            z[s, a] += 1.0 - alpha * gl * zsa                 # dutch trace
            Q += alpha * (delta + q - q_old) * z
            Q[s, a] -= alpha * (q - q_old)
            q_old = q1
            if term:
                break
            s, a = s1, a1
        steps.append(t + 1)
    return steps, Q, visited


def watkins_q_lambda(alpha, lam, n_episodes, rng):
    """Watkins's Q(lambda) with replacing traces: traces are cut after any exploratory
    (non-greedy) action, because the greedy target policy would not have taken it."""
    Q = np.zeros((N_S, N_A))
    visited = np.zeros((N_S, N_A), dtype=bool)
    steps = []
    for _ in range(n_episodes):
        z = np.zeros((N_S, N_A))
        s, a = S0, eps_greedy(Q, S0, rng)
        for t in range(MAX_STEPS):
            s1, r, term = step(s, a)
            visited[s, a] = True
            z[s, a] = 1.0
            if term:
                Q += alpha * (r - Q[s, a]) * z
                break
            a1 = eps_greedy(Q, s1, rng)
            q_max = Q[s1].max()
            greedy = Q[s1, a1] == q_max                       # A' ties for the max -> keep the trace
            delta = r + GAMMA * q_max - Q[s, a]
            Q += alpha * delta * z
            if greedy:
                z *= GAMMA * lam
            else:
                z[:] = 0.0
            s, a = s1, a1
        steps.append(t + 1)
    return steps, Q, visited


def greedy_path_length(Q, rng, n_rollouts=10, cap=200):
    """Mean length of epsilon = 0 rollouts from S (random tie-breaking); cap if it never arrives."""
    total = 0
    for _ in range(n_rollouts):
        s, t = S0, 0
        while s != SG and t < cap:
            q = Q[s]
            best = np.flatnonzero(q == q.max())
            s = int(NEXT[s, best[rng.integers(len(best))]])
            t += 1
        total += t
    return total / n_rollouts


LAM = 0.9
METHODS = {
    "SARSA (n=1)": lambda al, E, rng: n_step_sarsa(al, 1, E, rng),
    "4-step SARSA": lambda al, E, rng: n_step_sarsa(al, 4, E, rng),
    "16-step SARSA": lambda al, E, rng: n_step_sarsa(al, 16, E, rng),
    "SARSA(λ) accumulating": lambda al, E, rng: sarsa_lambda(al, LAM, E, rng, "accumulating"),
    "SARSA(λ) replacing": lambda al, E, rng: sarsa_lambda(al, LAM, E, rng, "replacing"),
    "true online SARSA(λ)": lambda al, E, rng: true_online_sarsa_lambda(al, LAM, E, rng),
    "Watkins's Q(λ)": lambda al, E, rng: watkins_q_lambda(al, LAM, E, rng),
}


# ------------------------------------------------------------------------------ part 1: first episode
def first_episode_credit(rng, alpha=0.5, n=10, lam=0.9):
    """Generate the uniformly random first episode and apply each method's updates to it."""
    S, A = [S0], []
    s = S0
    while s != SG:
        a = int(rng.integers(N_A))
        A.append(a)
        s = int(NEXT[s, a])
        S.append(s)
    T = len(A)
    out = {}
    # one-step SARSA: only the last update has a non-zero target
    Q = np.zeros((N_S, N_A)); Q[S[T - 1], A[T - 1]] += alpha * 1.0
    out["one-step SARSA"] = Q
    # n-step SARSA: every update whose window contains the final reward
    Q = np.zeros((N_S, N_A))
    for tau in range(max(0, T - n), T):
        G = GAMMA ** (T - 1 - tau)                                # only R_T = 1 is non-zero
        Q[S[tau], A[tau]] += alpha * (G - Q[S[tau], A[tau]])
    out[f"{n}-step SARSA"] = Q
    # SARSA(lambda), accumulating: all deltas are zero except the last, delta_{T-1} = 1
    z = np.zeros((N_S, N_A))
    for t in range(T):
        z *= GAMMA * lam
        z[S[t], A[t]] += 1.0
    out[f"SARSA(λ), λ={lam}"] = alpha * 1.0 * z
    return S, A, out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.quick:
        alphas, n_runs, n_episodes = [0.2, 0.6], 2, 10
    else:
        alphas, n_runs, n_episodes = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0], 30, 50
    print(f"Gridworld {W}x{H}, start {START}, goal {GOAL} | seed={args.seed} gamma={GAMMA} "
          f"epsilon={EPS} lambda={LAM} runs={n_runs} episodes={n_episodes}")
    print(f"alpha in {alphas}")
    t0 = time.time()

    # ---- part 1
    S, A, credit = first_episode_credit(np.random.default_rng(args.seed))
    print(f"\nPart 1: first (random) episode has T={len(A)} steps, visits {len(set(S))} distinct cells")
    for name, Q in credit.items():
        print(f"  {name:20s} changes {int((Q != 0).sum()):3d} action values "
              f"in {int((Q != 0).any(1).sum()):2d} cells")

    # ---- part 2
    curves, greedy_len, n_visited = {}, {}, {}
    mean_steps = {}
    for name, fn in METHODS.items():
        tm = time.time()
        res = np.zeros((len(alphas), n_runs, n_episodes))
        gpl = np.zeros((len(alphas), n_runs))          # greedy path length after the last episode
        nvis = np.zeros((len(alphas), n_runs))         # distinct (s, a) pairs tried during learning
        for i, al in enumerate(alphas):
            for run in range(n_runs):
                rng = np.random.default_rng(10_000 * args.seed + run)   # same seeds for every method/alpha
                st, Q, vis = fn(al, n_episodes, rng)
                res[i, run] = st
                gpl[i, run] = greedy_path_length(Q, rng)
                nvis[i, run] = vis.sum()
        curves[name] = res
        greedy_len[name] = gpl
        n_visited[name] = nvis
        mean_steps[name] = res.mean(axis=(1, 2))
        print(f"  {name:22s} done in {time.time() - tm:5.1f} s")
    print(f"\nPart 2: mean steps per episode over episodes 1-{n_episodes} ({n_runs} runs)")
    print("  method                 " + "".join(f"a={a:<6g}" for a in alphas) + "  best")
    best = {}
    for name, m in mean_steps.items():
        j = int(np.argmin(m))
        best[name] = j
        print(f"  {name:22s} " + "".join(f"{v:7.1f} " for v in m) + f"  {m[j]:.1f} (alpha={alphas[j]})")
    # Why is one-step SARSA flat in alpha? Check directly: compare every run's sequence of
    # episode lengths with the one obtained at the smallest alpha (same seeds, Section 13.3).
    res1 = curves["SARSA (n=1)"]
    same = [int(np.all(res1[i] == res1[0], axis=1).sum()) for i in range(len(alphas))]
    k = 0
    while k + 1 < len(alphas) and same[k + 1] == n_runs:
        k += 1
    print(f"\n  one-step SARSA: runs identical (all {n_episodes} episode lengths) to those at alpha={alphas[0]:g}: "
          + ", ".join(f"alpha={a:g}: {m}/{n_runs}" for a, m in zip(alphas, same)))
    print(f"  -> step-for-step identical for every alpha <= {alphas[k]:g} in the grid")
    print("\n  at the best alpha:     ep. 1   eps 2-10   last 10   s.e.(all)   greedy path   "
          "% optimal   (s,a) pairs tried")
    for name, res in curves.items():
        j = best[name]
        r = res[j]
        per_run = r.mean(axis=1)
        g = greedy_len[name][j]
        print(f"  {name:22s} {r[:, 0].mean():6.1f}   {r[:, 1:10].mean():7.1f}   {r[:, -10:].mean():7.1f}"
              f"   {per_run.std(ddof=1) / np.sqrt(len(per_run)):7.1f}   {g.mean():10.1f}   "
              f"{100 * np.mean(g == 9):8.0f}%   {n_visited[name][j].mean():10.1f} / {N_S * N_A}")
    print(f"\nelapsed {time.time() - t0:.1f} s")

    if args.quick:
        return
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    from plot_style import setup, C, BLUES, MARKERS
    plt = setup()
    os.makedirs(FIG_DIR, exist_ok=True)

    # figure 1: first-episode credit
    cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb"] + BLUES)     # visit counts: 0 = white
    # Credit panels: the ramp starts at a clearly visible light blue, so that *every* changed
    # cell is coloured; unchanged cells are masked and show the white background. The log
    # scale starts at the smallest credit actually given (about 1e-7 for SARSA(lambda)).
    cmap_credit = LinearSegmentedColormap.from_list("credit", BLUES)
    vmin = min(float(v[v > 0].min()) for v in credit.values())
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    xs = [s % W for s in S]
    ys = [s // W for s in S]
    visits = np.bincount(S, minlength=N_S).reshape(H, W)
    panels = [("path taken (visit counts)", visits)] + \
             [(f"Q increased by {k}", v.max(axis=1).reshape(H, W)) for k, v in credit.items()]
    for k, (ax, (title, M)) in enumerate(zip(axes, panels)):
        if k == 0:
            im = ax.imshow(M, origin="lower", cmap=cmap, vmin=0)
        else:   # log colour scale so that tiny (but non-zero) credit is visible; zero = background
            im = ax.imshow(np.ma.masked_less_equal(M, 0), origin="lower", cmap=cmap_credit,
                           norm=LogNorm(vmin=vmin, vmax=0.5))
        ax.plot(xs, ys, color=C[1], lw=0.5, alpha=0.6)
        ax.text(START[0], START[1], "S", ha="center", va="center", fontsize=10, weight="bold", color="#0b0b0b")
        ax.text(GOAL[0], GOAL[1], "G", ha="center", va="center", fontsize=10, weight="bold", color="#0b0b0b")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle(f"After the first episode (T = {len(A)} random steps, α = 0.5, γ = {GAMMA}): "
                 "max over actions of the increase in Q(s, ·) (log colour scale; white = unchanged)", y=1.02)
    print(f"  first-episode figure: log colour scale from {vmin:.2g} to 0.5")
    path1 = os.path.join(FIG_DIR, "gridworld_first_episode.png")
    fig.savefig(path1, bbox_inches="tight")
    plt.close(fig)

    # figure 2: parameter study and learning curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    for k, (name, m) in enumerate(mean_steps.items()):
        ax.plot(alphas, m, color=C[k], marker=MARKERS[k], label=name)
    ax.set_yscale("log")
    ax.set_yticks([20, 30, 40, 60, 100])
    ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"step size $\alpha$")
    ax.set_ylabel(f"mean steps per episode, episodes 1-{n_episodes} (log scale)")
    ax.set_title(f"Parameter study ({n_runs} runs, λ = {LAM})")
    ax.legend(fontsize=8)
    ax = axes[1]
    ep = np.arange(1, n_episodes + 1)
    for k, (name, res) in enumerate(curves.items()):
        ax.plot(ep, res[best[name]].mean(axis=0), color=C[k], marker=MARKERS[k], markevery=5,
                label=f"{name} (α={alphas[best[name]]})")
    ax.axhline(9, color="#898781", ls=":", lw=1.2)
    ax.text(n_episodes, 9.6, "shortest path (9)", ha="right", fontsize=8, color="#52514e")
    ax.set_yscale("log")
    ax.set_yticks([10, 20, 30, 50, 100, 200])
    ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("episode")
    ax.set_ylabel("steps per episode, mean over runs (log scale)")
    ax.set_title("Learning curves at each method's best α")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path2 = os.path.join(FIG_DIR, "gridworld_learning.png")
    fig.savefig(path2)
    plt.close(fig)
    print(f"saved {path1}\nsaved {path2}")


if __name__ == "__main__":
    main()
