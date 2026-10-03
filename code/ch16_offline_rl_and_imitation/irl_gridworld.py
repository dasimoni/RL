"""Inverse RL on a gridworld: apprenticeship learning (feature matching) and maximum-entropy IRL,
compared with behaviour cloning, including transfer to a changed world (Chapter 16, Sections 3-4).

World.  A 7 x 9 grid of terrain types: road, grass, mud and a goal.  Reward depends only on the
terrain, r(s) = omega . phi(s) with phi(s) the one-hot terrain indicator and
        omega_true = (road 0, grass -2, mud -6, goal +3).
Moves (N, S, E, W, stay) succeed with probability 0.9, otherwise the agent stays put.  The goal
is absorbing.  Episodes have a fixed horizon H = 30 and start in the top-left corner.

Demonstrations.  M = 30 trajectories from the soft-optimal (Boltzmann-rational) policy of the
true reward: exactly the behaviour model assumed by maximum-entropy IRL (Section 4.3).

Learners
  BC          per-state action frequencies of the demonstrations (uniform where unseen)
  Projection  Abbeel & Ng's projection algorithm (Algorithm 16.3): find a mixture of optimal
              policies whose feature expectations match the expert's
  MaxEnt IRL  gradient ascent on the demonstrations' log-likelihood (Algorithm 16.4): the
              gradient is  mu_expert - mu_omega  (Eq. 16.16), computed with a backward soft
              value-iteration pass and a forward state-visitation pass

Transfer test.  The road is blocked by mud and a new road is opened; the start moves.  BC
can only replay its copied actions; the IRL methods re-plan with the reward they recovered.

Run:  python code/ch16_offline_rl_and_imitation/irl_gridworld.py [--quick]
Output (full mode): figures/irl_gridworld.png
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from scipy.special import logsumexp

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# terrain codes: 0 road, 1 grass, 2 mud, 3 goal
TRAIN_MAP = [
    "rrrgggggg",
    "gmrmmmmgg",
    "gmrrrrmgg",
    "gmmmmrmgg",
    "ggggmrrrG",
    "ggggmmggg",
    "ggggggggg",
]
# transfer world: the old road is cut by mud at two places, a new road runs along the bottom
TEST_MAP = [
    "rmrgggggg",
    "rmmmmmmgg",
    "rmrrrrmgg",
    "rmmmmmmgg",
    "rgggmrrrG",
    "rmmmmmmmr",
    "rrrrrrrrr",
]
CODES = {"r": 0, "g": 1, "m": 2, "G": 3}
FEATURES = ["road", "grass", "mud", "goal"]
OMEGA_TRUE = np.array([0.0, -2.0, -6.0, 3.0])
MOVES = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]       # N, S, E, W, stay
NA = len(MOVES)
SUCCESS = 0.9
H = 30


class Grid:
    def __init__(self, rows, start):
        self.h, self.w = len(rows), len(rows[0])
        self.ns = self.h * self.w
        self.terrain = np.array([CODES[c] for row in rows for c in row])
        self.phi = np.eye(len(FEATURES))[self.terrain]            # phi(s), shape [S, 4]
        self.start = start[0] * self.w + start[1]
        self.goal = int(np.flatnonzero(self.terrain == 3)[0])
        P = np.zeros((self.ns, NA, self.ns))
        for s in range(self.ns):
            r, c = divmod(s, self.w)
            for a, (dr, dc) in enumerate(MOVES):
                if s == self.goal:                                 # absorbing goal
                    P[s, a, s] = 1.0
                    continue
                r2, c2 = min(max(r + dr, 0), self.h - 1), min(max(c + dc, 0), self.w - 1)
                P[s, a, r2 * self.w + c2] += SUCCESS
                P[s, a, s] += 1 - SUCCESS
        self.P = P

    def soft_policy(self, reward):
        """Finite-horizon soft (MaxEnt) value iteration, Eqs. (16.14)-(16.15):
        Q_t(s,a) = r(s) + sum_s' P(s'|s,a) V_{t+1}(s'),  V_t(s) = log sum_a exp Q_t(s,a).
        Returns the time-dependent policies pi_t(a|s) = exp(Q_t(s,a) - V_t(s))."""
        V = np.zeros(self.ns)
        pis = np.zeros((H, self.ns, NA))
        for t in reversed(range(H)):
            Q = reward[:, None] + self.P @ V
            V = logsumexp(Q, axis=1)
            pis[t] = np.exp(Q - V[:, None])
        return pis

    def optimal_policy(self, reward):
        """Ordinary (hard-max) finite-horizon value iteration -> deterministic pi_t."""
        V = np.zeros(self.ns)
        pis = np.zeros((H, self.ns, NA))
        for t in reversed(range(H)):
            Q = reward[:, None] + self.P @ V
            V = Q.max(1)
            pis[t] = np.eye(NA)[Q.argmax(1)]
        return pis

    def visitation(self, pis):
        """D_t(s) for t < H (forward pass); pis is [H, S, A] or a stationary [S, A]."""
        D = np.zeros((H, self.ns))
        D[0, self.start] = 1.0
        for t in range(H - 1):
            pi = pis[t] if pis.ndim == 3 else pis
            D[t + 1] = np.einsum("s,sa,sap->p", D[t], pi, self.P)
        return D

    def feature_expectations(self, pis):
        return self.visitation(pis).sum(0) @ self.phi                # mu(pi) = E[sum_t phi(s_t)]

    def value(self, pis, omega=OMEGA_TRUE):
        return float(self.feature_expectations(pis) @ omega)

    def sample(self, pis, n, rng):
        trajs = []
        for _ in range(n):
            s, traj = self.start, []
            for t in range(H):
                pi = pis[t] if pis.ndim == 3 else pis
                a = rng.choice(NA, p=pi[s])
                traj.append((s, a))
                s = rng.choice(self.ns, p=self.P[s, a])
            trajs.append(traj)
        return trajs


def behaviour_cloning(grid, trajs):
    counts = np.zeros((grid.ns, NA))
    for traj in trajs:
        for s, a in traj:
            counts[s, a] += 1
    pi = np.full((grid.ns, NA), 1.0 / NA)
    seen = counts.sum(1) > 0
    pi[seen] = counts[seen] / counts[seen].sum(1, keepdims=True)
    return pi, seen


def empirical_mu(grid, trajs):
    return np.mean([sum(grid.phi[s] for s, _ in traj) for traj in trajs], axis=0)


def maxent_irl(grid, mu_E, iters, lr, l2=0.0, log_every=None):
    """Algorithm 16.4: gradient ascent on the log-likelihood of the demonstrations, optionally
    with a Gaussian prior (L2 penalty l2/2 |omega|^2, i.e. MAP estimation)."""
    omega = np.zeros(len(FEATURES))
    hist = []
    for k in range(iters):
        pis = grid.soft_policy(grid.phi @ omega)
        mu = grid.feature_expectations(pis)
        gap = mu_E - mu                                             # Eq. (16.16)
        omega += lr * (gap - l2 * omega)
        if log_every and (k % log_every == 0 or k == iters - 1):
            hist.append((k, np.linalg.norm(gap), omega.copy()))
    return omega, hist


def projection_method(grid, mu_E, iters):
    """Algorithm 16.3 (Abbeel & Ng 2004, projection version).  Returns the weights w of the last
    iteration, the list of optimal policies found, and the mixture weights whose feature
    expectations are closest to mu_E, plus the distance history."""
    pis0 = grid.optimal_policy(np.zeros(grid.ns))       # any initial policy
    mus = [grid.feature_expectations(pis0)]
    policies = [pis0]
    mu_bar = mus[0].copy()
    lam = np.array([1.0])                                # mixture weights over `policies`
    dist = [np.linalg.norm(mu_E - mu_bar)]
    w = mu_E - mu_bar
    for _ in range(iters):
        w = mu_E - mu_bar                                # reward weights for this iteration
        pis = grid.optimal_policy(grid.phi @ w)          # RL step: optimal policy for r = w.phi
        mu = grid.feature_expectations(pis)
        policies.append(pis); mus.append(mu)
        # orthogonal projection of mu_E onto the segment [mu_bar, mu]
        d = mu - mu_bar
        alpha = 0.0 if d @ d < 1e-12 else np.clip(d @ (mu_E - mu_bar) / (d @ d), 0, 1)
        mu_bar = mu_bar + alpha * d
        lam = np.append(lam * (1 - alpha), alpha)        # mu_bar = sum_j lam_j mu_j
        dist.append(np.linalg.norm(mu_E - mu_bar))
    return w, policies, lam, dist


def mixture_value(grid, policies, lam, omega=OMEGA_TRUE):
    """A mixture policy picks policy j with probability lam_j at the start of the episode, so
    its value is the lam-weighted average of the component values."""
    return float(sum(l * grid.value(p, omega) for l, p in zip(lam, policies)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    quick = args.quick
    seed = 0
    rng = np.random.default_rng(seed)
    M = 30
    iters = 100 if quick else 2000
    l2 = 0.03                    # Gaussian prior on omega (MAP); l2 = 0 is plain maximum likelihood
    lr = 0.05
    proj_iters = 10 if quick else 30
    print(f"irl_gridworld: seed={seed} quick={quick} horizon H={H} demos M={M} success prob={SUCCESS} "
          f"omega_true={OMEGA_TRUE.tolist()} MaxEnt iters={iters} lr={lr} l2={l2}")
    t0 = time.time()

    train = Grid(TRAIN_MAP, start=(0, 0))
    test = Grid(TEST_MAP, start=(6, 0))
    r_true = train.phi @ OMEGA_TRUE
    expert_soft = train.soft_policy(r_true)
    demos = train.sample(expert_soft, M, rng)
    mu_E = empirical_mu(train, demos)
    print(f"expert feature expectations (empirical, M={M}): "
          + ", ".join(f"{f} {m:.2f}" for f, m in zip(FEATURES, mu_E)))
    print(f"exact soft-expert feature expectations:          "
          + ", ".join(f"{f} {m:.2f}" for f, m in zip(FEATURES, train.feature_expectations(expert_soft))))
    # Is mu_E achievable at all?  The range of each feature count over ALL policies is found by
    # planning optimally for the reward +phi_j (max) and -phi_j (min).  With stochastic moves the
    # empirical mu_E of a finite sample can lie OUTSIDE the achievable set (Section 4.4): then no
    # reward matches the features and the MaxEnt dual (16.16) is unbounded.
    print("achievable range of each feature count over all policies (optimal for +-phi_j):")
    for j, f in enumerate(FEATURES):
        e = np.eye(len(FEATURES))[j]
        hi = train.feature_expectations(train.optimal_policy(train.phi @ e))[j]
        lo = train.feature_expectations(train.optimal_policy(-train.phi @ e))[j]
        flag = "  <-- OUTSIDE the achievable range" if not (lo - 1e-9 <= mu_E[j] <= hi + 1e-9) else ""
        print(f"  {f:5s}: [{lo:.3f}, {hi:.3f}]   expert (empirical) {mu_E[j]:.3f}{flag}")

    # --- learners -------------------------------------------------------------------------
    pi_bc, seen = behaviour_cloning(train, demos)
    omega_ml, hist_ml = maxent_irl(train, mu_E, iters, lr, l2=0.0, log_every=max(iters // 50, 1))
    omega_me, hist = maxent_irl(train, mu_E, iters, lr, l2=l2, log_every=max(iters // 50, 1))
    w_proj, proj_pols, lam, dist = projection_method(train, mu_E, proj_iters)

    print(f"\nBC saw {seen.sum()} of {train.ns} states in the demonstrations")
    centred_true = OMEGA_TRUE - OMEGA_TRUE.mean()
    print("true omega minus its mean (the sum of omega is not identifiable):", np.round(centred_true, 2).tolist())
    for name, th, hi in [("max. likelihood (l2=0)", omega_ml, hist_ml), (f"MAP (l2={l2})", omega_me, hist)]:
        mid = hi[len(hi) // 2][2]
        print(f"MaxEnt IRL {name:22s}: omega at step {hi[len(hi) // 2][0]} = {np.round(mid, 2).tolist()}, "
              f"final = {np.round(th, 2).tolist()}, final |mu_E - mu_omega| = {hi[-1][1]:.4f}")
    print(f"  differences to road (grass, mud, goal): true {np.round(OMEGA_TRUE[1:] - OMEGA_TRUE[0], 2).tolist()}, "
          f"MAP {np.round(omega_me[1:] - omega_me[0], 2).tolist()}")
    mu_ml = train.feature_expectations(train.soft_policy(train.phi @ omega_ml))
    print(f"  max. likelihood end point: mu_omega = {np.round(mu_ml, 3).tolist()}, "
          f"residual mu_E - mu_omega = {np.round(mu_E - mu_ml, 3).tolist()}")
    mu_bar = sum(l * train.feature_expectations(p) for l, p in zip(lam, proj_pols))
    print(f"projection method: |mu_E - mu_bar| over iterations: "
          f"{dist[0]:.3f} -> {dist[min(5, len(dist) - 1)]:.3f} (it. 5) -> {dist[-1]:.4f} (it. {proj_iters}); "
          f"mu_bar = {np.round(mu_bar, 3).tolist()}; last w = {np.round(w_proj, 3).tolist()}")

    # --- evaluation in the training world and in the transfer world ------------------------
    rows = []
    for name, world in [("train", train), ("transfer", test)]:
        r_world = world.phi @ OMEGA_TRUE
        v_opt = world.value(world.optimal_policy(r_world))
        v_soft = world.value(world.soft_policy(r_world))
        # BC: both maps have the same 7 x 9 layout, so the copied state -> action table is
        # simply replayed in the new world (unseen states: uniformly random actions)
        v_bc = world.value(pi_bc)
        v_me = world.value(world.soft_policy(world.phi @ omega_me))
        v_me_hard = world.value(world.optimal_policy(world.phi @ omega_me))
        v_ml = world.value(world.soft_policy(world.phi @ omega_ml))
        if world is train:
            v_proj = mixture_value(world, proj_pols, lam)
        else:   # re-plan with the last reward weights found by the projection method
            v_proj = world.value(world.optimal_policy(world.phi @ w_proj))
        v_rand = world.value(np.full((world.ns, NA), 1.0 / NA))
        rows.append((name, v_opt, v_soft, v_bc, v_proj, v_me, v_me_hard, v_rand, v_ml))
    print(f"\ntrue return E[sum_t r(s_t)] over H={H} steps")
    print(f"{'world':9s} {'optimal':>8s} {'soft expert':>11s} {'BC':>8s} {'projection':>10s} "
          f"{'MaxEnt(soft)':>12s} {'MaxEnt(greedy)':>14s} {'random':>8s} {'MaxEnt-ML(soft)':>15s}")
    for r in rows:
        print(f"{r[0]:9s} " + " ".join(f"{x:{wd}.2f}" for x, wd in zip(r[1:], [8, 11, 8, 10, 12, 14, 8, 15])))
    print(f"total time {time.time() - t0:.1f}s")

    if quick:
        return
    import plot_style
    plt = plot_style.setup()
    C = plot_style.C
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#d9d4c7", "#b7d9a8", "#8a6d4b", "#f2c14e"])

    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 3)
    # (a) training map with demonstration visitation
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(train.terrain.reshape(train.h, train.w), cmap=cmap, vmin=0, vmax=3)
    vis = np.zeros(train.ns)
    for traj in demos:
        for s, _ in traj:
            vis[s] += 1
    rr, cc = np.divmod(np.arange(train.ns), train.w)
    ax.scatter(cc, rr, s=4 * vis, color=C[0], alpha=0.7)
    ax.set_title("Training world: road (grey), grass, mud (brown), goal;\ndots = visits by the 30 demonstrations", fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    # (b) transfer map with visitation of BC and MaxEnt policies
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(test.terrain.reshape(test.h, test.w), cmap=cmap, vmin=0, vmax=3)
    D_bc = test.visitation(pi_bc).sum(0)
    D_me = test.visitation(test.soft_policy(test.phi @ omega_me)).sum(0)
    rr, cc = np.divmod(np.arange(test.ns), test.w)
    ax.scatter(cc - 0.17, rr, s=12 * D_bc, color=C[1], alpha=0.8, label="BC (copied actions)")
    ax.scatter(cc + 0.17, rr, s=12 * D_me, color=C[0], alpha=0.8, label="MaxEnt IRL (re-planned)")
    ax.set_title("Transfer world (start bottom-left): expected visits", fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    leg = ax.legend(loc="upper right", fontsize=7.5, markerscale=1.0)
    for h in leg.legend_handles:
        h.set_sizes([30])
    # (c) values
    ax = fig.add_subplot(gs[0, 2])
    labels = ["optimal", "soft expert", "BC", "projection", "MaxEnt", "random"]
    idx = [1, 2, 3, 4, 5, 7]
    x = np.arange(len(labels))
    for k, (r, col) in enumerate(zip(rows, [C[2], C[3]])):
        ax.bar(x + (k - 0.5) * 0.38, [r[i] for i in idx], width=0.38, color=col, label=f"{r[0]} world")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30)
    ax.axhline(0, color=plot_style.INK, lw=0.8)
    ax.set_ylabel(f"true return over H = {H} steps")
    ax.set_title("Who solves the changed world?")
    ax.legend(fontsize=8)
    # (d) MaxEnt learning curve
    ax = fig.add_subplot(gs[1, 0])
    ks = [h[0] for h in hist]
    ax.semilogy(ks, [h[1] for h in hist], color=C[0], label=rf"MaxEnt MAP ($\ell_2$={l2}) $\|\mu_E-\mu_\omega\|$")
    ax.semilogy([h[0] for h in hist_ml], [h[1] for h in hist_ml], color=C[0], ls=":", lw=1.6,
                label=r"MaxEnt max. likelihood $\|\mu_E-\mu_\omega\|$")
    ax.set_xlabel("MaxEnt gradient step")
    ax.set_ylabel("feature-expectation gap")
    ax2 = ax.twiny()
    ax2.semilogy(np.arange(len(dist)), dist, color=C[1], ls="--", marker="o", ms=3,
                 label=r"projection $\|\mu_E-\bar\mu\|$")
    ax2.set_xlabel("projection iteration", color=C[1])
    ax2.grid(False)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8)
    ax.set_title("Matching feature expectations", pad=32, fontsize=10)
    # (e) recovered vs true reward weights
    ax = fig.add_subplot(gs[1, 1])
    traj = np.array([h[2] for h in hist])
    traj_ml = np.array([h[2] for h in hist_ml])
    for j, f in enumerate(FEATURES):
        ax.plot(ks, traj[:, j], color=C[j], label=f"{f}")
        ax.plot(ks, traj_ml[:, j], color=C[j], ls="--", lw=1.2)
        ax.axhline(centred_true[j], color=C[j], ls=":", lw=1.2)
    ax.set_xlabel("MaxEnt gradient step")
    ax.set_ylabel(r"$\omega$")
    ax.set_title(f"Reward weights: MAP l2={l2} (solid), ML (dashed),\ntrue minus its mean (dotted)", fontsize=9.5)
    ax.legend(fontsize=8, ncol=2)
    # (f) recovered reward map
    ax = fig.add_subplot(gs[1, 2])
    im = ax.imshow((train.phi @ omega_me).reshape(train.h, train.w), cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.035)
    ax.set_title(r"Recovered reward $\hat r(s)=\hat\omega^\top\phi(s)$")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "irl_gridworld.png"))
    plt.close(fig)
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
