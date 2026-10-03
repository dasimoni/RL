"""TD(0) vs constant-alpha Monte Carlo on the 5-state random walk.

Chapter 05, Section 3 ("TD vs MC on the random walk"), reproducing the setting
of Sutton & Barto (2018), Example 6.2.

The Markov reward process (no actions -- this is pure *prediction*):

    [T0] - A - B - C - D - E - [T6]
     0     1   2   3   4   5    6        <- state indices used below

* every episode starts in C (index 3);
* each step moves left or right with probability 1/2;
* reward is 0 on every transition except entering the right terminal T6 (+1);
* gamma = 1, so v(s) = Pr{finish on the right | start in s} = s / 6.

We compare
    TD(0):            V(S_t) <- V(S_t) + alpha [R_{t+1} + gamma V(S_{t+1}) - V(S_t)]   (Eq. 5.2)
    constant-alpha MC: V(S_t) <- V(S_t) + alpha [G_t - V(S_t)]                          (Eq. 5.1)
with V initialised to 0.5 for every non-terminal state (and 0 for terminals).

Outputs (full mode):
    figures/random_walk_values.png   -- TD(0) estimates after 0, 1, 10, 100 episodes
    figures/random_walk_rms.png      -- RMS error vs episodes, averaged over runs

Run:  python code/ch05_temporal_difference/random_walk_td_vs_mc.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

N_STATES = 5                      # non-terminal states A..E -> indices 1..5
LEFT_TERMINAL, RIGHT_TERMINAL = 0, N_STATES + 1
START = 3                         # state C
TRUE_V = np.arange(1, N_STATES + 1) / (N_STATES + 1)   # 1/6, ..., 5/6
INIT_V = 0.5
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def generate_episode(rng: np.random.Generator) -> tuple[list[int], list[float]]:
    """Return the state sequence S_0..S_T and rewards R_1..R_T of one episode."""
    states, rewards = [START], []
    s = START
    while s not in (LEFT_TERMINAL, RIGHT_TERMINAL):
        s = s + (1 if rng.random() < 0.5 else -1)
        states.append(s)
        rewards.append(1.0 if s == RIGHT_TERMINAL else 0.0)
    return states, rewards


def td0_episode(V: np.ndarray, states, rewards, alpha: float, gamma: float = 1.0) -> None:
    """TD(0) prediction (Algorithm in Section 2): update *during* the episode.

    Because we pre-generate the episode the order of updates is identical to the
    online algorithm: each update uses V as modified by all earlier updates.
    V[terminal] is always 0, so the target of the final step is just R_T.
    """
    for t in range(len(rewards)):
        s, s_next, r = states[t], states[t + 1], rewards[t]
        td_error = r + gamma * V[s_next] - V[s]          # delta_t (Eq. 5.3)
        V[s] += alpha * td_error


def mc_episode(V: np.ndarray, states, rewards, alpha: float, gamma: float = 1.0) -> None:
    """Every-visit constant-alpha MC: wait until the end, then move every V(S_t) towards G_t."""
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = gamma * G + rewards[t]                       # G_t = R_{t+1} + gamma G_{t+1}
        V[states[t]] += alpha * (G - V[states[t]])


def rms_error(V: np.ndarray) -> float:
    """Root-mean-square error over the 5 non-terminal states (uniform weighting)."""
    return float(np.sqrt(np.mean((V[1:N_STATES + 1] - TRUE_V) ** 2)))


def run_experiment(method: str, alpha: float, n_runs: int, n_episodes: int, seed: int,
                   init=INIT_V, return_sq_errors: bool = False):
    """Return an (n_runs, n_episodes + 1) array of RMS errors (column 0 = before learning).

    With return_sq_errors=True also return the per-state squared errors, shape (runs, episodes+1, 5).
    """
    update = td0_episode if method == "TD" else mc_episode
    errors = np.zeros((n_runs, n_episodes + 1))
    sq = np.zeros((n_runs, n_episodes + 1, N_STATES))
    for run in range(n_runs):
        # Same seed per run for every (method, alpha): common random numbers make
        # the comparison less noisy (all methods see the same episodes).
        rng = np.random.default_rng(seed + run)
        V = np.zeros(N_STATES + 2)
        V[1:N_STATES + 1] = init
        errors[run, 0] = rms_error(V)
        sq[run, 0] = (V[1:N_STATES + 1] - TRUE_V) ** 2
        for ep in range(n_episodes):
            states, rewards = generate_episode(rng)
            update(V, states, rewards, alpha)
            errors[run, ep + 1] = rms_error(V)
            sq[run, ep + 1] = (V[1:N_STATES + 1] - TRUE_V) ** 2
    return (errors, sq) if return_sq_errors else errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    n_runs = 20 if args.quick else 500
    n_episodes = 100
    td_alphas = [0.15, 0.10, 0.05]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    print(f"[random_walk_td_vs_mc] seed={args.seed} runs={n_runs} episodes={n_episodes} "
          f"init V=0.5 gamma=1  TD alphas={td_alphas}  MC alphas={mc_alphas}")
    t0 = time.time()

    # --- Left panel: one run of TD(0), alpha=0.1, snapshots of V -------------------------
    rng = np.random.default_rng(args.seed)
    V = np.zeros(N_STATES + 2)
    V[1:N_STATES + 1] = INIT_V
    snapshots = {0: V[1:N_STATES + 1].copy()}
    for ep in range(1, 101):
        states, rewards = generate_episode(rng)
        td0_episode(V, states, rewards, alpha=0.1)
        if ep in (1, 10, 100):
            snapshots[ep] = V[1:N_STATES + 1].copy()
    print("TD(0), alpha=0.1, single run -- estimates of v(A..E):")
    for ep, v in snapshots.items():
        print(f"  after {ep:3d} episodes: " + " ".join(f"{x:.3f}" for x in v))
    print("  true values:          " + " ".join(f"{x:.3f}" for x in TRUE_V))

    # --- Right panel: RMS error learning curves ------------------------------------------
    curves = {}
    for a in td_alphas:
        curves[("TD", a)] = run_experiment("TD", a, n_runs, n_episodes, args.seed + 1000).mean(axis=0)
    for a in mc_alphas:
        curves[("MC", a)] = run_experiment("MC", a, n_runs, n_episodes, args.seed + 1000).mean(axis=0)

    print(f"\nRMS error averaged over {n_runs} runs (and over the 5 states):")
    print("  method  alpha | ep 0   ep 10  ep 25  ep 50  ep 100 | min (episode)")
    for (m, a), c in curves.items():
        print(f"  {m:6s} {a:5.2f} | {c[0]:.3f}  {c[10]:.3f}  {c[25]:.3f}  {c[50]:.3f}  {c[100]:.3f} "
              f"| {c.min():.3f} ({int(c.argmin())})")

    # --- Why does TD with alpha=0.15 dip below its final error? (Exercise 5.9) -------------
    # Compare three initialisations.  The TD target R + V(S') has a variance that depends on V
    # itself: with a flat V (all 0.5) the targets of interior states are nearly constant.
    print(f"\nTD(0), alpha=0.15, effect of the initial values ({n_runs} runs):")
    print("  init V       | min mean-RMS (episode) | mean-RMS ep 100 | per-state MSE at episode 20 (A..E)")
    for name, init in (("0.5 (flat)", INIT_V), ("v_pi (exact)", TRUE_V), ("0.0", 0.0)):
        e, sq = run_experiment("TD", 0.15, n_runs, n_episodes, args.seed + 2000, init=init,
                               return_sq_errors=True)
        c = e.mean(axis=0)
        print(f"  {name:12s} | {c.min():.4f} ({int(c.argmin()):3d})           | {c[100]:.4f}          | "
              + " ".join(f"{x:.4f}" for x in sq[:, 20].mean(axis=0)))

    if not args.quick:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(FIG_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        labels = ["A", "B", "C", "D", "E"]
        ax.plot(labels, TRUE_V, "k--", lw=2, label="true values")
        for ep, v in snapshots.items():
            ax.plot(labels, v, "o-", label=f"after {ep} episode{'s' if ep != 1 else ''}")
        ax.set_xlabel("state")
        ax.set_ylabel("estimated value")
        ax.set_title("TD(0) on the random walk (one run, α = 0.1)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "random_walk_values.png"), dpi=110)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        td_colors = plt.cm.Blues(np.linspace(0.45, 0.95, len(td_alphas)))
        mc_colors = plt.cm.Reds(np.linspace(0.45, 0.95, len(mc_alphas)))
        for a, col in zip(td_alphas, td_colors):
            ax.plot(curves[("TD", a)], color=col, label=f"TD(0), α={a}")
        for a, col in zip(mc_alphas, mc_colors):
            ax.plot(curves[("MC", a)], color=col, ls="--", label=f"MC, α={a}")
        ax.set_xlabel("episodes (walks)")
        ax.set_ylabel("RMS error, averaged over states")
        ax.set_title(f"Random walk: TD(0) vs constant-α MC ({n_runs} runs)")
        ax.set_ylim(0, 0.26)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "random_walk_rms.png"), dpi=110)
        plt.close(fig)
        print(f"\nFigures written to {FIG_DIR}")

    best_td = min((c[100], a) for (m, a), c in curves.items() if m == "TD")
    best_mc = min((c[100], a) for (m, a), c in curves.items() if m == "MC")
    print(f"\nSummary: after 100 episodes best TD(0) RMS = {best_td[0]:.3f} (alpha={best_td[1]}), "
          f"best MC RMS = {best_mc[0]:.3f} (alpha={best_mc[1]}).  Runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
