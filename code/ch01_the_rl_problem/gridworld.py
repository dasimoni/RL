"""The 5x5 gridworld of Sutton & Barto (2018), Example 3.5 - as a model AND as an environment.

Rules (chapter Section 6.4):
* states: the 25 cells; s = 5 * row + col, row 0 at the top;
* actions: north, south, east, west - deterministic moves;
* an action that would leave the grid leaves the agent in place, reward -1;
* every action taken in A gives +10 and teleports the agent to A';
  every action taken in B gives +5 and teleports the agent to B';
* every other move gives reward 0.  The task is continuing (no terminal state).

The class exposes the environment in the two ways the chapter distinguishes:
* ``to_mdp()``  - the full model p(s', r | s, a) (what dynamic programming needs);
* ``reset() / step()`` - a Gymnasium-style sampling interface (all that a
  model-free learner gets): ``obs, info = env.reset(seed=...)`` and
  ``obs, reward, terminated, truncated, info = env.step(action)``.

Run this file to print the dynamics of a few states and play a few random steps;
in full mode it also draws the layout figure.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mdp import FiniteMDP

ACTION_NAMES = ["north", "south", "east", "west"]
ACTION_DELTAS = [(-1, 0), (1, 0), (0, 1), (0, -1)]
ARROWS = ["↑", "↓", "→", "←"]  # up, down, right, left


class GridWorld:
    """Sutton & Barto Example 3.5 (defaults) or a variant with other sizes/rewards."""

    def __init__(self, size: int = 5, a_pos=(0, 1), a_prime=(4, 1), b_pos=(0, 3),
                 b_prime=(2, 3), r_a: float = 10.0, r_b: float = 5.0,
                 off_grid_reward: float = -1.0, max_steps: int | None = None):
        self.size = size
        self.a_pos, self.a_prime, self.b_pos, self.b_prime = a_pos, a_prime, b_pos, b_prime
        self.r_a, self.r_b, self.off_grid_reward = r_a, r_b, off_grid_reward
        self.n_states, self.n_actions = size * size, 4
        self.max_steps = max_steps  # optional time limit -> truncation, never termination
        self._rng = np.random.default_rng()
        self._state = None
        self._t = 0

    # --------------------------------------------------------------- indices
    def index(self, row: int, col: int) -> int:
        return row * self.size + col

    def coords(self, s: int) -> tuple[int, int]:
        return divmod(s, self.size)

    # ----------------------------------------------------- the dynamics rule
    def transition(self, s: int, a: int) -> tuple[int, float]:
        """Deterministic (s, a) -> (s', r).  Section 6.4 lists the rules."""
        row, col = self.coords(s)
        if (row, col) == self.a_pos:
            return self.index(*self.a_prime), self.r_a          # +10 and jump to A'
        if (row, col) == self.b_pos:
            return self.index(*self.b_prime), self.r_b          # +5 and jump to B'
        dr, dc = ACTION_DELTAS[a]
        r2, c2 = row + dr, col + dc
        if not (0 <= r2 < self.size and 0 <= c2 < self.size):
            return s, self.off_grid_reward  # bumped into the wall: stay, pay -1
        return self.index(r2, c2), 0.0

    def tables(self) -> tuple[np.ndarray, np.ndarray]:
        """next_state[s, a] and reward[s, a] arrays (handy for fast vectorized simulation)."""
        nxt = np.zeros((self.n_states, self.n_actions), dtype=int)
        rew = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                nxt[s, a], rew[s, a] = self.transition(s, a)
        return nxt, rew

    def to_mdp(self) -> FiniteMDP:
        """The model: four-argument dynamics p(s', r | s, a), here all probabilities are 1."""
        outcomes = [[[(1.0, *self.transition(s, a))] for a in range(self.n_actions)]
                    for s in range(self.n_states)]
        return FiniteMDP(self.n_states, self.n_actions, outcomes,
                         state_names=[self.state_label(s) for s in range(self.n_states)],
                         action_names=list(ACTION_NAMES))

    def state_label(self, s: int) -> str:
        rc = self.coords(s)
        special = {self.a_pos: "A", self.a_prime: "A'", self.b_pos: "B", self.b_prime: "B'"}
        return special.get(rc, f"({rc[0]},{rc[1]})")

    # --------------------------------------------- Gymnasium-style interface
    def reset(self, seed: int | None = None, start: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = int(self._rng.integers(self.n_states)) if start is None else start
        self._t = 0
        return self._state, {}

    def step(self, action: int):
        s2, r = self.transition(self._state, action)
        self._state, self._t = s2, self._t + 1
        terminated = False  # continuing task: there is no terminal state
        truncated = self.max_steps is not None and self._t >= self.max_steps
        return s2, r, terminated, truncated, {}


# ------------------------------------------------------------------ drawing
def draw_policy(ax, world: GridWorld, action_sets, color: str = "#0b0b0b") -> None:
    """Arrows for every optimal action in every cell (ties drawn as several arrows)."""
    for s, acts in enumerate(action_sets):
        row, col = world.coords(s)
        for a in acts:
            dr, dc = ACTION_DELTAS[a]
            ax.annotate("", xy=(col + 0.36 * dc, row + 0.36 * dr), xytext=(col, row),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                        shrinkA=0, shrinkB=0, mutation_scale=11))


def draw_layout(ax, world: GridWorld) -> None:
    """The environment picture: cells, the special states, and the teleports."""
    from plotting import BLUE, INK, INK2, ORANGE  # local import keeps plotting optional
    n = world.size
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    for k in range(n + 1):
        ax.axhline(k - 0.5, color="#c3c2b7", lw=1)
        ax.axvline(k - 0.5, color="#c3c2b7", lw=1)
    for pos, lab, col in [(world.a_pos, "A", BLUE), (world.a_prime, "A'", BLUE),
                          (world.b_pos, "B", ORANGE), (world.b_prime, "B'", ORANGE)]:
        ax.text(pos[1], pos[0], lab, ha="center", va="center", fontsize=15,
                fontweight="bold", color=col)
    for src, dst, lab, col, rad in [(world.a_pos, world.a_prime, f"+{world.r_a:g}", BLUE, 0.35),
                                    (world.b_pos, world.b_prime, f"+{world.r_b:g}", ORANGE, 0.35)]:
        ax.annotate("", xy=(dst[1] + 0.22, dst[0] - 0.2), xytext=(src[1] + 0.22, src[0] + 0.2),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8,
                                    connectionstyle=f"arc3,rad={rad}", mutation_scale=14))
        # an arc3 curve bulges sideways by about rad * length / 2; put the label just beyond it
        length = abs(dst[0] - src[0]) - 0.4
        mid_r = (src[0] + dst[0]) / 2
        ax.text(src[1] + 0.22 - rad * length / 2 - 0.1, mid_r, lab, color=col, fontsize=11,
                ha="right", va="center", fontweight="bold")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("column", color=INK2)
    ax.set_ylabel("row", color=INK2)
    ax.grid(False)
    ax.set_title("Gridworld (Sutton & Barto Example 3.5)", color=INK)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the Example 3.5 gridworld")
    parser.add_argument("--quick", action="store_true", help="smoke test: no figure")
    args = parser.parse_args()
    seed = 0
    t0 = time.perf_counter()
    print(f"gridworld.py | seed={seed} quick={args.quick}")
    world = GridWorld()
    mdp = world.to_mdp()
    p, rewards = mdp.dynamics_tensor()
    print(f"|S| = {mdp.n_states}, |A| = {mdp.n_actions}, reward set R = {rewards.tolist()}")
    print(f"four-argument dynamics tensor p[s,a,s',r] has shape {p.shape}; "
          f"every p(.,.|s,a) sums to 1: {np.allclose(p.sum(axis=(2, 3)), 1.0)}")

    print("\nOutcomes p(s', r | s, a) > 0 for a few states:")
    for (row, col) in [(0, 1), (0, 3), (0, 0), (2, 2)]:
        s = world.index(row, col)
        cells = []
        for a in range(4):
            (prob, s2, r), = mdp.outcomes[s][a]
            cells.append(f"{ACTION_NAMES[a]:>5}: s'={world.state_label(s2):<6} r={r:+g} (p={prob:g})")
        print(f"  s = {world.state_label(s):<6} " + " | ".join(cells))
    R = mdp.expected_reward()
    print("\nExpected reward r(s, a) at the corner (0,0) for N,S,E,W:", R[0].tolist())

    print("\nA few steps of the agent-environment loop with a uniformly random policy:")
    rng = np.random.default_rng(seed)
    obs, info = world.reset(seed=seed, start=world.index(0, 2))
    for t in range(8):
        action = int(rng.integers(4))
        next_obs, reward, terminated, truncated, info = world.step(action)
        print(f"  t={t}: S_t={world.state_label(obs):<6} A_t={ACTION_NAMES[action]:<5} "
              f"-> R_t+1={reward:+g}, S_t+1={world.state_label(next_obs)}")
        obs = next_obs

    if not args.quick:
        import matplotlib.pyplot as plt
        from plotting import setup_style
        setup_style()
        fig, ax = plt.subplots(figsize=(4.6, 5.0))
        draw_layout(ax, world)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.text(0.5, 0.015, "moving off the grid: stay in place, reward -1;  any other move: reward 0",
                 ha="center", va="bottom", fontsize=8.5, color="#52514e")
        out = Path(__file__).parent / "figures" / "gridworld_layout.png"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        print(f"\nsaved {out}")
    print(f"done in {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
