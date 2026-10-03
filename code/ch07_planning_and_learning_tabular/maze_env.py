"""Deterministic gridworld mazes used throughout Chapter 07.

This is a *module*, imported by dyna_maze.py, changing_mazes.py and
prioritized_sweeping.py; it is not meant to be run on its own.

Conventions (Chapter 07, Section 3):
  * cells are (row, col) with row 0 at the TOP of the picture;
  * states are integer indices s = row * n_cols + col;
  * 4 actions: 0=up, 1=down, 2=left, 3=right;
  * moving into a wall or off the grid leaves the agent where it is;
  * reward is +1 on the transition INTO the goal, 0 otherwise; reaching the
    goal terminates the episode (a true termination: no bootstrapping).

For speed the full transition table is precomputed as plain Python lists,
because the Dyna inner loops do millions of tiny lookups and numpy scalar
indexing is much slower than list indexing for that access pattern.
"""

from __future__ import annotations

from collections import deque

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
N_ACTIONS = len(ACTIONS)
ARROWS = ["↑", "↓", "←", "→"]  # for text/figure rendering


class Maze:
    """A deterministic gridworld with walls, a start cell and a goal cell."""

    def __init__(self, n_rows, n_cols, start, goal, walls):
        self.n_rows, self.n_cols = n_rows, n_cols
        self.start_cell, self.goal_cell = tuple(start), tuple(goal)
        self.n_states = n_rows * n_cols
        self.start = self.index(self.start_cell)
        self.goal = self.index(self.goal_cell)
        self.set_walls(walls)

    # ----------------------------------------------------------------- helpers
    def index(self, cell):
        return cell[0] * self.n_cols + cell[1]

    def cell(self, s):
        return divmod(s, self.n_cols)

    def set_walls(self, walls):
        """(Re)build the transition table. Used to change the maze mid-run
        (blocking / shortcut mazes, Section 4)."""
        self.walls = {tuple(w) for w in walls}
        assert self.start_cell not in self.walls and self.goal_cell not in self.walls
        # table[s][a] = (next_state, reward, terminated)
        self.table = []
        for s in range(self.n_states):
            r, c = self.cell(s)
            row = []
            for dr, dc in ACTIONS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols) or (nr, nc) in self.walls:
                    nr, nc = r, c  # bump: stay in place
                s2 = self.index((nr, nc))
                if s2 == self.goal:
                    row.append((s2, 1.0, True))
                else:
                    row.append((s2, 0.0, False))
            self.table.append(row)

    def step(self, s, a):
        return self.table[s][a]

    def free_states(self):
        """All non-wall, non-goal states (the states an agent can occupy)."""
        return [s for s in range(self.n_states)
                if self.cell(s) not in self.walls and s != self.goal]

    def shortest_path_length(self):
        """Breadth-first search from start to goal (number of moves)."""
        dist = {self.start: 0}
        frontier = deque([self.start])
        while frontier:
            s = frontier.popleft()
            if s == self.goal:
                return dist[s]
            for a in range(N_ACTIONS):
                s2 = self.table[s][a][0]
                if s2 not in dist:
                    dist[s2] = dist[s] + 1
                    frontier.append(s2)
        return None  # goal unreachable

    def scaled(self, factor):
        """Return the same maze at a finer resolution: every cell becomes a
        factor x factor block (Peng & Williams-style scaling used in the
        prioritized sweeping experiment, Section 5)."""
        walls = [(r * factor + i, c * factor + j)
                 for (r, c) in self.walls for i in range(factor) for j in range(factor)]
        start = (self.start_cell[0] * factor, self.start_cell[1] * factor)
        goal = (self.goal_cell[0] * factor, self.goal_cell[1] * factor + factor - 1)
        return Maze(self.n_rows * factor, self.n_cols * factor, start, goal, walls)

    def render(self, policy_arrows=None):
        """ASCII picture of the maze; optionally overlay greedy-action arrows."""
        lines = []
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                s = self.index((r, c))
                if (r, c) in self.walls:
                    row.append("#")
                elif s == self.start:
                    row.append("S")
                elif s == self.goal:
                    row.append("G")
                elif policy_arrows is not None and policy_arrows.get(s):
                    row.append(policy_arrows[s])
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)


# --------------------------------------------------------------------- mazes
def dyna_maze():
    """The 6x9 maze of Sutton & Barto (2018) Example 8.1 / Figure 8.2."""
    walls = [(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)]
    return Maze(6, 9, start=(2, 0), goal=(0, 8), walls=walls)


def blocking_maze():
    """S&B Example 8.2. Returns (maze, walls_before, walls_after).
    Initially the gap in the long wall is on the RIGHT; after the change the
    right gap is closed and a new gap opens on the LEFT (longer path)."""
    before = [(3, c) for c in range(0, 8)]
    after = [(3, c) for c in range(1, 9)]
    return Maze(6, 9, start=(5, 3), goal=(0, 8), walls=before), before, after


def shortcut_maze():
    """S&B Example 8.3. Returns (maze, walls_before, walls_after).
    Initially the only gap is on the LEFT; after the change a SHORTCUT opens
    on the right while the left path stays open."""
    before = [(3, c) for c in range(1, 9)]
    after = [(3, c) for c in range(1, 8)]
    return Maze(6, 9, start=(5, 3), goal=(0, 8), walls=before), before, after
