"""Successor representation, successor features and generalized policy improvement (GPI).

Chapter 15, Section 6 (Algorithm 15.9).  Uses the slippery four-rooms dynamics of four_rooms_options.py.

Part 1 - the successor representation (SR) of the uniform random policy,
        M = sum_k gamma^k P_pi^k = (I - gamma P_pi)^{-1}            (Eq. 15.25)
    * v_pi = M r_pi for ANY reward: one matrix, many value functions (checked numerically);
    * M learned by TD (Eq. 15.26) with a constant step size approaches the exact M and then
      fluctuates around it (about 10% mean relative error at alpha = 0.1);
    * the slow eigenvectors of M (= those of P_pi) are nearly constant within rooms and vary most
      across the hallways: the bottleneck structure that "eigenoption" methods exploit.  We print,
      for eigenvectors 2-4, the fraction of each room's cells where the eigenvector is positive.

Part 2 - successor features (SFs) and GPI for zero-shot transfer (Barreto et al., 2017).
    Four objects, one in a corner of each room.  Entering an object's cell ends the episode and
    yields the feature phi = e_type (a one-hot vector in R^4); the reward of task w is phi . w.
    Base tasks w_i = e_i ("go to object i").  For each base policy pi_i we compute its SFs
        psi^{pi_i}(s, a) = E[ sum_k gamma^k phi_{t+k+1} | S_t = s, A_t = a, pi_i ]   (Eq. 15.27)
    so that q^{pi_i}_w(s, a) = psi^{pi_i}(s, a) . w for every w, at no extra cost.  On a NEW task
    w the GPI policy  pi(s) = argmax_a max_i psi^{pi_i}(s, a) . w  (Eq. 15.29) is evaluated exactly
    and compared with each base policy and with the optimal policy for w.

Run:  python code/ch15_beyond_mdps/successor_features_gpi.py [--quick]
Full mode writes figures/sr_four_rooms.png and figures/sf_gpi.png.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import time

import numpy as np

from four_rooms_options import FourRooms, LAYOUT

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
GAMMA = 0.95
OBJECTS = [(1, 1), (1, 11), (11, 1), (11, 11)]     # one per room, object type i at OBJECTS[i]


# ----------------------------------------------------------------------------------------------
# Part 1: the successor representation
# ----------------------------------------------------------------------------------------------
def successor_representation(env):
    P_pi = env.P.mean(axis=0)                                  # uniform random policy
    return np.linalg.inv(np.eye(env.N) - GAMMA * P_pi), P_pi


def td_learn_sr(env, rng, M_true, n_steps, alpha=0.1, log_every=5_000):
    """Tabular TD learning of the SR (Eq. 15.26): M(s,.) += alpha [1_s + gamma M(s',.) - M(s,.)]."""
    M = np.zeros((env.N, env.N))
    s = rng.integers(env.N)
    curve = []
    for t in range(1, n_steps + 1):
        s2 = env.step(rng, s, rng.integers(4))
        target = GAMMA * M[s2]
        target[s] += 1.0                    # the indicator of the CURRENT state (k = 0 term)
        M[s] += alpha * (target - M[s])
        s = s2
        if t % log_every == 0:
            curve.append((t, np.abs(M - M_true).max() / M_true.max(), np.abs(M - M_true).mean()))
    return M, np.array(curve)


# ----------------------------------------------------------------------------------------------
# Part 2: successor features and GPI
# ----------------------------------------------------------------------------------------------
class ObjectTasks:
    """Episodic tasks: entering an object cell ends the episode with feature phi = e_type."""

    def __init__(self, env):
        self.env = env
        self.N = env.N
        self.d = len(OBJECTS)
        self.phi_next = np.zeros((self.N, self.d))          # phi as a function of the next state
        self.terminal = np.zeros(self.N, dtype=bool)
        for i, rc in enumerate(OBJECTS):
            j = env.idx[rc]
            self.phi_next[j, i] = 1.0
            self.terminal[j] = True
        # expected immediate feature  E[phi(S') | s, a] and "continue" mask for bootstrapping
        self.Ephi = np.einsum("asn,nd->sad", env.P, self.phi_next)        # (N, 4, d)
        self.Pc = env.P * (~self.terminal)[None, None, :]                  # P(s'|s,a) 1[s' non-terminal]

    def optimal_q(self, w, tol=1e-12):
        """Value iteration for task w: q(s,a) = E[phi.w + gamma 1[non-term] max_a' q(s',a')]."""
        r = self.Ephi @ w                                   # (N, 4)
        q = np.zeros((self.N, 4))
        for _ in range(5000):
            v = q.max(axis=1)
            q_new = r + GAMMA * np.einsum("asn,n->sa", self.Pc, v)
            if np.abs(q_new - q).max() < tol:
                break
            q = q_new
        return q_new

    def policy_sf(self, pi):
        """SFs of a deterministic policy (Eq. 15.27) by a linear solve.

        psi(s,a) = E[phi(S')|s,a] + gamma sum_{s'} P(s'|s,a) 1[s' non-term] psi(s', pi(s')).
        First solve for psi_pi(s) = psi(s, pi(s)), then one backup gives psi(s, a) for all a.
        """
        P_pi = self.Pc[pi, np.arange(self.N)]               # (N, N), rows s, cols s'
        Ephi_pi = self.Ephi[np.arange(self.N), pi]          # (N, d)
        psi_v = np.linalg.solve(np.eye(self.N) - GAMMA * P_pi, Ephi_pi)
        return self.Ephi + GAMMA * np.einsum("asn,nd->sad", self.Pc, psi_v)   # (N, 4, d)

    def evaluate(self, pi, w):
        """Exact v^pi_w by policy evaluation (a linear solve) for the deterministic policy pi."""
        P_pi = self.Pc[pi, np.arange(self.N)]
        r_pi = (self.Ephi @ w)[np.arange(self.N), pi]
        return np.linalg.solve(np.eye(self.N) - GAMMA * P_pi, r_pi)


def gpi_policy(psis, w):
    """GPI (Eq. 15.29): pi(s) = argmax_a max_i psi_i(s,a).w ; also returns which i is used."""
    Qs = np.stack([psi @ w for psi in psis])                 # (n_policies, N, 4)
    Qmax = Qs.max(axis=0)
    pi = Qmax.argmax(axis=1)
    which = Qs.max(axis=2).argmax(axis=0)
    return pi, which


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figures")
    ap.add_argument("--tasks", type=int, default=500)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)
    n_tasks = 40 if args.quick else args.tasks
    n_td = 30_000 if args.quick else 600_000
    print(f"SR / SF / GPI in the four rooms | seed=0 gamma={GAMMA} objects={OBJECTS} "
          f"random test tasks={n_tasks} TD steps={n_td} quick={args.quick}")
    env = FourRooms()

    # ---- Part 1: SR -------------------------------------------------------------------------
    M, P_pi = successor_representation(env)
    r = rng.normal(size=env.N)
    v_direct = np.linalg.solve(np.eye(env.N) - GAMMA * P_pi, r)
    print(f"[SR] row sums of M = 1/(1-gamma) = {1 / (1 - GAMMA):.1f}: max deviation "
          f"{np.abs(M.sum(axis=1) - 1 / (1 - GAMMA)).max():.1e}; |M r - v_pi| for a random reward: "
          f"{np.abs(M @ r - v_direct).max():.1e}")
    M_td, td_curve = td_learn_sr(env, np.random.default_rng(1), M, n_td)
    print(f"[SR] TD-learned SR (alpha=0.1, random walk): max |M_td - M| / max M = "
          f"{td_curve[0, 1]:.3f} after {int(td_curve[0, 0])} steps, {td_curve[-1, 1]:.3f} after "
          f"{int(td_curve[-1, 0])} steps (mean abs error {td_curve[-1, 2]:.4f})")
    evals, evecs = np.linalg.eigh((M + M.T) / 2)            # M is symmetric here (P_pi is)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    evecs = evecs * np.where(evecs.sum(axis=0) < 0, -1.0, 1.0)   # sign convention: entries sum to >= 0
    print(f"[SR] asymmetry max |M - M^T| = {np.abs(M - M.T).max():.1e}; top eigenvalues of M: "
          + ", ".join(f"{e:.2f}" for e in evals[:5]))
    rooms = ["NW", "NE", "SW", "SE"]
    hall_cells = [i for i in range(env.N) if i not in env.room_of]
    for k in (1, 2, 3):
        v = evecs[:, k]
        frac = {rm: np.mean([v[i] > 0 for i in range(env.N) if env.room_of.get(i) == rm]) for rm in rooms}
        print(f"[SR] eigenvector {k + 1} (eigenvalue {evals[k]:.2f}): fraction of cells > 0 per room "
              + ", ".join(f"{rm} {frac[rm]:.2f}" for rm in rooms)
              + "; at the hallways " + ", ".join(f"{env.cells[i]} {v[i]:+.3f}" for i in hall_cells))

    # ---- Part 2: SFs and GPI ----------------------------------------------------------------
    tasks = ObjectTasks(env)
    base_w = [np.eye(4)[i] for i in range(4)]
    base_pi, psis = [], []
    worst_sf = 0.0
    for w in base_w:
        q = tasks.optimal_q(w)
        pi = q.argmax(axis=1)
        psi = tasks.policy_sf(pi)
        base_pi.append(pi)
        psis.append(psi)
        # check: psi . w' = q^pi_{w'} for the base task AND an unrelated task w'
        for w2 in [w, rng.normal(size=4)]:
            v = tasks.evaluate(pi, w2)
            worst_sf = max(worst_sf, np.abs((psi @ w2)[np.arange(env.N), pi] - v).max())
    print(f"[SF] max |psi^pi . w - v^pi_w| over base policies and test rewards: {worst_sf:.1e}")

    non_obj = ~tasks.terminal
    start = env.idx[(6, 9)] if (6, 9) in env.idx else 0

    def summary(w):
        q_star = tasks.optimal_q(w)
        v_star = q_star.max(axis=1)
        v_base = np.stack([tasks.evaluate(pi, w) for pi in base_pi])
        pi_gpi, which = gpi_policy(psis, w)
        v_gpi = tasks.evaluate(pi_gpi, w)
        return v_star, v_base, v_gpi, which

    showcase = {"w = (1, 1, 0, 0)": np.array([1.0, 1.0, 0.0, 0.0]),
                "w = (1, -1, 0.6, 0)": np.array([1.0, -1.0, 0.6, 0.0]),
                "w = (-1, -1, -1, -1)": -np.ones(4)}
    show = {}
    for name, w in showcase.items():
        v_star, v_base, v_gpi, which = summary(w)
        show[name] = (v_star, v_base, v_gpi, which)
        gpi_ge = np.all(v_gpi[non_obj] >= v_base[:, non_obj].max(axis=0) - 1e-10)
        print(f"[GPI] {name:<20} mean over states: v* = {v_star[non_obj].mean():.4f}, "
              f"best single base policy = {v_base[:, non_obj].mean(axis=1).max():.4f}, "
              f"GPI = {v_gpi[non_obj].mean():.4f}; GPI >= max_i v^pi_i everywhere: {gpi_ge}; "
              f"states where GPI is optimal (1e-9): {np.mean(np.abs(v_gpi - v_star)[non_obj] < 1e-9):.2f}")

    # random test tasks: w ~ uniform on [-1, 1]^4
    rows = []
    for _ in range(n_tasks):
        w = rng.uniform(-1, 1, size=4)
        v_star, v_base, v_gpi, _ = summary(w)
        ok = np.all(v_gpi[non_obj] >= v_base[:, non_obj].max(axis=0) - 1e-10)
        # regrets are >= 0 by optimality of v*; clip round-off of order 1e-15
        gap_best = max(0.0, (v_star[None, non_obj] - v_base[:, non_obj]).mean(axis=1).min())
        gap_gpi = max(0.0, (v_star[non_obj] - v_gpi[non_obj]).mean())
        rows.append((ok, gap_best, gap_gpi, (w > 0).sum(), v_star[non_obj].mean()))
    rows = np.array(rows, dtype=float)

    def fmt(x):                                  # small regrets in scientific notation
        return f"{x:.4f}" if x >= 1e-3 or x == 0 else f"{x:.1e}"
    print(f"[GPI] {n_tasks} random tasks w ~ U[-1,1]^4: GPI >= every base policy at every state in "
          f"{rows[:, 0].mean() * 100:.0f}% of tasks")
    for k in range(5):
        sel = rows[:, 3] == k
        if sel.any():
            print(f"      tasks with {k} positive weights ({int(sel.sum()):3d}): mean v* = {rows[sel, 4].mean():.4f}; "
                  f"mean regret (v* - v), averaged over states: best single base policy "
                  f"{fmt(rows[sel, 1].mean())}, GPI {fmt(rows[sel, 2].mean())}")

    if not args.quick:
        from plot_style import setup, C
        plt = setup()
        os.makedirs(FIG_DIR, exist_ok=True)
        walls = np.array([[1.0 if ch == "w" else np.nan for ch in row] for row in LAYOUT])

        def to_img(v):
            img = np.full(walls.shape, np.nan)
            for k, (rr, cc) in enumerate(env.cells):
                img[rr, cc] = v[k]
            return img

        def draw(ax, v, title, cmap="viridis", vmin=None, vmax=None, cbar=False):
            ax.imshow(walls, cmap="Greys", vmin=0, vmax=1.5)
            im = ax.imshow(to_img(v), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(title, fontsize=10)
            if cbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            return im

        fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
        s0 = env.idx[(2, 3)]
        draw(axes[0], M[s0], "SR row M(s0, ·), random policy", cbar=True)
        axes[0].text(3, 2, "s0", ha="center", va="center", color="white", fontsize=8)
        for j, k in enumerate([1, 2, 3]):
            vmax = np.abs(evecs[:, k]).max()
            draw(axes[1 + j], evecs[:, k], f"eigenvector {k + 1} of M (eigenvalue {evals[k]:.1f})",
                 cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax = axes[4]
        ax.semilogy(td_curve[:, 0] / 1000, td_curve[:, 1], color=C[0], label="max error / max M")
        ax.semilogy(td_curve[:, 0] / 1000, td_curve[:, 2] / M.mean(), color=C[1], ls="--",
                    label="mean error / mean M")
        ax.set_xlabel("thousands of random-walk steps")
        ax.set_title("TD learning of the SR (α = 0.1)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "sr_four_rooms.png"))
        plt.close(fig)

        fig = plt.figure(figsize=(15, 7.8), layout="constrained")
        gs = fig.add_gridspec(2, 4)
        name = "w = (1, -1, 0.6, 0)"
        v_star, v_base, v_gpi, which = show[name]
        panels = [(v_base[0], "v of π1 (go to object 1)"), (v_base[2], "v of π3 (go to object 3)"),
                  (v_gpi, "v of the GPI policy"), (v_star, "v* (optimal for this w)")]
        vmin = min(p_[0][non_obj].min() for p_ in panels)
        vmax = max(p_[0][non_obj].max() for p_ in panels)
        top_axes = []
        for j, (v, title) in enumerate(panels):
            ax = fig.add_subplot(gs[0, j])
            im = draw(ax, v, title, vmin=vmin, vmax=vmax)
            top_axes.append(ax)
            for i, (rr, cc) in enumerate(OBJECTS):
                ax.text(cc, rr, str(i + 1), ha="center", va="center", color="white", fontsize=9, weight="bold")
        fig.colorbar(im, ax=top_axes, fraction=0.015, pad=0.01, label="value (task w)")
        fig.suptitle(f"Task {name}: objects 1..4 sit in the four corners; base policies π_i = optimal for e_i")
        ax = fig.add_subplot(gs[1, 0])
        img = np.full(walls.shape, np.nan)
        for k, (rr, cc) in enumerate(env.cells):
            img[rr, cc] = which[k]
        ax.imshow(walls, cmap="Greys", vmin=0, vmax=1.5)
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        ax.imshow(img, cmap=ListedColormap(C[:4]), vmin=-0.5, vmax=3.5)
        for i, (rr, cc) in enumerate(OBJECTS):
            ax.text(cc, rr, str(i + 1), ha="center", va="center", color="white", fontsize=9, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title("base policy GPI follows in each state", fontsize=10)
        used = sorted(set(int(x) for x in which[non_obj]))
        ax.legend(handles=[Patch(color=C[i], label=f"π{i + 1}") for i in used], loc="lower center",
                  bbox_to_anchor=(0.5, -0.16), ncol=len(used), fontsize=9)
        ax = fig.add_subplot(gs[1, 1:])
        labels = []
        best, gpi = [], []
        for k in range(5):
            sel = rows[:, 3] == k
            if sel.any():
                labels.append(f"{k} positive weights\n(n = {int(sel.sum())} tasks)")
                best.append(rows[sel, 1].mean())
                gpi.append(rows[sel, 2].mean())
        x = np.arange(len(labels))
        ax.bar(x - 0.18, best, width=0.36, color=C[0], label="best single base policy")
        ax.bar(x + 0.18, gpi, width=0.36, color=C[2], label="GPI over the 4 base policies")
        for xi, (b_, g_) in enumerate(zip(best, gpi)):
            ax.text(xi - 0.18, b_ + 0.001, fmt(b_), ha="center", fontsize=8)
            ax.text(xi + 0.18, g_ + 0.001, fmt(g_), ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("mean regret  v*(s) − v(s)")
        ax.set_title(f"{n_tasks} random tasks w ~ U[-1,1]^4, grouped by number of positive weights")
        ax.legend(loc="upper center")
        fig.savefig(os.path.join(FIG_DIR, "sf_gpi.png"))
        plt.close(fig)
        print(f"figures written to {FIG_DIR}")
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
