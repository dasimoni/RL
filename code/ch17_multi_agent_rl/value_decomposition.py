"""What VDN and QMIX learn, and what the best monotonic fit would be: one-step
cooperative matrix games (Chapter 17, section 7).

Each method fits a joint action-value Q_tot(a1, a2) to the true team payoff r(a1, a2)
by minimising a (weighted) squared error over all 9 joint actions, sampled uniformly
(the "uniform exploration" setting used to compare decompositions in the QTRAN and
Weighted QMIX papers). Then each agent acts greedily on its own utility Q_i(a_i).

  central   unconstrained table Q(a1, a2)          (exact, but needs a joint argmax)
  VDN       Q_tot = Q_1(a1) + Q_2(a2)              (Sunehag et al., 2018)
  QMIX      Q_tot = g(Q_1(a1), Q_2(a2)), g monotone (Rashid et al., 2018): a 1-hidden-layer
            mixing network with non-negative weights (|W|) and ELU; with a single state
            the hypernetworks reduce to free parameters
  OW-QMIX   QMIX trained with the optimistic weighting of Weighted QMIX (Rashid et al., 2020):
            weight 1 where Q_tot underestimates the target, weight 0.1 elsewhere

Games: the climbing game and a non-monotonic game with payoff 8 on (A, A), -12 on the
rest of row A and column A, and 0 elsewhere (the matrix game of Son et al., 2019).

Best monotonic fit (exact). A matrix Z can be written as g(q1(a1), q2(a2)) with g
non-decreasing iff, for some ordering of the rows and of the columns, Z is non-decreasing
along every row and down every column (Exercise 17.9c). For a FIXED pair of orderings this
is a convex cone, and the least-squares projection of R onto it is a small quadratic
program, solved exactly here with NNLS on the dual (Moreau decomposition). Enumerating the
3! x 3! = 36 strict orderings gives the global optimum of the uniform-weight monotonic
projection (ties between rows or columns are limits of strict orderings, so weak orderings
add nothing). The union of 36 cones is NOT convex, which is why gradient training can stop
in a worse basin: for every trained QMIX seed the script also reports the optimum within
the cone of the ordering that seed learned.

Run:  python code/ch17_multi_agent_rl/value_decomposition.py [--quick]
"""
import argparse
import os
import time

import itertools

import numpy as np
import torch
from scipy.optimize import nnls

import games

torch.set_num_threads(1)


class Utilities(torch.nn.Module):
    """Per-agent utilities Q_i(a_i) for a single state: just two learnable vectors."""

    def __init__(self, n):
        super().__init__()
        self.q1 = torch.nn.Parameter(0.1 * torch.randn(n))
        self.q2 = torch.nn.Parameter(0.1 * torch.randn(n))


class MonotonicMixer(torch.nn.Module):
    """g(q1, q2) = w2^T ELU(W1 [q1, q2] + b1) + b2 with W1, w2 >= 0 (via abs), so that
    dg/dq_i >= 0: the QMIX monotonicity constraint (Eq. 7.4)."""

    def __init__(self, hidden=8):
        super().__init__()
        self.W1 = torch.nn.Parameter(torch.rand(2, hidden))
        self.b1 = torch.nn.Parameter(torch.zeros(hidden))
        self.w2 = torch.nn.Parameter(torch.rand(hidden, 1))
        self.b2 = torch.nn.Parameter(torch.zeros(1))

    def forward(self, qs):                       # qs: [batch, 2]
        h = torch.nn.functional.elu(qs @ self.W1.abs() + self.b1)
        return (h @ self.w2.abs() + self.b2).squeeze(-1)


def monotone_projection(R, rows, cols):
    """Least-squares projection of R onto {Z: Z non-decreasing along the row ordering `rows`
    and the column ordering `cols`} (orderings listed from lowest to highest).

    The set is a polyhedral cone K = {z : C z >= 0}. Its polar cone is {-C^T lam : lam >= 0},
    so by Moreau's decomposition P_K(r) = r + C^T lam*, with lam* = argmin_{lam >= 0}
    || C^T lam + r || -- a non-negative least-squares problem."""
    n = R.shape[0]
    pairs = []
    for k in range(n - 1):
        pairs += [(rows[k + 1] * n + j, rows[k] * n + j) for j in range(n)]   # down columns
        pairs += [(i * n + cols[k + 1], i * n + cols[k]) for i in range(n)]   # along rows
    C = np.zeros((len(pairs), n * n))
    for row, (hi, lo) in enumerate(pairs):
        C[row, hi], C[row, lo] = 1.0, -1.0
    r = R.reshape(-1)
    lam, _ = nnls(C.T, -r)
    z = r + C.T @ lam
    assert (C @ z).min() > -1e-9                 # feasible: monotone along both orderings
    return z.reshape(n, n), float(((z - r) ** 2).mean())


def best_monotonic_fit(R):
    """Global least-squares fit of R by a monotonic mixing g(q1(a1), q2(a2)): enumerate all
    strict row and column orderings and project onto each cone. Returns (Z, mse, greedy
    joint action = (top row, top column) of the best ordering)."""
    n = R.shape[0]
    best = None
    for rows in itertools.permutations(range(n)):
        for cols in itertools.permutations(range(n)):
            Z, mse = monotone_projection(R, rows, cols)
            if best is None or mse < best[1] - 1e-12:
                best = (Z, mse, (rows[-1], cols[-1]))
    return best


def fit(method, R, seed, steps, lr=0.03, ow_alpha=0.1):
    """Fit Q_tot to the payoff matrix R; return (Q_tot matrix, q1, q2, mse)."""
    torch.manual_seed(seed)
    n = R.shape[0]
    target = torch.tensor(R, dtype=torch.float32).reshape(-1)
    a1 = torch.arange(n).repeat_interleave(n)    # all 9 joint actions, uniform weight
    a2 = torch.arange(n).repeat(n)
    if method == "central":
        table = torch.nn.Parameter(torch.zeros(n * n))
        params = [table]
    else:
        util = Utilities(n)
        params = list(util.parameters())
        if method in ("QMIX", "OW-QMIX"):
            mixer = MonotonicMixer()
            params += list(mixer.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    def qtot():
        if method == "central":
            return table
        if method == "VDN":
            return util.q1[a1] + util.q2[a2]
        return mixer(torch.stack([util.q1[a1], util.q2[a2]], dim=1))

    for _ in range(steps):
        q = qtot()
        err = q - target
        if method == "OW-QMIX":
            w = torch.where(q.detach() < target, torch.ones_like(err), torch.full_like(err, ow_alpha))
            loss = (w * err ** 2).mean()
        else:
            loss = (err ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        Q = qtot().reshape(n, n).numpy()
        if method == "central":
            q1, q2 = Q.max(1), Q.max(0)          # each agent's best joint-table row/column
        else:
            q1, q2 = util.q1.numpy().copy(), util.q2.numpy().copy()
        mse = float(((Q.reshape(-1) - R.reshape(-1)) ** 2).mean())
    return Q, q1, q2, mse


def main():
    parser = argparse.ArgumentParser(description="VDN / QMIX fits vs the best monotonic fit on matrix games")
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    n_seeds = 2 if args.quick else 6
    steps = 600 if args.quick else 2500
    print(f"value_decomposition.py | seed={args.seed} | seeds per method={n_seeds} | Adam lr=0.03, "
          f"{steps} full-batch steps, uniform data over the 9 joint actions | QMIX hidden=8 | "
          f"OW-QMIX alpha=0.1 | quick={args.quick}")
    t0 = time.time()
    climb, _, _ = games.climbing_game()
    nonmono = np.array([[8.0, -12.0, -12.0], [-12.0, 0.0, 0.0], [-12.0, 0.0, 0.0]])
    gms = {"climbing game": (climb, "abc"), "non-monotonic game": (nonmono, "ABC")}
    methods = ["central", "VDN", "QMIX", "OW-QMIX"]
    store = {}
    for gname, (R, names) in gms.items():
        print(f"\n=== {gname} ===  true payoffs (optimum at {names[0]}{names[0]} = {R[0, 0]:g}):\n"
              + np.array2string(R, precision=0))
        print("VDN least-squares closed form (Eq. 7.3): row means "
              f"{np.array2string(R.mean(1), precision=3)}, column means {np.array2string(R.mean(0), precision=3)}")
        Zb, mse_b, gb = best_monotonic_fit(R)
        store[(gname, "best monotonic")] = (Zb, None, None, mse_b, gb, None)
        print(f"  best monotonic least-squares fit (exact, 36 orderings): MSE {mse_b:.2f}, greedy joint "
              f"action {names[gb[0]]}{names[gb[1]]} (true payoff {R[gb]:+g})\n" + "\n".join(
                  "       " + " ".join(f"{v:7.2f}" for v in row) for row in Zb))
        for method in methods:
            results = [fit(method, R, args.seed + s, steps) for s in range(n_seeds)]
            greedy = [(int(np.argmax(q1)), int(np.argmax(q2))) for _, q1, q2, _ in results]
            p_opt = np.mean([g == (0, 0) for g in greedy])
            vals = [R[g] for g in greedy]
            mses = [r[3] for r in results]
            counts = {}
            for g in greedy:
                key = names[g[0]] + names[g[1]]
                counts[key] = counts.get(key, 0) + 1
            modal = max(counts, key=counts.get)
            k_show = next(k for k, g in enumerate(greedy) if names[g[0]] + names[g[1]] == modal)
            Q, q1, q2, mse = results[k_show]
            store[(gname, method)] = (Q, q1, q2, mse, greedy[k_show], k_show)
            print(f"  {method:8s} greedy decentralised joint action over {n_seeds} seeds: {counts}; "
                  f"P(optimal) = {p_opt:.2f}; mean true payoff of greedy action = {np.mean(vals):+.2f}; "
                  f"fit MSE = {np.mean(mses):.2f}")
            if method == "QMIX":
                # Is QMIX's fit the best fit for the ordering of actions it learned? (basin vs capacity)
                cone = [monotone_projection(R, tuple(np.argsort(r[1], kind="stable")),
                                            tuple(np.argsort(r[2], kind="stable")))[1] for r in results]
                print("     QMIX fit MSE per seed:        " + " ".join(f"{m:7.2f}" for m in mses))
                print("     optimum within its ordering:  " + " ".join(f"{m:7.2f}" for m in cone)
                      + f"   (global optimum {mse_b:.2f})")
            print(f"     learned Q_tot (seed {k_show}, a seed with the most common greedy action):\n" + "\n".join(
                "       " + " ".join(f"{v:7.2f}" for v in row) for row in Q))
    print(f"\ntotal time {time.time() - t0:.1f} s")

    if not args.quick:
        import plot_style
        plt = plot_style.setup()
        figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
        os.makedirs(figdir, exist_ok=True)
        cols = methods + ["best monotonic"]
        fig, axes = plt.subplots(2, len(cols), figsize=(17, 7))
        for gi, (gname, (R, names)) in enumerate(gms.items()):
            vmax = np.abs(R).max()
            for mi, method in enumerate(cols):
                ax = axes[gi, mi]
                Q, q1, q2, mse, g, k_show = store[(gname, method)]
                ax.imshow(Q, cmap="RdBu", vmin=-vmax, vmax=vmax)
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, f"{Q[i, j]:.1f}\n({R[i, j]:g})", ha="center", va="center", fontsize=8,
                                color="white" if abs(Q[i, j]) > 0.55 * vmax else "black",
                                fontweight="bold" if (i, j) == g else "normal")
                ax.add_patch(plt.Rectangle((g[1] - 0.5, g[0] - 0.5), 1, 1, fill=False, lw=2.5,
                                           edgecolor=plot_style.INK))
                ax.set_xticks(range(3)); ax.set_yticks(range(3))
                ax.set_xticklabels([f"{c}" for c in names]); ax.set_yticklabels([f"{c}" for c in names])
                ax.set_xlabel("agent 2 action"); ax.set_ylabel("agent 1 action")
                ax.grid(False)
                sub = "exact least squares" if k_show is None else f"seed {k_show}"
                ax.set_title(f"{gname}: {method}\n{sub}, MSE {mse:.1f}", fontsize=9.5)
        fig.suptitle("Fitted $Q_{tot}$ (true payoff in brackets); box = greedy decentralised joint action; "
                     "trained methods show a seed with the most common greedy action", y=1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "value_decomposition.png"))
        plt.close(fig)
        print(f"figure written to {figdir}/value_decomposition.png")


if __name__ == "__main__":
    main()
