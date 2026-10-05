"""Stationary distribution of a Markov chain: power iteration vs eigenvector vs linear solve.

Chapter 00, Section 1.6. Demonstrates
  * three ways to compute the stationary distribution  d = d P  (Eq. 1.10),
  * that power iteration converges geometrically at rate |lambda_2| (second-largest
    eigenvalue modulus), so weakly-connected ("slow-mixing") chains converge slowly,
    and that stopping when the STEP is small (Algorithm 1.1) can leave a much larger error,
  * that a periodic chain has a stationary distribution but power iteration does not
    converge to it (a "lazy" chain or a Cesaro average fixes this),
  * the ergodic theorem: time-averaged visit frequencies along ONE long trajectory
    converge to the stationary distribution, with error ~ 1/sqrt(N).

Run:  python code/ch00_math_toolkit/markov_chain_stationary.py [--quick]
"""
import argparse
import os
import time

import numpy as np

from plot_style import setup, C, GREY

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ---------------------------------------------------------------------------
# Three solvers for the stationary distribution
# ---------------------------------------------------------------------------
def power_iteration(P, d0, num_iters, d_star=None):
    """Iterate the row vector d_{k+1} = d_k P (Eq. 1.9). Returns final d and
    (optionally) the L1 error ||d_k - d_star||_1 at every iteration."""
    d = d0.copy()
    errors = []
    for _ in range(num_iters):
        if d_star is not None:
            errors.append(np.abs(d - d_star).sum())
        d = d @ P                       # Eq. (1.9) on the whole distribution
    return d, np.array(errors)


def power_iteration_until(P, d0, tol, max_iters=100_000):
    """Algorithm 1.1: iterate d <- d P until the L1 step ||d_new - d||_1 < tol.
    Returns (d_new, number of iterations). NOTE: a small step is not a small error; near
    convergence the error of d_new is about |lambda_2|/(1-|lambda_2|) times the step (cf. Eq. 4.7)."""
    d = d0.copy()
    for k in range(1, max_iters + 1):
        d_new = d @ P
        if np.abs(d_new - d).sum() < tol:
            return d_new, k
        d = d_new
    return d, max_iters


def eigenvector_method(P):
    """d P = d  <=>  P^T d^T = d^T : d is the eigenvector of P^T with eigenvalue 1."""
    eigvals, eigvecs = np.linalg.eig(P.T)
    i = np.argmin(np.abs(eigvals - 1.0))
    v = np.real(eigvecs[:, i])
    return v / v.sum()  # eigenvectors are defined up to scale (and sign): normalise


def linear_solve_method(P):
    """Solve (P^T - I) d^T = 0 together with sum(d) = 1.
    The n equations of (P^T - I) d = 0 are linearly dependent (they sum to 0), so we
    replace the last one by the normalisation constraint and call a linear solver."""
    n = P.shape[0]
    A = P.T - np.eye(n); A[-1, :] = 1.0     # replace one equation by sum(d) = 1
    b = np.zeros(n); b[-1] = 1.0
    return np.linalg.solve(A, b)


def second_eigenvalue_modulus(P):
    mags = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    return mags[1]


# ---------------------------------------------------------------------------
# Example chains
# ---------------------------------------------------------------------------
def weather_chain():
    """The 3-state chain worked by hand in Section 1.6 (Sunny, Cloudy, Rainy)."""
    return np.array([[0.6, 0.3, 0.1],
                     [0.3, 0.4, 0.3],
                     [0.2, 0.3, 0.5]])


def two_rooms_chain(n_per_room=10, door_prob=0.01, rng=None):
    """Two densely-connected clusters of states joined by a rarely-used 'door'.
    Mixing between rooms is slow, so |lambda_2| is close to 1."""
    n = 2 * n_per_room
    P = np.zeros((n, n))
    for room in range(2):
        idx = np.arange(room * n_per_room, (room + 1) * n_per_room)
        for s in idx:
            w = rng.random(n_per_room) + 0.1
            P[s, idx] = w / w.sum()
    # each state leaks door_prob of its mass to a random state in the other room
    for s in range(n):
        other = rng.integers(0, n_per_room) + (n_per_room if s < n_per_room else 0)
        P[s] *= (1 - door_prob)
        P[s, other] += door_prob
    return P


def simulate_visit_frequencies(P, s0, num_steps, checkpoints, rng):
    """Run ONE trajectory of the chain and record empirical visit frequencies."""
    n = P.shape[0]
    cdf = np.cumsum(P, axis=1)
    counts = np.zeros(n)
    s = s0
    out = []
    u = rng.random(num_steps)
    cps = set(checkpoints)
    for t in range(1, num_steps + 1):
        counts[s] += 1
        if t in cps:
            out.append(counts / t)
        s = int(np.searchsorted(cdf[s], u[t - 1]))  # sample next state from row s
        s = min(s, n - 1)
    return np.array(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    print(f"[markov_chain_stationary] seed={args.seed} quick={args.quick}")
    t0 = time.time()

    # ---------------- (a) weather chain: three methods agree ------------------
    P = weather_chain()
    d_eig = eigenvector_method(P)
    d_lin = linear_solve_method(P)
    d_pow, _ = power_iteration(P, np.array([1.0, 0.0, 0.0]), 50)
    exact = np.array([7, 6, 5]) / 18  # hand computation in Section 1.6
    print("\nWeather chain P =\n", P)
    print("eigenvalues of P:", np.round(np.sort(np.real(np.linalg.eigvals(P)))[::-1], 6))
    print(f"  by hand        : {np.round(exact, 6)}")
    print(f"  power iter (50): {np.round(d_pow, 6)}")
    print(f"  eigenvector    : {np.round(d_eig, 6)}")
    print(f"  linear solve   : {np.round(d_lin, 6)}")
    max_diff = max(np.abs(d_eig - d_lin).max(), np.abs(d_pow - d_lin).max())
    print(f"  max |difference| between methods: {max_diff:.2e}")

    lam2 = second_eigenvalue_modulus(P)
    K = 30
    starts = {"start in Sunny": np.array([1.0, 0, 0]),
              "start in Rainy": np.array([0, 0, 1.0]),
              "uniform start": np.ones(3) / 3}
    weather_err = {k: power_iteration(P, v, K, exact)[1] for k, v in starts.items()}
    # empirical convergence rate: ratio of successive errors
    e = weather_err["start in Sunny"]
    print(f"  |lambda_2| = {lam2:.4f};  observed error ratio e_k+1/e_k at k=10: {e[11]/e[10]:.4f}")

    # ---------------- (b) slow-mixing "two rooms" chain ---------------------
    P2 = two_rooms_chain(n_per_room=10, door_prob=0.01, rng=rng)
    d2 = linear_solve_method(P2)
    lam2_rooms = second_eigenvalue_modulus(P2)
    K2 = 300 if args.quick else 1500
    d0 = np.zeros(P2.shape[0]); d0[0] = 1.0
    d2_pow, rooms_err = power_iteration(P2, d0, K2, d2)
    iters_to_1e6 = int(np.argmax(rooms_err < 1e-6)) if (rooms_err < 1e-6).any() else None
    iters_to_1e6_w = int(np.argmax(weather_err["start in Sunny"] < 1e-6))
    print(f"\nTwo-rooms chain (20 states, door prob 0.01): |lambda_2| = {lam2_rooms:.4f}")
    print(f"  eigvec vs linear solve max diff: {np.abs(eigenvector_method(P2) - d2).max():.2e}")
    print(f"  power-iteration L1 error after {K2} iters: {rooms_err[-1]:.2e}")
    print(f"  iterations to reach L1 error < 1e-6: weather={iters_to_1e6_w}, two-rooms={iters_to_1e6}")
    # Algorithm 1.1's stopping rule watches the STEP, not the (unknown) error.
    tol = 1e-6
    d_stop, k_stop = power_iteration_until(P2, d0, tol)
    tol_safe = tol * (1 - lam2_rooms) / lam2_rooms       # step tolerance that guarantees ~tol error
    d_safe, k_safe = power_iteration_until(P2, d0, tol_safe)
    print(f"  Algorithm 1.1 with step tolerance {tol:.0e}: stops at k={k_stop} with TRUE L1 error "
          f"{np.abs(d_stop - d2).sum():.1e} ({np.abs(d_stop - d2).sum() / tol:.0f}x the tolerance)")
    print(f"  with step tolerance tol*(1-|l2|)/|l2| = {tol_safe:.1e}: stops at k={k_safe}, true L1 error "
          f"{np.abs(d_safe - d2).sum():.1e}")

    # ---------------- (c) periodic chain -------------------------------------
    Pper = np.array([[0.0, 1.0], [1.0, 0.0]])
    Plazy = 0.1 * np.eye(2) + 0.9 * Pper  # small self-loop prob. breaks periodicity
    Kp = 20
    d = np.array([1.0, 0.0]); per_traj = []
    dl = np.array([1.0, 0.0]); lazy_traj = []
    for _ in range(Kp):
        per_traj.append(d[0]); lazy_traj.append(dl[0])
        d = d @ Pper; dl = dl @ Plazy
    per_traj = np.array(per_traj); lazy_traj = np.array(lazy_traj)
    cesaro = np.cumsum(per_traj) / np.arange(1, Kp + 1)
    print("\nPeriodic chain [[0,1],[1,0]]: stationary =", linear_solve_method(Pper))
    print("  Pr{X_k = 0} under power iteration, k=0..7:", per_traj[:8])
    print("  Cesaro average after 20 steps:", round(cesaro[-1], 4),
          "| lazy chain (eigenvalues 1, -0.8) after 20 steps:", round(lazy_traj[-1], 4))

    # ---------------- (d) ergodic theorem: time averages ----------------------
    N = 20_000 if args.quick else 1_000_000
    checkpoints = np.unique(np.logspace(1, np.log10(N), 40).astype(int))
    freqs = simulate_visit_frequencies(P, 0, N, checkpoints, rng)
    erg_err = np.abs(freqs - exact).sum(axis=1)
    print(f"\nErgodic theorem on the weather chain, one trajectory of N={N:,} steps:")
    print(f"  empirical visit frequencies: {np.round(freqs[-1], 4)}  (stationary {np.round(exact, 4)})")
    print(f"  L1 error at N=100: {erg_err[np.searchsorted(checkpoints, 100)]:.4f}, at N={N:,}: {erg_err[-1]:.5f}")

    if not args.quick:
        plt = setup()
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
        ax = axes[0, 0]
        styles = ["-", "--", ":"]
        for (name, err), ls, c in zip(weather_err.items(), styles, C):
            ax.semilogy(err, ls, color=c, label=name)
        ks = np.arange(K)
        ax.semilogy(ks, 2 * lam2 ** ks, color=GREY, lw=1.2, label=r"reference $2\cdot 0.4^k$")
        ax.set_title(f"(a) Weather chain: power iteration, |λ₂| = {lam2:.2f}")
        ax.set_xlabel("iteration k"); ax.set_ylabel(r"$\|\mathbf{d}_k - \mathbf{d}\|_1$")
        ax.set_ylim(1e-14, 3); ax.legend()

        ax = axes[0, 1]
        ks2 = np.arange(K2)
        ax.semilogy(rooms_err, color=C[0], label="power iteration")
        ax.semilogy(ks2, 10 * rooms_err[50] * lam2_rooms ** (ks2 - 50), color=GREY, lw=1.2,
                    ls="--", label=rf"$\propto|\lambda_2|^k$ (shifted up), $|\lambda_2|$={lam2_rooms:.3f}")
        ax.set_title("(b) Slow-mixing 'two rooms' chain (20 states)")
        ax.set_xlabel("iteration k"); ax.set_ylabel(r"$\|\mathbf{d}_k - \mathbf{d}\|_1$"); ax.legend()

        ax = axes[1, 0]
        kk = np.arange(Kp)
        ax.plot(kk, per_traj, "o-", color=C[1], ms=5, lw=1.2, label="periodic chain: Pr{state 0}")
        ax.plot(kk, cesaro, "s--", color=C[0], ms=4, lw=1.2, label="its running (Cesàro) average")
        ax.plot(kk, lazy_traj, "^:", color=C[2], ms=5, lw=1.2, label="lazy chain 0.1·I + 0.9·P")
        ax.axhline(0.5, color=GREY, lw=1, label="stationary value 0.5")
        ax.set_title("(c) Periodicity breaks power iteration")
        ax.set_ylim(-0.05, 1.5)
        ax.set_xlabel("iteration k"); ax.set_ylabel("probability of state 0")
        ax.legend(loc="upper center", ncol=2)

        ax = axes[1, 1]
        ax.loglog(checkpoints, erg_err, "o-", color=C[0], ms=3, lw=1.2, label="one trajectory")
        ax.loglog(checkpoints, 1.0 / np.sqrt(checkpoints), color=GREY, lw=1.2, ls="--",
                  label=r"reference $1/\sqrt{N}$")
        ax.set_title("(d) Ergodic theorem: visit frequencies → d")
        ax.set_xlabel("trajectory length N"); ax.set_ylabel("L1 error of visit frequencies"); ax.legend()
        fig.tight_layout()
        path = os.path.join(FIG_DIR, "markov_stationary.png")
        fig.savefig(path)
        print(f"\nsaved {path}")

    print(f"\nSummary: three methods agree to {max_diff:.1e}; power iteration "
          f"contracts at rate |lambda_2| ({lam2:.2f} weather, {lam2_rooms:.3f} two-rooms); "
          f"runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
