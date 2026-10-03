"""Numerical checks of the exact identities of Chapter 06.

Every identity is tested on random episodes of a random Markov reward process
(6 states, random rewards, gamma = 0.9, frequent revisits) and, where it applies,
with random *linear* features as well as one-hot (tabular) features.

  (1) n-step error = sum of TD errors          G_{t:t+n} - V(S_t) = sum_{k=t}^{t+n-1} gamma^{k-t} delta_k   (Sec. 2.1)
  (2) lambda-return error = sum of TD errors    G^lambda_t - V(S_t) = sum_k (gamma lambda)^{k-t} delta_k      (Sec. 8.2)
  (3) offline forward/backward equivalence: summed TD(lambda) increments with accumulating
      traces == summed lambda-return increments (Theorem in Sec. 10.1); replacing traces break it
  (4) the lambda = 1 case: offline TD(1) == every-visit constant-alpha Monte Carlo
  (5) true online TD(lambda) == online lambda-return algorithm after *every* step (Sec. 12)
  (6) true online TD(1) at episode end == forward-order constant-alpha MC
  (7) online TD(lambda) (accumulating) differs from the online lambda-return by O(alpha^2);
      the sequential offline lambda-return (S&B Eq. 12.4) differs from the summed one by O(alpha^2)
  (8) off-policy traces: the backward view z <- gamma c_t z + e(S_t, A_t) with expected-SARSA
      TD errors reproduces the forward recursion
      G_t = R_{t+1} + gamma [Vbar(S_{t+1}) + c_{t+1} (G_{t+1} - Q(S_{t+1}, A_{t+1}))]  (Sec. 6.2, 14.1)
      for c = lambda * min(1, rho) (Retrace), lambda*pi (Tree Backup), lambda*rho (IS)

Outputs (full mode): figures/online_equivalence.png

Run:  python code/ch06_n_step_and_eligibility_traces/equivalence_checks.py [--quick]
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one CPU thread (shared machine)
import numpy as np  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
N_S = 6          # non-terminal states 0..5; index 6 is the terminal state
TERM = N_S
GAMMA = 0.9
TOL = 1e-10


# ----------------------------------------------------------------------------- data
def random_mrp(rng):
    """Random transition matrix with termination probability 0.12 from every state."""
    P = rng.dirichlet(np.ones(N_S), size=N_S) * 0.88
    P = np.hstack([P, np.full((N_S, 1), 0.12)])
    return P


def episode(rng, P, max_len=60):
    s = int(rng.integers(N_S))
    states, rewards = [s], []
    while s != TERM and len(rewards) < max_len:
        s = int(rng.choice(N_S + 1, p=P[s]))
        states.append(s)
        rewards.append(float(rng.normal()))
    if s != TERM:                       # force termination (keeps the check simple)
        states[-1] = TERM
    return np.array(states), np.array(rewards)


def features(kind, rng, d=4):
    """Feature matrix X (N_S + 1, d) with x(terminal) = 0."""
    if kind == "tabular":
        X = np.vstack([np.eye(N_S), np.zeros(N_S)])
    else:
        X = rng.normal(size=(N_S, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)  # unit-norm features keep alpha in a stable range
        X = np.vstack([X, np.zeros(d)])
    return X


# ----------------------------------------------------------------------------- forward-view quantities
def td_errors(V, S, R, gamma=GAMMA):
    return R + gamma * V[S[1:]] - V[S[:-1]]                # V[TERM] must be 0


def lambda_returns(V, S, R, lam, gamma=GAMMA):
    """G^lambda_t for t = 0..T-1 with V fixed (recursive form, Eq. 6.18)."""
    T = len(R)
    G = np.zeros(T + 1)
    for t in range(T - 1, -1, -1):
        G[t] = R[t] + gamma * ((1 - lam) * V[S[t + 1]] + lam * G[t + 1])
    return G[:T]


def n_step_return(V, S, R, t, n, gamma=GAMMA):
    T = len(R)
    end = min(t + n, T)
    G = sum(gamma ** (i - t) * R[i] for i in range(t, end))
    if t + n < T:
        G += gamma ** n * V[S[t + n]]
    return G


# ----------------------------------------------------------------------------- algorithms (linear FA)
def offline_td_lambda_increment(w, X, S, R, lam, alpha, gamma=GAMMA, trace="accumulating"):
    """Sum of TD(lambda) increments over an episode with w held fixed (offline updating)."""
    z = np.zeros_like(w)
    dw = np.zeros_like(w)
    for t in range(len(R)):
        x, x1 = X[S[t]], X[S[t + 1]]
        delta = R[t] + gamma * w @ x1 - w @ x
        if trace == "accumulating":
            z = gamma * lam * z + x
        else:                                          # replacing (tabular only)
            z = gamma * lam * z
            z[x > 0] = 1.0
        dw += alpha * delta * z
    return dw


def offline_lambda_return_increment(w, X, S, R, lam, alpha, gamma=GAMMA):
    V = X @ w
    G = lambda_returns(V, S, R, lam, gamma)
    dw = np.zeros_like(w)
    for t in range(len(R)):
        dw += alpha * (G[t] - V[S[t]]) * X[S[t]]
    return dw


def offline_lambda_return_sequential(w, X, S, R, lam, alpha, gamma=GAMMA):
    """S&B (2018) Eq. 12.4: targets from the frozen w, updates applied one after another."""
    G = lambda_returns(X @ w, S, R, lam, gamma)
    w = w.copy()
    for t in range(len(R)):
        x = X[S[t]]
        w = w + alpha * (G[t] - w @ x) * x
    return w


def online_td_lambda(w, X, S, R, lam, alpha, gamma=GAMMA):
    """Online TD(lambda), accumulating traces; returns the weights after every step."""
    w = w.copy()
    z = np.zeros_like(w)
    ws = [w.copy()]
    for t in range(len(R)):
        x, x1 = X[S[t]], X[S[t + 1]]
        delta = R[t] + gamma * w @ x1 - w @ x
        z = gamma * lam * z + x
        w = w + alpha * delta * z
        ws.append(w.copy())
    return ws


def true_online_td_lambda(w, X, S, R, lam, alpha, gamma=GAMMA):
    """True online TD(lambda) with dutch traces (S&B 2018, Sec. 12.5 box)."""
    w = w.copy()
    z = np.zeros_like(w)
    v_old = 0.0
    ws = [w.copy()]
    for t in range(len(R)):
        x, x1 = X[S[t]], X[S[t + 1]]
        v, v1 = w @ x, w @ x1
        delta = R[t] + gamma * v1 - v
        z = gamma * lam * z + (1 - alpha * gamma * lam * (z @ x)) * x
        w = w + alpha * (delta + v - v_old) * z - alpha * (v - v_old) * x
        v_old = v1
        ws.append(w.copy())
    return ws


def online_lambda_return(w0, X, S, R, lam, alpha, gamma=GAMMA):
    """The online lambda-return algorithm (S&B 2018, Sec. 12.4): at every horizon h, redo
    all updates t = 0..h-1 from the episode's initial weights, using truncated
    lambda-returns G^lambda_{t:h}. The bootstrap value of state S_k is computed with
    w_{k-1}, the final weights of horizon k-1. O(T^2) per episode: a conceptual algorithm."""
    T = len(R)
    w_h = [w0.copy()]                                    # w_h[h] = w^h_h
    for h in range(1, T + 1):
        b = np.array([w_h[k - 1] @ X[S[k]] for k in range(1, h + 1)])   # b[k-1] = v(S_k, w_{k-1})
        G = np.empty(h)
        G[h - 1] = R[h - 1] + gamma * b[h - 1]
        for t in range(h - 2, -1, -1):
            G[t] = R[t] + gamma * ((1 - lam) * b[t] + lam * G[t + 1])
        w = w0.copy()
        for t in range(h):
            x = X[S[t]]
            w = w + alpha * (G[t] - w @ x) * x
        w_h.append(w)
    return w_h


def mc_forward_order(w, X, S, R, alpha, gamma=GAMMA):
    """Constant-alpha every-visit MC, updates applied in time order at the end of the episode."""
    G = lambda_returns(np.zeros(len(X)), S, R, 1.0, gamma)    # lambda = 1 -> Monte Carlo returns
    w = w.copy()
    for t in range(len(R)):
        x = X[S[t]]
        w = w + alpha * (G[t] - w @ x) * x
    return w


# ----------------------------------------------------------------------------- off-policy traces
def offpolicy_check(rng, lam=0.8, alpha=0.1, n_episodes=20):
    """Tabular action values, 2 actions. Target pi and behaviour b are random policies."""
    nA = 2
    pi = rng.dirichlet(np.ones(nA), size=N_S)
    b = rng.dirichlet(np.ones(nA), size=N_S)
    Pa = np.stack([random_mrp(rng) for _ in range(nA)])        # Pa[a, s, s']
    Q = np.vstack([rng.normal(size=(N_S, nA)), np.zeros((1, nA))])
    pi_full = np.vstack([pi, np.zeros((1, nA))])
    worst = {}
    for name, cfun in {"Retrace c = lam min(1, rho)": lambda p, q: lam * min(1.0, p / q),
                       "Tree Backup c = lam pi": lambda p, q: lam * p,
                       "per-decision IS c = lam rho": lambda p, q: lam * p / q}.items():
        err = 0.0
        for _ in range(n_episodes):
            s = int(rng.integers(N_S)); S, A, Rw = [s], [], []
            while s != TERM and len(Rw) < 60:
                a = int(rng.choice(nA, p=b[s])); A.append(a)
                s = int(rng.choice(N_S + 1, p=Pa[a, s])); S.append(s); Rw.append(float(rng.normal()))
            if s != TERM:
                S[-1] = TERM
            T = len(Rw)
            c = [cfun(pi[S[t], A[t]], b[S[t], A[t]]) for t in range(T)]
            Vbar = (pi_full * Q).sum(1)                         # Vbar(terminal) = 0
            dES = [Rw[t] + GAMMA * Vbar[S[t + 1]] - Q[S[t], A[t]] for t in range(T)]
            # backward view (offline: Q fixed, increments summed)
            z = np.zeros_like(Q); dQ_back = np.zeros_like(Q)
            for t in range(T):
                z *= GAMMA * c[t]
                z[S[t], A[t]] += 1.0
                dQ_back += alpha * dES[t] * z
            # forward view via the recursion
            G = np.zeros(T + 1)
            for t in range(T - 1, -1, -1):
                G[t] = Rw[t] + GAMMA * Vbar[S[t + 1]]
                if t + 1 < T:
                    G[t] += GAMMA * c[t + 1] * (G[t + 1] - Q[S[t + 1], A[t + 1]])
            dQ_fwd = np.zeros_like(Q)
            for t in range(T):
                dQ_fwd[S[t], A[t]] += alpha * (G[t] - Q[S[t], A[t]])
            err = max(err, np.abs(dQ_back - dQ_fwd).max())
        worst[name] = err
    return worst


# ----------------------------------------------------------------------------- main
def report(name, value, exact=True):
    status = ("OK" if value < TOL else "FAIL") if exact else ""
    print(f"  {name:72s} {value:10.2e}  {status}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    n_mdps = 3 if args.quick else 20
    n_eps = 5 if args.quick else 20
    print(f"Equivalence checks | seed={args.seed} random MRPs={n_mdps} episodes each={n_eps} "
          f"gamma={GAMMA} tolerance={TOL:g}")
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    worst = {k: 0.0 for k in ["n-step", "lambda", "offline", "replacing", "td1mc",
                              "trueonline_tab", "trueonline_lin", "td1_end"]}
    lens = []
    for _ in range(n_mdps):
        P = random_mrp(rng)
        lam = float(rng.uniform(0.2, 0.95))
        alpha = float(rng.uniform(0.01, 0.3))
        for kind in ("tabular", "linear"):
            X = features(kind, rng)
            for _ in range(n_eps):
                w = rng.normal(size=X.shape[1])              # fresh random weights for every episode
                S, R = episode(rng, P)
                lens.append(len(R))
                V = X @ w
                d = td_errors(V, S, R)
                T = len(R)
                # (1) and (2)
                for t in range(T):
                    for n in (1, 2, 3, 5):
                        rhs = sum(GAMMA ** (k - t) * d[k] for k in range(t, min(t + n, T)))
                        worst["n-step"] = max(worst["n-step"], abs(n_step_return(V, S, R, t, n) - V[S[t]] - rhs))
                G = lambda_returns(V, S, R, lam)
                for t in range(T):
                    rhs = sum((GAMMA * lam) ** (k - t) * d[k] for k in range(t, T))
                    worst["lambda"] = max(worst["lambda"], abs(G[t] - V[S[t]] - rhs))
                # (3) offline equivalence, (4) lambda = 1
                a = offline_td_lambda_increment(w, X, S, R, lam, alpha)
                b = offline_lambda_return_increment(w, X, S, R, lam, alpha)
                worst["offline"] = max(worst["offline"], np.abs(a - b).max())
                a1 = offline_td_lambda_increment(w, X, S, R, 1.0, alpha)
                b1 = offline_lambda_return_increment(w, X, S, R, 1.0, alpha)
                worst["td1mc"] = max(worst["td1mc"], np.abs(a1 - b1).max())
                if kind == "tabular":
                    ar = offline_td_lambda_increment(w, X, S, R, lam, alpha, trace="replacing")
                    worst["replacing"] = max(worst["replacing"], np.abs(ar - b).max())
                # (5) true online == online lambda-return at every step
                ws_true = true_online_td_lambda(w, X, S, R, lam, alpha)
                ws_olr = online_lambda_return(w, X, S, R, lam, alpha)
                key = "trueonline_tab" if kind == "tabular" else "trueonline_lin"
                worst[key] = max(worst[key], max(np.abs(p - q).max() for p, q in zip(ws_true, ws_olr)))
                # (6) true online TD(1) == forward-order MC at the end of the episode
                w_t1 = true_online_td_lambda(w, X, S, R, 1.0, alpha)[-1]
                worst["td1_end"] = max(worst["td1_end"], np.abs(w_t1 - mc_forward_order(w, X, S, R, alpha)).max())
    print(f"  (episodes: mean length {np.mean(lens):.1f}, max {max(lens)})\n")
    print("  identity" + " " * 64 + "max |diff|")
    report("(1) G_{t:t+n} - V(S_t) == sum of discounted TD errors", worst["n-step"])
    report("(2) G^lambda_t - V(S_t) == sum of (gamma lambda)-discounted TD errors", worst["lambda"])
    report("(3) offline TD(lambda), accumulating == offline lambda-return (summed)", worst["offline"])
    report("(4) offline TD(1) == offline every-visit constant-alpha MC", worst["td1mc"])
    report("(5a) true online TD(lambda) == online lambda-return, every step (tabular)", worst["trueonline_tab"])
    report("(5b) true online TD(lambda) == online lambda-return, every step (linear)", worst["trueonline_lin"])
    report("(6) true online TD(1) at episode end == forward-order constant-alpha MC", worst["td1_end"])
    report("    [contrast] offline TD(lambda) with REPLACING traces vs lambda-return", worst["replacing"],
           exact=False)
    print("    (replacing traces are not equivalent to the lambda-return when states repeat)")

    # (7) O(alpha^2) differences
    alphas = np.array([0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625])
    rng2 = np.random.default_rng(args.seed + 1)
    P = random_mrp(rng2)
    X = features("tabular", rng2)
    lam = 0.8
    eps = [episode(rng2, P) for _ in range(10)]
    w0 = rng2.normal(size=N_S)
    d_acc, d_true, d_seq = [], [], []
    for alpha in alphas:
        da = dt = ds = 0.0
        for S, R in eps:
            olr = online_lambda_return(w0, X, S, R, lam, alpha)[-1]
            da = max(da, np.abs(online_td_lambda(w0, X, S, R, lam, alpha)[-1] - olr).max())
            dt = max(dt, np.abs(true_online_td_lambda(w0, X, S, R, lam, alpha)[-1] - olr).max())
            seq = offline_lambda_return_sequential(w0, X, S, R, lam, alpha)
            summed = w0 + offline_lambda_return_increment(w0, X, S, R, lam, alpha)
            ds = max(ds, np.abs(seq - summed).max())
        d_acc.append(da); d_true.append(dt); d_seq.append(ds)
    print("\n(7) end-of-episode differences vs alpha (tabular, lambda = 0.8, 10 episodes):")
    print("   alpha    |TD(l) - online l-ret|  /alpha^2   |true online - online l-ret|   "
          "|sequential - summed offline l-ret|  /alpha^2")
    for a, x, y, z in zip(alphas, d_acc, d_true, d_seq):
        print(f"  {a:7.5f}   {x:12.3e}        {x / a**2:7.3f}      {y:12.3e}                 "
              f"{z:12.3e}             {z / a**2:7.3f}")

    # (8) off-policy traces
    print("\n(8) off-policy backward view == forward recursion (offline, summed increments):")
    for name, v in offpolicy_check(np.random.default_rng(args.seed + 2)).items():
        report(name, v)

    exact = [worst[k] for k in worst if k != "replacing"]
    assert max(exact) < TOL, "an exact identity failed"
    print(f"\nall exact identities hold to < {TOL:g}; elapsed {time.time() - t0:.1f} s")

    if args.quick:
        return
    from plot_style import setup, C, MARKERS
    plt = setup()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(alphas, d_acc, color=C[1], marker=MARKERS[0], label="online TD(λ), accumulating  vs  online λ-return")
    ax.loglog(alphas, d_seq, color=C[2], marker=MARKERS[1], label="sequential  vs  summed offline λ-return")
    ax.loglog(alphas, np.maximum(d_true, 1e-17), color=C[0], marker=MARKERS[2],
              label="true online TD(λ)  vs  online λ-return")
    ref = d_acc[2] * (alphas / alphas[2]) ** 2
    ax.loglog(alphas, ref, color="#898781", ls="--", lw=1.2, label=r"slope 2 ($\propto\alpha^2$)")
    ax.set_xlabel(r"step size $\alpha$")
    ax.set_ylabel("max |weight difference| after an episode")
    ax.set_title("Exact and approximate equivalences (tabular, λ = 0.8)")
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "online_equivalence.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
