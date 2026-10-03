"""Beyond linear: a small neural network and a memory-based (kernel) estimator on
the 1000-state random walk (Chapter 08, Sections 8 and 10).

Part A -- a 2-hidden-layer MLP v_hat(s, w) trained by minibatch SGD (Adam) on a fixed
          dataset of transitions collected from N episodes -- the setting of
          replay-based deep RL (Chapter 09) and offline RL (Chapter 16). Three losses:
            * gradient MC:             target G_t                      (SGD on VE)
            * semi-gradient TD(0):     target R + v_hat(S'), *detached* (stop-gradient)
            * naive residual gradient: same TD error but gradients flow through the
              target too, i.e. SGD on the mean squared TD error (TDE).
          Semi-gradient TD is what every deep value-based method uses; the "full
          gradient" alternative is stable, but converges to the wrong answer.
Part B -- memory-based / kernel regression (Nadaraya-Watson) on stored MC returns:
          v_hat(s) = sum_i k(s, s_i) G_i / sum_i k(s, s_i), Gaussian kernel.

Run:  python code/ch08_function_approximation/nonlinear_and_kernel.py [--quick]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread: the machine is shared
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn

import rw1000 as rw

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
torch.set_num_threads(1)


def make_net(hidden=32):
    return nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, 1))


def to_input(states):
    """Normalise states 1..1000 to [-1, 1] (Section 6: input scaling matters)."""
    return torch.as_tensor(2 * rw.unit(np.asarray(states)) - 1, dtype=torch.float32)[:, None]


ALL_IN = to_input(rw.ALL)


def sqrt_ve_net(net):
    with torch.no_grad():
        v_hat = net(ALL_IN).squeeze(1).numpy()
    e = rw.exact()
    return float(np.sqrt(e["mu"] @ (e["v"] - v_hat) ** 2))


def collect_dataset(n_episodes, seed):
    """Transitions (S_t, R_{t+1}, S_{t+1}, terminal flag, G_t) from n_episodes episodes."""
    rng = random.Random(seed)
    S, R, S2, term, G = [], [], [], [], []
    for _ in range(n_episodes):
        states, R_T = rw.generate_episode(rng)
        T = len(states)
        S += states
        S2 += states[1:] + [states[-1]]                  # dummy successor at the end (masked)
        R += [0.0] * (T - 1) + [R_T]
        term += [0.0] * (T - 1) + [1.0]
        G += [R_T] * T                                    # gamma = 1
    f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
    return to_input(S), f(R), to_input(S2), f(term), f(G)


def train_net(method, data, n_updates, seed, lr=1e-3, batch=256, eval_every=500):
    torch.manual_seed(seed)
    x, r, x2, term, G = data
    n = x.shape[0]
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # linear decay of the step size to 0: removes most of the constant-step-size noise,
    # so the end point reflects what each *objective* converges to (Robbins-Monro spirit)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 1.0 - k / n_updates)
    gen = torch.Generator().manual_seed(seed)
    curve = []
    for k in range(n_updates):
        i = torch.randint(n, (batch,), generator=gen)
        v = net(x[i]).squeeze(1)
        if method == "mc":
            loss = 0.5 * ((G[i] - v) ** 2).mean()
        else:
            v_next = net(x2[i]).squeeze(1) * (1 - term[i])  # v_hat(terminal) = 0
            if method == "td":
                target = (r[i] + v_next).detach()               # gamma = 1 here; semi-gradient: stop the gradient
                loss = 0.5 * ((target - v) ** 2).mean()
            else:                                               # naive residual gradient (TDE)
                loss = 0.5 * ((r[i] + v_next - v) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (k + 1) % eval_every == 0:
            curve.append(sqrt_ve_net(net))
    return net, np.array(curve)


def kernel_curve(n_episodes, bandwidth, seed, checkpoints):
    """Nadaraya-Watson regression of MC returns. States are discrete, so storing counts and
    return sums per state is equivalent to storing every (S_t, G_t) pair."""
    rng = random.Random(seed)
    s = rw.ALL.astype(float)
    K = np.exp(-0.5 * ((s[:, None] - s[None, :]) / bandwidth) ** 2)    # k(s, s')
    cnt = np.zeros(rw.N_STATES)
    sumG = np.zeros(rw.N_STATES)
    e = rw.exact()
    out = []
    for ep in range(1, n_episodes + 1):
        states, G = rw.generate_episode(rng)
        idx = np.asarray(states) - 1
        np.add.at(cnt, idx, 1.0)
        np.add.at(sumG, idx, G)
        if ep in checkpoints:
            v_hat = (K @ sumG) / np.maximum(K @ cnt, 1e-300)
            out.append(np.sqrt(e["mu"] @ (e["v"] - v_hat) ** 2))
    return np.array(out), int(cnt.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figures")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n_data, n_upd, seeds = (100, 600, [0]) if args.quick else (1000, 20_000, [0, 1, 2])
    eval_every = 200 if args.quick else 500
    print(f"seed base={args.seed} quick={args.quick}; MLP 1-32-32-1 tanh, Adam lr=1e-3 decayed linearly to 0, minibatch 256,"
          f" {n_upd} updates on a dataset of {n_data} episodes per seed, seeds={seeds}")

    curves = {}
    nets = {}
    names = {"mc": "gradient MC", "td": "semi-gradient TD(0)", "rg": "naive residual gradient (TDE)"}
    datasets = {sd: collect_dataset(n_data, args.seed + 100 + sd) for sd in seeds}
    print(f"  dataset sizes: {[d[0].shape[0] for d in datasets.values()]} transitions")
    for m in ("mc", "td", "rg"):
        t0 = time.time()
        cs = []
        for sd in seeds:
            net, c = train_net(m, datasets[sd], n_upd, args.seed + sd, eval_every=eval_every)
            cs.append(c)
            nets.setdefault(m, net)
        curves[m] = np.array(cs)
        fin = curves[m][:, -1]
        last = curves[m][:, -max(1, curves[m].shape[1] // 5):].mean()
        print(f"  {names[m]:32s}: sqrtVE after {n_upd} updates = {fin.mean():.4f}"
              f" (per seed {np.round(fin, 4).tolist()}); mean over last 20% of evaluations = {last:.4f}"
              f"  [{time.time() - t0:.1f}s]")

    bws = [10, 30, 100]
    checkpoints = [1, 3, 10, 30, 100, 300, 1000] + ([3000] if not args.quick else [])
    checkpoints = [c for c in checkpoints if c <= (100 if args.quick else 3000)]
    k_seeds = 5
    print(f"\n  kernel regression on MC returns, sqrtVE (mean of {k_seeds} seeds) at episodes {checkpoints}:")
    kern = {}
    for bw in bws:
        t0 = time.time()
        runs = [kernel_curve(max(checkpoints), bw, args.seed + 10 * j, set(checkpoints)) for j in range(k_seeds)]
        kern[bw] = np.mean([kc for kc, _ in runs], axis=0)
        print(f"    bandwidth {bw:4d} states: {np.round(kern[bw], 4).tolist()}  (memory after"
              f" {max(checkpoints)} episodes = {runs[0][1]:,} samples) [{time.time() - t0:.1f}s]")

    if args.quick:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    xs = np.arange(1, curves["mc"].shape[1] + 1) * eval_every
    for k, (m, c) in enumerate(curves.items()):
        for j, cj in enumerate(c):                                 # one thin line per seed
            axes[0].plot(xs, cj, color=f"C{k}", lw=1, alpha=0.8, label=names[m] if j == 0 else None)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("minibatch updates")
    axes[0].set_ylabel(r"$\sqrt{\overline{VE}}$ (log scale, one line per seed)")
    axes[0].set_title(f"MLP trained on a fixed dataset of {n_data} episodes")
    axes[0].legend(fontsize=8)
    e = rw.exact()
    axes[1].plot(rw.ALL, e["v"], "k", lw=2, label=r"$v_\pi$")
    for m, net in nets.items():
        with torch.no_grad():
            axes[1].plot(rw.ALL, net(ALL_IN).squeeze(1).numpy(), label=names[m])
    axes[1].set_xlabel("state")
    axes[1].set_ylabel("value")
    axes[1].set_title(f"Learned values after {n_upd} updates (first seed)")
    axes[1].legend(fontsize=8)
    for bw, kc in kern.items():
        axes[2].loglog(checkpoints, kc, "o-", label=f"kernel regression, bandwidth {bw}")
    axes[2].set_xlabel("episodes stored in memory")
    axes[2].set_ylabel(r"$\sqrt{\overline{VE}}$ (mean of 5 seeds)")
    axes[2].set_title("Memory-based (Nadaraya-Watson) estimate")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "nonlinear_and_kernel.png"), dpi=110)
    print(f"figure saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
