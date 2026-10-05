"""Uncertainty and compounding errors in learned dynamics models.

Chapter 13, Sections 2.1 and 3.3-3.4.

Part A (toy regression, figure ``uncertainty_toy.png``)
    An ensemble of B = 5 Gaussian-output networks is fit to 1-D data with input-dependent noise and a
    gap in the inputs.  The law of total variance splits the predictive variance into
        aleatoric  = mean over members of sigma_b^2(x)        (noise in the data; irreducible)
        epistemic  = variance over members of mu_b(x)          (disagreement; shrinks with data)
    (Eq. 13.8).  We expect epistemic uncertainty to be largest in the gap and outside the data range,
    and aleatoric uncertainty to track the true noise level where there is data (Section 3.3 reports
    what actually happens).

Part B (Pendulum-v1, figure ``compounding_error.png``)
    Five probabilistic networks are trained on 1,000 transitions from a uniformly random policy.
    Each member is trained on ALL the data (no bootstrap) from a different random initialisation, so
    "a single model" below is exactly what you would get by training one network.  We then predict
    open loop, h = 1..50 steps ahead, feeding the model's own predictions back in and using the
    recorded actions, and measure || s_hat_{t+h} - s_{t+h} ||_2 in observation space for
      * each member alone (mean prediction), averaged over members  -> "single model",
      * the ensemble-mean model  s_hat <- s_hat + (1/B) sum_b mu_b(s_hat, a)  -> "ensemble mean",
      * and the ensemble's disagreement (std across the members' own rollouts).
    Test trajectories come from (i) the random policy that generated the data and (ii) a
    hand-written energy-pumping swing-up controller that visits fast, high-energy states the random
    policy rarely reaches (distribution shift, as when a planner drives the system somewhere new).

Run (from the repository root):
  python code/ch13_model_based_rl/compounding_error.py           # full (figures)
  python code/ch13_model_based_rl/compounding_error.py --quick   # smoke test, no figures
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True          # importing the sibling modules must not leave __pycache__/
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mbrl_common import GaussianEnsemble, energy_swingup_action, fit_ensemble, train_ensemble  # noqa: E402

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")


# ---------------------------------------------------------------------------------------------
# Part A: aleatoric vs epistemic uncertainty on a 1-D toy problem
# ---------------------------------------------------------------------------------------------
def noise_sd(x):
    return 0.05 + 0.25 / (1.0 + np.exp(-2.0 * x))          # noisier on the right


def part_a(seed: int, n_steps: int, n_points: int, make_fig: bool):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    # inputs in [-3, -1] U [0.5, 3]: a gap in the middle, nothing beyond |x| = 3
    x = np.concatenate([rng.uniform(-3, -1, n_points // 2), rng.uniform(0.5, 3, n_points - n_points // 2)])
    y = np.sin(x) + noise_sd(x) * rng.standard_normal(len(x))
    model = GaussianEnsemble(1, 1, n_members=5, hidden=64, n_hidden=2, probabilistic=True)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    fit_ensemble(model, opt, torch.tensor(x[:, None], dtype=torch.float32),
                 torch.tensor(y[:, None], dtype=torch.float32), n_steps, 64, rng, bootstrap=True)
    xs = np.linspace(-5, 5, 401)
    with torch.no_grad():
        xt = torch.tensor(xs[:, None], dtype=torch.float32)
        xn = ((xt - model.in_mu) / model.in_sd).unsqueeze(0).expand(model.B, -1, -1)
        mu, logvar = model(xn)
        mu = (model.out_mu + model.out_sd * mu)[..., 0].numpy()            # (B, n)
        var = (model.out_sd ** 2 * logvar.exp())[..., 0].numpy()          # (B, n)
    alea = np.sqrt(var.mean(0))          # sqrt of E_b[sigma_b^2]
    epi = np.sqrt(mu.var(0))             # sqrt of Var_b[mu_b]
    regions = {"in data (x in [-3,-1] or [0.5,3])": ((xs >= -3) & (xs <= -1)) | ((xs >= 0.5) & (xs <= 3)),
               "gap (x in (-1, 0.5))": (xs > -1) & (xs < 0.5),
               "outside (|x| > 3.5)": np.abs(xs) > 3.5}
    print("Part A: 1-D regression, ensemble of 5 Gaussian networks, mean std by region:")
    for name, m in regions.items():
        print(f"  {name:34s} aleatoric {alea[m].mean():.3f}   epistemic {epi[m].mean():.3f}   "
              f"true noise {noise_sd(xs[m]).mean():.3f}")
    if not make_fig:
        return
    from plot_style import C, GREY, INK, setup
    plt = setup()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    tot = np.sqrt(alea ** 2 + epi ** 2)
    m = mu.mean(0)
    ax.fill_between(xs, m - 2 * tot, m + 2 * tot, color=C[0], alpha=0.15, lw=0, label="mean $\\pm$ 2 total std")
    for b in range(mu.shape[0]):
        ax.plot(xs, mu[b], color=C[0], lw=1, alpha=0.8, label="member means $\\mu_b(x)$" if b == 0 else None)
    ax.plot(xs, np.sin(xs), color=INK, lw=1.2, ls="--", label="true $\\sin x$")
    ax.scatter(x, y, s=6, color=GREY, alpha=0.6, label="data", zorder=0)
    ax.set_ylim(-3, 3); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("Ensemble of probabilistic networks")
    ax.legend(loc="lower left", fontsize=8)
    ax = axes[1]
    ax.plot(xs, alea, color=C[1], label="aleatoric $\\sqrt{\\mathbb{E}_b[\\sigma_b^2]}$")
    ax.plot(xs, epi, color=C[2], ls="-.", label="epistemic $\\sqrt{\\mathrm{Var}_b[\\mu_b]}$")
    ax.plot(xs, noise_sd(xs), color=INK, lw=1.2, ls="--", label="true noise std")
    for lo, hi in [(-3, -1), (0.5, 3)]:
        ax.axvspan(lo, hi, color=GREY, alpha=0.12, lw=0)
    ax.set_yscale("log"); ax.set_xlabel("$x$ (shaded: where the data are)"); ax.set_ylabel("standard deviation")
    ax.set_title("Decomposing predictive uncertainty")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "uncertainty_toy.png")
    fig.savefig(out); plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------------------------------------
# Part B: compounding multi-step error on Pendulum
# ---------------------------------------------------------------------------------------------
def collect(env, n_eps, policy, rng, seed):
    trajs = []
    for i in range(n_eps):
        obs, _ = env.reset(seed=seed + i)
        O, A = [obs], []
        done = False
        while not done:
            a = policy(obs, rng)
            obs, _, te, tr, _ = env.step(a)
            O.append(obs); A.append(a)
            done = te or tr
        trajs.append((np.array(O), np.array(A)))
    return trajs


def random_policy(obs, rng):
    return rng.uniform(-2, 2, size=1).astype(np.float32)


@torch.no_grad()
def open_loop_errors(model, trajs, H, starts):
    """For every start index, roll each member and the ensemble-mean model H steps open loop.

    Returns arrays of shape (n_rollouts, H): single-member error (averaged over members),
    ensemble-mean error, ensemble disagreement, and, to show which coordinates dominate the
    Euclidean error, the angle error |wrap(th_hat - th)| and the angular-velocity error
    |thdot_hat - thdot| of a single member (averaged over members).
    """
    single, ens, spread, ang, vel = [], [], [], [], []
    for O, A in trajs:
        for t0 in starts:
            if t0 + H >= len(O):
                continue
            s_b = torch.as_tensor(O[t0], dtype=torch.float32).expand(model.B, 1, -1).clone()  # per member
            s_m = torch.as_tensor(O[t0], dtype=torch.float32).view(1, 1, -1).clone()          # ens. mean
            e_single, e_ens, e_spread, e_ang, e_vel = [], [], [], [], []
            for h in range(H):
                a = torch.as_tensor(A[t0 + h], dtype=torch.float32)
                s_b = model.step(s_b, a.view(1, 1, 1).expand(model.B, 1, 1), sample=False)
                nxt = model.step(s_m.expand(model.B, 1, -1), a.view(1, 1, 1).expand(model.B, 1, 1), sample=False)
                s_m = nxt.mean(0, keepdim=True)                  # average of the members' mean predictions
                truth = torch.as_tensor(O[t0 + h + 1], dtype=torch.float32)
                e_single.append(torch.linalg.norm(s_b[:, 0] - truth, dim=-1).mean().item())
                e_ens.append(torch.linalg.norm(s_m[0, 0] - truth).item())
                e_spread.append(torch.linalg.norm(s_b[:, 0].std(0, unbiased=False)).item())
                d_th = torch.atan2(s_b[:, 0, 1], s_b[:, 0, 0]) - torch.atan2(truth[1], truth[0])
                e_ang.append(torch.atan2(torch.sin(d_th), torch.cos(d_th)).abs().mean().item())   # wrapped
                e_vel.append((s_b[:, 0, 2] - truth[2]).abs().mean().item())
            single.append(e_single); ens.append(e_ens); spread.append(e_spread)
            ang.append(e_ang); vel.append(e_vel)
    return np.array(single), np.array(ens), np.array(spread), np.array(ang), np.array(vel)


def part_b(seed: int, n_train_eps: int, n_steps: int, H: int, n_test_eps: int, make_fig: bool):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = gym.make("Pendulum-v1")
    train = collect(env, n_train_eps, random_policy, rng, seed=1000 + seed)
    obs = np.concatenate([O[:-1] for O, _ in train]); act = np.concatenate([A for _, A in train])
    nxt = np.concatenate([O[1:] for O, _ in train])
    print(f"\nPart B: Pendulum-v1, {len(obs)} random-policy transitions; "
          f"|thdot| <= {np.abs(obs[:, 2]).max():.2f} in the training data")
    model = GaussianEnsemble(4, 3, n_members=5, hidden=64, n_hidden=2, probabilistic=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    t0 = time.time()
    mse = train_ensemble(model, opt, obs, act, nxt, n_steps, 64, rng, bootstrap=False)
    print(f"  trained 5 members x {n_steps} steps in {time.time() - t0:.1f}s; one-step fit MSE {mse:.5f} "
          "(normalised units)")
    test_rand = collect(env, n_test_eps, random_policy, rng, seed=5000 + seed)
    test_swing = collect(env, n_test_eps, lambda o, r: energy_swingup_action(o, r), rng, seed=6000 + seed)
    vmax = np.max([np.abs(O[:, 2]).max() for O, _ in test_swing])
    print(f"  swing-up test trajectories reach |thdot| up to {vmax:.2f}")
    # Where does the one-step error come from?  Pendulum clips |thdot| at 8: a non-smooth effect that
    # the random-policy data never show (no training transition hits the limit).
    print(f"  training transitions that hit the speed limit |thdot'| >= 7.99: "
          f"{np.mean(np.abs(nxt[:, 2]) >= 7.99):.3f}")
    for name, trajs in [("random-policy test", test_rand), ("swing-up test", test_swing)]:
        O_ = np.concatenate([O[:-1] for O, _ in trajs]); A_ = np.concatenate([A for _, A in trajs])
        N_ = np.concatenate([O[1:] for O, _ in trajs])
        with torch.no_grad():
            pred = model.step(torch.as_tensor(O_).expand(model.B, -1, -1).clone(),
                              torch.as_tensor(A_).expand(model.B, -1, -1), sample=False).mean(0).numpy()
        err = np.linalg.norm(pred - N_, axis=1)
        clip = np.abs(N_[:, 2]) >= 7.99
        print(f"  {name}: one-step error (ensemble mean, all {len(err)} transitions) {err.mean():.4f}; "
              f"{clip.mean():.1%} hit the speed limit, error there {err[clip].mean() if clip.any() else float('nan'):.4f}"
              f" vs {err[~clip].mean():.4f} elsewhere")
    starts = list(range(0, 150, 10))
    results = {}
    for name, trajs in [("random-policy test data", test_rand), ("swing-up test data (shifted)", test_swing)]:
        single, ens, spread, ang, vel = open_loop_errors(model, trajs, H, starts)
        results[name] = (single, ens, spread)
        print(f"  {name}: {len(single)} rollouts; mean error ||s_hat - s|| at horizon h")
        hs = [h for h in (1, 5, 10, 25, 50) if h <= H]
        print("     h:            " + "".join(f"{h:>9d}" for h in hs))
        print("     single model: " + "".join(f"{single[:, h - 1].mean():9.4f}" for h in hs))
        print("     ensemble mean:" + "".join(f"{ens[:, h - 1].mean():9.4f}" for h in hs))
        print("     disagreement: " + "".join(f"{spread[:, h - 1].mean():9.4f}" for h in hs))
        # The Euclidean error in (cos th, sin th, thdot) is dominated by thdot (range +-8, while cos and
        # sin are bounded by 1), so also report the two physical coordinates separately.
        print("     single model, angle error |wrap(th_hat - th)| (rad):  "
              + "".join(f"{ang[:, h - 1].mean():9.4f}" for h in hs))
        print("     single model, velocity error |thdot_hat - thdot|:     "
              + "".join(f"{vel[:, h - 1].mean():9.4f}" for h in hs))
    if not make_fig:
        return
    from plot_style import C, GREY, INK, setup
    plt = setup()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    hs = np.arange(1, H + 1)
    for ax, (name, (single, ens, spread)) in zip(axes[:2], results.items()):
        ax.plot(hs, single.mean(0), color=C[1], label="single model (avg. over 5)")
        ax.plot(hs, ens.mean(0), color=C[0], ls="--", label="ensemble-mean model")
        ax.plot(hs, spread.mean(0), color=C[2], ls=":", label="ensemble disagreement")
        ax.set_yscale("log"); ax.set_xlabel("prediction horizon $h$ (steps, open loop)")
        ax.set_ylabel(r"mean $\Vert\hat s_{t+h}-s_{t+h}\Vert_2$")
        ax.set_title(name)
        ax.legend(loc="lower right", fontsize=8)
    axes[1].set_ylim(axes[0].get_ylim()[0], None)
    # Panel 3: one rollout from the shifted data, angle of each member vs truth
    O, A = test_swing[0]
    t0 = 0
    ax = axes[2]
    with torch.no_grad():
        s_b = torch.as_tensor(O[t0], dtype=torch.float32).expand(model.B, 1, -1).clone()
        traj = [s_b[:, 0].numpy().copy()]
        for h in range(H):
            a = torch.as_tensor(A[t0 + h], dtype=torch.float32).view(1, 1, 1).expand(model.B, 1, 1)
            s_b = model.step(s_b, a, sample=False)
            traj.append(s_b[:, 0].numpy().copy())
    traj = np.array(traj)                                  # (H+1, B, 3)
    th_true = np.unwrap(np.arctan2(O[t0:t0 + H + 1, 1], O[t0:t0 + H + 1, 0]))
    for b in range(model.B):
        th_b = np.unwrap(np.arctan2(traj[:, b, 1], traj[:, b, 0]))
        ax.plot(np.arange(H + 1), th_b, color=C[0], lw=1, alpha=0.8, label="each member" if b == 0 else None)
    ax.plot(np.arange(H + 1), th_true, color=INK, lw=2, label="true trajectory")
    ax.axhline(0, color=GREY, lw=0.8, ls=":")
    ax.set_xlabel("step"); ax.set_ylabel(r"angle $\theta$ (rad, unwrapped; 0 = upright)")
    ax.set_title("Open-loop rollouts, swing-up actions")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "compounding_error.png")
    fig.savefig(out); plt.close(fig)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    if args.quick:
        cfg_a = dict(n_steps=300, n_points=200)
        cfg_b = dict(n_train_eps=2, n_steps=300, H=10, n_test_eps=2)
    else:
        cfg_a = dict(n_steps=4000, n_points=400)
        cfg_b = dict(n_train_eps=5, n_steps=4000, H=50, n_test_eps=10)
    print(f"seed {args.seed} | part A {cfg_a} | part B {cfg_b}")
    os.makedirs(FIG_DIR, exist_ok=True)
    part_a(args.seed, make_fig=not args.quick, **cfg_a)
    part_b(args.seed, make_fig=not args.quick, **cfg_b)
    print(f"Total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
