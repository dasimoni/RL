"""PETS-lite on Pendulum-v1: probabilistic ensemble dynamics + CEM model-predictive control.

Chapter 13, Sections 3-5 (Algorithms 13.1-13.3).  The agent

  1. collects one episode with uniformly random torques,
  2. fits an ensemble of B = 5 Gaussian-output MLPs to the transitions (s, a, s' - s) by maximum
     likelihood, each member on a bootstrap resample (Section 3),
  3. acts by model-predictive control: at every step it optimises a 25-step torque sequence with
     the cross-entropy method against the learned model, using P = 5 "TS-infinity" particles per
     candidate, one per ensemble member (Section 5), executes the first torque and re-plans,
  4. adds the new episode to the data set, refits the ensemble, and repeats.

The reward function is assumed known (as in the PETS paper); only the dynamics are learned.
Pendulum never terminates; episodes are truncated at 200 steps.  MPC has no value function, so the
"bootstrap through truncation" rule is moot here: the planner simply looks H steps ahead.

The full run also plans with the TRUE dynamics on the same initial states ("oracle MPC"), so that
the gap between PETS and the oracle measures what the learned model costs, and tries two ablations
(seed 0, same budget): random shooting instead of CEM, and a single *deterministic* network instead
of the probabilistic ensemble ("D" in PETS).
If code/ch13_model_based_rl/results/sac_pendulum.json exists (written by sac_pendulum_baseline.py)
the learning-curve figure also shows the model-free SAC baseline.

Run (from the repository root):
  python code/ch13_model_based_rl/pets_pendulum.py            # full: 3 seeds x 10 episodes + oracle + ablations
  python code/ch13_model_based_rl/pets_pendulum.py --quick    # smoke test, no figures
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True          # importing the sibling modules must not leave __pycache__/
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass, replace  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mbrl_common import (GaussianEnsemble, MPCPlanner, TruePendulumDynamics,  # noqa: E402
                         pendulum_reward, train_ensemble)

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
RES_DIR = os.path.join(HERE, "results")


@dataclass
class PETSConfig:
    n_episodes: int = 10          # 1 random + 9 MPC episodes = 2,000 environment steps
    n_random_episodes: int = 1
    ensemble_size: int = 5        # B
    probabilistic: bool = True    # Gaussian outputs (P) vs deterministic MSE regression (D)
    hidden: int = 64
    n_hidden: int = 2
    lr: float = 1e-3
    first_train_steps: int = 2000  # gradient steps after the random episode
    train_steps: int = 600        # warm-started gradient steps after every later episode
    batch_size: int = 64
    bootstrap: bool = True
    method: str = "cem"           # "cem", "rs" (random shooting) or "random" (no planning: random torques)
    horizon: int = 25             # H
    pop: int = 100                # candidates per CEM iteration
    elites: int = 10
    iters: int = 4
    particles: int = 5            # P (multiple of ensemble_size): one TS-infinity particle per member
    episode_len: int = 200
    oracle: bool = False          # plan with the TRUE dynamics instead of a learned model


def run_pets(seed: int, cfg: PETSConfig, verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = gym.make("Pendulum-v1", max_episode_steps=cfg.episode_len)
    B = cfg.ensemble_size if cfg.probabilistic or cfg.ensemble_size > 1 else 1
    model = GaussianEnsemble(4, 3, n_members=B, hidden=cfg.hidden, n_hidden=cfg.n_hidden,
                             probabilistic=cfg.probabilistic)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    particles = cfg.particles if cfg.probabilistic else B   # a deterministic single model needs 1 particle
    if cfg.oracle:
        particles = 1
    planner = MPCPlanner(TruePendulumDynamics() if cfg.oracle else model, pendulum_reward,
                         horizon=cfg.horizon, pop=cfg.pop, elites=cfg.elites,
                         iters=cfg.iters, particles=particles, method=cfg.method,
                         sample_noise=cfg.probabilistic and not cfg.oracle)
    O, A, O2 = [], [], []
    returns, steps, fit_mse, times = [], [], [], []
    t0 = time.time()
    obs, _ = env.reset(seed=seed)
    for ep in range(cfg.n_episodes):
        if ep >= cfg.n_random_episodes and not cfg.oracle:   # (re)fit the model on all data so far
            n_steps = cfg.first_train_steps if ep == cfg.n_random_episodes else cfg.train_steps
            mse = train_ensemble(model, opt, np.array(O), np.array(A), np.array(O2), n_steps,
                                 cfg.batch_size, rng, bootstrap=cfg.bootstrap and B > 1)
            fit_mse.append(mse)
        if ep > 0:
            obs, _ = env.reset()
        planner.reset()
        ep_ret, done = 0.0, False
        while not done:
            if ep < cfg.n_random_episodes and not cfg.oracle:
                a = rng.uniform(-2.0, 2.0, size=1).astype(np.float32)
            else:
                a = planner.act(obs)
            obs2, r, terminated, truncated, _ = env.step(a)
            O.append(obs); A.append(a); O2.append(obs2)
            ep_ret += r
            obs = obs2
            done = terminated or truncated
        returns.append(ep_ret); steps.append(len(O)); times.append(time.time() - t0)
        if verbose:
            tag = "oracle" if cfg.oracle else ("random" if ep < cfg.n_random_episodes else cfg.method)
            extra = f"  model fit MSE {fit_mse[-1]:.4f}" if fit_mse and ep >= cfg.n_random_episodes else ""
            print(f"  seed {seed} ep {ep:2d} ({tag:6s}) steps {len(O):5d}  return {ep_ret:8.1f}{extra}"
                  f"  [{times[-1]:.0f}s]", flush=True)
    env.close()
    return {"seed": seed, "returns": returns, "steps": steps, "times": times}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 seed, 3 short episodes, no figures")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override PETSConfig fields, e.g. --set horizon=10 n_episodes=6 "
                         "(runs with overrides print results but save nothing)")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figure from results/pets_pendulum.json without re-running")
    args = ap.parse_args()

    cfg = PETSConfig()
    for kv in args.set:
        k, v = kv.split("=")
        typ = type(getattr(cfg, k))
        cfg = replace(cfg, **{k: (v.lower() == "true") if typ is bool else typ(v)})
    seeds = args.seeds
    if args.plot_only:
        with open(os.path.join(RES_DIR, "pets_pendulum.json")) as f:
            saved = json.load(f)
        plot(PETSConfig(**saved["config"]), saved["runs"], saved["oracle"], saved["ablations"])
        return
    if args.quick:
        cfg = replace(cfg, n_episodes=3, episode_len=40, first_train_steps=200, train_steps=50,
                      pop=40, iters=2, horizon=10)
        seeds = [0]
    print("PETS-lite on Pendulum-v1 | seeds", seeds)
    print("config:", asdict(cfg), flush=True)

    t_start = time.time()
    main_runs = [run_pets(s, cfg) for s in seeds]
    oracle_runs, ablations = [], {}
    if not args.quick and not args.no_oracle:
        print("Reference: the same CEM planner with the TRUE dynamics (oracle MPC), same initial states",
              flush=True)
        oracle_runs = [run_pets(s, replace(cfg, oracle=True)) for s in seeds]
    if not args.quick and not args.no_ablations:
        print("Ablation: random shooting (same number of model evaluations: pop*iters = "
              f"{cfg.pop * cfg.iters} uniform sequences, keep the best)", flush=True)
        ablations["random shooting"] = run_pets(seeds[0], replace(cfg, method="rs"))
        print("Ablation: single deterministic network (D), CEM", flush=True)
        ablations["single deterministic model"] = run_pets(seeds[0], replace(cfg, probabilistic=False,
                                                                             ensemble_size=1))
    total = time.time() - t_start

    R = np.array([r["returns"] for r in main_runs])            # (seeds, episodes)
    print("\nResults (return of each 200-step episode; episode 0 is random):")
    for r in main_runs:
        print(f"  seed {r['seed']}: " + " ".join(f"{x:7.0f}" for x in r["returns"]))
    print("  mean   : " + " ".join(f"{x:7.0f}" for x in R.mean(0)))
    if oracle_runs:
        RO = np.array([r["returns"] for r in oracle_runs])
        print("Oracle MPC (true dynamics) on the same initial states:")
        for r in oracle_runs:
            print(f"  seed {r['seed']}: " + " ".join(f"{x:7.0f}" for x in r["returns"]))
        last = R[:, -5:]
        print(f"Mean return over the last 5 episodes (steps {main_runs[0]['steps'][-6] + 1}-"
              f"{main_runs[0]['steps'][-1]}): PETS {last.mean():.1f} (per seed: "
              + ", ".join(f"{x:.1f}" for x in last.mean(1)) + f"); oracle MPC {RO[:, -5:].mean():.1f}")
        gap = R[:, 1:] - RO[:, 1:]                                # MPC episodes only
        print(f"Gap PETS - oracle over episodes 1-{cfg.n_episodes - 1}: mean {gap.mean():.1f}, "
              f"episode 1 (after 200 random transitions) {gap[:, 0].mean():.1f}, "
              f"episodes 2+ {gap[:, 1:].mean():.1f}; |gap| <= 10 in {np.mean(np.abs(gap) <= 10):.0%} of episodes")
        for name, r in ablations.items():
            g = np.array(r["returns"][1:]) - RO[0, 1:]
            print(f"  {name:28s} (seed {r['seed']}) returns: " + " ".join(f"{x:7.0f}" for x in r["returns"])
                  + f"   gap to oracle: mean {g.mean():.1f}, episodes 2+ {g[1:].mean():.1f}")
    print(f"Wall-clock: {total:.0f} s total; {np.mean([r['times'][-1] for r in main_runs]):.0f} s per main seed")

    if args.quick or args.set or not oracle_runs:
        return
    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "pets_pendulum.json"), "w") as f:
        json.dump({"config": asdict(cfg), "runs": main_runs, "oracle": oracle_runs, "ablations": ablations}, f)
    plot(cfg, main_runs, oracle_runs, ablations)


def plot(cfg, main_runs, oracle_runs, ablations):
    from plot_style import C, GREY, setup  # noqa: E402
    plt = setup()
    seeds = [r["seed"] for r in main_runs]
    R = np.array([r["returns"] for r in main_runs])
    RO = np.array([r["returns"] for r in oracle_runs])
    gap = R[:, 1:] - RO[:, 1:]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    ax = axes[0]
    steps = np.array(main_runs[0]["steps"])
    for r in main_runs:
        ax.plot(steps, r["returns"], color=C[0], alpha=0.3, lw=1, marker="o", ms=2.5)
    ax.plot(steps, R.mean(0), color=C[0], lw=2.4, marker="o", ms=4,
            label=f"PETS-lite (ensemble + CEM), mean of {len(seeds)} seeds")
    ax.plot(steps[1:], RO[:, 1:].mean(0), color=GREY, lw=1.5, marker="x", ms=5,
            label="oracle MPC (true dynamics), same initial states")
    sac_path = os.path.join(RES_DIR, "sac_pendulum.json")
    if os.path.exists(sac_path):
        with open(sac_path) as f:
            sac = json.load(f)
        S = np.array([r["eval_returns"] for r in sac["runs"]])
        xs = np.array(sac["runs"][0]["eval_steps"])
        ax.plot(xs, S.mean(0), color=C[3], lw=2.2, ls="-.", marker="D", ms=3,
                label=f"SAC (model-free), mean of {len(sac['runs'])} seeds")
        ax.fill_between(xs, S.min(0), S.max(0), color=C[3], alpha=0.15, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("environment steps (log scale)")
    ax.set_ylabel("episode return (200 steps)")
    ax.set_title("Sample efficiency on Pendulum-v1")
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[1]
    eps = np.arange(1, cfg.n_episodes)
    for r, ro in zip(main_runs, oracle_runs):
        ax.plot(eps, np.array(r["returns"][1:]) - np.array(ro["returns"][1:]), color=C[0], alpha=0.3, lw=1,
                marker="o", ms=2.5)
    ax.plot(eps, gap.mean(0), color=C[0], lw=2.4, marker="o", ms=4, label="PETS-lite (mean of seeds)")
    styles = {"random shooting": (C[1], "s", "--"), "single deterministic model": (C[2], "^", ":")}
    for name, r in ablations.items():
        col, mk, ls = styles[name]
        ax.plot(eps, np.array(r["returns"][1:]) - RO[0, 1:], color=col, marker=mk, ms=4, ls=ls, lw=1.6,
                label=f"random shooting instead of CEM (seed {r['seed']})" if name == "random shooting"
                else f"{name} (seed {r['seed']})")
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xlabel("episode (each = 200 steps; episode 0 was random)")
    ax.set_ylabel("return minus oracle-MPC return")
    ax.set_title("Gap to planning with the true model")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "pets_pendulum_learning_curve.png")
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    main()
