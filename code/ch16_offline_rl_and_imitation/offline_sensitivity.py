"""How sensitive are CQL and IQL to their one key hyperparameter?  (Chapter 16, Sections 8.6, 9.4 and 13)

Uses the medium and random CartPole datasets of offline_cartpole.py (same collection seeds)
and the same learners, and sweeps
  CQL:  alpha in {0, 0.1, 1, 10}                 (alpha = 0 is naive offline DQN)
  IQL:  (expectile tau, inverse temperature beta) in {(0.5, 1), (0.5, 10), (0.7, 1), (0.7, 3), (0.9, 1)}
with 3 seeds each, reporting the final greedy return and the mean of max_a Q(s, a) on data
states.  For IQL on the medium data we also report diagnostics that explain its failure:
  * how often the learned policy agrees with the medium controller on the dataset's states;
  * the mean estimated advantage Q(s, a) - V(s) of logged actions that agree / disagree with it;
  * per dataset state, the critic's preference for the controller's alternative action,
    Q(s, other) - Q(s, controller): its mean, its spread across states, and how often it is > 0.
Finally an expectile probe on the EXPERT data, which contain exactly one action per state:
IQL's values with tau = 0.5 and tau = 0.9 (beta does not affect Q or V), recorded every 2,000
steps.  If the upper expectile only "chased the rare action", it could not overestimate here.

The point (Section 13): in a real offline problem you could not run this sweep, because every
entry needs online evaluation.  Choosing hyperparameters offline is itself an off-policy
evaluation problem (Section 12).

Run:  python code/ch16_offline_rl_and_imitation/offline_sensitivity.py [--quick]
Output (full mode): figures/offline_sensitivity.png
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

import offline_cartpole as oc
from offline_lib import collect_dataset, medium_action, to_tensors

torch.set_num_threads(1)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def iql_diagnostics(agent, data):
    """Agreement with the medium controller and mean advantages of agreeing / other actions."""
    t = to_tensors(data)
    with torch.no_grad():
        s = agent.norm(t["s"]).unsqueeze(0).expand(agent.E, -1, -1)
        q = agent.q_targ(s)                                            # [E, N, 2]
        v = agent.v(s).squeeze(2)
        a = t["a"].unsqueeze(0).expand(agent.E, -1)
        adv = (oc.pick(q, a) - v).numpy()                              # [E, N]
        pi_a = agent.pi(s).argmax(2).numpy()                           # [E, N]
    med = medium_action(data["s"])
    is_med = data["a"] == med
    agree = (pi_a == med[None]).mean(1)
    # per-state preference for the alternative action: Q(s, 1 - med(s)) - Q(s, med(s))
    q = q.numpy()
    idx = np.arange(len(med))
    pref = q[:, idx, 1 - med] - q[:, idx, med]                         # [E, N]
    return dict(agree=agree.mean(), a_med=adv[:, is_med].mean(), a_oth=adv[:, ~is_med].mean(),
                pref_mean=pref.mean(1).mean(), pref_sd=pref.std(1).mean(),
                pref_pos=(pref > 0).mean(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    quick = args.quick
    n_data = 2_000 if quick else 20_000
    steps = 200 if quick else 8_000
    n_seeds = 2 if quick else 3
    eval_episodes = 3 if quick else 10
    alphas = [0.0, 1.0] if quick else [0.0, 0.1, 1.0, 10.0]
    iql_grid = [(0.7, 3.0)] if quick else [(0.5, 1.0), (0.5, 10.0), (0.7, 1.0), (0.7, 3.0),
                                           (0.9, 1.0)]
    probe_taus = [0.9] if quick else [0.5, 0.9]       # expectile probe on the expert data
    probe_every = steps // 4
    datasets = {"medium": 2000, "random": 3000}       # same collection seeds as offline_cartpole
    print(f"offline_sensitivity: quick={quick} seeds={n_seeds} dataset_size={n_data} "
          f"grad_steps={steps} polyak={oc.POLYAK} alphas={alphas} iql (tau, beta)={iql_grid}")
    t0 = time.time()
    data = {k: collect_dataset(k, n_data, seed=v) for k, v in datasets.items()}

    res_cql = {}
    print("\nCQL: final return mean (per seed) | mean max-Q on data")
    for dname in datasets:
        for al in alphas:
            oc.CQL_ALPHA = al
            _, rets, qs = oc.train("CQL", data[dname], n_seeds, steps, steps, eval_episodes, seed=7)
            res_cql[(dname, al)] = (rets[:, -1], qs[:, -1])
            print(f"  {dname:7s} alpha={al:5.1f}: {rets[:, -1].mean():6.1f} "
                  f"{np.round(rets[:, -1]).astype(int).tolist()} | {np.mean(qs[:, -1]):10.1f}")
    oc.CQL_ALPHA = 1.0

    res_iql = {}
    print("\nIQL: final return mean (per seed) | mean max-Q | [medium only: agreement with the "
          "medium controller; mean advantage of logged medium-controller / other actions]")
    for dname in datasets:
        for tau, beta in iql_grid:
            oc.IQL_EXPECTILE, oc.IQL_BETA = tau, beta
            _, rets, qs, agent = oc.train("IQL", data[dname], n_seeds, steps, steps,
                                          eval_episodes, seed=11, return_agent=True)
            res_iql[(dname, tau, beta)] = (rets[:, -1], qs[:, -1])
            extra = ""
            if dname == "medium":
                dg = iql_diagnostics(agent, data[dname])
                extra = (f" | agree {dg['agree']:.2f}; A(medium) {dg['a_med']:+.3f}, "
                         f"A(other) {dg['a_oth']:+.3f}; Q(s,other)-Q(s,ctrl): mean "
                         f"{dg['pref_mean']:+.3f}, sd over states {dg['pref_sd']:.2f}, "
                         f"> 0 in {dg['pref_pos']:.0%} of states")
            print(f"  {dname:7s} tau={tau:.1f} beta={beta:5.1f}: {rets[:, -1].mean():6.1f} "
                  f"{np.round(rets[:, -1]).astype(int).tolist()} | {np.mean(qs[:, -1]):7.1f}{extra}")

    # Expectile probe on the expert data (one logged action per state; true values <= 100).
    expert = collect_dataset("expert", n_data, seed=1000)  # same collection seed as offline_cartpole
    print("\nIQL expectile probe on the EXPERT data (one action per state): mean max-Q on data "
          f"states every {probe_every} steps | final return")
    for tau in probe_taus:
        oc.IQL_EXPECTILE, oc.IQL_BETA = tau, 1.0
        st, rets, qs = oc.train("IQL", expert, n_seeds, steps, probe_every, eval_episodes, seed=13)
        curve = ", ".join(f"{q:.1f}" for q in qs.mean(0))
        print(f"  expert  tau={tau:.1f}: max-Q at steps {st.tolist()}: {curve} | "
              f"return {rets[:, -1].mean():.1f}")
    oc.IQL_EXPECTILE, oc.IQL_BETA = 0.7, 3.0
    print(f"total time {time.time() - t0:.0f}s")

    if quick:
        return
    import plot_style
    plt = plot_style.setup()
    C = plot_style.C
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    ax = axes[0]
    xs = np.arange(len(alphas))
    for k, (dname, col) in enumerate([("medium", C[3]), ("random", C[0])]):
        m = [res_cql[(dname, a)][0].mean() for a in alphas]
        for j, a in enumerate(alphas):
            ax.scatter(np.full(n_seeds, j + (k - 0.5) * 0.18), res_cql[(dname, a)][0], color=col, s=14, alpha=0.6)
        ax.plot(xs + (k - 0.5) * 0.18, m, marker="D", color=col, label=f"{dname} data")
    ax.set_xticks(xs); ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_xlabel(r"CQL weight $\alpha$  ($\alpha = 0$: naive offline DQN)")
    ax.set_ylabel("final greedy return")
    ax.set_ylim(0, 520)
    ax.set_title("CQL: conservatism trades extrapolation for imitation")
    ax.legend(fontsize=8)
    ax = axes[1]
    xs = np.arange(len(iql_grid))
    for k, (dname, col) in enumerate([("medium", C[3]), ("random", C[0])]):
        m = [res_iql[(dname, t, b)][0].mean() for t, b in iql_grid]
        for j, (t, b) in enumerate(iql_grid):
            ax.scatter(np.full(n_seeds, j + (k - 0.5) * 0.18), res_iql[(dname, t, b)][0], color=col, s=14, alpha=0.6)
        ax.plot(xs + (k - 0.5) * 0.18, m, marker="v", color=col, label=f"{dname} data")
    ax.set_xticks(xs); ax.set_xticklabels([f"{t:g}, {b:g}" for t, b in iql_grid], rotation=20)
    ax.set_xlabel(r"IQL (expectile $\tau$, inverse temperature $\beta$)")
    ax.set_ylim(0, 520)
    ax.set_title(r"IQL: one $(\tau, \beta)$ can succeed on one dataset and collapse on another", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "offline_sensitivity.png"))
    plt.close(fig)
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
