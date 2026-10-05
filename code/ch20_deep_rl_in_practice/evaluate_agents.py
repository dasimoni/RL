"""Statistically sound evaluation of RL agents on a multi-task suite (Chapter 20, Section 6).

Four tabular TD agents (Q-learning, Double Q-learning, Expected SARSA, SARSA;
Chapter 05) are trained on five toy-text tasks with 100 independent runs each.
We then report results the way Agarwal et al. (NeurIPS 2021) recommend, using
the from-scratch tools in rl_stats.py:

  Part A  aggregate metrics (median of task means, IQM, mean, optimality gap)
          with 95% stratified bootstrap CIs, from a realistic budget of 10 runs/task;
  Part B  performance profiles with bootstrap bands, and probability of
          improvement P(X > Y) for pairs of agents;
  Part C  the "statistical precipice": how much point estimates computed from
          3, 5, 10 or 20 runs vary, how often the percentile bootstrap CI
          covers the 100-run value, and how often 3 runs rank two agents in
          the wrong order;
  Part D  IQM learning curves with CI bands.

Scores are exact (no evaluation noise) and normalized with lo = the best
TRIVIAL policy (random or constant action) and hi = the optimal policy: see
tabular_suite.py. The script also prints where the trivial policy would land
if the random policy were used as the lower reference instead.

Run:  python code/ch20_deep_rl_in_practice/evaluate_agents.py [--quick]
"""
import argparse
import time

import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

import rl_stats as st
import tabular_suite as ts

SEED = 0
N_POOL = 100          # runs per (agent, task): the "population" we subsample from
N_REPORT = 10         # runs a careful practitioner might report
BOOT_REPS = 2000
ALPHA, EPS, GAMMA = 0.1, 0.1, 0.99

# Task suite: (gymnasium id, kwargs, time limit H, training steps per run)
ts.TASKS.clear()
ts.TASKS.update({
    "FrozenLake4x4": ("FrozenLake-v1", {"map_name": "4x4", "is_slippery": True}, 100, 40_000),
    "FrozenLake8x8": ("FrozenLake-v1", {"map_name": "8x8", "is_slippery": True}, 200, 100_000),
    "FrozenLake6x6r": ("FrozenLake-v1", {"desc": generate_random_map(size=6, p=0.75, seed=3),
                                         "is_slippery": True}, 100, 40_000),
    "Taxi": ("Taxi-v4", {}, 200, 110_000),
    "CliffSlippery": ("CliffWalkingSlippery-v1", {}, 100, 40_000),
})


def run_suite(n_runs, step_scale, agents):
    tasks = [ts.TabularTask(name) for name in ts.TASKS]
    print("Tasks (exact H-step returns of the reference policies; lo = best trivial policy, hi = optimal):")
    for t in tasks:
        print(f"  {t.name:<15s} |S|={t.S:<4d} |A|={t.A}  H={t.H:<4d} train steps={int(t.train_steps * step_scale):>7,d}"
              f"  random={t.random_score:9.3f}  best constant action={t.constant_scores.max():9.3f}"
              f"  optimal={t.optimal_score:8.3f}")
    print("  with lo = the RANDOM policy instead, the best trivial policy would already score: "
          + ", ".join(f"{t.name} {t.normalized(t.trivial_score, lo=t.random_score):.3f}" for t in tasks))
    final, curves = {}, {}
    for agent in agents:
        f_cols, c_cols = [], []
        for j, t in enumerate(tasks):
            t0 = time.time()
            # distinct, reproducible seed per (agent, task)
            fin, cps, curve = ts.train(t, agent, n_runs, seed=SEED + 1000 * agents.index(agent) + j,
                                       steps=int(t.train_steps * step_scale), alpha=ALPHA, eps=EPS, gamma=GAMMA)
            f_cols.append(fin)
            c_cols.append(curve)
            print(f"  trained {agent:<15s} on {t.name:<15s} {n_runs} runs in {time.time() - t0:5.1f} s;"
                  f" mean normalized score {fin.mean():.3f}")
        final[agent] = np.stack(f_cols, axis=1)            # (runs, tasks)
        curves[agent] = np.stack(c_cols, axis=2)           # (checkpoints, runs, tasks)
    return tasks, final, curves


def part_a(final, agents, rng):
    print(f"\nPart A: aggregate metrics, first {N_REPORT} runs per task, 95% stratified bootstrap CIs "
          f"({BOOT_REPS} reps); in brackets the {N_POOL}-run point estimate")
    res = {}
    for agent in agents:
        sub = final[agent][:N_REPORT]
        res[agent] = {}
        line = f"  {agent:<15s}"
        for name, fn in st.AGGREGATES.items():
            p, lo, hi = st.stratified_bootstrap_ci(sub, fn, reps=BOOT_REPS, rng=rng)
            res[agent][name] = (p, lo, hi, st.aggregate(final[agent], fn))
            line += f" {name} {p:.3f} [{lo:.3f}, {hi:.3f}] ({res[agent][name][3]:.3f}) |"
        print(line)
    return res


def part_b(final, agents, rng):
    taus = np.round(np.linspace(-1.0, 1.05, 206), 10)   # scores below 0 are worse than the trivial policy
    prof = {a: st.performance_profile_ci(final[a][:N_REPORT], taus, reps=BOOT_REPS, rng=rng) for a in agents}
    print(f"\nPart B: performance profile at tau = 0 / 0.5 / 0.9 (fraction of runs above), {N_REPORT} runs:")
    for a in agents:
        cells = []
        for tau in (0.0, 0.5, 0.9):
            i = int(np.argmin(abs(taus - tau)))
            cells.append(f"F({tau}) = {prof[a][0][i]:.3f} [{prof[a][1][i]:.3f}, {prof[a][2][i]:.3f}]")
        print(f"  {a:<15s} " + "   ".join(cells))
    pairs = [(x, y) for i, x in enumerate(agents) for y in agents[i + 1:]]
    poi = {}
    print(f"  probability of improvement P(X > Y), {N_REPORT} runs (and {N_POOL} runs):")
    for x, y in pairs:
        p, lo, hi = st.probability_of_improvement_ci(final[x][:N_REPORT], final[y][:N_REPORT], reps=BOOT_REPS, rng=rng)
        pool = float(st.probability_of_improvement(final[x], final[y]))
        poi[(x, y)] = (p, lo, hi, pool)
        verdict = "X better" if lo > 0.5 else ("Y better" if hi < 0.5 else "inconclusive")
        print(f"    P({x} > {y}) = {p:.3f} [{lo:.3f}, {hi:.3f}]  ({pool:.3f})  -> {verdict}")
    return taus, prof, poi


PREC_METRICS = {"Median": st.agg_median, "Pooled median": st.agg_pooled_median, "IQM": st.agg_iqm, "Mean": st.agg_mean}


def part_c(final, agents, rng, ns, n_sub, n_cov, cov_reps, task_names):
    print("\nPart C: the statistical precipice (subsets drawn without replacement from the run pool)")
    focus = agents[0]
    pool = final[focus]
    n_pool, m = pool.shape
    # what the pool looks like: per-task spread, and the middle half that the IQM averages
    print(f"  {focus}, {n_pool} runs: per-task mean and 5-95% range of the normalized scores: "
          + "; ".join(f"{task_names[j]} {pool[:, j].mean():.3f} [{np.quantile(pool[:, j], .05):.3f}, "
                      f"{np.quantile(pool[:, j], .95):.3f}]" for j in range(m)))
    flat = np.sort(pool.ravel())
    cut = int(0.25 * flat.size)
    kept = flat[cut:flat.size - cut]
    print(f"  pooled scores: the IQM averages the middle half, which spans [{kept[0]:.3f}, {kept[-1]:.3f}];"
          f" pooled median {np.median(flat):.3f}; median of task means {st.aggregate(pool, st.agg_median):.3f}")
    spread = {}
    for name, fn in PREC_METRICS.items():
        truth = st.aggregate(pool, fn)
        spread[name] = {}
        for n in ns:
            # one random n-subset of runs per task, independently across tasks
            idx = np.argsort(rng.random((n_sub, n_pool, m)), axis=1)[:, :n, :]
            subs = np.take_along_axis(np.broadcast_to(pool, (n_sub, n_pool, m)), idx, axis=1)
            est = fn(subs)
            spread[name][n] = est
            print(f"  {focus} {name:<13s} from {n:2d} runs: estimates 5-95% range [{np.quantile(est, .05):.3f}, "
                  f"{np.quantile(est, .95):.3f}], mean {est.mean():.3f}, std {est.std():.3f}   ({n_pool}-run value {truth:.3f})")
    # Coverage of the nominal 95% percentile stratified bootstrap CI
    coverage = {}
    for name, fn in PREC_METRICS.items():
        truth = st.aggregate(pool, fn)
        coverage[name] = {}
        for n in ns:
            hit = 0
            for _ in range(n_cov):
                idx = np.argsort(rng.random((n_pool, m)), axis=0)[:n]
                sub = np.take_along_axis(pool, idx, axis=0)
                _, lo, hi = st.stratified_bootstrap_ci(sub, fn, reps=cov_reps, rng=rng)
                hit += lo <= truth <= hi
            coverage[name][n] = hit / n_cov
        print(f"  coverage of the nominal 95% CI for the {name:<13s} of {focus} ({n_cov} subsets, {cov_reps} resamples): "
              + ", ".join(f"{n} runs {coverage[name][n]:.2f}" for n in ns))
    # Ranking errors: two agents whose pool IQMs differ a little
    a, b = agents[0], agents[3] if len(agents) > 3 else agents[-1]
    truth_sign = np.sign(st.aggregate(final[a], st.agg_iqm) - st.aggregate(final[b], st.agg_iqm))
    flips = {}
    for n in ns:
        wrong = 0
        for _ in range(n_sub):
            ia = np.argsort(rng.random((n_pool, m)), axis=0)[:n]
            ib = np.argsort(rng.random((n_pool, m)), axis=0)[:n]
            ea = st.aggregate(np.take_along_axis(final[a], ia, axis=0), st.agg_iqm)
            eb = st.aggregate(np.take_along_axis(final[b], ib, axis=0), st.agg_iqm)
            wrong += np.sign(ea - eb) != truth_sign
        flips[n] = wrong / n_sub
    print(f"  {a} vs {b}: {N_POOL}-run IQMs {st.aggregate(final[a], st.agg_iqm):.3f} vs "
          f"{st.aggregate(final[b], st.agg_iqm):.3f}; probability that n-run IQMs rank them the other way: "
          + ", ".join(f"n={n}: {flips[n]:.2f}" for n in ns))
    return focus, spread, coverage, (a, b, flips)


def part_d(curves, agents, rng):
    out = {}
    for a in agents:
        c = curves[a][:, :N_REPORT]                      # (C, runs, tasks)
        out[a] = np.array([st.stratified_bootstrap_ci(c[i], st.agg_iqm, reps=500, rng=rng) for i in range(c.shape[0])])
    return out


def make_figures(tasks, final, agents, res_a, taus, prof, poi, prec, curves_iqm, checkpoints):
    import plot_style
    plt = plot_style.setup()
    C, MK = plot_style.C, ["o", "s", "^", "D"]
    fig_dir = __file__.rsplit("/", 1)[0] + "/figures/"

    # Figure 1: aggregate metrics with CIs
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for k, name in enumerate(st.AGGREGATES):
        ax = axes[k]
        for i, a in enumerate(agents):
            p, lo, hi, pool = res_a[a][name]
            ax.plot([lo, hi], [i, i], color=C[i], lw=6, alpha=0.45, solid_capstyle="butt")
            ax.plot(p, i, marker="|", color=C[i], ms=16, mew=2.5)
            ax.plot(pool, i, marker="x", color=plot_style.INK, ms=6, mew=1.2, ls="none",
                    label=f"{N_POOL}-run estimate" if i == 0 else None)
        ax.set_title({"Median": "Median of task means"}.get(name, name) + (" (lower is better)" if name == "Optimality gap" else ""))
        ax.set_xlabel("normalized score")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(agents)))
    axes[0].set_yticklabels(agents)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Aggregate metrics over 5 tasks, {N_REPORT} runs per task: point estimate and 95% stratified-bootstrap CI",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(fig_dir + "eval_aggregates.png")
    plt.close(fig)

    # Figure 2: performance profiles + probability of improvement
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    for i, a in enumerate(agents):
        p, lo, hi = prof[a]
        ax.plot(taus, p, color=C[i], ls=["-", "--", "-.", ":"][i], label=a)
        ax.fill_between(taus, lo, hi, color=C[i], alpha=0.15, lw=0)
    ax.set_xlabel(r"normalized score $\tau$")
    ax.set_ylabel(r"fraction of runs with score $> \tau$")
    ax.set_title(f"Performance profiles ({N_REPORT} runs x 5 tasks, 95% CI bands)")
    ax.legend()
    ax = axes[1]
    keys = list(poi)
    for k, (x, y) in enumerate(keys):
        p, lo, hi, pool = poi[(x, y)]
        ax.plot([lo, hi], [k, k], color=C[0], lw=6, alpha=0.45, solid_capstyle="butt")
        ax.plot(p, k, marker="|", color=C[0], ms=16, mew=2.5)
        ax.plot(pool, k, marker="x", color=plot_style.INK, ms=6, mew=1.2, ls="none",
                label=f"{N_POOL}-run estimate" if k == 0 else None)
    ax.axvline(0.5, color=plot_style.GREY, lw=1, ls="--")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"P({x} > {y})" for x, y in keys], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("probability of improvement")
    ax.set_title(f"Probability of improvement ({N_REPORT} runs, 95% CI)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(fig_dir + "eval_profiles_poi.png")
    plt.close(fig)

    # Figure 3: the statistical precipice
    focus, spread, coverage, (a, b, flips) = prec
    ns = sorted(next(iter(spread.values())).keys())
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    names = list(PREC_METRICS)
    width = 0.2
    for k, name in enumerate(names):
        data = [spread[name][n] for n in ns]
        pos = np.arange(len(ns)) + (k - 1.5) * width
        bp = ax.boxplot(data, positions=pos, widths=width * 0.85, whis=(5, 95), showfliers=False,
                        patch_artist=True, medianprops={"color": plot_style.INK})
        for box in bp["boxes"]:
            box.set_facecolor(C[k])
            box.set_alpha(0.55)
        ax.plot([], [], color=C[k], lw=6, alpha=0.55, label=name)
    ax.set_xticks(np.arange(len(ns)))
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("runs per task")
    ax.set_ylabel("point estimate")
    ax.set_title(f"{focus}: spread of estimates\n(boxes 25-75%, whiskers 5-95%)")
    ax.legend(fontsize=8)
    ax = axes[1]
    for k, name in enumerate(names):
        ax.plot(ns, [coverage[name][n] for n in ns], marker=MK[k], color=C[k], ls=["-", "--", "-.", ":"][k], label=name)
    ax.axhline(0.95, color=plot_style.GREY, ls="--", lw=1, label="nominal 95%")
    ax.set_xlabel("runs per task")
    ax.set_ylabel("coverage")
    ax.set_xticks(ns)
    ax.set_title("Coverage of the percentile\nstratified-bootstrap 95% CI")
    ax.legend(fontsize=8)
    ax = axes[2]
    ax.plot(ns, [flips[n] for n in ns], marker="o", color=C[0])
    ax.set_xlabel("runs per task")
    ax.set_ylabel("probability of the wrong ranking")
    ax.set_xticks(ns)
    ax.set_ylim(0, 0.5)
    ax.set_title(f"IQM ranks {a} vs {b}\nopposite to the {N_POOL}-run ranking")
    fig.tight_layout()
    fig.savefig(fig_dir + "eval_precipice.png")
    plt.close(fig)

    # Figure 4: IQM learning curves
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    frac = np.linspace(0.1, 1.0, len(checkpoints))
    for i, a in enumerate(agents):
        c = curves_iqm[a]
        ax.plot(frac, c[:, 0], color=C[i], marker=MK[i], ms=4, ls=["-", "--", "-.", ":"][i], label=a)
        ax.fill_between(frac, c[:, 1], c[:, 2], color=C[i], alpha=0.15, lw=0)
    ax.set_xlabel("fraction of each task's training budget")
    ax.set_ylabel("IQM normalized score")
    ax.set_title(f"IQM learning curves, {N_REPORT} runs x 5 tasks (95% CI bands)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir + "eval_curves.png")
    plt.close(fig)
    print("Figures written to", fig_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test: few runs, short training, no figures")
    args = ap.parse_args()
    global N_POOL, N_REPORT, BOOT_REPS
    if args.quick:
        N_POOL, N_REPORT, BOOT_REPS = 12, 6, 300
        agents, step_scale = ["Q-learning", "Double Q", "Expected SARSA", "SARSA"], 0.05
        ns, n_sub, n_cov, cov_reps = [3, 6], 100, 20, 200
    else:
        agents, step_scale = ["Q-learning", "Double Q", "Expected SARSA", "SARSA"], 1.0
        ns, n_sub, n_cov, cov_reps = [3, 5, 10, 20], 2000, 200, 500
    print(f"seed={SEED}  runs per (agent, task)={N_POOL}  reported runs={N_REPORT}  bootstrap reps={BOOT_REPS}")
    print(f"agents: alpha={ALPHA}, eps-greedy eps={EPS}, gamma={GAMMA}, zero-initialized Q; quick={args.quick}")
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    tasks, final, curves = run_suite(N_POOL, step_scale, agents)
    res_a = part_a(final, agents, rng)
    taus, prof, poi = part_b(final, agents, rng)
    prec = part_c(final, agents, rng, ns, n_sub, n_cov, cov_reps, [t.name for t in tasks])
    curves_iqm = part_d(curves, agents, rng)
    print("\nPart D: IQM at 10% / 50% / 100% of the budget (" + f"{N_REPORT} runs):")
    for a in agents:
        c = curves_iqm[a]
        print(f"  {a:<15s} {c[0, 0]:.3f}  {c[len(c) // 2 - 1, 0]:.3f}  {c[-1, 0]:.3f}")
    print("\nPer-task mean normalized score over all runs (rows: agents):")
    print("  " + " " * 15 + "".join(f"{t.name:>16s}" for t in tasks))
    for a in agents:
        print(f"  {a:<15s}" + "".join(f"{v:16.3f}" for v in final[a].mean(0)))
    if not args.quick:
        make_figures(tasks, final, agents, res_a, taus, prof, poi, prec, curves_iqm, np.arange(curves[agents[0]].shape[0]))
    print(f"Total time {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
