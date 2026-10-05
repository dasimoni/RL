"""Seeds, nondeterminism, sensitivity and diagnostics with A2C on CartPole (Chapter 20, Sections 5.4 and 7.4).

Part 1  Henderson et al. (2018)-style experiment: 10 runs of the SAME algorithm
        and hyperparameters, split into two groups of 5 seeds. How different
        can two groups look? We test the natural split (seeds 0-4 vs 5-9) and
        all 126 distinct splits with Welch's t-test.
Part 2  Determinism: re-running seed 0 must reproduce it bit for bit
        (CPU, one thread, fixed seeds).
Part 3  Which source of randomness matters? Vary only the network
        initialization, only the environment resets, or only the action
        sampling (5 runs each) and compare the spread with Part 1.
Part 4  Learning-rate sensitivity: 4 learning rates x 5 seeds.
Part 5  A diagnostics dashboard (Section 7.4): a run at the best learning rate
        of the sweep next to the worst seed at the largest learning rate.

Two summaries of a run, which answer different questions (Section 6.1):
  FINAL performance     mean return of the FINAL policy, acting greedily, on 10
                        episodes of a separately seeded evaluation environment
                        (what was learned);
  LEARNING performance  average, over all training steps, of the return of the
                        episode that step belongs to (step-weighted area under
                        the learning curve: how fast and how stably it learned).
For comparison we also print two training-return summaries over the last 10k
steps: episode-weighted (every episode that ended in the window counts once)
and step-weighted. The episode-weighted one over-weights short, failed episodes.

Run:  python code/ch20_deep_rl_in_practice/seed_variance.py [--quick]
"""
import argparse
import itertools
import time

import numpy as np
from scipy import stats

import cartpole_a2c as ca
import rl_stats as st

LRS = [3e-4, 1e-3, 3e-3, 1e-2]


def final_perf(eps, *_):
    """FINAL performance: mean return of the final greedy policy on 10 evaluation episodes."""
    return float(eps["eval_greedy"].mean())


def learning_perf(eps):
    """LEARNING performance: step-weighted mean training return over the whole run."""
    r, L = eps["return"], eps["length"]
    return float((r * L).sum() / L.sum())


def split_analysis(name, perf, half):
    """Henderson-style: compare seeds 0..half-1 with the rest, and look at all splits."""
    g1, g2 = perf[:half], perf[half:]
    t_nat = stats.ttest_ind(g1, g2, equal_var=False) if np.ptp(perf) > 0 else None
    diffs, pvals = [], []
    for grp in itertools.combinations(range(len(perf)), half):
        if 0 not in grp:
            continue                                 # each split counted once
        a = perf[list(grp)]
        b = perf[[i for i in range(len(perf)) if i not in grp]]
        diffs.append(abs(a.mean() - b.mean()))
        pvals.append(stats.ttest_ind(a, b, equal_var=False).pvalue if (np.ptp(a) > 0 or np.ptp(b) > 0) else 1.0)
    diffs, pvals = np.array(diffs), np.nan_to_num(np.array(pvals), nan=1.0)
    print(f"  {name}: seeds 0-{half - 1} mean {g1.mean():.1f} (sd {g1.std(ddof=1):.1f}); seeds {half}-{len(perf) - 1} mean"
          f" {g2.mean():.1f} (sd {g2.std(ddof=1):.1f})" + (f"; Welch t = {t_nat.statistic:.2f}, p = {t_nat.pvalue:.3f}" if t_nat else "")
          + f"\n    over all {len(diffs)} splits into two groups of {half}: |difference of means| median {np.median(diffs):.1f},"
          f" max {diffs.max():.1f}; splits with Welch p < 0.05: {np.mean(pvals < 0.05):.3f}")


def train_window(eps, total_steps, last=10_000):
    """Episode-weighted and step-weighted mean training return over the last `last` steps."""
    m = eps["step"] > total_steps - last
    r, L = eps["return"][m], eps["length"][m]
    return float(r.mean()), float((r * L).sum() / L.sum())


def run(total, **kw):
    t0 = time.time()
    log, eps = ca.train(ca.Config(total_steps=total, **kw))
    return log, eps, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        total, n_seeds, n_src, lrs = 4_000, 4, 2, [1e-3, 1e-2]
    else:
        total, n_seeds, n_src, lrs = 80_000, 10, 5, LRS
    base = ca.Config(total_steps=total)
    print(f"A2C on CartPole-v1, {total} steps per run; base config: lr={base.lr}, n_envs={base.n_envs}, "
          f"n_steps={base.n_steps}, gamma={base.gamma}, GAE lambda={base.gae_lambda}, entropy={base.ent_coef}, "
          f"reward scale={base.reward_scale}; quick={args.quick}")
    print("FINAL performance = mean return of the final policy, acting greedily, on 10 episodes of a separately seeded"
          " evaluation env; LEARNING performance = step-weighted mean training return over the whole run;"
          " last-window training returns: last 10k steps" + (" (2k in quick mode)" if args.quick else ""))
    last = 2_000 if args.quick else 10_000
    t_start = time.time()
    rng = np.random.default_rng(0)

    # ---------------- Part 1: same algorithm, two groups of seeds ----------------
    runs = {}
    for s in range(n_seeds):
        log, eps, dt = run(total, init_seed=s, env_seed=s, act_seed=s)
        runs[s] = (log, eps)
        ew, sw = train_window(eps, total, last)
        print(f"  seed {s}: final (greedy eval) {final_perf(eps):6.1f}   stochastic eval {eps['eval_stochastic'].mean():6.1f}"
              f"   learning {learning_perf(eps):6.1f}   last-window training return: episode-weighted {ew:6.1f},"
              f" step-weighted {sw:6.1f}   ({dt:.1f} s)")
    perf = np.array([final_perf(runs[s][1]) for s in range(n_seeds)])
    lperf = np.array([learning_perf(runs[s][1]) for s in range(n_seeds)])
    half = n_seeds // 2
    print("\nPart 1: two groups of seeds")
    split_analysis("final performance   ", perf, half)
    split_analysis("learning performance", lperf, half)
    for name, v in (("final", perf), ("learning", lperf)):
        p, lo, hi = st.stratified_bootstrap_ci(v[:, None], st.agg_iqm, reps=2000, rng=rng)
        print(f"  all {n_seeds} seeds, {name} performance: IQM {p:.1f} [95% CI {lo:.1f}, {hi:.1f}], mean {v.mean():.1f},"
              f" sd {v.std(ddof=1):.1f}, min {v.min():.1f}, max {v.max():.1f}")

    # ---------------- Part 2: determinism ----------------
    log_b, eps_b, _ = run(total, init_seed=0, env_seed=0, act_seed=0)
    same = np.array_equal(eps_b["return"], runs[0][1]["return"]) and np.array_equal(log_b["approx_kl"], runs[0][0]["approx_kl"])
    print(f"\nPart 2: re-running seed 0 reproduces every episode return and every logged KL bit for bit: {same}")

    # ---------------- Part 3: one source of randomness at a time ----------------
    print("\nPart 3: vary ONE random stream, keep the other two at seed 0")
    src_perf, src_lperf = {}, {}
    for name, key in (("network init", "init_seed"), ("env resets", "env_seed"), ("action sampling", "act_seed")):
        vals, lvals = [], []
        for s in range(1, n_src + 1):
            kw = dict(init_seed=0, env_seed=0, act_seed=0)
            kw[key] = s
            _, eps, _ = run(total, **kw)
            vals.append(final_perf(eps))
            lvals.append(learning_perf(eps))
        src_perf[name], src_lperf[name] = np.array(vals), np.array(lvals)
        print(f"  only {name:<16s} varies: final {np.round(vals, 1)} (sd {np.std(vals, ddof=1):.1f});"
              f" learning {np.round(lvals, 1)} (sd {np.std(lvals, ddof=1):.1f})")
    print(f"  everything varies (Part 1):   final sd {perf.std(ddof=1):.1f}; learning sd {lperf.std(ddof=1):.1f}")

    # ---------------- Part 4: learning-rate sensitivity ----------------
    print("\nPart 4: learning-rate sensitivity (seeds 0-%d)" % (n_src - 1))
    lr_perf, lr_lperf, lr_logs = {}, {}, {}
    for lr in lrs:
        vals = []
        for s in range(n_src):
            if lr == base.lr and s in runs:
                log, eps = runs[s]
            else:
                log, eps, _ = run(total, lr=lr, init_seed=s, env_seed=s, act_seed=s)
            vals.append(final_perf(eps))
            lr_logs[(lr, s)] = (log, eps)
        lr_perf[lr] = np.array(vals)
        lr_lperf[lr] = np.array([learning_perf(lr_logs[(lr, s)][1]) for s in range(n_src)])
        tw = np.array([train_window(lr_logs[(lr, s)][1], total, last) for s in range(n_src)])
        print(f"  lr {lr:.0e}: final {np.round(vals, 1)} (mean {np.mean(vals):.1f}); learning {np.round(lr_lperf[lr], 1)}"
              f" (mean {lr_lperf[lr].mean():.1f})\n             last-window training return, episode-weighted {np.round(tw[:, 0], 1)},"
              f" step-weighted {np.round(tw[:, 1], 1)}")

    # ---------------- Part 5: diagnostics summary ----------------
    # A run at the best learning rate of the sweep (seed 0) next to the WORST seed at
    # the largest learning rate: chosen deliberately, to show what a failure looks like.
    best_lr = max(lrs, key=lambda lr: (lr_perf[lr].mean(), lr_lperf[lr].mean()))   # ties broken by learning perf.
    worst_seed = int(np.argmin(lr_perf[lrs[-1]]))
    dash = [(best_lr, 0), (lrs[-1], worst_seed)]
    print(f"\nPart 5: diagnostics (averages over the last 20% of updates) for lr {best_lr:.0e} seed 0 (best lr)"
          f" and lr {lrs[-1]:.0e} seed {worst_seed} (worst seed at that lr)")
    vmax = 1.0 / (1.0 - base.gamma)
    for lr, s in dash:
        log, eps = lr_logs[(lr, s)]
        k = max(1, len(log["step"]) // 5)
        ev = log["explained_var"][-k:]
        ew, sw = train_window(eps, total, last)
        print(f"  lr {lr:.0e} seed {s}: entropy {log['entropy'][-k:].mean():.3f}  min entropy {log['entropy'].min():.3f}  approx KL median {np.median(log['approx_kl'][-k:]):.1e}"
              f" max {log['approx_kl'].max():.1e}  explained var median {np.nanmedian(ev):+.2f}"
              f"  |g_actor| median {np.median(log['gn_actor'][-k:]):.2f}  |g_critic| median {np.median(log['gn_critic'][-k:]):.2f}"
              f"  final {final_perf(eps):.1f}, learning {learning_perf(eps):.1f} (last-window training return: episode-weighted {ew:.1f}, step-weighted {sw:.1f})")
        q = len(log["step"]) // 4
        ent = log["entropy"]
        print(f"     entropy by quarter of training (mean): " + ", ".join(f"{ent[i * q:(i + 1) * q].mean():.2f}" for i in range(4))
              + f"; entropy at 10k steps {ent[min(np.searchsorted(log['step'], 10_000), len(ent) - 1)]:.2f}")
        print(f"     |g_actor| max {log['gn_actor'].max():.2f} at step {int(log['step'][np.argmax(log['gn_actor'])])} (median {np.median(log['gn_actor']):.3f});"
              f" |g_critic| max {log['gn_critic'].max():.2f}")
        vs = eps["v_start"]
        print(f"     critic V(s0) (original units): max {vs.max():.1f}; {int((vs > vmax).sum())} of {len(vs)} episode starts above"
              f" 1/(1-gamma) = {vmax:.0f}; max batch-mean V {log['value_mean'].max():.1f}; last-5-episode V(s0) {vs[-5:].mean():.1f}"
              f" vs realized discounted return {eps['disc_return'][-5:].mean():.1f}")
        cv = ca.curve(eps, total, n_points=80, window=4_000)
        print(f"     return curve (4k window), at 5k, 10k, ..., 80k steps: " + " ".join(f"{v:.0f}" for v in cv[1][4::5]))
    if not args.quick:
        make_figures(runs, (perf, lperf), total, (src_perf, src_lperf), (lr_perf, lr_lperf), lr_logs, dash)
    print(f"Total time {time.time() - t_start:.1f} s")


def make_figures(runs, perfs, total, src_perfs, lr_perfs, lr_logs, dash):
    import plot_style
    plt = plot_style.setup()
    C = plot_style.C
    fig_dir = __file__.rsplit("/", 1)[0] + "/figures/"
    n = len(runs)
    half = n // 2

    # Figure 1: two groups of seeds, sources of randomness, LR sensitivity
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    curves = np.array([ca.curve(runs[s][1], total, n_points=40, window=8_000)[1] for s in range(n)])
    grid = ca.curve(runs[0][1], total, n_points=40, window=8_000)[0]
    rng = np.random.default_rng(1)
    for k, (grp, lab, ls) in enumerate(((range(half), f"seeds 0-{half - 1}", "-"), (range(half, n), f"seeds {half}-{n - 1}", "--"))):
        c = curves[list(grp)]
        for row in c:
            ax.plot(grid, row, color=C[k], lw=0.6, alpha=0.35)
        m = np.nanmean(c, 0)
        boot = np.array([np.nanmean(c[rng.integers(0, len(c), len(c))], 0) for _ in range(1000)])
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        ax.plot(grid, m, color=C[k], ls=ls, lw=2.2, label=f"{lab}: mean, 95% bootstrap CI")
        ax.fill_between(grid, lo, hi, color=C[k], alpha=0.18, lw=0)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return (8k-step window)")
    ax.set_title("Same algorithm, same hyperparameters, two groups of 5 seeds")
    ax.legend(loc="upper left", fontsize=8)
    plot_style.kfmt(ax)

    ax = axes[1]
    (perf, lperf), (src_perf, src_lperf), (lr_perf, lr_lperf) = perfs, src_perfs, lr_perfs
    groups = [("init only", "network init"), ("env only", "env resets"), ("actions only", "action sampling"), ("all (Part 1)", None)]
    for k, (lab, key) in enumerate(groups):
        for off, vals, mk, fill in ((-0.17, perf if key is None else src_perf[key], "o", C[k]),
                                    (0.17, lperf if key is None else src_lperf[key], "s", "none")):
            jit = (np.random.default_rng(k).random(len(vals)) - 0.5) * 0.14
            ax.plot(np.full(len(vals), k + off) + jit, vals, mk, color=C[k], mfc=fill, ms=6, alpha=0.85)
            ax.plot([k + off - 0.12, k + off + 0.12], [np.mean(vals)] * 2, color=plot_style.INK, lw=1.5)
    ax.plot([], [], "o", color=plot_style.GREY, label="final (greedy eval)")
    ax.plot([], [], "s", color=plot_style.GREY, mfc="none", label="learning (avg over training)")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel("performance (return)")
    ax.set_title("Which random stream varies? (lines = means)")
    ax.set_ylim(0, 520)
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[2]
    lrs = sorted(lr_perf)
    for k, lr in enumerate(lrs):
        for off, vals, mk, fill in ((-0.06, lr_perf[lr], "o", C[0]), (0.06, lr_lperf[lr], "s", "none")):
            jit = (np.random.default_rng(10 + k).random(len(vals)) - 0.5) * 0.06
            ax.plot(np.log10(lr) + off + jit, vals, mk, color=C[0] if mk == "o" else C[1], mfc=fill, ms=6, alpha=0.8)
    ax.plot(np.log10(lrs) - 0.06, [lr_perf[lr].mean() for lr in lrs], color=C[0], lw=1.5, marker="_", ms=14,
            label="final (greedy eval): mean of 5 seeds")
    ax.plot(np.log10(lrs) + 0.06, [lr_lperf[lr].mean() for lr in lrs], color=C[1], lw=1.5, ls="--", marker="_", ms=14,
            label="learning (avg over training): mean")
    ax.set_xticks(np.log10(lrs))
    ax.set_xticklabels([f"{lr:.0e}" for lr in lrs])
    ax.set_xlabel("learning rate (Adam)")
    ax.set_ylabel("performance (return)")
    ax.set_title("Learning-rate sensitivity (dots = seeds)")
    ax.set_ylim(0, 520)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir + "seed_variance.png")
    plt.close(fig)

    # Figure 2: diagnostics dashboard
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.0))
    runs_d = [(lr, s, lr_logs[(lr, s)], col, ls) for (lr, s), col, ls in zip(dash, (C[0], C[1]), ("-", "--"))]

    def smooth(y, k=10):
        y = np.asarray(y, float)
        out = np.full_like(y, np.nan)
        for i in range(len(y)):
            w = y[max(0, i - k + 1): i + 1]
            w = w[~np.isnan(w)]
            out[i] = w.mean() if len(w) else np.nan
        return out

    for lr, sd, (log, eps), col, ls in runs_d:
        lab = f"lr {lr:.0e}, seed {sd}"
        g, c = ca.curve(eps, total, n_points=80, window=4_000)
        axes[0, 0].plot(g, c, color=col, ls=ls, label=lab)
        axes[0, 1].plot(log["step"], log["entropy"], color=col, ls=ls, label=lab)
        axes[0, 2].plot(log["step"], smooth(log["approx_kl"]), color=col, ls=ls, label=lab)
        axes[1, 0].plot(log["step"], smooth(log["explained_var"]), color=col, ls=ls, label=lab)
        axes[1, 1].plot(log["step"], smooth(log["gn_actor"]), color=col, ls=ls, label=f"{lab}: actor")
        axes[1, 1].plot(log["step"], smooth(log["gn_critic"]), color=col, ls=":", lw=1.4, label=f"{lab}: critic")
        es, vs, dr = eps["step"], eps["v_start"], eps["disc_return"]
        order = np.argsort(es)
        axes[1, 2].plot(es[order], smooth(vs[order], 15), color=col, ls=ls, label=f"{lab}: predicted V(s0)")
        axes[1, 2].plot(es[order], smooth(dr[order], 15), color=col, ls=":", lw=1.4, label=f"{lab}: realized G0")
    axes[0, 0].set_title("Episode return (4k-step window)")
    axes[0, 1].set_title("Policy entropy (max ln 2 = 0.69)")
    axes[0, 2].set_title("Approx. KL(old || new) per update (k3, smoothed)")
    axes[0, 2].set_yscale("log")
    axes[1, 0].set_title("Explained variance of the critic (smoothed)")
    axes[1, 0].set_ylim(-1.5, 1.05)
    axes[1, 1].set_title("Gradient norms before clipping (smoothed)")
    axes[1, 1].set_yscale("log")
    axes[1, 2].axhline(1.0 / (1.0 - ca.Config().gamma), color=plot_style.GREY, ls="--", lw=1)
    axes[1, 2].text(total, 1.0 / (1.0 - ca.Config().gamma) + 2, "1/(1-gamma) = 100: bound on the TRUE value",
                    ha="right", va="bottom", fontsize=7.5, color=plot_style.GREY)
    axes[1, 2].set_title("Critic's V(s0) vs realized discounted return")
    for ax in axes.flat:
        ax.set_xlabel("environment steps")
        plot_style.kfmt(ax)
        ax.legend(fontsize=7.5)
    (lr_a, s_a), (lr_b, s_b) = dash
    fig.suptitle(f"Diagnostics dashboard, A2C on CartPole: best learning rate of the sweep (lr {lr_a:.0e}, seed {s_a}) vs "
                 f"the worst of 5 seeds at lr {lr_b:.0e} (seed {s_b})", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir + "diagnostics_dashboard.png")
    plt.close(fig)
    print("Figures written to", fig_dir)


if __name__ == "__main__":
    main()
