"""Statistically sound evaluation of RL agents, from scratch (Chapter 20, Section 6).

Implements the toolkit recommended by Agarwal, Schwarzer, Castro, Courville &
Bellemare, "Deep Reinforcement Learning at the Edge of the Statistical
Precipice" (NeurIPS 2021), with numpy and scipy only:

* aggregate metrics on a (runs x tasks) matrix of normalized scores:
  mean, median of per-task means (as in rliable), pooled median, interquartile
  mean (IQM, Eq. 20.10) and optimality gap (Eq. 20.11);
* stratified bootstrap confidence intervals (Algorithm 20.3): resample runs
  with replacement *independently within each task*, recompute the aggregate;
* performance profiles (score distributions, Eq. 20.12) with bootstrap bands;
* probability of improvement P(X > Y) (Eq. 20.13), the average over tasks of
  the Mann-Whitney U statistic normalized to [0, 1], with bootstrap CIs.

Conventions: every score array has shape (n_runs, n_tasks); scores are
already normalized per task (e.g. 0 = random policy, 1 = optimal policy).
Bootstrapped versions operate on arrays of shape (reps, n_runs, n_tasks).

Run this file directly to execute the self-tests, which check the functions
against scipy and against the worked example of Section 6.3 of the chapter.
"""
import argparse

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# Aggregate metrics.  Each takes an array of shape (..., n_runs, n_tasks) and
# reduces the last two axes, so the same function serves the point estimate
# (a single (n, m) matrix) and thousands of bootstrap replicates at once
# (shape (reps, n, m)).
# --------------------------------------------------------------------------


def _pooled(x):
    """Flatten the (runs, tasks) axes into one axis of N*M scores."""
    return x.reshape(*x.shape[:-2], -1)


def agg_mean(x):
    return x.mean(axis=(-2, -1))


def agg_median(x):
    """Median of the M per-task mean scores (the traditional Atari summary).

    This is the definition used by Agarwal et al. (2021) and rliable
    (np.median(np.mean(scores, axis=0))). With few tasks it is just the mean
    of one or two tasks (Section 6.2)."""
    return np.median(x.mean(axis=-2), axis=-1)


def agg_pooled_median(x):
    """Median of all N*M run-task scores pooled together (NOT rliable's median)."""
    return np.median(_pooled(x), axis=-1)


def agg_iqm(x):
    """Interquartile mean of the N*M pooled scores (Eq. 20.10): drop floor(0.25 NM)
    values at each end, average the rest.

    Identical to scipy.stats.trim_mean(x, 0.25), which is what rliable uses.
    With NM not divisible by 4 the cut is conservative (Section 6.2)."""
    flat = _pooled(x)
    n = flat.shape[-1]
    cut = int(0.25 * n)
    return np.sort(flat, axis=-1)[..., cut:n - cut].mean(axis=-1)


def agg_optimality_gap(x, gamma=1.0):
    """Average shortfall below the target score gamma (lower is better), Eq. 20.11."""
    return gamma - np.minimum(x, gamma).mean(axis=(-2, -1))


AGGREGATES = {"Median": agg_median, "IQM": agg_iqm, "Mean": agg_mean,
              "Optimality gap": agg_optimality_gap}


def aggregate(scores, fn):
    """Point estimate of an aggregate over a (runs, tasks) score matrix."""
    scores = np.asarray(scores, dtype=float)
    return float(fn(scores[None])[0])


# --------------------------------------------------------------------------
# Stratified bootstrap (Algorithm 20.3)
# --------------------------------------------------------------------------


def stratified_resample(scores, reps, rng):
    """Return (reps, n_runs, n_tasks): runs resampled with replacement per task."""
    scores = np.asarray(scores, dtype=float)
    n, m = scores.shape
    idx = rng.integers(0, n, size=(reps, n, m))       # independent draws for every task
    return scores[idx, np.arange(m)[None, None, :]]


def stratified_bootstrap_ci(scores, fn, reps=2000, alpha=0.05, rng=None):
    """Percentile stratified-bootstrap CI for an aggregate.

    Returns (point_estimate, lower, upper)."""
    rng = np.random.default_rng(0) if rng is None else rng
    point = aggregate(scores, fn)
    vals = fn(stratified_resample(scores, reps, rng))     # (reps,)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------
# Performance profiles (Eq. 20.12)
# --------------------------------------------------------------------------


def performance_profile(scores, taus):
    """F(tau) = (1/M) sum_m (1/N) sum_n 1[x_{n,m} > tau], for each tau.

    Works on (n, m) or batched (reps, n, m) arrays."""
    scores = np.asarray(scores, dtype=float)
    taus = np.asarray(taus, dtype=float)
    above = scores[..., None] > taus                    # (..., n, m, n_tau)
    return above.mean(axis=-2).mean(axis=-2)            # average runs, then tasks


def performance_profile_ci(scores, taus, reps=2000, alpha=0.05, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    point = performance_profile(scores, taus)
    boot = performance_profile(stratified_resample(scores, reps, rng), taus)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2], axis=0)
    return point, lo, hi


# --------------------------------------------------------------------------
# Probability of improvement (Eq. 20.13)
# --------------------------------------------------------------------------


def probability_of_improvement(x, y):
    """P(X > Y): average over tasks of the normalized Mann-Whitney U statistic.

    x: (..., n, m), y: (..., k, m). Ties count 1/2. Uses ranks of the pooled
    sample, so it costs O((n+k) log(n+k)) per task instead of O(n k)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = x.shape[-2], y.shape[-2]
    pooled = np.concatenate([x, y], axis=-2)            # (..., n + k, m)
    ranks = stats.rankdata(pooled, axis=-2)             # average ranks for ties
    u = ranks[..., :n, :].sum(axis=-2) - n * (n + 1) / 2.0   # U statistic of x, per task
    return (u / (n * k)).mean(axis=-1)                  # normalize to [0, 1], average tasks


def probability_of_improvement_ci(x, y, reps=2000, alpha=0.05, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    point = float(probability_of_improvement(x, y))
    bx = stratified_resample(x, reps, rng)
    by = stratified_resample(y, reps, rng)
    vals = probability_of_improvement(bx, by)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def _poi_bruteforce(x, y):
    """O(n k) definition, used only to test the rank-based version."""
    per_task = []
    for j in range(x.shape[1]):
        s = 0.0
        for xi in x[:, j]:
            for yj in y[:, j]:
                s += 1.0 if xi > yj else (0.5 if xi == yj else 0.0)
        per_task.append(s / (x.shape[0] * y.shape[0]))
    return float(np.mean(per_task))


# --------------------------------------------------------------------------
# Self-tests
# --------------------------------------------------------------------------


def _self_test(quick):
    rng = np.random.default_rng(0)
    print("rl_stats self-test (seed 0)")

    # 1. The worked example of Section 6.3: 2 tasks (columns) x 4 runs (rows).
    X = np.array([[0.1, 0.3], [0.4, 0.5], [0.5, 0.7], [0.8, 1.6]])
    Y = np.array([[0.2, 0.2], [0.3, 0.5], [0.6, 0.5], [0.7, 0.9]])
    got = {
        "mean X": aggregate(X, agg_mean), "median X": aggregate(X, agg_median),
        "median Y": aggregate(Y, agg_median), "pooled median X": aggregate(X, agg_pooled_median),
        "IQM X": aggregate(X, agg_iqm), "IQM Y": aggregate(Y, agg_iqm),
        "opt. gap X": aggregate(X, agg_optimality_gap), "opt. gap Y": aggregate(Y, agg_optimality_gap),
        "P(X>Y)": float(probability_of_improvement(X, Y)),
        "profile X at 0.5": float(performance_profile(X, [0.5])[0]),
        "profile Y at 0.5": float(performance_profile(Y, [0.5])[0]),
    }
    expected = {"mean X": 0.6125, "median X": 0.6125, "median Y": 0.4875, "pooled median X": 0.5, "IQM X": 0.525, "IQM Y": 0.475,
                "opt. gap X": 0.4625, "opt. gap Y": 0.5125, "P(X>Y)": 0.5625,
                "profile X at 0.5": 0.375, "profile Y at 0.5": 0.375}
    print("  worked example (Section 6.3):")
    for key in expected:
        print(f"    {key:<18s} = {got[key]:.4f}   (by hand: {expected[key]:.4f})")
        assert abs(got[key] - expected[key]) < 1e-12, key

    # 2. IQM equals scipy's 25% trimmed mean for many sizes (including n % 4 != 0).
    for n in range(1, 60):
        v = rng.normal(size=n)
        assert abs(aggregate(v[:, None], agg_iqm) - stats.trim_mean(v, 0.25)) < 1e-12
    print("  IQM == scipy.stats.trim_mean(x, 0.25) for n = 1..59: OK")

    # 3. Rank-based P(X>Y) equals the brute-force double sum, with ties; and the
    #    one-task case equals scipy's Mann-Whitney U / (n k).
    for _ in range(50):
        n, k, m = rng.integers(1, 9), rng.integers(1, 9), rng.integers(1, 4)
        x = rng.integers(0, 4, size=(n, m)).astype(float)   # small integers -> many ties
        y = rng.integers(0, 4, size=(k, m)).astype(float)
        assert abs(probability_of_improvement(x, y) - _poi_bruteforce(x, y)) < 1e-12
    x, y = rng.normal(size=(12, 1)), rng.normal(0.3, 1, size=(9, 1))
    u = stats.mannwhitneyu(x[:, 0], y[:, 0]).statistic
    assert abs(probability_of_improvement(x, y) - u / (12 * 9)) < 1e-12
    print("  P(X>Y) == brute-force double sum (with ties) and == Mann-Whitney U/(nk): OK")

    # 4. Stratified resampling never mixes tasks: column j only contains column j's values.
    S = np.arange(20, dtype=float).reshape(5, 4)        # run i, task j -> 4 i + j
    B = stratified_resample(S, 100, rng)
    assert np.all(B % 4 == np.arange(4)), "a task received another task's scores"
    print("  stratified resampling keeps every resampled score in its own task: OK")

    # 5. Coverage sanity check of the percentile bootstrap for the mean of a
    #    normal sample with 3 vs 10 runs per task (5 tasks). With few runs the
    #    percentile interval is too narrow, a point the chapter returns to.
    #    The bootstrap's plug-in variance is too small by the factor (n - 1)/n,
    #    and each resampled column of 3 runs has only ~2.1 distinct values.
    reps, trials = (300, 100) if quick else (1000, 1000)
    for n in (3, 10):
        hit = 0
        for _ in range(trials):
            s = rng.normal(0.5, 0.2, size=(n, 5))
            _, lo, hi = stratified_bootstrap_ci(s, agg_mean, reps=reps, rng=rng)
            hit += lo <= 0.5 <= hi
        cov = hit / trials
        print(f"  coverage of the nominal 95% CI for the mean, {n:2d} runs x 5 tasks: {cov:.3f}"
              f"  ({trials} trials, Monte Carlo SE {np.sqrt(cov * (1 - cov) / trials):.3f})")
    print("All self-tests passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    _self_test(ap.parse_args().quick)
