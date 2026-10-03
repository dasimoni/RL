"""Classical conditioning and choice: Rescorla-Wagner, the TD model, and model-free vs model-based.

Chapter 05, Section 14 ("TD learning in the brain").  Four small, fully tabular experiments:

Part A -- blocking (Kamin, 1969) under two learning rules
    * Rescorla-Wagner (1972), a TRIAL-level rule.  Each stimulus i has an associative strength
      w_i; on a trial with the set P of stimuli present and US magnitude R (1 = US, 0 = none),
          w_i <- w_i + alpha * (R - sum_{j in P} w_j)        for every i in P.
      This is the LMS / delta rule for the linear prediction w^T x with binary x.
    * The TD model of conditioning (Sutton & Barto, 1990) with a complete-serial-compound (CSC)
      representation: a trial is a sequence of time steps; stimulus i present for k steps since its
      onset activates its own feature x_{i,k}; V_t = w^T x_t, and linear TD(0)
          delta_t = r_{t+1} + gamma * V_{t+1} - V_t,     w <- w + alpha * delta_t * x_t.
      Time steps with no stimulus have x = 0, so V = 0 there (the inter-trial interval).
    Blocking design: phase 1 pairs A with the US; phase 2 pairs the compound AX with the US.
    Control group: phase 1 pairs a different stimulus B with the US instead.  Test: X alone.

Part B -- TD errors over acquisition (the dopamine-like signal of Section 14.1)
    CS A followed by the US, CSC-TD(0).  delta_t at every within-trial step, for selected trials,
    and on an omission probe after training (US withheld once, no learning on the probe).

Part C -- the two-step task (Daw, Gershman, Seymour, Dayan & Dolan, 2011)
    Stage 1: choose a1 in {0, 1}.  a1 leads to stage-2 state a1 with probability 0.7 ("common")
    and to the other state with probability 0.3 ("rare").  Stage 2: choose one of two actions;
    each of the 4 stage-2 actions pays 1 with a probability that drifts as a Gaussian random walk
    (s.d. 0.025) reflected into [0.25, 0.75].
      model-free (MF):   SARSA(lambda)-style TD, lambda = 1: Q_MF(a1) <- Q_MF(a1) + alpha (r - Q_MF(a1))
      model-based (MB):  Q_MB(a1) = sum_s P(s | a1) max_b Q2(s, b), with the known 0.7 / 0.3 model
      hybrid:            Q = w Q_MB + (1 - w) Q_MF with w = 0.5
    All agents learn Q2(s, b) <- Q2(s, b) + alpha (r - Q2(s, b)) and choose by softmax with inverse
    temperature beta at both stages.  We report P(repeat a1 on the next trial) by the previous
    trial's reward and transition type.

Part D -- second-order conditioning (Exercise 5.16)
    Phase 1: A -> US.  Phase 2: B -> A with no US (B occupies the one time step before A's onset).
    Test: B alone.
    RW treats a phase-2 trial as the compound {B, A} with R = 0; the TD model sees B, then A.

Outputs (full mode):
    figures/conditioning_td.png  -- (a) TD errors over acquisition and on omission, (b) blocking,
                                    (c) second-order conditioning
    figures/two_step_task.png    -- stay probabilities of MF, MB and hybrid agents

Run:  python code/ch05_temporal_difference/conditioning_td.py [--quick] [--seed S]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# --- conditioning settings (Parts A, B, D) -----------------------------------------------
GAMMA = 0.95          # TD model discount per time step
ALPHA_TD = 0.1        # TD model step size (per feature)
ALPHA_RW = 0.1        # Rescorla-Wagner step size (equal saliences)
T_TRIAL = 20          # time steps per trial; x = 0 outside the stimuli (inter-trial interval)
ONSET = 6             # CS onset (time step)
ISI = 6               # inter-stimulus interval: the US arrives ISI steps after CS onset
US_TIME = ONSET + ISI  # r_{US_TIME} = 1, i.e. delta_{US_TIME - 1} is the TD error "at the US"
K = T_TRIAL           # CSC features per stimulus (one per time since onset)
STIMULI = ["A", "X", "B"]
N_PHASE1, N_PHASE2 = 200, 100


# ======================================================================================
# Rescorla-Wagner (trial level)
# ======================================================================================
def rw_trial(w, present, R, alpha=ALPHA_RW):
    """One Rescorla-Wagner trial: shared error R - sum of strengths of the stimuli present."""
    err = R - sum(w[s] for s in present)
    for s in present:
        w[s] += alpha * err
    return err


# ======================================================================================
# TD model with a complete serial compound (real time)
# ======================================================================================
def csc_features(stim_spans):
    """x[t] for t = 0..T_TRIAL; stim_spans = {stimulus: (onset, offset)} with offset exclusive.

    Feature index = STIMULI.index(stimulus) * K + (t - onset)."""
    x = np.zeros((T_TRIAL + 1, len(STIMULI) * K))
    for s, (on, off) in stim_spans.items():
        i = STIMULI.index(s)
        for t in range(on, off):
            x[t, i * K + (t - on)] = 1.0
    return x


def td_trial(w, stim_spans, us_time, learn=True, alpha=ALPHA_TD, gamma=GAMMA):
    """Run linear TD(0) through one trial; return the TD errors delta_0..delta_{T-1}.

    rewards: r[us_time] = 1 if us_time is not None.  delta_t = r[t+1] + gamma V_{t+1} - V_t."""
    x = csc_features(stim_spans)
    r = np.zeros(T_TRIAL + 1)
    if us_time is not None:
        r[us_time] = 1.0
    deltas = np.zeros(T_TRIAL)
    for t in range(T_TRIAL):
        v_t, v_next = x[t] @ w, x[t + 1] @ w
        d = r[t + 1] + gamma * v_next - v_t
        deltas[t] = d
        if learn:
            w += alpha * d * x[t]
    return deltas


def onset_value(w, s):
    """Prediction at the onset of stimulus s presented alone = weight of its first CSC feature."""
    return w[STIMULI.index(s) * K]


CS_A = {"A": (ONSET, US_TIME)}              # A on from its onset until the US arrives
CS_B = {"B": (ONSET, US_TIME)}
CS_AX = {"A": (ONSET, US_TIME), "X": (ONSET, US_TIME)}


def part_a():
    res = {}
    # Rescorla-Wagner
    for group, phase1 in (("blocking", ["A"]), ("control", ["B"])):
        w = {s: 0.0 for s in STIMULI}
        for _ in range(N_PHASE1):
            rw_trial(w, phase1, 1.0)
        for _ in range(N_PHASE2):
            rw_trial(w, ["A", "X"], 1.0)
        res[("RW", group)] = (w["A"], w["X"])
    # TD model
    for group, phase1 in (("blocking", CS_A), ("control", CS_B)):
        w = np.zeros(len(STIMULI) * K)
        for _ in range(N_PHASE1):
            td_trial(w, phase1, US_TIME)
        first_phase2 = td_trial(w.copy(), CS_AX, US_TIME, learn=False)
        for _ in range(N_PHASE2):
            td_trial(w, CS_AX, US_TIME)
        res[("TD", group)] = (onset_value(w, "A"), onset_value(w, "X"))
        res[("TD", group, "max|delta| first AX trial")] = np.abs(first_phase2[ONSET:US_TIME]).max()
    return res


def part_b(n_trials):
    w = np.zeros(len(STIMULI) * K)
    deltas = np.zeros((n_trials, T_TRIAL))
    for k in range(n_trials):
        deltas[k] = td_trial(w, CS_A, US_TIME)
    probe = td_trial(w.copy(), CS_A, None, learn=False)       # omission probe, no learning
    return deltas, probe, w


def part_d(n_phase2):
    """Second-order conditioning.  Phase 1: A -> US (N_PHASE1 trials).  Phase 2: B -> A, no US."""
    b_len = 1                     # B occupies the single time step before A's onset
    b_on = ONSET - b_len
    serial = {"B": (b_on, ONSET), "A": (ONSET, US_TIME)}
    # TD model
    w = np.zeros(len(STIMULI) * K)
    for _ in range(N_PHASE1):
        td_trial(w, CS_A, US_TIME)
    vB_td, vA_td = [onset_value(w, "B")], [onset_value(w, "A")]
    for _ in range(n_phase2):
        td_trial(w, serial, None)
        vB_td.append(onset_value(w, "B"))
        vA_td.append(onset_value(w, "A"))
    # Rescorla-Wagner: a phase-2 trial is the compound {B, A} without the US
    wr = {s: 0.0 for s in STIMULI}
    for _ in range(N_PHASE1):
        rw_trial(wr, ["A"], 1.0)
    vB_rw, vA_rw = [wr["B"]], [wr["A"]]
    for _ in range(n_phase2):
        rw_trial(wr, ["B", "A"], 0.0)
        vB_rw.append(wr["B"])
        vA_rw.append(wr["A"])
    return (np.array(vB_td), np.array(vA_td), np.array(vB_rw), np.array(vA_rw), b_len)


# ======================================================================================
# Two-step task
# ======================================================================================
P_COMMON = 0.7


def softmax_choice(q, beta, rng):
    """q: (n, 2) -> sampled action per row."""
    z = beta * (q - q.max(axis=1, keepdims=True))
    p1 = np.exp(z[:, 1]) / np.exp(z).sum(axis=1)
    return (rng.random(len(q)) < p1).astype(int)


def two_step(w_mb, n_agents, n_trials, rng, alpha=0.5, beta=5.0, lam=1.0):
    """Simulate n_agents independent agents in parallel.  Returns arrays (n_agents, n_trials)."""
    idx = np.arange(n_agents)
    q_mf = np.zeros((n_agents, 2))
    q2 = np.zeros((n_agents, 2, 2))                      # [agent, stage-2 state, action]
    p_rew = rng.uniform(0.25, 0.75, size=(n_agents, 2, 2))
    a1s = np.zeros((n_agents, n_trials), dtype=int)
    rews = np.zeros((n_agents, n_trials), dtype=int)
    commons = np.zeros((n_agents, n_trials), dtype=bool)
    for t in range(n_trials):
        q_mb = P_COMMON * q2.max(axis=2) + (1 - P_COMMON) * q2.max(axis=2)[:, ::-1]
        q_net = w_mb * q_mb + (1 - w_mb) * q_mf
        a1 = softmax_choice(q_net, beta, rng)
        common = rng.random(n_agents) < P_COMMON
        s2 = np.where(common, a1, 1 - a1)
        a2 = softmax_choice(q2[idx, s2], beta, rng)
        r = (rng.random(n_agents) < p_rew[idx, s2, a2]).astype(float)
        # model-free TD with an eligibility trace (Daw et al., 2011): lambda = 1 updates the
        # stage-1 value towards the reward itself
        d1 = q2[idx, s2, a2] - q_mf[idx, a1]
        d2 = r - q2[idx, s2, a2]
        q_mf[idx, a1] += alpha * (d1 + lam * d2)
        q2[idx, s2, a2] += alpha * d2
        # drifting reward probabilities, reflected into [0.25, 0.75]
        p_rew += rng.normal(0.0, 0.025, size=p_rew.shape)
        p_rew = np.where(p_rew > 0.75, 1.5 - p_rew, p_rew)
        p_rew = np.where(p_rew < 0.25, 0.5 - p_rew, p_rew)
        a1s[:, t], rews[:, t], commons[:, t] = a1, r, common
    return a1s, rews, commons


def stay_table(a1s, rews, commons):
    stay = (a1s[:, 1:] == a1s[:, :-1])
    r, c = rews[:, :-1], commons[:, :-1]
    tab = {}
    for rv in (1, 0):
        for cv in (True, False):
            m = (r == rv) & (c == cv)
            tab[(rv, cv)] = stay[m].mean()
    # per-agent stay probabilities, for a between-agent standard error of each cell
    se = {}
    for key in tab:
        rv, cv = key
        m = (r == rv) & (c == cv)
        per_agent = np.array([stay[i][m[i]].mean() for i in range(len(stay)) if m[i].any()])
        se[key] = per_agent.std(ddof=1) / np.sqrt(len(per_agent))
    reward_effect = 0.5 * ((tab[(1, True)] - tab[(0, True)]) + (tab[(1, False)] - tab[(0, False)]))
    interaction = (tab[(1, True)] - tab[(1, False)]) - (tab[(0, True)] - tab[(0, False)])
    return tab, se, reward_effect, interaction


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    n_agents = 100 if args.quick else 2000
    n_trials_2step = 201
    n_acq = N_PHASE1
    n_phase2_d = 100
    print(f"[conditioning_td] seed={args.seed}  TD model: gamma={GAMMA} alpha={ALPHA_TD} "
          f"CS onset t={ONSET}, US at t={US_TIME} (ISI={ISI}), {T_TRIAL} steps/trial, CSC features; "
          f"RW: alpha={ALPHA_RW}; phases {N_PHASE1}+{N_PHASE2} trials; "
          f"two-step: {n_agents} agents x {n_trials_2step} trials")
    t0 = time.time()

    # ---------------- Part A
    res = part_a()
    full = GAMMA ** (ISI - 1)
    print("\nPart A -- blocking (test: X presented alone)")
    print("  rule  group     | strength of A   strength of X")
    for rule in ("RW", "TD"):
        for group in ("blocking", "control"):
            a, x = res[(rule, group)]
            print(f"  {rule:4s}  {group:9s} |    {a:7.4f}         {x:7.4f}")
    print(f"  (TD strengths are predictions at stimulus onset; a fully trained CS predicts "
          f"gamma^(ISI-1) = {full:.4f})")
    print(f"  TD, blocking group: max |delta| while A and X are on, first AX trial = "
          f"{res[('TD', 'blocking', 'max|delta| first AX trial')]:.2e}")

    # ---------------- Part B
    deltas, probe, w_b = part_b(n_acq)
    print(f"\nPart B -- TD errors during acquisition of A -> US (CSC-TD(0), {n_acq} trials)")
    print(f"  delta at the cue = delta_{ONSET - 1} (arrival at the CS onset, step {ONSET}); "
          f"delta at the US = delta_{US_TIME - 1} (arrival at step {US_TIME}, when the US is delivered)")
    print("  trial | delta at cue  delta at US | largest delta between cue and US (time step)")
    for k in (1, 2, 5, 10, 20, 30, 50, 100, n_acq):
        d = deltas[k - 1]
        mid = d[ONSET:US_TIME - 1]          # errors on arrival at steps ONSET+1 .. US_TIME-1
        j = int(np.argmax(mid))
        print(f"  {k:5d} |   {d[ONSET - 1]:7.4f}      {d[US_TIME - 1]:7.4f}   |  {mid[j]:7.4f} "
              f"(on arrival at step {ONSET + j + 1})")
    print(f"  omission probe after training: delta at cue = {probe[ONSET - 1]:.4f}, "
          f"delta at the expected US time = {probe[US_TIME - 1]:.4f}; "
          f"elsewhere max |delta| = {np.abs(np.delete(probe, [ONSET - 1, US_TIME - 1])).max():.2e}")
    print(f"  asymptote: delta at cue -> gamma^ISI = {GAMMA ** ISI:.4f}; V just before the US -> 1")
    # trials until the cue response reaches 90% of its asymptote
    k90 = int(np.argmax(deltas[:, ONSET - 1] >= 0.9 * GAMMA ** ISI)) + 1
    k10 = int(np.argmax(deltas[:, US_TIME - 1] <= 0.1)) + 1
    print(f"  delta at cue first >= 90% of asymptote on trial {k90}; "
          f"delta at US first <= 0.1 on trial {k10}")

    # ---------------- Part C
    rng = np.random.default_rng(args.seed)
    agents = {"model-free (w=0)": 0.0, "model-based (w=1)": 1.0, "hybrid (w=0.5)": 0.5}
    stay_results = {}
    print(f"\nPart C -- two-step task: P(repeat stage-1 choice) after the previous trial's outcome "
          f"({n_agents} agents x {n_trials_2step} trials; alpha=0.5, beta=5, lambda=1)")
    print("  agent              | rew/common rew/rare unrew/common unrew/rare | reward effect  interaction | mean reward")
    for name, wmb in agents.items():
        a1s, rews, commons = two_step(wmb, n_agents, n_trials_2step, rng)
        tab, se, re, inter = stay_table(a1s, rews, commons)
        stay_results[name] = (tab, se)
        print(f"  {name:18s} |   {tab[(1, True)]:.3f}     {tab[(1, False)]:.3f}      "
              f"{tab[(0, True)]:.3f}        {tab[(0, False)]:.3f}   |    {re:+.3f}        {inter:+.3f}   |"
              f"   {rews.mean():.3f}")
    max_se = max(v for _, se in stay_results.values() for v in se.values())
    print(f"  (largest between-agent standard error of any cell: {max_se:.4f})")

    # ---------------- Part D
    vB_td, vA_td, vB_rw, vA_rw, b_len = part_d(n_phase2_d)
    jpk = int(np.argmax(vB_td))
    print(f"\nPart D -- second-order conditioning (Exercise 5.16): phase 1 A -> US ({N_PHASE1} trials), "
          f"phase 2 B -> A without US ({n_phase2_d} trials; B lasts {b_len} step(s) and ends when A starts)")
    print("  phase-2 trial |  TD: V(B onset)  V(A onset) |  RW: w_B    w_A")
    for k in (0, 1, 2, 5, 10, 20, 30, 50, n_phase2_d):
        print(f"  {k:13d} |      {vB_td[k]:7.4f}     {vA_td[k]:7.4f}   |  {vB_rw[k]:7.4f}  {vA_rw[k]:7.4f}")
    print(f"  TD: V(B onset) peaks at {vB_td[jpk]:.4f} after {jpk} phase-2 trials "
          f"(full first-order value of a CS {ISI + b_len} steps before the US: "
          f"gamma^{ISI + b_len - 1} = {GAMMA ** (ISI + b_len - 1):.4f})")
    print(f"  RW: max w_B over phase 2 = {vB_rw.max():.4f} (never positive)")

    if not args.quick:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(FIG_DIR, exist_ok=True)
        ink, ink2 = "#0b0b0b", "#52514e"
        c1, c2, c3 = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
        ax = axes[0]
        times = np.arange(1, T_TRIAL + 1)       # delta_t is the error on arrival at step t+1
        sel = [1, 10, 30, n_acq]
        blues = plt.cm.Blues(np.linspace(0.4, 0.95, len(sel)))
        for k, col in zip(sel, blues):
            ax.plot(times, deltas[k - 1], color=col, lw=2, label=f"trial {k}")
        ax.plot(times, probe, color=c2, lw=2, ls="--", label="trained, US omitted")
        ax.axvline(ONSET, color=ink2, lw=1, ls=":")
        ax.axvline(US_TIME, color=ink2, lw=1, ls=":")
        ax.text(ONSET, 1.05, " cue", color=ink2, va="bottom", fontsize=9)
        ax.text(US_TIME, 1.05, " US", color=ink2, va="bottom", fontsize=9)
        ax.axhline(0, color=ink2, lw=0.8)
        ax.set_ylim(-1.15, 1.25)
        ax.set_xticks(np.arange(0, T_TRIAL + 1, 2))
        ax.set_xlabel("time step within the trial (error on arrival at this step)")
        ax.set_ylabel("TD error δ")
        ax.set_title("(a) CSC-TD(0): δ moves from the US to the cue", color=ink)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.25)

        ax = axes[1]
        labels = ["RW\nblocking", "RW\ncontrol", "TD\nblocking", "TD\ncontrol"]
        vals = [res[("RW", "blocking")][1], res[("RW", "control")][1],
                res[("TD", "blocking")][1] / full, res[("TD", "control")][1] / full]
        bars = ax.bar(labels, vals, color=[c1, c1, c3, c3], width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", color=ink, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("strength of X (fraction of a fully trained CS)")
        ax.set_title("(b) Blocking: pretraining A leaves X unlearned", color=ink)
        ax.grid(alpha=0.25, axis="y")

        ax = axes[2]
        ks = np.arange(n_phase2_d + 1)
        ax.plot(ks, vB_td, color=c3, lw=2, label="TD: V(B onset)")
        ax.plot(ks, vA_td, color=c3, lw=1.5, ls="--", label="TD: V(A onset)")
        ax.plot(ks, vB_rw, color=c1, lw=2, label="RW: $w_B$")
        ax.plot(ks, vA_rw, color=c1, lw=1.5, ls="--", label="RW: $w_A$")
        ax.axhline(0, color=ink2, lw=0.8)
        ax.set_xlabel("phase-2 trials (B → A, no US)")
        ax.set_ylabel("prediction / associative strength")
        ax.set_title("(c) Second-order conditioning", color=ink)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "conditioning_td.png"), dpi=110)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
        for ax, (name, (tab, se)) in zip(axes, stay_results.items()):
            xs = np.array([0, 1])
            width = 0.36
            common_vals = [tab[(1, True)], tab[(0, True)]]
            rare_vals = [tab[(1, False)], tab[(0, False)]]
            ax.bar(xs - width / 2 - 0.01, common_vals, width, color=c1, label="common")
            ax.bar(xs + width / 2 + 0.01, rare_vals, width, color=c2, label="rare")
            ax.set_xticks(xs)
            ax.set_xticklabels(["rewarded", "unrewarded"])
            ax.set_ylim(0.0, 1.0)
            ax.set_title(name, color=ink)
            ax.grid(alpha=0.25, axis="y")
        axes[0].set_ylabel("P(repeat stage-1 choice)")
        axes[0].legend(title="previous transition", fontsize=8, title_fontsize=8)
        fig.suptitle("Two-step task: stay probabilities by previous outcome", color=ink)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "two_step_task.png"), dpi=110)
        plt.close(fig)
        print(f"\nFigures written to {FIG_DIR}")

    print(f"\nRuntime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
