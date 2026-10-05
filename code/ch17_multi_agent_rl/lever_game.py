"""Zero-shot coordination in the lever game (Chapter 17, section 9.6).

Two players each pull one of 10 levers. If they pull the same lever, both receive that
lever's payoff, otherwise both receive 0. Levers 0-8 pay 1.0 and lever 9 pays 0.9
(Hu, Lerer, Peysakhovich & Foerster, 2020). Nothing distinguishes the nine 1.0 levers
from each other: any relabelling phi of levers 0-8 (9! = 362,880 of them, lever 9 fixed)
leaves the game unchanged. These relabellings are the game's symmetries.

The zero-shot coordination protocol: train many agents independently (one per seed),
then pair seat 1 of seed s with seat 2 of seed s'. The cross-play matrix
    X[s, s'] = J(pi^1_s, pi^2_s') = sum_k r_k pi^1_s(k) pi^2_s'(k)
has the self-play scores on its diagonal and the cross-play scores off it.

Two learners, each trained with two objectives:
  * learners: an independent REINFORCE pair (two softmax policies, Chapter 10, as in
    lewis_signaling.py) and a central learner (stateless Q-learning over the 100 joint
    actions with the step size alpha = 0.1 of section 6.6's "central" learner, but slower-
    decaying exploration, eps_t = max(0.01, 0.999^t) instead of 0.998^t, because it has 100
    joint actions to explore rather than 9);
  * objectives: self-play (SP), max J(pi^1, pi^2), and other-play (OP),
    max E_phi[J(pi^1, phi(pi^2))]: in every training game a fresh, uniformly random
    relabelling phi is applied to seat 2's lever, so no convention that relies on the
    labels of the 1.0 levers can be learned.

The script also checks the OP value of every pure choice exactly, by enumerating all 9!
relabellings (1/9 for each 1.0 lever, 0.9 for lever 9; Exercise 17.16), and prints the
OP value of each lever against a uniformly random partner, which explains why gradient
learners started at the uniform policy drift away from lever 9.

Run:  python code/ch17_multi_agent_rl/lever_game.py [--quick]
"""
import argparse
import itertools
import os
import time

import numpy as np

K = 10                                       # levers
ODD = 9                                      # the 0.9 lever
R = np.array([1.0] * 9 + [0.9])              # payoff of a matched pull


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def sample(p, rng):
    """One categorical sample per row of p."""
    return (rng.random((p.shape[0], 1)) > np.cumsum(p, 1)).sum(1).clip(max=p.shape[1] - 1)


def random_symmetries(runs, rng):
    """One uniformly random relabelling per run: a permutation of levers 0-8, lever 9 fixed.
    Returned as perm[run, lever] = phi(lever)."""
    perm = np.argsort(rng.random((runs, 9)), axis=1)
    return np.concatenate([perm, np.full((runs, 1), ODD)], axis=1)


def all_symmetries():
    """All 9! = 362,880 relabellings, as an array [9!, 10]."""
    perms = np.array(list(itertools.permutations(range(9))), dtype=np.int8)
    return np.concatenate([perms, np.full((len(perms), 1), ODD, dtype=np.int8)], axis=1)


def payoff(p1, p2):
    """Expected common payoff J(pi^1, pi^2) for policies given as lever probabilities."""
    return (p1 * R * p2).sum(-1)


def op_payoff(p1, p2):
    """Other-play value E_phi J(pi^1, phi(pi^2)) in closed form: a uniformly relabelled
    partner puts mass q2/9 on every 1.0 lever, where q2 is its total mass on levers 0-8."""
    q1, q2 = p1[..., :9].sum(-1), p2[..., :9].sum(-1)
    return q1 * q2 / 9 + R[ODD] * p1[..., ODD] * p2[..., ODD]


def train_reinforce(runs, steps, rng, op, lr=0.5, baseline_lr=0.01, log_every=50):
    """Independent REINFORCE pair: two softmax policies from uniform logits, one shared
    running-average baseline (the reward is common). Returns final policies and a curve
    (step, fraction of runs whose seat-1 greedy lever is 9, mean seat-1 probability of 9)."""
    th = np.zeros((2, runs, K))
    b = np.zeros(runs)
    idx = np.arange(runs)
    curve = []
    for t in range(steps):
        p1, p2 = softmax(th[0]), softmax(th[1])
        if t % log_every == 0:
            curve.append((t, np.mean(p1.argmax(1) == ODD), p1[:, ODD].mean()))
        a1, a2 = sample(p1, rng), sample(p2, rng)
        # Other-play: seat 2 chooses a2, but the lever actually pulled is phi(a2).
        played2 = random_symmetries(runs, rng)[idx, a2] if op else a2
        r = np.where(a1 == played2, R[a1], 0.0)
        adv = r - b
        b += baseline_lr * (r - b)
        g1 = -p1                              # grad of log softmax = onehot(chosen) - probs
        g1[idx, a1] += 1.0
        g2 = -p2
        g2[idx, a2] += 1.0                    # seat 2's own choice, before relabelling
        th[0] += lr * adv[:, None] * g1
        th[1] += lr * adv[:, None] * g2
    p1, p2 = softmax(th[0]), softmax(th[1])
    curve.append((steps, np.mean(p1.argmax(1) == ODD), p1[:, ODD].mean()))
    return p1, p2, curve


def train_central(runs, steps, rng, op, alpha=0.1, eps_decay=0.999, eps_min=0.01, log_every=50):
    """Central learner: stateless Q-learning over the K*K joint actions (a 100-armed bandit),
    epsilon-greedy with eps_t = max(eps_min, eps_decay^t). Ties in the greedy choice are
    broken by a fixed random priority per run (so the seed, not the lever index, decides).
    Returns one-hot final policies for seat 1 and seat 2 and a curve as in train_reinforce."""
    Q = np.zeros((runs, K * K))
    tie = 1e-9 * rng.random((runs, K * K))
    idx = np.arange(runs)
    curve = []
    for t in range(steps):
        greedy = np.argmax(Q + tie, 1)
        if t % log_every == 0:
            curve.append((t, np.mean(greedy // K == ODD), np.mean(greedy // K == ODD)))
        eps = max(eps_min, eps_decay ** t)
        explore = rng.random(runs) < eps
        j = np.where(explore, rng.integers(K * K, size=runs), greedy)
        a1, a2 = j // K, j % K
        played2 = random_symmetries(runs, rng)[idx, a2] if op else a2
        r = np.where(a1 == played2, R[a1], 0.0)
        Q[idx, j] += alpha * (r - Q[idx, j])
    greedy = np.argmax(Q + tie, 1)
    curve.append((steps, np.mean(greedy // K == ODD), np.mean(greedy // K == ODD)))
    p1 = np.eye(K)[greedy // K]
    p2 = np.eye(K)[greedy % K]
    return p1, p2, curve


def cross_play(p1, p2):
    """X[s, s'] = J(seat 1 of run s, seat 2 of run s')."""
    return (p1 * R) @ p2.T


def summarise(name, p1, p2):
    X = cross_play(p1, p2)
    n = len(X)
    off = ~np.eye(n, dtype=bool)
    lever1 = p1.argmax(1)
    on_odd = np.mean(lever1 == ODD)
    distinct = len(np.unique(lever1))
    conf = p1.max(1).mean()
    opv = op_payoff(p1, p2).mean()
    print(f"{name:26s} {np.diag(X).mean():9.3f} {X[off].mean():10.3f} {opv:9.3f} "
          f"{on_odd:13.3f} {distinct:9d} {conf:10.3f}")
    return dict(X=X, sp=np.diag(X).mean(), xp=X[off].mean(), op=opv, odd=on_odd, lever=lever1)


def main():
    parser = argparse.ArgumentParser(description="Self-play vs other-play in the lever game")
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    runs = 50 if args.quick else 1000
    steps = 2_000 if args.quick else 5_000
    print(f"lever_game.py | seed={args.seed} | runs (independently trained pairs) per setting={runs} | "
          f"steps={steps} | REINFORCE lr=0.5, baseline lr=0.01 | central Q: alpha=0.1, "
          f"eps_t=max(0.01, 0.999^t) | quick={args.quick}")
    t0 = time.time()

    # --- exact other-play values of the pure choices, by enumerating all 9! relabellings
    perms = all_symmetries()
    exact = [np.mean(R[k] * (perms[:, k] == k)) for k in range(K)]
    print(f"\nenumerated {len(perms):,} relabellings; OP value E_phi J(k, phi(k)) of pulling lever k:")
    print("  " + ", ".join(f"{k}: {v:.4f}" for k, v in enumerate(exact)) + f"   (1/9 = {1 / 9:.4f})")
    u = np.full(K, 1 / K)
    vals = op_payoff(np.eye(K), u[None, :].repeat(K, 0))
    print(f"OP value of each lever against a uniform partner: 1.0 levers {vals[0]:.3f}, "
          f"lever 9 {vals[ODD]:.3f}")
    qstar = R[ODD] / (R[ODD] + 1 / 9)
    print(f"a 1.0 lever beats lever 9 under OP whenever the partner's mass q on the 1.0 levers "
          f"exceeds q* = {qstar:.3f}; the uniform partner has q = 0.9")

    # --- train SP and OP with both learners
    print(f"\n{'learner, objective':26s} {'SP score':>9s} {'XP score':>10s} {'OP value':>9s} "
          f"{'on lever 9':>13s} {'#levers':>9s} {'max prob':>10s}")
    print(f"{'':26s} {'(diagonal)':>9s} {'(off-diag)':>10s} {'':>9s} {'(fraction)':>13s} "
          f"{'(seat 1)':>9s} {'(seat 1)':>10s}")
    res = {}
    for k, (lname, fn) in enumerate([("REINFORCE pair", train_reinforce), ("central Q", train_central)]):
        for m, obj in enumerate(["SP", "OP"]):
            rng = np.random.default_rng(args.seed + 10 * k + m)
            p1, p2, curve = fn(runs, steps, rng, op=(obj == "OP"))
            res[(lname, obj)] = summarise(f"{lname}, {obj}", p1, p2)
            res[(lname, obj)]["curve"] = curve
            res[(lname, obj)]["p1"] = p1
    r_op = res[("REINFORCE pair", "OP")]
    print(f"\nREINFORCE pair, OP: seat-1 probability of lever 9 at the end: "
          f"median {np.median(r_op['p1'][:, ODD]):.3f}, runs with > 0.5: {np.mean(r_op['p1'][:, ODD] > 0.5):.3f}")
    miss = r_op["lever"] != ODD
    print(f"REINFORCE pair, OP: self-play score of the runs not on lever 9 (own training partner): "
          f"{np.diag(r_op['X'])[miss].mean():.3f}")
    for lname in ["REINFORCE pair", "central Q"]:
        lv = res[(lname, "SP")]["lever"]
        counts = np.bincount(lv, minlength=K)
        print(f"{lname}, SP: runs per seat-1 lever 0..9: {counts.tolist()}")
    print(f"\ntotal time {time.time() - t0:.1f} s")

    if not args.quick:
        import plot_style
        plt = plot_style.setup()
        figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
        os.makedirs(figdir, exist_ok=True)
        show = 30
        fig = plt.figure(figsize=(17.5, 4.4))
        gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 1, 0.05, 0.32, 1.35], wspace=0.12)
        panels = [("REINFORCE pair", "SP"), ("central Q", "SP"), ("REINFORCE pair", "OP"), ("central Q", "OP")]
        for i, key in enumerate(panels):
            ax = fig.add_subplot(gs[0, i])
            d = res[key]
            sel = np.arange(show)
            order = sel[np.argsort(d["lever"][sel], kind="stable")]   # group runs by convention
            X = d["X"][np.ix_(order, order)]
            im = ax.imshow(X, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
            ax.grid(False)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel("seat 2 from run s'")
            if i == 0:
                ax.set_ylabel("seat 1 from run s")
            ax.set_title(f"({'abcd'[i]}) {key[0]}, {key[1]}\nSP {d['sp']:.2f}, XP {d['xp']:.2f}")
        cax = fig.add_subplot(gs[0, 4])
        fig.colorbar(im, cax=cax, label="expected payoff J")
        ax = fig.add_subplot(gs[0, 6])
        for c, (key, ls) in enumerate([(("REINFORCE pair", "SP"), ":"), (("central Q", "SP"), "-."),
                                        (("REINFORCE pair", "OP"), "-"), (("central Q", "OP"), "--")]):
            t, frac, _ = zip(*res[key]["curve"])
            ax.plot(t, frac, color=plot_style.C[c], ls=ls, label=f"{key[0]}, {key[1]}")
        ax.set_xlabel("training game")
        ax.set_ylabel("fraction of runs on the 0.9 lever")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("(e) Which runs settle on the 0.9 lever")
        ax.legend(loc="center right")
        plot_style.kfmt(ax)
        fig.subplots_adjust(left=0.025, right=0.99, bottom=0.13, top=0.84)
        fig.savefig(os.path.join(figdir, "lever_game.png"))
        plt.close(fig)
        print(f"figure written to {figdir}/lever_game.png")


if __name__ == "__main__":
    main()
