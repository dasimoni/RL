"""Numerical checks and coding solutions for the Chapter 18 exercises.

Each function prints the numbers quoted in the corresponding solution.  Exercises 11-13
re-use the toy RLHF pipeline of rlhf_toy.py.  Exercise 14 (expert iteration) is a Monte Carlo
check on a softmax policy; Exercise 15 is a derivation whose numbers come from
multiturn_tool_toy.py.

Run from the repository root:
    python code/ch18_rl_for_language_models/exercise_solutions.py           # ~2.5 min (one core, shared machine)
    python code/ch18_rl_for_language_models/exercise_solutions.py --quick   # smoke test
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import itertools  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import integrate  # noqa: E402
from scipy.special import expit, softmax  # noqa: E402

torch.set_num_threads(1)
SEED = 0


def ex1_sequence_kl():
    """Chain rule: KL over whole sequences = E_pi[ sum_t KL(pi(.|s_t) || pi_ref(.|s_t)) ]."""
    rng = np.random.default_rng(SEED)
    V, L = 3, 3  # vocabulary of 3 tokens, responses of exactly 3 tokens (no EOS, for simplicity)
    # random next-token distributions for every prefix
    pi, ref = {}, {}
    for t in range(L):
        for prefix in itertools.product(range(V), repeat=t):
            pi[prefix] = softmax(rng.normal(size=V))
            ref[prefix] = softmax(rng.normal(size=V))
    kl_seq, kl_tok = 0.0, 0.0
    probs, sampled, exact = [], [], []
    for y in itertools.product(range(V), repeat=L):
        p = np.prod([pi[y[:t]][y[t]] for t in range(L)])
        q = np.prod([ref[y[:t]][y[t]] for t in range(L)])
        kl_seq += p * math.log(p / q)
        # sum of per-position exact KLs along y, weighted by the probability of y
        s_exact = sum(np.sum(pi[y[:t]] * np.log(pi[y[:t]] / ref[y[:t]])) for t in range(L))
        kl_tok += p * s_exact
        probs.append(p)
        sampled.append(math.log(p / q))  # the k1 estimate from this one response
        exact.append(s_exact)
    print(f"[Ex 1] sequence-level KL = {kl_seq:.6f};  E_pi[sum_t KL_t] = {kl_tok:.6f}")
    pr, sa, ex = np.array(probs), np.array(sampled), np.array(exact)
    var = lambda v: float(pr @ (v - pr @ v) ** 2)  # noqa: E731
    print(f"[Ex 1] variance over y ~ pi of one response's estimate: sampled log-ratios {var(sa):.4f},"
          f" exact per-token KLs {var(ex):.4f};  P(sampled estimate < 0) = {pr[sa < 0].sum():.3f}")
    # A counterexample: the exact-KL sum need not have the smaller variance.  Step 1: pi = (0.5, 0.5),
    # pi_ref = (0.25, 0.75), log-ratio l1(a).  Step 2: pi emits one token deterministically whose
    # pi_ref-probability is exp(-(1 - l1(a))), so its log-ratio is 1 - l1(a) and its exact KL is too.
    p1, q1 = np.array([0.5, 0.5]), np.array([0.25, 0.75])
    l1 = np.log(p1 / q1)
    kl1 = float(p1 @ l1)
    s_samp = l1 + (1.0 - l1)       # identically 1
    s_exact = kl1 + (1.0 - l1)     # KL_1 is a constant; KL_2 = log-ratio of the forced token
    v = lambda w: float(p1 @ (w - p1 @ w) ** 2)  # noqa: E731
    print(f"[Ex 1] counterexample: means {p1 @ s_samp:.3f} / {p1 @ s_exact:.3f};  variance of the sampled sum "
          f"{v(s_samp):.3f}, of the exact-KL sum {v(s_exact):.3f}")


def ex3_gumbel(n=2_000_000):
    rng = np.random.default_rng(SEED)
    for r1, r2 in [(1.0, 0.0), (0.3, 0.5), (2.0, -1.0)]:
        u1 = r1 + rng.gumbel(size=n)
        u2 = r2 + rng.gumbel(size=n)
        print(f"[Ex 3] r1={r1:+.1f} r2={r2:+.1f}: P(U1>U2) Monte Carlo {np.mean(u1 > u2):.4f}"
              f"   sigma(r1-r2) {expit(r1 - r2):.4f}")


def ex4_closed_form():
    pi_ref = np.array([0.5, 0.3, 0.2])
    r = np.array([0.0, 1.0, 2.0])
    for beta in [1.0, 0.5]:
        w = pi_ref * np.exp(r / beta)
        Z = w.sum()
        pi = w / Z
        kl = np.sum(pi * np.log(pi / pi_ref))
        print(f"[Ex 4] beta={beta}: unnormalised {np.round(w, 4)}  Z={Z:.4f}  pi*={np.round(pi, 4)}"
              f"  E[r]={pi @ r:.4f}  KL={kl:.4f}  E[r]-beta KL={pi @ r - beta * kl:.4f}"
              f"  beta log Z={beta * math.log(Z):.4f}")


def ex5_dpo_gradient():
    beta = 0.1
    for hw, hl in [(0.0, 0.0), (-2.0, 3.0), (10.0, -10.0)]:
        weight = expit(beta * (hl - hw))
        loss = -math.log(expit(beta * (hw - hl)))
        print(f"[Ex 5] h_w={hw:+.1f} h_l={hl:+.1f}: loss={loss:.4f}  gradient weight "
              f"sigma(beta (h_l - h_w))={weight:.4f}  (d loss/d h_w = {-beta * weight:+.4f})")


def ex6_best_of_n():
    """Continuous reward distribution: the best of n has density n F^{n-1} f, so its KL to the
    base is E[log(n F^{n-1})] = log n + (n-1) E[log U_max] with U_max ~ Beta(n, 1)."""
    for n in [2, 4, 16, 64]:
        val, _ = integrate.quad(lambda u: n * u ** (n - 1) * (math.log(n) + (n - 1) * math.log(u)), 0, 1)
        print(f"[Ex 6] n={n:<3} KL by numerical integration {val:.5f}   log n - (n-1)/n = "
              f"{math.log(n) - (n - 1) / n:.5f}")


def ex7_degenerate_groups():
    for G in [4, 8, 16]:
        row = []
        for p in [0.1, 0.5, 0.9, 0.99]:
            q = p**G + (1 - p) ** G
            row.append(f"p={p}: {q:.3f}")
        print(f"[Ex 7] G={G:<2}  P(group has no signal) " + "  ".join(row))
    p, G = 0.9, 8
    q = p**G + (1 - p) ** G
    print(f"[Ex 7] p=0.9, G=8: useful fraction {1 - q:.3f}; to fill 32 useful groups dynamic sampling "
          f"needs on average {32 / (1 - q):.1f} groups = {32 * G / (1 - q):.0f} responses")


def ex8_length_bias():
    """Per-token weight of a response's tokens in GRPO's loss: A_i / |o_i|."""
    for A in [+1.0, -1.0]:
        for L in [10, 100, 1000]:
            print(f"[Ex 8] advantage {A:+.0f}, length {L:>4}: per-token weight {A / L:+.4f};"
                  f"  total push on log pi(o|q) summed over tokens {A:+.1f}")


def ex9_rloo_and_std():
    rng = np.random.default_rng(SEED)
    G = 8
    r = rng.integers(0, 2, G).astype(float)
    loo = r - (r.sum() - r) / (G - 1)
    print(f"[Ex 9] rewards {r}:  RLOO advantages / (r - mean) = "
          f"{np.round(loo / np.where(r - r.mean() == 0, np.nan, r - r.mean()), 6)} (G/(G-1) = {G / (G - 1):.6f})")
    for p in [0.05, 0.2, 0.5, 0.8, 0.95]:
        sd = math.sqrt(p * (1 - p))
        print(f"[Ex 9] success prob {p:.2f}: group std ~ {sd:.3f}; GRPO weights this prompt by 1/std = "
              f"{1 / sd:.2f} relative to Dr. GRPO")


def ex10_k3_gradient():
    """The expected gradient of the k3 estimator, used as a loss, is grad KL(pi_ref || pi_theta)."""
    torch.manual_seed(SEED)
    V = 6
    theta = torch.randn(V, requires_grad=True)
    ref = torch.softmax(torch.randn(V), 0)
    p = torch.softmax(theta, 0)
    logp = torch.log_softmax(theta, 0)
    # E_{a ~ p}[ grad k3(a) ], k3 = q/p - log(q/p) - 1; differentiate through p only (a is fixed)
    k3 = ref / p - torch.log(ref / p) - 1.0
    exp_grad_k3 = torch.autograd.grad((p.detach() * k3).sum(), theta, retain_graph=True)[0]
    fwd = torch.autograd.grad((ref * (torch.log(ref) - logp)).sum(), theta, retain_graph=True)[0]
    rev = torch.autograd.grad((p * (logp - torch.log(ref))).sum(), theta)[0]
    print(f"[Ex 10] E[grad k3]        = {np.round(exp_grad_k3.numpy(), 4)}")
    print(f"[Ex 10] grad KL(ref||pi)  = {np.round(fwd.numpy(), 4)}")
    print(f"[Ex 10] grad KL(pi||ref)  = {np.round(rev.numpy(), 4)}")


def ex14_expert_iteration(n_groups=400_000):
    """Expected gradient of filtered SFT on a softmax 'policy' over K responses, binary reward.

    grad log pi(y) = e_y - pi for softmax logits.  Per-prompt average of the kept samples:
    E[g_bar] = (1 - (1-p)^k) grad p / p.  Sum over kept samples: E[g_sum] = k grad p."""
    rng = np.random.default_rng(SEED + 14)
    K, k = 6, 8
    for target_p in [0.05, 0.5, 0.95]:
        logits = rng.normal(size=K)
        r = np.zeros(K)
        r[:2] = 1.0  # responses 0 and 1 are correct; shift their logits to hit the target p
        q = softmax(logits)
        shift = np.log(target_p / (1 - target_p)) - np.log(q[:2].sum() / q[2:].sum())
        pi = softmax(logits + shift * r)
        p = pi @ r
        grad_p = (pi * r) @ (np.eye(K) - pi)          # sum_y pi(y) r(y) (e_y - pi)
        ys = rng.choice(K, size=(n_groups, k), p=pi)
        rs = r[ys]
        c = rs.sum(1)
        onehot = np.zeros((n_groups, K))
        np.add.at(onehot, (np.repeat(np.arange(n_groups), k), ys.ravel()), rs.ravel())
        # sum over kept samples of (e_y - pi), then the per-prompt average (0 if nothing kept)
        g_sum = onehot - c[:, None] * pi
        g_bar = g_sum / np.maximum(c, 1)[:, None]
        w_bar = (1 - (1 - p) ** k) / p
        err_bar = np.abs(g_bar.mean(0) - w_bar * grad_p).max() / np.abs(grad_p).max()
        err_sum = np.abs(g_sum.mean(0) - k * grad_p).max() / np.abs(grad_p).max()
        print(f"[Ex 14] p={p:.2f}, k={k}: weight on grad p  per-prompt average {w_bar:.3f} "
              f"(exact EM 1/p = {1 / p:.3f}), sum / k 1.000, GRPO 1/sqrt(p(1-p)) {1 / math.sqrt(p * (1 - p)):.3f};"
              f"  Monte Carlo ({n_groups} groups) relative error: average {err_bar:.4f}, sum {err_sum:.4f}")


# ---------------------------------------------------------------------------------------
# Coding exercises on the toy RLHF pipeline
# ---------------------------------------------------------------------------------------
def coding_exercises(quick: bool):
    import rlhf_toy as R
    t0 = time.time()
    R.seed_all(SEED)
    g = np.random.default_rng(SEED)
    task = R.Task(seed=SEED + 100)
    cfg = dict(batch=128 if quick else 256, minibatch=64, epochs=4, clip=0.2, lam=0.95, lr=3e-4, lr_v=1e-3,
               n_eval=256 if quick else 1024, lr_dpo=1e-3, dpo_batch=64, simpo_gamma=0.5)
    iters = 10 if quick else 120
    ref = R.train_sft(task, 300 if quick else 1500, 128, g)
    for p in ref.parameters():
        p.requires_grad_(False)
    val = R.make_preferences(task, ref, 300 if quick else 1000, g)
    prefs = R.make_preferences(task, ref, 600 if quick else 2000, g)
    rm, _ = R.train_reward_model(task, prefs, val, 3 if quick else 12, g)
    rmn = R.NormalizedRM(rm, task, ref, g)
    print(f"[setup] SFT + RM done ({time.time() - t0:.1f}s); RM corr with true reward {rmn.corr:.3f}")

    def show(tag, f):
        print(f"  {tag:<38} true {f['true']:+.3f}  RM {f['rm']:+.3f}  KL {f['kl']:5.2f}  repeats {f['rep']:.2f}")

    # Exercise 11: adaptive KL controller
    print("[Ex 11] PPO with a fixed beta vs an adaptive beta targeting KL = 4 nats")
    R.seed_all(SEED + 1)
    _, c_fixed = R.run_ppo(task, ref, rmn, rm, 0.01, iters, g, cfg, eval_every=iters // 2 if quick else 20)
    R.seed_all(SEED + 1)
    _, c_adapt = R.run_ppo(task, ref, rmn, rm, 0.01, iters, g, cfg, eval_every=iters // 2 if quick else 20,
                           kl_target=4.0)
    show("fixed beta = 0.01", c_fixed[-1])
    show(f"adaptive (start 0.01, final beta {c_adapt[-1]['beta']:.3f})", c_adapt[-1])
    print("  adaptive run, (iteration, beta, KL, true): "
          + ", ".join(f"({e['it']}, {e['beta']:.3f}, {e['kl']:.1f}, {e['true']:+.2f})" for e in c_adapt))

    # Exercise 12: SimPO vs DPO
    print("[Ex 12] DPO (beta=0.1) vs SimPO (beta=2, gamma=0.5) on the same pairs, 20 epochs")
    ep = 2 if quick else 20
    ref_len = R.evaluate_policy(task, ref, ref, rmn, g, n=cfg["n_eval"])["length"]
    for lt, b in [("dpo", 0.1), ("simpo", 2.0)]:
        R.seed_all(SEED + 3)
        pol, cv = R.run_dpo(task, ref, rmn, prefs, val, b, ep, g, cfg, eval_every=ep, loss_type=lt)
        ev = R.evaluate_policy(task, pol, ref, rmn, g, n=cfg["n_eval"])
        show(f"{lt.upper()} (beta={b})", ev)
        print(f"  {'':<38} mean length {ev['length']:.2f} words (pi_ref {ref_len:.2f});"
              f"  change in log pi(chosen) {cv[-1]['dlogp_w']:+.2f}, (rejected) {cv[-1]['dlogp_l']:+.2f}")

    # Exercise 13: iterated RLHF -- label the over-optimized policy's own samples and retrain the RM
    print("[Ex 13] Iterated RLHF: round 1 RM from pi_ref pairs; round 2 adds pairs sampled from the round-1 policy")
    R.seed_all(SEED + 4)
    pol1, c1 = R.run_ppo(task, ref, rmn, rm, 0.01, iters, g, cfg, eval_every=iters)
    show("round 1: PPO beta=0.01, RM-1", c1[-1])
    new = R.make_preferences(task, pol1, 600 if quick else 2000, g)
    both = {k: (torch.cat([prefs[k], new[k]]) if torch.is_tensor(prefs[k]) else
                (np.concatenate([prefs[k], new[k]]) if k != "bayes_acc" else prefs[k])) for k in prefs}
    rm2, _ = R.train_reward_model(task, both, val, 3 if quick else 12, g)
    rmn2 = R.NormalizedRM(rm2, task, ref, g)
    # the repeat probe (deterministic: draws no random numbers) for RM-1 and RM-2, all prompts
    for (x, z1, ts), (_, z2, _) in zip(R.probe_repeats(task, rmn), R.probe_repeats(task, rmn2)):
        print(f"  prompt {x}, best word repeated 1,2,3,5,9 times:  RM-1 " + " ".join(f"{z:+.2f}" for z in z1)
              + "   RM-2 " + " ".join(f"{z:+.2f}" for z in z2) + "   r* " + " ".join(f"{t:+.2f}" for t in ts))
    R.seed_all(SEED + 4)
    _, c2 = R.run_ppo(task, ref, rmn2, rm2, 0.01, iters, g, cfg, eval_every=iters)
    show("round 2: PPO beta=0.01, RM-2", c2[-1])
    print(f"  [{time.time() - t0:.1f}s]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    print(f"seed={SEED} quick={args.quick}")
    ex1_sequence_kl()
    ex3_gumbel(200_000 if args.quick else 2_000_000)
    ex4_closed_form()
    ex5_dpo_gradient()
    ex6_best_of_n()
    ex7_degenerate_groups()
    ex8_length_bias()
    ex9_rloo_and_std()
    ex10_k3_gradient()
    ex14_expert_iteration(40_000 if args.quick else 400_000)
    coding_exercises(args.quick)
    print(f"Total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
