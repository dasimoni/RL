"""RL with verifiable rewards (RLVR) on a toy "reasoning" task: GRPO and its relatives.

Chapter 18, Sections 9-12.  The task: given n digits d_1..d_n (n = 2..6), write the running
sums modulo 10, one per step, then EOS.  The last number written is the answer:

      prompt 3 9 4 7      ->   response  3 2 6 3 EOS      (3, 3+9=12->2, 2+4=6, 6+7=13->3)

The "chain of thought" makes each step an easy two-number addition; the outcome is checked by
a program, like a maths answer checker or a unit test:

      outcome reward  r(x, y) = 1  if y ends with EOS and its last digit = (d_1+...+d_n) mod 10
                                0  otherwise.

The base model is trained by maximum likelihood on *noisy* demonstrations with two kinds of
error (see demos()): a systematic "carry bug" (when a running sum exceeds 9 the demonstrator
writes the units digit plus one with probability P_WRAP = 0.6) and random +-1 slips (probability
P_SLIP = 0.05 on the other steps).  Errors propagate to later steps, like an arithmetic error in
a derivation.  The base model can produce correct solutions but is unreliable, its most likely
digit at a carry step is the wrong one, and longer problems are harder -- the situation of a
pretrained LLM before RLVR.

What runs (all variants share one update routine; only the knobs differ, Algorithm 18.7):
  REINFORCE   advantage = r                       (no baseline), no clip, no KL
  RLOO        advantage = r_i - mean_{j!=i} r_j   (leave-one-out baseline), no clip, no KL
              (both take several minibatch steps per batch with an unclipped per-token ratio, so
              after the first step they optimise a biased off-policy surrogate, not REINFORCE proper)
  GRPO        advantage = (r_i - mean) / std      per-response 1/|o_i| averaging, clip 0.2, KL 0.04 (k3)
  Dr. GRPO    advantage = r_i - mean              token sum / constant, clip 0.2, no KL (eq. 18.28)
  DAPO-lite   advantage = (r_i - mean) / std      token-level averaging, clip-higher (0.2, 0.28),
                                                  dynamic sampling (drop all-0 / all-1 groups), no KL
  GRPO+PRM    process rewards: each step is checked locally (is s_k = s_{k-1} + d_k mod 10?);
              advantages are DeepSeekMath's process-supervision advantages (Section 11)
  Two ablations that separate the KL term from the normalisations:
  GRPO, beta=0          GRPO without its KL term (the fair partner of Dr. GRPO)
  Dr. GRPO, beta=0.04   Dr. GRPO with GRPO's KL term.  Without std normalisation its advantages
                        are 2-3x smaller, so the same beta regularises it relatively more.
Then: pass@k and majority-vote accuracy (test-time compute) for the base and trained models.

Run from the repository root:
    python code/ch18_rl_for_language_models/grpo_rlvr.py            # full run with figures (~6 min)
    python code/ch18_rl_for_language_models/grpo_rlvr.py --quick    # smoke test, no figures
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from toy_lm import PolicyLM, token_entropy, token_kl_exact, token_logprobs  # noqa: E402

torch.set_num_threads(1)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

N_MIN, N_MAX = 2, 6        # problem sizes (number of digits)
EOS = 10
VOCAB = 11                 # digits 0..9 and EOS
T_MAX = N_MAX + 2          # generation budget: room for one extra step and EOS
COND_DIM = N_MAX * 11      # one-hot digit (or "empty") per prompt position
P_SLIP = 0.05              # random +-1 slip rate of the demonstrations
P_WRAP = 0.6               # rate of the systematic 'carry bug' when a sum exceeds 9


# ---------------------------------------------------------------------------------------
# The task and its verifier
# ---------------------------------------------------------------------------------------
def make_prompts(n_items, g, sizes=None):
    sizes = g.integers(N_MIN, N_MAX + 1, n_items) if sizes is None else np.asarray(sizes)
    digits = np.full((n_items, N_MAX), -1, dtype=np.int64)
    for i, n in enumerate(sizes):
        digits[i, :n] = g.integers(0, 10, n)
    return digits


def cond_of(digits):
    d = torch.as_tensor(np.where(digits < 0, 10, digits))
    return F.one_hot(d, 11).float().view(len(digits), -1)


def demos(digits, g):
    """Noisy demonstrations of the running sums, with two kinds of error:

    * a systematic "carry bug": when s_{k-1} + d_k >= 10, the demonstrator writes the units
      digit plus one (7 + 5 = 12 -> writes 3) with probability P_WRAP;
    * random slips: otherwise, the written digit is off by +-1 with probability P_SLIP.
    Errors propagate: the next step adds to the digit that was actually written.
    """
    out = np.full((len(digits), T_MAX), EOS, dtype=np.int64)
    for i, row in enumerate(digits):
        s = 0
        for k, d in enumerate(row[row >= 0]):
            wraps = s + d >= 10
            s = (s + d) % 10
            if wraps and g.random() < P_WRAP:
                s = (s + 1) % 10
            elif g.random() < P_SLIP:
                s = (s + (1 if g.random() < 0.5 else -1)) % 10
            out[i, k] = s
    return torch.as_tensor(out)


def verify(digits, tokens, mask):
    """Outcome reward (1/0), plus diagnostics: local step validity and chain correctness.

    Returns
      reward[b]       1 if the response ends with EOS and its last digit is the right answer
      step_ok[b, t]   1 if step t is locally valid: s_t = s_{t-1} + d_t mod 10 (PRM label)
      chain_ok[b]     1 if *every* step is valid and the number of steps is n
    """
    tok, m = tokens.numpy(), mask.numpy() > 0
    B = len(digits)
    reward = np.zeros(B)
    chain_ok = np.zeros(B)
    step_ok = np.zeros((B, T_MAX))
    for b in range(B):
        ds = digits[b][digits[b] >= 0]
        n = len(ds)
        resp = [int(v) for v, ok in zip(tok[b], m[b]) if ok]
        ended = len(resp) > 0 and resp[-1] == EOS
        nums = resp[:-1] if ended else resp
        prev, all_ok = 0, True
        for t, v in enumerate(nums):
            ok = t < n and v == (prev + ds[t]) % 10
            step_ok[b, t] = float(ok)
            all_ok &= ok
            prev = v
        if ended:  # the EOS decision is a step too: correct iff it comes right after n numbers
            step_ok[b, len(nums)] = float(len(nums) == n)
        answer = int(ds.sum() % 10)
        reward[b] = float(ended and len(nums) > 0 and nums[-1] == answer)
        chain_ok[b] = float(ended and all_ok and len(nums) == n)
    return reward, step_ok, chain_ok


# ---------------------------------------------------------------------------------------
# Base model ("pre-training" on noisy demonstrations)
# ---------------------------------------------------------------------------------------
def train_base(steps, batch, g, hidden, log_every=0):
    model = PolicyLM(VOCAB, COND_DIM, emb=32, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        dg = make_prompts(batch, g)
        toks = demos(dg, g)
        is_eos = (toks == EOS).float()
        mask = 1.0 - (torch.cumsum(is_eos, 1) - is_eos).clamp(max=1.0)
        loss = -token_logprobs(model, cond_of(dg), toks, mask).sum() / batch
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log_every and (s % log_every == 0 or s == steps - 1):
            print(f"    base step {s:5d}  NLL/sequence {loss.item():.3f}")
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------------------
# Advantages
# ---------------------------------------------------------------------------------------
def outcome_advantages(r, G, kind):
    """r: (n_prompts*G,) outcome rewards, grouped consecutively.  Returns per-response advantage."""
    rg = r.view(-1, G)
    mean = rg.mean(1, keepdim=True)
    if kind == "none":       # REINFORCE without a baseline
        a = rg
    elif kind == "rloo":     # leave-one-out baseline (Kool et al. 2019; Ahmadian et al. 2024)
        a = rg - (rg.sum(1, keepdim=True) - rg) / (G - 1)
    elif kind == "mean":     # Dr. GRPO: centre only
        a = rg - mean
    elif kind == "grpo":     # GRPO: centre and divide by the group's std (eq. 18.25)
        a = (rg - mean) / (rg.std(1, keepdim=True, unbiased=False) + 1e-4)
    else:
        raise ValueError(kind)
    return a.reshape(-1)


def process_advantages(step_r, mask, G):
    """DeepSeekMath's process supervision: normalise every step reward by the mean/std over *all*
    steps of all responses in the group, then the advantage of token t is the sum of the
    normalised rewards of steps >= t (a reward-to-go, eq. 18.30)."""
    B, T = step_r.shape
    sr = step_r.view(-1, G, T)
    mk = mask.view(-1, G, T)
    cnt = mk.sum((1, 2), keepdim=True)
    mu = (sr * mk).sum((1, 2), keepdim=True) / cnt
    sd = torch.sqrt((((sr - mu) * mk) ** 2).sum((1, 2), keepdim=True) / cnt)
    z = (sr - mu) / (sd + 1e-4) * mk
    togo = torch.flip(torch.cumsum(torch.flip(z, [2]), 2), [2])
    return (togo * mk).view(B, T)


# ---------------------------------------------------------------------------------------
# One RL run: sample groups, score with the verifier, update with the chosen knobs
# ---------------------------------------------------------------------------------------
METHODS = {
    #            advantage   loss normalisation  clip(low, high)  KL beta  dynamic sampling
    "REINFORCE": dict(adv="none", norm="seq_sum", clip=None, beta=0.0, dyn=False),
    "RLOO":      dict(adv="rloo", norm="seq_sum", clip=None, beta=0.0, dyn=False),
    "GRPO":      dict(adv="grpo", norm="seq_mean", clip=(0.2, 0.2), beta=0.04, dyn=False),
    "GRPO, beta=0": dict(adv="grpo", norm="seq_mean", clip=(0.2, 0.2), beta=0.0, dyn=False),
    "Dr. GRPO":  dict(adv="mean", norm="const", clip=(0.2, 0.2), beta=0.0, dyn=False),
    "Dr. GRPO, beta=0.04": dict(adv="mean", norm="const", clip=(0.2, 0.2), beta=0.04, dyn=False),
    "DAPO-lite": dict(adv="grpo", norm="token", clip=(0.2, 0.28), beta=0.0, dyn=True),
    "GRPO+PRM":  dict(adv="process", norm="seq_mean", clip=(0.2, 0.2), beta=0.04, dyn=False),
}
ABLATIONS = {"GRPO, beta=0", "Dr. GRPO, beta=0.04"}  # drawn dashed in the figure


def aggregate(per_token, mask, norm, G):
    """How token losses are combined -- the knob behind the Dr. GRPO and DAPO critiques.
      seq_mean: (1/n_resp) sum_i (1/|o_i|) sum_t      (GRPO as published)
      token:    sum_{i,t} / sum_i |o_i|                 (DAPO: every token weighs the same)
      const:    sum_{i,t} / (n_resp * T_MAX)            (Dr. GRPO: constant normaliser)
      seq_sum:  (1/n_resp) sum_i sum_t                  (plain REINFORCE on log pi(y|x))
    """
    if norm == "seq_mean":
        return ((per_token * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()
    if norm == "token":
        return (per_token * mask).sum() / mask.sum().clamp(min=1)
    if norm == "const":
        return (per_token * mask).sum() / (mask.shape[0] * T_MAX)
    if norm == "seq_sum":
        return (per_token * mask).sum(1).mean()
    raise ValueError(norm)


@torch.no_grad()
def sample_groups(policy, n_prompts, G, g):
    """G responses for each of n_prompts fresh prompts, scored by the verifier."""
    dg = np.repeat(make_prompts(n_prompts, g), G, axis=0)
    c = cond_of(dg)
    y, m = policy.sample(c, T_MAX)
    r, step_ok, _ = verify(dg, y, m)
    return dict(dg=dg, c=c, y=y, m=m, r=r, step_ok=step_ok)


def take_groups(batch, groups, G):
    idx = np.concatenate([np.arange(i * G, (i + 1) * G) for i in groups])
    return {k: v[torch.as_tensor(idx)] if torch.is_tensor(v) else v[idx] for k, v in batch.items()}


def cat_batches(a, b):
    return {k: torch.cat([a[k], b[k]]) if torch.is_tensor(a[k]) else np.concatenate([a[k], b[k]])
            for k in a}


def mixed_groups(batch, G):
    rg = batch["r"].reshape(-1, G)
    return [i for i in range(len(rg)) if rg[i].min() != rg[i].max()]


def run_rl(name, base, iters, g, cfg, eval_set, log_every=0):
    knobs = METHODS[name]
    policy = copy.deepcopy(base)
    for p in policy.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])
    G, P = cfg["G"], cfg["prompts_per_batch"]
    curve, n_samples, degen_hist = [], 0, []
    for it in range(iters):
        # ---- rollouts: G responses per prompt, scored by the verifier ----
        batch = sample_groups(policy, P, G, g)
        n_samples += len(batch["r"])
        frac_degenerate = 1.0 - len(mixed_groups(batch, G)) / P  # all-correct or all-wrong groups
        degen_hist.append(frac_degenerate)
        if knobs["dyn"]:
            # DAPO's dynamic sampling: keep only groups whose rewards are not all equal (the
            # others have zero advantage and contribute no gradient), and sample more prompts
            # until the batch is full again (at most 4 extra rounds here).
            batch = take_groups(batch, mixed_groups(batch, G), G)
            for _ in range(4):
                if len(batch["r"]) >= P * G:
                    break
                extra = sample_groups(policy, P, G, g)
                n_samples += len(extra["r"])
                batch = cat_batches(batch, take_groups(extra, mixed_groups(extra, G), G))
            if len(batch["r"]) == 0:
                continue
            batch = take_groups(batch, range(min(P, len(batch["r"]) // G)), G)
        c, y, m, r, step_ok = batch["c"], batch["y"], batch["m"], batch["r"], batch["step_ok"]
        rt = torch.as_tensor(r, dtype=torch.float)
        with torch.no_grad():
            logp_old = token_logprobs(policy, c, y, m)
            if knobs["adv"] == "process":
                adv = process_advantages(torch.as_tensor(step_ok, dtype=torch.float), m, G)
            else:
                adv = outcome_advantages(rt, G, knobs["adv"]).unsqueeze(1).expand_as(m) * m
        # ---- policy update: cfg["epochs"] passes over the batch in cfg["minibatches"] chunks ----
        B = len(rt)
        n_groups = B // G
        for _ in range(cfg["epochs"]):
            gperm = g.permutation(n_groups)
            for chunk in np.array_split(gperm, cfg["minibatches"]):
                idx = torch.as_tensor(np.concatenate([np.arange(i * G, (i + 1) * G) for i in chunk]))
                ci, yi, mi, ai = c[idx], y[idx], m[idx], adv[idx]
                logp = token_logprobs(policy, ci, yi, mi)
                ratio = torch.exp(logp - logp_old[idx])
                if knobs["clip"] is None:
                    # unclipped policy gradient.  On the first minibatch step ratio = 1 and this is
                    # REINFORCE.  On later steps the per-token ratio drops the product of the
                    # prefix ratios, so it is a (biased) off-policy surrogate, not REINFORCE proper.
                    surr = ratio * ai
                else:
                    lo, hi = knobs["clip"]
                    surr = torch.min(ratio * ai, torch.clamp(ratio, 1 - lo, 1 + hi) * ai)
                per_tok = -surr
                if knobs["beta"] > 0:
                    # k3 estimator of KL(pi || pi_ref) per token, as in GRPO (eq. 18.27):
                    # pi_ref/pi - log(pi_ref/pi) - 1  >= 0.  It is unbiased when y ~ pi_theta;
                    # here y ~ pi_old, so it is exact only on the first minibatch step.
                    with torch.no_grad():
                        logp_ref = token_logprobs(base, ci, yi, mi)
                    log_rr = logp_ref - logp
                    per_tok = per_tok + knobs["beta"] * (torch.exp(log_rr) - log_rr - 1)
                loss = aggregate(per_tok, mi, knobs["norm"], G)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
        if it % cfg["eval_every"] == 0 or it == iters - 1:
            ev = evaluate(policy, base, eval_set, g)
            # fraction of no-signal groups, averaged over the iterations since the last evaluation
            ev.update(it=it, samples=n_samples, degenerate=float(np.mean(degen_hist[-cfg["eval_every"]:])))
            curve.append(ev)
            if log_every and (it % log_every == 0 or it == iters - 1):
                print(f"    {name:<19} it {it:4d}  acc {ev['acc']:.3f}  greedy {ev['greedy']:.3f}  "
                      f"chain-ok|correct {ev['chain_given_correct']:.3f}  KL {ev['kl']:.2f}  "
                      f"entropy {ev['entropy']:.3f}  degenerate groups {frac_degenerate:.2f}")
    return policy, curve


# ---------------------------------------------------------------------------------------
# Evaluation: accuracy by problem size, KL, entropy, pass@k, majority vote
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(policy, base, eval_set, g, n_samples=4):
    dg = np.repeat(eval_set, n_samples, axis=0)
    c = cond_of(dg)
    y, m = policy.sample(c, T_MAX)
    r, _, chain_ok = verify(dg, y, m)
    sizes = (dg >= 0).sum(1)
    yg, mg = policy.sample(cond_of(eval_set), T_MAX, greedy=True)
    rg, _, _ = verify(eval_set, yg, mg)
    kl = token_kl_exact(policy, base, c, y, m).sum(1).mean().item()
    ent = (token_entropy(policy, c, y, m).sum() / m.sum()).item()
    out = dict(acc=r.mean(), greedy=rg.mean(), kl=kl, entropy=ent,
               chain_given_correct=chain_ok[r == 1].mean() if (r == 1).any() else np.nan,
               length=m.sum(1).float().mean().item())
    for n in range(N_MIN, N_MAX + 1):
        out[f"acc_n{n}"] = r[sizes == n].mean()
    return out


def pass_at_k(n, c, k):
    """Unbiased pass@k estimator from n samples with c correct (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)  # eq. 18.31


@torch.no_grad()
def test_time_scaling(policy, eval_set, n_samples, ks, g):
    """pass@k and majority-vote accuracy (self-consistency) as functions of k."""
    P = len(eval_set)
    dg = np.repeat(eval_set, n_samples, axis=0)
    y, m = policy.sample(cond_of(dg), T_MAX)
    r, _, _ = verify(dg, y, m)
    r = r.reshape(P, n_samples)
    # final answers (or -1 for a malformed response)
    tok, mk = y.numpy(), m.numpy() > 0
    ans = np.full(len(dg), -1)
    for b in range(len(dg)):
        resp = [int(v) for v, ok in zip(tok[b], mk[b]) if ok]
        if len(resp) >= 2 and resp[-1] == EOS:
            ans[b] = resp[-2]
    ans = ans.reshape(P, n_samples)
    truth = np.array([row[row >= 0].sum() % 10 for row in eval_set])
    passk = {k: np.mean([pass_at_k(n_samples, int(r[i].sum()), k) for i in range(P)]) for k in ks}
    maj = {}
    for k in ks:
        accs = []
        for rep in range(max(1, n_samples // k)):  # disjoint blocks of k samples
            block = ans[:, rep * k:(rep + 1) * k]
            for i in range(P):
                vals, counts = np.unique(block[i][block[i] >= 0], return_counts=True)
                if len(vals) == 0:
                    accs.append(0.0)
                    continue
                winners = vals[counts == counts.max()]
                accs.append(float(truth[i] in winners) / len(winners))  # ties broken at random
        maj[k] = np.mean(accs)
    never = np.mean(r.sum(1) == 0)
    return passk, maj, never


# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    quick = args.quick
    t_start = time.time()
    torch.manual_seed(args.seed)
    g = np.random.default_rng(args.seed)
    cfg = dict(base_steps=300 if quick else 2000, base_batch=256, hidden=128,
               G=8, prompts_per_batch=16 if quick else 32, lr=1e-3, epochs=2, minibatches=2,
               iters=8 if quick else 200, eval_every=4 if quick else 20,
               n_eval_per_size=20 if quick else 100, tts_samples=16 if quick else 64,
               seeds=1 if quick else 3)
    methods = ["GRPO", "Dr. GRPO"] if quick else list(METHODS)
    print(f"seed={args.seed} quick={quick}")
    print("config:", cfg)
    print("methods:", {k: METHODS[k] for k in methods})

    # ---- base model ----
    print("\n[1] Base model: maximum likelihood on noisy demonstrations (carry bug prob. "
          f"{P_WRAP}, random slip prob. {P_SLIP})")
    t0 = time.time()
    base = train_base(cfg["base_steps"], cfg["base_batch"], g, cfg["hidden"],
                      log_every=cfg["base_steps"] // 4)
    ge = np.random.default_rng(12345)  # fixed evaluation prompts
    eval_set = np.concatenate([make_prompts(cfg["n_eval_per_size"], ge, [n] * cfg["n_eval_per_size"])
                               for n in range(N_MIN, N_MAX + 1)])
    ev0 = evaluate(base, base, eval_set, g, n_samples=16)
    with torch.no_grad():  # how often are the demonstrations themselves fully correct?
        dd = demos(eval_set, np.random.default_rng(7))
        r_demo, _, _ = verify(eval_set, dd, torch.ones_like(dd, dtype=torch.float) *
                              (1.0 - (torch.cumsum((dd == EOS).float(), 1) - (dd == EOS).float()).clamp(max=1.0)))
    demo_acc = r_demo.mean()
    print(f"  base: sampled acc {ev0['acc']:.3f} (demonstrations: {demo_acc:.3f}), greedy acc "
          f"{ev0['greedy']:.3f}, by size n=2..6: "
          + " ".join(f"{ev0[f'acc_n{n}']:.2f}" for n in range(N_MIN, N_MAX + 1))
          + f"; chain fully valid among correct answers {ev0['chain_given_correct']:.3f}  [{time.time() - t0:.1f}s]")

    # ---- RL runs ----
    print("\n[2] RL with the verifier as reward")
    curves = {mname: [] for mname in methods}
    finals = {}
    for mname in methods:
        for s in range(cfg["seeds"]):
            t0 = time.time()
            torch.manual_seed(1000 + s)
            gs = np.random.default_rng(1000 + s)
            pol, curve = run_rl(mname, base, cfg["iters"], gs, cfg, eval_set,
                                log_every=(cfg["iters"] // 3) if s == 0 else 0)
            curves[mname].append(curve)
            if s == 0:
                finals[mname] = pol
            f = curve[-1]
            print(f"  {mname:<19} seed {s}: acc {f['acc']:.3f}  greedy {f['greedy']:.3f}  by size "
                  + " ".join(f"{f[f'acc_n{n}']:.2f}" for n in range(N_MIN, N_MAX + 1))
                  + f"  KL {f['kl']:.2f}  entropy {f['entropy']:.3f}  chain-ok|correct "
                    f"{f['chain_given_correct']:.3f}  samples {f['samples']}  [{time.time() - t0:.1f}s]")

    # ---- test-time compute ----
    print("\n[3] Test-time compute: pass@k and majority vote (maj@k), base vs RL (GRPO, seed 0)")
    t0 = time.time()
    ks = [1, 2, 4, 8, 16] if quick else [1, 2, 4, 8, 16, 32, 64]
    tts = {}
    for label, pol in [("base", base), ("GRPO", finals["GRPO"])]:
        passk, maj, never = test_time_scaling(pol, eval_set, cfg["tts_samples"], ks, g)
        tts[label] = (passk, maj)
        print(f"  {label:<5} pass@k " + " ".join(f"k={k}:{passk[k]:.3f}" for k in ks))
        print(f"  {label:<5} maj@k  " + " ".join(f"k={k}:{maj[k]:.3f}" for k in ks)
              + f"   (problems never solved in {cfg['tts_samples']} samples: {never:.3f})")
    print(f"  [{time.time() - t0:.1f}s]")

    print("\nSummary (final iterate, mean over seeds of sampled accuracy at temperature 1)")
    print(f"  {'method':<19} {'acc':>6} {'(min-max)':>12} {'greedy':>7} {'n=6 acc':>8} {'KL':>6} {'entropy':>8}"
          f" {'chain ok':>9} {'no-signal':>10} {'samples':>8}")
    print(f"  {'base':<19} {ev0['acc']:>6.3f} {'':>12} {ev0['greedy']:>7.3f} {ev0['acc_n6']:>8.3f} {0.0:>6.2f} "
          f"{ev0['entropy']:>8.3f} {ev0['chain_given_correct']:>9.3f} {'':>10} {0:>8}")
    for mname in methods:
        fs = [c[-1] for c in curves[mname]]
        accs = [f['acc'] for f in fs]
        # 'no-signal': fraction of groups that were all-correct or all-wrong over the last
        # eval_every iterations (before DAPO's filtering)
        print(f"  {mname:<19} {np.mean(accs):>6.3f} {f'({min(accs):.3f}-{max(accs):.3f})':>12}"
              f" {np.mean([f['greedy'] for f in fs]):>7.3f}"
              f" {np.mean([f['acc_n6'] for f in fs]):>8.3f} {np.mean([f['kl'] for f in fs]):>6.2f}"
              f" {np.mean([f['entropy'] for f in fs]):>8.3f} {np.mean([f['chain_given_correct'] for f in fs]):>9.3f}"
              f" {np.mean([f['degenerate'] for f in fs]):>10.3f} {int(np.mean([f['samples'] for f in fs])):>8}")
    print("  first evaluation (after one update): no-signal fraction "
          + ", ".join(f"{m} {np.mean([c[0]['degenerate'] for c in curves[m]]):.2f}" for m in methods))
    # accuracy of each method at the sample budget of the non-filtering methods (for DAPO-lite)
    budget = min(int(np.mean([c[-1]['samples'] for c in curves[m]])) for m in methods)
    print(f"  accuracy at about {budget} sampled responses: "
          + ", ".join(f"{m} {np.mean([np.interp(budget, [e['samples'] for e in c], [e['acc'] for e in c]) for c in curves[m]]):.3f}"
                      for m in methods))
    print(f"\nTotal time {time.time() - t_start:.1f}s")
    if quick:
        return
    make_figures(curves, tts, ev0, ks)


def make_figures(curves, tts, ev0, ks):
    os.makedirs(FIG_DIR, exist_ok=True)
    colors = dict(zip(METHODS, ["0.5", "tab:brown", "tab:blue", "tab:blue", "tab:orange", "tab:orange",
                                "tab:green", "tab:purple"]))
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for mname, cl in curves.items():
        # x axis: responses sampled so far (DAPO's dynamic sampling draws extra groups)
        xs = np.array([[e["samples"] for e in c] for c in cl]).mean(0) / 1000.0
        ls = "--" if mname in ABLATIONS else "-"
        for ax, key in zip(axes[:3], ["acc", "kl", "entropy"]):
            vals = np.array([[e[key] for e in c] for c in cl])
            ax.plot(xs, vals.mean(0), ls, color=colors[mname], label=mname)
            if len(cl) > 1 and mname not in ABLATIONS:
                ax.fill_between(xs, vals.min(0), vals.max(0), color=colors[mname], alpha=0.15)
        vals = np.array([[e["degenerate"] for e in c] for c in cl])
        axes[3].plot(xs, vals.mean(0), ls, color=colors[mname], label=mname)
    axes[0].axhline(ev0["acc"], color="k", ls=":", lw=1, label="base")
    for ax, yl, t in zip(axes, ["accuracy (sampled, T=1)", r"KL$(\pi_\theta\,\Vert\,\pi_{\mathrm{ref}})$",
                                 "token entropy (nats)", "fraction of groups with no signal"],
                         ["Outcome accuracy", "Distance from the base model", "Entropy collapse",
                          "All-correct / all-wrong groups"]):
        ax.set_xlabel("responses sampled (thousands)")
        ax.set_ylabel(yl)
        ax.set_title(t)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "grpo_rlvr_curves.png"), dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sizes = list(range(N_MIN, N_MAX + 1))
    x = np.arange(len(sizes))
    names = ["base"] + [mname for mname in curves if mname not in ABLATIONS]
    w = 0.8 / len(names)
    for j, mname in enumerate(names):
        if mname == "base":
            vals = [ev0[f"acc_n{n}"] for n in sizes]
            col = "k"
        else:
            vals = [np.mean([c[-1][f"acc_n{n}"] for c in curves[mname]]) for n in sizes]
            col = colors[mname]
        axes[0].bar(x + (j - len(names) / 2 + 0.5) * w, vals, w, color=col, label=mname)
    axes[0].set_xticks(x, [f"n={n}" for n in sizes])
    axes[0].set_ylabel("accuracy (sampled)")
    axes[0].set_title("Accuracy by problem length")
    axes[0].legend(fontsize=7, ncol=2)
    for label, ls in [("base", "--"), ("GRPO", "-")]:
        passk, maj = tts[label]
        axes[1].plot(ks, [passk[k] for k in ks], "o" + ls, color="tab:blue", label=f"{label}: pass@k")
        axes[1].plot(ks, [maj[k] for k in ks], "s" + ls, color="tab:red", label=f"{label}: majority vote")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("k (samples per problem)")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Test-time compute: base vs GRPO")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "grpo_rlvr_tts.png"), dpi=110)
    plt.close(fig)
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
