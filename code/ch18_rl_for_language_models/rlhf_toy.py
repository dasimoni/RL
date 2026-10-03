"""The whole RLHF pipeline on a toy language model, end to end, on one CPU thread.

Chapter 18, Sections 2-8.  Stages (the diagram of Section 1.4):

  0. "Pre-training/SFT": a GRU language model imitates demonstrations from a toy distribution.
     This is pi_ref.
  1. Preferences: pairs of SFT samples labelled by a *hidden* true reward through the
     Bradley-Terry model (Section 2.2) -- the stand-in for human labellers.
  2. Reward model r_phi trained with the Bradley-Terry loss (eq. 18.7, Algorithm 18.1).
  3. Policy optimization against r_phi, each with the same KL coefficient beta:
       (a) PPO with a per-token KL penalty, a value model initialised from the reward model
           and token-level GAE (the InstructGPT recipe, Algorithm 18.2);
       (b) RLOO: REINFORCE with the sequence-level KL in the reward and a leave-one-out
           baseline (Algorithm 18.3);
       (c) DPO directly on the preference pairs, no reward model, no sampling (Algorithm 18.4);
       (d) best-of-n sampling from pi_ref, reranked by r_phi (Section 8.3, Algorithm 18.6).
  4. A beta sweep for PPO and DPO and the "gold vs proxy" curves of Section 8: as the policy
     moves away from pi_ref, the reward model keeps saying "better" while the true reward
     peaks and then falls (over-optimization).

The toy task.  Prompts x in {0,1,2,3} ("topics"); a response is a string of up to T=10 tokens
from a 10-word vocabulary, then EOS.  The hidden true reward is
      r*(x, y) = sum over the *distinct* words v in y of w_x(v)  -  0.6 * (number of repeated words)
so a good answer says each relevant word once and stops.  Human demonstrations never repeat a
word (they are drawn without replacement), so the reward model sees few repeats -- the
analogue of a reward model that never saw degenerate, repetitive text.

Run from the repository root:
    python code/ch18_rl_for_language_models/rlhf_toy.py            # full run with figures
    python code/ch18_rl_for_language_models/rlhf_toy.py --quick    # smoke test, no figures
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

from toy_lm import (PolicyLM, ScalarModel, masked_mean, token_kl_exact,  # noqa: E402
                    token_logprobs)

torch.set_num_threads(1)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# ---------------------------------------------------------------------------------------
# The toy task
# ---------------------------------------------------------------------------------------
N_WORDS = 10            # content tokens 0..9
EOS = N_WORDS           # token 10
VOCAB = N_WORDS + 1
T_MAX = 10              # generation budget (tokens, EOS included)
N_PROMPTS = 4
LAMBDA_REP = 0.6        # true-reward penalty per repeated word
STOP_PROB = 0.3         # demonstrations stop after each word with this probability


class Task:
    def __init__(self, seed: int):
        g = np.random.default_rng(seed)
        # which words a "human" tends to use for each topic, and how good each word is
        self.q = g.dirichlet(0.8 * np.ones(N_WORDS), size=N_PROMPTS)
        self.w = g.normal(0.0, 1.0, size=(N_PROMPTS, N_WORDS))
        self.best = np.array([np.clip(self.w[x], 0, None).sum() for x in range(N_PROMPTS)])

    def cond(self, x):
        return F.one_hot(torch.as_tensor(x), N_PROMPTS).float()

    def demos(self, xs, g: np.random.Generator):
        """Human demonstrations: distinct words drawn without replacement from q_x."""
        toks = np.full((len(xs), T_MAX), EOS, dtype=np.int64)
        for i, x in enumerate(xs):
            p = self.q[x].copy()
            for t in range(T_MAX - 1):
                v = g.choice(N_WORDS, p=p / p.sum())
                toks[i, t] = v
                p[v] = 0.0
                if g.random() < STOP_PROB or p.sum() < 1e-12:
                    break
        return torch.as_tensor(toks)

    def true_reward(self, xs, tokens, mask):
        """r*(x,y) = sum_{distinct v in y} w_x(v) - LAMBDA_REP * (#words - #distinct words)."""
        xs = np.asarray(xs)
        tok = tokens.numpy()
        valid = (mask.numpy() > 0) & (tok != EOS)
        counts = np.zeros((len(xs), N_WORDS))
        for t in range(tok.shape[1]):  # word counts per response (vectorised over the batch)
            np.add.at(counts, (np.nonzero(valid[:, t])[0], tok[valid[:, t], t]), 1.0)
        present = counts > 0
        n_rep = counts.sum(1) - present.sum(1)
        out = (self.w[xs] * present).sum(1) - LAMBDA_REP * n_rep
        return out, n_rep


def seed_all(s):
    torch.manual_seed(s)
    np.random.seed(s)


# ---------------------------------------------------------------------------------------
# Stage 0: supervised fine-tuning (maximum likelihood on demonstrations)
# ---------------------------------------------------------------------------------------
def train_sft(task, steps, batch, g, log_every=0):
    model = PolicyLM(VOCAB, N_PROMPTS)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for s in range(steps):
        xs = g.integers(0, N_PROMPTS, batch)
        toks = task.demos(xs, g)
        mask = torch.ones_like(toks, dtype=torch.float)
        # mask everything after the first EOS (the EOS itself is a decision to imitate)
        is_eos = (toks == EOS).float()
        mask = 1.0 - (torch.cumsum(is_eos, 1) - is_eos).clamp(max=1.0)
        loss = -(token_logprobs(model, task.cond(xs), toks, mask)).sum() / batch
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and (s % log_every == 0 or s == steps - 1):
            print(f"    SFT step {s:5d}  NLL/sequence {loss.item():.3f}")
    return model


# ---------------------------------------------------------------------------------------
# Stage 1-2: preferences and the Bradley-Terry reward model
# ---------------------------------------------------------------------------------------
def make_preferences(task, policy, n_pairs, g):
    """Pairs of policy samples for the same prompt; label ~ Bernoulli(sigma(r*(y1) - r*(y2)))."""
    xs = g.integers(0, N_PROMPTS, n_pairs)
    c = task.cond(xs)
    y1, m1 = policy.sample(c, T_MAX)
    y2, m2 = policy.sample(c, T_MAX)
    r1, _ = task.true_reward(xs, y1, m1)
    r2, _ = task.true_reward(xs, y2, m2)
    p1 = 1.0 / (1.0 + np.exp(-(r1 - r2)))
    first_wins = g.random(n_pairs) < p1
    fw = torch.as_tensor(first_wins).unsqueeze(1)
    yw = torch.where(fw, y1, y2)
    mw = torch.where(fw, m1, m2)
    yl = torch.where(fw, y2, y1)
    ml = torch.where(fw, m2, m1)
    bayes_acc = np.mean(np.where(r1 == r2, 0.5, (r1 > r2) == first_wins))
    return dict(xs=xs, yw=yw, mw=mw, yl=yl, ml=ml, bayes_acc=bayes_acc)


def subset(d, idx):
    return {k: (v[idx] if k != "bayes_acc" else v) for k, v in d.items()}


def train_reward_model(task, prefs, val, epochs, g, batch=64):
    """Minimise -log sigma(r_phi(x,y_w) - r_phi(x,y_l)) (eq. 18.7); keep the best on validation."""
    rm = ScalarModel(VOCAB, N_PROMPTS)
    opt = torch.optim.Adam(rm.parameters(), lr=2e-3, weight_decay=1e-4)
    n = len(prefs["xs"])

    def evaluate(d):
        with torch.no_grad():
            c = task.cond(d["xs"])
            diff = rm.score(c, d["yw"], d["mw"]) - rm.score(c, d["yl"], d["ml"])
            return F.softplus(-diff).mean().item(), (diff > 0).float().mean().item()

    best, best_state, hist = 1e9, None, []
    for ep in range(epochs):
        perm = g.permutation(n)
        for i in range(0, n, batch):
            b = subset(prefs, perm[i:i + batch])
            c = task.cond(b["xs"])
            diff = rm.score(c, b["yw"], b["mw"]) - rm.score(c, b["yl"], b["ml"])
            loss = F.softplus(-diff).mean()  # = -log sigma(diff)
            opt.zero_grad()
            loss.backward()
            opt.step()
        vl, va = evaluate(val)
        tl, ta = evaluate(prefs)
        hist.append((ep, tl, ta, vl, va))
        if vl < best:
            best, best_state = vl, copy.deepcopy(rm.state_dict())
    rm.load_state_dict(best_state)
    for p in rm.parameters():
        p.requires_grad_(False)
    return rm, hist


class NormalizedRM:
    """r_phi standardised to mean 0, std 1 under pi_ref (InstructGPT normalised its RM so that
    reference outputs score 0 on average).  Also holds a linear calibration a + b r_phi of the
    true reward fitted on pi_ref samples, used to measure the proxy-vs-gold gap (Section 8.4)."""

    def __init__(self, rm, task, ref, g, n=4096):
        self.rm = rm
        xs = g.integers(0, N_PROMPTS, n)
        y, m = ref.sample(task.cond(xs), T_MAX)
        with torch.no_grad():
            s = rm.score(task.cond(xs), y, m).numpy()
        self.mu, self.sd = s.mean(), s.std()
        z = (s - self.mu) / self.sd
        rt, _ = task.true_reward(xs, y, m)
        self.b, self.a = np.polyfit(z, rt, 1)
        self.corr = np.corrcoef(z, rt)[0, 1]

    def __call__(self, cond, tokens, mask):
        with torch.no_grad():
            return (self.rm.score(cond, tokens, mask) - self.mu) / self.sd

    def predicted_true(self, z):
        return self.a + self.b * z


# ---------------------------------------------------------------------------------------
# Evaluation of a policy
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_policy(task, policy, ref, rmn, g, n=2048):
    xs = g.integers(0, N_PROMPTS, n)
    c = task.cond(xs)
    y, m = policy.sample(c, T_MAX)
    rt, n_rep = task.true_reward(xs, y, m)
    z = rmn(c, y, m).numpy()
    kl = token_kl_exact(policy, ref, c, y, m).sum(1).numpy()
    length = (m.sum(1) - (y == EOS).float().mul(m).sum(1)).numpy()
    return dict(true=rt.mean(), rm=z.mean(), rm_pred_true=rmn.predicted_true(z).mean(),
                kl=kl.mean(), rep=(n_rep > 0).mean(), length=length.mean(),
                frac_of_best=(rt / task.best[xs]).mean())


# ---------------------------------------------------------------------------------------
# Stage 3a: PPO with a per-token KL penalty, value model and token-level GAE (Algorithm 18.2)
# ---------------------------------------------------------------------------------------
def gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Token-level GAE (eq. 18.15).  V(s_{L+1}) = 0: the end of the response is terminal."""
    B, T = rewards.shape
    adv = torch.zeros(B, T)
    last = torch.zeros(B)
    for t in reversed(range(T)):
        v_next = values[:, t + 1] * mask[:, t + 1] if t + 1 < T else torch.zeros(B)
        delta = rewards[:, t] + gamma * v_next - values[:, t]
        last = delta + gamma * lam * last * (mask[:, t + 1] if t + 1 < T else 0.0)
        adv[:, t] = last * mask[:, t]
    return adv


def run_ppo(task, ref, rmn, rm_raw, beta, iters, g, cfg, log_every=0, eval_every=0, kl_target=None):
    """PPO-RLHF.  With kl_target set, beta is adapted after every batch by a proportional
    controller in the style of Ziegler et al. (2019):
        e = clip(KL/kl_target - 1, -0.2, 0.2),   beta <- beta * (1 + 0.1 e)."""
    policy = copy.deepcopy(ref)
    for p in policy.parameters():
        p.requires_grad_(True)
    critic = ScalarModel(VOCAB, N_PROMPTS)
    critic.load_state_dict(rm_raw.state_dict())  # InstructGPT: value model initialised from the RM
    for p in critic.parameters():
        p.requires_grad_(True)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])
    opt_v = torch.optim.Adam(critic.parameters(), lr=cfg["lr_v"])
    B, mb, eps = cfg["batch"], cfg["minibatch"], cfg["clip"]
    curve = []
    for it in range(iters):
        xs = g.integers(0, N_PROMPTS, B)
        c = task.cond(xs)
        y, m = policy.sample(c, T_MAX)
        with torch.no_grad():
            logp_old = token_logprobs(policy, c, y, m)
            logp_ref = token_logprobs(ref, c, y, m)
            values = critic.values(c, y) * m
            score = rmn(c, y, m)
            # per-token reward: -beta * log(pi_old/pi_ref) at every token, + r_phi at the last one
            rew = -beta * (logp_old - logp_ref)  # eq. 18.14
            last = (m.sum(1) - 1).long()
            rew[torch.arange(B), last] += score
            adv = gae(rew, values, m, 1.0, cfg["lam"])
            ret = adv + values
            a_mean = masked_mean(adv, m)
            a_std = torch.sqrt(masked_mean((adv - a_mean) ** 2, m))
            adv_n = (adv - a_mean) / (a_std + 1e-8) * m  # advantage whitening
            kl_batch = (logp_old - logp_ref).sum(1).mean().item()  # k1 estimate of the sequence KL
        if kl_target is not None:
            beta = beta * (1.0 + 0.1 * float(np.clip(kl_batch / kl_target - 1.0, -0.2, 0.2)))
        for _ in range(cfg["epochs"]):
            perm = torch.as_tensor(g.permutation(B))
            for i in range(0, B, mb):
                idx = perm[i:i + mb]
                ci, yi, mi = c[idx], y[idx], m[idx]
                logp = token_logprobs(policy, ci, yi, mi)
                ratio = torch.exp(logp - logp_old[idx])
                s1 = ratio * adv_n[idx]
                s2 = torch.clamp(ratio, 1 - eps, 1 + eps) * adv_n[idx]
                loss_pi = -masked_mean(torch.min(s1, s2), mi)  # clipped surrogate, eq. 18.16
                opt_pi.zero_grad()
                loss_pi.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt_pi.step()
                v = critic.values(ci, yi)
                loss_v = masked_mean((v - ret[idx]) ** 2, mi)
                opt_v.zero_grad()
                loss_v.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_v.step()
        if eval_every and (it % eval_every == 0 or it == iters - 1):
            ev = evaluate_policy(task, policy, ref, rmn, g, n=cfg["n_eval"])
            ev["it"] = it
            ev["beta"] = beta
            curve.append(ev)
            if log_every and (it % log_every == 0 or it == iters - 1):
                print(f"    PPO beta={beta:<5} it {it:4d}  true {ev['true']:+.3f}  rm {ev['rm']:+.3f}"
                      f"  KL {ev['kl']:.2f}  repeats {ev['rep']:.2f}  len {ev['length']:.1f}")
    return policy, curve


# ---------------------------------------------------------------------------------------
# Stage 3b: RLOO with the sequence-level KL in the reward (Algorithm 18.3)
# ---------------------------------------------------------------------------------------
def run_rloo(task, ref, rmn, beta, iters, g, cfg, log_every=0, eval_every=0):
    policy = copy.deepcopy(ref)
    for p in policy.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg["lr_rloo"])
    k, n_prompts = cfg["k"], cfg["batch"] // cfg["k"]
    curve = []
    for it in range(iters):
        xs = np.repeat(g.integers(0, N_PROMPTS, n_prompts), k)  # k samples per prompt
        c = task.cond(xs)
        y, m = policy.sample(c, T_MAX)
        logp = token_logprobs(policy, c, y, m)
        with torch.no_grad():
            logp_ref = token_logprobs(ref, c, y, m)
            R = rmn(c, y, m) - beta * (logp - logp_ref).sum(1)  # KL-shaped sequence reward (18.13)
            Rg = R.view(n_prompts, k)
            base = (Rg.sum(1, keepdim=True) - Rg) / (k - 1)  # leave-one-out baseline
            A = (Rg - base).view(-1)
        loss = -(A * logp.sum(1)).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if eval_every and (it % eval_every == 0 or it == iters - 1):
            ev = evaluate_policy(task, policy, ref, rmn, g, n=cfg["n_eval"])
            ev["it"] = it
            curve.append(ev)
            if log_every and (it % log_every == 0 or it == iters - 1):
                print(f"    RLOO beta={beta:<5} it {it:4d}  true {ev['true']:+.3f}  rm {ev['rm']:+.3f}"
                      f"  KL {ev['kl']:.2f}  repeats {ev['rep']:.2f}  len {ev['length']:.1f}")
    return policy, curve


# ---------------------------------------------------------------------------------------
# Stage 3c: DPO on the preference pairs (Algorithm 18.4)
# ---------------------------------------------------------------------------------------
def run_dpo(task, ref, rmn, prefs, val, beta, epochs, g, cfg, log_every=0, eval_every=0, loss_type="dpo"):
    policy = copy.deepcopy(ref)
    for p in policy.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg["lr_dpo"])
    n = len(prefs["xs"])
    with torch.no_grad():  # reference log-probabilities are computed once
        def ref_lp(d):
            c = task.cond(d["xs"])
            return (token_logprobs(ref, c, d["yw"], d["mw"]).sum(1),
                    token_logprobs(ref, c, d["yl"], d["ml"]).sum(1))
        ref_w, ref_l = ref_lp(prefs)
        vref_w, vref_l = ref_lp(val)
    curve, step = [], 0
    for ep in range(epochs):
        perm = g.permutation(n)
        for i in range(0, n, cfg["dpo_batch"]):
            idx = perm[i:i + cfg["dpo_batch"]]
            b = subset(prefs, idx)
            c = task.cond(b["xs"])
            lw = token_logprobs(policy, c, b["yw"], b["mw"]).sum(1)
            ll = token_logprobs(policy, c, b["yl"], b["ml"]).sum(1)
            hw, hl = lw - ref_w[idx], ll - ref_l[idx]  # log-ratios h = log pi_theta/pi_ref
            if loss_type == "dpo":
                loss = F.softplus(-beta * (hw - hl)).mean()  # -log sigma(beta (h_w - h_l)), eq. 18.19
            elif loss_type == "ipo":  # IPO (Azar et al.): squared loss toward a margin 1/(2 tau), tau = beta
                loss = ((hw - hl - 1.0 / (2.0 * beta)) ** 2).mean()
            elif loss_type == "simpo":  # SimPO: no reference model, length-normalised, margin gamma
                margin = beta * (lw / b["mw"].sum(1) - ll / b["ml"].sum(1)) - cfg.get("simpo_gamma", 0.5)
                loss = F.softplus(-margin).mean()
            else:
                raise ValueError(loss_type)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1
        if eval_every and (ep % eval_every == 0 or ep == epochs - 1):
            ev = evaluate_policy(task, policy, ref, rmn, g, n=cfg["n_eval"])
            with torch.no_grad():
                cv = task.cond(val["xs"])
                lw = token_logprobs(policy, cv, val["yw"], val["mw"]).sum(1)
                ll = token_logprobs(policy, cv, val["yl"], val["ml"]).sum(1)
                ev["val_acc"] = ((lw - vref_w) - (ll - vref_l) > 0).float().mean().item()
                ev["dlogp_w"] = (lw - vref_w).mean().item()  # change in log-prob of chosen
                ev["dlogp_l"] = (ll - vref_l).mean().item()  # ... and of rejected responses
            ev["it"] = ep
            curve.append(ev)
            if log_every and (ep % log_every == 0 or ep == epochs - 1):
                print(f"    {loss_type.upper()} beta={beta:<5} epoch {ep:3d}  true {ev['true']:+.3f}"
                      f"  rm {ev['rm']:+.3f}  KL {ev['kl']:.2f}  val-acc {ev['val_acc']:.3f}"
                      f"  dlogp(chosen) {ev['dlogp_w']:+.2f}  dlogp(rejected) {ev['dlogp_l']:+.2f}")
    return policy, curve


# ---------------------------------------------------------------------------------------
# Stage 3d: best-of-n reranking with the reward model
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def best_of_n(task, ref, rmn, ns, g, n_prompts=512):
    xs = g.integers(0, N_PROMPTS, n_prompts)
    nmax = max(ns)
    c = task.cond(np.repeat(xs, nmax))
    y, m = ref.sample(c, T_MAX)
    z = rmn(c, y, m).numpy().reshape(n_prompts, nmax)
    rt, _ = task.true_reward(np.repeat(xs, nmax), y, m)
    rt = rt.reshape(n_prompts, nmax)
    rows = []
    for n in ns:
        # use disjoint blocks of n samples to average over more draws
        blocks = nmax // n
        tr, rmv, oracle = [], [], []
        for b in range(blocks):
            zb, rb = z[:, b * n:(b + 1) * n], rt[:, b * n:(b + 1) * n]
            j = zb.argmax(1)
            tr.append(rb[np.arange(n_prompts), j].mean())
            rmv.append(zb[np.arange(n_prompts), j].mean())
            oracle.append(rb.max(1).mean())
        rows.append(dict(n=n, kl=math.log(n) - (n - 1) / n, true=np.mean(tr), rm=np.mean(rmv),
                         oracle=np.mean(oracle), rm_pred_true=rmn.predicted_true(np.mean(rmv))))
    return rows


# ---------------------------------------------------------------------------------------
@torch.no_grad()
def probe_repeats(task, rmn):
    """What does the reward model say about 'the best word, repeated L times'?  (r* says: worse)"""
    rows = []
    for x in range(N_PROMPTS):
        v = int(np.argmax(task.w[x]))
        zs, ts = [], []
        for L in [1, 2, 3, 5, 9]:
            t = torch.full((1, T_MAX), EOS)
            t[0, :L] = v
            m = torch.zeros(1, T_MAX)
            m[0, :L + 1] = 1
            zs.append(rmn(task.cond([x]), t, m).item())
            ts.append(task.true_reward([x], t, m)[0][0])
        rows.append((x, zs, ts))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    quick = args.quick
    t_start = time.time()
    seed_all(args.seed)
    g = np.random.default_rng(args.seed)
    task = Task(seed=args.seed + 100)

    cfg = dict(sft_steps=300 if quick else 1500, sft_batch=128,
               n_pairs=600 if quick else 2000, n_pairs_large=1200 if quick else 8000,
               n_val=300 if quick else 1000, rm_epochs=3 if quick else 12, rm_epochs_large=2 if quick else 6,
               batch=128 if quick else 256, minibatch=64, epochs=4, clip=0.2, lam=0.95,
               lr=3e-4, lr_v=1e-3, ppo_iters=12 if quick else 120, ppo_eval_every=4,
               k=4, lr_rloo=1e-3, rloo_iters=12 if quick else 400,
               lr_dpo=1e-3, dpo_batch=64, dpo_epochs=2 if quick else 20,
               n_eval=256 if quick else 1024)
    betas_ppo = [0.1] if quick else [0.01, 0.03, 0.1, 0.3, 1.0]
    betas_ppo_large = [0.01] if quick else [0.01, 0.1]
    betas_rloo = [0.1] if quick else [0.03, 0.1]
    betas_dpo = [0.1] if quick else [0.03, 0.1, 0.3, 1.0]
    print(f"seed={args.seed} quick={quick}  true-reward weights w (rows = prompts):")
    print(np.round(task.w, 2))
    print("config:", cfg)
    print(f"betas: PPO {betas_ppo}, PPO with large RM {betas_ppo_large}, RLOO {betas_rloo}, DPO {betas_dpo}")

    # ---- Stage 0 ----
    print("\n[Stage 0] SFT on demonstrations")
    t0 = time.time()
    ref = train_sft(task, cfg["sft_steps"], cfg["sft_batch"], g, log_every=cfg["sft_steps"] // 3)
    for p in ref.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        xs = g.integers(0, N_PROMPTS, 4096)
        y, m = ref.sample(task.cond(xs), T_MAX)
        rt, nrep = task.true_reward(xs, y, m)
        dem = task.demos(xs, g)
        rd, _ = task.true_reward(xs, dem, (dem != EOS).float())
    print(f"  SFT samples: true reward {rt.mean():+.3f} (demonstrations {rd.mean():+.3f}; best possible "
          f"{task.best.mean():+.3f}); fraction with a repeated word {np.mean(nrep > 0):.3f}  [{time.time() - t0:.1f}s]")

    # ---- Stage 1-2 ----
    print("\n[Stage 1-2] Preferences and Bradley-Terry reward models")
    t0 = time.time()
    val = make_preferences(task, ref, cfg["n_val"], g)
    rms = {}
    for name, n_pairs, ep in [("small", cfg["n_pairs"], cfg["rm_epochs"]),
                              ("large", cfg["n_pairs_large"], cfg["rm_epochs_large"])]:
        prefs_i = make_preferences(task, ref, n_pairs, g)
        rm_i, hist = train_reward_model(task, prefs_i, val, ep, g)
        rmn_i = NormalizedRM(rm_i, task, ref, g)
        be = int(np.argmin([h[3] for h in hist]))
        rms[name] = (rm_i, rmn_i, prefs_i)
        print(f"  RM-{name}: {n_pairs} pairs, best epoch {be}: train acc {hist[be][2]:.3f}, val acc {hist[be][4]:.3f}"
              f" (a perfect model of r* gets {val['bayes_acc']:.3f}); corr(r_phi, r*) on pi_ref samples {rmn_i.corr:.3f}")
        for x, zs, ts in probe_repeats(task, rmn_i):
            print(f"     prompt {x}, best word repeated 1,2,3,5,9 times: r_phi (std. units) "
                  + " ".join(f"{z:+.2f}" for z in zs) + "   r* " + " ".join(f"{t:+.2f}" for t in ts))
    rm, rmn, prefs = rms["small"]
    print(f"  [{time.time() - t0:.1f}s]")
    ev_ref = evaluate_policy(task, ref, ref, rmn, g, n=cfg["n_eval"])
    print(f"  pi_ref: true {ev_ref['true']:+.3f}  rm {ev_ref['rm']:+.3f}  repeats {ev_ref['rep']:.3f}"
          f"  length {ev_ref['length']:.2f}")

    results = {"ppo": {}, "ppo_large": {}, "rloo": {}, "dpo": {}, "ipo": {}}
    summary = []

    def report(tag, beta, curve, t0, extra=""):
        f = curve[-1]
        pk = max(curve, key=lambda e: e["true"])  # best point along the training trajectory
        summary.append((tag, beta, f))
        print(f"  {tag:<10} beta={beta:<5} final: true {f['true']:+.3f}  rm {f['rm']:+.3f} (RM-predicted true "
              f"{f['rm_pred_true']:+.3f})  KL {f['kl']:5.2f}  repeats {f['rep']:.2f}  len {f['length']:.2f}"
              f"  | peak true {pk['true']:+.3f} at KL {pk['kl']:.2f} (it {pk['it']}){extra}  [{time.time() - t0:.1f}s]")

    # ---- Stage 3a ----
    print("\n[Stage 3a] PPO with a per-token KL penalty (value model initialised from the RM, token-level GAE)")
    for key, (rm_k, rmn_k, _), betas in [("ppo", rms["small"], betas_ppo), ("ppo_large", rms["large"], betas_ppo_large)]:
        for beta in betas:
            t0 = time.time()
            seed_all(args.seed + 1)
            _, curve = run_ppo(task, ref, rmn_k, rm_k, beta, cfg["ppo_iters"], g, cfg,
                               eval_every=cfg["ppo_eval_every"])
            results[key][beta] = curve
            report("PPO" if key == "ppo" else "PPO-RM8k", beta, curve, t0)

    # ---- Stage 3b ----
    print("\n[Stage 3b] RLOO (k=4) with the sequence-level KL in the reward")
    for beta in betas_rloo:
        t0 = time.time()
        seed_all(args.seed + 2)
        _, curve = run_rloo(task, ref, rmn, beta, cfg["rloo_iters"], g, cfg, eval_every=20 if not quick else 4)
        results["rloo"][beta] = curve
        report("RLOO", beta, curve, t0)

    # ---- Stage 3c ----
    print("\n[Stage 3c] DPO (and IPO) on the same 2000 preference pairs, no reward model")
    for loss_type, betas in [("dpo", betas_dpo), ("ipo", [0.1])]:
        for beta in betas:
            t0 = time.time()
            seed_all(args.seed + 3)
            _, curve = run_dpo(task, ref, rmn, prefs, val, beta, cfg["dpo_epochs"], g, cfg,
                               eval_every=1, loss_type=loss_type)
            results[loss_type][beta] = curve
            f = curve[-1]
            report(loss_type.upper(), beta, curve, t0,
                   extra=f"  val-acc {f['val_acc']:.3f}  dlogp(chosen) {f['dlogp_w']:+.2f}"
                         f"  dlogp(rejected) {f['dlogp_l']:+.2f}")

    # ---- Stage 3d ----
    print("\n[Stage 3d] Best-of-n from pi_ref reranked by r_phi (KL <= log n - (n-1)/n)")
    t0 = time.time()
    ns = [1, 2, 4, 8, 16, 32] if quick else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    bon = {}
    for name in ["small", "large"]:
        bon[name] = best_of_n(task, ref, rms[name][1], ns, g, n_prompts=64 if quick else 512)
        for row in bon[name]:
            print(f"  RM-{name}: n={row['n']:<4} KL<={row['kl']:.2f}  true {row['true']:+.3f}  rm {row['rm']:+.3f}"
                  f"  (RM-predicted true {row['rm_pred_true']:+.3f}; oracle best-of-n by r*: {row['oracle']:+.3f})")
    print(f"  [{time.time() - t0:.1f}s]")

    print("\nSummary (final policies; 'true' = hidden reward r*, best achievable "
          f"{task.best.mean():+.3f}, pi_ref {ev_ref['true']:+.3f})")
    print(f"  {'method':<10} {'beta':>6} {'true':>7} {'RM(z)':>7} {'KL':>6} {'repeat':>7}")
    for tag, beta, f in summary:
        print(f"  {tag:<10} {beta:>6} {f['true']:>+7.3f} {f['rm']:>+7.3f} {f['kl']:>6.2f} {f['rep']:>7.2f}")
    print(f"\nTotal time {time.time() - t_start:.1f}s")
    if quick:
        return
    make_figures(results, bon, ev_ref, task)


def make_figures(results, bon, ev_ref, task):
    os.makedirs(FIG_DIR, exist_ok=True)
    cmap = plt.get_cmap("viridis")

    # 1. gold vs proxy along the optimization path (Section 8.4, cf. Gao et al. 2023)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, key, bkey, title in [(axes[0], "ppo", "small", "Reward model from 2000 pairs"),
                                 (axes[1], "ppo_large", "large", "Reward model from 8000 pairs")]:
        betas = sorted(results[key])
        for j, beta in enumerate(betas):
            col = cmap(j / max(1, len(betas) - 1) * 0.85)
            cv = results[key][beta]
            d = np.sqrt([max(e["kl"], 0) for e in cv])
            ax.plot(d, [e["true"] for e in cv], "-", color=col, lw=1.6, label=rf"PPO $\beta$={beta}: true reward")
            ax.plot(d, [e["rm_pred_true"] for e in cv], "--", color=col, lw=1.0)
        rows = bon[bkey]
        d = np.sqrt([r["kl"] for r in rows])
        ax.plot(d, [r["true"] for r in rows], "o-", color="tab:red", ms=3, lw=1.2, label="best-of-n: true reward")
        ax.plot(d, [r["rm_pred_true"] for r in rows], "o--", color="tab:red", ms=3, lw=0.8)
        ax.plot([], [], "k--", lw=1, label="dashed: what the RM predicts")
        ax.axhline(task.best.mean(), color="0.5", lw=0.8, ls=":", label="best achievable")
        ax.set_xlabel(r"$\sqrt{\mathrm{KL}(\pi\,\Vert\,\pi_{\mathrm{ref}})}$  (nats$^{1/2}$)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("reward (in units of the true reward)")
    axes[0].legend(fontsize=7, loc="lower right")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Over-optimization: the proxy keeps improving, the true reward peaks and falls", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "rlhf_overoptimization.png"), dpi=110)
    plt.close(fig)

    # 2. final true reward and KL vs beta
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for key, marker, lab in [("ppo", "o-", "PPO (RM-2k)"), ("ppo_large", "s-", "PPO (RM-8k)"),
                             ("rloo", "^-", "RLOO (RM-2k)"), ("dpo", "D-", "DPO (same 2k pairs)")]:
        betas = sorted(results[key])
        f = [results[key][b][-1] for b in betas]
        axes[0].plot(betas, [e["true"] for e in f], marker, label=lab)
        axes[1].plot(betas, [e["kl"] for e in f], marker, label=lab)
        axes[2].plot(betas, [e["rep"] for e in f], marker, label=lab)
    axes[0].axhline(ev_ref["true"], color="0.4", ls=":", lw=1, label=r"$\pi_{\mathrm{ref}}$")
    axes[0].axhline(task.best.mean(), color="0.6", ls="--", lw=1, label="best achievable")
    for ax, yl in zip(axes, ["final true reward", r"final KL$(\pi\,\Vert\,\pi_{\mathrm{ref}})$",
                              "fraction of responses with a repeat"]):
        ax.set_xscale("log")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Too small a $\\beta$ over-optimizes; too large under-optimizes")
    axes[1].set_title("KL budget used")
    axes[2].set_title("The hack: repeating words")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "rlhf_beta_sweep.png"), dpi=110)
    plt.close(fig)

    # 3. DPO dynamics
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    betas = sorted(results["dpo"])
    for j, beta in enumerate(betas):
        col = cmap(j / max(1, len(betas) - 1) * 0.85)
        cv = results["dpo"][beta]
        ep = [e["it"] + 1 for e in cv]
        axes[0].plot(ep, [e["dlogp_w"] for e in cv], "-", color=col, label=rf"$\beta$={beta} chosen")
        axes[0].plot(ep, [e["dlogp_l"] for e in cv], ":", color=col, label=rf"$\beta$={beta} rejected")
        axes[1].plot(ep, [e["kl"] for e in cv], "-", color=col, label=rf"DPO $\beta$={beta}")
        axes[2].plot(ep, [e["true"] for e in cv], "-", color=col, label=rf"DPO $\beta$={beta}")
    for tau, cv in results["ipo"].items():
        ep = [e["it"] + 1 for e in cv]
        axes[1].plot(ep, [e["kl"] for e in cv], "k--", label=rf"IPO $\tau$={tau}")
        axes[2].plot(ep, [e["true"] for e in cv], "k--", label=rf"IPO $\tau$={tau}")
    axes[2].axhline(ev_ref["true"], color="0.4", ls=":", lw=1, label=r"$\pi_{\mathrm{ref}}$")
    axes[0].set_ylabel(r"change in $\log\pi_\theta(y\mid x)$ on held-out pairs")
    axes[0].set_title("Both chosen and rejected log-probs fall")
    axes[1].set_ylabel(r"KL$(\pi_\theta\,\Vert\,\pi_{\mathrm{ref}})$")
    axes[1].set_title("KL keeps growing with epochs")
    axes[2].set_ylabel("true reward of samples")
    axes[2].set_title("True reward")
    for ax in axes:
        ax.set_xlabel("epoch over the 2000 pairs")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dpo_dynamics.png"), dpi=110)
    plt.close(fig)
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
