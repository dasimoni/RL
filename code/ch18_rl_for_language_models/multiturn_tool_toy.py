"""Multi-turn RL with a tool on a toy language model: observation masks, filtered SFT and stale samplers.

Chapter 18, Section 13 (and Section 9.5 for filtered SFT).  The task is the running-sum task of
grpo_rlvr.py (prompt: n = 2..6 digits; write the running sums mod 10, one per step, then EOS;
reward 1 if the response ends with EOS and its last value is the right answer), but now the
model has a calculator.  At any step it may emit the token CALL instead of writing a digit.
The *environment* then appends an observation token R_v, where v = (s + d_k) mod 10 is the
correct next running sum (s is the last value written), and generation continues.  So a
trajectory is

      x,  y_1, o_1, y_2, o_2, ..., y_K          e.g.  prompt 3 9 4 7  ->  3 CALL R2 6 CALL R3 EOS

with policy segments y_k (ending in CALL or EOS) and tool outputs o_k.  The policy never chooses
the R tokens after a CALL: they are observations, and the observation mask m_t is 0 there.
The R tokens are in the model's vocabulary, though, so the policy *could* write one itself; we
count that as a fabricated tool output (it is parsed as the step's value, like a real one).

Token budget.  A response may use at most T_MAX = 9 tokens.  Each step costs one token, a call
one more (CALL + R_v), and EOS one.  So a problem with n digits leaves room for 8 - n calls: every
step can be delegated when n <= 4, but only 3 of 5 steps when n = 5 and 2 of 6 when n = 6.  A
response that runs out of budget (including a CALL in the last position, whose output no longer
fits) is truncated and scored 0: nothing is bootstrapped past the budget (Pitfall 13).

The base model is trained by maximum likelihood on demonstrations from the noisy demonstrator of
grpo_rlvr.py (carry bug with probability P_WRAP, random slips P_SLIP), who also calls the tool at a
random Q_CALL fraction of steps when the budget allows, *regardless* of whether the step needs it.
The SFT loss is masked to the demonstrator's own tokens.  So the base model can use the tool but
does not know when it helps: at carry steps, where its own digit is wrong 60% of the time.

What runs (G = 8 responses per prompt, 32 prompts per batch, 2 epochs x 2 minibatches, 3 seeds):
  GRPO (masked)        trajectory-level GRPO: one standardized advantage per trajectory, applied to
                       the policy's tokens only; loss sum_t m_t clip(rho_t) A / sum_t m_t (eq. 18.35)
  Filtered SFT         expert iteration: log-likelihood of the policy tokens of successful
                       trajectories (Section 9.5)
  GRPO, unmasked       the same GRPO, but tool-output tokens are treated as if the policy had
                       chosen them (included in the loss and in the normaliser)
  GRPO, stale sampler  rollouts come from a copy of the policy LAG iterations old (an asynchronous
                       actor); the trainer ignores this and treats them as on-policy
  GRPO, stale + TIS    the same stale sampler, with each token's surrogate multiplied by the truncated
                       importance weight min(pi_old / pi_sampler, C) (Section 13.3)

Run from the repository root:
    python code/ch18_rl_for_language_models/multiturn_tool_toy.py           # full, figure (~4 min)
    python code/ch18_rl_for_language_models/multiturn_tool_toy.py --quick   # smoke test, no figure
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import collections  # noqa: E402
import copy  # noqa: E402
import time  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from toy_lm import PolicyLM, token_logprobs  # noqa: E402

torch.set_num_threads(1)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

N_MIN, N_MAX = 2, 6
OBS0 = 10                  # tool outputs R_0..R_9 are tokens 10..19
CALL = 20                  # "call the calculator"
EOS = 21
VOCAB = 22                 # PolicyLM: EOS = VOCAB - 1, BOS = VOCAB (input only)
T_MAX = 9                  # token budget: n steps + EOS + (8 - n) calls
COND_DIM = N_MAX * 11
P_SLIP = 0.05              # demonstrator's random slip rate (as in grpo_rlvr.py)
P_WRAP = 0.6               # demonstrator's carry bug (as in grpo_rlvr.py)
Q_CALL = 0.2               # demonstrator calls the tool at this fraction of steps, at random
TIS_C = 2.0                # truncation level of the importance weights

METHODS = {
    "GRPO (masked)":       dict(loss="grpo", mask_obs=True, lag=0, tis=False),
    "Filtered SFT":        dict(loss="sft", mask_obs=True, lag=0, tis=False),
    "GRPO, unmasked":      dict(loss="grpo", mask_obs=False, lag=0, tis=False),
    "GRPO, stale sampler": dict(loss="grpo", mask_obs=True, lag=None, tis=False),
    "GRPO, stale + TIS":   dict(loss="grpo", mask_obs=True, lag=None, tis=True),
}


# ---------------------------------------------------------------------------------------
# Task, demonstrations and the environment loop
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
    """Demonstrations with tool calls at random steps.  Returns tokens and the action mask
    (1 for the demonstrator's own tokens, 0 for tool outputs and padding)."""
    B = len(digits)
    out = np.full((B, T_MAX), EOS, dtype=np.int64)
    act = np.zeros((B, T_MAX), dtype=np.float32)
    for i, row in enumerate(digits):
        ds = row[row >= 0]
        n, s, pos = len(ds), 0, 0
        for k, d in enumerate(ds):
            spare = T_MAX - (pos + (n - k) + 1)  # tokens left if the rest are one-token steps + EOS
            if spare >= 1 and g.random() < Q_CALL:
                s = (s + d) % 10
                out[i, pos], act[i, pos], out[i, pos + 1] = CALL, 1.0, OBS0 + s
                pos += 2
            else:
                wraps = s + d >= 10
                s = (s + d) % 10
                if wraps and g.random() < P_WRAP:
                    s = (s + 1) % 10
                elif g.random() < P_SLIP:
                    s = (s + (1 if g.random() < 0.5 else -1)) % 10
                out[i, pos], act[i, pos] = s, 1.0
                pos += 1
        out[i, pos], act[i, pos] = EOS, 1.0
    return torch.as_tensor(out), torch.as_tensor(act)


@torch.no_grad()
def rollout(policy, digits):
    """Sample trajectories with the environment in the loop.

    The policy samples a token at every position, except right after a CALL, where the tool's
    output is written instead (and the observation mask is 0).  Returns tokens, the action mask
    `act` (tokens the policy chose), the mask `real` of all tokens up to EOS (actions and
    observations), the reward and diagnostics.
    """
    B = len(digits)
    c = cond_of(digits)
    n = (digits >= 0).sum(1)
    dpad = np.where(digits < 0, 0, digits)
    answer = dpad.sum(1) % 10
    ar = np.arange(B)
    tokens = torch.full((B, T_MAX), EOS, dtype=torch.long)
    act = torch.zeros(B, T_MAX)
    real = torch.zeros(B, T_MAX)
    alive = np.ones(B, dtype=bool)
    ended = np.zeros(B, dtype=bool)
    pending = np.full(B, -1)                      # tool output to write at this position
    s = np.zeros(B, dtype=np.int64)               # last value written (by the policy or the tool)
    k = np.zeros(B, dtype=np.int64)               # steps taken
    calls, fab, carry_steps, carry_calls, late_calls = (np.zeros(B) for _ in range(5))
    inp = torch.full((B, 1), policy.bos, dtype=torch.long)
    h = None
    for t in range(T_MAX):
        out, h = policy.trunk(c, inp, h)
        y = torch.multinomial(F.softmax(policy.head(out[:, 0]), dim=-1), 1)[:, 0].numpy()
        obs = pending >= 0
        y = np.where(obs, pending, y)
        y = np.where(alive, y, EOS)
        tokens[:, t] = torch.as_tensor(y)
        real[:, t] = torch.as_tensor(alive.astype(np.float32))
        a = alive & ~obs
        act[:, t] = torch.as_tensor(a.astype(np.float32))
        pending[:] = -1
        dk = np.where(k < n, dpad[ar, np.minimum(k, N_MAX - 1)], 0)
        carry = (k < n) & (s + dk >= 10)
        is_dig, is_fab = a & (y < OBS0), a & (y >= OBS0) & (y < OBS0 + 10)
        is_call, is_eos = a & (y == CALL), a & (y == EOS)
        step = is_dig | is_fab | is_call
        carry_steps += step & carry
        carry_calls += is_call & carry
        s_new = np.where(is_dig, y, np.where(is_fab, y - OBS0, np.where(is_call, (s + dk) % 10, s)))
        pending[is_call] = OBS0 + s_new[is_call]
        s, k = s_new, k + step
        calls += is_call
        late_calls += is_call & (k > n)          # a call after the last digit (returns s unchanged)
        fab += is_fab
        ended |= is_eos
        alive &= ~is_eos
        inp = torch.as_tensor(y).unsqueeze(1)
    reward = (ended & (k >= 1) & (s == answer)).astype(np.float64)
    return dict(c=c, y=tokens, act=act, real=real, r=reward, n=n, calls=calls, fab=fab, late_calls=late_calls,
                carry_steps=carry_steps, carry_calls=carry_calls, truncated=~ended,
                length=real.sum(1).numpy(), n_act=act.sum(1).numpy())


# ---------------------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------------------
def train_base(steps, batch, g, hidden, log_every=0):
    model = PolicyLM(VOCAB, COND_DIM, emb=32, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        dg = make_prompts(batch, g)
        toks, act = demos(dg, g)
        loss = -token_logprobs(model, cond_of(dg), toks, act).sum() / batch  # tool outputs masked
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
# RL
# ---------------------------------------------------------------------------------------
def run_rl(name, base, iters, g, cfg, eval_set, log_every=0):
    knobs = METHODS[name]
    lag = cfg["lag"] if knobs["lag"] is None else knobs["lag"]
    policy = copy.deepcopy(base)
    for p in policy.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])
    sampler = copy.deepcopy(base)
    snapshots = collections.deque([copy.deepcopy(policy.state_dict())], maxlen=lag + 1)
    G, P = cfg["G"], cfg["prompts_per_batch"]
    curve, n_samples, w_hist = [], 0, []
    for it in range(iters):
        # ---- rollouts, from the current policy or from one `lag` iterations old ----
        sampler.load_state_dict(snapshots[0])     # oldest snapshot kept = lag iterations ago
        dg = np.repeat(make_prompts(P, g), G, axis=0)
        b = rollout(sampler, dg)
        n_samples += len(dg)
        c, y, act, real = b["c"], b["y"], b["act"], b["real"]
        mask = act if knobs["mask_obs"] else real
        rg = torch.as_tensor(b["r"], dtype=torch.float).view(-1, G)
        with torch.no_grad():
            logp_old = token_logprobs(policy, c, y, real)   # the trainer's policy before this update
            if knobs["tis"]:
                logp_samp = token_logprobs(sampler, c, y, real)
                lw = (logp_old - logp_samp) * act        # log pi_old / pi_sampler on policy tokens
                w_all = torch.clamp(torch.exp(lw), max=TIS_C)
                w_hist.append((float(((lw > np.log(TIS_C)) * act).sum() / act.sum()),
                               float(lw.abs().sum() / act.sum())))
            if knobs["loss"] == "sft":
                adv = rg
            else:   # GRPO: one standardized advantage per trajectory (eq. 18.25)
                adv = (rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True, unbiased=False) + 1e-4)
            adv = adv.reshape(-1, 1) * mask
        for _ in range(cfg["epochs"]):
            for chunk in np.array_split(g.permutation(P), cfg["minibatches"]):
                idx = torch.as_tensor(np.concatenate([np.arange(i * G, (i + 1) * G) for i in chunk]))
                logp = token_logprobs(policy, c[idx], y[idx], real[idx])
                mi, ai = mask[idx], adv[idx]
                if knobs["loss"] == "sft":
                    # filtered SFT: log-likelihood of the policy tokens of successful trajectories
                    loss = (-(ai * logp) * mi).sum(1).mean()
                else:
                    ratio = torch.exp(logp - logp_old[idx])
                    surr = torch.min(ratio * ai, torch.clamp(ratio, 0.8, 1.2) * ai)
                    if knobs["tis"]:   # truncated importance weight pi_old / pi_sampler, no gradient
                        surr = w_all[idx] * surr
                    loss = ((-surr * mi).sum(1) / mi.sum(1).clamp(min=1.0)).mean()   # eq. 18.35
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
        snapshots.append(copy.deepcopy(policy.state_dict()))
        if it % cfg["eval_every"] == 0 or it == iters - 1:
            ev = evaluate(policy, eval_set, cfg["eval_samples"])
            ev.update(it=it, samples=n_samples)
            curve.append(ev)
            if log_every and (it % log_every == 0 or it == iters - 1):
                print(f"    {name:<20} it {it:4d}  success {ev['success']:.3f}  calls {ev['calls']:.2f}  "
                      f"tokens {ev['length']:.2f}  fabricated {ev['fab']:.3f}  "
                      f"log pi(tool output) {ev['obs_logp']:.2f}")
    extra = dict(tis_truncated=float(np.mean([w[0] for w in w_hist])) if w_hist else np.nan,
                 tis_abs_logw=float(np.mean([w[1] for w in w_hist])) if w_hist else np.nan)
    return policy, curve, extra


@torch.no_grad()
def evaluate(policy, eval_set, n_samples):
    dg = np.repeat(eval_set, n_samples, axis=0)
    b = rollout(policy, dg)
    obs = b["real"] - b["act"]
    lp_all = F.log_softmax(policy.logits(b["c"], b["y"]), dim=-1)
    logp = lp_all.gather(2, b["y"].unsqueeze(-1)).squeeze(-1) * b["real"]
    # the policy's next-token distribution at its first decision after a tool output
    after = torch.zeros_like(obs)
    after[:, 1:] = obs[:, :-1]
    after = (after * b["act"]) > 0
    p_after = lp_all[after].exp().mean(0) if after.any() else torch.full((VOCAB,), float("nan"))
    out = dict(success=b["r"].mean(), calls=b["calls"].mean(), length=b["length"].mean(),
               late_calls=b["late_calls"].mean(), p_call_after=float(p_after[CALL]),
               p_eos_after=float(p_after[EOS]),
               n_act=b["n_act"].mean(), fab=(b["fab"] > 0).mean(), truncated=b["truncated"].mean(),
               delegated=b["carry_calls"].sum() / max(b["carry_steps"].sum(), 1),
               at_carry=b["carry_calls"].sum() / max(b["calls"].sum(), 1),
               obs_logp=float((logp * obs).sum() / obs.sum().clamp(min=1.0)))
    for n in range(N_MIN, N_MAX + 1):
        sel = b["n"] == n
        out[f"success_n{n}"] = b["r"][sel].mean()
        out[f"calls_n{n}"] = b["calls"][sel].mean()
    return out


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
    cfg = dict(base_steps=300 if quick else 1500, base_batch=256, hidden=128, G=8,
               prompts_per_batch=16 if quick else 32, lr=1e-3, epochs=2, minibatches=2,
               iters=10 if quick else 200, eval_every=5 if quick else 20, lag=8,
               n_eval_per_size=20 if quick else 100, eval_samples=4, seeds=1 if quick else 3)
    methods = list(METHODS)
    print(f"seed={args.seed} quick={quick}")
    print("config:", cfg)
    print(f"token budget T_MAX={T_MAX}; demonstrator: carry bug {P_WRAP}, slips {P_SLIP}, "
          f"random tool calls {Q_CALL}; TIS truncation C={TIS_C}")

    print("\n[1] Base model: maximum likelihood on demonstrations (tool outputs masked out of the loss)")
    t0 = time.time()
    base = train_base(cfg["base_steps"], cfg["base_batch"], g, cfg["hidden"],
                      log_every=cfg["base_steps"] // 3)
    ge = np.random.default_rng(12345)
    eval_set = np.concatenate([make_prompts(cfg["n_eval_per_size"], ge, [n] * cfg["n_eval_per_size"])
                               for n in range(N_MIN, N_MAX + 1)])
    ev0 = evaluate(base, eval_set, 16)
    print(f"  base: success {ev0['success']:.3f}, by n=2..6 "
          + " ".join(f"{ev0[f'success_n{n}']:.2f}" for n in range(N_MIN, N_MAX + 1))
          + f"; calls/episode {ev0['calls']:.2f}; carry steps delegated {ev0['delegated']:.2f}; calls at carry "
            f"steps {ev0['at_carry']:.2f}; tokens {ev0['length']:.2f}; fabricated {ev0['fab']:.3f}; "
            f"truncated {ev0['truncated']:.3f}; log pi(tool output) {ev0['obs_logp']:.2f}  [{time.time() - t0:.1f}s]")

    print("\n[2] RL with the outcome reward")
    curves, extras = {m: [] for m in methods}, {m: [] for m in methods}
    for m in methods:
        for s in range(cfg["seeds"]):
            t0 = time.time()
            torch.manual_seed(1000 + s)
            gs = np.random.default_rng(1000 + s)
            _, curve, extra = run_rl(m, base, cfg["iters"], gs, cfg, eval_set,
                                     log_every=(cfg["iters"] // 2) if s == 0 else 0)
            curves[m].append(curve)
            extras[m].append(extra)
            f = curve[-1]
            print(f"  {m:<20} seed {s}: success {f['success']:.3f}  calls {f['calls']:.2f}  tokens "
                  f"{f['length']:.2f}  delegated {f['delegated']:.2f}  at-carry {f['at_carry']:.2f}  "
                  f"fabricated {f['fab']:.3f}  truncated {f['truncated']:.3f}  log pi(tool output) "
                  f"{f['obs_logp']:.2f}  [{time.time() - t0:.1f}s]")

    print("\nSummary (final iterate, mean over seeds; success = sampled at temperature 1)")
    keys = ["success", "calls", "length", "n_act", "delegated", "at_carry", "late_calls", "fab", "truncated",
            "obs_logp", "p_call_after", "p_eos_after"]
    heads = ["success", "calls", "tokens", "own tok", "deleg.", "at-carry", "late", "fabric.", "trunc.", "logp(o)",
             "CALL|o", "EOS|o"]
    print(f"  {'method':<20} {'(min-max)':>13} " + " ".join(f"{h:>8}" for h in heads))
    print(f"  {'base':<20} {'':>13} " + " ".join(f"{ev0[k]:>8.3f}" for k in keys))
    for m in methods:
        fs = [c[-1] for c in curves[m]]
        acc = [f["success"] for f in fs]
        print(f"  {m:<20} {f'({min(acc):.3f}-{max(acc):.3f})':>13} "
              + " ".join(f"{np.mean([f[k] for f in fs]):>8.3f}" for k in keys))
    print("  (deleg. = fraction of carry steps handed to the tool; at-carry = fraction of calls made at carry "
          "steps; late = calls per episode after the last digit; fabric. = episodes with a tool output written "
          "by the policy; logp(o) = mean log-prob the policy gives the actual tool outputs; CALL|o, EOS|o = its "
          "probability of CALL / EOS at the first decision after a tool output)")
    print("  success by n = 2..6 | calls by n = 2..6 (mean over seeds):")
    print(f"  {'base':<20} " + " ".join(f"{ev0[f'success_n{n}']:.3f}" for n in range(N_MIN, N_MAX + 1))
          + " | " + " ".join(f"{ev0[f'calls_n{n}']:.2f}" for n in range(N_MIN, N_MAX + 1)))
    for m in methods:
        fs = [c[-1] for c in curves[m]]
        print(f"  {m:<20} " + " ".join(f"{np.mean([f[f'success_n{n}'] for f in fs]):.3f}" for n in range(N_MIN, N_MAX + 1))
              + " | " + " ".join(f"{np.mean([f[f'calls_n{n}'] for f in fs]):.2f}" for n in range(N_MIN, N_MAX + 1)))
    for m in methods:
        if METHODS[m]["tis"]:
            print(f"  {m}: policy tokens with pi_old/pi_sampler > C={TIS_C} (truncated): "
                  f"{np.mean([e['tis_truncated'] for e in extras[m]]):.4f}; mean |log pi_old/pi_sampler| "
                  f"{np.mean([e['tis_abs_logw'] for e in extras[m]]):.3f} nats per token")
    mid = len(curves[methods[0]][0]) // 2
    print(f"  success at iteration {curves[methods[0]][0][mid]['it']} (mean over seeds): "
          + ", ".join(f"{m} {np.mean([c[mid]['success'] for c in curves[m]]):.3f}" for m in methods))
    print(f"\nTotal time {time.time() - t_start:.1f}s")
    if quick:
        return
    make_figure(curves, ev0)


def make_figure(curves, ev0):
    os.makedirs(FIG_DIR, exist_ok=True)
    colors = {"GRPO (masked)": "tab:blue", "Filtered SFT": "tab:green", "GRPO, unmasked": "tab:red",
              "GRPO, stale sampler": "tab:orange", "GRPO, stale + TIS": "tab:purple"}
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for m, cl in curves.items():
        xs = np.array([[e["samples"] for e in c] for c in cl]).mean(0) / 1000.0
        for ax, key in zip(axes, ["success", "calls", "obs_logp", "truncated"]):
            vals = np.array([[e[key] for e in c] for c in cl])
            ax.plot(xs, vals.mean(0), color=colors[m], label=m)
            if len(cl) > 1:
                ax.fill_between(xs, vals.min(0), vals.max(0), color=colors[m], alpha=0.15)
    for ax, key in zip(axes, ["success", "calls", "obs_logp", "truncated"]):
        ax.axhline(ev0[key], color="k", ls=":", lw=1, label="base" if key == "success" else None)
    for ax, yl, t in zip(axes, ["success rate (sampled)", "tool calls per episode",
                                "mean log-prob of the actual tool outputs", "fraction of episodes truncated"],
                         ["Outcome", "Tool use", "What the policy predicts for tool outputs",
                          "Episodes that run out of budget"]):
        ax.set_xlabel("trajectories sampled (thousands)")
        ax.set_ylabel(yl)
        ax.set_title(t)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "multiturn_tool_toy.png"), dpi=110)
    plt.close(fig)
    print("figure written to", FIG_DIR)


if __name__ == "__main__":
    main()
