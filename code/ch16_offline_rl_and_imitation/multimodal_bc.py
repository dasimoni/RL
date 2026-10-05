"""Behaviour cloning with expressive policy classes on a bimodal obstacle task (Chapter 16, Section 2.8).

Task.  A point at s = (x, y) starts at (x0, 0) with x0 ~ U(-0.02, 0.02) and must reach y >= 1,
within |x| <= 0.15 of the centre line, without touching a disk of radius 0.2 centred at
(0, 0.55).  Each step it moves by its action a = (dx, dy), |dx|, |dy| <= 0.12, plus Gaussian
noise (sd 0.004 per coordinate).  An episode fails if it collides, leaves |x| <= 0.7, misses the
gate, or runs out of its 40 steps.

Expert.  At the start of each episode it flips a fair coin c in {-1, +1}, which the learner never
sees.  It moves up by 0.05 per step and steers towards its lateral path at the next height:
straight up at x = x0 until y = 0.3, then a detour of 0.3 * c over three steps (0.1 per step),
then straight on, and back to x = 0 between y = 0.75 and y = 0.95.  So it passes left or right
with probability 1/2, and at the decision state (0, 0.3), just below the obstacle, its action is
dx = -0.1 or +0.1 with equal probability: the average action, dx = 0, leads into the obstacle.

Learners.  All see only the state (x, y), encoded as (x, y) plus sin/cos features at 5 frequencies,
are MLPs with two hidden layers of 128 units, and are trained by maximum likelihood (or, for
DDPM, its denoising analogue) on the same demonstrations, with Adam (learning rate 3e-3, batch 256)
and a cosine learning-rate decay: 4,000 gradient steps, or 16,000 for the two DDPMs.
  MSE          regression of a on s: Gaussian BC with a fixed variance, acting at its mean (Eq. 16.1)
  Gaussian     Gaussian with learned mean and diagonal variance; actions sampled
  MDN M=2, M=5 mixture density network (Bishop, 1994) with M diagonal-Gaussian components; sampled
  binned       each action dimension discretised into 41 bins with one softmax per dimension
               (the RT-1 / RT-2 recipe, which uses 256 bins); bins sampled
  DDPM         diffusion policy: noise-prediction network, K = 20 denoising steps, ancestral
               sampling (Eqs. 16.43-16.45, Algorithm 16.18)
  DDPM chunk4  the same, predicting a chunk of 4 future actions (an 8-dimensional "action")
               that is executed open loop before the policy is queried again
Two equal-budget controls (table only, not in the figure), because the DDPMs get 4x more steps:
  DDPM 4k steps      the DDPM trained for 4,000 steps, like the other learners
  MDN M=5 16k steps  the five-component MDN trained for 16,000 steps, like the DDPMs

Each seed draws a new set of 200 demonstrations and new initialisations; each policy is then
rolled out from 500 fresh start states.  We report collision and success rates, the other
failures (missed the gate, left the corridor, ran out of time), the side on which successful
rollouts passed the obstacle, and how often the policy was queried per episode.  A probe at the
decision state (0, 0.3) shows each policy's distribution of dx there.  The training loss printed
for each learner is its loss on the whole dataset at the end of training.

Run:  python code/ch16_offline_rl_and_imitation/multimodal_bc.py [--quick]
Output (full mode): figures/multimodal_bc.png
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

V = 0.05                        # expert's forward speed per step
A_MAX = 0.12                    # action bound per coordinate; networks see a / A_MAX
NOISE = 0.004                   # sd of the transition noise per coordinate
OBST = np.array([0.0, 0.55])    # obstacle centre
RADIUS = 0.2
X0_HALF = 0.02                  # start x0 ~ U(-X0_HALF, X0_HALF)
Y_DECIDE = 0.3                  # where the expert's detour starts
DETOUR = 0.3                    # lateral size of the detour, reached in 3 steps
Y_RETURN, Y_BACK = 0.75, 0.95   # the expert heads back to x = 0 between these heights
GATE = 0.15                     # success: reach y >= Y_GOAL with |x| <= GATE
X_LIMIT = 0.7                   # leaving |x| <= X_LIMIT ends the episode as a failure
Y_GOAL = 1.0
T_MAX = 40
PROBE = np.array([0.0, Y_DECIDE])   # the decision state, where both of the expert's modes occur

N_BINS = 41                     # binned policy: bins per action dimension
K_DIFF = 20                     # DDPM: number of denoising steps
HID = 128


# ------------------------------------------------------------------------------------ task
def expert_target(y, x0, c):
    """The expert's lateral path x*(y) for start x0 and side c in {-1, +1}."""
    ramp = np.clip((y - Y_DECIDE) / (3 * V), 0.0, 1.0)          # 0 -> 1 over three steps
    back = 1.0 - np.clip((y - Y_RETURN) / (Y_BACK - Y_RETURN), 0.0, 1.0)
    return (x0 + c * DETOUR * ramp) * back


def expert_action(p, x0, c):
    dx = np.clip(expert_target(p[:, 1] + V, x0, c) - p[:, 0], -A_MAX, A_MAX)
    return np.stack([dx, np.full_like(dx, V)], 1)


def step(p, a, rng):
    """One transition; returns the new position and whether the segment touched the obstacle."""
    p_new = p + a + NOISE * rng.standard_normal(p.shape)
    hit = np.zeros(len(p), bool)
    for f in (0.25, 0.5, 0.75, 1.0):          # check points along the segment, not just its end
        hit |= np.linalg.norm(p + f * (p_new - p) - OBST, axis=1) < RADIUS
    return p_new, hit


def rollout(act_fn, x0, rng, chunk=1):
    """Run len(x0) episodes in lockstep.  act_fn(states [n, 2]) -> actions [n, chunk, 2].
    Outcomes: collided, success (y >= Y_GOAL with |x| <= GATE), or failed otherwise (missed the
    gate, left |x| <= X_LIMIT, or ran out of time)."""
    n = len(x0)
    p = np.stack([x0, np.zeros(n)], 1)
    alive = np.ones(n, bool)
    collided = np.zeros(n, bool)
    success = np.zeros(n, bool)
    missed_gate = np.zeros(n, bool)
    left_corridor = np.zeros(n, bool)
    side = np.zeros(n)                        # sign of x when crossing the obstacle's centre line
    queries = np.zeros(n, int)
    traj = [p.copy()]
    plan, k = None, chunk
    for _ in range(T_MAX):
        if k == chunk:                        # query the policy for a new chunk of actions
            plan, k = act_fn(p), 0
            queries += alive
        a = np.clip(plan[:, k], -A_MAX, A_MAX)
        k += 1
        p_new, hit = step(p, a, rng)
        crossed = alive & (p[:, 1] < OBST[1]) & (p_new[:, 1] >= OBST[1])
        side[crossed] = np.sign(p_new[crossed, 0])
        collided |= alive & hit
        done = alive & ~hit & ((p_new[:, 1] >= Y_GOAL) | (np.abs(p_new[:, 0]) > X_LIMIT))
        success |= done & (p_new[:, 1] >= Y_GOAL) & (np.abs(p_new[:, 0]) <= GATE)
        missed_gate |= done & (p_new[:, 1] >= Y_GOAL) & (np.abs(p_new[:, 0]) > GATE)
        left_corridor |= done & (p_new[:, 1] < Y_GOAL)          # left |x| <= X_LIMIT first
        p = np.where(alive[:, None], p_new, p)
        alive &= ~collided & ~done
        traj.append(p.copy())
        if not alive.any():
            break
    timeout = alive                           # still going after T_MAX steps
    return dict(collided=collided, success=success, side=side, queries=queries,
                missed_gate=missed_gate, left_corridor=left_corridor, timeout=timeout,
                traj=np.stack(traj, 1))


def make_demos(n_ep, rng, chunk=4):
    """Expert episodes.  Returns states [N, 2], actions [N, 2] and action chunks [N, chunk, 2]
    (the next `chunk` expert actions, padded by repeating the episode's last action), the
    expert's sides, and the expert's collision rate."""
    x0 = rng.uniform(-X0_HALF, X0_HALF, n_ep)
    c = rng.choice([-1.0, 1.0], n_ep)
    p = np.stack([x0, np.zeros(n_ep)], 1)
    alive = np.ones(n_ep, bool)
    collided = np.zeros(n_ep, bool)
    S, A, M = [], [], []
    for _ in range(T_MAX):
        a = expert_action(p, x0, c)
        S.append(p.copy()); A.append(a); M.append(alive.copy())
        p_new, hit = step(p, a, rng)
        collided |= alive & hit
        p = np.where(alive[:, None], p_new, p)
        alive &= (p[:, 1] < Y_GOAL) & ~collided
        if not alive.any():
            break
    S, A, M = np.stack(S), np.stack(A), np.stack(M)          # [T, n, 2], [T, n, 2], [T, n]
    lengths = M.sum(0)
    idx = np.arange(len(M))[:, None] + np.arange(chunk)[None, :]
    states, acts, chunks = [], [], []
    for i in range(n_ep):
        L = lengths[i]
        j = np.minimum(idx[:L], L - 1)
        states.append(S[:L, i]); acts.append(A[:L, i]); chunks.append(A[j, i])
    return (np.concatenate(states), np.concatenate(acts), np.concatenate(chunks), c,
            collided.mean())


# ---------------------------------------------------------------------------------- models
def mlp(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, HID), nn.SiLU(), nn.Linear(HID, HID), nn.SiLU(),
                         nn.Linear(HID, n_out))


FREQS = 2.0 ** torch.arange(5)          # Fourier features of the state: frequencies 1..16
N_FEAT = 2 + 4 * len(FREQS)


def norm_s(s):
    """State features: (x, y) scaled to about [-1, 1], plus sin/cos of each at 5 frequencies,
    so that a small MLP can represent the sharp change of the expert's action at the decision
    height (the same trick NeRF-style networks use for low-dimensional inputs)."""
    z = torch.stack([s[:, 0] / 0.5, (s[:, 1] - 0.5) / 0.5], 1)
    w = (z[:, :, None] * FREQS * math.pi / 2).reshape(len(s), -1)
    return torch.cat([z, torch.sin(w), torch.cos(w)], 1)


class MSEPolicy:
    name = "MSE"

    def __init__(self):
        self.net = mlp(N_FEAT, 2)

    def loss(self, s, a):
        return (self.net(norm_s(s)) - a).pow(2).sum(1).mean()

    @torch.no_grad()
    def sample(self, s):
        return self.net(norm_s(s)).unsqueeze(1)


class GaussianPolicy:
    name = "Gaussian"

    def __init__(self):
        self.net = mlp(N_FEAT, 4)

    def dist(self, s):
        out = self.net(norm_s(s))
        return out[:, :2], out[:, 2:].clamp(-5.0, 1.0)

    def loss(self, s, a):
        mu, log_sd = self.dist(s)
        return (0.5 * ((a - mu) / log_sd.exp()).pow(2) + log_sd).sum(1).mean()

    @torch.no_grad()
    def sample(self, s):
        mu, log_sd = self.dist(s)
        return (mu + log_sd.exp() * torch.randn_like(mu)).unsqueeze(1)


class MDNPolicy:
    """Mixture density network (Bishop, 1994): K diagonal Gaussians with state-dependent weights."""
    def __init__(self, k=2):
        self.k = k
        self.name = f"MDN M={k}"
        self.net = mlp(N_FEAT, k + 4 * k)

    def dist(self, s):
        out = self.net(norm_s(s))
        k = self.k
        logits = out[:, :k]
        mu = out[:, k:3 * k].view(-1, k, 2)
        log_sd = out[:, 3 * k:].view(-1, k, 2).clamp(-5.0, 1.0)
        return logits, mu, log_sd

    def loss(self, s, a):
        logits, mu, log_sd = self.dist(s)
        comp = -(0.5 * ((a[:, None] - mu) / log_sd.exp()).pow(2) + log_sd).sum(2)    # [B, k]
        return -(torch.logsumexp(F.log_softmax(logits, 1) + comp, 1)).mean()

    @torch.no_grad()
    def sample(self, s):
        logits, mu, log_sd = self.dist(s)
        j = torch.distributions.Categorical(logits=logits).sample()
        r = torch.arange(len(j))
        return (mu[r, j] + log_sd[r, j].exp() * torch.randn_like(mu[r, j])).unsqueeze(1)


class BinnedPolicy:
    """Each action dimension discretised into N_BINS bins on [-1, 1], one softmax per dimension."""
    name = "binned"

    def __init__(self):
        self.net = mlp(N_FEAT, 2 * N_BINS)
        self.centres = torch.linspace(-1, 1, N_BINS)

    def loss(self, s, a):
        logits = self.net(norm_s(s)).view(-1, 2, N_BINS)
        target = ((a.clamp(-1, 1) + 1) / 2 * (N_BINS - 1)).round().long()            # [B, 2]
        return F.cross_entropy(logits.reshape(-1, N_BINS), target.reshape(-1)) * 2

    @torch.no_grad()
    def sample(self, s):
        logits = self.net(norm_s(s)).view(-1, 2, N_BINS)
        j = torch.distributions.Categorical(logits=logits).sample()                    # [n, 2]
        return self.centres[j].unsqueeze(1)


def cosine_schedule(K, s=0.008):
    """Nichol & Dhariwal's cosine schedule: abar_k for k = 0..K (abar_0 = 1)."""
    k = torch.arange(K + 1, dtype=torch.float64)
    f = torch.cos((k / K + s) / (1 + s) * math.pi / 2) ** 2
    abar = (f / f[0]).clamp(min=1e-5)
    beta = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
    abar = torch.cat([torch.ones(1, dtype=torch.float64), torch.cumprod(1 - beta, 0)])
    return beta.float(), abar.float()


class DDPMPolicy:
    """Diffusion policy (Chi et al., 2023) in its simplest form: an MLP eps_theta(a^k, k, s) trained
    with the noise-prediction loss (16.44) and sampled by DDPM ancestral sampling (16.45).
    With chunk > 1 the "action" is the next `chunk` actions stacked (dimension 2 * chunk)."""

    def __init__(self, chunk=1, K=K_DIFF):
        self.chunk, self.K = chunk, K
        self.d = 2 * chunk
        self.name = "DDPM" if chunk == 1 else f"DDPM chunk{chunk}"
        self.net = mlp(N_FEAT + self.d + 8, self.d)
        self.beta, self.abar = cosine_schedule(K)           # beta[k-1] = beta_k, abar[k] = abar_k
        self.freqs = torch.tensor([1.0, 2.0, 4.0, 8.0])

    def emb(self, k):
        x = (k.float() / self.K)[:, None] * self.freqs * math.pi
        return torch.cat([torch.sin(x), torch.cos(x)], 1)

    def eps(self, a_k, k, s):
        return self.net(torch.cat([norm_s(s), a_k, self.emb(k)], 1))

    def loss(self, s, a):
        a0 = a.reshape(len(a), -1)
        k = torch.randint(1, self.K + 1, (len(a0),))
        ab = self.abar[k][:, None]
        noise = torch.randn_like(a0)
        a_k = ab.sqrt() * a0 + (1 - ab).sqrt() * noise                                  # (16.43)
        return (noise - self.eps(a_k, k, s)).pow(2).sum(1).mean()                        # (16.44)

    @torch.no_grad()
    def sample(self, s):
        n = len(s)
        a = torch.randn(n, self.d)
        for k in range(self.K, 0, -1):
            kk = torch.full((n,), k)
            ab, ab_prev, b = self.abar[k], self.abar[k - 1], self.beta[k - 1]
            a0_hat = ((a - (1 - ab).sqrt() * self.eps(a, kk, s)) / ab.sqrt()).clamp(-1, 1)
            # posterior q(a^{k-1} | a^k, a^0 = a0_hat): this is (16.45) with the a^0 estimate clipped
            mean = (ab_prev.sqrt() * b / (1 - ab)) * a0_hat + ((1 - b).sqrt() * (1 - ab_prev) / (1 - ab)) * a
            if k > 1:
                var = b * (1 - ab_prev) / (1 - ab)
                a = mean + var.sqrt() * torch.randn_like(a)
            else:
                a = mean
        return a.view(n, self.chunk, 2)


def train(policy, S, A, steps, batch, lr, gen):
    opt = torch.optim.Adam(policy.net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    n = len(S)
    for _ in range(steps):
        i = torch.randint(0, n, (batch,), generator=gen)
        loss = policy.loss(S[i], A[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return float(loss.detach())


def as_act_fn(policy):
    def act(p):
        return policy.sample(torch.as_tensor(p, dtype=torch.float32)).numpy() * A_MAX
    return act


# ------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    quick = args.quick
    seeds = [0] if quick else [0, 1, 2]
    n_demo = 200
    n_eval = 100 if quick else 500
    steps = 300 if quick else 4000
    steps_ddpm = 600 if quick else 16000
    batch, lr = 256, 3e-3
    print(f"multimodal_bc: seeds={seeds} quick={quick} demos={n_demo} eval rollouts={n_eval} "
          f"steps={steps} (DDPM {steps_ddpm}) batch={batch} lr={lr} K={K_DIFF} bins={N_BINS}")
    t0 = time.time()

    names = ["expert", "MSE", "Gaussian", "MDN M=2", "MDN M=5", "binned", "DDPM", "DDPM chunk4"]
    kfmt = lambda n: f"{n // 1000}k" if n >= 1000 else str(n)
    controls = [f"DDPM {kfmt(steps)} steps", f"MDN M=5 {kfmt(steps_ddpm)} steps"]   # equal-budget controls
    res = {m: [] for m in names + controls}
    probe = {m: [] for m in names + controls}
    keep_traj = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        S, A, Ach, c, expert_coll = make_demos(n_demo, rng)
        St = torch.as_tensor(S, dtype=torch.float32)
        At = torch.as_tensor(A / A_MAX, dtype=torch.float32)
        Acht = torch.as_tensor(Ach / A_MAX, dtype=torch.float32)
        near = (np.abs(S[:, 0] - PROBE[0]) < 0.03) & (np.abs(S[:, 1] - PROBE[1]) < 0.04)
        probe["expert"].append(A[near, 0])
        print(f"\nseed {seed}: {len(S)} (s, a) pairs from {n_demo} demonstrations "
              f"({np.mean(c < 0):.0%} pass left; expert collisions {expert_coll:.1%}); "
              f"{near.sum()} pairs near the probe state, of which {np.mean(A[near, 0] < 0):.0%} steer left")

        policies = [MSEPolicy(), GaussianPolicy(), MDNPolicy(2), MDNPolicy(5), BinnedPolicy(), DDPMPolicy(1),
                    DDPMPolicy(4)]
        x0_eval = np.random.default_rng(1000 + seed).uniform(-X0_HALF, X0_HALF, n_eval)
        # expert rollouts from the same starts (fresh coins)
        c_eval = np.random.default_rng(2000 + seed).choice([-1.0, 1.0], n_eval)
        out = rollout(lambda p: expert_action(p, x0_eval, c_eval)[:, None], x0_eval,
                      np.random.default_rng(3000 + seed))
        res["expert"].append(out)
        keep_traj.setdefault("expert", out)

        def make_controls():
            # built only after the main learners have been trained, so that adding them leaves the
            # main learners' random streams (and results) unchanged
            short, long_ = DDPMPolicy(1), MDNPolicy(5)
            short.name, short.n_steps = controls[0], steps
            long_.name, long_.n_steps = controls[1], steps_ddpm
            return [short, long_]

        queue = list(policies)
        while queue:
            pol = queue.pop(0)
            t1 = time.time()
            chunked = pol.name == "DDPM chunk4"
            n_steps = getattr(pol, "n_steps", steps_ddpm if pol.name.startswith("DDPM") else steps)
            data = Acht if chunked else At
            train(pol, St, data, n_steps, batch, lr, gen)
            with torch.no_grad(), torch.random.fork_rng(devices=[]):     # DDPM's loss draws noise
                full_loss = float(pol.loss(St, data))
            t_train = time.time() - t1
            t1 = time.time()
            out = rollout(as_act_fn(pol), x0_eval, np.random.default_rng(3000 + seed),
                          chunk=4 if chunked else 1)
            res[pol.name].append(out)
            keep_traj.setdefault(pol.name, out)
            ps = torch.as_tensor(np.tile(PROBE, (4000, 1)), dtype=torch.float32)
            probe[pol.name].append(pol.sample(ps)[:, 0, 0].numpy() * A_MAX)
            print(f"  {pol.name:17s} steps {n_steps:5d}  loss on all data {full_loss:8.4f}  "
                  f"train {t_train:5.1f}s  rollouts {time.time() - t1:4.1f}s  "
                  f"collisions {out['collided'].mean():6.1%}  success {out['success'].mean():6.1%}")
            if isinstance(pol, MDNPolicy):        # what the mixture looks like at the decision state
                with torch.no_grad():
                    logits, mu, log_sd = pol.dist(torch.as_tensor(PROBE[None], dtype=torch.float32))
                w = torch.softmax(logits, 1)[0].numpy()
                order = np.argsort(-w)
                fmt3 = lambda v: "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"
                print(f"               at the decision state: weights "
                      f"[{', '.join(f'{x:.2f}' for x in w[order])}], "
                      f"means of dx {fmt3(mu[0, order, 0].numpy() * A_MAX)}, "
                      f"sds of dx {fmt3(log_sd[0, order, 0].exp().numpy() * A_MAX)}")
            if pol is policies[-1]:
                queue += make_controls()

    print(f"\nsummary over {len(seeds)} seed(s) x {n_eval} rollouts "
          f"(mean, and range over seeds in brackets when there are several); 'other' failures are "
          f"missed gate / left the corridor / out of time")
    print(f"{'policy':17s} {'collision':>21s} {'success':>23s} {'other: gate/out/time':>21s} "
          f"{'left|success':>13s} {'queries/episode':>16s} {'probe: P(dx<0)':>15s} {'probe: P(|dx|<0.05)':>20s}")

    def fmt(v):
        v = np.array(v)
        return f"{v.mean():6.1%}" + (f" [{v.min():.1%}-{v.max():.1%}]" if len(v) > 1 else "")

    summary = {}
    for m in names + controls:
        if m == controls[0]:
            print("equal-budget controls:")
        coll = [o["collided"].mean() for o in res[m]]
        succ = [o["success"].mean() for o in res[m]]
        other = "/".join(f"{np.mean([o[key].mean() for o in res[m]]):.1%}"
                         for key in ("missed_gate", "left_corridor", "timeout"))
        left = [np.mean(o["side"][o["success"]] < 0) if o["success"].any() else np.nan for o in res[m]]
        q = [o["queries"].mean() for o in res[m]]
        pr = np.concatenate(probe[m])
        summary[m] = (np.mean(coll), np.mean(succ), np.nanmean(left))
        print(f"{m:17s} {fmt(coll):>21s} {fmt(succ):>23s} {other:>21s} {np.nanmean(left):13.0%} "
              f"{np.mean(q):16.1f} {np.mean(pr < 0):15.0%} {np.mean(np.abs(pr) < 0.05):20.0%}")
    print("MSE's (deterministic) dx at the decision state, per seed: "
          + ", ".join(f"{float(v[0]):+.3f}" for v in probe["MSE"])
          + "   (the mean of the demonstrated modes; it shifts when one side is demonstrated more often)")
    print(f"total time {time.time() - t0:.1f}s")

    if quick:
        return
    import plot_style
    plt = plot_style.setup()
    C = plot_style.C
    fig, axes = plt.subplots(3, 3, figsize=(12, 12.5))
    panels = names
    for ax, m in zip(axes.flat, panels):
        out = keep_traj[m]
        tr = out["traj"]
        for i in range(60):
            if out["collided"][i]:
                col, ls = C[7], "--"
            elif out["success"][i]:
                col, ls = (C[0] if out["side"][i] < 0 else C[2]), "-"
            else:
                col, ls = plot_style.GREY, ":"
            ax.plot(tr[i, :, 0], tr[i, :, 1], color=col, ls=ls, lw=1.0, alpha=0.8)
            if out["collided"][i]:
                ax.plot(tr[i, -1, 0], tr[i, -1, 1], "x", color=C[7], ms=5)
        ax.add_patch(plt.Circle(OBST, RADIUS, color="#b9b8b3", alpha=0.6, lw=0))
        ax.plot([-GATE, GATE], [Y_GOAL, Y_GOAL], color=plot_style.INK, lw=2.0)     # the goal gate
        coll, succ, left = summary[m]
        ax.set_title(f"{m}: {out['collided'].mean():.0%} collide in this seed\n"
                     f"({coll:.0%} collide, {succ:.0%} succeed over all seeds)", fontsize=10)
        ax.set_xlim(-0.65, 0.65); ax.set_ylim(-0.05, 1.08)
        ax.set_aspect("equal")
        ax.set_xticks([-0.5, 0, 0.5]); ax.set_yticks([0, 0.5, 1])
    ax = axes.flat[-1]
    bins = np.linspace(-A_MAX, A_MAX, 41)
    styles = {"expert": (plot_style.INK, "-"), "MSE": (C[7], "-"), "Gaussian": (C[1], "--"),
              "MDN M=2": (C[4], "--"), "MDN M=5": (C[0], "-"), "binned": (C[3], ":"), "DDPM": (C[2], "-"), "DDPM chunk4": (C[6], "--")}
    for m in names:
        pr = np.concatenate(probe[m])
        h, e = np.histogram(pr, bins=bins, density=True)
        ax.step(0.5 * (e[1:] + e[:-1]), h, where="mid", color=styles[m][0], ls=styles[m][1], lw=1.4, label=m)
    ax.set_xlabel(rf"$dx$ at the decision state $({PROBE[0]:g}, {PROBE[1]:g})$")
    ax.set_ylabel("density")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title("Action distributions at the decision state", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"60 rollouts of each policy (first seed): passed left (blue), right (green), collided "
                 f"(red, dashed, x), other failure (grey, dotted).\nAll-seed rates are over "
                 f"{len(seeds)} seeds x {n_eval} rollouts; the black bar is the goal gate.", fontsize=10.5)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "multimodal_bc.png"))
    plt.close(fig)
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
