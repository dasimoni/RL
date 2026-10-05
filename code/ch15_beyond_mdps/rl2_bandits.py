"""RL^2 on two-armed Bernoulli bandits: a recurrent policy that learns a bandit algorithm.

Chapter 15, Section 7 (Algorithm 15.10).  Meta-RL in its "black-box" form (Duan et al., 2016;
Wang et al., 2016): a GRU receives (previous action, previous reward, time) and keeps its hidden
state for the whole trial of H = 20 pulls on ONE task; across tasks the hidden state is reset.
Its weights are trained by policy gradient (A2C) on many tasks drawn from a distribution.  The
weights then encode an exploration strategy; the hidden state is the agent's (implicit) belief.

Two task distributions:
  independent : p1, p2 ~ U[0, 1] independently
  dependent   : p1 ~ U[0, 1], p2 = 1 - p1   (one pull is informative about BOTH arms)

We compare the meta-learned agent with generic bandit algorithms that assume independent arms
(UCB1, Thompson sampling with Beta(1,1) priors, greedy) and with the Bayes-optimal policy for each
distribution, computed EXACTLY by dynamic programming over the posterior (the belief state of the
Bayes-adaptive MDP, Section 7.2, Eq. 15.31):
  independent : belief = (s1, f1, s2, f2) success/failure counts, posteriors Beta(1+s, 1+f)
  dependent   : belief = (a, b) with a = s1 + f2 (evidence that p1 is high), b = f1 + s2.

Metric: expected regret over the trial, sum_t (max(p1, p2) - p_{A_t}), on 10,000 fresh test tasks
(the same tasks for every method).

Run:  python code/ch15_beyond_mdps/rl2_bandits.py [--quick]
Full mode writes figures/rl2_bandits.png.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
H = 20                 # pulls per trial (one task)


def sample_tasks(rng, n, dist):
    p1 = rng.random(n)
    p2 = rng.random(n) if dist == "independent" else 1.0 - p1
    return np.stack([p1, p2], axis=1)


# ----------------------------------------------------------------------------------------------
# Bayes-optimal policies by dynamic programming over the belief (Section 7.2, Eq. 15.31)
# ----------------------------------------------------------------------------------------------
def bayes_optimal_independent(H):
    """V[s1,f1,s2,f2] = max_a { m_a + m_a V[.. s_a+1 ..] + (1-m_a) V[.. f_a+1 ..] }, m_a = (1+s_a)/(2+s_a+f_a).

    Computed backwards over the number of pulls made so far; entries whose counts sum to t are
    the beliefs reachable after t pulls.  Returns the greedy-in-the-Bayes-value policy table.
    """
    n = H + 2
    idx = np.indices((n, n, n, n))
    total = idx.sum(axis=0)
    m1 = (1 + idx[0]) / (2 + idx[0] + idx[1])
    m2 = (1 + idx[2]) / (2 + idx[2] + idx[3])
    V = np.zeros((n,) * 4)                      # V after H pulls is 0
    policy = np.zeros((n,) * 4, dtype=np.int8)
    for t in range(H - 1, -1, -1):
        Vs1 = np.zeros_like(V); Vs1[:-1] = V[1:]               # V[s1+1, f1, s2, f2]
        Vf1 = np.zeros_like(V); Vf1[:, :-1] = V[:, 1:]          # V[s1, f1+1, s2, f2]
        Vs2 = np.zeros_like(V); Vs2[:, :, :-1] = V[:, :, 1:]
        Vf2 = np.zeros_like(V); Vf2[:, :, :, :-1] = V[:, :, :, 1:]
        Q1 = m1 * (1 + Vs1) + (1 - m1) * Vf1
        Q2 = m2 * (1 + Vs2) + (1 - m2) * Vf2
        layer = total == t
        V = np.where(layer, np.maximum(Q1, Q2), V)
        policy[layer] = (Q2 > Q1 + 1e-12)[layer]
    return policy, V[0, 0, 0, 0]


def bayes_optimal_dependent(H):
    """Belief (a, b): p1 ~ Beta(1+a, 1+b).  Arm 1 succeeds w.p. m, arm 2 w.p. 1-m, m = (1+a)/(2+a+b).

    Both arms move the belief identically (a success on arm 1 and a failure on arm 2 are the same
    evidence), so the future term is the same for both actions and the Bayes-optimal policy is
    simply greedy in the posterior mean.  We still run the DP to show it.
    """
    n = H + 2
    a, b = np.indices((n, n))
    m = (1 + a) / (2 + a + b)
    V = np.zeros((n, n))
    policy = np.zeros((n, n), dtype=np.int8)
    for t in range(H - 1, -1, -1):
        Va = np.zeros_like(V); Va[:-1] = V[1:]
        Vb = np.zeros_like(V); Vb[:, :-1] = V[:, 1:]
        future = m * Va + (1 - m) * Vb                          # identical for both arms
        Q1, Q2 = m + future, (1 - m) + future
        layer = (a + b) == t
        V = np.where(layer, np.maximum(Q1, Q2), V)
        policy[layer] = (Q2 > Q1 + 1e-12)[layer]
    return policy, V[0, 0]


# ----------------------------------------------------------------------------------------------
# Running a policy on a batch of tasks
# ----------------------------------------------------------------------------------------------
def run_baseline(rng, P, method, tables):
    """Vectorised over tasks.  Returns per-step expected regret (n_tasks, H) and actions."""
    n = len(P)
    S = np.zeros((n, 2), dtype=int)
    F = np.zeros((n, 2), dtype=int)
    regret = np.zeros((n, H))
    best = P.max(axis=1)
    rows = np.arange(n)
    for t in range(H):
        if method == "random":
            a = rng.integers(0, 2, size=n)
        elif method == "greedy":
            m = (1 + S) / (2 + S + F)
            a = np.where(np.abs(m[:, 0] - m[:, 1]) < 1e-12, rng.integers(0, 2, size=n), m.argmax(axis=1))
        elif method == "ucb1":
            N = S + F
            mean = np.where(N > 0, S / np.maximum(N, 1), 0.0)
            bonus = np.sqrt(2 * np.log(max(t, 1)) / np.maximum(N, 1))
            u = np.where(N == 0, np.inf, mean + bonus)
            a = np.where(u[:, 0] == u[:, 1], rng.integers(0, 2, size=n), u.argmax(axis=1))
        elif method == "thompson":
            a = rng.beta(1 + S, 1 + F).argmax(axis=1)
        elif method == "bayes_independent":
            a = tables["independent"][S[:, 0], F[:, 0], S[:, 1], F[:, 1]].astype(int)
        elif method == "bayes_dependent":
            a = tables["dependent"][S[:, 0] + F[:, 1], F[:, 0] + S[:, 1]].astype(int)
        else:
            raise ValueError(method)
        r = rng.random(n) < P[rows, a]
        S[rows, a] += r
        F[rows, a] += ~r
        regret[:, t] = best - P[rows, a]
    return regret


class RL2Agent(nn.Module):
    """GRU policy: input (one-hot previous action, previous reward, t/H) -> (logits, value / H)."""

    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRUCell(4, hidden)
        self.pi = nn.Linear(hidden, 2)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x, h):
        h = self.gru(x, h)
        return self.pi(h), self.v(h).squeeze(-1), h


def rollout(agent, P_t, rng_t, greedy=False):
    """Run one trial of H pulls on each task of the batch, keeping the hidden state throughout."""
    n = P_t.shape[0]
    h = torch.zeros(n, agent.hidden)
    x = torch.zeros(n, 4)                          # nothing observed before the first pull
    logps, ents, vals, rews, acts = [], [], [], [], []
    for t in range(H):
        logits, v, h = agent(x, h)
        dist = torch.distributions.Categorical(logits=logits)
        a = logits.argmax(dim=1) if greedy else dist.sample()
        u = torch.rand(n, generator=rng_t) if rng_t is not None else torch.rand(n)
        r = (u < P_t.gather(1, a[:, None]).squeeze(1)).float()
        logps.append(dist.log_prob(a))
        ents.append(dist.entropy())
        vals.append(v)
        rews.append(r)
        acts.append(a)
        # Algorithm 15.10: the next input is (a_t, r_t, t+1); the hidden state is NOT reset
        x = torch.cat([nn.functional.one_hot(a, 2).float(), r[:, None],
                       torch.full((n, 1), (t + 1) / H)], dim=1)
    return (torch.stack(logps, 1), torch.stack(ents, 1), torch.stack(vals, 1),
            torch.stack(rews, 1), torch.stack(acts, 1))


def train_rl2(dist, updates, batch, seed, lr=2e-3, ent0=0.05, log_every=100, raw_value_target=False):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    rng_t = torch.Generator().manual_seed(seed)
    agent = RL2Agent()
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    # linear step-size decay (to 5% of lr): reduces the noise of the final policy
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: max(0.05, 1.0 - k / updates))
    curve = []
    diag = []            # (update, |grad of value loss|, |grad of policy loss|) on the GRU weights
    clipped = []         # total gradient norm > 5 at this update?
    gru_params = list(agent.gru.parameters())
    for u in range(1, updates + 1):
        P = torch.as_tensor(sample_tasks(rng, batch, dist), dtype=torch.float32)
        logp, ent, val, rew, act = rollout(agent, P, rng_t)
        # Monte Carlo return within the trial (gamma = 1: the trial objective is total reward).
        # The value head predicts G / H (a number in [0, 1]).  Without this normalisation the
        # policy barely learns: with --raw-value-target --updates 1500 the test regret ends at
        # 3.010 / 4.991 (independent / dependent arms), against 3.318 / 5.008 for random actions.
        # The diagnostic below measures the likely mechanism: the value loss (returns up to 20)
        # dominates the gradient reaching the shared GRU, and norm clipping then shrinks the
        # policy-gradient part along with it.
        G = rew.flip(1).cumsum(1).flip(1)
        scale = 1.0 if raw_value_target else H                # ablation: regress the raw return
        adv = (G - scale * val).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ent_coef = ent0 * max(0.0, 1.0 - u / (0.8 * updates))   # anneal exploration bonus to 0
        pg_loss = -(logp * adv).mean()
        v_loss = 0.5 * ((G / scale - val) ** 2).mean()
        loss = pg_loss + v_loss - ent_coef * ent.mean()
        if u % log_every == 0:
            # gradient norms of the two terms on the SHARED recurrent weights (read-only: the
            # update below is unchanged, so the training run is identical with or without this)
            g_v = torch.autograd.grad(v_loss, gru_params, retain_graph=True)
            g_pg = torch.autograd.grad(pg_loss, gru_params, retain_graph=True)
            norm = lambda gs: float(torch.sqrt(sum((g ** 2).sum() for g in gs)))
            diag.append((u, norm(g_v), norm(g_pg)))
        opt.zero_grad()
        loss.backward()
        total_norm = nn.utils.clip_grad_norm_(agent.parameters(), 5.0)
        clipped.append(float(total_norm) > 5.0)
        opt.step()
        sched.step()
        if u % log_every == 0:
            exp_regret = (P.max(1).values[:, None] - P.gather(1, act)).sum(1).mean().item()
            curve.append((u, exp_regret))
    return agent, np.array(curve), np.array(diag), np.array(clipped)


def eval_rl2(agent, P, seed):
    with torch.no_grad():
        P_t = torch.as_tensor(P, dtype=torch.float32)
        _, _, _, _, act = rollout(agent, P_t, torch.Generator().manual_seed(seed))
    act = act.numpy()
    return P.max(axis=1)[:, None] - np.take_along_axis(P, act, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figures")
    ap.add_argument("--updates", type=int, default=3000)
    ap.add_argument("--raw-value-target", action="store_true",
                    help="ablation: value head regresses the raw return G instead of G/H (no figures)")
    ap.add_argument("--no-figures", action="store_true",
                    help="do not write figures (e.g. for a controlled run with --updates 1500)")
    args = ap.parse_args()
    t0 = time.time()
    updates = 60 if args.quick else args.updates
    batch = 256
    n_test = 2_000 if args.quick else 10_000
    print(f"RL^2 on 2-armed Bernoulli bandits | H={H} GRU hidden=64 A2C batch={batch} updates={updates} "
          f"lr=2e-3 (linear decay) entropy 0.05 -> 0 test tasks={n_test} raw_value_target={args.raw_value_target} "
          f"quick={args.quick}")
    print("seeds: RL^2 training (numpy + torch) 0, test tasks 999, baselines 7, RL^2 evaluation 11")
    tables = {}
    tables["independent"], v_ind = bayes_optimal_independent(H)
    tables["dependent"], v_dep = bayes_optimal_dependent(H)
    print(f"[Bayes-optimal DP] expected total reward under the prior: independent {v_ind:.4f}, "
          f"dependent {v_dep:.4f}")
    results, curves = {}, {}
    for dist in ["independent", "dependent"]:
        P = sample_tasks(np.random.default_rng(999), n_test, dist)
        e_best = P.max(axis=1).mean() * H
        res = {}
        for method in ["random", "greedy", "ucb1", "thompson", f"bayes_{dist}"]:
            reg = run_baseline(np.random.default_rng(7), P, method, tables)
            res[method] = reg
        t1 = time.time()
        agent, curve, diag, clipped = train_rl2(dist, updates, batch, seed=0,
                                                log_every=10 if args.quick else 100,
                                                raw_value_target=args.raw_value_target)
        res["RL^2 (GRU)"] = eval_rl2(agent, P, seed=11)
        curves[dist] = curve
        results[dist] = res
        print(f"\n[{dist}] E[max(p1,p2)] * H = {e_best:.3f}; exact Bayes-optimal regret under the prior "
              f"H*E[max] - V = {H * (2 / 3 if dist == 'independent' else 3 / 4) - (v_ind if dist == 'independent' else v_dep):.5f}; "
              f"RL^2 trained in {time.time() - t1:.0f} s")
        ratio = diag[:, 1] / diag[:, 2]
        first = diag[:, 0] <= max(0.1 * updates, diag[0, 0])
        print(f"   gradient diagnostic on the shared GRU weights (every {int(diag[1, 0] - diag[0, 0])} updates): "
              f"|grad value loss| / |grad policy loss| median {np.median(ratio):.2f} "
              f"(first 10% of training: {np.median(ratio[first]) if first.any() else np.nan:.2f}); "
              f"updates whose total gradient norm was clipped (> 5): {100 * clipped.mean():.1f}% "
              f"(first 10%: {100 * clipped[:max(1, updates // 10)].mean():.1f}%)")
        print("   method              total regret over H=20 pulls (± s.e.)   regret in pulls 11-20")
        for method, reg in res.items():
            tot = reg.sum(axis=1)
            print(f"   {method:<20} {tot.mean():8.3f} ± {tot.std(ddof=1) / np.sqrt(len(tot)):.3f}"
                  f"                      {reg[:, 10:].sum(axis=1).mean():.3f}")

    if args.raw_value_target or args.no_figures:
        print("ablation / --no-figures run: no figures written")
    if not args.quick and not args.raw_value_target and not args.no_figures:
        from plot_style import setup, C
        plt = setup()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        style = {"random": (C[7], ":"), "greedy": (C[4], ":"), "ucb1": (C[3], "--"),
                 "thompson": (C[0], "--"), "RL^2 (GRU)": (C[1], "-")}
        for ax, dist in zip(axes[:2], ["independent", "dependent"]):
            for method, reg in results[dist].items():
                if method == "random":
                    continue
                col, ls = style.get(method, (C[2], "-."))
                label = "Bayes-optimal (DP)" if method.startswith("bayes") else method
                ax.plot(np.arange(1, H + 1), reg.mean(axis=0).cumsum(), color=col, ls=ls, label=label,
                        lw=2.6 if method.startswith("RL") else 1.8)
            ax.set_xlabel("pull t")
            ax.set_ylabel("expected cumulative regret")
            ax.set_title(f"{dist} arms" + (": p1, p2 ~ U[0,1]" if dist == "independent" else ": p2 = 1 − p1"))
            ax.legend(fontsize=8)
        ax = axes[2]
        for i, dist in enumerate(["independent", "dependent"]):
            c = curves[dist]
            bayes = results[dist][f"bayes_{dist}"].sum(axis=1).mean()
            ax.plot(c[:, 0], c[:, 1], color=C[1 + 2 * i], ls=["-", "--"][i], label=f"RL^2, {dist}")
            ax.axhline(bayes, color=C[1 + 2 * i], lw=1, ls=":", label=f"Bayes-optimal, {dist}")
        ax.set_xlabel("meta-training update (256 tasks each)")
        ax.set_ylabel("regret per trial (training batch)")
        ax.set_title("Meta-training")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "rl2_bandits.png"))
        plt.close(fig)
        print(f"figures written to {FIG_DIR}")
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
