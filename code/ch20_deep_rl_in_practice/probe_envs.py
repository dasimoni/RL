"""Probe environments as unit tests for RL agents (Chapter 20, Section 7.2).

Seven tiny environments, in the spirit of Andy Jones's "Debugging RL, without
the agonizing pain" (2021), each with a known analytic answer (gamma = 0.9):

  P1 constant reward      1 action, obs 0, 1 step, r = +1                     -> V(0) = 1
  P2 obs-dependent reward 1 action, obs +-1, 1 step, r = obs                  -> V(+-1) = +-1
  P3 discounting          1 action, obs 0 then 1, r = 0 then +1               -> V(0) = gamma, V(1) = 1
  P4 action-dependent     2 actions, obs 0, 1 step, r = +1 if a = 0 else -1   -> pi(0|0) = 1, Q(0,.) = (1, -1)
  P5 action-and-obs       2 actions, obs +-1, r = +1 if a = 1[obs > 0] else -1 -> correct action in both states
  P6 delayed credit       2 actions, obs 0, then obs 1 + a_0, then r = +1 if a_0 = 0 else -1
                                                                              -> pi(0|0) = 1, Q(0,.) = (gamma, -gamma)
  P7 time limit           1 action, obs 0, r = +1 forever, TRUNCATED after 3 steps -> V(0) = 1/(1-gamma) = 10

Two small agents are tested: an advantage actor-critic (A2C, n-step returns
over 16-step rollout segments) and DQN (replay + target network), each with a
correct version and several deliberately planted bugs (Section 8 of the
chapter). We also run a static "gradient isolation" check, which catches the
bugs that probes cannot see. The check calls the SAME loss function that
training uses (A2C.losses, DQN.target) on a small hand-made batch, so a missing
.detach() anywhere in that code path is detected; a check that re-implemented
the loss would only test its own copy.

Run:  python code/ch20_deep_rl_in_practice/probe_envs.py [--quick] [--long-p6]
      --long-p6 additionally trains the correct A2C and the "adv. not detached"
      A2C on P6 for 30,000 steps and prints V, pi and the entropy at 6k and 30k
      steps (the numbers quoted in Section 7.2 and Exercise 20.9).
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
GAMMA = 0.9
SEED = 0


# ---------------------------------------------------------------------------
# Probe environments (Gymnasium-style API: reset -> obs, info; step -> 5-tuple)
# ---------------------------------------------------------------------------
class Probe:
    n_actions = 1
    time_limit = None

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.t = 0
        self.s = self._start()
        return np.array([self.s], dtype=np.float32), {}

    def step(self, a):
        self.t += 1
        s2, r, term = self._transition(self.s, a)
        self.s = s2
        trunc = (self.time_limit is not None and self.t >= self.time_limit and not term)
        return np.array([s2], dtype=np.float32), float(r), term, trunc, {}

    def _start(self):
        return 0.0


class P1Constant(Probe):
    def _transition(self, s, a):
        return 0.0, 1.0, True


class P2ObsReward(Probe):
    def _start(self):
        return float(self.rng.choice([-1.0, 1.0]))

    def _transition(self, s, a):
        return 0.0, s, True


class P3Discount(Probe):
    def _transition(self, s, a):
        return (1.0, 0.0, False) if s == 0.0 else (0.0, 1.0, True)


class P4ActionReward(Probe):
    n_actions = 2

    def _transition(self, s, a):
        return 0.0, (1.0 if a == 0 else -1.0), True


class P5ActionObsReward(P2ObsReward):
    n_actions = 2

    def _transition(self, s, a):
        return 0.0, (1.0 if a == int(s > 0) else -1.0), True


class P6DelayedCredit(Probe):
    n_actions = 2

    def _transition(self, s, a):
        if s == 0.0:
            return 1.0 + a, 0.0, False             # the next observation remembers a_0 (Markov)
        return 0.0, (1.0 if s == 1.0 else -1.0), True


class P7TimeLimit(Probe):
    time_limit = 3

    def _transition(self, s, a):
        return 0.0, 1.0, False                      # never terminates: only the time limit ends it


PROBES = {"P1": P1Constant, "P2": P2ObsReward, "P3": P3Discount, "P4": P4ActionReward,
          "P5": P5ActionObsReward, "P6": P6DelayedCredit, "P7": P7TimeLimit}


def mlp(out):
    return nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, out))


def T(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float32)).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Agent 1: advantage actor-critic with n-step returns (Chapter 10; tested as in Section 7.2)
# ---------------------------------------------------------------------------
A2C_BUGS = {
    "correct": "",
    "no done mask": "returns flow across episode boundaries",
    "trunc = term": "time-limit truncation treated as termination",
    "done off-by-one": "uses the done flag of step t-1 for step t",
    "obs off-by-one": "stores s_{t+1} where s_t belongs",
    "policy-loss sign": "gradient DEscent on the policy objective",
    "adv. not detached": "actor loss back-propagates into the critic",
}


class A2C:
    def __init__(self, n_actions, bug="correct", seed=0, lr=3e-3, rollout=16, ent_coef=0.01, vf_coef=0.5):
        torch.manual_seed(seed)
        self.nA, self.bug, self.T = n_actions, bug, rollout
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.actor, self.critic = mlp(n_actions), mlp(1)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.rng = np.random.default_rng(seed)

    def V(self, obs):
        return self.critic(T(obs)).squeeze(-1)

    def pi(self, obs):
        return torch.softmax(self.actor(T(obs)), -1)

    def act(self, obs):
        with torch.no_grad():
            p = self.pi(obs)[0].numpy()
        return int(self.rng.choice(self.nA, p=p / p.sum()))

    def returns(self, rew, term, trunc, next_obs):
        """n-step bootstrapped returns over one rollout segment, computed backwards (Section 7.2)."""
        with torch.no_grad():
            v_next = self.V(next_obs).numpy()
        Tn = len(rew)
        G = np.zeros(Tn)
        g = 0.0
        prev_end = True
        for t in reversed(range(Tn)):
            term_t, trunc_t = term[t], trunc[t]
            if self.bug == "done off-by-one":
                # BUG: looks at the flags of the previous transition
                term_t = term[t - 1] if t > 0 else True
                trunc_t = trunc[t - 1] if t > 0 else False
            if self.bug == "trunc = term":
                term_t, trunc_t = term_t or trunc_t, False   # BUG
            ends = term_t or trunc_t or t == Tn - 1
            if self.bug == "no done mask":
                ends = t == Tn - 1                         # BUG: ignore episode boundaries
                term_t = False
            if ends:
                bootstrap = 0.0 if term_t else v_next[t]   # truncated or segment end: bootstrap from s_{t+1}
            else:
                bootstrap = g                              # same episode continues: use G_{t+1}
            g = rew[t] + GAMMA * bootstrap
            G[t] = g
        return G

    def losses(self, obs, act, rew, term, trunc, next_obs):
        """The three loss terms of one A2C update. Used by update() AND by the static check."""
        G = torch.as_tensor(self.returns(rew, term, trunc, next_obs), dtype=torch.float32)
        x = next_obs if self.bug == "obs off-by-one" else obs     # BUG variant stores the wrong state
        v = self.V(x)
        logits = self.actor(T(x))
        logp_all = torch.log_softmax(logits, -1)
        logp = logp_all.gather(1, torch.as_tensor(act).view(-1, 1)).squeeze(1)
        adv = G - v if self.bug == "adv. not detached" else (G - v).detach()
        sign = -1.0 if self.bug == "policy-loss sign" else 1.0
        policy_loss = -sign * (logp * adv).mean()
        entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
        value_loss = 0.5 * ((G - v) ** 2).mean()
        return policy_loss, value_loss, entropy

    def update(self, obs, act, rew, term, trunc, next_obs):
        policy_loss, value_loss, entropy = self.losses(obs, act, rew, term, trunc, next_obs)
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return policy_loss

    def train(self, env, steps):
        obs, _ = env.reset()
        buf = {k: [] for k in ("obs", "act", "rew", "term", "trunc", "next_obs")}
        for _ in range(steps):
            a = self.act(obs)
            obs2, r, term, trunc, _ = env.step(a)
            for k, v in zip(buf, (obs[0], a, r, term, trunc, obs2[0])):
                buf[k].append(v)
            obs = env.reset()[0] if (term or trunc) else obs2
            if len(buf["obs"]) == self.T:
                self.update(*[np.array(buf[k]) for k in buf])
                buf = {k: [] for k in buf}

    def grad_isolation_ok(self):
        """Static check: the actor loss must not produce gradients in the critic.

        Runs the agent's real loss code (self.losses) on a hand-made 4-transition batch
        containing a terminal and a truncated step, back-propagates ONLY the policy
        loss, and checks that no critic parameter received a gradient."""
        batch = (np.array([0.0, 1.0, 0.0, 1.0]), np.array([0, 1, 1, 0]),       # obs, actions
                 np.array([0.0, 1.0, 0.0, -1.0]),                             # rewards
                 np.array([False, True, False, False]), np.array([False, False, False, True]),  # term, trunc
                 np.array([1.0, 0.0, 2.0, 0.0]))                              # next observations
        self.opt.zero_grad(set_to_none=True)
        policy_loss, _, _ = self.losses(*batch)
        policy_loss.backward()
        leaked = sum(float(p.grad.abs().sum()) for p in self.critic.parameters() if p.grad is not None)
        self.opt.zero_grad(set_to_none=True)
        return leaked == 0.0


# ---------------------------------------------------------------------------
# Agent 2: DQN with replay and a target network (Chapter 09)
# ---------------------------------------------------------------------------
DQN_BUGS = {
    "correct": "",
    "no done mask": "target bootstraps through termination",
    "trunc = term": "time-limit truncation treated as termination",
    "shape broadcast": "Q(s,a) of shape (B,1) minus target of shape (B,)",
    "target not detached": "target computed with the ONLINE net and gradients enabled (as without a target network)",
}


class DQN:
    def __init__(self, n_actions, bug="correct", seed=0, lr=1e-3, eps=0.2, batch=64, buffer=5000,
                 target_every=25, learn_start=200):
        torch.manual_seed(seed)
        self.nA, self.bug = n_actions, bug
        self.q, self.q_targ = mlp(n_actions), mlp(n_actions)
        self.q_targ.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.eps, self.batch, self.cap = eps, batch, buffer
        self.target_every, self.learn_start = target_every, learn_start
        self.rng = np.random.default_rng(seed)
        self.mem = []

    def Q(self, obs):
        return self.q(T(obs))

    def act(self, obs):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.nA))
        with torch.no_grad():
            return int(self.Q(obs)[0].argmax())

    def target(self, r, o2, term, trunc):
        """TD target y = r + gamma (1 - terminated) max_a' Q_target(s', a')   (shape (B,))."""
        done = term | trunc if self.bug == "trunc = term" else term          # BUG variant
        if self.bug == "no done mask":
            done = np.zeros_like(term)                                       # BUG
        not_done = torch.as_tensor(1.0 - done.astype(np.float32))
        if self.bug == "target not detached":
            q_next = self.q(T(o2)).max(1).values                             # BUG: online net, with grad
        else:
            with torch.no_grad():
                q_next = self.q_targ(T(o2)).max(1).values
        return torch.as_tensor(r, dtype=torch.float32) + GAMMA * not_done * q_next

    def loss(self, o, a, r, o2, term, trunc):
        q_sa = self.q(T(o)).gather(1, torch.as_tensor(a).view(-1, 1))      # shape (B, 1)
        y = self.target(r, o2, term, trunc)                                  # shape (B,)
        if self.bug == "shape broadcast":
            return ((q_sa - y) ** 2).mean()                                  # BUG: (B,1) - (B,) -> (B,B)
        return ((q_sa.squeeze(1) - y) ** 2).mean()

    def train(self, env, steps):
        obs, _ = env.reset()
        for t in range(1, steps + 1):
            a = self.act(obs)
            obs2, r, term, trunc, _ = env.step(a)
            self.mem.append((obs[0], a, r, obs2[0], term, trunc))
            if len(self.mem) > self.cap:
                self.mem.pop(0)
            obs = env.reset()[0] if (term or trunc) else obs2
            if t >= self.learn_start:
                idx = self.rng.integers(0, len(self.mem), size=self.batch)
                o, a_, r_, o2, te, tr = (np.array(x) for x in zip(*[self.mem[i] for i in idx]))
                loss = self.loss(o, a_, r_, o2, te, tr)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            if t % self.target_every == 0:
                self.q_targ.load_state_dict(self.q.state_dict())

    def grad_isolation_ok(self):
        """Static check: the TD target must carry no gradient (a semi-gradient method)."""
        o = np.zeros(4, dtype=np.float32)
        y = self.target(np.ones(4), o, np.zeros(4, bool), np.zeros(4, bool))
        return not y.requires_grad


# ---------------------------------------------------------------------------
# Checks: compare learned quantities with the analytic answers
# ---------------------------------------------------------------------------
def check(agent, probe, kind):
    """Return (passed, worst normalized error, description of the worst quantity)."""
    g = GAMMA
    rows = []                                    # (name, learned, expected)
    with torch.no_grad():
        if kind == "a2c":
            V = lambda s: float(agent.V(np.array([s]))[0])
            p0 = lambda s: float(agent.pi(np.array([s]))[0, 0])
            if probe == "P1":
                rows.append(("V(0)", V(0), 1.0))
            elif probe == "P2":
                rows += [("V(+1)", V(1), 1.0), ("V(-1)", V(-1), -1.0)]
            elif probe == "P3":
                rows += [("V(0)", V(0), g), ("V(1)", V(1), 1.0)]
            elif probe == "P4":
                rows += [("pi(0|0)", p0(0), 1.0), ("V(0)", V(0), 2 * p0(0) - 1)]
            elif probe == "P5":
                rows += [("pi(1|+1)", 1 - p0(1), 1.0), ("pi(0|-1)", p0(-1), 1.0),
                         ("V(+1)", V(1), 1 - 2 * p0(1)), ("V(-1)", V(-1), 2 * p0(-1) - 1)]
            elif probe == "P6":
                # A probe may only test what the agent actually experiences. An on-policy
                # agent that (almost) always takes a_0 = 0 never visits obs 2, so V(2) is
                # never trained, and vice versa for obs 1. Check the branch it visits.
                rows += [("pi(0|0)", p0(0), 1.0), ("V(0)", V(0), g * (2 * p0(0) - 1))]
                rows.append(("V(1)", V(1), 1.0) if p0(0) >= 0.5 else ("V(2)", V(2), -1.0))
            elif probe == "P7":
                rows.append(("V(0)", V(0), 1 / (1 - g)))
        else:
            Q = lambda s: agent.Q(np.array([s]))[0].numpy()
            if probe == "P1":
                rows.append(("Q(0)", Q(0)[0], 1.0))
            elif probe == "P2":
                rows += [("Q(+1)", Q(1)[0], 1.0), ("Q(-1)", Q(-1)[0], -1.0)]
            elif probe == "P3":
                rows += [("Q(0)", Q(0)[0], g), ("Q(1)", Q(1)[0], 1.0)]
            elif probe == "P4":
                rows += [("Q(0,0)", Q(0)[0], 1.0), ("Q(0,1)", Q(0)[1], -1.0)]
            elif probe == "P5":
                rows += [("Q(+1,1)", Q(1)[1], 1.0), ("Q(+1,0)", Q(1)[0], -1.0),
                         ("Q(-1,0)", Q(-1)[0], 1.0), ("Q(-1,1)", Q(-1)[1], -1.0)]
            elif probe == "P6":
                rows += [("Q(0,0)", Q(0)[0], g), ("Q(0,1)", Q(0)[1], -g),
                         ("Q(1,.)", Q(1).max(), 1.0), ("Q(2,.)", Q(2).max(), -1.0)]
            elif probe == "P7":
                rows.append(("Q(0)", Q(0)[0], 1 / (1 - g)))
    worst, desc = 0.0, ""
    for name, got, exp in rows:
        if name.startswith("pi"):
            err = max(0.0, 0.9 - got) / 0.1          # a policy check passes if pi >= 0.9
        else:
            err = abs(got - exp) / (0.1 * max(1.0, abs(exp)))   # value tolerance: 10% (at least 0.1)
        if err > worst:
            worst, desc = err, f"{name} = {got:.2f} (expected {exp:.2f})"
    return worst <= 1.0, worst, desc


def run_suite(kind, bugs, probes, steps, seed):
    results = {}
    for bug in bugs:
        cls = A2C if kind == "a2c" else DQN
        for k, pname in enumerate(probes):
            env = PROBES[pname](seed + k)
            agent = cls(env.n_actions, bug=bug, seed=seed + k)
            agent.train(env, steps)
            results[(bug, pname)] = check(agent, pname, kind)
        results[(bug, "grad")] = (cls(2, bug=bug, seed=seed).grad_isolation_ok(), 0.0, "")
    return results


def report(kind, bugs, probes, res):
    title = "A2C (n-step actor-critic)" if kind == "a2c" else "DQN"
    print(f"\n{title}: PASS/FAIL per probe (worst error in units of the tolerance)")
    print("  " + " " * 19 + "".join(f"{p:>11s}" for p in probes) + "   grad-check")
    for bug in bugs:
        cells = "".join(f"{('ok' if res[(bug, p)][0] else 'FAIL'):>5s}{min(res[(bug, p)][1], 999):6.1f}" for p in probes)
        print(f"  {bug:<19s}{cells}   {'ok' if res[(bug, 'grad')][0] else 'FAIL':>6s}")
    for bug in bugs:
        fails = [f"{p}: {res[(bug, p)][2]}" for p in probes if not res[(bug, p)][0]]
        if fails:
            print(f"    {bug}: " + "; ".join(fails))


def make_figure(a2c_bugs, dqn_bugs, probes, ra, rd):
    import plot_style
    from matplotlib.colors import ListedColormap
    plt = plot_style.setup()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [len(a2c_bugs), len(dqn_bugs) + 0.4]})
    cmap = ListedColormap(["#e8f4ee", "#f6d4cf"])        # pass (pale green), fail (pale red)
    for ax, bugs, res, title in ((axes[0], a2c_bugs, ra, "A2C"), (axes[1], dqn_bugs, rd, "DQN")):
        cols = probes + ["grad"]
        M = np.array([[0 if res[(b, p)][0] else 1 for b in bugs] for p in cols])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i, p in enumerate(cols):
            for j, b in enumerate(bugs):
                ok, err, _ = res[(b, p)]
                txt = ("pass" if ok else "FAIL") if p == "grad" else (f"{err:.1f}" if ok else f"FAIL\n{min(err, 999):.0f}")
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color=plot_style.INK, fontweight="normal" if ok else "bold")
        ax.set_xticks(range(len(bugs)))
        ax.set_xticklabels(bugs, rotation=30, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(["P1 constant", "P2 obs-dep.", "P3 discount", "P4 action-dep.", "P5 act.+obs",
                            "P6 delayed", "P7 time limit", "grad isolation"][:len(cols)], fontsize=8.5)
        ax.grid(False)
        ax.set_title(f"{title}: probe results (cell = worst error / tolerance)")
    fig.tight_layout()
    path = __file__.rsplit("/", 1)[0] + "/figures/probe_results.png"
    fig.savefig(path)
    plt.close(fig)
    print("Figure written to", path)


def long_p6(seed=SEED):
    """Section 7.2 / Exercise 20.9: how the bias of an undetached advantage evolves on P6."""
    k = list(PROBES).index("P6")
    print("\nLong P6 run (A2C, same seeds as the suite): V, pi and entropy after 6,000 and 30,000 steps")
    for bug in ("correct", "adv. not detached"):
        env = PROBES["P6"](seed + k)
        agent = A2C(env.n_actions, bug=bug, seed=seed + k)
        done = 0
        for target in (6000, 30000):
            agent.train(env, target - done)          # 6000 = 375 full segments, so no data is lost
            done = target
            with torch.no_grad():
                V = agent.V(np.array([0.0, 1.0, 2.0])).numpy()
                P = agent.pi(np.array([0.0, 1.0])).numpy()
            H = -(P * np.log(P)).sum(1)
            print(f"  {bug:<18s} {target:>6,d} steps: V(0) = {V[0]:.3f} (true {GAMMA * (2 * P[0, 0] - 1):.3f}),"
                  f" V(1) = {V[1]:.3f} (true 1), pi(.|0) = [{P[0, 0]:.3f}, {P[0, 1]:.3f}],"
                  f" pi(.|1) = [{P[1, 0]:.3f}, {P[1, 1]:.3f}], entropy at obs 0 / 1 = {H[0]:.3f} / {H[1]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--long-p6", action="store_true", help="also run the 30k-step P6 diagnostic (about 20 s)")
    args = ap.parse_args()
    probes = list(PROBES)
    if args.quick:
        a2c_bugs, dqn_bugs = ["correct", "no done mask"], ["correct", "shape broadcast"]
        probes = ["P1", "P2", "P4"]
        a2c_steps, dqn_steps = 1500, 800
    else:
        a2c_bugs, dqn_bugs = list(A2C_BUGS), list(DQN_BUGS)
        a2c_steps, dqn_steps = 6000, 3000
    print(f"seed={SEED} gamma={GAMMA}; A2C: lr 3e-3, 16-step segments, entropy 0.01, {a2c_steps} steps per probe;"
          f" DQN: lr 1e-3, eps 0.2, batch 64, target sync every 25 steps, {dqn_steps} steps per probe; quick={args.quick}")
    print("tolerance: |V - V_true| <= 0.1 max(1, |V_true|); policy probabilities >= 0.9")
    t0 = time.time()
    ra = run_suite("a2c", a2c_bugs, probes, a2c_steps, SEED)
    report("a2c", a2c_bugs, probes, ra)
    t1 = time.time()
    rd = run_suite("dqn", dqn_bugs, probes, dqn_steps, SEED)
    report("dqn", dqn_bugs, probes, rd)
    print(f"\nA2C suite {t1 - t0:.0f} s, DQN suite {time.time() - t1:.0f} s")
    caught = {b: any(not ra[(b, p)][0] for p in probes + ["grad"]) for b in a2c_bugs[1:]}
    caught.update({"DQN " + b: any(not rd[(b, p)][0] for p in probes + ["grad"]) for b in dqn_bugs[1:]})
    print("bug caught by at least one check: " + ", ".join(f"{b}: {'yes' if c else 'NO'}" for b, c in caught.items()))
    if args.long_p6:
        long_p6()
    if not args.quick:
        make_figure(a2c_bugs, dqn_bugs, probes, ra, rd)
    print(f"Total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
