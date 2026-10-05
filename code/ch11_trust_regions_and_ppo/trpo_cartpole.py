"""Natural policy gradient and TRPO on CartPole: Fisher-vector products, conjugate gradient,
backtracking line search -- and why a fixed parameter-space step size is the wrong knob.

Chapter 11, Sections 5-6 (Algorithms 11.3-11.5).

Part A (checks, float64, small 16-16 policy so the Fisher matrix can be built explicitly):
  1. Fisher-vector products by double back-propagation through the mean KL agree with
     F v for the explicit Fisher F = E_s sum_a pi(a|s) grad log pi grad log pi^T  (Eq. 11.19).
  2. Conjugate gradient on (F + zeta I) x = g (damping zeta) converges in far fewer iterations than the
     number of parameters, because F's spectrum is dominated by a few large eigenvalues.
  3. The quadratic model 1/2 d^T F d of the KL is accurate for the TRPO step and degrades
     for larger steps.
Part B (learning, 64-64 policy): vanilla policy gradient with three fixed step sizes vs.
  natural policy gradient with a KL-normalised step (no line search) vs. TRPO (line search),
  same batches (2048 steps), same GAE advantages, same value-function fitting, 3 seeds.

Run:  python code/ch11_trust_regions_and_ppo/trpo_cartpole.py [--quick]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.distributions import Categorical  # noqa: E402

torch.set_num_threads(1)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ---------------------------------------------------------------------------------------------
# Small helpers: flat parameter vectors
# ---------------------------------------------------------------------------------------------
def flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p))
        i += n


def flat_grad(y: torch.Tensor, model: nn.Module, **kw) -> torch.Tensor:
    gs = torch.autograd.grad(y, list(model.parameters()), **kw)
    return torch.cat([g.reshape(-1) for g in gs])


def mlp(inp, out, hidden, out_gain):
    m = nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                      nn.Linear(hidden, out))
    for layer, g in zip([m[0], m[2], m[4]], [np.sqrt(2), np.sqrt(2), out_gain]):
        nn.init.orthogonal_(layer.weight, g)
        nn.init.zeros_(layer.bias)
    return m


# ---------------------------------------------------------------------------------------------
# The three ingredients of TRPO (Section 6)
# ---------------------------------------------------------------------------------------------
def mean_kl(policy: nn.Module, obs: torch.Tensor, old_logits: torch.Tensor) -> torch.Tensor:
    """Average over states of KL(pi_old(.|s) || pi_theta(.|s)) for categorical policies."""
    old_logp = torch.log_softmax(old_logits, -1)
    new_logp = torch.log_softmax(policy(obs), -1)
    return (old_logp.exp() * (old_logp - new_logp)).sum(-1).mean()


def fisher_vector_product(policy, obs, old_logits, v: torch.Tensor, damping: float) -> torch.Tensor:
    """(F + damping I) v without ever forming F (Eq. 11.26).

    F is the Hessian of the mean KL at theta = theta_old, so
    F v = grad_theta [ (grad_theta KL)^T v ],  two backward passes ("double backprop").
    """
    kl = mean_kl(policy, obs, old_logits)
    g = flat_grad(kl, policy, create_graph=True)       # = 0 at theta_old, but its graph is not
    return flat_grad((g * v).sum(), policy) + damping * v


def conjugate_gradient(Avp, b: torch.Tensor, iters: int = 10, tol: float = 1e-10, trace=None):
    """Solve A x = b for symmetric positive-definite A using only products A p (Algorithm 11.4)."""
    x = torch.zeros_like(b)
    r = b.clone()            # residual b - A x  (x = 0)
    p = b.clone()            # search direction
    rr = r @ r
    for k in range(iters):
        Ap = Avp(p)
        alpha = rr / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rr_new = r @ r
        if trace is not None:
            trace.append(x.clone())
        if rr_new < tol:
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x


# ---------------------------------------------------------------------------------------------
# Rollouts with GAE (same conventions as ppo.py: bootstrap through truncation)
# ---------------------------------------------------------------------------------------------
class Sampler:
    def __init__(self, env_id, num_envs, seed):
        self.envs = [gym.make(env_id) for _ in range(num_envs)]
        self.obs = np.stack([e.reset(seed=seed * 1000 + i)[0] for i, e in enumerate(self.envs)]).astype(np.float32)
        self.ep_ret = np.zeros(num_envs)
        self.episodes = []     # (global step, return)
        self.step = 0

    def rollout(self, policy, vf, T, gamma, lam):
        N, obs_dim = len(self.envs), self.obs.shape[1]
        O = np.zeros((T, N, obs_dim), np.float32)
        Aa = np.zeros((T, N), np.int64)
        R = np.zeros((T, N), np.float32)
        D = np.zeros((T, N), np.float32)
        V = np.zeros((T, N), np.float32)
        for t in range(T):
            self.step += N
            with torch.no_grad():
                o = torch.as_tensor(self.obs)
                a = Categorical(logits=policy(o)).sample().numpy()
                V[t] = vf(o).squeeze(-1).numpy()
            O[t], Aa[t] = self.obs, a
            for i, env in enumerate(self.envs):
                o2, r, te, tr, _ = env.step(int(a[i]))
                self.ep_ret[i] += r
                if tr and not te:          # time-limit truncation: bootstrap
                    with torch.no_grad():
                        r += gamma * vf(torch.as_tensor(o2, dtype=torch.float32)).item()
                if te or tr:
                    self.episodes.append((self.step, self.ep_ret[i]))
                    self.ep_ret[i] = 0.0
                    o2, _ = env.reset()
                R[t, i], D[t, i] = r, float(te or tr)
                self.obs[i] = o2
        with torch.no_grad():
            last_v = vf(torch.as_tensor(self.obs)).squeeze(-1).numpy()
        adv = np.zeros_like(R)
        gae = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else V[t + 1]
            delta = R[t] + gamma * (1 - D[t]) * nv - V[t]
            gae = delta + gamma * lam * (1 - D[t]) * gae
            adv[t] = gae
        ret = adv + V
        f = lambda x: torch.as_tensor(x.reshape(T * N, *x.shape[2:]))  # noqa: E731
        return f(O), f(Aa), f(adv), f(ret)


def fit_value(vf, opt, obs, ret, epochs=10, mb=256):
    n = obs.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, mb):
            idx = perm[s:s + mb]
            loss = ((vf(obs[idx]).squeeze(-1) - ret[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()


# ---------------------------------------------------------------------------------------------
# One policy update: vanilla PG / natural PG / TRPO
# ---------------------------------------------------------------------------------------------
def policy_update(policy, obs, act, adv, method, step_size=None, delta=0.01, damping=0.1,
                  cg_iters=10, backtrack=0.8, max_backtracks=10):
    """Returns a dict with the realised KL and line-search information."""
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    with torch.no_grad():
        old_logits = policy(obs)
        old_logp = Categorical(logits=old_logits).log_prob(act)

    def surrogate():
        # L(theta) = E[ pi_theta(a|s)/pi_old(a|s) * A ]  (Eqs. 11.7, 11.24); grad at theta_old = PG
        logp = Categorical(logits=policy(obs)).log_prob(act)
        return (torch.exp(logp - old_logp) * adv).mean()

    L_old = surrogate()
    g = flat_grad(L_old, policy)
    theta_old = flat_params(policy)
    info = {"backtracks": 0, "accepted": True}

    if method == "vpg":
        # parameter-space step of fixed length scale: theta <- theta + alpha g
        set_flat_params(policy, theta_old + step_size * g)
    else:
        Fvp = lambda v: fisher_vector_product(policy, obs, old_logits, v, damping)  # noqa: E731
        x = conjugate_gradient(Fvp, g, iters=cg_iters)                 # x ~ F^{-1} g
        xFx = x @ Fvp(x)
        full_step = torch.sqrt(2 * delta / (xFx + 1e-12)) * x          # 1/2 d^T F d = delta (Eq. 11.25)
        if method == "npg":
            set_flat_params(policy, theta_old + full_step)
        else:  # trpo: backtracking line search on the true surrogate and the true KL (Eq. 11.27)
            info["accepted"] = False
            for j in range(max_backtracks):
                set_flat_params(policy, theta_old + (backtrack ** j) * full_step)
                with torch.no_grad():
                    kl = mean_kl(policy, obs, old_logits).item()
                    improve = (surrogate() - L_old).item()
                if kl <= delta and improve > 0:
                    info["accepted"], info["backtracks"] = True, j
                    break
            if not info["accepted"]:
                set_flat_params(policy, theta_old)                     # reject: keep old policy
                info["backtracks"] = max_backtracks
    with torch.no_grad():
        info["kl"] = mean_kl(policy, obs, old_logits).item()
    return info


def run(method, seed, iters, step_size=None, delta=0.01, num_envs=4, T=512, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sampler = Sampler("CartPole-v1", num_envs, seed)
    policy = mlp(4, 2, 64, 0.01)
    vf = mlp(4, 1, 64, 1.0)
    vopt = torch.optim.Adam(vf.parameters(), lr=1e-3)
    kls, steps, backtracks = [], [], []
    for it in range(iters):
        obs, act, adv, ret = sampler.rollout(policy, vf, T, gamma=0.99, lam=0.97)
        info = policy_update(policy, obs, act, adv, method, step_size=step_size, delta=delta)
        fit_value(vf, vopt, obs, ret)
        kls.append(info["kl"])
        backtracks.append(info["backtracks"])
        steps.append(sampler.step)
        if verbose and (it + 1) % 10 == 0:
            rec = [e[1] for e in sampler.episodes[-20:]]
            print(f"    {method:5s} seed {seed} iter {it + 1:3d} step {sampler.step:6d} "
                  f"ret {np.mean(rec):6.1f}  KL {info['kl']:.2e}  backtracks {info['backtracks']}", flush=True)
    return {"episodes": np.array(sampler.episodes), "kl": np.array(kls), "step": np.array(steps),
            "backtracks": np.array(backtracks)}


# ---------------------------------------------------------------------------------------------
# Part A: checks of the Fisher machinery
# ---------------------------------------------------------------------------------------------
def part_a(n_states=512, seed=0):
    torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    # States from a random policy (any state distribution will do for the identity checks)
    env.action_space.seed(seed)
    obs, o = [], env.reset(seed=seed)[0]
    while len(obs) < n_states:
        obs.append(o)
        o, _, te, tr, _ = env.step(env.action_space.sample())
        if te or tr:
            o = env.reset()[0]
    obs = torch.as_tensor(np.array(obs), dtype=torch.float64)

    policy = mlp(4, 2, 16, 1.0).double()   # gain 1 so the policy is not ~uniform
    P = sum(p.numel() for p in policy.parameters())
    with torch.no_grad():
        old_logits = policy(obs)
    probs = torch.softmax(old_logits, -1)

    # Explicit Fisher: F = (1/|S|) sum_s sum_a pi(a|s) psi(s,a) psi(s,a)^T,  psi = grad log pi
    F = torch.zeros(P, P, dtype=torch.float64)
    for s in range(n_states):
        logp = torch.log_softmax(policy(obs[s:s + 1]), -1)[0]
        for a in range(2):
            psi = flat_grad(logp[a], policy, retain_graph=(a == 0))
            F += probs[s, a] * torch.outer(psi, psi)
    F /= n_states

    # 1. FVP vs explicit F v
    rel_errs = []
    for _ in range(5):
        v = torch.randn(P, dtype=torch.float64)
        fv = fisher_vector_product(policy, obs, old_logits, v, damping=0.0)
        rel_errs.append(((fv - F @ v).norm() / (F @ v).norm()).item())

    # 2. CG on (F + zeta I) x = g (zeta = damping), with g the policy gradient for random "advantages"
    damping = 0.1
    acts = Categorical(logits=old_logits).sample()
    adv = torch.randn(n_states, dtype=torch.float64)
    logp = Categorical(logits=policy(obs)).log_prob(acts)
    g = flat_grad((logp * adv).mean(), policy)
    Fd = F + damping * torch.eye(P, dtype=torch.float64)
    x_star = torch.linalg.solve(Fd, g)
    trace = []
    conjugate_gradient(lambda v: fisher_vector_product(policy, obs, old_logits, v, damping), g,
                       iters=50, tol=1e-30, trace=trace)
    cg_err = [((x - x_star).norm() / x_star.norm()).item() for x in trace]
    # quality of the step direction: cosine with the exact natural gradient, and the
    # first-order improvement g^T x relative to the exact g^T x*
    cg_cos = [(x @ x_star / (x.norm() * x_star.norm())).item() for x in trace]
    cg_gain = [((g @ x) / (g @ x_star)).item() for x in trace]
    eig_raw = torch.linalg.eigvalsh(F).flip(0)
    n_clamped = int((eig_raw <= 1e-16).sum())          # zero up to round-off (some even slightly < 0)
    eig = eig_raw.clamp_min(1e-16)                     # clamp so they can be drawn on a log axis

    # 3. KL along the TRPO direction: exact vs quadratic model
    delta = 0.01
    x10 = trace[9]
    step = torch.sqrt(2 * delta / (x10 @ (Fd @ x10))) * x10
    theta0 = flat_params(policy)
    ts = np.linspace(0, 3, 31)
    kl_true, kl_quad = [], []
    for t in ts:
        set_flat_params(policy, theta0 + t * step)
        with torch.no_grad():
            kl_true.append(mean_kl(policy, obs, old_logits).item())
        kl_quad.append((0.5 * t**2 * step @ (F @ step)).item())
    set_flat_params(policy, theta0)
    return dict(P=P, n_clamped=n_clamped, rel_errs=rel_errs, cg_err=cg_err, cg_cos=cg_cos, cg_gain=cg_gain, eig=eig.numpy(),
                ts=ts, kl_true=np.array(kl_true), kl_quad=np.array(kl_quad), damping=damping)


# ---------------------------------------------------------------------------------------------
def smoothed(episodes, grid, window=20):
    out = np.full(len(grid), np.nan)
    for j, gstep in enumerate(grid):
        k = np.searchsorted(episodes[:, 0], gstep, side="right")
        if k > 0:
            out[j] = episodes[max(0, k - window):k, 1].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    print("=== Part A: Fisher-vector products and conjugate gradient (float64, 16-16 policy) ===")
    A = part_a(n_states=128 if args.quick else 512)
    print(f"  parameters P = {A['P']}")
    print(f"  FVP (double backprop) vs explicit F v: max relative error {max(A['rel_errs']):.2e}")
    e = A["eig"]
    print(f"  Fisher eigenvalues: largest {e[0]:.3e}, 10th {e[9]:.3e}, 50th {e[49]:.3e}, "
          f"#eig > 1% of largest: {(e > 0.01 * e[0]).sum()} / {len(e)}; "
          f"#eig <= 1e-16 (zero up to round-off, clamped for plotting): {A['n_clamped']}")
    for k in [1, 3, 5, 10, 20]:
        if k <= len(A["cg_err"]):
            print(f"  CG iteration {k:2d}: relative error to (F+{A['damping']}I)^-1 g = {A['cg_err'][k - 1]:.3f}, "
                  f"cosine to exact = {A['cg_cos'][k - 1]:.3f}, g^T x / g^T x* = {A['cg_gain'][k - 1]:.3f}")
    i1 = np.argmin(np.abs(A["ts"] - 1.0))
    i3 = np.argmin(np.abs(A["ts"] - 3.0))
    print(f"  KL at the TRPO step (delta = 0.01): exact {A['kl_true'][i1]:.5f}, quadratic model {A['kl_quad'][i1]:.5f}")
    print(f"  KL at 3x the step:                  exact {A['kl_true'][i3]:.5f}, quadratic model {A['kl_quad'][i3]:.5f}")

    print("\n=== Part B: VPG vs NPG vs TRPO on CartPole-v1 (batch 2048 = 4 envs x 512 steps) ===")
    iters = 4 if args.quick else 60
    seeds = [1] if args.quick else [1, 2, 3]
    methods = [("vpg", dict(step_size=0.3), "VPG, alpha = 0.3"),
               ("vpg", dict(step_size=1.0), "VPG, alpha = 1"),
               ("vpg", dict(step_size=3.0), "VPG, alpha = 3"),
               ("npg", dict(delta=0.01), "NPG, delta = 0.01"),
               ("trpo", dict(delta=0.01), "TRPO, delta = 0.01"),
               ("npg", dict(delta=0.1), "NPG, delta = 0.1"),
               ("trpo", dict(delta=0.1), "TRPO, delta = 0.1")]
    print(f"  iterations={iters}, seeds={seeds}, gamma=0.99, GAE lambda=0.97, CG iters=10, damping=0.1, "
          f"backtrack coeff=0.8, value fn: Adam 1e-3, 10 epochs  (NPG = same step as TRPO, no line search)")
    results = {}
    for m, kw, label in methods:
        results[label] = [run(m, s, iters, verbose=not args.quick, **kw) for s in seeds]
        r = results[label]
        last = [res["episodes"][-20:, 1].mean() for res in r]
        g = np.linspace(0, iters * 2048, 100)
        auc = np.mean([np.nanmean(smoothed(res["episodes"], g)) for res in r])
        kl = np.concatenate([res["kl"] for res in r])
        print(f"  {label:20s} final return per seed: {np.round(last, 1)}  mean return over training {auc:6.1f}  "
              f"KL per update: median {np.median(kl):.2e}, min {kl.min():.1e}, max {kl.max():.1e}", flush=True)
        if m == "trpo":
            bt = np.concatenate([res["backtracks"] for res in r])
            print(f"  {'':20s} line search: mean backtracks {bt.mean():.2f}, updates with >= 1 backtrack "
                  f"{(bt > 0).sum()} / {len(bt)}, rejected updates {(bt == 10).sum()}")

    if not args.quick:
        from plot_style import C, kfmt, setup
        plt = setup()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
        ax[0].semilogy(np.arange(1, len(A["eig"]) + 1), A["eig"], color=C[0])
        if A["n_clamped"]:
            ax[0].annotate(f"{A['n_clamped']} eigenvalues <= 1e-16\n(round-off), clamped to 1e-16",
                           xy=(len(A["eig"]) - A["n_clamped"] / 2, 1e-16), xytext=(0.30, 0.22),
                           textcoords="axes fraction", fontsize=8,
                           arrowprops=dict(arrowstyle="->", color="#8a8985", lw=0.8))
        ax[0].set_title(f"(a) Fisher spectrum ({A['P']} params)")
        ax[0].set_xlabel("index")
        ax[0].set_ylabel("eigenvalue")
        k = np.arange(1, len(A["cg_err"]) + 1)
        ax[1].plot(k, A["cg_err"], color=C[0], label="relative error ||x_k - x*|| / ||x*||")
        ax[1].plot(k, A["cg_gain"], color=C[1], ls="--", label="g^T x_k / g^T x*")
        ax[1].plot(k, A["cg_cos"], color=C[2], ls=":", label="cosine(x_k, x*)")
        ax[1].axvline(10, color="#8a8985", lw=0.8)
        ax[1].set_title("(b) conjugate gradient on (F + 0.1 I) x = g")
        ax[1].set_xlabel("CG iteration k")
        ax[1].legend(fontsize=8)
        ax[2].plot(A["ts"], A["kl_true"], color=C[0], label="exact mean KL")
        ax[2].plot(A["ts"], A["kl_quad"], color=C[1], ls="--", label="quadratic model 1/2 t^2 d^T F d")
        ax[2].axhline(0.01, color="#8a8985", lw=0.8)
        ax[2].axvline(1.0, color="#8a8985", lw=0.8)
        ax[2].set_title("(c) KL along the TRPO direction t * d")
        ax[2].set_xlabel("t (t = 1 is the full TRPO step)")
        ax[2].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "trpo_fisher_cg.png"))
        plt.close(fig)

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.9))
        grid = np.linspace(0, iters * 2048, 150)
        look = {"VPG, alpha = 0.3": (C[2], ":"), "VPG, alpha = 1": (C[3], "-."), "VPG, alpha = 3": (C[4], "--"),
                "NPG, delta = 0.01": (C[0], "--"), "TRPO, delta = 0.01": (C[0], "-"),
                "NPG, delta = 0.1": (C[1], "--"), "TRPO, delta = 0.1": (C[1], "-")}
        panel = {"VPG, alpha = 0.3": 0, "VPG, alpha = 1": 0, "VPG, alpha = 3": 0, "TRPO, delta = 0.01": 0,
                 "NPG, delta = 0.01": 1, "NPG, delta = 0.1": 1, "TRPO, delta = 0.1": 1}
        with np.errstate(all="ignore"):
            import warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            for label, res in results.items():
                c, ls = look[label]
                curves = np.array([smoothed(r["episodes"], grid) for r in res])
                axes = [panel[label]] + ([1] if label == "TRPO, delta = 0.01" else [])
                for k in axes:
                    ax[k].plot(grid, np.nanmean(curves, 0), color=c, ls=ls, label=label)
                    ax[k].fill_between(grid, np.nanmin(curves, 0), np.nanmax(curves, 0), color=c, alpha=0.10)
                kl = np.array([r["kl"] for r in res])
                ax[2].semilogy(np.arange(1, iters + 1), np.exp(np.log(np.maximum(kl, 1e-10)).mean(0)),
                               color=c, ls=ls, label=label)
        ax[0].set_title("(a) fixed parameter-space steps vs TRPO")
        ax[1].set_title("(b) NPG (no line search) vs TRPO")
        for k in (0, 1):
            ax[k].set_xlabel("environment steps")
            kfmt(ax[k])
            ax[k].set_ylabel("return (last 20 episodes)")
            ax[k].legend(fontsize=8, loc="lower right")
        ax[2].set_title("(c) realised KL(pi_old || pi_new) per update")
        ax[2].set_xlabel("iteration")
        ax[2].set_ylabel("geometric mean over 3 seeds")
        ax[2].axhline(0.01, color="#8a8985", lw=0.8)
        ax[2].axhline(0.1, color="#8a8985", lw=0.8)
        # legend below the axes, so that it hides none of the curves
        ax[2].legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "trpo_vs_vpg.png"))
        plt.close(fig)
        print("  wrote figures/trpo_fisher_cg.png, figures/trpo_vs_vpg.png")
    print(f"total wall time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
