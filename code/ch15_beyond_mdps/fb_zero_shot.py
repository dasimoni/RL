"""Forward-backward (FB) representations versus successor features + GPI for zero-shot RL.

Chapter 15, Section 6.6 (Eqs. 15.29b-c).  Same slippery four-rooms dynamics as
successor_features_gpi.py, but a CONTINUING task (no terminal states, gamma = 0.95) whose
reward depends on the next state, r(s'), so that one object of interest, the successor measure,
covers every reward at once:

    M^pi(s, a, s') = sum_t gamma^t Pr{S_{t+1} = s' | S_0 = s, A_0 = a, pi},
    q^pi_r(s, a)   = sum_{s'} M^pi(s, a, s') r(s').

FB (Touati & Ollivier, 2021) learns, from the dynamics alone and with NO reward, a backward
embedding B(s') in R^d and a forward embedding F(s, a, z) in R^d for a family of policies
pi_z(s) = argmax_a F(s, a, z).z, such that

    M^{pi_z}(s, a, s') ~= F(s, a, z).B(s') rho(s')          (Eq. 15.29b; rho, written nu in the
                                                              chapter, is uniform over cells here)

Training minimizes the FB Bellman residual (the measure-valued Bellman equation, written for the
density m = M / rho)
    m(s, a, s') = P(s'|s, a) / rho(s') + gamma sum_{s''} P(s''|s, a) m(s'', pi_z(s''), s')
in rho-weighted least squares, with the KNOWN P (an exact expectation instead of samples), target
networks, and an orthonormality penalty E_rho[B B^T] ~= I.  B is a table; F is a small MLP of
(one-hot s, z) with one d-vector output per action.

Zero-shot test: given a NEW reward r (never seen in training), set z_R = E_rho[B(s) r(s)], rescale
it to the sphere of radius sqrt(d) on which z was sampled in training (scaling a reward does not
change its optimal policy), and act with pi_z; no planning, no learning. Each policy is then
evaluated EXACTLY (linear solve) and compared with the optimal policy and with the uniform random
policy: normalized regret = sum_s (v* - v^pi) / sum_s (v* - v^random) (0 = optimal, 1 = random).

Baseline: SF + GPI with hand-designed features phi(s) = one-hot of the four object cells and the
four base policies "go to object i" (Section 6.4); for a test reward we fit w by least squares,
r ~= phi w, then act by GPI.  Three families of test rewards:
    objects : r = phi w, w ~ U[-1,1]^4                 (linear in phi: SF+GPI's home turf)
    goals   : r = 1[s' = g], g uniform over cells      (not linear in phi)
    rooms   : r = w_k on every cell of room k, w ~ U[-1,1]^4  (not linear in phi, but smooth)

Run:  python code/ch15_beyond_mdps/fb_zero_shot.py [--quick]
Full mode writes figures/fb_zero_shot.png.
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

from four_rooms_options import FourRooms, LAYOUT

torch.set_num_threads(1)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
GAMMA = 0.95
OBJECTS = [(1, 1), (1, 11), (11, 1), (11, 11)]     # the corner cells of successor_features_gpi.py
ROOMS = ["NW", "NE", "SW", "SE"]


# ----------------------------------------------------------------------------------------------
# Exact tabular tools (reward on the next state, continuing task)
# ----------------------------------------------------------------------------------------------
class Tabular:
    def __init__(self, env):
        self.env, self.N, self.P = env, env.N, env.P          # P[a, s, s']
        self.rho = np.full(self.N, 1.0 / self.N)

    def optimal(self, r, tol=1e-11):
        v = np.zeros(self.N)
        Pr = np.einsum("asn,n->sa", self.P, r)
        for _ in range(20_000):
            q = Pr + GAMMA * np.einsum("asn,n->sa", self.P, v)
            v_new = q.max(axis=1)
            if np.abs(v_new - v).max() < tol:
                break
            v = v_new
        return q, v_new

    def P_pi(self, pi):
        return self.P[pi, np.arange(self.N)]                 # rows s, columns s'

    def evaluate(self, pi, r):
        """v^pi(s) = E[sum_t gamma^t r(S_{t+1})] for a deterministic policy pi."""
        Ppi = self.P_pi(pi)
        return np.linalg.solve(np.eye(self.N) - GAMMA * Ppi, Ppi @ r)

    def evaluate_random(self, r):
        Ppi = self.P.mean(axis=0)
        return np.linalg.solve(np.eye(self.N) - GAMMA * Ppi, Ppi @ r)

    def successor_measure(self, pi):
        """Exact M^pi(s, a, s') = P(s'|s,a) + gamma sum_s'' P(s''|s,a) M^pi(s'', pi(s''), s')."""
        Ppi = self.P_pi(pi)
        Mv = np.linalg.solve(np.eye(self.N) - GAMMA * Ppi, Ppi)       # M(s, pi(s), .)
        return self.P + GAMMA * np.einsum("asn,nm->asm", self.P, Mv)  # (A, N, N)


# ----------------------------------------------------------------------------------------------
# Forward-backward representation
# ----------------------------------------------------------------------------------------------
class ForwardNet(nn.Module):
    """F(s, a, z): first layer = embedding of s + linear in z (the same as a linear layer on the
    concatenation [one-hot(s), z], computed without the one-hot product); then one hidden layer and
    a d-vector per action."""

    def __init__(self, N, d, hidden=128, A=4):
        super().__init__()
        self.N, self.d, self.A = N, d, A
        self.emb = nn.Embedding(N, hidden)
        self.lin_z = nn.Linear(d, hidden)
        self.body = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, A * d))

    def forward(self, z):
        """z (bz, d) -> F (bz, N, A, d) for every state and action."""
        h = self.emb.weight[None, :, :] + self.lin_z(z)[:, None, :]          # (bz, N, hidden)
        return self.body(h).view(z.shape[0], self.N, self.A, self.d)


def sample_z(B, bz, d, rng_t):
    """Half on the sphere of radius sqrt(d); half sqrt(d)-normalized B(s) of a random cell
    (the z of the reward "reach s"), as in Touati & Ollivier's mixing of task vectors."""
    g = torch.randn(bz, d, generator=rng_t)
    idx = torch.randint(B.shape[0], (bz,), generator=rng_t)
    use_b = torch.rand(bz, generator=rng_t) < 0.5
    g = torch.where(use_b[:, None], B[idx].detach(), g)
    return np.sqrt(d) * g / g.norm(dim=1, keepdim=True).clamp_min(1e-8)


def train_fb(tab, d, steps, seed, bz=16, lr=1e-3, tau=0.01, ortho=1.0, log_every=0):
    torch.manual_seed(seed)
    rng_t = torch.Generator().manual_seed(seed)
    N = tab.N
    P = torch.tensor(tab.P, dtype=torch.float32)                         # (A, N, N)
    inv_rho = float(N)                                                     # rho uniform
    Fnet = ForwardNet(N, d)
    Ftarg = ForwardNet(N, d)
    Ftarg.load_state_dict(Fnet.state_dict())
    B = nn.Parameter(torch.randn(N, d))                                   # E_rho[B B^T] ~ I at init
    Btarg = B.detach().clone()
    opt = torch.optim.Adam(list(Fnet.parameters()) + [B], lr=lr)
    PP_rho = (tab.P ** 2 * inv_rho).sum(axis=2).mean()                    # E_{s,a} E_rho[(P/rho)^2], a constant
    eye_d = torch.eye(d)
    log = []
    for it in range(1, steps + 1):
        z = sample_z(B, bz, d, rng_t)
        F = Fnet(z)                                                        # (bz, N, A, d)
        with torch.no_grad():
            Ft = Ftarg(z)                                                  # (bz, N, A, d)
            a_next = torch.einsum("bsad,bd->bsa", Ft, z).argmax(dim=2)     # pi_z(s'') (bz, N)
            Fpi = torch.gather(Ft, 2, a_next[:, :, None, None].expand(-1, -1, 1, d)).squeeze(2)
            G = torch.einsum("ask,bkd->bsad", P, Fpi)                      # E[F(S'', pi_z(S''), z) | s, a]
        # E_{s'~rho}[(F.B(s') - P(s'|s,a)/rho(s') - gamma G.Btarg(s'))^2], expanded so that the N x N
        # density matrix is never formed (Touati & Ollivier's loss, with exact expectations):
        #   F'C F - 2 F.(P B)(s,a) - 2 gamma F'C' G + const,   C = E_rho[B B'], C' = E_rho[B Btarg']
        C = B.T @ B / N
        Cx = B.T @ Btarg / N
        PB = torch.einsum("ask,kd->sad", P, B)                             # sum_s' P(s'|s,a) B(s')
        quad = torch.einsum("bsad,de,bsae->bsa", F, C, F)
        cross_p = 2 * (F * PB[None]).sum(-1)
        cross_g = 2 * GAMMA * torch.einsum("bsad,de,bsae->bsa", F, Cx, G)
        loss_fb = (quad - cross_p - cross_g).mean()
        loss_ortho = ((C - eye_d) ** 2).sum()                              # E_rho[B B^T] ~= I
        if log_every and it % log_every == 0:
            with torch.no_grad():                                          # add back the constant: E[(m - target)^2]
                Ct = Btarg.T @ Btarg / N
                PBt = torch.einsum("ask,kd->sad", P, Btarg)
                const = (PP_rho + 2 * GAMMA * (G * PBt[None]).sum(-1)
                         + GAMMA ** 2 * torch.einsum("bsad,de,bsae->bsa", G, Ct, G))
                log.append((it, loss_fb.item() + const.mean().item(), loss_ortho.item()))
        loss = loss_fb + ortho * loss_ortho
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p, pt in zip(Fnet.parameters(), Ftarg.parameters()):
                pt.mul_(1 - tau).add_(tau * p)
            Btarg.mul_(1 - tau).add_(tau * B)
    return Fnet, B.detach(), log


def fb_policy(Fnet, B, r, rho):
    """Zero-shot: z_R = E_rho[B(s) r(s)]; the policy index is z_R rescaled to the training sphere,
    z = sqrt(d) z_R / |z_R| (rescaling r does not change the optimal policy), pi(s) = argmax_a
    F(s, a, z).z.  FB's value estimate for r is q(s, a) = F(s, a, z).z_R.  Returns (pi, q, z_R)."""
    d = B.shape[1]
    z_r = (rho * r) @ B.numpy()
    z = torch.tensor(np.sqrt(d) * z_r / max(np.linalg.norm(z_r), 1e-12), dtype=torch.float32)[None]
    with torch.no_grad():
        F = Fnet(z)[0].numpy()                                             # (N, A, d)
    pi = (F @ z[0].numpy()).argmax(axis=1)
    return pi, F @ z_r, z_r


# ----------------------------------------------------------------------------------------------
# SF + GPI baseline with hand features
# ----------------------------------------------------------------------------------------------
def sf_gpi_library(tab, phi):
    psis = []
    for i in range(phi.shape[1]):
        q, _ = tab.optimal(phi[:, i])
        pi = q.argmax(axis=1)
        Ppi = tab.P_pi(pi)
        psi_v = np.linalg.solve(np.eye(tab.N) - GAMMA * Ppi, Ppi @ phi)            # psi(s, pi(s))
        psis.append(np.einsum("asn,nd->sad", tab.P, phi + GAMMA * psi_v))          # psi(s, a)
    return psis


def sf_gpi_policy(psis, phi, r, rho):
    D = np.diag(rho)
    w = np.linalg.lstsq(phi.T @ D @ phi, phi.T @ D @ r, rcond=None)[0]             # r ~= phi w
    Q = np.stack([psi @ w for psi in psis]).max(axis=0)
    return Q.argmax(axis=1)


def make_tasks(env, rng, n, phi):
    room_ind = np.zeros((env.N, 4))
    for i in range(env.N):
        if env.room_of.get(i) in ROOMS:
            room_ind[i, ROOMS.index(env.room_of[i])] = 1.0
    goals = rng.choice(env.N, size=n, replace=n > env.N)
    return {
        "objects": [phi @ rng.uniform(-1, 1, 4) for _ in range(n)],
        "goals": [np.eye(env.N)[g] for g in goals],
        "rooms": [room_ind @ rng.uniform(-1, 1, 4) for _ in range(n)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test, no figures")
    ap.add_argument("--steps", type=int, default=4000, help="FB training steps per rank")
    ap.add_argument("--tasks", type=int, default=100, help="test rewards per family")
    ap.add_argument("--ranks", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    ranks = [16] if args.quick else args.ranks
    steps = 300 if args.quick else args.steps
    n_tasks = 20 if args.quick else args.tasks
    seeds = [0] if args.quick else args.seeds
    print(f"FB vs SF+GPI zero-shot | four rooms (slippery), continuing, gamma={GAMMA}, rho uniform | "
          f"ranks={ranks} steps={steps} seeds={seeds} test rewards per family={n_tasks} quick={args.quick}")
    env = FourRooms()
    tab = Tabular(env)
    rng = np.random.default_rng(0)
    phi = np.zeros((env.N, 4))
    for i, rc in enumerate(OBJECTS):
        phi[env.idx[rc], i] = 1.0
    tasks = make_tasks(env, rng, n_tasks, phi)

    # optimal and random values once per task
    ref = {fam: [] for fam in tasks}
    for fam, rs in tasks.items():
        for r in rs:
            _, v_star = tab.optimal(r)
            ref[fam].append((v_star, tab.evaluate_random(r)))

    def norm_regret(fam, k, pi):
        v_star, v_rand = ref[fam][k]
        v = tab.evaluate(pi, tasks[fam][k])
        return (v_star - v).sum() / max((v_star - v_rand).sum(), 1e-12)

    results = {}
    psis = sf_gpi_library(tab, phi)
    results["SF+GPI"] = {fam: np.mean([norm_regret(fam, k, sf_gpi_policy(psis, phi, r, tab.rho))
                                       for k, r in enumerate(rs)]) for fam, rs in tasks.items()}
    print("[SF+GPI] hand features = 4 object indicators; base policies = optimal for e_i; "
          "normalized regret: " + ", ".join(f"{f} {v:.4f}" for f, v in results["SF+GPI"].items()))

    fb_res = {}
    examples = {}
    for d in ranks:
        per_seed = []
        for seed in seeds:
            t1 = time.time()
            Fnet, B, log = train_fb(tab, d, steps, seed, log_every=max(1, steps // 4))
            row = {}
            for fam, rs in tasks.items():
                row[fam] = np.mean([norm_regret(fam, k, fb_policy(Fnet, B, r, tab.rho)[0])
                                    for k, r in enumerate(rs)])
            # fit diagnostic: FB's own value estimate F(s,a,z_R).z_R against the exact q of pi_{z_R},
            # i.e. sum_s' M^{pi_z}(s,a,s') r(s') (relative L2 error, first 10 rewards per family)
            for fam, rs in tasks.items():
                errs = []
                for r in rs[:10]:
                    pi, q_hat, _ = fb_policy(Fnet, B, r, tab.rho)
                    q_true = np.einsum("asn,n->sa", tab.successor_measure(pi), r)
                    errs.append(np.linalg.norm(q_hat - q_true) / np.linalg.norm(q_true))
                row["qerr_" + fam] = np.mean(errs)
            per_seed.append(row)
            print(f"[FB] d={d:3d} seed={seed}: FB Bellman loss {log[0][1]:.2f} (step {log[0][0]}) -> "
                  f"{log[-1][1]:.2f} (step {log[-1][0]}), ortho {log[-1][2]:.4f}; normalized regret: "
                  + ", ".join(f"{f} {row[f]:.3f}" for f in tasks)
                  + "; rel. error of F.z_R vs exact q: " + ", ".join(f"{f} {row['qerr_' + f]:.3f}" for f in tasks)
                  + f"  ({time.time() - t1:.1f} s)")
            if seed == seeds[0] and d == ranks[-1]:
                examples[d] = (Fnet, B)
        fb_res[d] = {k: (np.mean([r_[k] for r_ in per_seed]), np.min([r_[k] for r_ in per_seed]),
                         np.max([r_[k] for r_ in per_seed])) for k in per_seed[0]}

    print("\nSummary: normalized regret, sum_s (v* - v^pi) / sum_s (v* - v^random), mean over "
          f"{n_tasks} test rewards (FB: mean over seeds {seeds}, [min, max])")
    print(f"{'method':<14}" + "".join(f"{f:>22}" for f in tasks))
    print(f"{'SF+GPI':<14}" + "".join(f"{results['SF+GPI'][f]:>22.4f}" for f in tasks))
    for d in ranks:
        print(f"{'FB d=' + str(d):<14}" + "".join(
            f"{fb_res[d][f][0]:>10.3f} [{fb_res[d][f][1]:.3f},{fb_res[d][f][2]:.3f}]" for f in tasks))

    if not (args.quick or args.no_figures):
        from plot_style import setup, C
        plt = setup()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1.4, 1, 1]})
        ax = axes[0]
        markers = ["o", "s", "^"]
        for j, fam in enumerate(tasks):
            mean = [fb_res[d][fam][0] for d in ranks]
            lo = [fb_res[d][fam][1] for d in ranks]
            hi = [fb_res[d][fam][2] for d in ranks]
            ax.plot(ranks, mean, marker=markers[j], color=C[j], label=f"FB, {fam}")
            ax.fill_between(ranks, lo, hi, color=C[j], alpha=0.15)
            ax.hlines(results["SF+GPI"][fam], ranks[0] * 0.8, ranks[-1] * 1.25, color=C[j], ls="--", lw=1.2)
            ax.text(ranks[-1] * 1.4, results["SF+GPI"][fam], f"SF+GPI, {fam}", color=C[j],
                    fontsize=8, va="center")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ranks)
        ax.set_xticklabels([str(d) for d in ranks])
        ax.set_xlim(ranks[0] * 0.8, ranks[-1] * 4.5)
        ax.set_xlabel("FB rank d")
        ax.set_ylabel("normalized regret (0 = optimal, 1 = random)")
        ax.set_title("Zero-shot regret on unseen rewards (dashed: SF+GPI)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8)
        # an example goal task: value of the FB policy vs optimal
        d = ranks[-1]
        Fnet, B = examples[d]
        g = env.idx[(9, 4)]
        r = np.eye(env.N)[g]
        pi, _, _ = fb_policy(Fnet, B, r, tab.rho)
        _, v_star = tab.optimal(r)
        v_fb = tab.evaluate(pi, r)
        walls = np.array([[1.0 if ch == "w" else np.nan for ch in row] for row in LAYOUT])
        vmax = v_star.max()
        for ax, v, title in [(axes[1], v_fb, f"FB (d = {d}) zero-shot policy, goal cell G"),
                             (axes[2], v_star, "optimal policy, goal cell G")]:
            img = np.full(walls.shape, np.nan)
            for k, (rr, cc) in enumerate(env.cells):
                img[rr, cc] = v[k]
            ax.imshow(walls, cmap="Greys", vmin=0, vmax=1.5)
            im = ax.imshow(img, cmap="viridis", vmin=0, vmax=vmax)
            ax.text(4, 9, "G", ha="center", va="center", color="white", fontsize=9, weight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=axes[1:], fraction=0.025, pad=0.02, label="value")
        fig.savefig(os.path.join(FIG_DIR, "fb_zero_shot.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"figures written to {FIG_DIR}")
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
