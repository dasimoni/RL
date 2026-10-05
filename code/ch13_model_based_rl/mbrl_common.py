"""Shared building blocks for the Chapter 13 Pendulum experiments (not run directly).

* ``GaussianEnsemble``  : B probabilistic neural-network dynamics models (Section 3), trained in
                          parallel with batched matrix multiplies.  Each member outputs a Gaussian
                          over the state *change*  Delta s = s' - s  (Eq. 13.6), with the learned
                          soft bounds on the log-variance used by PETS.
* ``train_ensemble``    : Gaussian negative log-likelihood training (Eq. 13.7), optional bootstrap.
* ``MPCPlanner``        : model-predictive control (Section 4) with random shooting or the
                          cross-entropy method, and PETS' TS-infinity trajectory sampling (Section 5).
* ``pendulum_reward``   : the *known* Pendulum-v1 reward, computed from an observation and action.

Pendulum-v1 observation: (cos th, sin th, thdot); action: torque u in [-2, 2];
reward r(s, u) = -(th^2 + 0.1 thdot^2 + 0.001 u^2) with th normalised to [-pi, pi], evaluated at
the state *before* the step (this matches gymnasium's implementation exactly).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------
# Reward and a hand-written swing-up controller (used only to generate *test* trajectories)
# ---------------------------------------------------------------------------------------------
def pendulum_reward(obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """Exact Pendulum-v1 reward for a batch of observations (..., 3) and actions (..., 1)."""
    th = torch.atan2(obs[..., 1], obs[..., 0])            # angle in [-pi, pi]; 0 = upright
    thdot = obs[..., 2]
    u = act[..., 0].clamp(-2.0, 2.0)
    return -(th ** 2 + 0.1 * thdot ** 2 + 0.001 * u ** 2)


def energy_swingup_action(obs: np.ndarray, rng: np.random.Generator, noise: float = 0.3) -> np.ndarray:
    """A crude energy-pumping controller plus a PD stabiliser near the top.

    It is NOT used by any learning agent; compounding_error.py uses it to generate test
    trajectories that visit high-energy states which a random policy rarely reaches
    (a deliberate distribution shift).
    """
    c, s, thdot = obs
    th = math.atan2(s, c)
    if c > 0.85:                                   # near upright: PD control
        u = -8.0 * th - 1.5 * thdot
    else:                                          # pump energy in the direction of motion
        u = 2.0 * np.sign(thdot if abs(thdot) > 1e-3 else 1.0)
    u += noise * rng.standard_normal()
    return np.array([np.clip(u, -2.0, 2.0)], dtype=np.float32)


class TruePendulumDynamics:
    """The real Pendulum-v1 equations of motion, written in torch with the same ``step`` interface
    as the learned ensemble.  Planning with it ("oracle MPC") tells us how good the *planner* is,
    so that the gap between oracle MPC and PETS can be attributed to model error."""

    B = 1

    @torch.no_grad()
    def step(self, obs: torch.Tensor, act: torch.Tensor, sample: bool = False) -> torch.Tensor:
        g, dt = 10.0, 0.05
        th = torch.atan2(obs[..., 1], obs[..., 0])
        thdot = obs[..., 2]
        u = act[..., 0].clamp(-2.0, 2.0)
        newthdot = (thdot + (3 * g / 2 * torch.sin(th) + 3.0 * u) * dt).clamp(-8.0, 8.0)
        newth = th + newthdot * dt
        return torch.stack([torch.cos(newth), torch.sin(newth), newthdot], dim=-1)


# ---------------------------------------------------------------------------------------------
# Probabilistic ensemble (Section 3, "PE" in PETS terminology)
# ---------------------------------------------------------------------------------------------
class GaussianEnsemble(nn.Module):
    """B independent MLPs evaluated together.  Member b maps normalised (s, a) to the mean and
    log-variance of a diagonal Gaussian over the normalised state change.

    Shapes: inputs are (B, N, in_dim); outputs are (B, N, out_dim).  Weights are (B, in, out),
    so one ``torch.baddbmm`` evaluates every member on its own batch.
    """

    def __init__(self, in_dim: int, out_dim: int, n_members: int = 5, hidden: int = 64,
                 n_hidden: int = 2, probabilistic: bool = True):
        super().__init__()
        self.B, self.in_dim, self.out_dim = n_members, in_dim, out_dim
        self.probabilistic = probabilistic
        dims = [in_dim] + [hidden] * n_hidden + [2 * out_dim if probabilistic else out_dim]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            w = torch.empty(n_members, d_in, d_out)
            for b in range(n_members):          # independent random init per member (diversity!)
                nn.init.trunc_normal_(w[b], std=1.0 / (2.0 * math.sqrt(d_in)))
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(torch.zeros(n_members, 1, d_out)))
        # Learned soft bounds on the log-variance (PETS, appendix): stop the variance from
        # collapsing to 0 or exploding on inputs far from the data.
        self.max_logvar = nn.Parameter(torch.full((1, 1, out_dim), 0.5))
        self.min_logvar = nn.Parameter(torch.full((1, 1, out_dim), -10.0))
        # Normalisation statistics (set from data by ``set_normalizer``).
        self.register_buffer("in_mu", torch.zeros(in_dim))
        self.register_buffer("in_sd", torch.ones(in_dim))
        self.register_buffer("out_mu", torch.zeros(out_dim))
        self.register_buffer("out_sd", torch.ones(out_dim))

    def set_normalizer(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.in_mu.copy_(x.mean(0)); self.in_sd.copy_(x.std(0).clamp_min(1e-6))
        self.out_mu.copy_(y.mean(0)); self.out_sd.copy_(y.std(0).clamp_min(1e-6))

    def forward(self, xn: torch.Tensor):
        """xn: normalised inputs (B, N, in).  Returns (mean, logvar) in normalised target units."""
        h = xn
        n_layers = len(self.weights)
        for i in range(n_layers):
            h = torch.baddbmm(self.biases[i], h, self.weights[i])
            if i < n_layers - 1:
                h = F.silu(h)                    # "swish", as in PETS
        if not self.probabilistic:
            return h, None
        mean, logvar = h[..., : self.out_dim], h[..., self.out_dim:]
        # Soft clamp:  min_logvar <= logvar <= max_logvar, differentiably.
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    @torch.no_grad()
    def step(self, obs: torch.Tensor, act: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """One simulated step for a (B, N, obs_dim) batch: member b advances its own N particles.

        sample=True draws  s' = s + Delta,  Delta ~ N(mu_b(s,a), sigma_b^2(s,a))   (aleatoric noise)
        sample=False uses the member's mean prediction.
        """
        x = torch.cat([obs, act], dim=-1)
        mean, logvar = self((x - self.in_mu) / self.in_sd)
        if sample and logvar is not None:
            mean = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return obs + self.out_mu + self.out_sd * mean


def fit_ensemble(model: GaussianEnsemble, opt: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor,
                 n_steps: int, batch_size: int, rng: np.random.Generator, bootstrap: bool = True) -> float:
    """Fit every member by maximum likelihood (Eq. 13.7) to the regression data (x, y).

    With ``bootstrap`` each member trains on its own resample (with replacement) of the n points,
    so the ensemble's disagreement reflects how much the fit depends on which data happened to be
    collected.  Returns the mean squared error of the members' mean predictions on all the data
    (normalised units): a training-fit diagnostic, not a validation score.
    """
    model.set_normalizer(x, y)
    xn = (x - model.in_mu) / model.in_sd
    yn = (y - model.out_mu) / model.out_sd
    n = len(x)
    idx_sets = [rng.integers(0, n, size=n) if bootstrap else np.arange(n) for _ in range(model.B)]
    for _ in range(n_steps):
        batch = np.stack([s[rng.integers(0, n, size=batch_size)] for s in idx_sets])   # (B, bs)
        bx, by = xn[batch], yn[batch]                                                  # (B, bs, d)
        mean, logvar = model(bx)
        if model.probabilistic:
            # Gaussian NLL up to constants: sum_d [ (mu - y)^2 / sigma^2 + log sigma^2 ]   (Eq. 13.7)
            loss = (((mean - by) ** 2) * torch.exp(-logvar) + logvar).sum(-1).mean()
            loss = loss + 0.01 * (model.max_logvar.sum() - model.min_logvar.sum())
        else:
            loss = ((mean - by) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        mean, _ = model(xn.unsqueeze(0).expand(model.B, -1, -1))
        return float(((mean - yn) ** 2).mean())


def train_ensemble(model: GaussianEnsemble, opt: torch.optim.Optimizer, obs: np.ndarray, act: np.ndarray,
                   next_obs: np.ndarray, n_steps: int, batch_size: int, rng: np.random.Generator,
                   bootstrap: bool = True) -> float:
    """Dynamics-model training: inputs (s, a), targets the state change s' - s (Eq. 13.6)."""
    x = torch.as_tensor(np.concatenate([obs, act], axis=1), dtype=torch.float32)
    y = torch.as_tensor(next_obs - obs, dtype=torch.float32)
    return fit_ensemble(model, opt, x, y, n_steps, batch_size, rng, bootstrap)


# ---------------------------------------------------------------------------------------------
# Model-predictive control: random shooting and CEM, with TS-infinity particles (Sections 4-5)
# ---------------------------------------------------------------------------------------------
class MPCPlanner:
    """Plan an action sequence of length H against the learned model, execute the first action,
    re-plan at the next step (receding horizon).

    method = "cem": cross-entropy method, ``iters`` rounds of ``pop`` Gaussian samples, refit to the
                    ``elites`` best, warm-started from the shifted previous solution (Alg. 13.2).
    method = "rs" : random shooting, ``pop * iters`` uniform sequences, keep the best (Alg. 13.1).
    method = "random": no planning at all, a uniformly random torque (a baseline that can be run on
                    exactly the same initial states as the planners).

    Each candidate sequence is scored by the average return of ``particles`` simulated
    trajectories.  Particle p always uses ensemble member p mod B ("TS-infinity"), so the spread of
    returns across particles mixes epistemic (which member) and aleatoric (sampled noise)
    uncertainty.
    """

    def __init__(self, model: GaussianEnsemble, reward_fn, horizon: int = 25, pop: int = 100,
                 elites: int = 10, iters: int = 4, particles: int = 10, alpha: float = 0.1,
                 act_low: float = -2.0, act_high: float = 2.0, method: str = "cem",
                 sample_noise: bool = True):
        assert particles % model.B == 0, "particles must be a multiple of the ensemble size"
        self.model, self.reward_fn = model, reward_fn
        self.H, self.pop, self.elites, self.iters, self.P = horizon, pop, elites, iters, particles
        self.alpha, self.low, self.high, self.method = alpha, act_low, act_high, method
        self.sample_noise = sample_noise
        self.reset()

    def reset(self):
        self.mean = torch.zeros(self.H)                       # previous solution (shifted each step)

    @torch.no_grad()
    def evaluate(self, obs: np.ndarray, seqs: torch.Tensor) -> torch.Tensor:
        """Average predicted return of each action sequence in seqs (n, H) from state obs."""
        n, B = seqs.shape[0], self.model.B
        k = self.P // B                                        # particles per member per candidate
        s = torch.as_tensor(obs, dtype=torch.float32).expand(B, n * k, -1).clone()
        # row j of member b belongs to candidate j // k
        acts = seqs.repeat_interleave(k, dim=0)                # (n*k, H)
        ret = torch.zeros(B, n * k)
        for t in range(self.H):
            a = acts[:, t].view(1, n * k, 1).expand(B, -1, -1)
            ret += self.reward_fn(s, a)                        # r(s_t, a_t): reward of the state we are in
            s = self.model.step(s, a, sample=self.sample_noise)
        return ret.view(B, n, k).mean(dim=(0, 2))              # average over all P particles

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.method == "random":
            return torch.empty(1).uniform_(self.low, self.high).numpy().astype(np.float32)
        if self.method == "rs":
            seqs = torch.empty(self.pop * self.iters, self.H).uniform_(self.low, self.high)
            best = seqs[self.evaluate(obs, seqs).argmax()]
            return best[:1].numpy().astype(np.float32)
        mean = self.mean.clone()
        var = torch.full((self.H,), ((self.high - self.low) / 4.0) ** 2)
        for _ in range(self.iters):
            seqs = (mean + var.sqrt() * torch.randn(self.pop, self.H)).clamp(self.low, self.high)
            elite = seqs[self.evaluate(obs, seqs).topk(self.elites).indices]
            mean = self.alpha * mean + (1 - self.alpha) * elite.mean(0)
            var = self.alpha * var + (1 - self.alpha) * elite.var(0, unbiased=False)
        self.mean = torch.cat([mean[1:], torch.zeros(1)])     # warm start for the next step
        return mean[:1].numpy().astype(np.float32)
