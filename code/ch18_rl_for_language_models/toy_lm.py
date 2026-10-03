"""A tiny autoregressive "language model" and the sequence utilities that RLHF needs.

Chapter 18, Section 1.  Imported by rlhf_toy.py and grpo_rlvr.py (this file is a library
with nothing to run on its own).

The policy is a conditional GRU:  pi_theta(y_t | x, y_<t).

* The prompt x enters as a feature vector `cond` (a one-hot topic, or the one-hot digits of an
  arithmetic problem).  It initialises the hidden state and is also fed at every step, so the
  model never has to memorise it.
* Responses are token sequences y = (y_1, ..., y_L) over a vocabulary whose last token is EOS.
  A special BOS token (index = vocab size) is used only as the first *input*.
* Tensors of responses have a fixed width T (the generation budget).  After EOS a row is padded
  with EOS, and `mask[b, t] = 1` exactly for the decisions the policy really took (the tokens up
  to and including the first EOS).  Every sum over tokens in the chapter is a masked sum.

Token-level MDP view (Section 1.2): state s_t = (x, y_<t), action a_t = y_t, deterministic
transition "append the token", episode ends at EOS or when the budget T is used up.  The budget
is part of the task, so hitting it is a genuine termination (no bootstrapping), unlike a
Gymnasium time limit.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUTrunk(nn.Module):
    """Embeds (prompt features, previous token) and runs a single-layer GRU."""

    def __init__(self, n_input_tokens: int, cond_dim: int, emb: int = 32, hidden: int = 64):
        super().__init__()
        self.emb = nn.Embedding(n_input_tokens, emb)
        self.cond = nn.Linear(cond_dim, emb)
        self.init_h = nn.Linear(cond_dim, hidden)
        self.gru = nn.GRU(2 * emb, hidden, batch_first=True)

    def forward(self, cond, inp, h=None):
        B, T = inp.shape
        c = self.cond(cond).unsqueeze(1).expand(B, T, -1)
        z = torch.cat([self.emb(inp), c], dim=-1)
        if h is None:
            h = torch.tanh(self.init_h(cond)).unsqueeze(0)
        out, h = self.gru(z, h)
        return out, h


class PolicyLM(nn.Module):
    """pi_theta(y_t | x, y_<t) over `vocab` output tokens; EOS = vocab - 1, BOS = vocab."""

    def __init__(self, vocab: int, cond_dim: int, emb: int = 32, hidden: int = 64):
        super().__init__()
        self.vocab, self.eos, self.bos = vocab, vocab - 1, vocab
        self.trunk = GRUTrunk(vocab + 1, cond_dim, emb, hidden)
        self.head = nn.Linear(hidden, vocab)

    def logits(self, cond, tokens):
        """Teacher-forced logits: entry [b, t] predicts tokens[b, t] from (x, tokens[b, :t])."""
        bos = torch.full_like(tokens[:, :1], self.bos)
        out, _ = self.trunk(cond, torch.cat([bos, tokens[:, :-1]], dim=1))
        return self.head(out)

    @torch.no_grad()
    def sample(self, cond, T: int, temperature: float = 1.0, greedy: bool = False):
        """Autoregressive sampling with the GRU state carried along (no recomputation).

        Returns tokens (B, T) padded with EOS, and mask (B, T) of the real decisions.
        """
        B = cond.shape[0]
        inp = torch.full((B, 1), self.bos, dtype=torch.long)
        tokens = torch.full((B, T), self.eos, dtype=torch.long)
        mask = torch.zeros(B, T)
        alive = torch.ones(B, dtype=torch.bool)
        h = None
        for t in range(T):
            out, h = self.trunk(cond, inp, h)
            logits = self.head(out[:, 0])
            if greedy:
                y = logits.argmax(-1)
            else:
                y = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)[:, 0]
            y = torch.where(alive, y, torch.full_like(y, self.eos))
            tokens[:, t] = y
            mask[:, t] = alive.float()
            alive = alive & (y != self.eos)
            if not alive.any():
                break
            inp = y.unsqueeze(1)
        return tokens, mask


class ScalarModel(nn.Module):
    """A GRU with a scalar head: the reward model r_phi(x, y) and the PPO critic V(s_t).

    Input is [BOS, y_1, ..., y_T]; output position j has read (x, y_1..y_j).
    * reward: the output at the position of the last real token (after EOS, or the last token
      of a truncated response), as in InstructGPT's reward model;
    * values: output j-1 is V(s_j) for the state s_j = (x, y_<j), j = 1..T.
    """

    def __init__(self, vocab: int, cond_dim: int, emb: int = 32, hidden: int = 64):
        super().__init__()
        self.vocab, self.bos = vocab, vocab
        self.trunk = GRUTrunk(vocab + 1, cond_dim, emb, hidden)
        self.head = nn.Linear(hidden, 1)

    def per_position(self, cond, tokens):
        bos = torch.full_like(tokens[:, :1], self.bos)
        out, _ = self.trunk(cond, torch.cat([bos, tokens], dim=1))
        return self.head(out).squeeze(-1)  # (B, T+1)

    def score(self, cond, tokens, mask):
        out = self.per_position(cond, tokens)
        last = mask.sum(1).long()  # number of real tokens = index of the last one in `out`
        return out.gather(1, last.unsqueeze(1)).squeeze(1)

    def values(self, cond, tokens):
        return self.per_position(cond, tokens)[:, :-1]  # (B, T): V(s_1), ..., V(s_T)


# ---------------------------------------------------------------------------------------
# Sequence quantities
# ---------------------------------------------------------------------------------------
def token_logprobs(model: PolicyLM, cond, tokens, mask):
    """log pi(y_t | s_t) for every position, zeroed where mask = 0.  Shape (B, T)."""
    logp = F.log_softmax(model.logits(cond, tokens), dim=-1)
    return logp.gather(2, tokens.unsqueeze(-1)).squeeze(-1) * mask


def token_kl_exact(model: PolicyLM, ref: PolicyLM, cond, tokens, mask):
    """Per-position exact KL( pi(.|s_t) || pi_ref(.|s_t) ), summed over the vocabulary.

    Summed over t along responses sampled from pi, this is an unbiased, low-variance estimate
    of the sequence-level KL(pi(.|x) || pi_ref(.|x)) (Section 1.3, eq. 18.5).
    """
    lp = F.log_softmax(model.logits(cond, tokens), dim=-1)
    with torch.no_grad():
        lq = F.log_softmax(ref.logits(cond, tokens), dim=-1)
    return (lp.exp() * (lp - lq)).sum(-1) * mask


def token_entropy(model: PolicyLM, cond, tokens, mask):
    lp = F.log_softmax(model.logits(cond, tokens), dim=-1)
    return -(lp.exp() * lp).sum(-1) * mask


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def clone_frozen(model: nn.Module, cls, *args, **kwargs):
    """A frozen copy (used for pi_ref)."""
    m = cls(*args, **kwargs)
    m.load_state_dict(model.state_dict())
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    return m


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
