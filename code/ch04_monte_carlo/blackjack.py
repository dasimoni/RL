"""Blackjack exactly as in Sutton & Barto (2018), Example 5.1 -- simulator AND exact solver.

This module is imported by the other Chapter 04 scripts. Run it directly to check it:

    python code/ch04_monte_carlo/blackjack.py [--quick]

which (1) checks this simulator against gymnasium's ``Blackjack-v1`` with ``sab=True`` for
three policies: game by game with a shared card stream, and statistically with independent
streams (mean return vs the exact value, chi-square test on win/draw/loss counts),
(2) prints exact values computed by recursion over the infinite deck, and (3) prints the
exact optimal policy.

Rules (S&B Example 5.1, identical to gymnasium ``Blackjack-v1(sab=True)``):
  * Infinite deck: every card is drawn i.i.d.; values 1 (ace) .. 9 w.p. 1/13 each and
    10 (ten, J, Q, K) w.p. 4/13.
  * Player and dealer get two cards; one dealer card is face up ("showing").
  * An ace is "usable" if it can count as 11 without the hand exceeding 21.
  * A player whose sum is below 12 always hits (hitting can never bust, so this is never
    a real decision). The decision states are therefore
        (player_sum in 12..21, dealer_showing in 1..10, usable_ace in {0, 1})  -> 200 states.
  * Actions: 0 = stick, 1 = hit (gymnasium's encoding).
  * The player busts (> 21): reward -1. When the player sticks, the dealer hits until
    his sum is >= 17 (he sticks on soft 17) and the higher sum wins: +1 / 0 / -1.
  * Natural (ace + ten-card as the first two cards): if the player has one and sticks,
    he wins (+1) unless the dealer also has a natural (draw, 0). As in gymnasium, a natural
    is shown to the player as the ordinary state (21, d, usable ace) and the player may act
    on it (hitting loses natural status); S&B's text settles naturals at the deal. This only
    matters for policies that would hit on 21.
  * All intermediate rewards are 0 and gamma = 1, so the return is the final reward.

Exact solution. Because the deck is infinite and the dealer's play does not depend on
the player, everything can be computed exactly by a short recursion over the player's
hand (raw sum counting aces as 1, has-ace flag). This gives us ground truth to measure the
Monte Carlo estimates against. (S&B estimated the one value of their Example 5.4 by
averaging 10^8 simulated games; the solver computes it, and every other value, exactly.)
"""
from __future__ import annotations

import argparse
import time
from functools import lru_cache

import numpy as np

STICK, HIT = 0, 1
N_SUMS, N_DEALER, N_ACE = 10, 10, 2          # player sum 12..21, dealer 1..10, usable 0/1
STATE_SHAPE = (N_SUMS, N_DEALER, N_ACE)      # array index: [sum-12, dealer-1, usable]
CARD_VALUES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
CARD_PROBS = np.array([1 / 13] * 9 + [4 / 13])
_DECK = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])
_CARDS = list(zip(CARD_VALUES.tolist(), CARD_PROBS.tolist()))


def hand_value(raw: int, has_ace: bool) -> tuple[int, bool]:
    """(sum, usable_ace) of a hand whose cards add up to `raw` with aces counted as 1."""
    if has_ace and raw + 10 <= 21:
        return raw + 10, True
    return raw, False


# --------------------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------------------
class CardShoe:
    """An infinite deck. Cards are pre-drawn in blocks with NumPy because calling a NumPy
    generator once per card is ~50x slower than reading a Python list."""

    def __init__(self, rng: np.random.Generator, block: int = 1 << 16):
        self.rng, self.block = rng, block
        self._refill()

    def _refill(self):
        self.buf = _DECK[self.rng.integers(0, 13, size=self.block)].tolist()
        self.i = 0

    def draw(self) -> int:
        if self.i >= self.block:
            self._refill()
        c = self.buf[self.i]
        self.i += 1
        return c


class Blackjack:
    """S&B Blackjack. Observations are tuples (player_sum, dealer_showing, usable_ace) with
    player_sum >= 12, exactly as in the book; use `state_index` to index NumPy tables."""

    def __init__(self, seed: int | np.random.SeedSequence = 0):
        self.shoe = CardShoe(np.random.default_rng(seed))

    def reset(self, start: tuple[int, int, int] | None = None):
        """Deal a new hand. `start=(player_sum, dealer_showing, usable)` forces the initial
        state (used for exploring starts); such hands are never naturals."""
        draw = self.shoe.draw
        if start is None:
            self.dealer_showing, self.dealer_hidden = draw(), draw()
            c1, c2 = draw(), draw()
            self.raw, self.ace = c1 + c2, (c1 == 1 or c2 == 1)
            self.natural = self.ace and self.raw == 11
            while hand_value(self.raw, self.ace)[0] < 12:   # sums < 12: always hit
                c = draw()
                self.raw += c
                self.ace = self.ace or c == 1
        else:
            player_sum, self.dealer_showing, usable = start
            self.dealer_hidden = draw()
            # A usable-ace hand of sum s has raw sum s-10 with an ace; a no-usable-ace hand
            # behaves identically whether or not it holds an ace (it can never become usable).
            self.raw, self.ace = (player_sum - 10, True) if usable else (player_sum, False)
            self.natural = False
        return self._obs()

    def _obs(self):
        s, usable = hand_value(self.raw, self.ace)
        return (s, self.dealer_showing, int(usable))

    def step(self, action: int):
        """Returns (obs, reward, terminated). Blackjack has no time limit, so there is no
        truncation: every episode ends by termination."""
        if action == HIT:
            c = self.shoe.draw()
            self.raw += c
            self.ace = self.ace or c == 1
            self.natural = False
            if self.raw > 21:                       # raw counts aces as 1: bust for sure
                return self._obs(), -1, True
            return self._obs(), 0, False
        # STICK: the dealer plays out his hand (fixed policy: hit below 17).
        d_raw = self.dealer_showing + self.dealer_hidden
        d_ace = self.dealer_showing == 1 or self.dealer_hidden == 1
        dealer_natural = d_ace and d_raw == 11
        while hand_value(d_raw, d_ace)[0] < 17:
            c = self.shoe.draw()
            d_raw += c
            d_ace = d_ace or c == 1
        if self.natural:
            return self._obs(), (0 if dealer_natural else 1), True
        if d_raw > 21:
            return self._obs(), 1, True
        p, d = hand_value(self.raw, self.ace)[0], hand_value(d_raw, d_ace)[0]
        return self._obs(), (p > d) - (p < d), True


def state_index(obs) -> tuple[int, int, int]:
    """Map an observation (sum, dealer, usable) to an index into a STATE_SHAPE array."""
    s, d, u = obs
    return (s - 12, d - 1, u)


def flat_index(obs) -> int:
    """Map an observation to 0..199 (fast Python-list indexing in the learning loops)."""
    s, d, u = obs
    return ((s - 12) * N_DEALER + (d - 1)) * N_ACE + u


def all_states():
    """The 200 decision states in flat_index order."""
    return [(s, d, u) for s in range(12, 22) for d in range(1, 11) for u in (0, 1)]


def stick_on_20_policy() -> np.ndarray:
    """Probability of HIT for S&B's evaluation policy: stick only on 20 or 21."""
    phit = np.ones(STATE_SHAPE)
    phit[20 - 12:, :, :] = 0.0
    return phit


# --------------------------------------------------------------------------------------
# Exact solver (infinite deck)
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _dealer_from(raw: int, ace: bool) -> tuple:
    """Distribution of the dealer's final total over [17, 18, 19, 20, 21, bust] when his
    current (non-natural) hand is (raw, ace)."""
    total, _ = hand_value(raw, ace)
    out = np.zeros(6)
    if raw > 21:
        out[5] = 1.0
    elif total >= 17:
        out[total - 17] = 1.0
    else:
        for c, p in _CARDS:
            out += p * np.array(_dealer_from(raw + c, ace or c == 1))
    return tuple(out)


@lru_cache(maxsize=None)
def dealer_outcomes(showing: int) -> tuple[np.ndarray, float]:
    """(probabilities of dealer final total [17..21, bust] including naturals as 21,
    probability that the dealer has a natural) given the up-card."""
    dist, p_nat = np.zeros(6), 0.0
    for h, p in _CARDS:
        raw, ace = showing + h, (showing == 1 or h == 1)
        if ace and raw == 11:
            p_nat += p
            dist[4] += p                 # a dealer natural is a 21 for comparison purposes
        else:
            dist += p * np.array(_dealer_from(raw, ace))
    return dist, p_nat


def stick_value(player_sum: int, showing: int) -> float:
    """Expected reward of sticking with a non-natural hand of sum `player_sum` <= 21."""
    dist, _ = dealer_outcomes(showing)
    totals = np.arange(17, 22)
    win = dist[5] + dist[:5][totals < player_sum].sum()
    lose = dist[:5][totals > player_sum].sum()
    return float(win - lose)


def natural_stick_value(showing: int) -> float:
    return 1.0 - dealer_outcomes(showing)[1]


def solve(phit: np.ndarray | None = None, mode: str = "policy", eps: float = 0.0):
    """Exact values by backward recursion over the player's hand.

    mode='policy' : evaluate the stochastic policy phit[sum-12, dealer-1, usable] = P(hit).
    mode='optimal': Bellman optimality (max over actions)            -> q_*, v_*, pi_*.
    mode='eps'    : best epsilon-soft policy (S&B Sec 5.4): value of an epsilon-greedy
                    choice, (1 - eps + eps/2) max_a q + (eps/2) min_a q.

    Returns dict with
      Q[sum-12, dealer-1, usable, action]  exact action values of non-natural hands,
      V[...]                               state values of non-natural hands,
      greedy[...]                          argmax_a Q (1 = hit),
      v_natural[dealer-1]                  value of a natural under the policy,
      J                                    expected return of one game from the deal.
    """
    Q = np.zeros(STATE_SHAPE + (2,))
    V = np.zeros(STATE_SHAPE)
    v_nat = np.zeros(N_DEALER)
    J = 0.0
    for d in range(1, 11):
        val = {}                                  # (raw, ace) -> value
        def get(raw, ace):
            return -1.0 if raw > 21 else val[(raw, ace)]

        def backup(q_stick, q_hit, idx):
            if mode == "optimal":
                return max(q_stick, q_hit)
            if mode == "eps":
                return (1 - eps / 2) * max(q_stick, q_hit) + (eps / 2) * min(q_stick, q_hit)
            h = phit[idx]
            return h * q_hit + (1 - h) * q_stick

        for raw in range(21, 1, -1):              # raw sum only increases -> go backwards
            for ace in (False, True):
                s, usable = hand_value(raw, ace)
                q_hit = sum(p * get(raw + c, ace or c == 1) for c, p in _CARDS)
                if s < 12:                        # automatic hit
                    val[(raw, ace)] = q_hit
                    continue
                q_stick = stick_value(s, d)
                idx = (s - 12, d - 1, int(usable))
                val[(raw, ace)] = backup(q_stick, q_hit, idx)
                Q[idx + (STICK,)], Q[idx + (HIT,)] = q_stick, q_hit
                V[idx] = val[(raw, ace)]
        # Natural (raw 11 with an ace, two cards): same observation (21, d, usable=1).
        q_stick_nat = natural_stick_value(d)
        q_hit_nat = sum(p * get(11 + c, True) for c, p in _CARDS)
        v_nat[d - 1] = backup(q_stick_nat, q_hit_nat, (21 - 12, d - 1, 1))
        # Expected return from the initial deal.
        pd = CARD_PROBS[d - 1]
        for c1, p1 in _CARDS:
            for c2, p2 in _CARDS:
                raw, ace = c1 + c2, (c1 == 1 or c2 == 1)
                v0 = v_nat[d - 1] if (ace and raw == 11) else val[(raw, ace)]
                J += pd * p1 * p2 * v0
    greedy = (Q[..., HIT] > Q[..., STICK]).astype(int)
    return dict(Q=Q, V=V, greedy=greedy, v_natural=v_nat, J=J)


def outcome_probs(phit: np.ndarray) -> np.ndarray:
    """Exact probabilities [P(win), P(draw), P(loss)] of one game from the deal under the
    stochastic policy phit. Same backward recursion as solve(), but each hand carries the
    vector of outcome probabilities instead of the expected reward (so J = P(win) - P(loss))."""
    totals = np.arange(17, 22)
    out = np.zeros(3)
    for d in range(1, 11):
        dist, p_nat = dealer_outcomes(d)
        val = {}

        def get(raw, ace):
            return np.array([0.0, 0.0, 1.0]) if raw > 21 else val[(raw, ace)]

        for raw in range(21, 1, -1):
            for ace in (False, True):
                s, usable = hand_value(raw, ace)
                q_hit = sum(p * get(raw + c, ace or c == 1) for c, p in _CARDS)
                if s < 12:
                    val[(raw, ace)] = q_hit
                    continue
                win = dist[5] + dist[:5][totals < s].sum()
                draw = dist[:5][totals == s].sum()
                h = phit[s - 12, d - 1, int(usable)]
                val[(raw, ace)] = h * q_hit + (1 - h) * np.array([win, draw, 1 - win - draw])
        h = phit[21 - 12, d - 1, 1]
        nat = (h * sum(p * get(11 + c, True) for c, p in _CARDS)
               + (1 - h) * np.array([1 - p_nat, p_nat, 0.0]))
        for c1, p1 in _CARDS:
            for c2, p2 in _CARDS:
                raw, ace = c1 + c2, (c1 == 1 or c2 == 1)
                out += CARD_PROBS[d - 1] * p1 * p2 * (nat if (ace and raw == 11) else val[(raw, ace)])
    return out


def onpolicy_targets(phit: np.ndarray):
    """What first-visit MC prediction converges to: E_pi[G_t | S_t = s] for every state.

    This equals solve(phit)['V'] everywhere except at (21, d, usable): that observation
    lumps together naturals and non-natural 21s, which have different values. The
    observation is therefore not Markov, and MC (which never bootstraps) simply averages
    over the two cases in proportion to how often the policy reaches each of them.
    Also returns P(state is visited in an episode)."""
    sol = solve(phit)
    target, visit = sol["V"].copy(), np.zeros(STATE_SHAPE)
    for d in range(1, 11):
        mass = {}
        m_nat = 0.0
        for c1, p1 in _CARDS:
            for c2, p2 in _CARDS:
                raw, ace = c1 + c2, (c1 == 1 or c2 == 1)
                if ace and raw == 11:
                    m_nat += p1 * p2
                else:
                    mass[(raw, ace)] = mass.get((raw, ace), 0.0) + p1 * p2
        h_nat = phit[21 - 12, d - 1, 1]
        for c, p in _CARDS:                       # a natural that hits loses its status
            mass[(11 + c, True)] = mass.get((11 + c, True), 0.0) + m_nat * h_nat * p
        for raw in range(2, 22):                  # forward in raw sum
            for ace in (False, True):
                m = mass.get((raw, ace), 0.0)
                if m == 0.0:
                    continue
                s, usable = hand_value(raw, ace)
                h = 1.0
                if s >= 12:
                    idx = (s - 12, d - 1, int(usable))
                    visit[idx] += m
                    h = phit[idx]
                for c, p in _CARDS:
                    if raw + c <= 21:
                        key = (raw + c, ace or c == 1)
                        mass[key] = mass.get(key, 0.0) + m * h * p
        idx = (21 - 12, d - 1, 1)
        m_non = visit[idx]
        visit[idx] += m_nat
        target[idx] = (m_non * sol["V"][idx] + m_nat * sol["v_natural"][d - 1]) / (m_non + m_nat)
    pd = CARD_PROBS[None, :, None]
    return target, visit * pd                     # visit prob includes P(dealer card)


def value_of_greedy(Q_table) -> float:
    """Exact expected return J of the deterministic greedy policy w.r.t. a Q table of
    shape STATE_SHAPE + (2,) (ties -> stick)."""
    phit = (Q_table[..., HIT] > Q_table[..., STICK]).astype(float)
    return solve(phit)["J"]


def format_policy(policy: np.ndarray, usable: int) -> str:
    """ASCII picture of a deterministic policy (H = hit, S = stick), rows 21 -> 12."""
    lines = ["      dealer: A  2  3  4  5  6  7  8  9 10"]
    for s in range(21, 11, -1):
        row = "".join(f"  {'H' if policy[s - 12, d - 1, usable] else 'S'}" for d in range(1, 11))
        lines.append(f"  sum {s:2d}:{row}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------------------
def _act(phit, obs, u01):
    """Action of the policy phit (P(hit) table) in observation obs, given a uniform number."""
    if obs[0] < 12:                                  # gymnasium shows sums < 12; always hit
        return HIT
    return HIT if u01 < phit[state_index(obs)] else STICK


def _play_own(phit, n, card_seed, act_seed):
    """n games in our simulator. Returns rewards and number of decisions (sum >= 12)."""
    import random
    env, arng = Blackjack(card_seed), random.Random(act_seed)
    rewards, decisions = np.empty(n), np.empty(n, dtype=int)
    for i in range(n):
        obs, done, k = env.reset(), False, 0
        while not done:
            k += 1
            obs, r, done = env.step(_act(phit, obs, arng.random()))
        rewards[i], decisions[i] = r, k
    return rewards, decisions


def _play_gym(phit, n, gym_seed, act_seed, card_seed=None):
    """n games in gymnasium Blackjack-v1(sab=True). If card_seed is given, gymnasium's card
    source is replaced by our CardShoe with that seed. Both simulators draw cards in the same
    order (dealer showing, dealer hidden, player, player, player hits, dealer hits), so with a
    shared card stream and a shared action stream the two must agree game by game."""
    import random
    import gymnasium as gym
    import gymnasium.envs.toy_text.blackjack as gbj
    original = gbj.draw_card
    if card_seed is not None:
        shoe = CardShoe(np.random.default_rng(card_seed))
        gbj.draw_card = lambda np_random: shoe.draw()
    try:
        env, arng = gym.make("Blackjack-v1", sab=True), random.Random(act_seed)
        rewards, decisions = np.empty(n), np.empty(n, dtype=int)
        obs, _ = env.reset(seed=gym_seed)
        for i in range(n):
            if i > 0:
                obs, _ = env.reset()
            terminated = truncated = False
            k = 0
            while not (terminated or truncated):
                u = 0.0
                if obs[0] >= 12:                     # a real decision: consume one uniform
                    k += 1
                    u = arng.random()
                obs, r, terminated, truncated, _ = env.step(_act(phit, obs, u))
            rewards[i], decisions[i] = r, k
        env.close()
    finally:
        gbj.draw_card = original
    return rewards, decisions


def _wdl(rewards):
    return [int((rewards == 1).sum()), int((rewards == 0).sum()), int((rewards == -1).sum())]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n_ind = 5_000 if args.quick else 200_000          # independent-stream games per simulator
    n_cpl = 2_000 if args.quick else 100_000          # coupled (shared-stream) games
    print(f"Blackjack self-check  seed={args.seed}  independent games per simulator and policy="
          f"{n_ind:,}  coupled games per policy={n_cpl:,}")
    from scipy.stats import chisquare

    phit = stick_on_20_policy()
    exact = solve(phit)
    opt = solve(mode="optimal")
    policies = {"stick on 20/21": phit, "optimal pi_*": opt["greedy"].astype(float),
                "uniform random": np.full(STATE_SHAPE, 0.5)}
    print("\nCheck against gymnasium Blackjack-v1(sab=True).")
    print("  (a) coupled: same card stream and same action stream -> games must agree exactly")
    print("  (b) independent streams: mean return vs exact J (z-score), and a chi-square")
    print("      goodness-of-fit test of the win/draw/loss counts against the exact probabilities")
    for name, ph in policies.items():
        J = solve(ph)["J"]
        wdl = outcome_probs(ph)
        assert abs(wdl[0] - wdl[2] - J) < 1e-12           # the two exact recursions agree
        r1, k1 = _play_own(ph, n_cpl, args.seed + 7, args.seed + 8)
        r2, k2 = _play_gym(ph, n_cpl, args.seed, args.seed + 8, card_seed=args.seed + 7)
        mismatch = int(((r1 != r2) | (k1 != k2)).sum())
        t0 = time.time()
        ro, _ = _play_own(ph, n_ind, args.seed, args.seed + 1)
        t_own = time.time() - t0
        t0 = time.time()
        rg, _ = _play_gym(ph, n_ind, args.seed + 2, args.seed + 3)
        t_gym = time.time() - t0
        print(f"\n  policy: {name}   exact J = {J:+.5f}; exact W/D/L = "
              f"{wdl[0]:.4f}/{wdl[1]:.4f}/{wdl[2]:.4f}")
        print(f"    (a) coupled games that differ (reward or #decisions): {mismatch} of {n_cpl:,}")
        for lab, g, t in (("own simulator", ro, t_own), ("gymnasium", rg, t_gym)):
            se = g.std(ddof=1) / np.sqrt(n_ind)
            w, d, l = (np.array(_wdl(g)) / n_ind).tolist()
            p_val = chisquare(_wdl(g), wdl * n_ind).pvalue
            print(f"    (b) {lab:14s}: mean {g.mean():+.5f} +- {se:.5f} (z = {(g.mean() - J) / se:+.2f}); "
                  f"W/D/L = {w:.4f}/{d:.4f}/{l:.4f} (chi-square p = {p_val:.3f}); "
                  f"{1e6 * t / n_ind:.1f} us/game")

    target, visit = onpolicy_targets(phit)
    print("\nThe observation (21, d, usable ace) is not Markov: naturals and non-natural 21s differ.")
    for d in (1, 2, 10):
        i = (21 - 12, d - 1, 1)
        p_nat = (8 / 169) / (visit[i] / CARD_PROBS[d - 1])
        print(f"  dealer {d:2d}: v(non-natural 21) = {exact['V'][i]:.4f}, v(natural) = "
              f"{exact['v_natural'][d - 1]:.4f}, share of visits that are naturals = {p_nat:.3f}, "
              f"first-visit MC limit = {target[i]:.4f}")

    v_ex54 = exact["V"][13 - 12, 2 - 1, 1]
    print(f"\nExact v(sum 13, dealer 2, usable ace) under stick-on-20: {v_ex54:.6f}"
          "  (S&B Example 5.4 quote: -0.27726 from 1e8 episodes)")

    print(f"\nExact optimal J* = {opt['J']:+.5f};  J(stick-on-20) = {exact['J']:+.5f}")
    print("Optimal policy, usable ace:\n" + format_policy(opt["greedy"], 1))
    print("Optimal policy, no usable ace:\n" + format_policy(opt["greedy"], 0))
    for eps in (0.05, 0.1, 0.2):
        se = solve(mode="eps", eps=eps)
        print(f"Best eps-soft policy, eps={eps:.2f}: J = {se['J']:+.5f}, greedy part equals "
              f"pi_*: {bool((se['greedy'] == opt['greedy']).all())}")


if __name__ == "__main__":
    main()
