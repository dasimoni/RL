"""Kuhn poker: game tree, exact evaluation, best response, exploitability and the
sequence-form linear program (Chapter 17, section 10).

Rules (Kuhn, 1950). A deck of three cards J < Q < K. Each player antes 1 chip and is
dealt one private card (6 equally likely deals). Player 1 acts first; each player can
pass ('p': check, or fold when facing a bet) or bet ('b': bet 1 chip, or call a bet).

    history   who acts / what happens
    ''        player 1: check (p) or bet (b)
    'p'       player 2: check (-> showdown for 1) or bet
    'b'       player 2: fold (-> player 1 wins 1) or call (-> showdown for 2)
    'pb'      player 1: fold (-> player 2 wins 1) or call (-> showdown for 2)
    'pp', 'bp', 'bb', 'pbp', 'pbb' are terminal.

An information set is the acting player's card plus the public history, written e.g.
'K:' (player 1 holding K at the root) or 'Q:pb' (player 1 holding Q, facing a bet after
checking). Each player has 6 information sets with 2 actions each.

A behavioural strategy is a dict {infoset: np.array([P(pass), P(bet)])}.
All utilities are for player 1 (the game is zero-sum). The value of the game for
player 1 is -1/18 (Kuhn, 1950).

Run `python code/ch17_multi_agent_rl/kuhn.py` for a self-test that solves the game
with the sequence-form LP and checks the known equilibrium family.
"""
import argparse
import itertools

import numpy as np
from scipy.optimize import linprog

CARDS = "JQK"
ACTIONS = "pb"
DEALS = list(itertools.permutations(range(3), 2))    # (card of P1, card of P2), prob 1/6 each
TERMINALS = ("pp", "bp", "bb", "pbp", "pbb")
P1_HISTORIES = ("", "pb")
P2_HISTORIES = ("p", "b")
GAME_VALUE = -1.0 / 18.0


def player(history):
    """1 or 2 for decision nodes; 0 for terminal histories."""
    if history in TERMINALS:
        return 0
    return 1 if len(history) % 2 == 0 else 2


def infoset_key(card, history):
    return f"{CARDS[card]}:{history}"


def infosets(p):
    hs = P1_HISTORIES if p == 1 else P2_HISTORIES
    return [infoset_key(c, h) for h in hs for c in range(3)]


def utility(c1, c2, history):
    """Payoff to player 1 at a terminal history."""
    if history == "bp":
        return 1.0                       # player 2 folded
    if history == "pbp":
        return -1.0                      # player 1 folded
    stake = 2.0 if history in ("bb", "pbb") else 1.0
    return stake if c1 > c2 else -stake


def uniform_strategy(p):
    return {I: np.array([0.5, 0.5]) for I in infosets(p)}


# ----------------------------------------------------------------------------------
# Exact evaluation
# ----------------------------------------------------------------------------------
def _value(c1, c2, h, s1, s2):
    p = player(h)
    if p == 0:
        return utility(c1, c2, h)
    sigma = (s1 if p == 1 else s2)[infoset_key(c1 if p == 1 else c2, h)]
    return sum(sigma[k] * _value(c1, c2, h + a, s1, s2) for k, a in enumerate(ACTIONS))


def expected_value(s1, s2):
    """Expected payoff to player 1 when both players use the given behavioural strategies."""
    return sum(_value(c1, c2, "", s1, s2) for c1, c2 in DEALS) / len(DEALS)


# ----------------------------------------------------------------------------------
# Best response and exploitability (section 10.2)
# ----------------------------------------------------------------------------------
def best_response(sigma_opp, br_player):
    """Exact best response of `br_player` to the opponent's behavioural strategy.

    At an information set I the best responder cannot see the opponent's card, so it
    picks the action maximising the counterfactual value
        sum_{h in I} eta_{-i}(h) * u_i(h.a),
    where eta_{-i}(h) is the chance-times-opponent probability of reaching h. Deeper
    information sets are decided first so that u_i(h.a) can use the best responder's
    own later choices (perfect recall makes this order well defined).
    Returns (pure behavioural strategy, value of the best response for br_player).
    """
    sign = 1.0 if br_player == 1 else -1.0
    br = {}
    my_hists = P1_HISTORIES if br_player == 1 else P2_HISTORIES

    def val(c1, c2, h):                 # value for br_player at node h (BR fixed below h)
        p = player(h)
        if p == 0:
            return sign * utility(c1, c2, h)
        if p == br_player:
            I = infoset_key(c1 if p == 1 else c2, h)
            return val(c1, c2, h + ACTIONS[int(np.argmax(br[I]))])
        sigma = sigma_opp[infoset_key(c1 if p == 1 else c2, h)]
        return sum(sigma[k] * val(c1, c2, h + a) for k, a in enumerate(ACTIONS))

    def reach_opp(c1, c2, h):            # chance x opponent probability of reaching h
        prob = 1.0 / len(DEALS)
        for k in range(len(h)):
            if player(h[:k]) != br_player:
                opp_card = c1 if player(h[:k]) == 1 else c2
                prob *= sigma_opp[infoset_key(opp_card, h[:k])][ACTIONS.index(h[k])]
        return prob

    for h in sorted(my_hists, key=len, reverse=True):      # deepest information sets first
        for my_card in range(3):
            q = np.zeros(2)
            for opp_card in range(3):
                if opp_card == my_card:
                    continue
                c1, c2 = (my_card, opp_card) if br_player == 1 else (opp_card, my_card)
                w = reach_opp(c1, c2, h)
                for k, a in enumerate(ACTIONS):
                    q[k] += w * val(c1, c2, h + a)
            br[infoset_key(my_card, h)] = np.eye(2)[int(np.argmax(q))]
    value = sum(val(c1, c2, "") for c1, c2 in DEALS) / len(DEALS)
    return br, value


def exploitability(s1, s2):
    """(BR value of player 1 vs s2 + BR value of player 2 vs s1) / 2  >= 0.

    In a zero-sum game the two best-response values sum to
    max_{s1'} EV(s1', s2) - min_{s2'} EV(s1, s2') >= 0 (NashConv), which is zero exactly at
    a Nash equilibrium; exploitability is half of it (the average amount a best
    responder wins beyond the game value)."""
    _, v1 = best_response(s2, 1)
    _, v2 = best_response(s1, 2)
    return 0.5 * (v1 + v2)


# ----------------------------------------------------------------------------------
# Mixtures of strategies -> behavioural strategy (Kuhn's theorem; used by PSRO)
# ----------------------------------------------------------------------------------
def own_reach(strategy, p, card, h):
    """Probability that player p's own actions lead to history h (its 'sequence weight')."""
    prob = 1.0
    for k in range(len(h)):
        if player(h[:k]) == p:
            prob *= strategy[infoset_key(card, h[:k])][ACTIONS.index(h[k])]
    return prob


def mixture_to_behavioural(strategies, weights, p):
    """Behavioural strategy that is outcome-equivalent to playing strategies[k] with
    probability weights[k] (choice made once, before the game):
        pi(a | I) = sum_k w_k x_k(I) pi_k(a | I) / sum_k w_k x_k(I),
    where x_k(I) is the probability that strategy k's own actions reach I. This is the
    same reach-weighted average that CFR uses for its average strategy (Eq. 10.6)."""
    out = {}
    hs = P1_HISTORIES if p == 1 else P2_HISTORIES
    for h in hs:
        for c in range(3):
            I = infoset_key(c, h)
            num, den = np.zeros(2), 0.0
            for s, w in zip(strategies, weights):
                x = w * own_reach(s, p, c, h)
                num += x * s[I]
                den += x
            out[I] = num / den if den > 1e-15 else np.array([0.5, 0.5])
    return out


# ----------------------------------------------------------------------------------
# Sequence-form linear program (Koller, Megiddo & von Stengel, 1994)
# ----------------------------------------------------------------------------------
def _sequences(p):
    """Sequences of player p: the empty sequence plus (infoset, action) pairs, and for
    each infoset the index of its parent sequence."""
    seqs = [None]                         # index 0 = empty sequence
    parent = {}
    hs = P1_HISTORIES if p == 1 else P2_HISTORIES
    for h in hs:                          # parents always precede children in this order
        for c in range(3):
            I = infoset_key(c, h)
            # the player's own last action before h (player 1 at 'pb' previously chose 'p')
            prev = [k for k in range(len(h)) if player(h[:k]) == p]
            parent[I] = 0 if not prev else seqs.index((infoset_key(c, h[:prev[-1]]), h[prev[-1]]))
            for a in ACTIONS:
                seqs.append((I, a))
    return seqs, parent


def _last_seq(seqs, p, card, h):
    prev = [k for k in range(len(h)) if player(h[:k]) == p]
    if not prev:
        return 0
    k = prev[-1]
    return seqs.index((infoset_key(card, h[:k]), h[k]))


def sequence_form():
    """Return (A, E, e, F, f, seqs1, parent1, seqs2, parent2) for the sequence-form game
    max_x min_y x^T A y  s.t.  E x = e, F y = f, x, y >= 0."""
    seqs1, par1 = _sequences(1)
    seqs2, par2 = _sequences(2)
    A = np.zeros((len(seqs1), len(seqs2)))
    for c1, c2 in DEALS:
        for z in TERMINALS:
            A[_last_seq(seqs1, 1, c1, z), _last_seq(seqs2, 2, c2, z)] += utility(c1, c2, z) / len(DEALS)

    def constraints(seqs, parent):
        Is = list(parent)
        M = np.zeros((1 + len(Is), len(seqs)))
        M[0, 0] = 1.0                          # x_empty = 1
        for r, I in enumerate(Is, start=1):    # sum_a x_{I,a} = x_{parent(I)}
            M[r, parent[I]] -= 1.0
            for a in ACTIONS:
                M[r, seqs.index((I, a))] += 1.0
        rhs = np.zeros(1 + len(Is))
        rhs[0] = 1.0
        return M, rhs

    E, e = constraints(seqs1, par1)
    F, f = constraints(seqs2, par2)
    return A, E, e, F, f, seqs1, par1, seqs2, par2


def solve_sequence_form_lp():
    """Solve Kuhn poker exactly. Returns (value for P1, behavioural s1, behavioural s2).

    Player 1's problem  max_x min_{y: Fy=f, y>=0} x^T A y.  Replacing the inner LP by its
    dual (max_q f^T q s.t. F^T q <= A^T x) gives one LP in (x, q):
        max f^T q   s.t.  F^T q - A^T x <= 0,  E x = e,  x >= 0,  q free.
    Player 2's strategy comes from the symmetric LP with A replaced by -A^T.
    """
    A, E, e, F, f, seqs1, par1, seqs2, par2 = sequence_form()

    def solve(A, E, e, F, f):
        nx, nq = A.shape[0], F.shape[0]
        c = np.concatenate([np.zeros(nx), -f])              # maximise f^T q
        A_ub = np.hstack([-A.T, F.T])
        A_eq = np.hstack([E, np.zeros((E.shape[0], nq))])
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(A.shape[1]), A_eq=A_eq, b_eq=e,
                      bounds=[(0, None)] * nx + [(None, None)] * nq, method="highs")
        assert res.success, res.message
        return res.x[:nx], -res.fun

    x, v1 = solve(A, E, e, F, f)
    y, v2 = solve(-A.T, F, f, E, e)
    assert abs(v1 + v2) < 1e-8, (v1, v2)     # the two LPs certify the same value

    def behavioural(z, seqs, parent):
        s = {}
        for I, par in parent.items():
            w = np.array([z[seqs.index((I, a))] for a in ACTIONS])
            s[I] = w / z[par] if z[par] > 1e-12 else np.array([0.5, 0.5])
        return s

    return v1, behavioural(x, seqs1, par1), behavioural(y, seqs2, par2)


def describe(s1, s2):
    """The handful of probabilities that characterise Kuhn equilibria (section 10.1)."""
    return {
        "alpha = P1 bets J": s1["J:"][1],
        "P1 bets Q": s1["Q:"][1],
        "P1 bets K": s1["K:"][1],
        "P1 calls with Q after p-b": s1["Q:pb"][1],
        "P1 calls with J after p-b": s1["J:pb"][1],
        "P1 calls with K after p-b": s1["K:pb"][1],
        "P2 bets J after check": s2["J:p"][1],
        "P2 bets Q after check": s2["Q:p"][1],
        "P2 bets K after check": s2["K:p"][1],
        "P2 calls with Q": s2["Q:b"][1],
        "P2 calls with J": s2["J:b"][1],
        "P2 calls with K": s2["K:b"][1],
    }


def main():
    parser = argparse.ArgumentParser(description="Kuhn poker self-test")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    print(f"kuhn.py self-test | seed={args.seed} quick={args.quick}")

    u1, u2 = uniform_strategy(1), uniform_strategy(2)
    print(f"EV(uniform, uniform) = {expected_value(u1, u2):+.4f};  "
          f"exploitability(uniform) = {exploitability(u1, u2):.4f}")
    _, v = best_response(u2, 1)
    print(f"  best response of P1 to uniform P2 earns {v:+.4f}")
    _, v = best_response(u1, 2)
    print(f"  best response of P2 to uniform P1 earns {v:+.4f}")

    v, s1, s2 = solve_sequence_form_lp()
    print(f"\nsequence-form LP: game value for P1 = {v:+.6f}  (-1/18 = {GAME_VALUE:+.6f})")
    print(f"  EV of LP strategies = {expected_value(s1, s2):+.6f}; "
          f"exploitability = {exploitability(s1, s2):.2e}")
    for k, val in describe(s1, s2).items():
        print(f"  {k:28s} {val:.4f}")
    assert abs(v - GAME_VALUE) < 1e-9

    # The known one-parameter family for player 1 (Kuhn 1950): for alpha in [0, 1/3],
    # bet J w.p. alpha, bet K w.p. 3 alpha, call with Q w.p. alpha + 1/3; never bet Q,
    # fold J and call K when facing a bet. Player 2's equilibrium strategy is unique.
    n_alpha = 3 if args.quick else 7
    print("\nchecking the equilibrium family alpha in [0, 1/3] against the LP's player-2 strategy:")
    for alpha in np.linspace(0, 1 / 3, n_alpha):
        f1 = {"J:": np.array([1 - alpha, alpha]), "Q:": np.array([1.0, 0.0]),
              "K:": np.array([1 - 3 * alpha, 3 * alpha]), "J:pb": np.array([1.0, 0.0]),
              "Q:pb": np.array([2 / 3 - alpha, alpha + 1 / 3]), "K:pb": np.array([0.0, 1.0])}
        ex = exploitability(f1, s2)
        print(f"  alpha = {alpha:.4f}: EV = {expected_value(f1, s2):+.6f}, exploitability = {ex:.2e}")
        assert ex < 1e-9

    # Random strategies are exploitable, and exploitability is never negative.
    worst = min(exploitability({I: rng.dirichlet([1, 1]) for I in infosets(1)},
                               {I: rng.dirichlet([1, 1]) for I in infosets(2)})
                for _ in range(20 if args.quick else 200))
    print(f"\nmin exploitability over random strategy pairs = {worst:.4f} (> 0)")
    print("all self-tests passed")


if __name__ == "__main__":
    main()
