"""Self-play, fictitious play and PSRO on Kuhn poker (Chapter 17, section 9).

Policy-Space Response Oracles (PSRO; Lanctot et al., 2017) keeps a population of
policies for each player, builds the "meta-game" whose entries are the expected payoffs
of every pair of population members, solves it with a meta-solver, and adds a best
response to the opponent's meta-strategy. Different meta-solvers recover familiar
training schemes:

  meta-solver 'last'     -> naive self-play (best-respond to the opponent's latest policy)
  meta-solver 'uniform'  -> fictitious play in policy space (best-respond to the uniform
                            mixture of all past policies; the idea behind fictitious self-play)
  meta-solver 'nash'     -> PSRO-Nash = the double oracle algorithm (McMahan et al., 2003)

Here the best-response oracle is exact (kuhn.best_response), so the experiment isolates
the effect of the meta-solver; in deep-RL PSRO the oracle is an RL training run.
Exploitability of the meta-strategy mixture is computed exactly after converting the
mixture to an equivalent behavioural strategy (Kuhn's theorem, kuhn.mixture_to_behavioural).

Run:  python code/ch17_multi_agent_rl/psro_kuhn.py [--quick]
"""
import argparse
import os
import time

import numpy as np

import games
import kuhn


def same_policy(s, t):
    return all(np.allclose(s[I], t[I]) for I in s)


def psro(meta_solver, n_iters):
    pop1, pop2 = [kuhn.uniform_strategy(1)], [kuhn.uniform_strategy(2)]
    M = np.array([[kuhn.expected_value(pop1[0], pop2[0])]])   # meta-game payoffs for player 1
    log = {"expl": [], "n_distinct1": [], "n_distinct2": [], "meta_value": [], "betJ": []}
    for it in range(n_iters + 1):
        k1, k2 = len(pop1), len(pop2)
        if meta_solver == "last":
            w1, w2 = np.eye(k1)[-1], np.eye(k2)[-1]
        elif meta_solver == "uniform":
            w1, w2 = np.full(k1, 1 / k1), np.full(k2, 1 / k2)
        else:
            w1, w2, _ = games.solve_zero_sum(M)
        mix1 = kuhn.mixture_to_behavioural(pop1, w1, 1)
        mix2 = kuhn.mixture_to_behavioural(pop2, w2, 2)
        log["expl"].append(kuhn.exploitability(mix1, mix2))
        log["meta_value"].append(w1 @ M @ w2)
        log["betJ"].append(mix1["J:"][1])
        distinct1 = [s for i, s in enumerate(pop1) if not any(same_policy(s, t) for t in pop1[:i])]
        distinct2 = [s for i, s in enumerate(pop2) if not any(same_policy(s, t) for t in pop2[:i])]
        log["n_distinct1"].append(len(distinct1))
        log["n_distinct2"].append(len(distinct2))
        if it == n_iters:
            break
        # response oracle: exact best response to the opponent's meta-strategy mixture
        br1, _ = kuhn.best_response(mix2, 1)
        br2, _ = kuhn.best_response(mix1, 2)
        pop1.append(br1)
        pop2.append(br2)
        # extend the meta-game payoff matrix with the new row and column
        new_row = np.array([kuhn.expected_value(br1, s2) for s2 in pop2])
        new_col = np.array([kuhn.expected_value(s1, br2) for s1 in pop1[:-1]])
        M = np.vstack([np.hstack([M, new_col[:, None]]), new_row[None, :]])
    return {k: np.array(v) for k, v in log.items()}, pop1, pop2


def main():
    parser = argparse.ArgumentParser(description="PSRO meta-solvers on Kuhn poker")
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0, help="unused: everything is deterministic")
    args = parser.parse_args()
    n_iters = 10 if args.quick else 60
    print(f"psro_kuhn.py | seed={args.seed} (deterministic) | iterations={n_iters} | quick={args.quick}")
    t0 = time.time()
    logs = {}
    for ms in ("last", "uniform", "nash"):
        log, pop1, pop2 = psro(ms, n_iters)
        logs[ms] = log
        e = log["expl"]
        zero_at = next((i for i, v in enumerate(e) if v < 1e-9), None)
        print(f"\nmeta-solver = {ms:8s}: exploitability at iterations 0,1,2,5,10,{n_iters}: "
              + ", ".join(f"{e[i]:.4f}" for i in (0, 1, 2, 5, 10, n_iters) if i <= n_iters))
        print(f"  distinct policies in final population: P1 {log['n_distinct1'][-1]}, "
              f"P2 {log['n_distinct2'][-1]} (of {n_iters + 1} added)")
        if zero_at is not None:
            print(f"  exploitability reached 0 (< 1e-9) at iteration {zero_at}; "
                  f"meta-game value there {log['meta_value'][zero_at]:+.6f} (game value -1/18 = {kuhn.GAME_VALUE:+.6f})")
        if ms == "last":
            print("  last 8 exploitabilities: " + ", ".join(f"{v:.3f}" for v in e[-8:]))
    print(f"\ntotal time {time.time() - t0:.1f} s")

    if not args.quick:
        import plot_style
        plt = plot_style.setup()
        figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
        os.makedirs(figdir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
        labels = {"last": "naive self-play (BR to latest)", "uniform": "fictitious play (BR to uniform mix)",
                  "nash": "PSRO-Nash / double oracle"}
        st = {"last": dict(color=plot_style.C[0], ls="-", marker="o", ms=3),
              "uniform": dict(color=plot_style.C[1], ls="--", marker="s", ms=3),
              "nash": dict(color=plot_style.C[2], ls="-.", marker="^", ms=4)}
        for ms, log in logs.items():
            it = np.arange(len(log["expl"]))
            axes[0].semilogy(it, np.maximum(log["expl"], 1e-6), label=labels[ms], **st[ms])
            axes[1].plot(it, log["n_distinct1"], label=labels[ms], **st[ms])
        axes[0].set_xlabel("PSRO iteration (best responses added per player)")
        axes[0].set_ylabel("exploitability of meta-strategy")
        axes[0].set_title("(a) Exploitability (values < 1e-6 drawn at 1e-6)")
        axes[0].legend()
        axes[1].set_xlabel("PSRO iteration")
        axes[1].set_ylabel("distinct policies (player 1)")
        axes[1].set_title("(b) Population size")
        fig.suptitle("Kuhn poker with an exact best-response oracle: the meta-solver decides convergence", y=1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "psro_kuhn.png"))
        plt.close(fig)
        print(f"figure written to {figdir}/psro_kuhn.png")


if __name__ == "__main__":
    main()
