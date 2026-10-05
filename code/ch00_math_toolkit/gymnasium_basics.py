"""Gymnasium in one script: spaces, reset/step, terminated vs truncated, seeding, wrappers,
vectorised environments.

Chapter 00, Section 8.
  (a) spaces of three environments (Discrete / Box observations and actions);
  (b) a random agent on CartPole-v1: every episode ends by TERMINATION (pole falls);
  (c) the same random agent under a 30-step time limit: some episodes are TRUNCATED;
  (d) a simple hand-written controller survives to the 500-step limit (always truncated),
      and the state it is truncated in is still worth ~1/(1-gamma) -- which is why the TD
      target must bootstrap through truncation (Eq. 8.1);
  (e) seeding makes runs reproducible;
  (f) wrappers: RecordEpisodeStatistics, TimeLimit, and a custom RewardWrapper;
  (g) a vector environment with next-step autoreset, and the mask that drops the
      'transition' across an episode boundary.

Run:  python code/ch00_math_toolkit/gymnasium_basics.py [--quick]
"""
import argparse
import os
import time

import gymnasium as gym
import numpy as np

from plot_style import setup, C, GREY

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def run_episode(env, policy, seed):
    """Algorithm 8.1: the canonical agent-environment loop for ONE episode.
    Returns (steps, return, terminated, truncated, last obs, info, last internal state); the
    internal state (CartPole: float64 physics state) is used in part (d) to continue exactly."""
    obs, info = env.reset(seed=seed)
    total_reward, steps = 0.0, 0
    while True:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:            # the episode is over either way...
            state = getattr(env.unwrapped, "state", None)
            state = None if state is None else np.array(state, dtype=np.float64)
            return steps, total_reward, terminated, truncated, obs, info, state
        # ...but only `terminated` means "the next state has value 0" (see Eq. 8.1)


def balance_policy(obs):
    """Push the cart toward the side the pole is falling to (a hand-tuned linear rule)."""
    x, x_dot, theta, theta_dot = obs
    return int(0.1 * x + 0.5 * x_dot + 3.0 * theta + theta_dot > 0)


class ScaledReward(gym.RewardWrapper):
    """A minimal custom wrapper: multiplies every reward by a constant."""

    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return self.scale * reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke test, no figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    n_eps = 200 if args.quick else 2000
    print(f"[gymnasium_basics] seed={args.seed} episodes={n_eps} gymnasium={gym.__version__} quick={args.quick}")
    t0 = time.time()

    # ------------------------------------------------------------------ (a) spaces
    print("\n(a) Spaces")
    for env_id in ["CartPole-v1", "FrozenLake-v1", "Pendulum-v1"]:
        env = gym.make(env_id)
        env.action_space.seed(args.seed)
        a = env.action_space.sample()
        print(f"  {env_id:14s} obs space {str(env.observation_space):60s} action space {env.action_space}"
              f"  sample action {a}  time limit {env.spec.max_episode_steps}")
        env.close()
    env = gym.make("CartPole-v1")
    print(f"  CartPole obs dtype {env.observation_space.dtype}, shape {env.observation_space.shape}; "
          f"contains([0,0,0,0])? {env.observation_space.contains(np.zeros(4, dtype=np.float32))}; "
          f"contains([0,0,1,0])? {env.observation_space.contains(np.array([0, 0, 1.0, 0], dtype=np.float32))} "
          f"(pole angle limit ±0.418 rad)")

    # ------------------------------------------------------------------ (b) random agent
    env.action_space.seed(args.seed)
    random_policy = lambda obs: env.action_space.sample()
    res_b = [run_episode(env, random_policy, seed=args.seed + i) for i in range(n_eps)]
    len_b = np.array([r[0] for r in res_b]); term_b = np.array([r[2] for r in res_b]); trunc_b = np.array([r[3] for r in res_b])
    print(f"\n(b) Random agent, CartPole-v1 (limit 500), {n_eps} episodes")
    print(f"  mean length {len_b.mean():.1f} (median {np.median(len_b):.0f}, max {len_b.max()}); "
          f"terminated {term_b.sum()}, truncated {trunc_b.sum()}")

    # ------------------------------------------------------------------ (c) short time limit
    env30 = gym.make("CartPole-v1", max_episode_steps=30)
    env30.action_space.seed(args.seed)
    res_c = [run_episode(env30, lambda o: env30.action_space.sample(), seed=args.seed + i) for i in range(n_eps)]
    len_c = np.array([r[0] for r in res_c]); term_c = np.array([r[2] for r in res_c]); trunc_c = np.array([r[3] for r in res_c])
    trunc_only = trunc_c & ~term_c
    print(f"\n(c) Random agent with max_episode_steps=30: terminated {term_c.sum()}, truncated {trunc_c.sum()}; "
          f"BOTH flags True in {(term_c & trunc_c).sum()} episodes (pole fell exactly at step 30)")
    print(f"  -> truncated-only (must bootstrap): {trunc_only.sum()} ({100 * trunc_only.mean():.1f}%); "
          f"terminated (must not bootstrap, even if also truncated): {term_c.sum()}")

    # ------------------------------------------------------------------ (d) truncation still has value
    n_d = 20 if args.quick else 100
    res_d = [run_episode(env, balance_policy, seed=args.seed + i) for i in range(n_d)]
    len_d = np.array([r[0] for r in res_d])
    print(f"\n(d) Hand-written controller, {n_d} episodes: mean length {len_d.mean():.1f}, "
          f"truncated {sum(r[3] for r in res_d)}, terminated {sum(r[2] for r in res_d)}")
    # What is the state at truncation worth? Continue from that exact physical state without a limit.
    gamma = 0.99
    _, _, _, _, final_obs, _, final_state = res_d[0]
    long_env = gym.make("CartPole-v1", max_episode_steps=100_000)
    long_env.reset(seed=args.seed)
    # teleport to the truncated state: copy the internal float64 physics state, not the
    # float32-rounded observation, so the continuation is exactly the episode's future
    long_env.unwrapped.state = final_state.copy()
    print(f"  max |float64 state - float32 observation| at truncation: "
          f"{np.abs(final_state - final_obs.astype(np.float64)).max():.1e}")
    obs, G, disc, k = np.array(final_obs), 0.0, 1.0, 0
    horizon = 500 if args.quick else 2000
    while k < horizon:
        obs, reward, terminated, truncated, _ = long_env.step(balance_policy(obs))
        G += disc * reward; disc *= gamma; k += 1
        if terminated:
            break
    print(f"  continuing from the truncated state for up to {horizon} more steps: survived {k} steps, "
          f"discounted return (gamma={gamma}) = {G:.2f}  vs 1/(1-gamma) = {1 / (1 - gamma):.0f}")
    print(f"  -> treating truncation as termination would use 0 instead of ~{G:.0f} in the TD target")

    # ------------------------------------------------------------------ (e) seeding
    def rollout(seed):
        e = gym.make("CartPole-v1"); e.action_space.seed(seed)
        obs, _ = e.reset(seed=seed); traj = [obs]
        for _ in range(20):
            obs, _, te, tr, _ = e.step(e.action_space.sample()); traj.append(obs)
            if te or tr:
                break
        return np.array(traj)
    same = np.array_equal(rollout(123), rollout(123))
    diff = rollout(123).shape != rollout(124).shape or not np.array_equal(rollout(123), rollout(124))
    print(f"\n(e) Seeding: identical trajectories with the same seed: {same}; different seeds differ: {diff}")

    # ------------------------------------------------------------------ (f) wrappers
    wrapped = gym.wrappers.RecordEpisodeStatistics(ScaledReward(gym.make("CartPole-v1", max_episode_steps=50), 0.1))
    wrapped.action_space.seed(args.seed)
    print(f"\n(f) Wrapper stack: {wrapped}")
    _, ret, _, _, _, info, _ = run_episode(wrapped, lambda o: wrapped.action_space.sample(), seed=args.seed)
    print(f"  episode return seen by agent {ret:.2f}; info['episode'] = "
          f"{{'r': {float(info['episode']['r']):.2f}, 'l': {int(info['episode']['l'])}}}")

    # ------------------------------------------------------------------ (g) vector env
    num_envs, steps = 4, (100 if args.quick else 1000)
    envs = gym.make_vec("CartPole-v1", num_envs=num_envs, vectorization_mode="sync")
    envs.action_space.seed(args.seed)
    obs, info = envs.reset(seed=args.seed)
    print(f"\n(g) Vector env: {envs}, autoreset mode {envs.metadata['autoreset_mode']}")
    print(f"  batched obs shape {obs.shape}, action space {envs.action_space}")
    prev_done = np.zeros(num_envs, dtype=bool)
    n_valid, n_skipped, finished = 0, 0, 0
    for _ in range(steps):
        actions = envs.action_space.sample()
        next_obs, rewards, terminated, truncated, info = envs.step(actions)
        # NEXT_STEP autoreset: the step AFTER an episode ends only resets that sub-env; the
        # (obs -> next_obs) pair for it is not a real transition and must not be learned from.
        valid = ~prev_done
        n_valid += valid.sum(); n_skipped += (~valid).sum()
        finished += (terminated | truncated).sum()
        prev_done = terminated | truncated
        obs = next_obs
    envs.close()
    print(f"  {steps} vector steps x {num_envs} envs: {finished} episodes finished, {n_valid} valid transitions, "
          f"{n_skipped} reset-steps masked out")

    if not args.quick:
        plt = setup()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
        ax = axes[0]
        bins = np.arange(0, len_b.max() + 2, 2)
        ax.hist(len_b, bins=bins, color=C[0], edgecolor="#fcfcfb", linewidth=0.8, label="terminated (pole fell / cart left track)")
        ax.set_title(f"(a) Random agent, CartPole-v1 (limit 500), {n_eps} episodes")
        ax.set_xlabel("episode length (steps)"); ax.set_ylabel("count"); ax.legend()
        ax = axes[1]
        bins = np.arange(0, 32, 1)
        ax.hist([len_c[term_c], len_c[trunc_only]], bins=bins, stacked=True, color=[C[0], C[1]],
                edgecolor="#fcfcfb", linewidth=0.8,
                label=[f"terminated ({term_c.sum()}, incl. {(term_c & trunc_c).sum()} also truncated)",
                       f"truncated only ({trunc_only.sum()})"])
        ax.set_title("(b) Same agent with max_episode_steps=30")
        ax.set_xlabel("episode length (steps)"); ax.set_ylabel("count"); ax.legend(loc="upper left")
        fig.tight_layout()
        path = os.path.join(FIG_DIR, "cartpole_terminated_truncated.png")
        fig.savefig(path)
        print(f"\nsaved {path}")

    print(f"\nSummary: random agent mean length {len_b.mean():.1f} (all terminated); with a 30-step limit "
          f"{100 * trunc_only.mean():.1f}% truncated-only; controller always truncated at 500 with value ~{G:.0f} left. "
          f"runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
