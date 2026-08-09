"""Headless validation of envs/mountaincar.py + envs/rollout.py.

ponytail: stdlib only, no pytest. One file, asserts, exit 0/1.
"""
from __future__ import annotations

import sys

import gymnasium as gym
import numpy as np

from envs.mountaincar import MountainCarEnv
from envs.rollout import RolloutBatch, rollout


def test_env_obs_normalized() -> bool:
    env = MountainCarEnv()
    obs, _ = env.reset(seed=0)
    return obs.shape == (2,) and -1.0 <= obs[0] <= 1.0 and -1.0 <= obs[1] <= 1.0


def test_env_step_returns_correct_shapes() -> bool:
    env = MountainCarEnv()
    env.reset(seed=0)
    next_obs, reward, done = env.step(0)
    return next_obs.shape == (2,) and isinstance(reward, float) and isinstance(done, bool)


def test_env_render_returns_rgb() -> bool:
    env = MountainCarEnv()
    env.reset(seed=0)
    env.step(0)
    frame = env.render()
    # gymnasium render() returns (H, W, 3) uint8 by default in 1.x
    return frame.ndim == 3 and frame.shape[2] == 3


def test_rollout_accumulates_correctly() -> bool:
    """Run a constant-action policy; assert shapes, accumulation, and
    that sum of rewards per episode equals gymnasium's default."""
    env = MountainCarEnv()
    rng = np.random.default_rng(0)

    def policy(state):
        a = int(rng.integers(0, 3))
        log_prob = float(np.log(1.0 / 3.0))  # uniform
        return a, log_prob

    batch = rollout(env, policy, n_episodes=3, max_steps=50, seed=0)
    assert isinstance(batch, RolloutBatch)
    assert batch.states.ndim == 2 and batch.states.shape[1] == 2
    assert batch.actions.ndim == 1
    assert batch.rewards.ndim == 1
    assert batch.log_probs.ndim == 1
    assert (
        batch.states.shape[0]
        == batch.actions.shape[0]
        == batch.rewards.shape[0]
        == batch.log_probs.shape[0]
    )
    # Episode boundaries sum to gymnasium's total reward per episode.
    raw = gym.make("MountainCar-v0")
    for start, end in zip(batch.episode_starts, batch.episode_ends):
        o, _ = raw.reset(seed=0)
        total = 0.0
        for t in range(end - start):
            a = int(batch.actions[start + t])
            _, r, term, trunc, _ = raw.step(a)
            total += float(r)
            if term or trunc:
                break
        assert abs(total - float(batch.rewards[start:end].sum())) < 1e-6, (
            f"reward sum mismatch: ours={batch.rewards[start:end].sum():.3f} gym={total:.3f}"
        )
    raw.close()
    return True


def main() -> int:
    results = [
        ("env_obs_normalized", test_env_obs_normalized()),
        ("env_step_returns_correct_shapes", test_env_step_returns_correct_shapes()),
        ("env_render_returns_rgb", test_env_render_returns_rgb()),
        ("rollout_accumulates_correctly", test_rollout_accumulates_correctly()),
    ]
    for name, ok in results:
        flag = "PASS" if ok else "FAIL"
        print(f"  {name:38s} {flag}")
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
