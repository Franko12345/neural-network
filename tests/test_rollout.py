"""Headless validation of envs/mountaincar.py + envs/rollout.py.

ponytail: stdlib only, no pytest. One file, asserts, exit 0/1.
"""
from __future__ import annotations

import sys

import gymnasium as gym
import numpy as np

from envs.mountaincar import MountainCarEnv, POS_HI, POS_LO, VEL_HI, VEL_LO, _normalize_obs
from envs.rollout import RolloutBatch, rollout


def test_env_obs_normalized_in_range() -> bool:
    env = MountainCarEnv()
    obs = env.reset(seed=0)
    return obs.shape == (2,) and -1.0 <= obs[0] <= 1.0 and -1.0 <= obs[1] <= 1.0


def test_env_normalize_endpoints_exact() -> bool:
    """Normalization: endpoints map to ±1 exactly (catches sign-flip bugs)."""
    raw_lo = np.array([POS_LO, VEL_LO])
    raw_hi = np.array([POS_HI, VEL_HI])
    raw_mid = np.array([(POS_LO + POS_HI) / 2, (VEL_LO + VEL_HI) / 2])
    out_lo = _normalize_obs(raw_lo)
    out_hi = _normalize_obs(raw_hi)
    out_mid = _normalize_obs(raw_mid)
    return (
        np.allclose(out_lo, [-1.0, -1.0])
        and np.allclose(out_hi, [1.0, 1.0])
        and np.allclose(out_mid, [0.0, 0.0])
    )


def test_env_step_returns_4_tuple_with_done_and_info() -> bool:
    env = MountainCarEnv()
    env.reset(seed=0)
    next_obs, reward, done, info = env.step(0)
    assert next_obs.shape == (2,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "terminated" in info and "truncated" in info
    return True


def test_env_render_returns_rgb() -> bool:
    env = MountainCarEnv()
    env.reset(seed=0)
    env.step(0)
    frame = env.render()
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

    batch = rollout(env, policy, n_episodes=3, max_steps=200, seed=0)
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
    # Episode boundaries: starts/ends adjacent slices.
    assert len(batch.episode_starts) == 3
    assert len(batch.episode_ends) == 3
    assert batch.episode_starts[0] == 0
    assert batch.episode_ends[-1] == len(batch.states)
    # Reward sum parity with raw gymnasium (action-only dynamics).
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


def test_rollout_done_breaks_episode() -> bool:
    """An env that dones on step 1 must produce 1-step episodes."""
    class _OneStepEnv:
        action_space_n = 3
        def reset(self, seed=None):
            return np.array([0.0, 0.0])
        def step(self, a):
            return np.array([0.0, 0.0]), -1.0, True, {"terminated": True, "truncated": False}

    def policy(state):
        return 0, 0.0

    batch = rollout(_OneStepEnv(), policy, n_episodes=3, max_steps=200, seed=0)
    return (
        len(batch.states) == 3
        and batch.episode_starts == [0, 1, 2]
        and batch.episode_ends == [1, 2, 3]
    )


def main() -> int:
    results = [
        ("env_obs_normalized_in_range", test_env_obs_normalized_in_range()),
        ("env_normalize_endpoints_exact", test_env_normalize_endpoints_exact()),
        ("env_step_returns_4_tuple_with_done_and_info", test_env_step_returns_4_tuple_with_done_and_info()),
        ("env_render_returns_rgb", test_env_render_returns_rgb()),
        ("rollout_accumulates_correctly", test_rollout_accumulates_correctly()),
        ("rollout_done_breaks_episode", test_rollout_done_breaks_episode()),
    ]
    for name, ok in results:
        flag = "PASS" if ok else "FAIL"
        print(f"  {name:42s} {flag}")
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
