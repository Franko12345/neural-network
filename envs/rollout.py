"""Rollout batch collector: runs N episodes, packs into RolloutBatch.

ponytail: policy_fn(state) -> (action, log_prob). Caller owns the
policy (built on layers.Linear); this module just accumulates.
Episode boundaries tracked separately for downstream per-episode
discounts (ticket 03).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class RolloutBatch:
    states: np.ndarray       # (T, obs_dim) float64
    actions: np.ndarray      # (T,) int64
    rewards: np.ndarray      # (T,) float64
    log_probs: np.ndarray    # (T,) float64
    episode_starts: list[int]   # index of first step of each episode
    episode_ends: list[int]     # index past last step of each episode


def rollout(
    env,
    policy_fn,
    n_episodes: int = 10,
    max_steps: int = 200,
    seed: int | None = None,
) -> RolloutBatch:
    """Run `n_episodes` episodes; collect (state, action, reward, log_prob)."""
    states, actions, rewards, log_probs = [], [], [], []
    starts, ends = [], []
    rng = np.random.default_rng(seed)

    for _ in range(n_episodes):
        obs = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        starts.append(len(states))
        for _t in range(max_steps):
            a, lp = policy_fn(obs)
            a = int(a)
            states.append(obs)
            actions.append(a)
            log_probs.append(lp)
            obs, reward, done, _info = env.step(a)
            rewards.append(reward)
            if done:
                break
        ends.append(len(states))

    return RolloutBatch(
        states=np.asarray(states, dtype=np.float64),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float64),
        log_probs=np.asarray(log_probs, dtype=np.float64),
        episode_starts=starts,
        episode_ends=ends,
    )
