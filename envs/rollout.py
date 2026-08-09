"""Rollout batch collector: runs N episodes, packs into RolloutBatch.

ponytail: policy_fn(state) -> (action, log_prob). Caller owns the
policy (built on layers.Linear); this module just accumulates.
Episode boundaries are tracked separately so downstream trainers
can apply per-episode discounts without re-scanning.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class RolloutBatch:
    states: np.ndarray       # (T, obs_dim) float32
    actions: np.ndarray      # (T,) int64
    rewards: np.ndarray      # (T,) float32
    log_probs: np.ndarray    # (T,) float32
    episode_starts: list[int]
    episode_ends: list[int]


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
    if seed is not None:
        np.random.seed(seed)

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(np.random.randint(0, 2**31 - 1)))
        starts.append(int(np.sum([len(a) for a in (actions,)])))
        for _t in range(max_steps):
            a, lp = policy_fn(obs)
            a = int(a)
            states.append(obs)
            actions.append(a)
            log_probs.append(lp)
            next_obs, reward, done = env.step(a)
            rewards.append(reward)
            obs = next_obs
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
