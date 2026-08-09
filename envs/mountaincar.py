"""MountainCar-v0 wrapper. Normalizes obs to [-1, 1]^2.

ponytail: gymnasium import only here (ticket 02 keeps it local).
Obs/velocity ranges are env constants; normalize by raw bounds, not
by per-episode scaling.
"""
import gymnasium as gym
import numpy as np


# MountainCar-v0 raw bounds (gymnasium 0.29+ / 1.x)
POS_LO, POS_HI = -1.2, 0.6
VEL_LO, VEL_HI = -0.07, 0.07


def _normalize_obs(obs: np.ndarray) -> np.ndarray:
    """Map raw obs (position, velocity) to [-1, 1]^2."""
    pos = 2.0 * (obs[0] - POS_LO) / (POS_HI - POS_LO) - 1.0
    vel = 2.0 * (obs[1] - VEL_LO) / (VEL_HI - VEL_LO) - 1.0
    return np.array([pos, vel], dtype=np.float64)


class MountainCarEnv:
    """Thin wrapper over gymnasium MountainCar-v0."""

    def __init__(self):
        # ponytail: rgb_array mode so render() returns np.ndarray, not None
        self._env = gym.make("MountainCar-v0", render_mode="rgb_array")
        self.action_space_n = self._env.action_space.n

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        obs, info = self._env.reset(seed=seed)
        return _normalize_obs(obs), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        obs, reward, terminated, truncated, _info = self._env.step(int(action))
        return _normalize_obs(obs), float(reward), bool(terminated or truncated)

    def render(self) -> np.ndarray:
        """Return the latest frame as RGB array (H, W, 3) uint8."""
        return self._env.render()

    def close(self) -> None:
        self._env.close()
