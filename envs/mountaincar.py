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
    """Map raw obs (position, velocity) to [-1, 1]^2. Endpoints exact."""
    pos = 2.0 * (obs[0] - POS_LO) / (POS_HI - POS_LO) - 1.0
    vel = 2.0 * (obs[1] - VEL_LO) / (VEL_HI - VEL_LO) - 1.0
    return np.array([pos, vel], dtype=np.float64)


class MountainCarEnv:
    """Thin wrapper over gymnasium MountainCar-v0.

    reset() returns normalized obs (info dropped; ticket 02 contract).
    step() returns (next_obs, reward, done, info). info carries
    {'terminated': bool, 'truncated': bool} so future ticket 03 work
    can bootstrap on truncated states if needed; 'done' is OR of both.
    """

    def __init__(self):
        # ponytail: rgb_array mode so render() returns np.ndarray, not None
        self._env = gym.make("MountainCar-v0", render_mode="rgb_array")
        self.action_space_n = self._env.action_space.n

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs, _info = self._env.reset(seed=seed)
        return _normalize_obs(obs)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminated, truncated, _info = self._env.step(int(action))
        next_obs = _normalize_obs(obs)
        done = bool(terminated or truncated)
        info = {"terminated": bool(terminated), "truncated": bool(truncated)}
        return next_obs, float(reward), done, info

    def render(self) -> np.ndarray:
        """Return the latest frame as RGB array (H, W, 3) uint8."""
        return self._env.render()

    def close(self) -> None:
        self._env.close()
