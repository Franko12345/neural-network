# 02 — MountainCar env wrapper + rollout batch

**What to build:** `envs/mountaincar.py` (gymnasium wrapper,
normalized observations) and `envs/rollout.py` (collects trajectories
into `RolloutBatch`). From the user's perspective: `env = MountainCarEnv();
batch = rollout(env, policy, n_episodes=10)` returns a dataclass with
`(states, actions, rewards, log_probs)` ready to feed into a trainer.

**Blocked by:** 01 (need Linear + Softmax for the policy network).

**Status:** implemented

- [x] `envs/__init__.py` empty
- [x] `envs/mountaincar.py`: `MountainCarEnv` class wrapping gymnasium's
      `MountainCar-v0`. `reset()` returns normalized obs (position,
      velocity) in `[-1, 1]²`. `step(action)` returns
      `(next_obs, reward, done, info)` where info carries
      `{"terminated", "truncated"}` (keeps value-bootstrapping open
      for future work; ticket 03 REINFORCE only needs `done`).
      `render()` returns the gym render frame as RGB array.
- [x] `envs/rollout.py`: `RolloutBatch` dataclass with the four
      required fields plus `episode_starts`/`episode_ends` (needed
      for per-episode returns in ticket 03 — same pattern as PR #11
      ticket wording update). `rollout(env, policy_fn, n_episodes=10,
      max_steps=200, seed=None)` accepts `max_steps` and `seed` kwargs
      with defaults; `seed` for determinism, `max_steps=200` matches
      gymnasium's default episode length.
- [x] `tests/test_rollout.py`: rollout returns correct shapes; obs
      are in `[-1, 1]²`; rewards sum to gymnasium's default;
      normalization endpoints exact (`_normalize_obs` maps
      `[POS_LO, VEL_LO]` → `[-1, -1]` etc.).
- [x] `python3 -m tests.test_rollout` exits 0 in <5s
- [x] ponytail: no flag on `step()`; gymnasium import only inside
      `envs/mountaincar.py` (test suite imports it too)

## Review findings applied (PR #12)

- Matt Pocock: missing `envs/__init__.py` (created); `reset()` returned
  tuple `(obs, info)` — now returns obs only per ticket literal; ticket
  wording updated to bless 4-tuple `step()` with `info` dict (preserves
  terminated/truncated separation).
- Correctness: test only checked obs in range; now asserts endpoints
  exact (`±1, 0`) which catches sign-flip normalization bugs.
- Ponytail: replaced global `np.random.seed` with `np.random.default_rng(seed)`
  for reproducibility without global state mutation; replaced ceremony
  `np.sum([len(a) for a in (actions,)])` with `len(actions)`.
- New test: `rollout_done_breaks_episode` uses a 1-step stub env to
  verify the done-break path.
