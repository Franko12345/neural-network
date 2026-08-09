# 02 — MountainCar env wrapper + rollout batch

**What to build:** `envs/mountaincar.py` (gymnasium wrapper,
normalized observations) and `envs/rollout.py` (collects trajectories
into `RolloutBatch`). From the user's perspective: `env = MountainCarEnv();
batch = rollout(env, policy, n_episodes=10)` returns a dataclass with
`(states, actions, rewards, log_probs)` ready to feed into a trainer.

**Blocked by:** 01 (need Linear + Softmax for the policy network).

**Status:** ready-for-agent

- [ ] `envs/__init__.py` empty
- [ ] `envs/mountaincar.py`: `MountainCarEnv` class wrapping gymnasium's
      `MountainCar-v0`. `reset()` returns normalized state (position,
      velocity) in `[-1, 1]²`. `step(action)` returns next state,
      reward, done. `render()` returns the gym render frame as RGB
      array.
- [ ] `envs/rollout.py`: `RolloutBatch = dataclass(states, actions,
      rewards, log_probs)`. `rollout(env, policy_fn, n_episodes=10)`
      runs episodes, calls `policy_fn(state)` which returns
      `(action, log_prob)`, accumulates everything into a batch.
- [ ] `tests/test_rollout.py`: rollout returns correct shapes; obs
      are in `[-1, 1]²`; rewards sum to gymnasium's default.
- [ ] `python3 -m tests.test_rollout` exits 0 in <5s
- [ ] ponytail: no flag on `step()`; gymnasium import only inside
      `envs/mountaincar.py` (test suite imports it too)