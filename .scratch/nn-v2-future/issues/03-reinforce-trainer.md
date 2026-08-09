# 03 — REINFORCE trainer with baseline

**What to build:** `train_rl.py` that takes a policy network (built
from `layers.Linear` + activations) and a `RolloutBatch`, computes
the policy gradient with constant baseline, and updates weights.
From the user's perspective: `policy = make_policy([2, 16, 3]);
train_rl.train(policy, batch)` returns the updated policy and a
running mean reward.

**Blocked by:** 01 (Linear, Softmax), 02 (RolloutBatch).

**Status:** ready-for-agent

- [ ] `train_rl.py`: `train(policy, batch, lr=0.01)` computes:
      - `G_t = Σ γ^k * r_{t+k}` per timestep
      - `b = mean(G)` over the batch
      - `advantage_t = G_t - b`
      - `loss = -mean(log_prob * advantage)`
      - backward through policy layers (built on `layers.Linear`)
      - in-place SGD update on each Linear's W/b
- [ ] Headless smoke: train for 50 episodes on MountainCar, save
      policy weights to `mountaincar_policy.npz`, exit 0
- [ ] `tests/test_rl.py`:
      - Numerical gradient check on `-log π(a|s) * advantage` w.r.t.
        Linear weights (small input)
      - Baseline subtraction: gradient with `b=0` matches baseline
        gradient up to the constant offset
- [ ] `python3 -m tests.test_rl` exits 0 in <10s
- [ ] ponytail: no flag on `train()`; `γ` and `lr` have defaults but
      are explicit kwargs
- [ ] v1 tests still pass (no shared code with v1; sanity check)