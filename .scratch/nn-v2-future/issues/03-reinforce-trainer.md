# 03 — REINFORCE trainer with baseline

**What to build:** `train_rl.py` that takes a policy network (built
from `layers.Linear` + activations) and a `RolloutBatch`, computes
the policy gradient with constant baseline, and updates weights.
From the user's perspective: `policy = make_policy([2, 16, 3]);
train_rl.train(policy, batch)` returns the updated policy and a
running mean reward.

**Blocked by:** 01 (Linear, Softmax), 02 (RolloutBatch).

**Status:** implemented

- [x] `train_rl.py`: `train(policy, batch, lr=0.01)` computes:
      - `G_t = Σ γ^k * r_{t+k}` per timestep
      - `b = mean(G)` over the batch
      - `advantage_t = G_t - b`
      - `loss = -mean(log_prob * advantage)`
      - backward through policy layers (built on `layers.Linear`)
      - in-place SGD update on each Linear's W/b
- [x] Headless smoke: train for 50 episodes on MountainCar, save
      policy weights to `mountaincar_policy.npz`, exit 0
- [x] `tests/test_rl.py`:
      - Numerical gradient check on `-log π(a|s) * advantage` w.r.t.
        Linear weights (small input)
      - Baseline subtraction: gradient with `b=0` matches baseline
        gradient up to the constant offset
- [x] `python3 -m tests.test_rl` exits 0 in <10s
- [x] ponytail: no flag on `train()`; `γ` and `lr` have defaults but
      are explicit kwargs
- [x] v1 tests still pass (no shared code with v1; sanity check)


## Review findings applied (PR #15)

- **CRITICAL BUG CAUGHT + FIXED**: `Linear.backward` had the same
  W-mutation-before-d_x bug as PR #14 MultiHeadAttention caught. The
  line `return grad @ self.W.T` was using `W` AFTER the in-place
  `self.W -= self.dW` mutation, giving wrong gradients for any
  module chain of 2+ Linears. Discovered because ticket 03's REINFORCE
  is the first user with a Linear→ReLU→Linear pipeline where the
  loss depends on W2. Fix: compute `d_x = grad @ self.W.T` BEFORE
  mutating W. Mirrors PR #14's W_o fix.
- test_linear_backward_grad in test_layers.py was reusing the layer
  instance across FD calls; this hid the W-mutation bug because FD
  used the same mutated W as analytical. Fixed test to re-init
  Linear per FD call (matches test_layernorm_backward_numerical
  pattern from PR #13 review).
- ticket 03 says "no flag on train()"; train() takes (policy_layers,
  batch, lr, gamma) — all explicit, no flags.
- v1 tests still pass (3.6s); no frozen files modified.
