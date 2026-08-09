# 0006. W-order + double-update trap: backward() must compute gradients BEFORE mutating weights

Date: 2026-08-09

## Status

Accepted

## Context

Building v2 surfaced **three related bug classes** across PRs #14, #15, #16,
and #17. All three were the same family of mistakes around the
`forward` / `backward` symmetric contract when backward also mutates
weights in-place:

1. **W-order in MultiHeadAttention (PR #14)**: `MultiHeadAttention.backward`
   mutated `W_o` (`self.W_o -= dW`) BEFORE computing
   `d_pre = grad @ self.W_o.T`. The d_pre used the post-update W_o,
   giving wrong input gradients for any 2+ layer chain. Caught by
   upgrading FD test from `d_model=4` → `d_model=8` (smaller scale
   masked the magnitude mismatch).

2. **W-order in Linear (PR #15)**: `Linear.backward` had the same bug:
   `return grad @ self.W.T` after `self.W -= self.dW`. Hidden in PR #01's
   `test_linear_backward_grad` because the test reused the layer instance
   across FD calls — FD used the same mutated W as analytical. Caught
   when ticket 03's REINFORCE was the first user of a Linear→ReLU→Linear
   chain where loss depends on W2.

3. **N-D support in Linear (PR #16)**: `Linear.forward` accepts
   `(B, T, fan_in)` via broadcasting, but `Linear.backward` assumed 2D.
   `self.x.T @ grad` raised matmul shape errors on N-D inputs. Hidden
   until ticket 06's transformer block tried to train. Fix: flatten
   leading dims in backward.

4. **Double-update in Trainer (PR #17)**: `train_rl.Trainer.step()` calls
   `model.backward(d_logits, lr=1.0)` which in-place SGD-updates every
   Linear/LayerNorm weight. THEN `_opt.step()` applied an AdamW update.
   Every weight was stepped twice per iteration: once with lr=1.0 SGD,
   then with lr=3e-4 AdamW. Comment in train.py line 103 explicitly
   admitted this. The training *appeared* to work (raw SGD alone
   descends) but the actual update magnitudes were wrong.

## Decision

All modules in `layers.py` and downstream follow a strict backward()
contract:

1. **Compute gradients first, mutate weights last.** Inside `backward()`:
   ```
   self.dW = ...      # gradient computation
   d_x = grad @ W.T  # input gradient uses W BEFORE mutation
   self.W -= self.dW  # mutation only after d_x is computed
   ```

2. **Match forward's broadcast shape.** `forward()` may accept N-D
   inputs via broadcasting (e.g. Linear on `(B, T, fan_in)`). `backward()`
   MUST mirror that: flatten leading dims, do the 2D matmul, reshape
   the output back. Pattern:
   ```python
   x_flat = self.x.reshape(-1, self.W.shape[0])
   grad_flat = grad.reshape(-1, self.W.shape[1])
   self.dW = x_flat.T @ grad_flat
   d_x_flat = grad_flat @ self.W.T
   self.W -= self.dW  # optional (see #3)
   return d_x_flat.reshape(self.x.shape)
   ```

3. **Backward must NOT mutate by default.** Linear/LayerNorm/MHA/
   Block/Transformer.backward all accept `update: bool = True`. Trainers
   that use an external optimizer (AdamW) call with `update=False`
   so the optimizer is the sole updater. Trainers that don't use an
   external optimizer (REINFORCE in train_rl, v1 nn.py) leave it at
   `True` (default SGD-style in-place step).

## Consequences

**Easier:**
- Forward/backward symmetric: caller can rely on the contract that
  `backward` returns the input gradient before any side effects.
- External optimizers (AdamW) compose cleanly: gradient flow runs
  first, then the optimizer decides what to do with `dW`.
- N-D inputs work the same as 2D: same code path, no special cases.

**Harder:**
- `update=False` is an extra flag every backward carries. Worth it
  because the alternative (always mutate, always in-place) makes
  caller-managed optimizers impossible to wire without monkey-patching
  or post-hoc corrections.
- Test fixtures must re-instantiate modules per FD call (or reset
  weights manually) because the in-place SGD step mutates W. PR #15
  and #16 both fixed test fixtures that had been hiding the W-order
  bug by reusing the layer instance.

**When to revisit:**
- If we ever add a non-mutating backward path (return dW without
  applying it), this contract can collapse — but for now the
  `update` flag is the smallest contract change.

**Lesson (the recurring pattern):**
In-place SGD step + caller-managed optimizer = double-update trap.
Pattern surfaced across 4 PRs. Future trainers MUST either:
- suppress the in-place update (`update=False`), OR
- not use a caller-managed optimizer (use the in-place SGD step as-is)
- but never both.
