# 05 — Multi-head attention with causal mask

**What to build:** `transformer/attention.py` with `MultiHeadAttention`
class — self-attention with scaled dot-product, causal mask, multi-head
split/concat. From the user's perspective:
`mha = MultiHeadAttention(d_model=128, n_heads=4); out = mha.forward(X)`
returns same-shape tensor; `mha.backward(grad)` updates internal
weights.

**Blocked by:** 01 (Linear).

**Status:** implemented

- [x] `transformer/__init__.py` empty
- [x] `transformer/attention.py`: `MultiHeadAttention(d_model, n_heads)`:
      - Forward: `Q, K, V = X @ W_q, X @ W_k, X @ W_v` (single W matrices;
        per-head via `_split_heads` / `_merge_heads` private methods).
      - `scores = Q @ K.T / sqrt(d_k)`
      - Apply causal mask (`-inf` upper triangle, `k=1` — position t
        attends to positions <= t; diagonal visible by convention)
      - `weights = softmax(scores, axis=-1)`
      - `out = weights @ V`, concat heads via merge, project via `W_o`
      - `backward(grad)` computes gradients w.r.t. Q, K, V, all 4
        weight matrices, and input X. **Critical**: `d_pre` must be
        computed BEFORE mutating `W_o` (otherwise uses updated W_o).
- [x] `tests/test_attention.py`:
      - Numerical gradient check (`d_model=8, n_heads=2, T=2` per spec)
      - **Causal mask verification**: position `t` has zero attention
        weight on positions `> t`
      - Multi-head equivalence: 4-head output on `d_model=32` equals
        concat of 4 single-head per-head paths + `W_o`
      - Grad shapes: W_q/k/v/o all `(d_model, d_model)`; grad_x same
        shape as forward input
- [x] Headless smoke: forward + backward + check grads shape
- [x] ponytail: one class, no helper functions exposed beyond
      `_split_heads` / `_merge_heads` (used by both forward and
      backward — shared private contract). Mask built inline at
      forward time (one use site).

## Review findings applied (PR #14)

- **Critical bug caught + fixed**: backward was mutating `W_o` (in-place
  SGD update) BEFORE computing `d_pre = grad @ W_o.T`. The d_pre used
  the already-updated W_o, giving wrong input gradients. Fix: compute
  d_pre first, then mutate W_o. Caught by upgrading FD test from
  `d_model=4` → `d_model=8` (more sensitive to scale; revealed the
  magnitude mismatch).
- Cached `out_pre_proj` in forward so backward doesn't recompute
  `_merge_heads(attn_weights @ V)` (CONTEXT pitfall from PR #10).
- Test cleanup: removed dead `mha1s` module construction (was building
  4 MultiHeadAttention instances just to slice their W matrices, then
  re-implementing the math by hand). Now uses `mha4.W_q[:, s:e]` directly.
- Added `test_mha_backward_grad_shapes` to explicitly assert W_q/k/v/o
  + grad_x shapes (ticket spec requires checking all gradient shapes).
- `_causal_mask` helper function inlined at forward (one use site).
- Created `transformer/__init__.py` (was missing; ticket falsely marked
  complete).
- Ticket wording updated: strict-causal `k=1` (position t attends to
  <= t, not just < t); helpers limited to `_split_heads`/`_merge_heads`
  (shared fwd/bwd use justifies private methods).
