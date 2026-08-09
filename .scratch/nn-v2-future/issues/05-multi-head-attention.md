# 05 — Multi-head attention with causal mask

**What to build:** `transformer/attention.py` with `MultiHeadAttention`
class — self-attention with scaled dot-product, causal mask, multi-head
split/concat. From the user's perspective:
`mha = MultiHeadAttention(d_model=128, n_heads=4); out = mha.forward(X)`
returns same-shape tensor; `mha.backward(grad)` updates internal
weights.

**Blocked by:** 01 (Linear).

**Status:** ready-for-agent

- [x] `transformer/__init__.py` empty
- [x] `transformer/attention.py`: `MultiHeadAttention(d_model, n_heads)`:
      - Forward: `Q, K, V = X @ W_q, X @ W_k, X @ W_v` (per head, split)
      - `scores = Q @ K.T / sqrt(d_k)`
      - Apply causal mask (`-inf` upper triangle, including diagonal
        for strict causal)
      - `weights = softmax(scores, axis=-1)`
      - `out = weights @ V`, concat heads, project via `W_o`
      - `backward(grad)` computes gradients w.r.t. Q, K, V, all 4
        weight matrices, and input X
- [x] `tests/test_attention.py`:
      - Numerical gradient check (small d_model=8, n_heads=2)
      - **Causal mask verification**: position `t` has zero attention
        weight on positions `> t`
      - Multi-head equivalence: 4-head output on d_model=32 equals
        concat of 4 single-head outputs
- [x] Headless smoke: forward + backward + check grads shape
- [x] ponytail: one class, no helper functions exposed; mask applied
      via `scores += mask` (not `np.where` per cell)
