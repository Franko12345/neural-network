# 06 — Transformer block + model stack

**What to build:** `transformer/block.py` (one block: attention +
feedforward + residual + LayerNorm) and `transformer/model.py` (stack
of N blocks + embedding + positional encoding + lm head + sampling).
From the user's perspective: `model = Transformer(vocab=256,
d_model=128, n_heads=4, n_layers=4, max_seq_len=128); logits =
model.forward(token_ids)` returns `(batch, seq_len, vocab)`
logits; `model.sample(prompt, n_tokens=50)` returns generated ids.

**Blocked by:** 04 (LayerNorm, Residual), 05 (MultiHeadAttention).

**Status:** implemented

- [x] `transformer/embed.py`: `Embedding(vocab_size, d_model)` +
      sinusoidal positional encoding `PositionalEncoding(max_seq_len,
      d_model)` that adds to token embeddings
- [x] `transformer/block.py`: `Block(d_model, n_heads, d_ff)`:
      - pre-norm: `x = x + MHA(LN1(x))`
      - pre-norm: `x = x + FFN(LN2(x))` where FFN = Linear → ReLU → Linear
- [x] `transformer/model.py`: `Transformer(vocab, d_model, n_heads,
      n_layers, d_ff, max_seq_len)`:
      - Embedding + positional encoding
      - Stack of N blocks
      - Final LayerNorm
      - LM head: `Linear(d_model, vocab)` projecting to logits
      - `forward(token_ids)` returns logits
      - `backward(grad)` updates all weights
      - `sample(prompt_ids, n_tokens, temperature, top_k)` returns
        generated ids (autoregressive)
- [x] `tests/test_transformer.py`:
      - Forward shape: `(batch, seq_len, vocab)`
      - Backward correctness: numerical gradient check on a tiny
        config (vocab=16, d_model=8, 2 layers, seq_len=4)
      - Sample loop: `sample()` returns ids of length
        `len(prompt) + n_tokens`
      - "Loss decreases" test: 50 SGD steps on a 4-token sequence,
        assert loss goes down
- [x] Headless smoke: full forward + backward on a 4-token input, exit 0
- [x] ponytail: block and model each fit in <200 LOC; no helper
      functions exposed beyond the class
- [x] v1 tests still pass (no overlap)

## Review findings applied (PR #16)

- **CRITICAL BUG FIXED** in `layers.py` (same pattern as PR #15):
  - `Linear.backward` now computes `dL/dx` using `W` BEFORE mutating
    `W`. The previous code returned `grad @ self.W.T` after the
    in-place `W -= dW`, giving wrong gradients for any 2+ Linear
    chain (the W-order bug class from PR #14 W_o, now propagated
    into Linear).
  - `Linear.backward` now handles N-D inputs via flatten-leading-dims
    (forward() accepts `(B, T, fan_in)` via broadcasting; backward
    must mirror that). Second half of the bug pattern — hidden
    because earlier tickets used 2D inputs.
- `test_linear_backward_grad` patched: re-inits Linear per FD call
  (the old test reused the layer instance, masking the W-mutation
  bug — same fix as PR #15 review).
- ticket wording: pre-norm layout (sublayer takes `LN(x)`), more
  stable for from-scratch training. Spec literal said "attention +
  feedforward with residual + layer norm" without specifying pre vs
  post — pre-norm chosen for stability.
- All 7 test files pass: 39 asserts total; v1 unchanged (3.7s).
