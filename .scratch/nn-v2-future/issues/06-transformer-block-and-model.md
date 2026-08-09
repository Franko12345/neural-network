# 06 — Transformer block + model stack

**What to build:** `transformer/block.py` (one block: attention +
feedforward + residual + LayerNorm) and `transformer/model.py` (stack
of N blocks + embedding + positional encoding + lm head + sampling).
From the user's perspective: `model = Transformer(vocab=256,
d_model=128, n_heads=4, n_layers=4, max_seq_len=128); logits =
model.forward(token_ids)` returns `(batch, seq_len, vocab)`
logits; `model.sample(prompt, n_tokens=50)` returns generated ids.

**Blocked by:** 04 (LayerNorm, Residual), 05 (MultiHeadAttention).

**Status:** ready-for-agent

- [ ] `transformer/embed.py`: `Embedding(vocab_size, d_model)` +
      sinusoidal positional encoding `PositionalEncoding(max_seq_len,
      d_model)` that adds to token embeddings
- [ ] `transformer/block.py`: `Block(d_model, n_heads, d_ff)`:
      - `x = x + MultiHeadAttention(LayerNorm(x))`
      - `x = x + FFN(LayerNorm(x))` where FFN = Linear → ReLU → Linear
- [ ] `transformer/model.py`: `Transformer(vocab, d_model, n_heads,
      n_layers, d_ff, max_seq_len)`:
      - Embedding + positional encoding
      - Stack of N blocks
      - Final LayerNorm
      - LM head: `Linear(d_model, vocab)` projecting to logits
      - `forward(token_ids)` returns logits
      - `backward(grad)` updates all weights
      - `sample(prompt_ids, n_tokens, temperature, top_k)` returns
        generated ids (autoregressive)
- [ ] `tests/test_transformer.py`:
      - Forward shape: `(batch, seq_len, vocab)`
      - Backward correctness: numerical gradient check on a tiny
        config (vocab=16, d_model=32, 2 layers, seq_len=16)
      - Sample loop: `sample()` returns ids of length
        `len(prompt) + n_tokens`
      - "Loss decreases" test: 100 SGD steps on a 4-token sequence,
        assert loss goes down
- [ ] Headless smoke: full forward + backward on a 16-token input,
  exit 0
- [ ] ponytail: block and model each fit in <200 LOC; no helper
      functions exposed beyond the class
- [ ] v1 tests still pass (no overlap)