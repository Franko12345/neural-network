"""Headless validation of transformer/ block + model.

ponytail: stdlib only, no pytest.
"""
from __future__ import annotations

import sys

import numpy as np

from transformer.attention import MultiHeadAttention
from transformer.block import Block
from transformer.embed import Embedding, PositionalEncoding
from transformer.model import Transformer


def test_embedding_forward_shape() -> bool:
    """Embedding(vocab, d_model): forward token_ids -> (B, T, d_model)."""
    emb = Embedding(vocab_size=16, d_model=8, seed=0)
    ids = np.array([[1, 5, 3, 7]], dtype=np.int64)
    out = emb.forward(ids)
    return out.shape == (1, 4, 8)


def test_positional_encoding_adds_to_emb() -> bool:
    """PositionalEncoding adds sinusoidal vector; shape (1, T, d_model)."""
    pe = PositionalEncoding(max_seq_len=16, d_model=8)
    x = np.zeros((1, 4, 8))
    out = pe.forward(x)
    return out.shape == (1, 4, 8) and not np.allclose(out, x)


def test_block_forward_shape() -> bool:
    """Block(d_model, n_heads, d_ff): forward (B, T, d_model) -> same shape."""
    block = Block(d_model=8, n_heads=2, d_ff=16, seed=0)
    x = np.random.default_rng(0).standard_normal((1, 4, 8))
    out = block.forward(x)
    return out.shape == (1, 4, 8)


def test_transformer_forward_shape() -> bool:
    """Transformer(vocab, d_model, n_heads, n_layers, d_ff, max_seq_len):
    forward (B, T) token ids -> (B, T, vocab) logits."""
    model = Transformer(vocab=16, d_model=8, n_heads=2, n_layers=2,
                        d_ff=16, max_seq_len=8, seed=0)
    ids = np.array([[1, 5, 3, 7]], dtype=np.int64)
    logits = model.forward(ids)
    return logits.shape == (1, 4, 16)


def test_transformer_sample_returns_extended_ids() -> bool:
    """sample(prompt_ids, n_tokens): returns ids of length
    len(prompt) + n_tokens."""
    model = Transformer(vocab=16, d_model=8, n_heads=2, n_layers=2,
                        d_ff=16, max_seq_len=16, seed=0)
    prompt = np.array([[1, 5, 3]], dtype=np.int64)
    out = model.sample(prompt, n_tokens=5, temperature=1.0, top_k=None)
    return out.shape == (1, 3 + 5)


def test_transformer_loss_decreases() -> bool:
    """100 SGD steps on a 4-token synthetic sequence: loss should drop."""
    from layers import Softmax
    softmax = Softmax(axis=-1)
    model = Transformer(vocab=16, d_model=8, n_heads=2, n_layers=2,
                        d_ff=16, max_seq_len=8, seed=0)
    seq = np.array([1, 5, 3, 7], dtype=np.int64)
    losses = []
    for step in range(50):
        ids = seq[:-1].reshape(1, -1)
        targets = seq[1:].reshape(1, -1)
        logits = model.forward(ids)
        probs = softmax.forward(logits)
        log_probs = np.log(np.maximum(probs[0, np.arange(3), targets[0]], 1e-12))
        loss = -float(np.mean(log_probs))
        losses.append(loss)
        one_hot = np.zeros_like(logits)
        one_hot[0, np.arange(3), targets[0]] = 1.0
        d_logits = (probs - one_hot) / 3
        model.backward(d_logits, lr=0.1)
    return losses[-1] < losses[0]


def main() -> int:
    results = [
        ("embedding_forward_shape", test_embedding_forward_shape()),
        ("positional_encoding_adds_to_emb", test_positional_encoding_adds_to_emb()),
        ("block_forward_shape", test_block_forward_shape()),
        ("transformer_forward_shape", test_transformer_forward_shape()),
        ("transformer_sample_returns_extended_ids", test_transformer_sample_returns_extended_ids()),
        ("transformer_loss_decreases", test_transformer_loss_decreases()),
    ]
    for name, ok in results:
        flag = "PASS" if ok else "FAIL"
        print(f"  {name:42s} {flag}")
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
