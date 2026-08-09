"""Headless validation of transformer/attention.py — MultiHeadAttention.

ponytail: stdlib only, no pytest. One file, asserts, exit 0/1.
"""
from __future__ import annotations

import sys

import numpy as np

from transformer.attention import MultiHeadAttention


def test_mha_forward_shape() -> bool:
    """Forward (B, T, d_model) -> (B, T, d_model)."""
    mha = MultiHeadAttention(d_model=8, n_heads=2, seed=0)
    x = np.random.default_rng(0).standard_normal((1, 4, 8))
    out = mha.forward(x)
    return out.shape == (1, 4, 8)


def test_mha_causal_mask_blocks_future() -> bool:
    """Position t must have zero attention weight on positions > t."""
    mha = MultiHeadAttention(d_model=8, n_heads=1, seed=0)
    x = np.random.default_rng(1).standard_normal((1, 4, 8))
    mha.forward(x)  # populates self.attn_weights (n_heads, B, T, T)
    # Take head 0, batch 0
    aw = mha.attn_weights[0, 0]  # (T, T)
    T = aw.shape[0]
    for t in range(T):
        for s in range(t + 1, T):
            if aw[t, s] != 0.0:
                return False
    return True


def test_mha_multihead_split_equivalence() -> bool:
    """4-head on d_model=32 equals concat of 4 single-head per-head paths,
    then mha4's W_o. Single-head math is replicated inline (avoids
    constructing 4 MHA instances just to slice their W matrices)."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((1, 3, 32))
    d_k = 8

    mha4 = MultiHeadAttention(d_model=32, n_heads=4, seed=42)
    out4 = mha4.forward(x)  # (1, 3, 32)

    head_outs = []
    for h in range(4):
        s, e = h * d_k, (h + 1) * d_k
        Q = x @ mha4.W_q[:, s:e]  # (B, T, d_k)
        K = x @ mha4.W_k[:, s:e]
        V = x @ mha4.W_v[:, s:e]
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
        T = scores.shape[-1]
        mask = np.triu(np.full((T, T), -np.inf), k=1)
        scores = scores + mask
        sh = scores - scores.max(axis=-1, keepdims=True)
        aw = np.exp(sh) / np.exp(sh).sum(axis=-1, keepdims=True)
        head_outs.append(aw @ V)  # (B, T, d_k)
    out_concat = np.concatenate(head_outs, axis=-1) @ mha4.W_o  # (B, T, 32)
    return bool(np.allclose(out4, out_concat, atol=1e-6))


def test_mha_backward_finite_diff() -> bool:
    """d_model=8, n_heads=2, T=2 (matches ticket 05 spec)."""
    d_model, n_heads, T = 8, 2, 2
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, seed=0)
    x = np.random.default_rng(3).standard_normal((1, T, d_model))
    eps = 1e-5

    def loss(x_in):
        m = MultiHeadAttention(d_model=d_model, n_heads=n_heads, seed=0)
        return float(m.forward(x_in).sum())

    mha.forward(x)
    grad_x = mha.backward(np.ones_like(x))

    grad_num = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        xp, xm = x.copy(), x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        grad_num[idx] = (loss(xp) - loss(xm)) / (2 * eps)
        it.iternext()
    return np.allclose(grad_x, grad_num, atol=1e-5)


def test_mha_backward_grad_shapes() -> bool:
    """Verify W_q/k/v/o and grad_x have the expected shapes after backward."""
    d_model, n_heads, T = 8, 2, 3
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, seed=0)
    x = np.random.default_rng(4).standard_normal((1, T, d_model))
    mha.forward(x)
    grad_x = mha.backward(np.ones((1, T, d_model)))
    return (
        grad_x.shape == x.shape
        and mha.W_q.shape == (d_model, d_model)
        and mha.W_k.shape == (d_model, d_model)
        and mha.W_v.shape == (d_model, d_model)
        and mha.W_o.shape == (d_model, d_model)
    )


def main() -> int:
    results = [
        ("mha_forward_shape", test_mha_forward_shape()),
        ("mha_causal_mask_blocks_future", test_mha_causal_mask_blocks_future()),
        ("mha_multihead_split_equivalence", test_mha_multihead_split_equivalence()),
        ("mha_backward_finite_diff", test_mha_backward_finite_diff()),
        ("mha_backward_grad_shapes", test_mha_backward_grad_shapes()),
    ]
    for name, ok in results:
        flag = "PASS" if ok else "FAIL"
        print(f"  {name:38s} {flag}")
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
