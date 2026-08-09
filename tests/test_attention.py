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
    """Multi-head with n_heads=4 is equivalent to concat of 4 single-head
    modules. Each single-head has d_model=8 (= d_k * n_heads), W_q shape
    (8,8) — sliced columns of the multi-head's W_q, etc. After concat,
    project through the multi-head's W_o to recover d_model=32."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((1, 3, 32))

    mha4 = MultiHeadAttention(d_model=32, n_heads=4, seed=42)
    mha1s = []
    for h in range(4):
        m = MultiHeadAttention(d_model=8, n_heads=1, seed=0)
        s, e = h * 8, (h + 1) * 8
        m.W_q = mha4.W_q[:, s:e].copy()  # (32, 8)
        m.W_k = mha4.W_k[:, s:e].copy()
        m.W_v = mha4.W_v[:, s:e].copy()
        m.W_o = np.eye(8)  # single-head d_model=8, no projection; identity here
        mha1s.append(m)

    out4 = mha4.forward(x)  # (1, 3, 32)
    # Single-head: project x via sliced W_q,k,v, attend, project via identity.
    # For the equivalence we need to ALSO use the same per-head W_o slices.
    # mha4's W_o maps merged (B, T, 32) -> (B, T, 32). Per-head slice maps
    # d_k=8 -> d_k=8 (then concat). To make single-heads identical to
    # mha4's internal path, we must NOT use mha4's W_o here — single-head
    # output is d_model=8 = d_k, then concat gives (B, T, 32) and we apply
    # mha4's W_o externally.
    head_outs = []
    for h in range(4):
        m = mha1s[h]
        s, e = h * 8, (h + 1) * 8
        Q = x @ m.W_q  # (B, T, 8)
        K = x @ m.W_k
        V = x @ m.W_v
        # Single-head scores, mask, softmax, attend — replicate mha internals
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(m.d_k)
        T = scores.shape[-1]
        mask = np.triu(np.full((T, T), -np.inf), k=1)
        scores = scores + mask
        sh = scores - scores.max(axis=-1, keepdims=True)
        aw = np.exp(sh) / np.exp(sh).sum(axis=-1, keepdims=True)
        head_outs.append(aw @ V)  # (B, T, 8)
    out_concat = np.concatenate(head_outs, axis=-1) @ mha4.W_o  # (B, T, 32)
    return bool(np.allclose(out4, out_concat, atol=1e-6))


def test_mha_backward_finite_diff() -> bool:
    """Small config: d_model=4, n_heads=2, seq=2. FD vs analytical."""
    d_model, n_heads, T = 4, 2, 2
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, seed=0)
    x = np.random.default_rng(3).standard_normal((1, T, d_model))
    eps = 1e-5

    def loss(x_in):
        # Re-init MHA to reset any cached state from prior in-place updates.
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
    # atol loose: FD on small RNG noise
    return np.allclose(grad_x, grad_num, atol=1e-5)


def main() -> int:
    results = [
        ("mha_forward_shape", test_mha_forward_shape()),
        ("mha_causal_mask_blocks_future", test_mha_causal_mask_blocks_future()),
        ("mha_multihead_split_equivalence", test_mha_multihead_split_equivalence()),
        ("mha_backward_finite_diff", test_mha_backward_finite_diff()),
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
