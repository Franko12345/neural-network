"""Multi-head self-attention with causal mask.

ponytail: one class, no helper functions exposed. mask applied via
`scores += mask` (not np.where per cell). Per-head parameters are
concatenated into single matrices for compactness (W_q, W_k, W_v,
W_o each have shape (d_model, d_model)).
"""
import numpy as np


def _causal_mask(T: int) -> np.ndarray:
    """Upper-triangular -inf mask (strict causal: position t sees <= t)."""
    m = np.triu(np.full((T, T), -np.inf), k=1)
    return m


class MultiHeadAttention:
    """Self-attention with multiple heads. Forward input: (B, T, d_model)."""

    def __init__(self, d_model: int, n_heads: int, seed: int = 42):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        rng = np.random.default_rng(seed)
        # Xavier-scale init (matches layers.Linear convention)
        scale = np.sqrt(2.0 / (d_model + d_model))
        self.W_q = rng.standard_normal((d_model, d_model)) * scale
        self.W_k = rng.standard_normal((d_model, d_model)) * scale
        self.W_v = rng.standard_normal((d_model, d_model)) * scale
        self.W_o = rng.standard_normal((d_model, d_model)) * scale
        # Cache for backward
        self.x = None
        self.Q = None
        self.K = None
        self.V = None
        self.scores = None
        self.attn_weights = None  # (n_heads, B, T, T) — exposed for ticket 09
        self._mask = None

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, T, d_model) -> (n_heads, B, T, d_k)."""
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.d_k).transpose(2, 0, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """(n_heads, B, T, d_k) -> (B, T, d_model)."""
        _, B, T, _ = x.shape
        return x.transpose(1, 2, 0, 3).reshape(B, T, self.d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (B, T, d_model). returns (B, T, d_model)."""
        self.x = x
        B, T, _ = x.shape
        # Project Q/K/V per head: (B, T, d_model) @ W -> (B, T, d_model)
        Q = x @ self.W_q  # (B, T, d_model)
        K = x @ self.W_k
        V = x @ self.W_v
        # Split heads
        Qh = self._split_heads(Q)  # (H, B, T, d_k)
        Kh = self._split_heads(K)
        Vh = self._split_heads(V)
        # Scores: (H, B, T, d_k) @ (H, B, d_k, T) -> (H, B, T, T)
        scores = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        # Causal mask
        if self._mask is None or self._mask.shape[0] != T:
            self._mask = _causal_mask(T)
        scores = scores + self._mask  # broadcasts over heads and batch
        # Softmax along last axis
        shift = scores - scores.max(axis=-1, keepdims=True)
        e = np.exp(shift)
        aw = e / e.sum(axis=-1, keepdims=True)  # (H, B, T, T)
        # Attention output: (H, B, T, T) @ (H, B, T, d_k) -> (H, B, T, d_k)
        out_h = aw @ Vh
        # Merge heads: (B, T, d_model)
        out = self._merge_heads(out_h)
        # Output projection: (B, T, d_model)
        out = out @ self.W_o

        # Cache for backward
        self.Q, self.K, self.V = Qh, Kh, Vh
        self.scores = scores
        self.attn_weights = aw
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """grad: (B, T, d_model) — dL/d(out). Returns dL/d(x), updates W in place.

        ponytail: SGD with lr=1.0 (caller scales) — same as layers.Linear.
        """
        B, T, _ = grad.shape
        # dL/d(out_proj_input): backprop through W_o
        d_out_proj_input = grad @ self.W_o.T  # (B, T, d_model)
        # dL/d(W_o) = out_proj_input.T @ grad
        out_proj_input = self._merge_heads(self.attn_weights @ self.V)
        self.W_o -= out_proj_input.reshape(-1, self.d_model).T @ grad.reshape(-1, self.d_model)

        # Split dL/d(merged) back to heads
        dh = self._split_heads(d_out_proj_input)  # (H, B, T, d_k)
        # Backprop through attention: out_h = aw @ V
        # dL/d(aw) = dL/d(out_h) @ V.T
        d_aw = dh @ self.V.transpose(0, 1, 3, 2)
        d_Vh = self.attn_weights.transpose(0, 1, 3, 2) @ dh

        # Softmax backward: dL/d(scores) = aw * (d_aw - sum(d_aw * aw, -1, keepdims))
        d_scores = self.attn_weights * (d_aw - (d_aw * self.attn_weights).sum(axis=-1, keepdims=True))
        # Note: mask was applied by adding -inf, so masked positions already
        # have aw=0 and contribute nothing — no explicit zeroing needed.
        # Scale by 1/sqrt(d_k)
        d_scores = d_scores / np.sqrt(self.d_k)

        # dL/d(Qh) = d_scores @ Kh
        d_Qh = d_scores @ self.K
        d_Kh = d_scores.transpose(0, 1, 3, 2) @ self.Q

        # Merge heads back to (B, T, d_model)
        dQ = self._merge_heads(d_Qh)
        dK = self._merge_heads(d_Kh)
        dV = self._merge_heads(d_Vh)

        # dL/d(x) = dQ @ W_q.T + dK @ W_k.T + dV @ W_v.T
        d_x = dQ @ self.W_q.T + dK @ self.W_k.T + dV @ self.W_v.T

        # Parameter gradients (SGD with lr=1.0 in-place)
        x_flat = self.x.reshape(-1, self.d_model)
        self.W_q -= x_flat.T @ dQ.reshape(-1, self.d_model)
        self.W_k -= x_flat.T @ dK.reshape(-1, self.d_model)
        self.W_v -= x_flat.T @ dV.reshape(-1, self.d_model)

        return d_x
