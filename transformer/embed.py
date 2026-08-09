"""Embedding + sinusoidal positional encoding.

ponytail: Embedding is a lookup table (vocab_size, d_model); PE adds a
fixed sinusoidal vector to token embeddings. No learned position
embedding — matches 3Blue1Brown viz and ticket spec.
"""
import numpy as np


class Embedding:
    """Token embedding lookup. token_ids: (B, T) int -> (B, T, d_model)."""

    def __init__(self, vocab_size: int, d_model: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        # Small init scale to keep activations sane at the start.
        self.embedding = rng.standard_normal((vocab_size, d_model)) * 0.02

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        return self.embedding[token_ids]


class PositionalEncoding:
    """Sinusoidal positional encoding. Adds a fixed vector to inputs.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, max_seq_len: int, d_model: int):
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        # Precompute (max_seq_len, d_model)
        pos = np.arange(max_seq_len).reshape(-1, 1)
        i = np.arange(d_model).reshape(1, -1)
        div = np.exp(-np.log(10000.0) * (2 * (i // 2)) / d_model)
        pe = np.zeros((max_seq_len, d_model))
        pe[:, 0::2] = np.sin(pos * div[:, 0::2])
        pe[:, 1::2] = np.cos(pos * div[:, 1::2])
        self.pe = pe.astype(np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (B, T, d_model). Adds PE[:T, :] to each batch."""
        T = x.shape[1]
        return x + self.pe[:T].reshape(1, T, self.d_model)
