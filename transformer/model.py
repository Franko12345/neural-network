"""Decoder-only transformer: embedding + pos enc + N blocks + LM head.

ponytail: caller owns the Softmax (passed at training time) so they
can fuse with cross-entropy. Sample loop uses softmax independently.

Architecture (defaults match v2 spec):
  vocab_size = 256, d_model = 128, n_heads = 4, n_layers = 4,
  d_ff = 512, max_seq_len = 128.
"""
import numpy as np

from layers import LayerNorm, Linear
from transformer.block import Block
from transformer.embed import Embedding, PositionalEncoding


class Transformer:
    """Decoder-only transformer with stack of blocks + LM head."""

    def __init__(
        self,
        vocab: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 128,
        seed: int = 42,
    ):
        self.vocab = vocab
        self.d_model = d_model
        self.n_layers = n_layers
        self.token_emb = Embedding(vocab_size=vocab, d_model=d_model, seed=seed)
        self.pos_enc = PositionalEncoding(max_seq_len=max_seq_len, d_model=d_model)
        self.blocks = [
            Block(d_model=d_model, n_heads=n_heads, d_ff=d_ff, seed=seed + i)
            for i in range(n_layers)
        ]
        self.final_ln = LayerNorm(d_model=d_model)
        self.lm_head = Linear(d_model, vocab, rng_seed=seed + 100)
        # Cache for backward

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """token_ids: (B, T) int -> logits: (B, T, vocab)."""
        x = self.token_emb.forward(token_ids)  # (B, T, d_model)
        x = self.pos_enc.forward(x)
        for block in self.blocks:
            x = block.forward(x)
        x = self.final_ln.forward(x)
        logits = self.lm_head.forward(x)  # (B, T, vocab)
        return logits

    def backward(self, grad_logits: np.ndarray, lr: float = 1.0,
                update: bool = True) -> None:
        """grad_logits: (B, T, vocab) dL/d(logits). Updates all params in place
        UNLESS update=False (PR #17 trainer uses AdamW and must suppress
        the in-place SGD to avoid double-updating)."""
        # Scale upstream gradient by lr so Linear in-place lr=1.0 applies
        # the requested step size.
        grad_logits = grad_logits * lr
        # lm_head backward
        grad = self.lm_head.backward(grad_logits, update=update)
        # final_ln backward
        grad = self.final_ln.backward(grad, update=update)
        # blocks backward (reverse order)
        for block in reversed(self.blocks):
            grad = block.backward(grad, update=update)
        # token_emb + pos_enc have no params; gradients on x are discarded.

    def sample(
        self,
        prompt_ids: np.ndarray,
        n_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> np.ndarray:
        """Autoregressive decode. prompt_ids: (B, T_prompt) -> (B, T_prompt + n_tokens)."""
        ids = prompt_ids.copy()
        for _ in range(n_tokens):
            logits = self.forward(ids)[:, -1, :]  # (B, vocab) — last position
            # Apply temperature
            scaled = logits / max(temperature, 1e-8)
            # Softmax
            shifted = scaled - scaled.max(axis=-1, keepdims=True)
            probs = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
            # top-k truncation
            if top_k is not None and top_k < probs.shape[-1]:
                topk_idx = np.argpartition(-probs, top_k, axis=-1)[:, :top_k]
                mask = np.zeros_like(probs, dtype=bool)
                np.put_along_axis(mask, topk_idx, True, axis=-1)
                probs = np.where(mask, probs, 0.0)
                probs /= probs.sum(axis=-1, keepdims=True)
            next_ids = np.array([
                np.random.choice(probs.shape[-1], p=probs[b])
                for b in range(probs.shape[0])
            ]).reshape(-1, 1)
            ids = np.concatenate([ids, next_ids], axis=1)
        return ids
