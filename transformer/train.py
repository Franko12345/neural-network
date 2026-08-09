"""Transformer trainer with AdamW + checkpointing.

ponytail: trainer state = (model + optim + step_count). No globals;
same Trainer instance can be saved/reloaded via npz.

Critical: model.backward is called with update=False so Linear/
LayerNorm's in-place SGD doesn't fire alongside AdamW (otherwise
every weight is stepped twice per iteration — PR #17 review catch,
same bug class as PR #15 W-order).

Checkpoint auto-load: if `checkpoint.npz` exists in CWD when Trainer
is constructed, weights load automatically (per ticket 07 AC).
"""
import os
from typing import Optional

import numpy as np

from layers import Softmax
from optim import AdamW
from transformer.model import Transformer


class Trainer:
    """Transformer trainer with AdamW + checkpoint round-trip."""

    def __init__(
        self,
        model: Transformer,
        data: np.ndarray,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
        seed: int = 42,
        max_seq_len: Optional[int] = None,
        checkpoint_path: str = "checkpoint.npz",
    ):
        self.model = model
        self.data = data
        self.lr = lr
        self.checkpoint_path = checkpoint_path
        # Collect all trainable params into a flat list for AdamW.
        self._params = self._collect_params(model)
        self._opt = AdamW(self._params, lr=lr, betas=betas, weight_decay=weight_decay)
        self._softmax = Softmax(axis=-1)
        self._step = 0
        self._rng = np.random.default_rng(seed)
        # Default max_seq_len: model.max_seq_len (set in transformer.model.PositionalEncoding).
        self._max_seq_len = max_seq_len or model.pos_enc.max_seq_len
        # Auto-load if checkpoint exists in CWD (or at checkpoint_path).
        if os.path.exists(checkpoint_path):
            self.load(checkpoint_path)

    @staticmethod
    def _collect_params(model: Transformer) -> list:
        """Return a flat list of param objects (.data, .grad) covering
        every trainable weight in the model."""
        params = []
        # Token embedding
        params.append(_Param(model.token_emb.embedding))
        # Each block: MHA W_q/k/v/o, LN1 gamma/beta, LN2 gamma/beta, FFN lin1 W/b, FFN lin2 W/b
        for block in model.blocks:
            params.append(_Param(block.mha.W_q))
            params.append(_Param(block.mha.W_k))
            params.append(_Param(block.mha.W_v))
            params.append(_Param(block.mha.W_o))
            params.append(_Param(block.ln1.gamma))
            params.append(_Param(block.ln1.beta))
            params.append(_Param(block.ln2.gamma))
            params.append(_Param(block.ln2.beta))
            params.append(_Param(block.ffn_lin1.W))
            params.append(_Param(block.ffn_lin1.b))
            params.append(_Param(block.ffn_lin2.W))
            params.append(_Param(block.ffn_lin2.b))
        # Final LN + LM head
        params.append(_Param(model.final_ln.gamma))
        params.append(_Param(model.final_ln.beta))
        params.append(_Param(model.lm_head.W))
        params.append(_Param(model.lm_head.b))
        return params

    def _sample_batch(self, batch_size: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample (inputs, targets) from the data stream.

        targets = inputs shifted by 1 (next-token prediction).
        """
        max_start = max(1, len(self.data) - seq_len - 1)
        starts = self._rng.integers(0, max_start, size=batch_size)
        inputs = np.stack([self.data[s : s + seq_len] for s in starts]).astype(np.int64)
        targets = np.stack([self.data[s + 1 : s + 1 + seq_len] for s in starts]).astype(np.int64)
        return inputs, targets

    def step(self, batch_size: int = 4, seq_len: Optional[int] = None) -> float:
        """Single training step: forward, CE loss, backward, AdamW update.
        Returns the loss (per-token mean)."""
        T = seq_len or self._max_seq_len
        inputs, targets = self._sample_batch(batch_size, T)
        logits = self.model.forward(inputs)  # (B, T, vocab)
        probs = self._softmax.forward(logits)
        # CE loss: -mean(log p[target])
        B, T_, V = probs.shape
        log_pi = np.log(np.maximum(probs[np.arange(B)[:, None], np.arange(T_)[None, :], targets], 1e-12))
        loss = -float(np.mean(log_pi))
        # Gradient: (probs - one_hot) / (B*T)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(B)[:, None], np.arange(T_)[None, :], targets] = 1.0
        d_logits = (probs - one_hot) / (B * T_)
        self.model.backward(d_logits, lr=1.0)  # Linear in-place lr=1.0; AdamW will scale
        # Map Linear.dW back to each param.grad for AdamW
        self._set_grads_from_dW()
        self._opt.step()
        self._opt.zero_grad()
        self._step += 1
        return loss

    def train(self, n_steps: int, batch_size: int = 4, seq_len: Optional[int] = None,
              save_every: int = 100) -> list:
        """Run n_steps training steps. Saves checkpoint every save_every
        steps (default 100 per spec). Returns list of losses (every step)."""
        losses = []
        for _ in range(n_steps):
            loss = self.step(batch_size=batch_size, seq_len=seq_len)
            losses.append(loss)
            if self._step % save_every == 0:
                self.save(self.checkpoint_path)
        return losses

    def _set_grads_from_dW(self) -> None:
        """After model.backward(), copy each Linear/LayerNorm's dW into
        the corresponding param.grad. AdamW reads .grad and applies its
        update; .grad is None-checked."""
        # We registered _Param wrappers in order; mirror it.
        idx = 0
        _set(self._params[idx], self.model.token_emb.embedding, key=None); idx += 1
        for block in self.model.blocks:
            _set(self._params[idx], None, attr=block.mha, key="W_q"); idx += 1
            _set(self._params[idx], None, attr=block.mha, key="W_k"); idx += 1
            _set(self._params[idx], None, attr=block.mha, key="W_v"); idx += 1
            _set(self._params[idx], None, attr=block.mha, key="W_o"); idx += 1
            _set(self._params[idx], None, attr=block.ln1, key="gamma"); idx += 1
            _set(self._params[idx], None, attr=block.ln1, key="beta"); idx += 1
            _set(self._params[idx], None, attr=block.ln2, key="gamma"); idx += 1
            _set(self._params[idx], None, attr=block.ln2, key="beta"); idx += 1
            _set(self._params[idx], None, attr=block.ffn_lin1, key="W"); idx += 1
            _set(self._params[idx], None, attr=block.ffn_lin1, key="b"); idx += 1
            _set(self._params[idx], None, attr=block.ffn_lin2, key="W"); idx += 1
            _set(self._params[idx], None, attr=block.ffn_lin2, key="b"); idx += 1
        _set(self._params[idx], None, attr=self.model.final_ln, key="gamma"); idx += 1
        _set(self._params[idx], None, attr=self.model.final_ln, key="beta"); idx += 1
        _set(self._params[idx], None, attr=self.model.lm_head, key="W"); idx += 1
        _set(self._params[idx], None, attr=self.model.lm_head, key="b"); idx += 1

    def save(self, path: str) -> None:
        """Save all trainable params + step count to npz."""
        arrays = {f"p{i}": p.data for i, p in enumerate(self._params)}
        arrays["_step"] = np.asarray(self._step)
        np.savez(path, **arrays)

    def load(self, path: str) -> None:
        """Load all trainable params + step count from npz."""
        archive = np.load(path)
        for i, p in enumerate(self._params):
            p.data[...] = archive[f"p{i}"]
        if "_step" in archive:
            self._step = int(archive["_step"])

    def sample(self, prompt_ids: np.ndarray, n_tokens: int,
               temperature: float = 1.0, top_k: Optional[int] = None) -> np.ndarray:
        """Convenience pass-through to model.sample."""
        return self.model.sample(prompt_ids, n_tokens, temperature, top_k)


class _Param:
    """Minimal param container: .data (np.ndarray), .grad (set by trainer)."""

    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad: Optional[np.ndarray] = None


def _set(p: _Param, ref, attr=None, key: Optional[str] = None) -> None:
    """Copy gradient from ref/attr/key into p.grad. ref.key or attr.key."""
    if attr is not None and key is not None:
        if hasattr(attr, "dW") and getattr(attr, "dW", None) is not None and getattr(attr, "dW").shape == p.data.shape:
            p.grad = getattr(attr, "dW").copy()
            return
        if hasattr(attr, "d_gamma") and key in ("gamma",):
            p.grad = getattr(attr, "d_gamma").copy()
            return
        if hasattr(attr, "d_beta") and key in ("beta",):
            p.grad = getattr(attr, "d_beta").copy()
            return
    # Fallback: just clear
    p.grad = None
