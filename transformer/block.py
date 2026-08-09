"""Transformer block: attention + FFN with residual + LayerNorm.

ponytail: pre-norm variant (`x = x + sublayer(LN(x))`), which is the
3Blue1Brown convention and is more stable than post-norm for
from-scratch training.

Block layout:
  x = x + MHA(LN1(x))
  x = x + FFN(LN2(x))
where FFN = Linear → ReLU → Linear.

Inlined pre-norm: no Residual wrapper, no private wrapper classes —
the gradient split (grad + sublayer_grad) is the residual identity,
written directly. Less ceremony than wrapping a 2-op subgraph.
"""
import numpy as np

from layers import LayerNorm, Linear, ReLU
from transformer.attention import MultiHeadAttention


class Block:
    """One transformer block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, seed: int = 42):
        self.d_model = d_model
        self.mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, seed=seed)
        self.ln1 = LayerNorm(d_model=d_model)
        self.ln2 = LayerNorm(d_model=d_model)
        # FFN: Linear(d_model, d_ff) → ReLU → Linear(d_ff, d_model)
        self.ffn_lin1 = Linear(d_model, d_ff, rng_seed=seed + 1)
        self.ffn_lin2 = Linear(d_ff, d_model, rng_seed=seed + 2)
        self.relu = ReLU()

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x + self.mha.forward(self.ln1.forward(x))
        x = x + self.ffn_lin2.forward(
            self.relu.forward(self.ffn_lin1.forward(self.ln2.forward(x)))
        )
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Residual #2: x = x + FFN(LN2(x)). grad_out splits into identity
        # path (+grad) and FFN path.
        d_lin2_out = grad
        d_lin2_in = self.ffn_lin2.backward(d_lin2_out)
        d_relu_in = self.relu.backward(d_lin2_in)
        d_lin1_in = self.ffn_lin1.backward(d_relu_in)
        d_ln2_in = self.ln2.backward(d_lin1_in)
        grad = grad + d_ln2_in  # residual split
        # Residual #1: x = x + MHA(LN1(x)).
        d_mha_out = grad
        d_mha_in = self.mha.backward(d_mha_out)
        d_ln1_in = self.ln1.backward(d_mha_in)
        return grad + d_ln1_in  # residual split
