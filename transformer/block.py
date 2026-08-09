"""Transformer block: attention + FFN with residual + LayerNorm.

ponytail: pre-norm variant (`x = x + sublayer(LN(x))`), which is the
3Blue1Brown convention and is more stable than post-norm for
from-scratch training.

Block layout:
  x = x + MHA(LN1(x))
  x = x + FFN(LN2(x))
where FFN = Linear → ReLU → Linear.
"""
import numpy as np

from layers import LayerNorm, Linear, ReLU
from modules import Residual
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
        # Residual wrappers (pre-norm: sublayer takes LN(x))
        self.res1 = Residual(_MHAWrapper(self.mha, self.ln1))
        self.res2 = Residual(_FFNWrapper(self.ffn_lin1, self.relu, self.ffn_lin2, self.ln2))
        self.x = None  # cache for backward

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return self.res2.forward(self.res1.forward(x))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Backward through res2 → res1 → input
        grad = self.res2.backward(grad)
        grad = self.res1.backward(grad)
        return grad


class _MHAWrapper:
    """Apply LN first, then MHA. Used by Residual as the wrapped fn."""

    def __init__(self, mha: MultiHeadAttention, ln: LayerNorm):
        self.mha = mha
        self.ln = ln

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.mha.forward(self.ln.forward(x))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # grad is dL/d(MHA_out). dL/d(LN_out) = dL/d(MHA_in) via MHA backward.
        d_ln_out = self.mha.backward(grad)
        # LN backward returns dL/d(LN_in) = dL/d(block_x).
        return self.ln.backward(d_ln_out)


class _FFNWrapper:
    """Apply LN first, then FFN (Linear→ReLU→Linear)."""

    def __init__(self, lin1: Linear, relu: ReLU, lin2: Linear, ln: LayerNorm):
        self.lin1 = lin1
        self.relu = relu
        self.lin2 = lin2
        self.ln = ln

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.lin2.forward(self.relu.forward(self.lin1.forward(self.ln.forward(x))))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        d_lin2_in = self.lin2.backward(grad)
        d_relu_in = self.relu.backward(d_lin2_in)
        d_lin1_in = self.lin1.backward(d_relu_in)
        return self.ln.backward(d_lin1_in)
