"""Building blocks for v2: Linear, ReLU, Tanh, Sigmoid, Softmax.

ponytail: each module is a flat forward/backward pair, no flags, no
shape dispatch. v1 NeuralNetwork stays frozen — it does not import
from here.
"""
import numpy as np


class Linear:
    """Affine: y = x @ W + b. Xavier init matches v1."""

    def __init__(self, fan_in: int, fan_out: int, rng_seed: int = 42):
        rng = np.random.default_rng(rng_seed)
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        self.W: np.ndarray = rng.standard_normal((fan_in, fan_out)) * scale
        self.b: np.ndarray = np.zeros((1, fan_out))
        self.x: np.ndarray | None = None  # cached input for backward

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """grad has the shape of forward's output. Returns dL/dx.

        Updates W and b in-place with plain SGD (lr=1.0; caller scales).
        """
        assert self.x is not None, "backward called before forward"
        self.dW = self.x.T @ grad
        self.db = grad.sum(axis=0, keepdims=True)
        self.W -= self.dW  # ponytail: SGD with lr=1.0; trainer multiplies
        self.b -= self.db
        return grad @ self.W.T


class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(0.0, x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (self.x > 0).astype(float)


class Tanh:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.tanh(x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        a = np.tanh(self.x)
        return grad * (1.0 - a * a)


class Sigmoid:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        # clip to avoid overflow — same trick v1 uses
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        a = 1.0 / (1.0 + np.exp(-np.clip(self.x, -500, 500)))
        return grad * a * (1.0 - a)


class Softmax:
    """Softmax along `axis`. Exposes `.x` for fused CE gradient."""

    def __init__(self, axis: int = -1):
        self.axis = axis

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        shift = x - x.max(axis=self.axis, keepdims=True)
        e = np.exp(shift)
        return e / e.sum(axis=self.axis, keepdims=True)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Generic softmax Jacobian. Callers should fuse with CE
        (use `self.x - y_target`) — that's faster and numerically nicer."""
        a = np.exp(self.x - self.x.max(axis=self.axis, keepdims=True))
        a = a / a.sum(axis=self.axis, keepdims=True)
        # diag(a) - a a^T applied to grad, vectorized per row
        dot = (a * grad).sum(axis=self.axis, keepdims=True)
        return a * (grad - dot)
