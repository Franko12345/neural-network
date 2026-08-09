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
        Critical: compute dL/dx using W BEFORE mutating W (PR #14/15
        pitfall: same bug existed in MultiHeadAttention and in Linear).

        Handles N-D input via flatten-leading-dims: forward() accepts
        (B, T, fan_in) etc. via broadcasting; backward mirrors that.
        """
        assert self.x is not None, "backward called before forward"
        x_flat = self.x.reshape(-1, self.W.shape[0])
        grad_flat = grad.reshape(-1, self.W.shape[1])
        self.dW = x_flat.T @ grad_flat
        self.db = grad_flat.sum(axis=0, keepdims=True)
        d_x_flat = grad_flat @ self.W.T  # use W before mutation
        self.W -= self.dW
        self.b -= self.db
        return d_x_flat.reshape(self.x.shape)


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
    """Softmax along `axis`. After forward(), `self.a` holds the probs.

    ponytail: callers in this codebase fuse with cross-entropy
    (use `self.a - y_target` directly; cheaper + numerically nicer
    than generic backward). backward() stays for non-CE cases."""

    def __init__(self, axis: int = -1):
        self.axis = axis

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        shift = x - x.max(axis=self.axis, keepdims=True)
        e = np.exp(shift)
        self.a = e / e.sum(axis=self.axis, keepdims=True)
        return self.a

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Generic Jacobian-vector product per row.
        a = self.a
        dot = (a * grad).sum(axis=self.axis, keepdims=True)
        return a * (grad - dot)


class LayerNorm:
    """Layer normalization over the last axis. Learnable gamma/beta.

    ponytail: gamma init = 1, beta init = 0; standard parametrization.
    eps = 1e-5 to match PyTorch default.
    """

    def __init__(self, d_model: int, eps: float = 1e-5,
                 gamma: np.ndarray | None = None,
                 beta: np.ndarray | None = None):
        self.d_model = d_model
        self.eps = eps
        # trainable params; init gamma=1, beta=0
        self.gamma = np.ones((d_model,)) if gamma is None else gamma.astype(float)
        self.beta = np.zeros((d_model,)) if beta is None else beta.astype(float)
        self.x = None
        self.mu = None
        self.var = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        # mean/var along last axis; keepdims so broadcast works on (..., d_model)
        self.mu = x.mean(axis=-1, keepdims=True)
        self.var = x.var(axis=-1, keepdims=True)
        self.x_hat = (x - self.mu) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_hat + self.beta

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Standard LN backward. d_model = D for shorthand.
        x_hat = self.x_hat
        D = x_hat.shape[-1]
        # d/dx_hat
        dx_hat = grad * self.gamma
        # d/dvar
        dvar = (dx_hat * (self.x - self.mu) * -0.5 *
                (self.var + self.eps) ** -1.5).sum(axis=-1, keepdims=True)
        # d/dmu
        dmu = (dx_hat * -1.0 / np.sqrt(self.var + self.eps)).sum(
            axis=-1, keepdims=True
        ) + dvar * ((self.x - self.mu) * -2.0).mean(axis=-1, keepdims=True)
        # d/dx
        dx = dx_hat / np.sqrt(self.var + self.eps) + dvar * 2.0 * (
            self.x - self.mu
        ) / D + dmu / D
        # gradients on gamma/beta; in-place SGD update like Linear
        self.d_gamma = (grad * x_hat).sum(axis=tuple(range(grad.ndim - 1)))
        self.d_beta = grad.sum(axis=tuple(range(grad.ndim - 1)))
        self.gamma -= self.d_gamma
        self.beta -= self.d_beta
        return dx
