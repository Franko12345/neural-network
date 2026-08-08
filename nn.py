"""Feedforward neural network with backprop. NumPy only.

ponytail: full-batch SGD for clarity. Swap to minibatches if/when
datasets get bigger than ~10k samples.
"""
from __future__ import annotations

import numpy as np


def _activate(name: str, z: np.ndarray) -> np.ndarray:
    if name == "relu":
        return np.maximum(0.0, z)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    if name == "tanh":
        return np.tanh(z)
    if name == "softmax":
        z_shift = z - z.max(axis=1, keepdims=True)
        e = np.exp(z_shift)
        return e / e.sum(axis=1, keepdims=True)
    raise ValueError(f"unknown activation {name!r}")


def _activate_deriv(name: str, z: np.ndarray, a: np.ndarray) -> np.ndarray:
    if name == "relu":
        return (z > 0).astype(float)
    if name == "sigmoid":
        return a * (1.0 - a)
    if name == "tanh":
        return 1.0 - a * a
    if name == "softmax":
        # softmax + cross-entropy handles this on the caller side
        return np.ones_like(z)
    raise ValueError(f"unknown activation {name!r}")


class NeuralNetwork:
    """Stack of fully-connected layers with backprop."""

    def __init__(self, layer_sizes: list[int], activations: list[str]):
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("activations must be len(layer_sizes)-1")
        rng = np.random.default_rng(42)  # reproducible
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))  # Xavier
            self.layers.append({
                "W": rng.standard_normal((fan_in, fan_out)) * scale,
                "b": np.zeros((1, fan_out)),
                "activation": activations[i],
                "z": None, "a": None, "dz": None,
            })

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            layer["z"] = out @ layer["W"] + layer["b"]
            layer["a"] = _activate(layer["activation"], layer["z"])
            out = layer["a"]
        return out

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, lr: float) -> None:
        n = y_true.shape[0]
        last = self.layers[-1]
        # Fused softmax + cross-entropy gradient: δ = ŷ - y.
        if last["activation"] == "softmax":
            last["dz"] = (y_pred - y_true) / n
        else:
            da = -(y_true - y_pred) / n
            last["dz"] = da * _activate_deriv(last["activation"], last["z"], last["a"])

        for i in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[i]
            prev = self.layers[i - 1]
            layer["dW"] = prev["a"].T @ layer["dz"]
            layer["db"] = layer["dz"].sum(axis=0, keepdims=True)
            da_prev = layer["dz"] @ layer["W"].T
            prev["dz"] = da_prev * _activate_deriv(prev["activation"], prev["z"], prev["a"])

        layer0 = self.layers[0]
        layer0["dW"] = self.input_cache.T @ layer0["dz"]
        layer0["db"] = layer0["dz"].sum(axis=0, keepdims=True)

        for layer in self.layers:
            layer["W"] -= lr * layer["dW"]
            layer["b"] -= lr * layer["db"]

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 1000,
        lr: float = 0.05,
        log_every: int = 0,
    ) -> list[float]:
        losses: list[float] = []
        for epoch in range(epochs):
            self.input_cache = X
            Y_hat = self.forward(X)
            loss = self._loss(Y, Y_hat)
            losses.append(loss)
            self.backward(Y, Y_hat, lr)
            if log_every and epoch % log_every == 0:
                acc = (Y_hat.argmax(axis=1) == Y.argmax(axis=1)).mean()
                print(f"epoch {epoch:4d}  loss={loss:.4f}  acc={acc:.3f}")
        return losses

    @staticmethod
    def _loss(Y: np.ndarray, Y_hat: np.ndarray) -> float:
        eps = 1e-12
        return float(-np.mean(np.sum(Y * np.log(Y_hat + eps), axis=1)))


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes))
    out[np.arange(y.shape[0]), y] = 1.0
    return out


if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (40, 2))
    Y = one_hot((X[:, 0] * X[:, 1] > 0).astype(int), 2)
    nn = NeuralNetwork([2, 8, 8, 2], ["relu", "relu", "softmax"])
    losses = nn.fit(X, Y, epochs=500, lr=0.05)
    assert losses[-1] < losses[0], f"loss not decreasing: {losses[0]:.4f} -> {losses[-1]:.4f}"
    acc = (nn.forward(X).argmax(axis=1) == Y.argmax(axis=1)).mean()
    print(f"sanity ok: loss {losses[0]:.3f}->{losses[-1]:.3f}, acc={acc:.2f}")
    sys.exit(0)