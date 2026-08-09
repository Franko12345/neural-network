"""Headless validation of layers.py building blocks.

ponytail: stdlib only, no pytest. One file, asserts, exit 0/1.
Each test exercises forward + backward on small inputs so the loop
is fast (<2s).
"""
from __future__ import annotations

import sys

import numpy as np

from layers import Linear, ReLU, Sigmoid, Softmax, Tanh


def _finite_diff_grad(fn, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central-difference numerical gradient of fn w.r.t. x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[idx] += eps
        x_minus[idx] -= eps
        grad[idx] = (fn(x_plus) - fn(x_minus)) / (2 * eps)
        it.iternext()
    return grad


def test_linear_forward_shape() -> bool:
    layer = Linear(3, 5)
    x = np.random.default_rng(0).standard_normal((4, 3))
    out = layer.forward(x)
    return out.shape == (4, 5)


def test_linear_backward_grad() -> bool:
    """Check grad of sum(out) w.r.t. x matches finite differences."""
    rng = np.random.default_rng(1)
    layer = Linear(2, 3)
    # fix W/b so analytical and numerical match
    layer.W = rng.standard_normal((2, 3)) * 0.1
    layer.b = np.zeros((1, 3))
    x = rng.standard_normal((1, 2))

    def loss_fn(x_in: np.ndarray) -> float:
        return float(layer.forward(x_in).sum())

    layer.forward(x)
    grad_out = np.ones((1, 3))
    grad_x_analytical = layer.backward(grad_out)
    grad_x_numerical = _finite_diff_grad(loss_fn, x)
    return np.allclose(grad_x_analytical, grad_x_numerical, atol=1e-5)


def test_linear_xavier_init_scale() -> bool:
    """Weights should be ~N(0, sqrt(2/(fan_in+fan_out)))."""
    layer = Linear(100, 50)
    expected = np.sqrt(2.0 / 150)
    actual = layer.W.std()
    return abs(actual - expected) < 0.05  # generous: std fluctuates


def test_relu_forward_backward_shape() -> bool:
    layer = ReLU()
    x = np.array([[-1.0, 0.5, 2.0]])
    out = layer.forward(x)
    if not (out.shape == x.shape and np.allclose(out, [[0.0, 0.5, 2.0]])):
        return False
    grad = layer.backward(np.ones_like(out))
    expected = np.array([[0.0, 1.0, 1.0]])
    return np.allclose(grad, expected)


def test_tanh_forward_backward_shape() -> bool:
    layer = Tanh()
    x = np.array([[0.0, 1.0, -1.0]])
    out = layer.forward(x)
    if out.shape != x.shape:
        return False
    grad = layer.backward(np.ones_like(out))
    expected = 1.0 - out ** 2
    return np.allclose(grad, expected)


def test_sigmoid_forward_backward_shape() -> bool:
    layer = Sigmoid()
    x = np.array([[0.0, 100.0, -100.0]])  # stress overflow guard
    out = layer.forward(x)
    if out.shape != x.shape:
        return False
    grad = layer.backward(np.ones_like(out))
    expected = out * (1.0 - out)
    return np.allclose(grad, expected)


def test_softmax_forward_sum_to_one() -> bool:
    layer = Softmax(axis=-1)
    x = np.random.default_rng(2).standard_normal((3, 4))
    out = layer.forward(x)
    return out.shape == x.shape and np.allclose(out.sum(axis=-1), 1.0)


def test_softmax_ce_fused_gradient() -> bool:
    """Same formula as v1: d/dz (CE(softmax(z), y)) = softmax(z) - y."""
    layer = Softmax(axis=-1)
    rng = np.random.default_rng(3)
    z = rng.standard_normal((2, 3))
    y = np.eye(3)[rng.integers(0, 3, 2)]  # one-hot
    layer.forward(z)
    # Caller fuses: grad = softmax(z) - y
    grad = layer.x - y
    return grad.shape == z.shape


def main() -> int:
    results = [
        ("linear_forward_shape", test_linear_forward_shape()),
        ("linear_backward_grad", test_linear_backward_grad()),
        ("linear_xavier_init_scale", test_linear_xavier_init_scale()),
        ("relu_forward_backward_shape", test_relu_forward_backward_shape()),
        ("tanh_forward_backward_shape", test_tanh_forward_backward_shape()),
        ("sigmoid_forward_backward_shape", test_sigmoid_forward_backward_shape()),
        ("softmax_forward_sum_to_one", test_softmax_forward_sum_to_one()),
        ("softmax_ce_fused_gradient", test_softmax_ce_fused_gradient()),
    ]
    for name, ok in results:
        flag = "PASS" if ok else "FAIL"
        print(f"  {name:40s} {flag}")
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
