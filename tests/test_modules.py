"""Headless validation of LayerNorm, Residual, AdamW.

ponytail: stdlib only, no pytest. One file, asserts, exit 0/1.
"""
from __future__ import annotations

import sys

import numpy as np

from layers import LayerNorm  # noqa: F401
from modules import Residual
from optim import AdamW


def test_layernorm_forward_normalizes_last_axis() -> bool:
    ln = LayerNorm(d_model=4)
    x = np.random.default_rng(0).standard_normal((2, 4)) * 3 + 5
    out = ln.forward(x)
    # after LN: mean ~ 0, std ~ 1 along last axis (gamma=1, beta=0)
    return (
        out.shape == x.shape
        and np.allclose(out.mean(axis=-1), 0.0, atol=1e-6)
        and np.allclose(out.std(axis=-1), 1.0, atol=1e-5)
    )


def test_layernorm_gamma_beta_trainable() -> bool:
    ln = LayerNorm(d_model=3, gamma=np.array([2.0, 0.5, -1.0]), beta=np.array([1.0, -1.0, 0.0]))
    x = np.zeros((1, 3))
    out = ln.forward(x)
    # mean=0,std=1 -> normalized is 0; then * gamma + beta = beta
    return np.allclose(out, np.array([[1.0, -1.0, 0.0]]))


def test_layernorm_backward_numerical() -> bool:
    """Finite-diff check on LayerNorm small input. Re-init LN each FD
    call so the in-place gamma/beta update doesn't poison later samples."""
    x = np.random.default_rng(1).standard_normal((1, 3))
    eps = 1e-5

    # Re-init LN at the same gamma/beta for every loss eval.
    template = LayerNorm(d_model=3)
    template.forward(x)  # populate mu/var cache only — gamma/beta still default
    g0, b0 = template.gamma.copy(), template.beta.copy()

    def loss(x_in):
        ln = LayerNorm(d_model=3, gamma=g0.copy(), beta=b0.copy())
        return float(ln.forward(x_in).sum())

    ln = LayerNorm(d_model=3, gamma=g0.copy(), beta=b0.copy())
    ln.forward(x)
    grad_x = ln.backward(np.ones_like(x))

    grad_num = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            xp, xm = x.copy(), x.copy()
            xp[i, j] += eps
            xm[i, j] -= eps
            grad_num[i, j] = (loss(xp) - loss(xm)) / (2 * eps)
    return np.allclose(grad_x, grad_num, atol=1e-5)


class _Scale2:
    """Test stub: y = 2x, dy/dx = 2."""
    def forward(self, x):
        return x * 2.0
    def backward(self, grad):
        return grad * 2.0


def test_residual_forward_adds_identity() -> bool:
    """Residual(fn) should produce fn(x) + x."""
    res = Residual(_Scale2())
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = res.forward(x)
    return np.allclose(out, 3.0 * x)


def test_residual_backward_splits() -> bool:
    """dL/dx from Residual = dL/dout + dL/d(fn(x))."""
    res = Residual(_Scale2())
    x = np.array([[1.0, 2.0]])
    res.forward(x)
    grad = res.backward(np.ones_like(x))
    # grad_out (1) + fn'(x)*grad_out (2) = 3 per element
    return np.allclose(grad, 3.0)


class _Param:
    """Minimal param container for AdamW tests."""
    def __init__(self, data, grad=None):
        self.data = data
        self.grad = grad


def test_adamw_zero_grad_resets_state() -> bool:
    p = _Param(np.zeros((2, 2)), np.ones_like(np.zeros((2, 2))))
    adam = AdamW([p], lr=1e-2)
    adam.step()
    assert not np.allclose(p.data, 0)
    p.grad = np.zeros_like(p.data)
    adam.zero_grad()
    return p.grad is None


def test_adamw_bias_correction_at_step_zero() -> bool:
    """First step is finite (bias correction handles v=0 case)."""
    p = _Param(np.array([[1.0, 2.0]]), np.ones((1, 2)))
    before = p.data.copy()
    adam = AdamW([p], lr=0.1, betas=(0.9, 0.95), weight_decay=0.0)
    adam.step()
    return bool(np.all(np.isfinite(p.data)) and not np.allclose(p.data, before))


def test_adamw_weight_decay_decoupled() -> bool:
    """With grad=0 and lr>0, weight decay alone shrinks data."""
    p = _Param(np.array([[2.0, 4.0]]), np.zeros((1, 2)))
    before = p.data.copy()
    adam = AdamW([p], lr=0.1, betas=(0.9, 0.95), weight_decay=0.5)
    adam.step()
    return bool(float(np.abs(p.data).sum()) < float(np.abs(before).sum()))


def main() -> int:
    results = [
        ("layernorm_forward_normalizes_last_axis", test_layernorm_forward_normalizes_last_axis()),
        ("layernorm_gamma_beta_trainable", test_layernorm_gamma_beta_trainable()),
        ("layernorm_backward_numerical", test_layernorm_backward_numerical()),
        ("residual_forward_adds_identity", test_residual_forward_adds_identity()),
        ("residual_backward_splits", test_residual_backward_splits()),
        ("adamw_zero_grad_resets_state", test_adamw_zero_grad_resets_state()),
        ("adamw_bias_correction_at_step_zero", test_adamw_bias_correction_at_step_zero()),
        ("adamw_weight_decay_decoupled", test_adamw_weight_decay_decoupled()),
    ]
    for name, ok in results:
        flag = "PASS" if ok else "FAIL"
        print(f"  {name:42s} {flag}")
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
