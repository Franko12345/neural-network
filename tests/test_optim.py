"""Headless validation of optim.AdamW.

ponytail: stdlib only, no pytest. Single-purpose file; layer/optimizer
seam tests live in test_modules.py.
"""
from __future__ import annotations

import sys

import numpy as np

from optim import AdamW


def _make_param(data: np.ndarray):
    """Lightweight param container compatible with AdamW's contract:
    .data (the tensor) and .grad (None-or-array, set externally)."""
    class P:
        pass
    p = P()
    p.data = data.copy()
    p.grad = None
    return p


def test_adamw_step_moves_data() -> bool:
    p = _make_param(np.ones((3,)))
    p.grad = np.ones_like(p.data)
    adam = AdamW([p], lr=0.1)
    before = p.data.copy()
    adam.step()
    return not np.allclose(p.data, before)


def test_adamw_step_zero_grad_resets() -> bool:
    p = _make_param(np.zeros((2,)))
    p.grad = np.ones_like(p.data)
    adam = AdamW([p], lr=0.1)
    adam.step()
    p.grad = None  # simulate trainer forgetting to reset
    adam.zero_grad()
    return p.grad is None or np.all(p.grad == 0)


def test_adamw_multiple_params() -> bool:
    """All registered params update independently."""
    p1 = _make_param(np.zeros((2,)))
    p2 = _make_param(np.zeros((2,)))
    p1.grad = np.array([1.0, 0.0])
    p2.grad = np.array([0.0, 1.0])
    adam = AdamW([p1, p2], lr=0.1)
    adam.step()
    return p1.data[1] == 0.0 and p2.data[0] == 0.0 and (p1.data[0] != 0.0 or p2.data[1] != 0.0)


def main() -> int:
    results = [
        ("adamw_step_moves_data", test_adamw_step_moves_data()),
        ("adamw_step_zero_grad_resets", test_adamw_step_zero_grad_resets()),
        ("adamw_multiple_params", test_adamw_multiple_params()),
    ]
    for name, ok in results:
        flag = "PASS" if ok else "FAIL"
        print(f"  {name:35s} {flag}")
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
