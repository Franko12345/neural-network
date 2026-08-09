"""Headless smoke for the refactored visualizer.

ponytail: the visualizer is render-only and pygame-bound. We can't
unit-test pixels. We test the new API surface: update(metrics, panel)
exists, set_panel(name) switches the right-panel mode, and the
legacy adapter still drives frames end-to-end under dummy SDL.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402  (used in v1 adapter test)
import pygame  # noqa: E402  (env must be set before init)

from visualizer import Visualizer  # noqa: E402


def test_update_accepts_metrics_dict() -> bool:
    viz = Visualizer()
    try:
        metrics = {"epoch": 100, "loss": 0.3, "acc": 0.9, "dataset": "xor"}
        viz.update(metrics=metrics, panel="boundary")
        return True
    finally:
        viz.close()


def test_set_panel_switches_mode() -> bool:
    viz = Visualizer()
    try:
        viz.set_panel("weight_graph")
        return viz.panel == "weight_graph"
    finally:
        viz.close()


def test_v1_adapter_smoke() -> bool:
    """Legacy kwargs path must still drive 10 frames under dummy SDL."""
    from datasets import xor
    from nn import NeuralNetwork, one_hot

    viz = Visualizer()
    try:
        X, y = xor(n=50, seed=0)
        Y = one_hot(y, 2)
        nn = NeuralNetwork([2, 8, 8, 2], ["relu", "relu", "softmax"])
        # Adapter signature: same old kwargs, internally builds metrics dict.
        for _ in range(10):
            nn.fit(X, Y, epochs=5, lr=0.05)
            Y_hat = nn.forward(X)
            acc = float((Y_hat.argmax(axis=1) == y).mean())
            loss = float(-(((Y * np.log(Y_hat + 1e-12))).sum(axis=1)).mean())
            viz.update_legacy(nn, X, y, epoch=50, loss=loss, acc=acc, dataset_name="xor")
        return True
    finally:
        viz.close()


def main() -> int:
    results = [
        ("update_accepts_metrics_dict", test_update_accepts_metrics_dict()),
        ("set_panel_switches_mode", test_set_panel_switches_mode()),
        ("v1_adapter_smoke", test_v1_adapter_smoke()),
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
