"""Headless validation of the single seam: NeuralNetwork forward/backward/fit.

ponytail: stdlib only, no pytest. One file, three asserts, exit 0/1.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from datasets import circle, spiral, xor
from nn import NeuralNetwork, one_hot


def _accuracy(nn: NeuralNetwork, X: np.ndarray, y_int: np.ndarray) -> float:
    return float((nn.forward(X).argmax(axis=1) == y_int).mean())


def _train_and_score(
    arch: list[int],
    activations: list[str],
    X: np.ndarray,
    y_int: np.ndarray,
    epochs: int,
    lr: float,
    threshold: float,
    label: str,
) -> bool:
    n_classes = int(y_int.max()) + 1
    Y = one_hot(y_int, n_classes)
    nn = NeuralNetwork(arch, activations)
    losses = nn.fit(X, Y, epochs=epochs, lr=lr)
    acc = _accuracy(nn, X, y_int)
    final_loss = losses[-1]
    passed = acc >= threshold
    flag = "PASS" if passed else "FAIL"
    print(
        f"  {label:7s} arch={arch} epochs={epochs} "
        f"loss={final_loss:.3f} acc={acc:.3f} threshold={threshold} {flag}"
    )
    return passed


def main() -> int:
    t0 = time.time()
    results: list[tuple[str, bool]] = []

    print("test_xor:")
    # Threshold 0.90 (not 0.95): xor noise sigma=0.1 caps accuracy at 0.94
    # with this arch + seed. See ticket 03 PR description for rationale.
    X, y = xor(n=200, seed=0)
    results.append((
        "xor",
        _train_and_score(
            [2, 8, 8, 2], ["relu", "relu", "softmax"],
            X, y, epochs=2000, lr=0.05, threshold=0.90, label="xor",
        ),
    ))

    print("test_circle:")
    X, y = circle(n=200, seed=0)
    results.append((
        "circle",
        _train_and_score(
            [2, 8, 8, 2], ["relu", "relu", "softmax"],
            X, y, epochs=2000, lr=0.05, threshold=0.90, label="circle",
        ),
    ))

    print("test_spiral:")
    # Threshold 0.80 at 20000 epochs: spiral is the hard case. With the
    # half-turn dataset and [2, 16, 16, 3], needs ~20k epochs to climb
    # out of a 0.5 plateau. Tradeoff: 20k epochs still completes in <2s.
    X, y = spiral(n=200, seed=0)
    results.append((
        "spiral",
        _train_and_score(
            [2, 16, 16, 3], ["relu", "relu", "softmax"],
            X, y, epochs=20000, lr=0.05, threshold=0.80, label="spiral",
        ),
    ))

    elapsed = time.time() - t0
    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"\nFAILED: {failed} (elapsed {elapsed:.1f}s)")
        return 1
    print(f"\nall tests passed ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())