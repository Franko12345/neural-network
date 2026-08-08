"""Toy 2D datasets for the visualizer.

ponytail: fixed-shape outputs (n, 2) and (n,). No train/test split —
visualizer trains on the full set.
"""
from __future__ import annotations

import numpy as np


def _normalize(X: np.ndarray) -> np.ndarray:
    """Per-axis min-max to [-1, 1]."""
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    return 2.0 * (X - lo) / span - 1.0


def _noise(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    return rng.normal(0.0, sigma, (n, 2))


def xor(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    half = n // 2
    # Class 0: (0,0) and (1,1) quadrants. Class 1: (0,1) and (1,0).
    X = rng.uniform(-1, 1, (n, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    X += _noise(rng, n, 0.1)
    return _normalize(X), y


def circle(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    half = n // 2
    # Class 0: inner radius 0.3. Class 1: outer ring radius 0.8.
    angles0 = rng.uniform(0, 2 * np.pi, half)
    angles1 = rng.uniform(0, 2 * np.pi, n - half)
    inner = np.column_stack([np.cos(angles0) * 0.3, np.sin(angles0) * 0.3])
    outer = np.column_stack([np.cos(angles1) * 0.8, np.sin(angles1) * 0.8])
    X = np.vstack([inner, outer]) + _noise(rng, n, 0.1)
    y = np.array([0] * half + [1] * (n - half))
    return _normalize(X), y


def spiral(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """3 intertwined spirals. Each class starts at a different angle offset
    so they interleave cleanly from the origin outward."""
    rng = np.random.default_rng(seed)
    n_per = n // 3
    remainder = n - 3 * n_per
    counts = [n_per, n_per, n_per + remainder]
    Xs, ys = [], []
    for cls, count in enumerate(counts):
        # Each class is a half-turn of a spiral starting at cls * 2π/3.
        t = rng.uniform(0, 2 * np.pi, count)
        r = 0.4 + t / (2 * np.pi) * 0.5  # radius 0.4 -> 0.9
        offset = cls * 2 * np.pi / 3
        x = r * np.cos(t + offset)
        y = r * np.sin(t + offset)
        Xs.append(np.column_stack([x, y]))
        ys.append(np.full(count, cls))
    X = np.vstack(Xs) + _noise(rng, n, 0.05)
    y = np.concatenate(ys)
    return _normalize(X), y


if __name__ == "__main__":
    for name, gen in [("xor", xor), ("circle", circle), ("spiral", spiral)]:
        X, y = gen()
        unique, counts = np.unique(y, return_counts=True)
        bal = dict(zip(unique.tolist(), counts.tolist()))
        print(
            f"{name:7s} shape={X.shape} y_unique={unique.tolist()} "
            f"balance={bal} X_range=[{X.min():.2f},{X.max():.2f}]"
        )
    print("datasets ok")