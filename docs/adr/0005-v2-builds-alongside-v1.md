# 0005. v2 builds alongside v1; v1 modules are frozen

Date: 2026-08-08

## Status

Accepted

## Context

v1 (`nn-visualizer`) shipped with `nn.py`, `datasets.py`,
`tests/test_nn.py`, and a v1 visualizer surface. When designing v2
(MountainCar + REINFORCE + transformer), the temptation is to
**refactor `nn.py`** so it supports arbitrary input shapes (3D for
transformer sequences), expose activations as classes, and split
`backward` into per-layer gradients.

Doing that breaks v1: `NeuralNetwork.forward` is called by v1
visualizer and v1 tests with a 2D `(N, fan_in)` input; the refactored
signature would ripple through 7 files. v1 visualizer depends on
`forward(X)` populating `layer["a"]` for the weight graph — a hidden
behavioral contract, not part of the type signature.

## Decision

v2 builds **alongside** v1, not on top of it:

- v1 files (`nn.py`, `datasets.py`, `tests/test_nn.py`, the v1
  visualizer surface) are **frozen**. They are not modified by v2
  work. v1 tests keep passing unchanged.
- v2 introduces `layers.py` (Linear, ReLU, Tanh, Sigmoid, Softmax,
  LayerNorm, Embedding) — the math primitives as standalone modules
  with their own `forward`/`backward` and seam tests.
- v2 visualizer is built on the **refactored** `Visualizer.update()`
  (ticket 08) which accepts a `metrics: dict` instead of fixed
  kwargs. The v1 visual surface is preserved by an adapter in the
  v1 `__main__` block.

## Consequences

**Easier:**
- v1 keeps shipping; v2 work can't regress it.
- Each module in `layers.py` is testable in isolation with numerical
  gradient checks against a known-correct reference.
- v1 visualizer stays simple (no metrics dict); v2 visualizer gets
  the flexibility without inheriting v1's flags.

**Harder:**
- Some duplication between v1 `NeuralNetwork._activate` and v2
  `layers.Softmax`. Both exist; both are tested. The duplication is
  the cost of frozen-v1.
- `Visualizer.update()` signature changes in ticket 08. v1 callers
  must be updated in the same commit (small adapter).

**When to revisit:**
- When v1 is no longer a target (e.g. v3 ships and v1 is just an
  example). Then `nn.py` and `layers.py` can be unified.