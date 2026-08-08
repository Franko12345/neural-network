# 0002. Numpy-only, no PyTorch / TF

Date: 2026-08-08

## Status

Accepted

## Context

The project visualizes a neural network to **teach** how it works.
A working framework (PyTorch, TensorFlow) hides the gradient, the
weight initialization, the loss curve. The whole point is to see those.

## Decision

Core math lives in **`nn.py`** using only `numpy`. No PyTorch, no TF,
no JAX, no autograd library. Activations, loss, forward, backward are
all explicit numpy code.

The pygame visualizer (`visualizer.py`) reads numpy arrays directly.

## Consequences

**Easier:**
- Each component is ~30 lines and reads top-to-bottom.
- Forces a clean backprop implementation that mirrors the 3Blue1Brown
  chapter-4 math.

**Harder:**
- No autograd → manual gradient bookkeeping. Bug-prone; covered by
  `tests/test_nn.py`.
- No GPU acceleration. Fine: datasets are tiny (≤200 samples).

**When to revisit:**
- If the project grows to real-scale data (>10k samples) or needs
  modern architectures (transformers, convnets), swap the core for
  PyTorch and keep the visualizer.