# 01 — Extract `layers.py` building blocks

**What to build:** A new module `layers.py` with `Linear`, `ReLU`,
`Tanh`, `Sigmoid`, `Softmax` modules — each with `forward(X)` and
`backward(grad)` methods. From the user's perspective: `from layers
import Linear; layer = Linear(2, 8); out = layer.forward(X);
layer.backward(grad)` works standalone, with no dependency on v1's
`NeuralNetwork`. v1 stays frozen.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [x] `layers.py` module exists at repo root, no imports from `nn.py`
- [x] `Linear(fan_in, fan_out)` with Xavier init (`sqrt(2/(fan_in+fan_out))`),
      `forward(X)` returns `X @ W + b`, `backward(grad)` returns input
      gradient and updates `W`/`b` in place
- [x] `ReLU()`, `Tanh()`, `Sigmoid()`, `Softmax(axis=-1)` — each with
      `forward(X)` and `backward(grad)` matching the v1 activation
      derivative formulas
- [x] `tests/test_layers.py` exists with:
      - Linear: numerical gradient check (finite differences vs
        analytical, small input)
      - Softmax+CE fused derivative verified (same formula as v1:
        `δ = ŷ - y`)
      - Each activation: forward shape == input shape, backward
        shape == forward shape
- [x] Headless smoke: `python3 -m tests.test_layers` exits 0 in <5s
- [x] v1 tests still pass: `python3 -m tests.test_nn` exits 0 (v1
      untouched)
- [x] ponytail: no flag on `forward()`; no `if isinstance` shape
      dispatch; one method per module