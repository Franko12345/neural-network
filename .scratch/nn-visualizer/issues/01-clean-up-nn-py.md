# 01 — Clean up `nn.py` to match the v1 spec

**What to build:** A working `NeuralNetwork` class that matches the v1
spec end-to-end. From the user's perspective: `import numpy as np;
from nn import NeuralNetwork, one_hot; nn = NeuralNetwork([2, 8, 8, 3],
["relu", "relu", "softmax"]); losses = nn.fit(X, Y, epochs=500,
lr=0.05)` returns a list of decreasing losses. Demonstrated by a
sanity script that exits 0 in under 2 seconds on the headless LXC.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] `NeuralNetwork.__init__(layer_sizes, activations)` constructs
      `n_layers - 1` layers with Xavier-init weights, zero biases, and
      each layer caches `z`, `a`, `dz` for backprop.
- [ ] `NeuralNetwork.forward(x)` returns the network's output and
      populates each layer's cache.
- [ ] `NeuralNetwork.backward(y_true, y_pred, lr)` computes the
      gradient via the fused softmax+cross-entropy derivative
      (`δ = ŷ - y`) at the output layer, propagates backward through
      ReLU/sigmoid/tanh via `_activate_deriv`, updates all weights
      and biases with `lr * gradient`.
- [ ] `NeuralNetwork.fit(X, Y, epochs, lr, log_every=0)` runs the
      forward/backward loop, returns a `list[float]` of per-epoch
      cross-entropy losses.
- [ ] `one_hot(y, n_classes)` helper exists.
- [ ] Reproducible: `np.random.default_rng(42)` seeded once in
      `__init__`.
- [ ] A small sanity script (in-repo, not in tests/) runs a tiny net
      on a synthetic dataset, prints loss going down, exits 0 in
      under 2 seconds.
- [ ] The draft's redundant `inp = ...` lines (from the earlier
      rewrite) are removed; the file reads top-to-bottom in under
      100 lines.
- [ ] A `ponytail:` comment names the full-batch SGD ceiling and
      points to minibatch as the upgrade path.