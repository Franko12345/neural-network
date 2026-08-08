"""Sanity check: a tiny net on synthetic XOR-ish labels must decrease loss.

ponytail: standalone, exits 0 on pass. Lives outside nn.py to keep the
core module under ticket 01's line ceiling.
"""
import numpy as np

from nn import NeuralNetwork, one_hot

rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, (40, 2))
Y = one_hot((X[:, 0] * X[:, 1] > 0).astype(int), 2)
nn = NeuralNetwork([2, 8, 8, 2], ["relu", "relu", "softmax"])
losses = nn.fit(X, Y, epochs=500, lr=0.05)
assert losses[-1] < losses[0], f"loss not decreasing: {losses[0]:.4f} -> {losses[-1]:.4f}"
acc = (nn.forward(X).argmax(axis=1) == Y.argmax(axis=1)).mean()
print(f"sanity ok: loss {losses[0]:.3f}->{losses[-1]:.3f}, acc={acc:.2f}")