"""Compositional modules for v2.

ponytail: Residual is a thin wrapper — splits a function into f(x) + x
with the right gradient. No flags.
"""
import numpy as np


class Residual:
    """forward(x) = fn(x) + x. backward(grad) returns grad + grad @ fn'(x).

    ponytail: assumes fn.forward/backward follow the layers convention.
    fn must accept (x,) and return y; backward accepts dL/dy and returns
    dL/dx into the inner fn. Residual splits grad back to both paths.
    """

    def __init__(self, fn):
        self.fn = fn
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.fx = self.fn.forward(x)
        return self.fx + x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # grad from loss comes in as dL/d(out); out = fx + x.
        # dL/d(fx) = grad (same); dL/dx via fn = grad_fn(grad)
        # identity path adds +grad; total dL/dx = grad + fn_grad.
        fn_grad = self.fn.backward(grad)
        return grad + fn_grad
