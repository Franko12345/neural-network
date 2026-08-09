"""AdamW optimizer — Adam with decoupled weight decay.

ponytail: each param is a duck-typed object with .data (np.ndarray) and
.grad (np.ndarray or None). step() updates .data in-place; zero_grad()
clears .grad to None. bias correction follows Loshchilov & Hutter.
"""
import numpy as np


class AdamW:
    """AdamW. params: list of objects with .data, .grad. Bias-corrected."""

    def __init__(
        self,
        params: list,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
        eps: float = 1e-8,
    ):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.wd = weight_decay
        self.eps = eps
        self.t = 0
        # State per param
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            # Decoupled weight decay (applied directly to data, not via grad).
            if self.wd != 0.0:
                p.data -= self.lr * self.wd * p.data
            # Adam moments
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)
            # Bias correction
            m_hat = self.m[i] / (1.0 - self.b1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None
