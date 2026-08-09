"""REINFORCE trainer with constant baseline.

ponytail: caller owns the policy modules (Linear stacks + ReLU/Softmax
in between) and the policy_fn passed to rollout. Trainer re-runs the
forward pass to compute gradients — log_probs from rollout are only
used for masking sanity, not the gradient itself.

Gradient scaling trick: Linear.backward applies W -= dW (lr=1.0). To
get a real SGD step with learning rate `lr`, we just scale the
upstream gradient by lr before backward. No post-hoc correction.
"""
import numpy as np

from layers import Linear, ReLU, Softmax


def _discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Compute G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}, per timestep."""
    T = len(rewards)
    G = np.zeros(T)
    cum = 0.0
    for t in reversed(range(T)):
        cum = rewards[t] + gamma * cum
        G[t] = cum
    return G


def train(policy_layers: list, batch, lr: float = 0.01, gamma: float = 0.99):
    """REINFORCE with constant baseline.

    policy_layers: [Linear(in, h), Linear(h, out)]. Trainer adds ReLU +
        Softmax between them in the internal forward.
    batch: RolloutBatch with episode boundaries.
    Returns: (policy_layers, mean_episode_reward).
    """
    assert len(policy_layers) == 2, "expected [Linear(in, h), Linear(h, out)]"
    lin1, lin2 = policy_layers
    relu = ReLU()
    softmax = Softmax(axis=-1)
    states = batch.states
    actions = batch.actions
    T = len(states)

    # Per-episode G_t (avoids bleeding across episode boundaries).
    G = np.zeros(T)
    for start, end in zip(batch.episode_starts, batch.episode_ends):
        G[start:end] = _discounted_returns(batch.rewards[start:end], gamma)

    # Constant baseline = mean(G); reduces variance, unbiased.
    advantages = G - G.mean()

    # Re-run forward to get gradients.
    h = relu.forward(lin1.forward(states))
    logits = lin2.forward(h)
    probs = softmax.forward(logits)

    # Fused softmax+CE: dL/d_logits = (probs - one_hot(a)) * advantage / T
    N = T
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(T), actions] = 1.0
    d_logits = (probs - one_hot) * advantages.reshape(-1, 1) / N

    # Scale upstream gradient by lr so Linear.backward (which uses lr=1.0)
    # effectively applies the requested lr.
    d_logits *= lr

    # Backprop through lin2 → ReLU → lin1.
    grad_h = lin2.backward(d_logits)
    d_pre_relu = relu.backward(grad_h)
    lin1.backward(d_pre_relu)

    # Mean episode reward (caller metric).
    ep_rewards = [
        float(batch.rewards[start:end].sum())
        for start, end in zip(batch.episode_starts, batch.episode_ends)
    ]
    mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0

    return policy_layers, mean_reward
