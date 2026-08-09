"""Headless validation of train_rl.REINFORCE.

ponytail: stdlib only, no pytest.
"""
from __future__ import annotations

import sys

import numpy as np

from layers import Linear, ReLU, Softmax
from envs.rollout import RolloutBatch
from train_rl import train


def _make_policy(rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    lin1 = Linear(2, 16)
    lin2 = Linear(16, 3)
    lin1.W = rng.standard_normal(lin1.W.shape) * 0.3
    lin2.W = rng.standard_normal(lin2.W.shape) * 0.3
    return lin1, lin2


def _policy_fn_factory(lin1, lin2):
    """Returns policy(state) -> (action, log_prob). Accepts 1D or 2D state."""
    relu = ReLU()
    softmax = Softmax(axis=-1)
    def policy(state):
        s = state if state.ndim == 2 else state.reshape(1, -1)
        h = relu.forward(lin1.forward(s))
        logits = lin2.forward(h)
        probs = softmax.forward(logits)
        a = int(np.random.choice(probs.shape[-1], p=probs[0]))
        log_prob = float(np.log(max(probs[0, a], 1e-12)))
        return a, log_prob
    return policy


def test_train_returns_policy_and_running_mean() -> bool:
    lin1, lin2 = _make_policy(0)
    T = 50
    states = np.random.default_rng(0).standard_normal((T, 2)) * 0.5
    actions = np.random.default_rng(1).integers(0, 3, T)
    rewards = np.full(T, -1.0)
    log_probs = np.full(T, np.log(1/3))
    batch = RolloutBatch(
        states=states, actions=actions, rewards=rewards, log_probs=log_probs,
        episode_starts=[0], episode_ends=[T],
    )
    policy_layers, mean_reward = train([lin1, lin2], batch, lr=0.01, gamma=0.99)
    return (
        len(policy_layers) == 2
        and policy_layers[0].W.shape == lin1.W.shape
        and policy_layers[1].W.shape == lin2.W.shape
        and isinstance(mean_reward, float)
    )


def test_train_gradient_numerical() -> bool:
    """FD vs analytical on REINFORCE loss w.r.t. Linear weights.

    Loss = -mean(log_pi * advantage) with advantage = G - baseline.
    We use a 2-step batch so advantages are non-zero; FD uses the
    identical loss expression so we compare apples to apples.
    """
    rng = np.random.default_rng(0)
    lin1 = Linear(2, 4)
    lin2 = Linear(4, 3)
    lin1.W = rng.standard_normal(lin1.W.shape) * 0.3
    lin2.W = rng.standard_normal(lin2.W.shape) * 0.3

    state = np.array([[0.3, -0.5], [0.4, -0.2]])
    actions = np.array([1, 1])
    rewards = np.array([2.0, -1.0])
    G = rewards  # gamma=0 → G = rewards
    baseline = G.mean()
    advantages = G - baseline

    # Loss = -mean(log_pi * advantage), exactly matching the trainer.
    def loss(lin1_w, lin2_w):
        l1 = Linear(2, 4); l1.W = lin1_w.copy()
        l2 = Linear(4, 3); l2.W = lin2_w.copy()
        relu = ReLU(); softmax = Softmax(axis=-1)
        h = relu.forward(l1.forward(state))
        logits = l2.forward(h)
        probs = softmax.forward(logits)
        log_pi = np.log(np.maximum(probs[np.arange(2), actions], 1e-12))
        return -float(np.mean(log_pi * advantages))

    eps = 1e-5
    grad_w1_num = np.zeros_like(lin1.W)
    it = np.nditer(lin1.W, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        wp, wm = lin1.W.copy(), lin1.W.copy()
        wp[idx] += eps
        wm[idx] -= eps
        grad_w1_num[idx] = (loss(wp, lin2.W) - loss(wm, lin2.W)) / (2 * eps)
        it.iternext()
    grad_w2_num = np.zeros_like(lin2.W)
    it = np.nditer(lin2.W, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        wp, wm = lin2.W.copy(), lin2.W.copy()
        wp[idx] += eps
        wm[idx] -= eps
        grad_w2_num[idx] = (loss(lin1.W, wp) - loss(lin1.W, wm)) / (2 * eps)
        it.iternext()

    batch = RolloutBatch(
        states=state, actions=actions, rewards=rewards,
        log_probs=np.zeros(2),
        episode_starts=[0], episode_ends=[2],
    )
    w1_before = lin1.W.copy()
    w2_before = lin2.W.copy()
    train([lin1, lin2], batch, lr=1.0, gamma=0.0)
    # lr=1.0 → W_after = W_before - dW → dW = W_before - W_after.
    grad_w1_an = w1_before - lin1.W
    grad_w2_an = w2_before - lin2.W

    return (
        np.allclose(grad_w1_an, grad_w1_num, atol=1e-5)
        and np.allclose(grad_w2_an, grad_w2_num, atol=1e-5)
    )


def test_train_baseline_subtraction_invariant() -> bool:
    """With baseline b=0, gradient equals mean-centered targets gradient
    (the constant shift cancels)."""
    lin1_a, lin2_a = _make_policy(42)
    lin1_b, lin2_b = _make_policy(42)

    T = 2
    state = np.array([[0.3, -0.5], [0.4, -0.2]])
    # batch_a: rewards [2, 0] → G=[2, 0] → mean=1 → advantages=[1, -1]
    # batch_b: rewards [3, 1] → G=[3, 1] → mean=2 → advantages=[1, -1]
    # So gradients must be identical (baseline subtraction makes them
    # mean-invariant).
    batch_a = RolloutBatch(
        states=state, actions=np.array([1, 2]), rewards=np.array([2.0, 0.0]),
        log_probs=np.zeros(T),
        episode_starts=[0], episode_ends=[T],
    )
    batch_b = RolloutBatch(
        states=state, actions=np.array([1, 2]), rewards=np.array([3.0, 1.0]),
        log_probs=np.zeros(T),
        episode_starts=[0], episode_ends=[T],
    )
    train([lin1_a, lin2_a], batch_a, lr=1.0, gamma=0.0)
    train([lin1_b, lin2_b], batch_b, lr=1.0, gamma=0.0)

    return (
        np.allclose(lin1_a.W, lin1_b.W, atol=1e-6)
        and np.allclose(lin2_a.W, lin2_b.W, atol=1e-6)
    )


def test_train_50_episodes_smoke() -> bool:
    """Headless: train for 50 episodes on MountainCar, exit 0."""
    from envs.mountaincar import MountainCarEnv
    from envs.rollout import rollout

    lin1, lin2 = _make_policy(0)
    policy = _policy_fn_factory(lin1, lin2)
    env = MountainCarEnv()
    try:
        np.random.seed(0)
        batch = rollout(env, policy, n_episodes=50, max_steps=200, seed=0)
        _, mean_reward = train([lin1, lin2], batch, lr=0.01, gamma=0.99)
        return mean_reward < 0  # MountainCar reward always negative until goal
    finally:
        env.close()


def main() -> int:
    results = [
        ("train_returns_policy_and_running_mean", test_train_returns_policy_and_running_mean()),
        ("train_gradient_numerical", test_train_gradient_numerical()),
        ("train_baseline_subtraction_invariant", test_train_baseline_subtraction_invariant()),
        ("train_50_episodes_smoke", test_train_50_episodes_smoke()),
    ]
    for name, ok in results:
        flag = "PASS" if ok else "FAIL"
        print(f"  {name:42s} {flag}")
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
