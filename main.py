"""Main loop with task registry.

ponytail: glue. Each task (xor/circle/spiral/mountaincar/transformer)
exposes reset(), step(), render(viz, metrics), metrics(). The main
loop dispatches the active task; the registry is a flat dict.

v1 surface (XorTask / CircleTask / SpiralTask) keeps the v1 key
bindings (1/2/3) and v1 behavior. v2 tasks add 4 (MountainCar) and
5 (Transformer). Tab cycles the v1 panel; in v2 panels, the click
inspector is task-specific (handled inside each task's render()).
"""
import argparse
import sys
from typing import Protocol

import numpy as np
import pygame

from datasets import circle, spiral, xor
from nn import NeuralNetwork, one_hot
from visualizer import Visualizer


KEY_TO_TASK = {
    pygame.K_1: "xor",
    pygame.K_2: "circle",
    pygame.K_3: "spiral",
    pygame.K_4: "mountaincar",
    pygame.K_5: "transformer",
}


class Task(Protocol):
    """Protocol every task implements."""

    def reset(self) -> Visualizer: ...
    def step(self) -> None: ...
    def render(self, viz: Visualizer, metrics: dict) -> None: ...
    def metrics(self) -> dict: ...


def parse_arch(s: str) -> list[int]:
    try:
        return [int(x) for x in s.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--arch: {e}")


# ----- v1 tasks: wrap the existing v1 main loop logic -------------------

def _make_network(arch: list[int]) -> NeuralNetwork:
    activations = ["relu"] * (len(arch) - 2) + ["softmax"]
    return NeuralNetwork(arch, activations)


class V1DatasetTask:
    """Base for xor/circle/spiral: trains NeuralNetwork on a 2D dataset."""

    DATASET_NAME: str = ""  # overridden

    def __init__(self, arch: list[int]):
        self.arch = arch
        self.X, self.y, self.Y = self._load()
        self.nn = _make_network(arch)
        self.epoch = 0
        self.lr = 0.05
        self.paused = False

    def _load(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fn = {"xor": xor, "circle": circle, "spiral": spiral}[self.DATASET_NAME]
        X, y = fn(n=200, seed=0)
        return X, y, one_hot(y, self.arch[-1])

    def reset(self) -> Visualizer:
        return Visualizer()

    def step(self) -> None:
        if not self.paused:
            self.nn.fit(self.X, self.Y, epochs=10, lr=self.lr)
            self.epoch += 10

    def render(self, viz: Visualizer, metrics: dict) -> None:
        Y_hat = self.nn.forward(self.X)
        acc = float((Y_hat.argmax(axis=1) == self.y).mean())
        loss = float(-np.mean(np.sum(self.Y * np.log(Y_hat + 1e-12), axis=1)))
        viz.update_legacy(self.nn, self.X, self.y, self.epoch, loss, acc, self.DATASET_NAME)

    def metrics(self) -> dict:
        return {"epoch": self.epoch, "dataset": self.DATASET_NAME}

    def handle_key(self, key: int) -> bool:
        """Returns True if key was handled. Used by main loop."""
        if key == pygame.K_SPACE:
            self.paused = not self.paused
            return True
        if key in (pygame.K_r,):
            self.nn = _make_network(self.arch)
            self.epoch = 0
            return True
        if key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.lr = min(1.0, self.lr * 1.5)
            return True
        if key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
            self.lr = max(0.001, self.lr / 1.5)
            return True
        if key == pygame.K_f:
            # Toggle epochs_per_frame (handled outside; here just toggle state)
            return True
        return False

    def handle_click(self, pos: tuple, viz: Visualizer) -> None:
        j = viz.hit_test_input_node(*pos)
        if j is not None:
            self.X[:, j] = 1.0 - self.X[:, j]
            self.nn.forward(self.X)
            self.paused = True


class XorTask(V1DatasetTask):
    DATASET_NAME = "xor"


class CircleTask(V1DatasetTask):
    DATASET_NAME = "circle"


class SpiralTask(V1DatasetTask):
    DATASET_NAME = "spiral"


# ----- v2 tasks: MountainCar + Transformer ----------------------------

class MountainCarTask:
    """MountainCar with REINFORCE (constant baseline)."""

    def __init__(self, n_episodes: int = 5, max_steps: int = 200, seed: int = 0):
        from envs.mountaincar import MountainCarEnv
        from envs.rollout import rollout
        from train_rl import train
        from layers import Linear, ReLU, Softmax

        self.env = MountainCarEnv()
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.seed = seed
        # Policy: [2, 16, 3] per ticket 03
        rng = np.random.default_rng(seed)
        self.lin1 = Linear(2, 16)
        self.lin2 = Linear(16, 3)
        self.lin1.W = rng.standard_normal(self.lin1.W.shape) * 0.3
        self.lin2.W = rng.standard_normal(self.lin2.W.shape) * 0.3
        self._relu = ReLU()
        self._softmax = Softmax(axis=-1)
        self._step_count = 0

    def _policy_fn(self, state):
        # state may be 1D (env) or 2D (rollout). Reshape to 2D for forward.
        s = np.atleast_2d(state)
        h = self._relu.forward(self.lin1.forward(s))
        logits = self.lin2.forward(h)
        probs = self._softmax.forward(logits)
        a = int(np.random.choice(probs.shape[-1], p=probs[0]))
        log_prob = float(np.log(max(probs[0, a], 1e-12)))
        return a, log_prob

    def reset(self) -> Visualizer:
        return Visualizer()

    def step(self) -> None:
        from envs.rollout import rollout
        from train_rl import train
        np.random.seed(self.seed + self._step_count)
        batch = rollout(self.env, self._policy_fn,
                        n_episodes=self.n_episodes,
                        max_steps=self.max_steps,
                        seed=self.seed + self._step_count)
        train([self.lin1, self.lin2], batch, lr=0.01, gamma=0.99)
        self._step_count += 1

    def render(self, viz: Visualizer, metrics: dict) -> None:
        # MountainCarEnv.reset returns obs only (per ticket 02 fix);
        # step returns (next_obs, reward, done, info).
        obs = self.env.reset(seed=self.seed + self._step_count)
        frame = self.env.render()
        for _ in range(20):
            a, _ = self._policy_fn(obs)
            obs, _, done, _ = self.env.step(a)
            if done:
                break
        viz.update(metrics={"frame": frame, "epoch": self._step_count,
                            "panel": "gym_render"}, panel="gym_render")

    def metrics(self) -> dict:
        return {"epoch": self._step_count}


class TransformerTask:
    """Decoder-only transformer with bundled Shakespeare text."""

    def __init__(self, vocab: int = 256, d_model: int = 32, n_heads: int = 2,
                 n_layers: int = 2, d_ff: int = 64, max_seq_len: int = 32,
                 seed: int = 0):
        from transformer.model import Transformer
        from transformer.train import Trainer
        from data.text import load_text, DEFAULT_PATH

        self.model = Transformer(vocab=vocab, d_model=d_model, n_heads=n_heads,
                                 n_layers=n_layers, d_ff=d_ff,
                                 max_seq_len=max_seq_len, seed=seed)
        data, _ = load_text(DEFAULT_PATH)
        self.trainer = Trainer(self.model, data=data, lr=3e-4, seed=seed)
        self._step_count = 0
        self._last_attn = None

    def reset(self) -> Visualizer:
        return Visualizer()

    def step(self) -> None:
        self.trainer.train(n_steps=1)
        self._step_count += 1
        # Capture attention weights from the most recent forward by
        # re-running forward and reading attn_weights off the MHA layers.
        # The first block's MHA exposes attn_weights after forward.
        from data.text import DEFAULT_PATH
        from data.text import load_text
        # We need fresh attn — run a tiny forward.
        ids = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=np.uint8)
        self.model.forward(ids)
        # Read first block's first head attn weights
        self._last_attn = self.model.blocks[0].mha.attn_weights[0]  # (B, T, T)

    def render(self, viz: Visualizer, metrics: dict) -> None:
        attn = self._last_attn
        if attn is None:
            # Fall back to random attention weights if no forward has run yet
            T = self.model.pos_enc.max_seq_len
            attn = np.full((self.model.blocks[0].mha.n_heads, T, T), 1.0 / T)
        viz.update(metrics={"attention_weights": attn, "epoch": self._step_count,
                            "panel": "attention_heatmap"}, panel="attention_heatmap")

    def metrics(self) -> dict:
        return {"epoch": self._step_count}


TASKS = {
    "xor": XorTask([2, 8, 8, 3]),
    "circle": CircleTask([2, 8, 8, 2]),
    "spiral": SpiralTask([2, 16, 16, 3]),
    "mountaincar": MountainCarTask(),
    "transformer": TransformerTask(),
}


# ----- main loop ---------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(TASKS), default="xor")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--arch", type=parse_arch, default="2,8,8,3")
    args = parser.parse_args()

    active_name = args.task
    task = TASKS[active_name]
    viz = task.reset()
    clock = pygame.time.Clock()
    running = True

    def switch_task(name: str) -> None:
        nonlocal active_name, task, viz
        active_name = name
        task = TASKS[name]
        viz.close()
        viz = task.reset()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_TAB and isinstance(task, V1DatasetTask):
                    viz.set_panel("weight_graph" if viz.panel == "boundary"
                                  else "boundary")
                elif event.key in KEY_TO_TASK:
                    switch_task(KEY_TO_TASK[event.key])
                else:
                    if isinstance(task, V1DatasetTask):
                        task.handle_key(event.key)
            elif (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                  and isinstance(task, V1DatasetTask)):
                task.handle_click(event.pos, viz)

        task.step()
        metrics = task.metrics()
        task.render(viz, metrics)
        clock.tick(60)

    viz.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
