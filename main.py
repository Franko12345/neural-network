"""Main loop: trains a NeuralNetwork live in the pygame visualizer.

ponytail: glue. Drives training, handles key/mouse input, swaps
datasets. Click hit-testing lives on the visualizer.
"""
import argparse
import sys

import numpy as np
import pygame

from datasets import circle, spiral, xor
from nn import NeuralNetwork, one_hot
from visualizer import Visualizer


DATASETS = {"xor": xor, "circle": circle, "spiral": spiral}
KEY_TO_DATASET = {pygame.K_1: "xor", pygame.K_2: "circle", pygame.K_3: "spiral"}


def make_network(arch: list[int]) -> NeuralNetwork:
    activations = ["relu"] * (len(arch) - 2) + ["softmax"]
    return NeuralNetwork(arch, activations)


def load_dataset(name: str, n_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y = DATASETS[name](n=200, seed=0)
    return X, y, one_hot(y, n_classes)


def parse_arch(s: str) -> list[int]:
    try:
        return [int(x) for x in s.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--arch: {e}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS), default="xor")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--arch", type=parse_arch, default="2,8,8,3")
    args = parser.parse_args()
    lr = min(1.0, max(0.001, args.lr))

    arch = args.arch
    dataset_name = args.dataset
    epochs_per_frame = 10
    paused = False

    X, y, Y = load_dataset(dataset_name, arch[-1])
    nn = make_network(arch)
    epoch = 0

    viz = Visualizer()
    clock = pygame.time.Clock()
    running = True

    def swap_dataset(name: str) -> None:
        nonlocal dataset_name, X, y, Y, nn, epoch
        dataset_name = name
        X, y, Y = load_dataset(name, arch[-1])
        nn = make_network(arch)
        epoch = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    nn = make_network(arch)
                    epoch = 0
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    lr = min(1.0, lr * 1.5)
                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
                    lr = max(0.001, lr / 1.5)
                elif event.key == pygame.K_f:
                    epochs_per_frame = 200 if epochs_per_frame < 100 else 10
                elif event.key == pygame.K_TAB:
                    viz.set_panel("weight_graph" if viz.panel == "boundary" else "boundary")
                elif event.key in KEY_TO_DATASET:
                    swap_dataset(KEY_TO_DATASET[event.key])
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                j = viz.hit_test_input_node(*event.pos)
                if j is not None:
                    X[:, j] = 1.0 - X[:, j]
                    nn.forward(X)  # refresh input_cache so visualizer renders the click
                    paused = True

        if not paused and epochs_per_frame > 0:
            nn.fit(X, Y, epochs=epochs_per_frame, lr=lr)
            epoch += epochs_per_frame

        Y_hat = nn.forward(X)
        acc = float((Y_hat.argmax(axis=1) == y).mean())
        loss = float(-np.mean(np.sum(Y * np.log(Y_hat + 1e-12), axis=1)))
        viz.update_legacy(nn, X, y, epoch, loss, acc, dataset_name)
        clock.tick(60)

    viz.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())