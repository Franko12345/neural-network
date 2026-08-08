"""Main loop: trains a NeuralNetwork live in the pygame visualizer.

ponytail: glue. Drives training, handles key/mouse input, swaps
datasets. Coordinate math for input-node clicks mirrors the visualizer.
"""
import argparse
import sys

import numpy as np
import pygame

from datasets import circle, spiral, xor
from nn import NeuralNetwork, one_hot
from visualizer import Visualizer


DATASETS = {"xor": xor, "circle": circle, "spiral": spiral}

# Right-panel layout (mirrors visualizer constants; visualizer.py owns the
# authoritative versions — if you change them there, update here too).
PANEL_RIGHT_X = 1280 // 2 + 40
GRAPH_W = 1280 // 2 - 80
TOP_BAR_H = 60
GRAPH_PAD_Y = 20


def input_node_pos(node_idx: int, n_total: int) -> tuple[int, int]:
    """Mirror visualizer._calc_x/_calc_y for layer 0 (input column)."""
    inset = int(GRAPH_W * 0.12)
    pos_x = PANEL_RIGHT_X + inset
    pos_y = TOP_BAR_H + GRAPH_PAD_Y
    size_h = (720 - TOP_BAR_H) - 2 * GRAPH_PAD_Y
    if n_total <= 1:
        return pos_x, pos_y + size_h // 2
    dy = size_h // (n_total - 1)
    total_used = (n_total - 1) * dy
    y = pos_y + (size_h - total_used) // 2 + node_idx * dy
    return pos_x, y


def make_network(arch: list[int]) -> NeuralNetwork:
    activations = ["relu"] * (len(arch) - 2) + ["softmax"]
    return NeuralNetwork(arch, activations)


def load_dataset(name: str, n_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y = DATASETS[name](n=200, seed=0)
    return X, y, one_hot(y, n_classes)


def parse_arch(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS), default="xor")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--arch", type=str, default="2,8,8,3")
    args = parser.parse_args()

    arch = parse_arch(args.arch)
    dataset_name = args.dataset
    lr = args.lr
    epochs_per_frame = 10
    paused = False

    X, y, Y = load_dataset(dataset_name, arch[-1])
    nn = make_network(arch)
    epoch = 0
    n_total = max(arch)

    viz = Visualizer()
    clock = pygame.time.Clock()
    running = True

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
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    lr = min(1.0, lr * 1.5)
                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    lr = max(0.001, lr / 1.5)
                elif event.key == pygame.K_f:
                    epochs_per_frame = 200 if epochs_per_frame < 100 else 10
                elif event.key == pygame.K_1:
                    dataset_name = "xor"
                    X, y, Y = load_dataset(dataset_name, arch[-1])
                    nn = make_network(arch)
                    epoch = 0
                elif event.key == pygame.K_2:
                    dataset_name = "circle"
                    X, y, Y = load_dataset(dataset_name, arch[-1])
                    nn = make_network(arch)
                    epoch = 0
                elif event.key == pygame.K_3:
                    dataset_name = "spiral"
                    X, y, Y = load_dataset(dataset_name, arch[-1])
                    nn = make_network(arch)
                    epoch = 0
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                for j in range(arch[0]):
                    nx, ny = input_node_pos(j, n_total)
                    if (mx - nx) ** 2 + (my - ny) ** 2 <= 18 ** 2:
                        X[:, j] = 1.0 - X[:, j]
                        paused = True
                        break

        if not paused and epochs_per_frame > 0:
            nn.fit(X, Y, epochs=epochs_per_frame, lr=lr)
            epoch += epochs_per_frame

        acc = float((nn.forward(X).argmax(axis=1) == y).mean())
        loss = float(-np.mean(np.sum(Y * np.log(nn.forward(X) + 1e-12), axis=1)))
        viz.update(nn, X, y, epoch, loss, acc, dataset_name)
        clock.tick(60)

    viz.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())