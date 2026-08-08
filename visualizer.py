"""Pygame visualizer: decision boundary (left) + weight graph (right).

ponytail: render-only, no training logic. Caller drives training and
calls update() each frame.
"""
import numpy as np
import pygame
import pygame.gfxdraw


# --- layout constants -------------------------------------------------------

WIDTH, HEIGHT = 1280, 720
TOP_BAR_H = 60
PANEL_W = WIDTH // 2
PANEL_H = HEIGHT - TOP_BAR_H

BG_COLOR = (15, 15, 20)
BAR_COLOR = (25, 25, 32)
TEXT_COLOR = (220, 220, 220)
NODE_OUTLINE = (0, 0, 0)
CONN_NEG = (60, 120, 255)
CONN_POS = (255, 80, 80)
CLASS_COLORS = [(40, 200, 180), (255, 150, 50), (220, 60, 200)]

GRID_RES = 40
NODE_RADIUS = 18
NODE_OUTLINE_W = 2


def _calc_x(layer_idx: int, n_layers: int, pos_x: int, width: int) -> int:
    return pos_x + layer_idx * width // max(n_layers - 1, 1)


def _calc_y(node_idx: int, n_nodes: int, pos_y: int, height: int) -> int:
    if n_nodes == 1:
        return pos_y + height // 2
    return pos_y + node_idx * height // (n_nodes - 1)


def _connection_thickness(weight: float) -> int:
    # ponytail: |w| * 2 per spec, floored at 1 so tiny weights still draw.
    return max(1, min(5, int(abs(weight) * 2)))


class Visualizer:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("neural-network visualizer")
        self.big_font = pygame.font.SysFont("consolas", 18, bold=True)

    def close(self) -> None:
        pygame.quit()

    def update(
        self,
        nn,  # NeuralNetwork — typed loosely to avoid circular import
        X: np.ndarray,
        y_int: np.ndarray,
        epoch: int,
        loss: float,
        acc: float,
        dataset_name: str,
    ) -> None:
        self.screen.fill(BG_COLOR)
        self._draw_top_bar(epoch, loss, acc, dataset_name)
        self._draw_decision_boundary(nn, X, y_int)
        self._draw_weight_graph(nn)
        pygame.display.flip()

    def _draw_top_bar(self, epoch: int, loss: float, acc: float, name: str) -> None:
        pygame.draw.rect(self.screen, BAR_COLOR, (0, 0, WIDTH, TOP_BAR_H))
        text = f"dataset={name}  epoch={epoch}  loss={loss:.3f}  acc={acc:.2f}"
        surf = self.big_font.render(text, True, TEXT_COLOR)
        self.screen.blit(surf, (12, TOP_BAR_H // 2 - surf.get_height() // 2))

    def _draw_decision_boundary(self, nn, X: np.ndarray, y_int: np.ndarray) -> None:
        x_lo, x_hi = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_lo, y_hi = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xs = np.linspace(x_lo, x_hi, GRID_RES)
        ys = np.linspace(y_lo, y_hi, GRID_RES)
        grid = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)
        preds = nn.forward(grid).argmax(axis=1)

        cell_w = PANEL_W / GRID_RES
        cell_h = PANEL_H / GRID_RES
        for i, cls in enumerate(preds):
            gx = i % GRID_RES
            gy = i // GRID_RES
            color = CLASS_COLORS[cls % len(CLASS_COLORS)]
            rect = pygame.Rect(
                int(gx * cell_w),
                TOP_BAR_H + int(gy * cell_h),
                int(cell_w) + 1,
                int(cell_h) + 1,
            )
            pygame.draw.rect(self.screen, color, rect)

        for pt, cls in zip(X, y_int):
            px = int((pt[0] - x_lo) / (x_hi - x_lo) * PANEL_W)
            py = TOP_BAR_H + int((pt[1] - y_lo) / (y_hi - y_lo) * PANEL_H)
            color = CLASS_COLORS[int(cls) % len(CLASS_COLORS)]
            pygame.draw.circle(self.screen, color, (px, py), 4)
            pygame.draw.circle(self.screen, (0, 0, 0), (px, py), 4, 1)

    def _draw_weight_graph(self, nn) -> None:
        pos_x = PANEL_W + 40
        pos_y = TOP_BAR_H + 20
        size_w = PANEL_W - 80
        size_h = PANEL_H - 40
        layers: list[dict] = nn.layers
        n_layers = len(layers) + 1

        for i, layer in enumerate(layers):
            fan_in, fan_out = layer["W"].shape
            x_in = _calc_x(i, n_layers, pos_x, size_w)
            x_out = _calc_x(i + 1, n_layers, pos_x, size_w)
            for k in range(fan_in):
                y_in = _calc_y(k, fan_in, pos_y, size_h)
                for j in range(fan_out):
                    y_out = _calc_y(j, fan_out, pos_y, size_h)
                    w = layer["W"][k, j]
                    color = CONN_POS if w > 0 else CONN_NEG
                    thick = _connection_thickness(w)
                    pygame.draw.line(self.screen, color, (x_in, y_in), (x_out, y_out), thick)
                    # AA blend on top so the edges aren't stair-stepped.
                    pygame.draw.aaline(self.screen, color, (x_in, y_in), (x_out, y_out))

        for i in range(n_layers):
            if i == 0:
                values = nn.input_cache[0]
            else:
                values = layers[i - 1]["a"][0]
            n_nodes = values.shape[0]
            for j in range(n_nodes):
                x = _calc_x(i, n_layers, pos_x, size_w)
                y = _calc_y(j, n_nodes, pos_y, size_h)
                gray = int(np.clip(values[j] * 255, 0, 255))
                fill = (gray, gray, gray)
                pygame.gfxdraw.filled_circle(self.screen, x, y, NODE_RADIUS, fill)
                pygame.gfxdraw.aacircle(self.screen, x, y, NODE_RADIUS, fill)
                pygame.gfxdraw.aacircle(self.screen, x, y, NODE_RADIUS, NODE_OUTLINE)


# --- standalone demo: run on hardcoded XOR ---------------------------------

if __name__ == "__main__":
    from datasets import xor
    from nn import NeuralNetwork, one_hot

    X, y = xor(n=200, seed=0)
    Y = one_hot(y, 2)
    nn = NeuralNetwork([2, 8, 8, 2], ["relu", "relu", "softmax"])

    viz = Visualizer()
    clock = pygame.time.Clock()
    losses = nn.fit(X, Y, epochs=2000, lr=0.05)
    print(f"final loss: {losses[-1]:.3f}")
    try:
        epoch = 2000
        for _ in range(60):
            epoch += 10
            losses = nn.fit(X, Y, epochs=10, lr=0.05)
            acc = float((nn.forward(X).argmax(axis=1) == y).mean())
            viz.update(nn, X, y, epoch, losses[-1], acc, "xor")
            clock.tick(60)
    finally:
        viz.close()