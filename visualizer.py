"""Pygame visualizer: decision boundary (left) + weight graph (right).

ponytail: render-only, no training logic. Caller drives training and
calls update() each frame. v2 refactor: update() takes a metrics dict
and a panel selector; set_panel(name) switches the right-panel mode.
The v1 surface is preserved via update_legacy() (tiny adapter).
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
NODE_OUTLINE = (45, 50, 65)
CONN_NEG = (60, 120, 255)
CONN_POS = (255, 80, 80)
CLASS_COLORS = [(40, 200, 180), (255, 150, 50), (220, 60, 200)]

GRID_RES = 40
NODE_RADIUS = 18
NODE_OUTLINE_W = 6
NODE_END_INSET = 0.12


def _calc_x(layer_idx: int, n_layers: int, pos_x: int, width: int) -> int:
    inset = int(width * NODE_END_INSET)
    if n_layers <= 1:
        return pos_x + width // 2
    if layer_idx == 0:
        return pos_x + inset
    if layer_idx == n_layers - 1:
        return pos_x + width - inset
    n_hidden = n_layers - 2
    span = width - 2 * inset
    return pos_x + inset + (layer_idx * span) // (n_hidden + 1)


def _calc_y(node_idx: int, n_nodes: int, n_total: int, pos_y: int, height: int) -> int:
    if n_total <= 1 or n_nodes == 1:
        return pos_y + height // 2
    dy = height // max(n_total - 1, 1)
    used = (n_nodes - 1) * dy
    top = pos_y + (height - used) // 2
    return top + node_idx * dy


def _connection_thickness(weight: float) -> int:
    return max(1, min(5, int(abs(weight) * 2)))


# v2 panels available. Ticket 09 adds gym_render / attention_heatmap /
# attention_inspector; for now they fall back to weight_graph.
KNOWN_PANELS = {"boundary", "weight_graph"}


class Visualizer:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("neural-network visualizer")
        self.big_font = pygame.font.SysFont("consolas", 18, bold=True)
        self.panel = "boundary"  # default right-panel mode
        self._last_nn = None  # cached for hit_test_input_node

    def close(self) -> None:
        pygame.quit()

    def set_panel(self, name: str) -> None:
        """Switch the right-panel mode. Unknown names fall back to weight_graph."""
        # ponytail: silent fallback over raising; viz keeps running on bad input
        self.panel = name if name in KNOWN_PANELS else "weight_graph"

    def update(self, metrics: dict, panel: str | None = None) -> None:
        """v2 entry point. metrics carries epoch/loss/acc/dataset/etc.
        panel overrides the active right-panel mode for this frame."""
        if panel is not None:
            self.set_panel(panel)
        self._render(metrics)

    def update_legacy(
        self,
        nn,
        X: np.ndarray,
        y_int: np.ndarray,
        epoch: int,
        loss: float,
        acc: float,
        dataset_name: str,
    ) -> None:
        """v1 shim: same signature as the old update(). Builds a metrics
        dict and delegates. Used by main.py v1 path until ticket 10 lands."""
        self._last_nn = nn
        metrics = {
            "epoch": epoch,
            "loss": loss,
            "acc": acc,
            "dataset": dataset_name,
            "_nn": nn,
            "_X": X,
            "_y": y_int,
        }
        self.update(metrics, panel="boundary")

    # --- internal ---------------------------------------------------------

    def _render(self, metrics: dict) -> None:
        nn = metrics.get("_nn")
        self.screen.fill(BG_COLOR)
        self._draw_top_bar(metrics)
        if nn is not None:
            X = metrics.get("_X")
            y_int = metrics.get("_y")
            assert X is not None and y_int is not None, (
                "metrics must carry _X and _y when _nn is set"
            )
            self._draw_decision_boundary(nn, X, y_int)
            self._draw_weight_graph(nn)
        pygame.display.flip()

    def _draw_top_bar(self, metrics: dict) -> None:
        pygame.draw.rect(self.screen, BAR_COLOR, (0, 0, WIDTH, TOP_BAR_H))
        parts = []
        if "dataset" in metrics:
            parts.append(f"dataset={metrics['dataset']}")
        if "epoch" in metrics:
            parts.append(f"epoch={metrics['epoch']}")
        if "loss" in metrics:
            parts.append(f"loss={metrics['loss']:.3f}")
        if "acc" in metrics:
            parts.append(f"acc={metrics['acc']:.2f}")
        if "reward" in metrics:
            parts.append(f"reward={metrics['reward']:.1f}")
        if "tokens" in metrics:
            parts.append(f"tokens={metrics['tokens']}")
        if "panel" in metrics or self.panel:
            parts.append(f"panel={self.panel}")
        text = "  ".join(parts) if parts else "neural-network"
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
        n_total = max(nn.input_cache.shape[1], max(layer["W"].shape[1] for layer in layers))

        for i, layer in enumerate(layers):
            fan_in, fan_out = layer["W"].shape
            x_in = _calc_x(i, n_layers, pos_x, size_w)
            x_out = _calc_x(i + 1, n_layers, pos_x, size_w)
            for k in range(fan_in):
                y_in = _calc_y(k, fan_in, n_total, pos_y, size_h)
                for j in range(fan_out):
                    y_out = _calc_y(j, fan_out, n_total, pos_y, size_h)
                    w = layer["W"][k, j]
                    color = CONN_POS if w > 0 else CONN_NEG
                    thick = _connection_thickness(w)
                    pygame.draw.line(self.screen, color, (x_in, y_in), (x_out, y_out), thick)
                    pygame.draw.aaline(self.screen, color, (x_in, y_in), (x_out, y_out))

        for i in range(n_layers):
            if i == 0:
                values = nn.input_cache[0]
            else:
                values = layers[i - 1]["a"][0]
            n_nodes = values.shape[0]
            for j in range(n_nodes):
                x = _calc_x(i, n_layers, pos_x, size_w)
                y = _calc_y(j, n_nodes, n_total, pos_y, size_h)
                gray = int(np.clip(values[j] * 255, 0, 255))
                fill = (gray, gray, gray)
                pygame.draw.circle(self.screen, NODE_OUTLINE, (x, y), NODE_RADIUS, NODE_OUTLINE_W)
                inner = NODE_RADIUS - NODE_OUTLINE_W // 2
                pygame.gfxdraw.filled_circle(self.screen, x, y, inner, fill)
                pygame.gfxdraw.aacircle(self.screen, x, y, inner, fill)
                pygame.gfxdraw.aacircle(self.screen, x, y, NODE_RADIUS, NODE_OUTLINE)

    def hit_test_input_node(self, mx: int, my: int) -> int | None:
        if not hasattr(self, "_last_nn") or self._last_nn is None:
            return None
        nn = self._last_nn
        layers: list[dict] = nn.layers
        pos_x = PANEL_W + 40
        pos_y = TOP_BAR_H + 20
        size_w = PANEL_W - 80
        size_h = PANEL_H - 40
        n_layers = len(layers) + 1
        n_nodes = nn.input_cache.shape[1]
        n_total = max(nn.input_cache.shape[1], max(layer["W"].shape[1] for layer in layers))
        x = _calc_x(0, n_layers, pos_x, size_w)
        for j in range(n_nodes):
            y = _calc_y(j, n_nodes, n_total, pos_y, size_h)
            if (mx - x) ** 2 + (my - y) ** 2 <= NODE_RADIUS ** 2:
                return j
        return None


# --- standalone demo (v1 surface preserved via update_legacy) ----------------

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
            viz.update_legacy(nn, X, y, epoch, losses[-1], acc, "xor")
            clock.tick(60)
    finally:
        viz.close()
