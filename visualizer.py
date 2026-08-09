"""Pygame visualizer: decision boundary (left) + weight graph (right).

ponytail: render-only, no training logic. Caller drives training and
calls update() each frame. v2 refactor: update() takes a metrics dict
plus optional nn/X/y kwargs for the v1-shim path; set_panel(name)
switches the right-panel mode (raises on unknown).
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
    # ponytail: |w|*2 per spec, floored at 1 so tiny weights still draw
    return max(1, min(5, int(abs(weight) * 2)))


def _heat_color(v: float) -> tuple:
    """Blue (low) → yellow (mid) → red (high). v in [0, 1]."""
    v = max(0.0, min(1.0, v))
    if v < 0.5:
        # blue → yellow
        t = v * 2.0
        r = int(40 + t * (255 - 40))
        g = int(80 + t * (255 - 80))
        b = int(200 - t * (200 - 60))
    else:
        # yellow → red
        t = (v - 0.5) * 2.0
        r = int(255 - t * (255 - 200))
        g = int(255 - t * (255 - 60))
        b = int(60 - t * 60)
    return (r, g, b)


# v2 panels available. Ticket 09 adds gym_render / attention_heatmap /
# attention_inspector. Unknown panel names now raise (caller bug, not ours).
KNOWN_PANELS = {"boundary", "weight_graph", "gym_render",
                 "attention_heatmap", "attention_inspector"}


class Visualizer:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("neural-network visualizer")
        self.big_font = pygame.font.SysFont("consolas", 18, bold=True)
        self.panel = "boundary"
        self._last_nn = None  # cached for hit_test_input_node

    def close(self) -> None:
        pygame.quit()

    def set_panel(self, name: str) -> None:
        """Switch the right-panel mode. Raises on unknown panel (caller bug)."""
        if name not in KNOWN_PANELS:
            raise ValueError(
                f"unknown panel: {name!r}; known={sorted(KNOWN_PANELS)}"
            )
        self.panel = name

    def update(
        self,
        metrics: dict,
        panel: str | None = None,
        *,
        nn=None,
        X: np.ndarray | None = None,
        y_int: np.ndarray | None = None,
    ) -> None:
        """v2 entry point. metrics carries flat scalar fields (epoch/loss/acc
        /dataset/etc). nn/X/y are kwargs so the metrics dict stays flat;
        omit them for non-network panels (gym render, attention inspector).
        """
        if panel is not None:
            self.set_panel(panel)
        if nn is not None:
            self._last_nn = nn  # cache for hit_test_input_node
        self._render(metrics, nn, X, y_int)

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
        """v1 shim: same signature as the old update(). Builds a flat
        metrics dict + kwargs and delegates. Used by main.py v1 path
        until ticket 10 lands."""
        metrics = {"epoch": epoch, "loss": loss, "acc": acc, "dataset": dataset_name}
        self.update(metrics, panel="boundary", nn=nn, X=X, y_int=y_int)

    # --- internal ---------------------------------------------------------

    def _render(
        self,
        metrics: dict,
        nn,
        X: np.ndarray | None,
        y_int: np.ndarray | None,
    ) -> None:
        self.screen.fill(BG_COLOR)
        self._draw_top_bar(metrics)
        # Right-panel mode routes to a different renderer. The left side
        # stays empty for gym_render / attention_* modes (the panel is
        # the whole right half). For boundary / weight_graph, we render
        # the v1 left+right split.
        if self.panel in ("gym_render", "attention_heatmap", "attention_inspector"):
            self._draw_right_panel(metrics)
        elif nn is not None:
            assert X is not None and y_int is not None, (
                "nn given without X/y_int"
            )
            self._draw_decision_boundary(nn, X, y_int)
            self._draw_weight_graph(nn)
        pygame.display.flip()

    def _draw_right_panel(self, metrics: dict) -> None:
        """Dispatch to the right-panel renderer based on self.panel."""
        if self.panel == "gym_render":
            self._draw_gym_render(metrics.get("frame"))
        elif self.panel == "attention_heatmap":
            self._draw_attention_heatmap(metrics.get("attention_weights"),
                                         head=metrics.get("selected_head", 0))
        elif self.panel == "attention_inspector":
            head = metrics.get("selected_head", 0)
            layer = metrics.get("selected_layer", 0)
            self._draw_attention_inspector(metrics.get("attention_weights"),
                                           head=head, layer=layer,
                                           n_layers=metrics.get("n_layers", 1))

    def _draw_gym_render(self, frame: np.ndarray | None) -> None:
        """Render the gym env frame as a pygame surface on the right
        panel. Frame is (H, W, 3) uint8."""
        if frame is None:
            return
        # Transpose to (W, H, 3) for pygame.surfarray
        h, w = frame.shape[:2]
        # Scale frame to fit the right panel
        surf = pygame.Surface((w, h))
        pygame.surfarray.blit_array(surf, np.transpose(frame, (1, 0, 2)))
        # Blit onto right panel area
        dest_x = PANEL_W + (PANEL_W - w) // 2
        dest_y = TOP_BAR_H + (PANEL_H - h) // 2
        self.screen.blit(surf, (dest_x, dest_y))

    def _draw_attention_heatmap(self, attn: np.ndarray | None,
                                head: int = 0) -> None:
        """Render attn[head] as a heatmap (T x T) on the right panel.
        Color: dark blue (low) → yellow → red (high)."""
        if attn is None:
            return
        if head >= attn.shape[0]:
            head = 0
        mat = attn[head]
        T = mat.shape[0]
        if T == 0:
            return
        # Cell size: fit T x T into PANEL_W x PANEL_H minus margins.
        cell = min((PANEL_W - 40) // T, (PANEL_H - 40) // T)
        ox = PANEL_W + 20
        oy = TOP_BAR_H + 20
        # Color mapping: low=blue, mid=yellow, high=red (perceptually OK).
        for i in range(T):
            for j in range(T):
                v = float(mat[i, j])
                color = _heat_color(v)
                pygame.draw.rect(self.screen, color,
                                 (ox + j * cell, oy + i * cell, cell, cell))

    def _draw_attention_inspector(self, attn: np.ndarray | None,
                                   head: int = 0, layer: int = 0,
                                   n_layers: int = 1) -> None:
        """Render selected layer/head's attention on right panel;
        draw a small grid of head thumbnails for the inspector."""
        if attn is None:
            return
        if attn.ndim == 3:
            # Single layer: shape (H, T, T)
            layers = [attn]
        else:
            # Multi-layer: shape (L, H, T, T)
            layer = min(layer, attn.shape[0] - 1)
            layers = [attn[layer]]
        # Big heatmap on top
        self._draw_attention_heatmap(layers[0], head=head)
        # Head thumbnails on bottom
        n_heads = layers[0].shape[0]
        T = layers[0].shape[1]
        thumb_size = max(8, min(40, (PANEL_H // 2) // max(n_heads, 1)))
        ox = PANEL_W + 20
        oy = TOP_BAR_H + PANEL_H // 2 + 10
        for h in range(n_heads):
            sel = (h == head)
            color = (255, 255, 0) if sel else (80, 80, 80)
            pygame.draw.rect(self.screen, color,
                             (ox + h * (thumb_size + 2),
                              oy, thumb_size, thumb_size), 1)

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


# --- standalone demo --------------------------------------------------------

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
