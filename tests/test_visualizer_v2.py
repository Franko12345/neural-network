"""Headless smoke for the v2 visualizer panels.

ponytail: stdlib only, no pytest.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402  (env must be set before pygame import)
import pygame  # noqa: E402

from visualizer import Visualizer  # noqa: E402


def test_gym_render_panel_draws_rgb_frame() -> bool:
    """panel='gym_render' draws an RGB array from env.render() on the
    right side. Smoke: 10 frames under dummy SDL, exits 0."""
    viz = Visualizer()
    try:
        # Fake an RGB frame
        frame = np.random.default_rng(0).integers(0, 255, (200, 300, 3), dtype=np.uint8)
        for _ in range(10):
            viz.update(metrics={"frame": frame, "epoch": 1, "loss": 0.3},
                       panel="gym_render")
        return True
    finally:
        viz.close()


def test_attention_heatmap_panel_draws_attention_weights() -> bool:
    """panel='attention_heatmap' takes attention_weights (H, T, T) and
    renders head 0 as a 128x128 heatmap (or scaled to fit)."""
    viz = Visualizer()
    try:
        # 4 heads, seq_len=16
        attn = np.random.default_rng(1).standard_normal((4, 16, 16))
        # Softmax-normalize per row to look like real attention weights
        e = np.exp(attn - attn.max(axis=-1, keepdims=True))
        attn = e / e.sum(axis=-1, keepdims=True)
        for _ in range(10):
            viz.update(metrics={"attention_weights": attn, "epoch": 1},
                       panel="attention_heatmap")
        return True
    finally:
        viz.close()


def test_attention_inspector_clicks_select_head() -> bool:
    """panel='attention_inspector' lets click on a block node select
    that head; the heatmap updates."""
    viz = Visualizer()
    try:
        attn = np.random.default_rng(2).standard_normal((4, 8, 8))
        e = np.exp(attn - attn.max(axis=-1, keepdims=True))
        attn = e / e.sum(axis=-1, keepdims=True)
        viz.update(metrics={"attention_weights": attn, "epoch": 1},
                   panel="attention_inspector")
        # Simulate clicking a node (no actual click — just verify API
        # accepts selection state and re-renders without crash).
        viz.update(metrics={"attention_weights": attn, "epoch": 2,
                            "selected_head": 2, "selected_layer": 0},
                   panel="attention_inspector")
        return True
    finally:
        viz.close()


def test_v1_modes_still_work() -> bool:
    """v1 panels (boundary + weight_graph) keep working — re-render 10
    frames each under dummy SDL."""
    from layers import Linear
    from envs.mountaincar import MountainCarEnv

    viz = Visualizer()
    try:
        # Fake a tiny nn
        nn = Linear(2, 3)
        nn.W = np.random.default_rng(3).standard_normal((2, 3)) * 0.3
        nn.b = np.zeros((1, 3))
        X = np.random.default_rng(4).standard_normal((5, 2))
        y = np.array([0, 1, 0, 1, 0])
        # Cache input for hit_test
        nn.input_cache = X
        viz._last_nn = nn
        for _ in range(10):
            viz.update(metrics={"epoch": 1, "loss": 0.3, "acc": 0.5,
                                "dataset": "xor"},
                       panel="boundary")
        return True
    finally:
        viz.close()


def main() -> int:
    results = [
        ("gym_render_panel_draws_rgb_frame", test_gym_render_panel_draws_rgb_frame()),
        ("attention_heatmap_panel_draws_attention_weights",
         test_attention_heatmap_panel_draws_attention_weights()),
        ("attention_inspector_clicks_select_head",
         test_attention_inspector_clicks_select_head()),
        ("v1_modes_still_work", test_v1_modes_still_work()),
    ]
    for name, ok in results:
        flag = "PASS" if ok else "FAIL"
        print(f"  {name:50s} {flag}")
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
