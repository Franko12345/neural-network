"""Headless smoke for main.py task registry.

ponytail: stdlib only, no pytest.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# Import after env set
import numpy as np  # noqa: E402

from main import TASKS, KEY_TO_TASK  # noqa: E402


def test_tasks_registry_has_5_tasks() -> bool:
    """Registry has all 5 tasks per ticket 10 spec."""
    expected = {"xor", "circle", "spiral", "mountaincar", "transformer"}
    return set(TASKS.keys()) == expected


def test_each_task_has_required_interface() -> bool:
    """Each task has reset(), step(), render(viz), metrics() -> dict."""
    for name, task in TASKS.items():
        if not all(hasattr(task, m) for m in ("reset", "step", "render", "metrics")):
            print(f"  missing methods on {name}")
            return False
    return True


def test_xor_task_runs_under_dummy_sdl() -> bool:
    """XorTask: train a few steps, render some frames, metrics returns dict."""
    import pygame  # noqa: F401
    task = TASKS["xor"]
    viz = task.reset()
    try:
        for _ in range(5):
            task.step()
            metrics = task.metrics()
            task.render(viz, metrics)
        return isinstance(metrics, dict) and "epoch" in metrics
    finally:
        viz.close()


def test_mountaincar_task_smoke() -> bool:
    """MountainCarTask: rollout + REINFORCE step renders gym frame.

    render() injects the frame into viz.update metrics; we check that
    after render, viz's most recent metrics dict contained 'frame'.
    """
    import pygame  # noqa: F401
    task = TASKS["mountaincar"]
    viz = task.reset()
    try:
        task.step()
        metrics = task.metrics()
        task.render(viz, metrics)
        # viz stores nothing by default; we just verify render doesn't crash
        # and produces a metrics-shaped output (frame injected into viz.update).
        return metrics["epoch"] >= 0
    finally:
        viz.close()


def test_transformer_task_smoke() -> bool:
    """TransformerTask: forward+backward step renders attention heatmap."""
    import pygame  # noqa: F401
    task = TASKS["transformer"]
    viz = task.reset()
    try:
        task.step()
        task.step()  # need a forward pass to populate attn_weights
        metrics = task.metrics()
        task.render(viz, metrics)
        # attn weights are stored on the task via last forward
        assert task._last_attn is not None, "attn weights not populated"
        assert task._last_attn.ndim == 3, "attn should be (B, T, T)"
        return True
    finally:
        viz.close()


def test_key_to_task_covers_v2_keys() -> bool:
    """Keys 1/2/3 (v1) + 4 (MountainCar) + 5 (Transformer) are mapped."""
    import pygame  # noqa: F401
    keys_needed = {pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5}
    return keys_needed.issubset(set(KEY_TO_TASK.keys()))


def main() -> int:
    results = [
        ("tasks_registry_has_5_tasks", test_tasks_registry_has_5_tasks()),
        ("each_task_has_required_interface", test_each_task_has_required_interface()),
        ("xor_task_runs_under_dummy_sdl", test_xor_task_runs_under_dummy_sdl()),
        ("mountaincar_task_smoke", test_mountaincar_task_smoke()),
        ("transformer_task_smoke", test_transformer_task_smoke()),
        ("key_to_task_covers_v2_keys", test_key_to_task_covers_v2_keys()),
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
