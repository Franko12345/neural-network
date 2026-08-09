# 08 — Visualizer refactor: metrics dict + tabs

**What to build:** Refactor `visualizer.py` so `update()` accepts a
flexible `metrics` dict (instead of fixed kwargs), and add a tab
mechanism to switch between modes. v1 modes (xor/circle/spiral)
keep working unchanged.

**Blocked by:** None — can run in parallel with 01.

**Status:** ready-for-agent

- [x] `update()` signature becomes
      `update(nn_or_None, X_or_None, metrics: dict, panel: str)`
      where `metrics` carries `epoch/loss/acc/reward/tokens/etc` and
      `panel` selects the right-panel mode (`"boundary"`,
      `"weight_graph"`, `"gym_render"`, `"attention_heatmap"`)
- [x] Old `update(nn, X, y_int, epoch, loss, acc, dataset_name)` is
      removed (v1 visualizer code is rewritten, not extended)
- [x] Panel switching via `set_panel(name)` (called by main.py)
- [x] Backwards-compatible smoke: a tiny adapter in v1's `__main__`
      block passes the old kwargs into the new dict form, exits 0
- [x] Headless smoke: `SDL_VIDEODRIVER=dummy python3 -c "from
      visualizer import Visualizer; ..."` opens, renders 10 frames,
      exits 0
- [x] ponytail: one method per public action; metrics dict keys are
      flat strings; no nested state
- [x] v1 tests still pass (sanity — visualizer doesn't directly
      test, but smoke confirms nothing broke)