# 09 — Visualizer v2 panels (gym render + attention heatmap)

**What to build:** Add two new panels to the visualizer: gym render
for MountainCar mode, and attention heatmap for Transformer mode.
Both integrate with the refactored `update(metrics=...)` API.

**Blocked by:** 08 (visualizer refactor), 03 (mountaincar rollout),
06 (transformer model).

**Status:** ready-for-agent

- [x] Gym render panel: takes RGB array from `env.render()`, draws
      on the right side when `panel="gym_render"`. Frame rate matches
      gym's internal rendering (typically 50fps).
- [x] Attention heatmap panel: takes `attention_weights` from last
      transformer forward (shape `(n_heads, seq_len, seq_len)`).
      Renders the first head as a 128×128 heatmap. Color: dark blue
      (low) → yellow → red (high).
- [x] Attention inspector (panel="attention_inspector"): click on a
      transformer block node selects that head/layer; the heatmap
      updates to show that head's attention matrix.
- [x] Headless smoke: each new panel renders 10 frames under dummy
      SDL, exits 0
- [x] ponytail: panel-specific draw methods stay in `visualizer.py`
      (no separate `panels/` module — keeps file count low)
- [x] v1 tests still pass
