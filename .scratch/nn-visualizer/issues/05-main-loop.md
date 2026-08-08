# 05 — Main loop with controls (keys + click)

**What to build:** `main.py` that wires the cleaned-up network, the
three datasets, and the visualizer into a single training loop with
the spec's interactivity (keys + click). From the user's perspective:
`python3 main.py` opens the visualizer on XOR; pressing `1`/`2`/`3`
swaps datasets, `SPACE` pauses, `R` resets weights, `+`/`-` adjusts
learning rate, `F` toggles fast-forward, and clicking on an input
node toggles its value.

**Blocked by:** 02 (need all three datasets), 04 (need the
visualizer).

**Status:** ready-for-agent

- [ ] `main.py` is the CLI entrypoint. Accepts `--dataset`,
      `--epochs`, `--lr`, `--arch` with sensible defaults
      (`xor`, `2000`, `0.05`, `[2, 8, 8, 3]`).
- [ ] Training loop runs at 60fps via `clock.tick(60)`. Each frame,
      call `nn.fit(...)` for `epochs_per_frame` epochs (default 10;
      toggleable to 200 with `F` for fast-forward).
- [ ] `1` / `2` / `3` — swap dataset (re-create network, reset
      weights, restart training on the new dataset).
- [ ] `SPACE` — pause/resume the training loop (clock still ticks,
      `epochs_per_frame` drops to 0).
- [ ] `R` — re-init weights with the same seed, reset epoch counter.
- [ ] `+` / `-` — increase / decrease learning rate (clamped
      0.001 ≤ lr ≤ 1.0). New lr applies to subsequent epochs only.
- [ ] `F` — toggle `epochs_per_frame` between 10 and 200.
- [ ] `ESC` — quit cleanly (close window, exit 0).
- [ ] **Click on input node** — toggle that input between 0 and 1,
      pause training automatically while the click is held, re-run
      forward pass so the user sees the network react. Uses the
      same `calcX` / `calcY` layout pattern the spec describes
      (no copy from old `ai` repo — re-implement clean).
- [ ] Total LOC ≤ 150 lines.
- [ ] `__main__` block at the bottom so `python3 -m main` also
      works.

## Notes

This ticket is pure glue. No math, no new rendering. The
non-obvious part is the click-on-input-node handler — coordinate
math must match the visualizer's layout exactly. Coordinate the
layout helper with ticket 04 if it isn't already shared.