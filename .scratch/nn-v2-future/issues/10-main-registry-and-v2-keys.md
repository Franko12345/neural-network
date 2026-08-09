# 10 — `main.py` registry + v2 keys (4/5/6/T)

**What to build:** Refactor `main.py` to use a task registry. Each
task (xor/circle/spiral/mountaincar/transformer) has the same
interface: `reset()`, `step()`, `render()`, `metrics()`. Main loop
just dispatches the active task.

**Blocked by:** 09 (v2 panels), 03 (mountaincar rollout), 06 (model).

**Status:** ready-for-agent

- [x] `main.py`: `TASKS = {"xor": XorTask(), "circle": CircleTask(),
      "spiral": SpiralTask(), "mountaincar": MountainCarTask(),
      "transformer": TransformerTask()}` — uniform interface
- [x] `Task` protocol (or ABC): `.reset()`, `.step()`, `.render()`,
      `.metrics() -> dict`
- [x] Key bindings:
      - `1/2/3` — xor/circle/spiral (v1, unchanged)
      - `4` — MountainCar
      - `5` — Transformer
      - `6` — Transformer attention inspector
      - `T` — type a prompt (transformer mode)
      - `SPACE` pause, `R` reset, `+/-` lr, `F` fast-forward, `ESC`
        quit (all v1, unchanged)
- [x] `XorTask` and friends wrap existing v1 logic; new tasks
      compose the v2 modules
- [x] Headless smoke: launch each task for 10 frames under dummy
      SDL, exit 0; no crashes
- [x] ponytail: tasks are simple objects with no inheritance
      hierarchy beyond a `Task` protocol; no flags on `step()`
- [x] v1 tests still pass (sanity)
- [x] Total LOC ≤ 250 (more than v1's 108 due to 5 tasks vs 3)
