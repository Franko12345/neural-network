# 08 — Visualizer refactor: metrics dict + tabs

**What to build:** Refactor `visualizer.py` so `update()` accepts a
flexible `metrics` dict (instead of fixed kwargs), and add a tab
mechanism to switch between modes. v1 modes (xor/circle/spiral)
keep working unchanged.

**Blocked by:** None — can run in parallel with 01.

**Status:** ready-for-agent

- [x] `update()` is `update(metrics: dict, panel: str | None = None,
      *, nn=None, X=None, y_int=None)` — flat metrics dict for scalars,
      kwargs-only for the network/data (so future gym / attention
      panels can pass metrics without nn)
- [x] Old `update(nn, X, y_int, epoch, loss, acc, dataset_name)` is
      removed (v1 visualizer code is rewritten, not extended)
- [x] Panel switching via `set_panel(name)` (called by main.py)
- [x] Backwards-compatible smoke: a tiny adapter in v1's `__main__`
      block passes the old kwargs into the new dict form, exits 0
- [x] Headless smoke: `SDL_VIDEODRIVER=dummy python3 -c "from
      visualizer import Visualizer; ..."` opens, renders 10 frames,
      exits 0
- [x] ponytail: one method per public action; metrics dict keys flat
      strings; nn/X/y live as kwargs not nested keys

## Review findings applied (PR #11)

- API split: nn/X/y as kwargs → metrics dict stays flat (review #6).
- `set_panel` raises on unknown panel — no silent fallback (review #5).
- Deleted `reward` / `tokens` top-bar pre-stubs; add when ticket 09 lands (review #3).
- Tab binding in main.py makes `set_panel` literally "called by main.py" (Matt Pocock).
- `_draw_top_bar` no longer dead-branches `or self.panel`; always emits `panel=...` (review #8).
- [x] v1 tests still pass (sanity — visualizer doesn't directly
      test, but smoke confirms nothing broke)
