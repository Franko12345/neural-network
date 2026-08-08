# 06 — Headless smoke test + README

**What to build:** A `README.md` that explains how to install, run,
and what each key does, plus a documented headless smoke test that
verifies the visualizer doesn't crash without a display. From the
user's perspective: a fresh clone of this repo, after
`pip install numpy pygame-ce gymnasium`, runs `python3
tests/test_nn.py` to verify math, then `python3 main.py` on their
desktop to see the visualizer.

**Blocked by:** 03 (tests must pass), 05 (main loop must work).

**Status:** ready-for-agent

- [ ] `README.md` at repo root with sections: Quick start, What it
      does, Key bindings (table of `1/2/3/space/R/+/-/F/ESC` + click),
      File layout, Limitations, Reference to `docs/agents/issue-tracker.md`
      and `CONTEXT.md`.
- [ ] `requirements.txt` lists `numpy`, `pygame-ce` (and
      `gymnasium` only when v2 lands — for now, just the first
      two).
- [ ] Documented headless smoke test command:
      `SDL_VIDEODRIVER=dummy python3 main.py --dataset xor --epochs 5`
      must exit 0 on the LXC after a few seconds.
- [ ] The smoke test is reproducible: any agent on a fresh LXC
      can run it and confirm v1 is shippable.
- [ ] No fabricated screenshots — README describes what the user
      will see ("left panel: decision boundary deforming; right
      panel: weight graph with blue/red connections"). User adds
      screenshots themselves if they want.
- [ ] `CONTEXT.md` updated to flip v1's **Status** from
      "Pre-implementation" → "Shipped (v1)" and add a one-line
      "How to run" section pointing at the README.
- [ ] The v1 spec at `.scratch/nn-visualizer/README.md` has its
      frontmatter `status` flipped from `ready-for-agent` →
      `implemented` (Matt Pocock workflow — see
      `feature-loop` skill's "Spec doesn't get marked
      'Implementada' after the parent ticket closes" pitfall).
- [ ] Final commit message references the parent spec.

## Notes

This ticket is **wrap-up only** — no logic, no rendering. The
pitfall the `feature-loop` skill warns about (spec not flipped to
"Implementada") is explicitly called out so it doesn't get
forgotten.