# 11 — Wrap-up: requirements + v0.2.0 + release notes

**What to build:** Final ticket. Update `requirements.txt` to add
`gymnasium`, bump version to `v0.2.0`, generate 3 new GIFs (mountaincar,
transformer training, transformer sampling), write release notes.

**Blocked by:** 10 (main with all 5 tasks working).

**Status:** ready-for-agent

- [x] `requirements.txt` gains `gymnasium>=0.29` (v1 install path still
      works without it; v2 install pulls it in)
- [x] README updated with new section: "MountainCar + Transformer
      (v2)" showing 2-3 new GIFs in a second visual table
- [x] Tag `v0.2.0` on the merge commit, push tag
- [x] GitHub release notes for v0.2.0:
      - What: MountainCar + Transformer added
      - Numbers: 11 tickets closed, 11 new files, total LOC ~2200
      - How to use: `python3 main.py` then press `4` for
        MountainCar or `5` for Transformer
- [x] `CONTEXT.md`: status flipped from "Shipped (v1)" → "v2 ready";
      roadmap v1 → implemented, v2 → implemented
- [x] v1 spec at `.scratch/nn-visualizer/README.md` stays
      `status: implemented` (no change)
- [x] v2 spec at `.scratch/nn-v2-future/README.md` frontmatter flipped
      `status: ready-for-agent` → `status: implemented` (per Matt
      Pocock pitfall: spec not marked "Implementada" after parent
      closes)
- [x] `.scratch/nn-v2-future/issues/README.md` tickets table all
      flipped to `implemented`
- [x] Final commit message references parent spec
