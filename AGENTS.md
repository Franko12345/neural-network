# neural-network

Feedforward neural network from scratch in numpy, with a pygame
visualizer (decision boundary + animated weight graph). Visualizes
XOR, circle, and spiral datasets.

**v1 shipped** (release `v0.1.0`). v2 (MountainCar + transformer) is
ready-to-build — see `.scratch/nn-v2-future/`.

## Agent skills

### Issue tracker

Issues live as markdown files under `.scratch/<feature>/` (local-only,
no remote). See [`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md).

### Triage labels

Uses the Matt Pocock default five-label vocabulary
(`needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human`
/ `wontfix`). See [`docs/agents/triage-labels.md`](docs/agents/triage-labels.md).

### Domain docs

Single-context layout: one `CONTEXT.md` at the repo root, ADRs under
`docs/adr/`. See [`docs/agents/domain.md`](docs/agents/domain.md).

**Always read `CONTEXT.md` first** — it has the current state, the
roadmap, and the active pitfalls from past sessions.

## Quick start

```bash
python3 -c "import numpy, pygame; print('ok'"   # verify deps
python3 -m tests.test_nn                        # headless test
python3 main.py --dataset xor                   # visualizer (needs display)
SDL_VIDEODRIVER=dummy python3 main.py ...       # headless smoke
```

For the full README, see `README.md`.

## Author

Franko.

## Versions

- **v1** — `nn-visualizer`: implemented (v0.1.0 tag).
  Spec: `.scratch/nn-visualizer/README.md`.
- **v2** — `nn-v2-future`: ready-for-agent.
  Spec: `.scratch/nn-v2-future/README.md`. 11 tickets published.
  v1 files stay frozen throughout v2.