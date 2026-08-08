# neural-network

Feedforward neural network from scratch in numpy, with a pygame
visualizer (decision boundary + animated weight graph). Visualizes
XOR, circle, and spiral datasets.

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

## Quick start

```bash
python3 -c "import numpy, pygame; print('ok')"   # verify deps
python3 tests/test_nn.py                        # headless test
python3 main.py --dataset xor                   # visualizer (needs display)
```

## Author

Franko.