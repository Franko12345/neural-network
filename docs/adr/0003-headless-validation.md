# 0003. Headless-first validation

Date: 2026-08-08

## Status

Accepted

## Context

The dev environment is a headless Proxmox LXC. Pygame windows cannot
open here. The visualizer must be run on Franko's local desktop. If we
build the visualizer first, we can't validate anything until the user
runs it themselves — slow feedback loop.

## Decision

Development order:

1. **Core math (`nn.py`)** — pure numpy, no pygame.
2. **Tests (`tests/test_nn.py`)** — headless assertions on accuracy
   per dataset. **Run these on the LXC** to validate.
3. **Visualizer (`visualizer.py`)** — only after tests pass.
4. **Smoke test** — `SDL_VIDEODRIVER=dummy python3 main.py` on the LXC
   to verify it doesn't crash; visual correctness verified by the
   user on their desktop.

## Consequences

**Easier:**
- We know the math is correct before any visualization work.
- Bugs caught in numpy are easier to fix than bugs in pygame + numpy.

**Harder:**
- Need a way to "fake" the display for the smoke test.
  `SDL_VIDEODRIVER=dummy` works.

**When to revisit:**
- If we add datasets that require GPU compute, validation strategy
  changes.