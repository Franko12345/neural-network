# nn-visualizer — Issues (v1)

Vertical slices for the v1 implementation. Each ticket is a
tracer-bullet: cuts a narrow but complete path through math / data /
viz / main / docs.

## Dependency graph

```
01 ─┬─→ 02 ─┐
    └─→ 04 ──┴─→ 05 ─→ 06
        ↘    ↗
         03 ────────→ 06
```

## Tickets

| # | Title | Blocked by | Status |
|---|---|---|---|
| [01](./01-clean-up-nn-py.md) | Clean up `nn.py` to match spec | None | implemented |
| [02](./02-dataset-generators.md) | Dataset generators (XOR / circle / spiral) | 01 | implemented |
| [03](./03-headless-tests.md) | Headless tests for the single seam | 01, 02 | implemented |
| [04](./04-pygame-visualizer.md) | Pygame visualizer (boundary + weight graph) | 01 | implemented |
| [05](./05-main-loop.md) | Main loop with controls (keys + click) | 02, 04 | implemented |
| [06](./06-headless-smoke-and-readme.md) | Headless smoke test + README | 03, 05 | implemented |

## Frontier (parallelizable)

After **01** lands, the frontier splits:
- **02** and **04** are independent — both can run in parallel.
- **03** waits for 01+02.
- **05** waits for 02+04.
- **06** waits for everything (wrap-up + spec flip).

## Per-ticket cycle

For each ticket in dependency order:

1. `git checkout -b feat/<NN>-<slug>`
2. TDD: write the failing test (or smoke check) FIRST, see it fail,
   implement, see it pass.
3. `git commit` — single commit, conventional message,
   `Refs #<NN>`.
4. Smoke test locally (on LXC: tests + `SDL_VIDEODRIVER=dummy`).
5. Merge to `main` (this is a single-actor repo — no PR review,
   straight merge).
6. Update `issues/README.md` status if needed.
7. Move to next frontier ticket.