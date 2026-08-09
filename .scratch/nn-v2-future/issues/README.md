# nn-v2-future — Issues

Vertical slices for the v2 implementation: MountainCar-v0 with
REINFORCE + decoder-only transformer from scratch. Built on top of
v1's frozen `nn.py`/`visualizer.py`/`datasets.py` via a new
`layers.py` (linear + activations) and `modules.py` (layernorm,
residual, multi-head attention).

## Dependency graph

```
01 ─┬─→ 02 ─→ 03 ─┐
    │    ├─→ 05 ─→ 06 ─→ 07 ─┐
    │    └─→ 04 ────────────┘
    └─→ 08 (paralelo) ─────→ 09 ─→ 10 ─→ 11
```

## Tickets

| # | Title | Blocked by | Status |
|---|---|---|---|
| [01](./01-extract-layers.md) | Extract `layers.py` building blocks | None | implemented |
| [02](./02-mountaincar-env-and-rollout.md) | MountainCar env wrapper + rollout batch | 01 | ready-for-agent |
| [03](./03-reinforce-trainer.md) | REINFORCE trainer with baseline | 01, 02 | ready-for-agent |
| [04](./04-layernorm-residual-adamw.md) | LayerNorm + Residual + AdamW | 01 | ready-for-agent |
| [05](./05-multi-head-attention.md) | Multi-head attention with causal mask | 01 | ready-for-agent |
| [06](./06-transformer-block-and-model.md) | Transformer block + model stack | 04, 05 | ready-for-agent |
| [07](./07-transformer-trainer.md) | Transformer trainer + checkpointing | 06, 04 | ready-for-agent |
| [08](./08-visualizer-refactor-metrics-tabs.md) | Visualizer refactor: metrics dict + tabs | None | implemented |
| [09](./09-visualizer-v2-panels.md) | Visualizer v2 panels (gym + attention) | 08, 03, 06 | ready-for-agent |
| [10](./10-main-registry-and-v2-keys.md) | `main.py` registry + v2 keys | 09, 03, 06 | ready-for-agent |
| [11](./11-wrap-up-v0-2-0.md) | Wrap-up: requirements + v0.2.0 | 10 | ready-for-agent |

## Frontier (parallelizable)

**01 and 08 landed (PRs #10, #11, merged 2026-08-08).** Frontier now:
- **02** and **04** are independent — both can run in parallel.
- **05** can run in parallel with **04** (multi-head attention only
  needs 01).
- **03** waits for 02 (01 done).
- **06** waits for 04+05.
- **07** waits for 06+04.
- **09** waits for 03+06 (08 done).
- **10** waits for 09.
- **11** waits for everything.

## Per-ticket cycle

For each ticket in dependency order:

1. `git checkout -b feat/<NN>-<slug>`
2. TDD: write the failing test, see it fail, implement, see it pass
3. Verify locally: `python3 -m tests.test_*` + headless smoke
4. Commit + push + PR with reviews (3 reviewers: ponytail, Matt
   Pocock, correctness)
5. Apply findings, merge, close ticket
6. Move to next frontier ticket

## Cross-cutting rules

- v1 stays frozen. `nn.py`, `visualizer.py` (v1 surface), `datasets.py`,
  `tests/test_nn.py` are not modified.
- Each new module gets its own test file under `tests/`.
- `python3 -m tests.test_nn` must keep passing unchanged.
- `python3 -m tests.test_layers` etc. are headless (sub-10s).
- No new deps beyond `gymnasium` until v2 ships.

## Reference

- `../README.md` — spec for v1 (shipped in v0.1.0)
- `../CONTEXT.md` — project context
- `neural-networks-foundation` skill — chapters 5-8 cover exactly
  the concepts this ticket queue implements
