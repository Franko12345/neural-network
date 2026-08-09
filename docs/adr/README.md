# Architecture Decision Records

ADRs capture the **why** behind significant project decisions. Each
decision gets one file, numbered sequentially.

## Index

- [0001 — Record architecture decisions](./0001-record-architecture-decisions.md)
- [0002 — Numpy-only, no PyTorch / TF](./0002-numpy-only-no-pytorch.md)
- [0003 — Headless-first validation, pygame display on user's machine](./0003-headless-validation.md)
- [0004 — Single-context domain docs](./0004-single-context-docs.md)
- [0005 — v2 builds alongside v1; v1 modules are frozen](./0005-v2-builds-alongside-v1.md)
- [0006 — W-order + double-update trap: backward() must compute gradients BEFORE mutating weights](./0006-w-order-double-update-trap.md)

## Template

Every ADR follows the Nygard template:

```markdown
# NNNN. <Short noun phrase>

Date: YYYY-MM-DD

## Status

Accepted | Superseded by NNNN | Deprecated

## Context

What is the issue we're seeing? What's motivating this decision?

## Decision

What did we choose to do?

## Consequences

What becomes easier? What becomes harder? What did we give up?
```