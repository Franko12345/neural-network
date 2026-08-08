# 0004. Single-context domain docs

Date: 2026-08-08

## Status

Accepted

## Context

This is a small single-package Python project. The Matt Pocock
engineering skills support both single-context (one `CONTEXT.md` at
repo root) and multi-context (root `CONTEXT-MAP.md` + per-context
`CONTEXT.md` files). Multi-context is for genuine monorepos.

## Decision

**Single-context.** One `CONTEXT.md` at repo root, ADRs in
`docs/adr/`, skill config in `docs/agents/`. See
[`docs/agents/domain.md`](../agents/domain.md).

## Consequences

**Easier:**
- Agents have exactly one place to look for project context.
- No ambiguity about which context file to load.

**Harder:**
- If the project ever splits into `core/` and `viz/` packages, we
  re-run `setup-matt-pocock-skills` to migrate.

**When to revisit:**
- A genuine monorepo split. Until then, single-context stays.