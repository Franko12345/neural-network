# 0001. Record architecture decisions

Date: 2026-08-08

## Status

Accepted

## Context

We are starting a small project (`neural-network`) that will grow beyond
a single file. Decisions about deps, layout, and workflow will be made
quickly. Without a record, future agents (and Franko) will re-litigate
or forget the reasoning.

## Decision

We use **Architecture Decision Records (ADRs)** following the Nygard
template (Context / Decision / Consequences). ADRs live in
`docs/adr/`, numbered sequentially, never deleted — superseded ones
get a `Status: Superseded by NNNN` line.

## Consequences

**Easier:**
- Any agent can reconstruct the project's history from the ADRs.
- New decisions get a fresh number, old ones are immutable.

**Harder:**
- Slight overhead per decision. Worth it for anything that affects
  layout, deps, or workflow.