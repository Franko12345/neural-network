# Triage Labels

This repo uses the **Matt Pocock default** triage label vocabulary.
The `triage` skill applies these labels (or sets the corresponding
`status:` field on the local markdown issue) when moving work through
the queue.

## The five canonical labels

| Label | Meaning | Typical action |
|---|---|---|
| `needs-triage` | Just landed; not yet looked at. | Review, classify, decide path. |
| `needs-info` | Blocked waiting for human clarification. | Ask the user, then re-classify. |
| `ready-for-agent` | Spec is clear; an agent can implement it. | Pick up via `implement`. |
| `ready-for-human` | Needs human judgment, code review, or manual work. | Hand back to the user. |
| `wontfix` | Considered and rejected; document why. | Close with reason. |

## Local-markdown mapping

In `.scratch/<feature>/README.md`, the same five states map to the
`status:` frontmatter field:

```markdown
---
status: ready-for-agent
---

# <feature title>
...
```

Status values (lowercase, hyphenated):

- `needs-triage`
- `needs-info`
- `ready-for-agent`
- `ready-for-human`
- `wontfix`

## Override

If the user later wants to rename a label (e.g. `bug:triage` instead of
`needs-triage`), edit the table here AND update the `triage` skill's
behavior. This file is the source of truth for label *strings*.