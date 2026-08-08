# Domain Docs — Layout & Consumer Rules

This repo uses the **single-context** domain doc layout.

## Layout

```
neural-network/
├── CONTEXT.md                # the one and only context doc
├── docs/
│   ├── agents/               # this directory — engineering skill config
│   │   ├── issue-tracker.md
│   │   ├── triage-labels.md
│   │   └── domain.md         # this file
│   └── adr/                  # Architecture Decision Records
│       ├── README.md
│       ├── 0001-record-architecture-decisions.md
│       ├── 0002-numpy-only-no-pytorch.md
│       └── ...
├── AGENTS.md                 # agent entry point
└── .scratch/                 # local-markdown issues (see issue-tracker.md)
```

## Rules for consumers (skills / agents / humans)

1. **Start with `CONTEXT.md`.** It is the one doc loaded on every session
   that touches this repo. Keep it current; treat it as the project's
   shared memory.
2. **ADRs in `docs/adr/`** capture decisions that shape the project
   (e.g. "we use numpy-only, no PyTorch"; "headless LXC, no display").
   Number them sequentially (`NNNN-kebab-title.md`). Each ADR follows
   the Nygard template (Context / Decision / Consequences).
3. **`docs/agents/`** is for engineering-skill configuration only —
   NOT for project domain docs. If you need to add a new convention,
   edit the file here; do not create ad-hoc READMEs at the repo root.
4. **`AGENTS.md`** is the entry point that names this layout. It points
   back at the three files in `docs/agents/`.
5. **When in doubt about layout**, edit an existing file rather than
   creating a new one. The skill ecosystem is more reliable when docs
   stay flat and predictable.

## When to switch to multi-context

Multi-context (root `CONTEXT-MAP.md` + per-context `CONTEXT.md`) is for
genuine monorepos: separate `packages/*` with their own context. This
repo is a single Python project — single-context is correct. If a
future split happens (e.g. `core/` and `viz/` packages), re-run
`setup-matt-pocock-skills` to migrate.