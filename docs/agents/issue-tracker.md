# Issue Tracker — Local Markdown

Issues for this project live as **markdown files under `.scratch/<feature>/`**
in the repo root. This is a solo / no-remote workflow; no GitHub / GitLab
backend.

## Layout

```
.scratch/
├── README.md                    # index of all open features
├── <feature-slug>/
│   ├── README.md                # issue spec — what & why
│   ├── plan.md                  # implementation plan / approach
│   ├── adr-NNNN-title.md        # linked ADRs (or symlink to docs/adr/)
│   └── notes.md                 # working notes / scratch
```

`<feature-slug>` is kebab-case (e.g. `nn-core`, `pygame-visualizer`,
`xor-dataset`).

## File conventions

- **`README.md`** is the entry point. Skills that "list issues" read this
  file. Frontmatter is optional; YAML `status` is allowed.
- Issues have a single status: `draft`, `ready`, `in-progress`, `done`,
  `wontfix`. Default to `draft`; flip to `ready` once the spec is solid.
- The `## Agent skills` block in `AGENTS.md` points here.
- PRs as a request surface: **off** (no PRs in this workflow).

## Skills that read / write this tracker

- `to-spec` — drafts an issue from a conversation, saves to
  `.scratch/<feature>/README.md`
- `to-tickets` — breaks a spec into vertical-slice issues under
  `.scratch/<feature>-<slice>/`
- `triage` — moves issues between `needs-triage` → `ready-for-agent` /
  `ready-for-human` (state in the `status:` frontmatter, label taxonomy
  in `triage-labels.md`)
- `implement` — picks a `ready-for-agent` issue, implements, opens a PR
- `qa` — review-checked runs

## When to switch trackers

If the repo later gets a `git remote` pointing at GitHub / GitLab,
re-run `setup-matt-pocock-skills` to migrate.

## Editing

This file describes the convention. To change the layout, edit this
file — the skills read it on each invocation.