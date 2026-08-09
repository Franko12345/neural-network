# neural-network — Context

> The one doc loaded on every session that touches this repo. Keep
> it current.

## What this is

A from-scratch feedforward neural network in **numpy**, with a
**pygame** visualizer that animates the decision boundary and the
weight graph during training. Three toy datasets: XOR, circle, spiral.

**Shipped as v0.1.0.** See https://github.com/Franko12345/neural-network/releases/tag/v0.1.0

## Goals

- **Educational**: see backprop actually run, weight values shift,
  decision boundary deform.
- **Minimal**: no PyTorch, no TF. Each component reads top-to-bottom.
- **Reproducible**: fixed seed (`np.random.default_rng(42)`).
- **Extensible**: v2 (MountainCar + transformer) builds on top via
  `layers.py` building blocks without modifying v1 files.

## Non-goals

- Real-scale datasets (ImageNet, language at scale).
- GPU acceleration.
- Production deployment / serving.
- Multi-contributor workflow (single maintainer only).

> Note: "Modern architectures (transformers, convnets)" was a v1
> non-goal. As of the v2 spec, transformers are now in scope via
> `.scratch/nn-v2-future/`.

## Stack

- Python 3.11+
- numpy 2.x
- pygame-ce 2.5+
- (v2 only) gymnasium 0.29+

All three v1 deps verified present in the LXC dev env. No `pip
install` needed for the v1 core path.

## File layout

```
neural-network/
├── CONTEXT.md                    # this file
├── AGENTS.md                     # agent entry point
├── README.md                     # user-facing docs (with GIFs)
├── nn.py                         # v1 core math (FROZEN — do not modify)
├── datasets.py                   # v1 toy dataset generators (FROZEN)
├── visualizer.py                 # v1 pygame render (FROZEN v1 surface)
├── main.py                       # v1 CLI loop with key/click controls
├── sanity.py                     # v1 one-shot smoke check for nn.py
├── tests/
│   └── test_nn.py                # v1 headless validation (FROZEN)
├── requirements.txt              # numpy + pygame-ce pins
├── docs/
│   ├── agents/                   # engineering-skill config
│   └── adr/                      # ADRs (NNNN-kebab-title.md)
└── .scratch/
    ├── nn-visualizer/            # v1 spec + 6 tickets (all implemented)
    │   ├── README.md             # status: implemented
    │   └── issues/               # 01-06 all closed
    ├── nn-v2-future/             # v2 spec + 11 tickets (ready-for-agent)
    │   ├── README.md             # status: ready-for-agent
    │   └── issues/               # 01-11 (01-08 unblocked; 09-11 downstream)
    └── screenshots/              # README GIFs + asset-generation script
```

## Status

**v1 shipped** (`v0.1.0` tag, GitHub release published). Headless tests
pass in ~3.6s. Visualizer runs at 60fps on a desktop with display.

**v2 ready-to-build.** Spec reviewed structurally (10 changes for
v1-compat); 11 tickets published. v1 stays frozen throughout v2.

## How to run

```bash
pip install -r requirements.txt
python3 -m tests.test_nn                  # verify math
python3 main.py --dataset xor             # visualizer (needs display)
SDL_VIDEODRIVER=dummy python3 main.py ... # headless smoke (no display)
```

See `README.md` for key bindings, CLI options, and visual demo GIFs.

## Roadmap

| Version | Status | Spec | Scope |
|---|---|---|---|
| **v1 — nn-visualizer** | `implemented` (v0.1.0 released) | `.scratch/nn-visualizer/README.md` | Toy datasets (XOR/circle/spiral) + backprop + pygame viz. ~600 LOC shipped. |
| **v2 — nn-v2-future** | `ready-for-agent` | `.scratch/nn-v2-future/README.md` | MountainCar-v0 + REINFORCE + decoder-only transformer from scratch. Adds gymnasium as only new dep. ~2200 LOC estimated. |
| **v3 — diffusion** | `not specced` | — | 3Blue1Brown cap. 10. Spec when v2 ships. |

## Versions must sequence

- **v1 first** ✅ — validates backprop primitives, softmax+CE gradient,
  visualizer framework. Shipped.
- **v2 builds alongside v1** — `nn.py`, `datasets.py`, `tests/test_nn.py`
  stay frozen. New `layers.py` + `modules.py` + `optim.py` + `envs/` +
  `data/` + `transformer/` provide the math. `visualizer.py` v1 surface
  gets refactored in ticket 08 to accept a `metrics` dict (v1 visual
  mode preserved via adapter in v1's `__main__`).
- **v3** depends on v2's transformer modules + interpretability tools.

## v2 ticket queue (next session)

11 tickets; **5 landed** (PRs #10, #11, #12, #13, #14, merged 2026-08-08):
01, 08, 02, 04, 05. Remaining 6 tickets, dependency graph:
```
03 ──�
06 ──┼─→ 07
    │
    └→ 09 ─→ 10 ─→ 11
```

- **03** REINFORCE trainer (needs 02, done)
- **06** transformer block + model stack (needs 04+05, both done)
- **07** transformer trainer (needs 06+04)
- **09** v2 visualizer panels (gym render + attention heatmap)
- **10** main.py registry + v2 keys
- **11** wrap-up: requirements + v0.2.0

All in `.scratch/nn-v2-future/issues/03-reinforce-trainer.md` etc.

## Conventions

- `ponytail:` comments mark deliberate simplifications and name the
  ceiling (e.g. "full-batch SGD; swap to minibatches if data grows").
- v1 architecture: default `[2, 8, 8, 3]` (covers all 3 datasets).
- v1 activation defaults: `relu` for hidden, `softmax` for output.
- v1 loss: cross-entropy (fused with softmax at the output layer for
  clean `δ = ŷ - y` gradient).
- PRs use `gh pr merge --squash --delete-branch --admin` (solo
  workflow, no human reviewer available).
- Reviews fan out in parallel via `delegate_task` (ponytail + Matt
  Pocock + correctness), but skip for docs-only changes (README, spec
  flips, ADRs).

## Pitfalls (real ones)

- **PR #14 critical bug catch (2026-08-08)**: in-place SGD updates
  of weights (`W -= dW` with lr=1.0) inside `backward()` mutate the
  weight matrix BEFORE the gradient flow computes `d_pre = grad @
  W.T`. Always compute `d_pre` (and any downstream gradient that
  uses the weight) FIRST, then mutate. Pattern: keep mutation at
  the END of backward(). Caught by upgrading FD test from
  `d_model=4` → `d_model=8` — smaller scale masked the bug.
- **PR #13 review catch (2026-08-08)**: test scope creep. Ticket 04
  said `test_modules.py` = LN + Residual only; PR duplicated AdamW
  tests in both `test_modules.py` and `test_optim.py`. Lesson: when
  ticket explicitly scopes a test file, respect it — duplicates make
  refactors painful.
- **PR #11 review catch (2026-08-08)**: silent fallback in
  `set_panel()` hid caller bugs. Now raises `ValueError` on unknown
  panel name. Pattern: silent fallback over raising almost always
  means caller typo goes undetected — raise, don't guess.
- **PR #10 review catch**: `Softmax.forward` recomputed probabilities
  on every `backward()` call. Cached in `self.a`. Pattern: if a
  backward() needs a forward output, cache it on forward, don't
  recompute.
- **PR #11 ticket wording drift**: ticket 08 said
  `update(nn_or_None, X_or_None, metrics, panel)` literally; the
  implemented `update(metrics, panel, *, nn=None, X=None, y=None)`
  satisfied the SPIRIT (flat metrics dict + panel selector) but
  violated the literal signature. Resolved by updating the ticket
  wording to match the simpler API — split render payload from
  metrics via kwargs. Lesson: ticket text can drift; when reviewer
  flags a literal vs spirit mismatch, fix the ticket, not the code.



- **Pygame won't open on the LXC** — display is on Franko's desktop.
- **Spiral with 3 classes is hard for a tiny arch** — `[2, 16, 16, 3]`
  + 20k epochs needed (test ticket 03 documents this).
- **Random seed**: the constructor seeds once with `42` for
  reproducibility across runs.
- **Click hit-test off-by-264px bug** (caught + fixed by Matt Pocock
  review in PR #6): `input_node_pos` was using `n_total` instead of
  `n_nodes` for the y-formula. Fix: hit-test lives on `Visualizer`,
  reuses `_calc_y` with correct `n_nodes`. Lesson: any click/mouse
  handler that mirrors visualizer layout math **must** call the same
  helpers, not re-implement them.
- **GitHub Telegram image cache**: same path = same cached image even
  if content changed. Workaround: rename file with different suffix
  (e.g. `-v3.png`) when sending "updated" screenshots through chat.
- **v1 tests must keep passing** during v2 — see ADR-0003 and ticket
  01 acceptance criteria.

## Related skills (in `~/.hermes/skills/`)

- `neural-networks-foundation` — conceptual reference for the math
  (load chapters 5-8 when starting v2)
- `research-then-code-then-visualize` — workflow we used for v1
- `setup-matt-pocock-skills` (already applied) — generated this
  repo's `docs/agents/` config
- `feature-loop-single-actor` — adapted feature-loop for solo repos
  (created this session; no reviewers, branch protection with admin
  bypass)
- `ponytail` — over-engineering discipline (always-on)
- `caveman` — terse prose (always-on)
