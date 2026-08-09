# neural-network — Context

> The one doc loaded on every session that touches this repo. Keep
> it current.

## What this is

A from-scratch educational deep learning library in **numpy**, with a
**pygame** visualizer that animates everything live: decision boundary,
weight graph, MountainCar environment, multi-head attention heatmap.

**v1 (shipped v0.1.0)** — feedforward NN on three toy datasets
(XOR, circle, spiral). Validates backprop primitives + visualizer.

**v2 (shipped v0.2.0)** — MountainCar-v0 with REINFORCE (constant
baseline) + decoder-only transformer from scratch (~600K params,
char-level Shakespeare). Built alongside v1 — v1 files stayed frozen
throughout (0 diff lines per merge on nn.py / datasets.py /
tests/test_nn.py).

Releases:
- https://github.com/Franko12345/neural-network/releases/tag/v0.1.0
- https://github.com/Franko12345/neural-network/releases/tag/v0.2.0

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
├── visualizer.py                 # pygame render (v1 + v2 panels)
├── main.py                       # CLI loop with task registry (5 tasks)
├── sanity.py                     # v1 one-shot smoke check for nn.py
├── layers.py                     # v2 building blocks (Linear, activations, LayerNorm)
├── modules.py                    # v2 Residual wrapper
├── optim.py                      # v2 AdamW (decoupled weight decay, bias correction)
├── envs/                         # v2 MountainCar wrapper + rollout batch collector
│   ├── __init__.py
│   ├── mountaincar.py            #   gymnasium MountainCar-v0 wrapper
│   └── rollout.py                #   RolloutBatch + rollout(env, policy_fn)
├── transformer/                  # v2 attention, block, model, train
│   ├── __init__.py
│   ├── attention.py              #   MultiHeadAttention + strict-causal mask
│   ├── block.py                  #   pre-norm transformer block
│   ├── embed.py                  #   Embedding + sinusoidal PositionalEncoding
│   ├── model.py                  #   Transformer stack + LM head + autoregressive sample()
│   └── train.py                  #   Trainer (AdamW + checkpoint + auto-load)
├── data/                         # v2 char-level text loader
│   ├── __init__.py
│   ├── text.py                   #   load_text(path) -> (uint8 ids, vocab_size=256)
│   └── shakespeare.txt           #   bundled clean excerpt (~1.5KB)
├── train_rl.py                   # v2 REINFORCE trainer with constant baseline
├── tests/                        # 12 test files, 67 asserts total
│   ├── test_nn.py                #   v1: 3 datasets, acc thresholds (FROZEN)
│   ├── test_layers.py            #   v2: Linear/ReLU/Tanh/Sigmoid/Softmax + FD
│   ├── test_modules.py           #   v2: LayerNorm + Residual
│   ├── test_optim.py             #   v2: AdamW (bias correction, decoupled WD)
│   ├── test_attention.py         #   v2: MHA causal mask + multi-head split + FD
│   ├── test_rollout.py           #   v2: MountainCar env + rollout shape
│   ├── test_transformer.py       #   v2: Transformer forward/backward + sample + loss drop
│   ├── test_rl.py                #   v2: REINFORCE gradient + baseline invariance
│   ├── test_train.py             #   v2: AdamW trainer + checkpoint round-trip + auto-load
│   ├── test_visualizer.py        #   v2: refactored update(metrics, panel) + set_panel
│   ├── test_visualizer_v2.py     #   v2: gym_render + attention panels
│   └── test_main.py              #   v2: task registry + 5 keys under dummy SDL
├── docs/
│   ├── agents/                   # engineering-skill config
│   └── adr/                      # ADRs (NNNN-kebab-title.md) — see list below
├── requirements.txt              # numpy + pygame-ce + gymnasium
├── AGENTS.md                     # agent entry point
├── CONTEXT.md                    # this file
└── .scratch/
    ├── nn-visualizer/            # v1 spec + 6 tickets (all implemented)
    │   ├── README.md             #   status: implemented
    │   └── issues/               #   01-06 all closed
    ├── nn-v2-future/             # v2 spec + 11 tickets (all implemented)
    │   ├── README.md             #   status: implemented
    │   └── issues/               #   01-11 all closed
    └── screenshots/              # README GIFs + asset-generation scripts
        ├── xor.gif               #   v1: XOR decision boundary
        ├── circle.gif            #   v1: circle decision boundary
        ├── spiral.gif            #   v1: spiral decision boundary
        ├── spiral_deep.gif       #   v1: spiral with [2, 32, 32, 32, 3]
        ├── mountaincar.gif       #   v2: REINFORCE on MountainCar
        ├── transformer_train.gif #   v2: attention heatmap during training
        ├── transformer_sample.gif#   v2: attention during autoregressive sampling
        ├── _make_gifs.py         #   regenerate v1 GIFs
        └── _make_v2_gifs.py      #   regenerate v2 GIFs
```

## Status

**v1 shipped** (`v0.1.0` tag, GitHub release published). Headless tests
pass in ~3.6s. Visualizer runs at 60fps on a desktop with display.

**v2 shipped** (`v0.2.0` tag). 11 tickets closed in 4 rounds of
parallel worktrees. v1 stayed frozen throughout v2 (verified via
`git diff main -- nn.py datasets.py tests/test_nn.py` = 0 lines per
merge).

## How to run

```bash
pip install -r requirements.txt          # v1: numpy + pygame-ce; v2 adds gymnasium

# Run any test file (all 12, 67 asserts total, ~5s combined):
python3 -m tests.test_nn                 # v1 feedforward (~3.7s)
python3 -m tests.test_layers             # v2 Linear/ReLU/Tanh/Sigmoid/Softmax + FD
python3 -m tests.test_modules            # v2 LayerNorm + Residual
python3 -m tests.test_optim              # v2 AdamW
python3 -m tests.test_attention          # v2 MultiHeadAttention
python3 -m tests.test_rollout            # v2 MountainCar env + rollout
python3 -m tests.test_transformer        # v2 Transformer forward/backward + sample
python3 -m tests.test_rl                 # v2 REINFORCE trainer
python3 -m tests.test_train              # v2 AdamW trainer + checkpoint
python3 -m tests.test_visualizer         # v2 refactored visualizer
python3 -m tests.test_visualizer_v2      # v2 gym_render + attention panels
python3 -m tests.test_main               # v2 task registry

# Run the visualizer (needs a real display):
python3 main.py                          # default task = xor
python3 main.py --task mountaincar        # REINFORCE
python3 main.py --task transformer        # decoder-only transformer

# Headless smoke (no display required):
SDL_VIDEODRIVER=dummy python3 main.py --task xor --epochs 5
SDL_VIDEODRIVER=dummy python3 main.py --task mountaincar
SDL_VIDEODRIVER=dummy python3 main.py --task transformer
```

See `README.md` for key bindings, CLI options, and visual demo GIFs.

## Roadmap

| Version | Status | Spec | Scope |
|---|---|---|---|
| **v1 — nn-visualizer** | `implemented` (v0.1.0 released) | `.scratch/nn-visualizer/README.md` | Toy datasets (XOR/circle/spiral) + backprop + pygame viz. ~600 LOC shipped. |
| **v2 — nn-v2-future** | `implemented` (v0.2.0) | `.scratch/nn-v2-future/README.md` | MountainCar-v0 + REINFORCE + decoder-only transformer from scratch. Adds gymnasium as only new dep. ~2200 LOC estimated. |
| **v3 — diffusion** | `not specced` | — | 3Blue1Brown ch. 10. Spec when v2 ships — now unblocked since v0.2.0 is released. |

## Versions must sequence

- **v1 first** ✅ — validates backprop primitives, softmax+CE gradient,
  visualizer framework. Shipped.
- **v2 builds alongside v1** — `nn.py`, `datasets.py`, `tests/test_nn.py`
  stay frozen. New `layers.py` + `modules.py` + `optim.py` + `envs/` +
  `data/` + `transformer/` provide the math. `visualizer.py` v1 surface
  gets refactored in ticket 08 to accept a `metrics` dict (v1 visual
  mode preserved via adapter in v1's `__main__`).
- **v3** depends on v2's transformer modules + interpretability tools.

## v2 ticket queue (final)

11 tickets; **all 11 shipped** (PRs #10-#20, merged 2026-08-08/09):
01, 08, 02, 04, 05, 03, 06, 07, 09, 10, 11. `v0.2.0` tag.

Final commit on main tagged `v0.2.0`. See [v0.2.0 release notes](
https://github.com/Franko12345/neural-network/releases/tag/v0.2.0)
for ticket-by-ticket summary.

| # | Ticket | Status | PR |
|---|---|---|---|
| 01 | Extract `layers.py` building blocks | implemented | #10 |
| 02 | MountainCar env wrapper + rollout batch | implemented | #12 |
| 03 | REINFORCE trainer with baseline | implemented | #15 |
| 04 | LayerNorm + Residual + AdamW | implemented | #13 |
| 05 | Multi-head attention + causal mask | implemented | #14 |
| 06 | Transformer block + model stack | implemented | #16 |
| 07 | Transformer trainer + AdamW + checkpointing | implemented | #17 |
| 08 | Visualizer refactor: metrics dict + tabs | implemented | #11 |
| 09 | Visualizer v2 panels (gym + attention) | implemented | #18 |
| 10 | `main.py` registry + v2 keys | implemented | #19 |
| 11 | Wrap-up: requirements + v0.2.0 | implemented | #20 |

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

## Architecture Decision Records (ADRs)

Five ADRs in `docs/adr/` capture the project history:

| ADR | Decision |
|---|---|
| 0001 | Record architecture decisions (Nygard template) |
| 0002 | Numpy-only, no PyTorch / TF |
| 0003 | Headless-first validation, pygame display on user's machine |
| 0004 | Single-context domain docs |
| 0005 | v2 builds alongside v1; v1 modules are frozen |

See `docs/adr/README.md` for the index. ADRs are immutable — superseded
ones get a `Status: Superseded by NNNN` line.

## Pitfalls (real ones)

- **PR #17 critical bug catch (2026-08-09)**: DOUBLE-UPDATE in trainer.
  `model.backward(d_logits, lr=1.0)` in-place SGD-updates Linear/
  LayerNorm weights; then `AdamW.step()` applied a SECOND update.
  Every weight stepped twice per iteration. Same bug class as
  PR #15 W-order. Fix: `update=False` flag on Linear/LayerNorm/MHA/
  Block/Transformer.backward so trainer can compute gradients
  without mutating weights; AdamW is the sole updater. Pattern
  recurring across PR #14/#15/#16/#17: in-place SGD step +
  caller-managed optimizer = double-update trap.
- **PR #16 critical bug catch (2026-08-09)**: Linear.backward had
  TWO latent bugs that PR #14/15 partially exposed:
  1. W-order (same as PR #14 W_o): `return grad @ self.W.T` after
     `self.W -= self.dW` — used post-update W. PR #15 fixed for 2D
     inputs; PR #16 fixed for N-D.
  2. N-D support: forward() accepts `(B, T, fan_in)` via broadcasting,
     but backward assumed 2D. Hidden until a real N-D user
     (transformer block) tried to train. Fix: flatten leading dims
     in backward.
  Lesson: when a module's forward is broadcasting-friendly, its
  backward MUST mirror that via flatten/reshape. Pattern repeated
  across 3 PRs (W_o, Linear 2D, Linear N-D) — same root cause class.
- **PR #15 critical bug catch (2026-08-09)**: in-place SGD updates
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

## Session state (2026-08-09 close)

**Repo state:** v1 + v2 both shipped. `main` at tag `v0.2.0`.
12 test files, 67 asserts. 0 diff lines on v1 frozen files across all
v2 merges.

**Documentation state:** AGENTS.md, README.md, CONTEXT.md current.
5 → 6 ADRs (0001-0006). All spec/issue frontmatter flipped to
`status: implemented`. Release notes published on GitHub for v0.1.0
and v0.2.0.

**Next session entry point:** the user may pick up v3 (diffusion)
whenever they're ready. Spec doesn't exist yet — first step is to
draft `.scratch/nn-v3-diffusion/README.md` (3Blue1Brown ch. 10).
Alternatively, the project may be considered "done" at v0.2.0 — no
v3 work required.

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
