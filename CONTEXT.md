# neural-network — Context

> The one doc loaded on every session that touches this repo. Keep
> it current.

## What this is

A from-scratch feedforward neural network in **numpy**, with a
**pygame** visualizer that animates the decision boundary and the
weight graph during training. Three toy datasets: XOR, circle, spiral.

## Goals

- **Educational**: see backprop actually run, weight values shift,
  decision boundary deform.
- **Minimal**: no PyTorch, no TF. Each component reads top-to-bottom.
- **Reproducible**: fixed seed (`np.random.default_rng(42)`).

## Non-goals

- Real-scale datasets (ImageNet, language).
- GPU acceleration.
- Modern architectures (transformers, convnets).
- Production training loops (no LR schedulers, no early stopping,
  no checkpointing beyond in-memory).

## Stack

- Python 3.11+
- numpy 2.x
- pygame-ce 2.5+

All three verified present in the LXC dev env. No `pip install`
needed for the core path.

## File layout

```
neural-network/
├── CONTEXT.md                    # this file
├── AGENTS.md                     # agent entry point
├── nn.py                         # core math (forward, backward, fit)
├── datasets.py                   # XOR / circle / spiral generators
├── visualizer.py                 # pygame render (boundary + weight graph)
├── main.py                       # CLI + loop
├── tests/
│   └── test_nn.py                # headless validation
├── requirements.txt              # numpy + pygame-ce pins
├── README.md                     # user-facing docs
├── docs/
│   ├── agents/                   # engineering-skill config
│   └── adr/                      # ADRs (NNNN-kebab-title.md)
└── .scratch/                     # local-markdown issues
```

## Development order (headless-first)

1. `nn.py` core math
2. `tests/test_nn.py` — assertions on XOR / circle / spiral
3. `datasets.py` — generators
4. `visualizer.py` — pygame render
5. `main.py` — loop
6. `SDL_VIDEODRIVER=dummy python3 main.py` — smoke test on LXC

## Conventions

- `ponytail:` comments mark deliberate simplifications and name the
  ceiling (e.g. "full-batch SGD; swap to minibatches if data grows").
- Architecture: default `[2, 8, 8, 3]` (covers all 3 datasets).
- Activation defaults: `relu` for hidden, `softmax` for output.
- Loss: cross-entropy (fused with softmax at the output layer for
  clean `δ = ŷ - y` gradient).

## Pitfalls (real ones)

- **Pygame won't open on the LXC** — display is on Franko's desktop.
- **Spiral with 3 classes is hard for a tiny arch** — `[2, 16, 16, 3]`
  may be needed; covered in tests.
- **Random seed**: the constructor seeds once with `42` for
  reproducibility across runs.

## Status

**Pre-implementation.** `nn.py` exists as a draft awaiting cleanup.
No datasets, tests, visualizer, or main loop yet.

## Related skills (in `~/.hermes/skills/`)

- `neural-networks-foundation` — conceptual reference for the math.
- `research-then-code-then-visualize` — workflow we used to plan this.
- `setup-matt-pocock-skills` (already applied) — generated this
  repo's `docs/agents/` config.