# neural-network

Feedforward neural network from scratch in **numpy**, with a **pygame**
visualizer that animates the decision boundary and the weight graph
during training. Three toy datasets of escalating difficulty: **XOR**,
**circle**, **spiral**.

The project exists to make backprop **visible**: every component is
~100 lines, weights update in real time on screen, and there are no
deep-learning frameworks hiding the gradient behind autograd.

## Visual

The visualizer shows two panels side-by-side:

- **Left**: a 40×40 decision boundary grid. Each cell is colored by
  the predicted class for that input. Dataset points are overlaid as
  small circles.
- **Right**: a weight graph. Nodes are circles filled by activation
  (white = full, black = zero), connections are lines **blue for
  negative weight, red for positive**, thickness by |weight|.

## Quick start

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Verify the math (headless, <10s)
python3 -m tests.test_nn

# 3. Run the visualizer (needs a display)
python3 main.py --dataset xor

# 4. Try other datasets
python3 main.py --dataset circle
python3 main.py --dataset spiral
```

## Key bindings

| Key       | Action                                                    |
|-----------|-----------------------------------------------------------|
| `1` `2` `3` | Swap dataset (xor / circle / spiral). Resets weights.   |
| `SPACE`   | Pause / resume training.                                  |
| `R`       | Reset weights (same seed), restart training.              |
| `+` / `-` | Adjust learning rate (1.5× steps, clamped 0.001–1.0).    |
| `F`       | Toggle fast-forward (10 → 200 epochs/frame).              |
| `ESC`     | Quit.                                                     |

**Click** on an input node to toggle its value (0 ↔ 1) and pause
training so you can see the network react.

## CLI options

```
python3 main.py --dataset xor|circle|spiral
                 --epochs 2000          # not enforced; training runs continuously
                 --lr 0.05              # initial learning rate
                 --arch 2,8,8,3         # layer sizes (default matches spec)
```

## Headless smoke test

If you can't open a display (CI, LXC, SSH without X11):

```bash
SDL_VIDEODRIVER=dummy python3 main.py --dataset xor --epochs 5
```

This initializes pygame under the `dummy` SDL driver, runs the loop
for a few seconds, and exits 0. Use this to confirm the project works
on a fresh machine.

## File layout

```
neural-network/
├── nn.py              # NeuralNetwork class (numpy-only, ~115 LOC)
├── datasets.py        # xor / circle / spiral generators (~70 LOC)
├── visualizer.py      # pygame render — decision boundary + weight graph
├── main.py            # CLI loop with key + click controls
├── sanity.py          # one-shot smoke check for nn.py
├── tests/
│   └── test_nn.py     # stdlib-only headless validation
├── docs/
│   ├── agents/        # engineering-skill config
│   └── adr/           # architecture decision records
├── .scratch/          # local-markdown issues (see docs/agents/issue-tracker.md)
├── requirements.txt
├── AGENTS.md          # agent entry point
└── CONTEXT.md         # project context (loaded on every session)
```

## Architecture

| Module          | Responsibility                                        |
|-----------------|-------------------------------------------------------|
| `nn.py`         | Single seam: `NeuralNetwork.forward/backward/fit`. No deps beyond numpy. |
| `datasets.py`   | Three 2D generators, all return `(X, y)` with `X ∈ [-1, 1]²`. |
| `visualizer.py` | Render-only. Caller drives training and calls `update()` per frame. |
| `main.py`       | Glue: argparse, 60fps loop, key/mouse handlers.        |
| `tests/test_nn.py` | Stdlib-only validation: 3 datasets, acc thresholds.   |

## How backprop works here

Cross-entropy + softmax fuses into a clean `δ = ŷ - y` gradient at
the output layer. Hidden layers propagate via the chain rule:
`prev.δ = (Wᵀ · layer.δ) ⊙ σ'(prev.z)`. Xavier init keeps activations
sane at start. Full-batch SGD — swap to minibatch if you scale past
~10k samples.

## Limitations

- Full-batch SGD, no momentum, no Adam. Fine for toy datasets, slow
  for anything bigger.
- Cross-entropy loss only. Regressão com sigmoid+CE tem bug latente
  (TODO no código, não exercitado).
- Reproducible: seed `42` hardcoded in `NeuralNetwork.__init__`. Add
  a `seed=` parameter if you need to vary runs.
- Visualizer is fixed 1280×720. Resize requires code change.
- No save/load weights, no checkpointing, no video capture.

## What's next

- **v2** — `MountainCar-v0` with REINFORCE + a decoder-only transformer
  from scratch. Spec: `.scratch/nn-v2-future/README.md` (draft).

## See also

- `CONTEXT.md` — project context, loaded every session
- `docs/agents/issue-tracker.md` — how issues are tracked
- `docs/adr/` — architecture decision records
- `neural-networks-foundation` skill — chapter-by-chapter notes from
  3Blue1Brown + Coding Train