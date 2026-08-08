---
status: implemented
---

# NN Visualizer — Spec

## Problem Statement

The user is re-learning neural networks from scratch (using 3Blue1Brown's
chapter series as the conceptual guide). They want a **single
self-contained project** that:

- Implements a feedforward neural network with backprop **by hand** in
  numpy — no PyTorch, no TF, no autograd library — so the gradient
  computation is visible, not magical.
- Visualizes the network **training live** in pygame: the decision
  boundary deforms as weights update, and the weight graph's color /
  thickness shifts in sync.
- Covers three toy datasets of escalating difficulty: **XOR** (the
  canonical "linear models can't do this" test), **circle** (inner vs
  outer ring, needs 1 hidden layer minimum), and **spiral** (3-class
  intertwined spirals, needs deeper / wider net).

The user already explored this space years ago in the GitHub repo
`Franko12345/ai` (2023) using a genetic-algorithm approach with
pygame. That repo is **reference only** — not a dependency. The current
project is backprop-first (matches 3Blue1Brown chapter 4), not GA-first.

## Solution

A 4-file Python project:

- `nn.py` — `NeuralNetwork` class with `forward`, `backward`, `fit`.
  NumPy only. Reproducible via `np.random.default_rng(42)`.
- `datasets.py` — three generators (`xor`, `circle`, `spiral`) returning
  `(X, y)` with `X` normalized to `[-1, 1]²`.
- `visualizer.py` — pygame window with two side-by-side panels:
  - **Left (~50%)**: decision boundary rendered as a colored grid
    (predicted class per pixel).
  - **Right (~50%)**: weight graph — circles for nodes (filled by
    activation, black outline), lines for connections (blue for
    negative weights, red for positive, thickness by |weight|).
  - **Top bar**: current epoch, loss, accuracy, dataset name.
- `main.py` — CLI entrypoint with `--dataset`, `--epochs`, `--lr`,
  `--arch`. Drives `train_n_epochs_per_frame=10` between pygame frames
  so training feels live at 60fps.

Headless validation via `tests/test_nn.py` (assertions on accuracy
per dataset, runs on the LXC). Visualizer is smoke-tested on LXC with
`SDL_VIDEODRIVER=dummy`, visual correctness verified by the user on
their desktop.

## User Stories

1. As a learner, I want to see a neural network's decision boundary
   deform in real time, so that I can connect gradient descent to
   geometric intuition.
2. As a learner, I want to see the weights of the network update
   visibly (color + thickness of the connection lines), so that I
   understand what each weight is doing.
3. As a learner, I want to switch between XOR / circle / spiral
   datasets with a single keystroke, so that I can compare how the
   network handles different problem geometries.
4. As a learner, I want to pause / resume training, so that I can
   freeze a frame and inspect what the network is doing.
5. As a learner, I want to reset the network (re-init weights, restart
   training), so that I can compare different runs on the same dataset.
6. As a learner, I want to adjust the learning rate with `+` / `-`,
   so that I can feel how lr affects convergence (too small = glacial,
   too big = bouncing).
7. As a learner, I want a fast-forward mode (e.g. `F`) that trains
   many epochs per frame, so that I can skip the boring initial phase
   to see what a converged network looks like.
8. As a learner, I want to click on an input node to toggle its value
   between 0 and 1, so that I can see how the network reacts to
   specific inputs (forward pass visualized).
9. As a learner, I want the visualizer to fit on a standard 1366×768
   laptop screen, so that I can run it without window scrolling.
10. As a learner, I want the project to run on the headless LXC for
    tests, so that I don't need a display to validate the math.
11. As a learner, I want the dataset to be drawn over the decision
    boundary, so that I can see the training points relative to the
    predicted classes.
12. As a learner, I want to see the loss and accuracy update live in
    the top bar, so that I know if training is working.
13. As a learner, I want the activation values of hidden neurons to
    be visible (node fill color), so that I can see which neurons are
    firing for which inputs.
14. As a learner, I want the output layer's class probabilities to be
    visible (e.g. node fill or label), so that I can see how confident
    the network is.
15. As a developer, I want the project to have a single seam
    (`NeuralNetwork` class) that is testable without pygame, so that
    I can validate the math in <1 second.
16. As a developer, I want no external dependencies beyond numpy and
    pygame, so that the project boots on a fresh Python install with
    one `pip install`.
17. As a developer, I want each component (`nn.py`, `datasets.py`,
    `visualizer.py`, `main.py`) to be <200 lines, so that the whole
    project is readable in one sitting.
18. As a developer, I want deterministic results (`np.random.seed=42`
    in the network constructor), so that visualizer runs are
    reproducible for bug reports.

## Implementation Decisions

### Architecture

- **Single seam**: `NeuralNetwork` class in `nn.py`. Visualizer and
  main loop are out-of-seam (validated by headless smoke test only).
- **Default architecture**: `[2, 8, 8, 3]` — covers all 3 datasets,
  small enough to draw legibly.
- **Activations**: ReLU for hidden layers, softmax for output.
- **Loss**: cross-entropy with the fused softmax gradient
  (`δ = ŷ - y`) at the output layer.
- **Initialization**: Xavier (`sqrt(2 / (fan_in + fan_out))`).

### Training loop

- **Full-batch SGD** for clarity (ponytail ceiling: swap to minibatch
  if datasets grow past 10k samples).
- Default `lr=0.05`, `epochs=2000`. Visualizer trains
  `10 epochs / frame` at 60fps → ~600 epochs/sec.
- `fit(X, Y, epochs, lr)` returns `losses: list[float]` for plotting.

### Datasets

- All return `(X, y)` with `X.shape = (N, 2)`, `y.shape = (N,)`
  integer labels, `X` normalized to `[-1, 1]²`.
- `xor(n=200)` — 2 classes, balanced.
- `circle(n=200)` — 2 classes (inner radius 0.3, outer radius 0.8,
  noise ±0.1).
- `spiral(n=200)` — 3 classes, 3 spirals of ~67 samples each.

### Visualizer

- **Resolution**: 1280×720 (fits 1366×768 screens with room for OS
  chrome).
- **Layout**: top bar (60px) for info, body split 50/50 horizontally.
- **Decision boundary** rendered at low resolution (e.g. 40×40 grid,
  upscaled) for speed — 1600 pixels/frame is well under budget.
- **Weight graph**: nodes drawn as `pygame.draw.circle` with radius
  18px, fill = activation clamped to [0, 255], outline = black 2px.
  Connections drawn as lines, color by sign of weight, thickness
  by `min(5, int(abs(weight) * 2))` or similar. No labels on weights
  or biases.
- **Dataset points**: drawn over the decision boundary as small
  circles (radius 4px), color = class (3 distinct colors).

### Interactivity (combined: keys + click)

- `1` / `2` / `3` — switch dataset (xor / circle / spiral)
- `SPACE` — pause / resume training
- `R` — reset weights, restart training on current dataset
- `+` / `-` — increase / decrease learning rate (clamped 0.001–1.0)
- `F` — toggle fast-forward (10 → 200 epochs/frame)
- `ESC` — quit
- **Click on input node** — toggle that input between 0 and 1,
  re-run forward pass (pauses training automatically while held)

### Determinism

- `np.random.default_rng(42)` seeded once in `NeuralNetwork.__init__`.
- Visualizer's per-frame training is deterministic given the seed
  and the same `epochs_per_frame`.

## Testing Decisions

- **The single testable seam**: `NeuralNetwork` (`forward`, `backward`,
  `fit`).
- **What makes a good test**: asserts behavior on external observables
  (accuracy on a fixed dataset after N epochs of training), not on
  intermediate gradient values (which are implementation-detail).
- **Modules tested**: `nn.py` only.
- **Prior art**: none in-repo (greenfield project). Standard pattern:
  a single `tests/test_nn.py` file with one `assert` per dataset.
- **Headless test runs**: `python3 tests/test_nn.py`. Exits 0 on
  pass, non-zero on fail.
- **No pytest** — keep the test runner to a single file with
  `assert ... else sys.exit(1)`, matching the project's ponytail /
  small-diff stance.

### Test cases

- `test_xor`: train 2000 epochs on XOR, assert accuracy ≥ 0.95.
- `test_circle`: train 2000 epochs on circle, assert accuracy ≥ 0.90.
- `test_spiral`: train 3000 epochs on spiral with arch `[2, 16, 16, 3]`,
  assert accuracy ≥ 0.85.

## Out of Scope

- Modern architectures (transformers, convnets, RNNs).
- GPU acceleration.
- Production training loops (LR schedulers, early stopping,
  checkpointing beyond in-memory).
- Real-scale datasets (ImageNet, language).
- Minibatch SGD, momentum, Adam.
- Saving / loading model weights.
- Saving video / GIF of training.
- Network architecture search.
- Per-class confusion matrix or other metrics.
- Web UI (this is a desktop pygame app).

## Further Notes

### Why numpy-only (ADR-0002)

The project's reason for existing is to **see backprop work**. PyTorch /
TF would hide the gradient behind autograd. Each component must be
~30 lines and read top-to-bottom.

### Why headless-first (ADR-0003)

The dev environment is a headless Proxmox LXC. Pygame windows won't
open here. The math must validate before any visualization work, and
the visualizer gets smoke-tested with `SDL_VIDEODRIVER=dummy`.

### Why a single seam (ponytail / YAGNI)

Multiple testable surfaces = more code, more fixtures, more failure
modes. The `NeuralNetwork` class is the only place where correctness
matters at the math level. Everything else (datasets, visualizer,
main loop) is plumbing that gets smoke-tested at the integration
boundary.

### Visual style summary

- **Background**: black / very dark gray (`(15, 15, 20)`).
- **Decision boundary**: 3 class colors chosen for max contrast on
  dark bg (e.g. teal, orange, magenta). Dimmed slightly so dataset
  points pop on top.
- **Connection lines**: blue `(60, 120, 255)` for negative weights,
  red `(255, 80, 80)` for positive. Alpha or thickness modulated by
  |weight|.
- **Nodes**: black outline (2px), fill = `(activation * 255, activation
  * 255, activation * 255)` (white at full activation, black at zero)
  — gives a clear "which neuron is firing" signal.
- **Text**: white on dark bg, monospace for numbers (e.g.
  `pygame.font.SysFont('consolas', 16)`).

### Reference (not dependency)

The repo `github.com/Franko12345/ai` (2023) used a genetic-algorithm
approach with similar visualizer code (node/connection rendering,
`calcX`/`calcY` layout helpers, click-to-toggle input). That repo is
**reference only** for visual style — this project does not import or
fork any of its code. The current implementation is backprop-first
and written from scratch.