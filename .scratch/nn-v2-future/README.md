---
status: draft
---

# NN v2 — Spec Futura (backprop + transformers)

## Problem Statement

O `nn-visualizer` (spec pronta, `ready-for-agent`) cobre o **hello world**
de redes neurais: toy datasets estáticos, backprop do zero, visualização
em pygame. É o suficiente pra internalizar 3Blue1Brown capítulos 1-4.

Mas você já foi além disso anos atrás no `Franko12345/ai` — controlou
**MountainCar-v0** com redes neurais (GA na época, com a função de
fitness multi-critério que você mesmo marcou pra revisar). A spec v1
não cobre isso. Pior: ela **nem chega perto** do que é o estado da arte
hoje — **transformers** (3Blue1Brown cap. 5-8) — que é o que de fato
move o mundo (LLMs, diffusion guidance, etc.).

Esta spec v2 é o **próximo passo natural** depois da v1 terminar:
mover do "rede feedforward de 2 camadas vendo pontos 2D" para
"rede controlando ambiente real + modelo transformer completo".

## Solution

### Parte A — MountainCar-v0 com backprop (substitui o GA antigo)

Reaproveita `nn.py` e `visualizer.py` da v1. Adiciona:

- `envs/mountaincar.py` — wrapper do `gymnasium` (sucessor do `gym`)
  que normaliza observações para `[-1, 1]²` (position, velocity).
- `train_rl.py` — loop de **REINFORCE** (policy gradient) escrito do
  zero em numpy. Sem stable-baselines, sem torch.
- `datasets.py` ganha `mountaincar_rollout(n_episodes)` que gera
  trajetórias e retorna `(states, actions, rewards)` pra treinar.
- Visualizer da v1 ganha um **3º painel**: o ambiente gymnasium
  renderizado ao vivo enquanto a rede joga, lado a lado com o
  grafo de pesos. Teclas `1/2/3` (feedforward), `4` (mountaincar).

### Parte B — Transformer from scratch (3Blue1Brown cap. 6-7)

Implementa o **transformer decoder-only** usado em GPTs, do zero, em
numpy. Sem torch. Sem huggingface.

Componentes:

- `transformer/embed.py` — token embedding + positional encoding
  sinusoidal.
- `transformer/attention.py` — self-attention com **causal mask**,
  scaled dot-product, multi-head. Inclui **softmax + cross-entropy**
  fused gradient (igual à v1).
- `transformer/block.py` — transformer block: attention + feedforward
  com residual + layer norm.
- `transformer/model.py` — stack de N blocos, lm head, sampling loop
  (argmax / temperature / top-k).
- `transformer/train.py` — trainer com **AdamW** (subindo o nível de
  optimizer — SGD puro é inviável pra transformer).

Dataset: **texto** (ex: coletar livros do Project Gutenberg, ou um
`.txt` que o usuário fornece). Tokenização: **character-level** (evita
dependência de BPE/tokenizer externo). Vocabulário: ~256 (ASCII).

Visualização específica do transformer:

- Attention maps head-by-head (heatmap de pesos de atenção por
  camada/cabeça) — o destaque visual.
- Loss curve por step.
- Sampling: digitar um prompt, ver a rede continuar token a token.
- Teclas `5` (transformer), `6` (attention heatmap por head).

## User Stories

### Parte A (RL)

1. As a learner, I want to see the MountainCar agent **learn to swing
   up the hill** in real time, so that I can connect policy gradient
   to a tangible task.
2. As a learner, I want the gymnasium render side-by-side with the
   weight graph, so that I can correlate "what the network is doing"
   with "what the car is doing".
3. As a learner, I want to see the **policy distribution** per state
   (probabilities of left / no-op / right), so that I understand what
   a stochastic policy looks like.
4. As a learner, I want REINFORCE implemented from scratch, so that
   the gradient through the sampling step is visible.
5. As a developer, I want `gymnasium` as the only new dep, so that
   the project stays minimal.
6. As a developer, I want the existing v1 tests to keep passing
   unchanged, so that the v2 doesn't regress v1.

### Parte B (transformer)

7. As a learner, I want to see the **attention pattern** shift during
   training, so that I can connect chapter 7 ("attention, step-by-step")
   to something happening on screen.
8. As a learner, I want to **type a prompt and watch the transformer
   complete it** token-by-token, so that I can feel what generation
   actually is (ch. 5).
9. As a learner, I want to **pause and inspect** any attention head
   in any layer, so that I can hunt for circuits (ch. 8).
10. As a learner, I want the **loss curve** to update live during
    training, so that I know if my hyperparameters are sane.
11. As a developer, I want **multi-head attention** to be its own
    module with its own forward/backward tests, so that the math is
    verified before stacking blocks.
12. As a developer, I want **layer norm** and **residual** to be
    testable modules with numerical gradient checks.
13. As a developer, I want **AdamW** to be its own module (not just
    `lr * grad`), because SGD alone is too slow for transformers.
14. As a developer, I want character-level tokenization so that the
    project has zero external tokenizer deps.
15. As a developer, I want a **small-enough model** (e.g. 4 layers,
    4 heads, 128 hidden) so that training fits in a few minutes on
    CPU and inference is interactive.
16. As a developer, I want the training to be **interruptible** (Ctrl-C
    saves current weights to `checkpoint.npz`) so that I can resume
    without losing progress.

## Implementation Decisions

### Reuse from v1

- `nn.py` — keep as-is, including `NeuralNetwork.forward/backward/fit`.
  Add `forward_return_activations()` flag if needed by attention viz.
- `visualizer.py` — keep layout (top bar + left/right panels); add
  3rd panel (gym render) and 4th panel (attention heatmap) via
  additional key bindings.
- `datasets.py` — keep xor/circle/spiral; add `mountaincar_rollout`.
- `tests/` — keep v1 tests; add `tests/test_attention.py` and
  `tests/test_transformer.py`.

### Part A — MountainCar + REINFORCE

- **Architecture**: `[2, 16, 3]` (position, velocity → softmax over
  3 actions). Tanh on hidden.
- **Policy**: stochastic (sample action from softmax probabilities).
- **REINFORCE**: `loss = -log π(a|s) * G_t`, where `G_t` is the
  discounted return from step `t`.
- **Discount**: `γ = 0.99`.
- **Reward shaping**: use gymnasium's default (negative per step,
  +100 on goal). No shaping.
- **Training**: 500 episodes per visualizer session; resume-able.

### Part B — Transformer

- **Architecture**:
  - `vocab_size = 256` (ASCII byte-level)
  - `d_model = 128`
  - `n_heads = 4`
  - `n_layers = 4`
  - `d_ff = 512` (4× d_model)
  - `max_seq_len = 128` (context window)
  - **~600K parameters** total (fits in CPU, trains in minutes).
- **Tokenization**: character-level. Each byte → one token.
- **Positional encoding**: sinusoidal (matches 3Blue1Brown viz).
- **Causal mask**: triangular `-inf` mask in pre-softmax scores.
- **Optimizer**: AdamW with `lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1`.
- **Training data**: a single `.txt` file (default: a Shakespeare
  play from Project Gutenberg, ~1MB). User can swap via `--data`.
- **Sampling**:
  - Temperature (default 1.0)
  - Top-k (default None = full softmax)
  - Max new tokens (default 200)
- **Checkpointing**: save `W_pe, W_e, blocks[*], W_head` to
  `checkpoint.npz` every 100 steps.

### Part B — Visualization

- **Attention heatmap** per head, per layer:
  - Click on a node in the transformer block diagram → see that
    head's attention pattern as a 128×128 heatmap.
  - Color: dark blue (low attention) → yellow → red (high).
- **Sampling console** at the bottom:
  - `T` to enter a prompt (type, Enter to submit).
  - Output streams character-by-character.
- **Loss curve** in top bar (mini-chart, last 200 steps).

### Interactivity (additions to v1 keys)

- `4` — switch to MountainCar panel
- `5` — switch to Transformer training panel
- `6` — switch to Attention heatmap inspector
- `T` — type a prompt (transformer)
- `P` — pause training (already in v1 as SPACE; reuse)

### Dependencies

- **New**: `gymnasium` (RL env), nothing else.
- **Still nothing else**: no torch, no huggingface, no tiktoken.

## Testing Decisions

### New seams to test

1. **Multi-head attention** — `tests/test_attention.py`:
   - Numerical gradient check vs analytical gradient (small inputs).
   - Causal mask verification: position `t` has zero attention weight
     on positions `> t`.
2. **Layer norm + residual** — backward pass correctness.
3. **AdamW** — bias correction, weight decay decoupled from gradient.
4. **Transformer block** — end-to-end forward + backward sanity
   (loss decreases after 100 steps on a tiny synthetic dataset).
5. **REINFORCE** — `tests/test_rl.py`:
   - Gradient of `-log π(a|s) * R` w.r.t. weights is correct
     (numerical check).

### What stays the same

- v1 tests (`test_xor`, `test_circle`, `test_spiral`) keep running
  unchanged. v2 must not regress v1.
- The single-seam philosophy still applies: each component has its
  own seam-tested module.

### Performance / scale guardrails

- "Loss decreases" test on transformer uses `vocab=16, d_model=32,
  2 layers, seq_len=16` — completes in <5s on CPU.
- Full-size model training is **not** tested in CI; it's a manual
  visualizer run.

## Out of Scope

- Training at scale (>10MB datasets, GPU, >10M params).
- Modern tokenizer (BPE, SentencePiece) — character-level only.
- Encoder-only or encoder-decoder transformers (BERT, T5) — decoder-
  only (GPT-style) is enough for this educational project.
- KV-cache optimization (recompute full attention each step; fine at
  seq_len=128).
- Distributed training, mixed precision, flash attention.
- RL algorithms beyond REINFORCE (no PPO, A2C, DQN).
- Saving video / GIF of training.
- Web UI.
- Production deployment / serving.

## Future Work (v3+)

- **Diffusion model from scratch** — natural next step after
  transformer; covers 3Blue1Brown cap. 10.
- **Mechanistic interpretability tools** — automated circuit
  discovery (matches cap. 8 deeper).
- **GPT-style pretraining on a real corpus** (Wikipedia, OpenWebText)
  — but only after v2 ships and validates the math.

## Further Notes

### Why split into v1 + v2

v1 (`nn-visualizer`) is **ready-for-agent** NOW and is small enough
to ship in a week. v2 is bigger (transformer alone is ~3× the LOC of
v1). Doing v1 first means:
- The math primitives (forward/backward, softmax+CE gradient) are
  battle-tested before transformer reuses them.
- The visualizer framework is in place; v2 just adds panels.
- We learn our lessons on a smaller system before scaling up.

### Why character-level tokenization

BPE / SentencePiece add a real dep (`tiktoken`, `sentencepiece`)
and a learning curve that distracts from the transformer itself.
For a 600K-param model on 1MB of text, char-level produces
**worse samples** than BPE — but the **math is identical**, and
that's the point. A future v3 can add BPE as a drop-in
`Tokenizer` interface with no model changes.

### Why AdamW and not just SGD

SGD with `lr=0.05` works for 2-layer feedforward on 200 points
(v1). It does **not** work for transformers — gradients explode /
vanish and the loss is flat. AdamW with decoupled weight decay is
the standard. Adding it here is necessary; the cost is one new
module with a clean API.

### Why MountainCar specifically

It's the canonical "control task that linear / shallow nets fail on".
Small (2D state, 3 actions, episodes in <200 steps), satisfying to
watch (car actually learns to swing), and historically the first
thing you tried (your `AIgym.py`). Familiar ground for you.

### Why decoder-only

Encoder-decoder (translation-style) requires a separate
training regime and paired data. Decoder-only is what GPTs use,
covers cap. 5-8, and is **simpler** (no cross-attention to the
encoder, just self-attention with a causal mask).

### Reference

- `Franko12345/ai/AIgym.py` (2023) — same MountainCar-v0 setup,
  but with GA. We replace GA with REINFORCE. Visualization style
  follows the dark-background spec from v1 (not the salmon
  background of the old code).
- `neural-networks-foundation` skill — chapters 5-8 cover exactly
  the concepts this spec implements.
- 3Blue1Brown chapters 5-8 — visual reference for attention maps
  and "how LLMs store facts".