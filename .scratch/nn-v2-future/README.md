---
status: implemented
---

# NN v2 — MountainCar-v0 + Transformer (revisão estrutural)

## Problem Statement

O `nn-visualizer` (v1, shippada em `v0.1.0`) cobre o **hello world** de
redes neurais: toy datasets estáticos, backprop do zero, visualização
em pygame. É o suficiente pra internalizar 3Blue1Brown capítulos 1-4.

Mas você já foi além disso anos atrás no `Franko12345/ai` — controlou
**MountainCar-v0** com redes neurais (GA na época, com a função de
fitness multi-critério que você mesmo marcou pra revisar). A v1 não
cobre isso. Pior: ela **nem chega perto** do que é o estado da arte
hoje — **transformers** (3Blue1Brown cap. 5-8) — que é o que de fato
move o mundo (LLMs, diffusion guidance, etc.).

Esta spec v2 é o **próximo passo natural** depois da v1: mover do
"rede feedforward de 2 camadas vendo pontos 2D" para "rede
controlando ambiente real + modelo transformer completo".

## Solution

### Estrutura proposta (vs. v1)

```
neural-network/
├── nn.py                  # v1, frozen. NeuralNetwork class.
├── visualizer.py          # v1, refactored: accepts metric dict, panel layout extensible
├── main.py                # v1 + v2: registry of "tasks" (xor/circle/spiral/mountaincar/transformer)
├── datasets.py            # v1 only. Toy datasets live here.
├── sanity.py              # v1
├── layers.py              # NEW. Linear, ReLU, Tanh, Softmax, LayerNorm — building blocks.
├── modules.py             # NEW. MultiHeadAttention, Embedding, Residual.
├── optim.py               # NEW. AdamW (decoupled weight decay).
├── envs/
│   └── mountaincar.py     # NEW. gymnasium wrapper, normalized obs.
├── data/
│   └── text.py            # NEW. Loads a .txt file, returns token ids.
├── transformer/
│   ├── model.py           # NEW. Stack of blocks, lm head, sampling loop.
│   ├── block.py           # NEW. Attention + feedforward + residual + layernorm.
│   ├── attention.py       # NEW. Self-attention, causal mask, multi-head.
│   ├── embed.py           # NEW. Token embedding + sinusoidal positional encoding.
│   └── train.py           # NEW. Trainer with AdamW, checkpointing.
└── tests/
    ├── test_nn.py         # v1, unchanged
    ├── test_attention.py  # NEW
    ├── test_transformer.py # NEW
    └── test_rl.py         # NEW
```

### Parte A — MountainCar-v0 com REINFORCE

Reaproveita **componentes da v1**, não `nn.py` direto:

- `envs/mountaincar.py` — wrapper do `gymnasium` (sucessor do `gym`),
  normaliza observações para `[-1, 1]²` (position, velocity).
- `envs/rollout.py` — função que roda N episódios coletando
  `(states, actions, rewards, log_probs)` em batches.
- `train_rl.py` — loop de **REINFORCE com baseline** (constant
  baseline = média de retornos do batch), escrito do zero em numpy
  usando `layers.Linear`.
- `main.py` ganha registro de tarefa `mountaincar`; tecla `4` ativa.

### Parte B — Transformer from scratch (3Blue1Brown cap. 6-7)

Implementa **decoder-only** (GPT-style) do zero, em numpy, usando
`layers.Linear` + módulos próprios:

- `transformer/embed.py` — token embedding + positional encoding
  sinusoidal.
- `transformer/attention.py` — self-attention com **causal mask**,
  scaled dot-product, multi-head. Inclui **softmax + cross-entropy**
  fused gradient (mesma fórmula da v1, reaproveitada).
- `transformer/block.py` — transformer block: attention + feedforward
  com residual + layer norm.
- `transformer/model.py` — stack de N blocos, lm head, sampling loop
  (temperature, top-k).
- `transformer/train.py` — trainer com **AdamW** (do `optim.py`).
- `data/text.py` — carrega `.txt` file e retorna token ids (char-level,
  256 vocab = ASCII byte).

Dataset: **`.txt` file** (default: um Shakespeare **já limpo** — sem
cabeçalho/rodapé do Project Gutenberg). User pode swap via `--data`.

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
   training, so that I can connect chapter 7 ("attention,
   step-by-step") to something happening on screen.
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
16. As a developer, I want the training to be **interruptible**
    (Ctrl-C saves current weights to `checkpoint.npz`, next start
    auto-loads if file exists) so that I can resume without losing
    progress.

## Implementation Decisions

### Reuse from v1 (compatível, sem mexer em `nn.py`)

- `nn.py` — **frozen**. Não adicionar flag, não mexer em API.
  `NeuralNetwork.forward(X)` continua esperando `X.shape == (N, fan_in)`.
  v2 não usa essa classe diretamente — usa `layers.Linear` (mesma
  matemática, mas com shape flexível).
- `datasets.py` — **frozen**. xor/circle/spiral continuam.
- `tests/test_nn.py` — **frozen**. v1 deve passar inalterado.
- `visualizer.py` — refactored com cuidado:
  - `update()` aceita `metrics: dict` em vez de kwargs fixos
    (`{"epoch", "loss", "acc", "reward", "tokens"}`).
  - Layout extensível: 2 painéis visíveis por vez; teclas
    `4` (mountaincar) / `5` (transformer) trocam o painel direito
    entre gym render e weight graph.
- `main.py` — vira registry de tarefas. Cada task tem
  `.reset()`, `.step()`, `.render()`, `.metrics()`. Loop principal
  só chama esses métodos.

### Módulos novos (Parte B)

- `layers.py` — building blocks. **Todos** com `forward(X)` e
  `backward(grad)` simétricos. Sem flags, sem `if` por shape.
  - `Linear(fan_in, fan_out)` — `X @ W + b`, gradientes padrão
  - `ReLU()`, `Tanh()`, `Sigmoid()`, `Softmax(axis=-1)`
  - `LayerNorm(d_model, eps=1e-5)`
  - `Embedding(vocab_size, d_model)`
- `modules.py` — composições:
  - `MultiHeadAttention(d_model, n_heads)`
  - `Residual(fn, x)` — `out = fn(x) + x`, com gradiente propagado
- `optim.py` — `AdamW(params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)`
  com bias correction.

### Parte A — MountainCar + REINFORCE

- **Architecture**: `[2, 16, 3]` (position, velocity → softmax over
  3 actions). ReLU on hidden (não Tanh — alinhado com v1; Tanh é relic
  pré-2015).
- **Policy**: stochastic (sample action from softmax probabilities).
- **REINFORCE com baseline** (não vanilla):
  - `G_t = Σ γ^k * r_{t+k}` (discounted return)
  - `advantage_t = G_t - b`, onde `b = mean(G)` no batch
  - `loss = -log π(a_t|s_t) * advantage_t`
  - Baseline reduz variância ~10× sem mudar o bias do gradiente.
- **Discount**: `γ = 0.99`.
- **Reward shaping**: use gymnasium's default (negative per step,
  +100 on goal). No shaping.
- **Rollout batch**: `(states, actions, rewards, log_probs)` como
  dataclass `RolloutBatch` em `envs/rollout.py`.
- **Training**: 500 episodes per visualizer session; resume-able
  via checkpoint.

### Parte B — Transformer

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
- **Optimizer**: AdamW (do `optim.py`) com defaults:
  `lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1`.
- **Training data**: a single `.txt` file (**already cleaned** —
  no Project Gutenberg headers/footers; see `data/text.py`).
  Default: Shakespeare excerpt, ~100KB clean.
- **Sampling**:
  - Temperature (default 1.0)
  - Top-k (default None = full softmax)
  - Max new tokens (default 200)
- **Checkpointing**:
  - Save to `checkpoint.npz` every 100 steps.
  - On startup, if `checkpoint.npz` exists in CWD, auto-load
    weights after constructing the model.

### Visualização v2

- **2 painéis visíveis** (não 3 — não cabe em 1280×720):
  - Esquerda: decision boundary OU gym render OU attention heatmap
    (depende do modo ativo).
  - Direita: weight graph v1 (mesmo layout).
- **Tabs** via teclas numéricas:
  - `1/2/3` — feedforward (xor/circle/spiral) [v1, sem mudança]
  - `4` — MountainCar: esquerda = gym render, direita = weight graph
  - `5` — Transformer: esquerda = attention heatmap (last batch),
    direita = weight graph
  - `6` — Transformer inspector: clique num nó do bloco transformer
    mostra a attention matrix daquela head específica (128×128)
- **Attention heatmap**: dark blue (low) → yellow → red (high).
  Renderizado a 128×128, scaled up pro painel.
- **Sampling console** at bottom (modo transformer):
  - `T` to enter a prompt (type, Enter to submit).
  - Output streams character-by-character.
- **Loss curve** in top bar (mini-chart, last 1000 steps — não 200).

### Interatividade

| Tecla | Ação |
|---|---|
| `1` `2` `3` | Dataset v1 (xor / circle / spiral) |
| `4` | MountainCar |
| `5` | Transformer |
| `6` | Transformer attention inspector |
| `SPACE` | Pause / resume training |
| `R` | Reset weights (same seed), restart |
| `+` / `-` | Adjust learning rate |
| `F` | Fast-forward (10 → 200 epochs/frame) |
| `T` | Type a prompt (transformer) |
| `ESC` | Quit |

### Dependencies

- **v1** (já em `requirements.txt`): `numpy`, `pygame-ce`. **Nada muda**.
- **v2 adiciona** (só quando v2 entrar): `gymnasium`. **Não adicionar
  agora** — `pip install -r requirements.txt` na v1 não deve puxar
  gymnasium à toa.

## Testing Decisions

### Módulos novos, com seam test

Cada módulo em `layers.py` e `modules.py` é testado isoladamente:

1. **`Linear.forward/backward`** — `tests/test_layers.py`:
   - Numerical gradient check (finite differences vs analytical).
   - Shape correctness for various input shapes (2D, 3D).
2. **`MultiHeadAttention`** — `tests/test_attention.py`:
   - Numerical gradient check vs analytical gradient (small inputs).
   - **Causal mask verification**: position `t` has zero attention
     weight on positions `> t`.
   - Multi-head split: 4 heads on d_model=32 → output equals
     concat(heads).
3. **`LayerNorm + Residual`** — `tests/test_modules.py`:
   - Backward pass correctness.
   - Residual gradient: `∂L/∂x = ∂L/∂out + ∂L/∂x_fn`.
4. **`AdamW`** — `tests/test_optim.py`:
   - Bias correction: at step 0, effective step = bias_correction.
   - Weight decay decoupled from gradient (per Loshchilov & Hutter).
5. **`TransformerBlock` end-to-end** — `tests/test_transformer.py`:
   - Forward + backward sanity (loss decreases after 100 steps on a
     tiny synthetic dataset: `vocab=16, d_model=32, 2 layers,
     seq_len=16`).
6. **`REINFORCE`** — `tests/test_rl.py`:
   - Gradient of `-log π(a|s) * advantage` w.r.t. weights is correct
     (numerical check).
   - Baseline subtraction: gradient w/ baseline equals gradient w/o
     baseline + mean-centered targets.

### What stays the same

- v1 tests (`test_xor`, `test_circle`, `test_spiral`) keep running
  **unchanged**. v2 must not regress v1.
- The single-seam philosophy still applies: each component has its
  own seam-tested module.

### Performance / scale guardrails

- "Loss decreases" test on transformer uses `vocab=16, d_model=32,
  2 layers, seq_len=16` — completes in <5s on CPU.
- Numerical gradient checks use small dimensions (d_model=8) so the
  finite-difference loop is fast.
- Full-size model training is **not** tested in CI; manual visualizer
  run only.

## Out of Scope

- Training at scale (>10MB datasets, GPU, >10M params).
- Modern tokenizer (BPE, SentencePiece) — character-level only.
- Encoder-only or encoder-decoder transformers (BERT, T5) —
  decoder-only (GPT-style) is enough for this educational project.
- KV-cache optimization (recompute full attention each step; fine
  at seq_len=128).
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

v1 (`nn-visualizer`) **shipped** in v0.1.0. v2 is bigger (transformer
alone is ~3× the LOC of v1). Doing v1 first meant:
- The math primitives (forward/backward, softmax+CE gradient) are
  battle-tested before transformer reuses them.
- The visualizer framework is in place; v2 just adds panels.
- We learn our lessons on a smaller system before scaling up.

### Why character-level tokenization

BPE / SentencePiece add a real dep (`tiktoken`, `sentencepiece`)
and a learning curve that distracts from the transformer itself.
For a 600K-param model on 100KB of text, char-level produces
**worse samples** than BPE — but the **math is identical**, and
that's the point. A future v3 can add BPE as a drop-in
`Tokenizer` interface with no model changes.

### Why clean training text

Project Gutenberg files have ~50KB of license boilerplate at the
start and end. Char-level models happily memorize and reproduce
"PROJECT GUTENBERG LICENSE TERMS…" as the first sample. To get
useful samples, default to a **pre-cleaned** excerpt (~100KB of
just the play text).

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

### Why REINFORCE with baseline

Vanilla REINFORCE has very high variance — same action can give
different rewards, and the gradient signal is noisy. A constant
baseline `b = mean(G)` doesn't change the **expectation** of the
gradient (unbiased) but reduces **variance** ~10×. This means
training converges with fewer episodes and is more stable across
seeds. Cost: 2 extra lines.

### Why modules not flags

Earlier draft proposed `forward_return_activations()` flag on
`NeuralNetwork.forward()`. Rejected: flags are bad design (they
couple unrelated concerns, force callers to know about
implementation). v2 builds on top of `layers.Linear` which already
returns its output AND can be queried for its intermediate state
(`layer.a` attribute, same pattern v1 already uses). v2 modules
follow this convention.

### Why modules not a refactored `nn.py`

Refactoring `nn.py` v1 to support arbitrary shapes would break the
v1 API and force v1 visualizer/tests to change. Instead, extract
the math into `layers.py` (new) and leave `nn.py` v1 frozen. v2
builds on `layers.py` directly; v1 still uses `nn.py`. Both files
can coexist indefinitely.

### Reference

- `Franko12345/ai/AIgym.py` (2023) — same MountainCar-v0 setup,
  but with GA. We replace GA with REINFORCE. Visualization style
  follows the dark-background spec from v1 (not the salmon
  background of the old code).
- `neural-networks-foundation` skill — chapters 5-8 cover exactly
  the concepts this spec implements.
- 3Blue1Brown chapters 5-8 — visual reference for attention maps
  and "how LLMs store facts".

## Spec revision history

- **v2.0 (original)**: drafted ad-hoc, dense, no structure review.
- **v2.1 (this revision)**: structural review applied — 10 changes:
  1. Extract `layers.py` instead of refactoring `nn.py`
  2. Separate `envs/mountaincar.py` and `data/text.py` from
     `datasets.py`
  3. `update()` accepts metric dict; 2 panels visible, tabs for mode
  4. `main.py` vira registry de tarefas
  5. `nn.py` v1 stays frozen, no flags added
  6. `gymnasium` only added when v2 ships (not in v1 requirements)
  7. `RolloutBatch` dataclass for `(states, actions, rewards, log_probs)`
  8. Cleaned training text default, not raw Gutenberg
  9. Removed `forward_return_activations()` flag proposal
  10. Documented checkpoint auto-load semantics