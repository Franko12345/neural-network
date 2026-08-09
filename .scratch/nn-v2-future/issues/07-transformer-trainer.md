# 07 — Transformer trainer with AdamW + checkpointing

**What to build:** `transformer/train.py` that takes a `Transformer`
model + tokenized text corpus, runs training with AdamW, saves
checkpoints, and auto-loads on startup. From the user's perspective:
`trainer = Trainer(model, data, lr=3e-4); trainer.train(n_steps=1000)`
runs training and saves `checkpoint.npz` every 100 steps. A second
`train()` call with `checkpoint.npz` present auto-resumes.

**Blocked by:** 06 (Transformer model), 04 (AdamW).

**Status:** ready-for-agent

- [ ] `data/text.py`: `load_text(path) -> tuple[np.ndarray, int]`
      returning `(token_ids, vocab_size)`. Char-level: each byte → one
      token. Default path is a cleaned Shakespeare excerpt (~100KB
      bundled in `data/shakespeare.txt`, no Project Gutenberg headers).
- [ ] `transformer/train.py`: `Trainer(model, data, lr=3e-4)`:
      - On init: if `checkpoint.npz` exists in CWD, load it into model
      - `step()`: forward on a random batch, compute loss
        (cross-entropy), backward, AdamW update
      - `train(n_steps)`: loops calling step(), saves
        `checkpoint.npz` every 100 steps with all model weights
      - `sample(prompt, n_tokens=200)` autoregressive decode
- [ ] `tests/test_train.py`:
      - Checkpoint round-trip: save → load produces identical logits
      - Auto-load: with `checkpoint.npz` present, `Trainer(...)`
        loads weights without training
      - "Loss decreases" over 100 steps on a 4-token sequence
- [ ] Headless smoke: train 50 steps on bundled Shakespeare, save
      checkpoint, reload, exit 0
- [ ] ponytail: trainer state = `model + optim + step_count`; no
      hidden state in globals
- [ ] v1 tests still pass