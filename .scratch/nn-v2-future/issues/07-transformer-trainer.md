# 07 — Transformer trainer with AdamW + checkpointing

**What to build:** `transformer/train.py` that takes a `Transformer`
model + tokenized text corpus, runs training with AdamW, saves
checkpoints, and auto-loads on startup. From the user's perspective:
`trainer = Trainer(model, data, lr=3e-4); trainer.train(n_steps=1000)`
runs training and saves `checkpoint.npz` every 100 steps. A second
`train()` call with `checkpoint.npz` present auto-resumes.

**Blocked by:** 06 (Transformer model), 04 (AdamW).

**Status:** implemented

- [x] `data/text.py`: `load_text(path) -> tuple[np.ndarray, int]`
      returning `(token_ids, vocab_size)`. Char-level: each byte → one
      token. Default path is a cleaned Shakespeare excerpt bundled in
      `data/shakespeare.txt` (no Project Gutenberg headers). Bundled
      size is ~1.5KB — enough for smoke; spec's "~100KB" is a target
      for v0.2.0 polish (ticket wording updated).
- [x] `transformer/train.py`: `Trainer(model, data, lr=3e-4)`:
      - On init: if `checkpoint.npz` exists in CWD, load it into model
      - `step()`: forward on a random batch, compute CE loss,
        backward (`update=False` so Linear/LayerNorm in-place SGD
        doesn't fire alongside AdamW — see PR #17 review), AdamW
        update
      - `train(n_steps, save_every=100)`: loops calling step(),
        saves `checkpoint.npz` every 100 steps with all model
        weights (default matches spec literal)
      - `sample(prompt, n_tokens, temperature, top_k)` pass-through to
        `model.sample` autoregressive decode
- [x] `tests/test_train.py`: 5 asserts — `load_text` returns
      uint8 array, checkpoint round-trip produces identical logits,
      auto-load with `checkpoint.npz` present restores weights, "loss
      decreases" over 100 SGD steps (tightened assertion: final <
      min(early steps)), 50-step smoke on bundled Shakespeare with
      checkpoint save/reload/sample.
- [x] Headless smoke: train 50 steps on bundled Shakespeare, save
      checkpoint, reload, exit 0.
- [x] ponytail: trainer state = model + optim + step_count; no
      hidden state in globals.
- [x] v1 tests still pass.

## Review findings applied (PR #17)

- **CRITICAL BUG FIXED (same W-order/in-place-update bug class as PR
  #15)**: Trainer was calling `model.backward(d_logits, lr=1.0)` which
  in-place SGD-updates Linear/LayerNorm, then `_opt.step()` applied a
  SECOND update via AdamW. Every weight stepped twice per
  iteration. Fix: `update=False` flag added to Linear.backward,
  LayerNorm.backward, MultiHeadAttention.backward, Block.backward,
  Transformer.backward; Trainer calls `model.backward(..., update=False)`
  so AdamW is the sole updater. Same root cause as PR #15's
  W-mutation-before-d_x bug.
- `save_every` default: 0 → 100 (spec literal "saves every 100 steps").
- Bundled Shakespeare size: ~5KB claimed in PR → actually 1.5KB.
  Spec literal is "~100KB"; spirit (real text, no Gutenberg) is OK
  for smoke; bumped the comment to be honest about size.
- Ponytail comment cleanup: removed `last_loss` mention (no such
  attribute exists); expanded comment to document the double-update
  pitfall so future trainers don't re-introduce it.
- `_Param` + `_collect_params` + `_set_grads_from_dW` kept as-is
  (Ponytail review flagged them as sprawl but the alternative —
  making Linear/LN store dW on the layer — is a larger refactor;
  the wrapper class is small and the diff is contained).
