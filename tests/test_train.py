"""Headless validation of transformer/train.py and data/text.py.

ponytail: stdlib only, no pytest.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

from layers import Softmax
from transformer.model import Transformer
from transformer.train import Trainer
from data.text import load_text, ASCII_VOCAB_SIZE


def test_load_text_returns_int_array_and_vocab() -> bool:
    """load_text returns (token_ids uint8, vocab_size)."""
    # Write a small temp file
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tiny.txt")
        with open(path, "wb") as f:
            f.write(b"hello world\n")
        ids, vocab = load_text(path)
    return (
        ids.dtype == np.uint8
        and ids.ndim == 1
        and vocab == ASCII_VOCAB_SIZE
        and len(ids) > 0
        and ids.max() < ASCII_VOCAB_SIZE
    )


def test_checkpoint_roundtrip() -> bool:
    """Save → load produces identical forward logits on same input."""
    model = Transformer(vocab=16, d_model=8, n_heads=2, n_layers=2,
                        d_ff=16, max_seq_len=8, seed=0)
    trainer = Trainer(model, data=(np.arange(20) % 16).astype(np.uint8), lr=3e-4)
    ids = np.array([[1, 5, 3]], dtype=np.uint8)
    logits_before = model.forward(ids).copy()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "checkpoint.npz")
        trainer.save(path)
        # Build a fresh model, load
        model2 = Transformer(vocab=16, d_model=8, n_heads=2, n_layers=2,
                             d_ff=16, max_seq_len=8, seed=0)
        # Init defaults differ — manually zero to match before-save
        # Easier: load the checkpoint INTO model2 first
        Trainer(model2, data=(np.arange(20) % 16).astype(np.uint8), lr=3e-4).load(path)
        logits_after = model2.forward(ids)
    return np.allclose(logits_before, logits_after, atol=1e-6)


def test_checkpoint_auto_load_on_init() -> bool:
    """With checkpoint.npz present in CWD, Trainer loads weights without
    manual load() call."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "checkpoint.npz")
        # First: train 100 steps and save
        model = Transformer(vocab=16, d_model=8, n_heads=2, n_layers=2,
                            d_ff=16, max_seq_len=8, seed=0)
        trainer = Trainer(model, data=(np.arange(50) % 16).astype(np.uint8), lr=3e-4)
        trainer.train(n_steps=10)
        logits_trained = model.forward(np.array([[1, 2, 3]], dtype=np.uint8)).copy()
        trainer.save(path)
        # Now: new Trainer in same CWD — should auto-load
        # Temporarily chdir
        old_cwd = os.getcwd()
        try:
            os.chdir(td)
            model2 = Transformer(vocab=16, d_model=8, n_heads=2, n_layers=2,
                                 d_ff=16, max_seq_len=8, seed=0)
            Trainer(model2, data=(np.arange(50) % 16).astype(np.uint8), lr=3e-4)
            logits_loaded = model2.forward(np.array([[1, 2, 3]], dtype=np.uint8))
        finally:
            os.chdir(old_cwd)
    return np.allclose(logits_trained, logits_loaded, atol=1e-6)


def test_train_loss_decreases_100_steps() -> bool:
    """100 SGD steps on a small synthetic dataset: loss drops below the
    early-step minimum (true descent)."""
    # Tiny config: vocab=16, d_model=8, 2 layers
    model = Transformer(vocab=16, d_model=8, n_heads=2, n_layers=2,
                        d_ff=16, max_seq_len=16, seed=0)
    np.random.seed(0)
    data = (np.random.randint(0, 16, 200) % 16).astype(np.uint8)
    trainer = Trainer(model, data=data, lr=3e-3, seed=1)
    losses = []
    for _ in range(100):
        losses.append(trainer.step(seq_len=8))
    return losses[-1] < min(losses[1:20])


def test_train_smoke_50_steps_on_bundled_shakespeare() -> bool:
    """50 steps on bundled Shakespeare, save checkpoint, reload, exit 0."""
    from data.text import DEFAULT_PATH
    ids, vocab = load_text(DEFAULT_PATH)
    assert vocab == ASCII_VOCAB_SIZE
    model = Transformer(vocab=vocab, d_model=32, n_heads=2, n_layers=2,
                        d_ff=64, max_seq_len=32, seed=0)
    trainer = Trainer(model, data=ids, lr=3e-4, seed=1)
    trainer.train(n_steps=50)
    # Save + reload
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "checkpoint.npz")
        trainer.save(path)
        model2 = Transformer(vocab=vocab, d_model=32, n_heads=2, n_layers=2,
                             d_ff=64, max_seq_len=32, seed=0)
        Trainer(model2, data=ids, lr=3e-4, seed=1).load(path)
        # Sample
        out = model2.sample(np.array([[10, 20, 30]], dtype=np.uint8),
                            n_tokens=5, temperature=1.0, top_k=None)
    return out.shape == (1, 8)


def main() -> int:
    results = [
        ("load_text_returns_int_array_and_vocab", test_load_text_returns_int_array_and_vocab()),
        ("checkpoint_roundtrip", test_checkpoint_roundtrip()),
        ("checkpoint_auto_load_on_init", test_checkpoint_auto_load_on_init()),
        ("train_loss_decreases_100_steps", test_train_loss_decreases_100_steps()),
        ("train_smoke_50_steps_on_bundled_shakespeare", test_train_smoke_50_steps_on_bundled_shakespeare()),
    ]
    for name, ok in results:
        flag = "PASS" if ok else "FAIL"
        print(f"  {name:42s} {flag}")
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
