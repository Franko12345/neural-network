"""Generate GIF frames for the v2 panels: MountainCar (gym render) and
Transformer (attention heatmap, both training and sampling).

ponytail: ad-hoc script for README asset generation. Not part of the
shipped visualizer. Each GIF is 120 frames at 20fps = 6s loop.

Outputs (under .scratch/screenshots/):
  - mountaincar.gif     — REINFORCE training rollout per frame
  - transformer_train.gif — AdamW training, attention heatmap evolving
  - transformer_sample.gif — same model sampling tokens autoregressively
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from layers import Linear, ReLU, Softmax
from transformer.attention import MultiHeadAttention
from transformer.model import Transformer
from visualizer import Visualizer


FRAMES = 120
FPS = 20
EPOCHS_PER_FRAME = 30  # MountainCar REINFORCE updates per frame
TRAIN_STEPS_PER_FRAME = 1  # transformer AdamW step per frame


def _save_frame(viz: Visualizer, path: str) -> None:
    pygame.image.save(viz.screen, path)


def _make_gif(frames_dir: str, gif_path: str) -> None:
    """ffmpeg palettegen + paletteuse to keep file size sane."""
    palette = f"{frames_dir}/palette.png"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-framerate", str(FPS),
         "-i", f"{frames_dir}/frame_%03d.png",
         "-vf", "palettegen", palette],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-framerate", str(FPS),
         "-i", f"{frames_dir}/frame_%03d.png",
         "-i", palette,
         "-lavfi", "paletteuse", gif_path],
        check=True,
    )


# ----- MountainCar ---------------------------------------------------------

def capture_mountaincar(out_dir: str, seed: int = 0) -> None:
    """Train REINFORCE for FRAMES steps; per frame, render a fresh policy
    rollout in the gym env and save the frame."""
    from envs.mountaincar import MountainCarEnv
    from envs.rollout import rollout
    from train_rl import train

    env = MountainCarEnv()
    rng = np.random.default_rng(seed)
    # Policy: [2, 16, 3] per ticket 03
    lin1 = Linear(2, 16); lin2 = Linear(16, 3)
    lin1.W = rng.standard_normal(lin1.W.shape) * 0.3
    lin2.W = rng.standard_normal(lin2.W.shape) * 0.3
    relu = ReLU(); softmax = Softmax(axis=-1)

    def policy(state):
        s = np.atleast_2d(state)
        h = relu.forward(lin1.forward(s))
        logits = lin2.forward(h)
        probs = softmax.forward(logits)
        a = int(np.random.choice(probs.shape[-1], p=probs[0]))
        lp = float(np.log(max(probs[0, a], 1e-12)))
        return a, lp

    viz = Visualizer()
    os.makedirs(out_dir, exist_ok=True)
    try:
        for frame in range(FRAMES):
            # Train one batch of 3 episodes
            np.random.seed(seed + frame)
            batch = rollout(env, policy, n_episodes=3, max_steps=200,
                            seed=seed + frame)
            train([lin1, lin2], batch, lr=0.01, gamma=0.99)

            # Roll out 1 episode with current policy to capture a frame.
            obs = env.reset(seed=seed * 1000 + frame)
            env_frame = env.render()
            for _ in range(200):
                a, _ = policy(obs)
                obs, _, done, _ = env.step(a)
                if done:
                    break
            # Inject the gym frame; visualizer renders it on the right panel.
            viz.update(metrics={"frame": env_frame, "epoch": frame,
                                "panel": "gym_render"},
                       panel="gym_render")
            _save_frame(viz, f"{out_dir}/frame_{frame:03d}.png")
    finally:
        env.close()
        viz.close()


# ----- Transformer: training (attention heatmap evolves) -------------------

def capture_transformer_train(out_dir: str, seed: int = 0) -> None:
    """Train a tiny transformer on bundled Shakespeare; capture the
    first block's first-head attention heatmap after each AdamW step."""
    from data.text import DEFAULT_PATH, load_text
    from transformer.train import Trainer

    raw_ids, _ = load_text(DEFAULT_PATH)
    # Tiny config so 120 frames show meaningful learning: vocab=16
    # (first 16 ASCII bytes), d_model=16, 1 block, 2 heads. Bigger
    # model would not learn in 120 AdamW steps with this much data.
    # Bundled Shakespeare has bytes > 16; clip into [0, vocab).
    ids = (raw_ids[:200] % 16).astype(np.uint8)
    model = Transformer(vocab=16, d_model=16, n_heads=2, n_layers=1,
                        d_ff=32, max_seq_len=16, seed=seed)
    trainer = Trainer(model, data=ids, lr=3e-3, seed=seed)

    viz = Visualizer()
    os.makedirs(out_dir, exist_ok=True)
    try:
        for frame in range(FRAMES):
            # One AdamW step per frame (per spec, Trainer.train defaults
            # to save_every=100; here we call step() directly to avoid
            # touching checkpoint logic in the GIF generator).
            trainer.step(batch_size=2, seq_len=8)

            # Capture attention weights from the most recent forward by
            # re-running forward on a tiny fixed prompt.
            prompt = np.array([[3, 8, 5, 12, 1, 7, 2, 14]], dtype=np.uint8)
            model.forward(prompt)
            attn = model.blocks[0].mha.attn_weights[0]  # (B, T, T) -> (T, T)

            viz.update(metrics={"attention_weights": attn,
                                "epoch": frame, "loss": 0.0,
                                "panel": "attention_heatmap"},
                       panel="attention_heatmap")
            _save_frame(viz, f"{out_dir}/frame_{frame:03d}.png")
    finally:
        viz.close()


# ----- Transformer: sampling (token-by-token autoregressive decode) -----

def capture_transformer_sample(out_dir: str, seed: int = 0) -> None:
    """Train briefly, then sample token-by-token; per frame, capture the
    attention heatmap of the current context (so the GIF shows the
    attention pattern shifting as new tokens are appended)."""
    from data.text import DEFAULT_PATH, load_text
    from transformer.train import Trainer

    raw_ids, _ = load_text(DEFAULT_PATH)
    # Tiny config so 120 frames show meaningful learning: vocab=16
    # (first 16 ASCII bytes), d_model=16, 1 block, 2 heads. Bigger
    # model would not learn in 120 AdamW steps with this much data.
    # Bundled Shakespeare has bytes > 16; clip into [0, vocab).
    ids = (raw_ids[:200] % 16).astype(np.uint8)
    model = Transformer(vocab=16, d_model=16, n_heads=2, n_layers=1,
                        d_ff=32, max_seq_len=16, seed=seed)
    trainer = Trainer(model, data=ids, lr=3e-3, seed=seed)

    # Sample token-by-token.
    prompt_ids = np.array([[3, 8, 5]], dtype=np.uint8)
    sampled = prompt_ids.copy()
    np.random.seed(seed)

    viz = Visualizer()
    os.makedirs(out_dir, exist_ok=True)
    try:
        for frame in range(FRAMES):
            # Forward + capture attention on the current context
            # (T x T square matrix). For a sampled sequence of length T,
            # build a per-token attention matrix by running forward on
            # each prefix and stitching the last row of each.
            T = sampled.shape[1]
            if T > 16:
                ctx = sampled[:, -16:]
                T = 16
            else:
                ctx = sampled
            attn_sq = np.zeros((T, T))
            for i in range(T):
                model.forward(ctx[:, : i + 1])
                attn_sq[i, : i + 1] = model.blocks[0].mha.attn_weights[0][0, -1]

            viz.update(metrics={"attention_weights": attn_sq,
                                "epoch": frame,
                                "panel": "attention_heatmap"},
                       panel="attention_heatmap")
            _save_frame(viz, f"{out_dir}/frame_{frame:03d}.png")

            # Append the next sampled token (autoregressive). If
            # sampled has grown past max_seq_len, feed the model the
            # last max_seq_len tokens (KV-cache would be better, but
            # pos_enc is hard-bounded by max_seq_len).
            ctx = sampled[:, -16:] if sampled.shape[1] > 16 else sampled
            next_id = model.sample(ctx, n_tokens=1, temperature=0.8,
                                   top_k=4)[:, -1:]
            sampled = np.concatenate([sampled, next_id], axis=1)
            sampled = np.concatenate([sampled, next_id], axis=1)
    finally:
        viz.close()


# ----- main -----------------------------------------------------------------

def main() -> int:
    out_root = ".scratch/screenshots/gif-frames"
    final_dir = ".scratch/screenshots"

    targets = [
        ("mountaincar", capture_mountaincar),
        ("transformer_train", capture_transformer_train),
        ("transformer_sample", capture_transformer_sample),
    ]
    for name, fn in targets:
        d = f"{out_root}/{name}"
        if os.path.isdir(d):
            shutil.rmtree(d)
        print(f"capturing {name} ({FRAMES} frames)...")
        fn(d)
        gif_path = f"{final_dir}/{name}.gif"
        print(f"  encoding {gif_path}...")
        _make_gif(d, gif_path)
        print(f"  done ({os.path.getsize(gif_path)} bytes)")
    print("all v2 GIFs ready in .scratch/screenshots/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
