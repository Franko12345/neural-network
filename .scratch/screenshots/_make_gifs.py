"""Generate GIF frames by training each dataset under dummy SDL and saving
frames to disk. Then ffmpeg joins them into a GIF.

ponytail: ad-hoc script for README asset generation. Not part of the
shipped visualizer.
"""
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pygame

from datasets import circle, spiral, xor
from nn import NeuralNetwork, one_hot
from visualizer import Visualizer


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


DATASETS = {"xor": xor, "circle": circle, "spiral": spiral}
FRAMES_PER_DATASET = 60
EPOCHS_PER_FRAME = 30  # so each GIF shows meaningful learning progress


def capture_dataset(name: str, arch: list[int], out_dir: str) -> int:
    gen = DATASETS[name]
    X, y = gen(n=200, seed=0)
    Y = one_hot(y, arch[-1])
    nn = NeuralNetwork(arch, ["relu"] * (len(arch) - 2) + ["softmax"])

    viz = Visualizer()
    os.makedirs(out_dir, exist_ok=True)
    epoch = 0
    for frame in range(FRAMES_PER_DATASET):
        nn.fit(X, Y, epochs=EPOCHS_PER_FRAME, lr=0.05)
        epoch += EPOCHS_PER_FRAME
        Y_hat = nn.forward(X)
        acc = float((Y_hat.argmax(axis=1) == y).mean())
        loss = float(-np.mean(np.sum(Y * np.log(Y_hat + 1e-12), axis=1)))
        viz.update(nn, X, y, epoch, loss, acc, name)
        pygame.image.save(viz.screen, f"{out_dir}/frame_{frame:03d}.png")
    viz.close()
    return FRAMES_PER_DATASET


def make_gif(frames_dir: str, gif_path: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", "20",
        "-i", f"{frames_dir}/frame_%03d.png",
        "-vf", "palettegen",
        f"{frames_dir}/palette.png",
    ]
    subprocess.run(cmd, check=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", "20",
        "-i", f"{frames_dir}/frame_%03d.png",
        "-i", f"{frames_dir}/palette.png",
        "-lavfi", "paletteuse",
        gif_path,
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    out_root = ".scratch/screenshots/gif-frames"
    final_dir = ".scratch/screenshots"
    for name, arch in [("xor", [2, 8, 8, 2]), ("circle", [2, 8, 8, 2]), ("spiral", [2, 16, 16, 3])]:
        d = f"{out_root}/{name}"
        if os.path.isdir(d):
            shutil.rmtree(d)
        print(f"capturing {name} ({FRAMES_PER_DATASET} frames)...")
        capture_dataset(name, arch, d)
        gif_path = f"{final_dir}/{name}.gif"
        print(f"  encoding {gif_path}...")
        make_gif(d, gif_path)
        print(f"  done ({os.path.getsize(gif_path)} bytes)")
    print("all GIFs ready in .scratch/screenshots/")
    return 0


if __name__ == "__main__":
    sys.exit(main())