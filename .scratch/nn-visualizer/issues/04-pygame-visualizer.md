# 04 — Pygame visualizer (decision boundary + weight graph)

**What to build:** A pygame window that draws the network's decision
boundary on the left and the weight graph on the right, with a top
bar showing epoch / loss / accuracy / dataset name. From the user's
perspective: `python3 visualizer.py --dataset xor` opens a
1280×720 window with both panels animating live as the network
trains.

**Blocked by:** 01 (need the cleaned-up network exposing weights,
biases, activations per layer). Does **not** need ticket 02 — the
visualizer can start with XOR hardcoded and pick up the other
datasets via the main loop in ticket 05.

**Status:** ready-for-agent

- [ ] `visualizer.py` exposes a `Visualizer` class taking a
      `NeuralNetwork` and a window size.
- [ ] Left panel (~640×660 below the top bar): decision boundary
      rendered as a 40×40 colored grid, each cell colored by the
      predicted class for that (x, y) input. Three class colors
      (teal / orange / magenta) for max contrast on dark bg.
- [ ] Right panel (~640×660): weight graph. Nodes drawn as circles
      (radius 18px, 2px black outline, fill = activation clamped to
      [0, 255]). Connections drawn as lines, **blue for negative
      weight, red for positive**, thickness = `min(5, int(abs(w)
      * 2))`. No labels on weights or biases (per spec).
- [ ] Top bar (60px tall): white monospace text showing current
      epoch, loss (rounded to 3 decimals), accuracy (rounded to 2
      decimals), and dataset name.
- [ ] Background: dark `(15, 15, 20)`.
- [ ] Animation: `visualizer.update(network, epoch, loss, acc,
      dataset_name)` is called from outside; this ticket's job is
      to **render**, not to drive training.
- [ ] `__main__` block runs a minimal demo (hardcoded XOR, 10
      epochs/frame at 60fps) so the file is testable on its own.
- [ ] Total LOC ≤ 200 lines.
- [ ] Headless smoke: `SDL_VIDEODRIVER=dummy python3 visualizer.py`
      runs the demo loop for 60 frames without crashing on the LXC
      (validated in ticket 06's smoke test).

## Notes

This ticket ends with a **demoable visualizer** but no controls yet
(those come with the main loop in ticket 05). Visual style is
locked by the spec — do not invent new colors / layouts here.