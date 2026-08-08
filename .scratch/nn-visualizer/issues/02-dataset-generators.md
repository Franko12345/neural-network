# 02 — Dataset generators (XOR / circle / spiral)

**What to build:** Three importable generators that each return
`(X, y)` with `X` in `[-1, 1]²` and integer class labels. From the
user's perspective: `from datasets import xor, circle, spiral;
X, y = xor(n=200)` produces a balanced 2-class dataset that is
plotable as a 2D scatter.

**Blocked by:** 01 (the generators don't strictly need the network,
but sequencing keeps the math primitives validated first — and the
test ticket that gates "is the math correct" needs them).

**Status:** ready-for-agent

- [ ] `xor(n=200, seed=0)` returns `(X, y)` with shape `(n, 2)` and
      `(n,)`. Two balanced classes — `(0,0)` and `(1,1)` → class 1;
      `(0,1)` and `(1,0)` → class 0. Each point gets Gaussian noise
      (σ=0.1) so the network can't memorize a hard step.
- [ ] `circle(n=200, seed=0)` returns `(X, y)` with inner circle
      (radius ~0.3, σ=0.1) as class 0 and outer ring (radius ~0.8,
      σ=0.1) as class 1.
- [ ] `spiral(n=200, seed=0)` returns `(X, y)` with **3 classes**,
      ~67 points each, intertwined spirals with σ=0.1 noise. y values
      in `{0, 1, 2}`.
- [ ] All three normalize `X` to `[-1, 1]²` (per-axis min-max).
- [ ] Each generator accepts a `seed` argument for reproducibility
      (default `0` so tests are stable).
- [ ] Sanity script (in-repo) loads each dataset, prints shape and
      class balance, exits 0.

## Notes

This ticket **only** delivers the generators — no visualization, no
tests. Visualization lands in ticket 04. Headless correctness tests
land in ticket 03 (which depends on this one + 01).