# 04 — LayerNorm, Residual, AdamW

**What to build:** Three new modules — `layers.LayerNorm`,
`modules.Residual`, `optim.AdamW` — each with their own seam tests.
From the user's perspective: each is independently importable and
testable, with a clean API.

**Blocked by:** 01 (Linear, needed for AdamW parameter iteration).

**Status:** ready-for-agent

- [ ] `layers.LayerNorm(d_model, eps=1e-5)`: `forward(X)` normalizes
      last axis, scales + shifts with learnable γ/β. `backward(grad)`
      computes gradient w.r.t. input, γ, β.
- [ ] `modules.Residual(fn)`: composes a callable such that
      `forward(x) = fn(x) + x`, with `backward(grad)` correctly
      splitting gradients between fn and identity.
- [ ] `optim.AdamW(params, lr=3e-4, betas=(0.9, 0.95),
      weight_decay=0.1)`: Adam with decoupled weight decay (per
      Loshchilov & Hutter). Bias correction on first few steps.
      `step()` and `zero_grad()` methods.
- [ ] `tests/test_modules.py`: LayerNorm forward/backward correctness;
      Residual gradient split
- [ ] `tests/test_optim.py`: AdamW bias correction at step 0;
      weight decay decoupled from gradient
- [ ] Headless smoke: each module runs standalone, exits 0
- [ ] ponytail: no flag on `step()`; one module per file