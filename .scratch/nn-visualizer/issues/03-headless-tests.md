# 03 — Headless tests for the single seam

**What to build:** A single test file at `tests/test_nn.py` that
exercises the `NeuralNetwork` class against all three datasets and
exits 0 on pass. From the user's perspective: running
`python3 tests/test_nn.py` on the LXC takes <10 seconds total and
prints "all tests passed".

**Blocked by:** 01 (need the cleaned-up network), 02 (need the
generators).

**Status:** ready-for-agent

- [ ] `tests/test_nn.py` exists with three test functions:
      `test_xor`, `test_circle`, `test_spiral`.
- [ ] `test_xor`: train 2000 epochs at lr=0.05 on `xor(200)` with
      arch `[2, 8, 8, 2]`, assert final accuracy ≥ 0.95.
- [ ] `test_circle`: train 2000 epochs at lr=0.05 on `circle(200)`
      with arch `[2, 8, 8, 2]`, assert final accuracy ≥ 0.90.
- [ ] `test_spiral`: train 3000 epochs at lr=0.05 on `spiral(200)`
      with arch `[2, 16, 16, 3]`, assert final accuracy ≥ 0.85.
- [ ] Each test prints a one-line summary (`dataset: xor, acc:
      0.97, PASS`) before its assertion.
- [ ] Test runner uses **stdlib only**: a top-level `if __name__ ==
      "__main__":` block that calls each test, `sys.exit(1)` on
      failure, `print("all tests passed")` on success. No pytest.
- [ ] Tests use the fixed seeds from tickets 01 and 02 — results
      are reproducible.
- [ ] Sanity: total runtime < 10 seconds on the headless LXC.

## Notes

This ticket exists because the v1 spec chose **one testable seam**
(`NeuralNetwork`). Everything else (visualizer, main loop) is
smoke-tested only. If a future ticket regresses the math, this
file is where you'd catch it first.