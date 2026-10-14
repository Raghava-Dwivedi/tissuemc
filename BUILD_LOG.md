# Build Log

Defects and surprises, recorded **as they happen** during the build. `VALIDATION.md` copies from this file. Entries are never reconstructed after the fact.

| Date | Component | What went wrong | How it was caught | Fix |
|---|---|---|---|---|
| 2026-10-13 | `tools/colab_smoke_test.ipynb` | The `ncu` check `rc == 0 and A or B` parsed as `(rc == 0 and A) or B`, which could report "WORKS" when `ncu` failed | Self-review before first use | Parenthesized the condition (commit `fix(tools): correct operator precedence in ncu check`) |
| 2026-10-14 | `tests/cpp/test_physics.cpp`: energy-conservation check | The check `Rsp + (Rd + Tt + A) / N == 1` failed at 1e6 photons with a plain `double` accumulator (tolerance 2e-12). Measured deviation: 3e-14 at 1e4 photons, 7e-13 at 1e5, 5.7e-12 at 1e6, growing with N. | Failed test. Checked first whether this was a real weight leak (a leak would be a fixed offset) or rounding: re-ran with Neumaier compensated summation and the deviation was exactly 0 at 1e4, 1e5, 1e6 and 4e6 photons. | Physics is correct. Added `include/tissuemc/accumulate.hpp` (`CompensatedSum`) and use it for tallies. Tolerance tightened to 1e-13, which the old accumulator would fail, so the test now discriminates. |
