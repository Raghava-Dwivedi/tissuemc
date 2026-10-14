# Build Log

Defects and surprises, recorded **as they happen** during the build. `VALIDATION.md` copies from this file. Entries are never reconstructed after the fact.

| Date | Component | What went wrong | How it was caught | Fix |
|---|---|---|---|---|
| 2026-10-13 | `tools/colab_smoke_test.ipynb` | The `ncu` check `rc == 0 and A or B` parsed as `(rc == 0 and A) or B`, which could report "WORKS" when `ncu` failed | Self-review before first use | Parenthesized the condition (commit `fix(tools): correct operator precedence in ncu check`) |
