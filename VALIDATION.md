# Validation

Each claim made in this repository, the check that supports it, and where that check runs. A check counts only if it does not depend on code written alongside the thing it checks.

## How this was built

The author designed and specified the project and reviews every result. The implementation is AI-assisted. Two kinds of independent check keep that honest: published reference values, and a first-principles calculation that shares no code with the engine. Tests written alongside the code are listed separately and are not treated as validation on their own.

Written by hand by the author without AI: *nothing yet.*

## Claims and the checks behind them

| Claim | Independent check | Where it runs | Result |
|---|---|---|---|
| Thin-slab R_d and T_t agree with published values | van de Hulst 1980, MCML and Prahl et al. 1989 (MCML manual, Table 6.1) | `tissuemc-validate`, CI | R_d z = +0.15, T_t z = +0.11 against van de Hulst |
| Semi-infinite reflectance with a refractive-index mismatch agrees | Giovanelli 1955, MCML, Prahl et al. (Table 6.2) and the H-function calculation | `tissuemc-validate`, CI | total z = −0.99 (Giovanelli); diffuse z = +1.06 (H-function) |
| Matched-boundary semi-infinite reflectance is correct | H-function calculation (no published table involved) | `tissuemc-validate`, CI | z = −0.37 |
| Three-layer R_d agrees | MCML and Gardner et al. (Table 6.7); the original program at 4×10⁷ photons | `tissuemc-validate`, `tissuemc-compare-mcml` | z = +1.26 (Table 6.7), z = −1.23 (original program) |
| Thin slab and semi-infinite cases agree with the original program | Original MCML at 1×10⁷ photons, 40 independent runs | `tissuemc-compare-mcml` (needs the sources, which are not distributed) | all within 3σ (|z| ≤ 1.23). Too weak to be conclusive for differences below about 2×10⁻⁴ |
| Three-layer T_t agrees | MCML manual Table 6.7 values; the original program itself run at 4×10⁷ photons; the same program with only its random number generator replaced | `tissuemc-validate`, `tissuemc-compare-mcml`, CI (validation only) | 2.6σ below the printed Table 6.7 value, and **5.2σ below the unmodified original program**; **agrees with the original with a modern generator (z = −0.53)**. The difference comes from the original's generator; why is not established (see README and `BUILD_LOG.md`) |
| Energy is conserved | Weight balance, within standard error | every validation run | passes in all cases |
| No invalid photon state occurs | Unit-direction and finite-weight checks at every scoring call | every validation run | 0 anomalies |
| Results do not depend on thread count | Bitwise comparison at 1, 2, 3, 8, 24 and auto threads | `tissuemc_tests`, CI | identical |
| The threaded engine has no data races | ThreadSanitizer | CI | clean |
| The float32 transport path (the one the GPU runs) is correct | The full validation suite on `--precision float32`, plus float32 against float64 on identical photons (totals agree to about 1% of their standard error on V3) | `tissuemc-validate --precision float32`, `tissuemc_tests`, CI | 24/24 checks pass; V3 R_d differs by 2.8e-6 against a standard error of 3.0e-4 |
| The GPU engine (rung R0) is correct | The full validation suite on a T4; the CPU's float32 path on identical photons, with limits fixed before any GPU run | `tissuemc-validate --engine cuda`, `tissuemc-cross-engine` (Colab) | 24/24 checks pass; gaps 0.0013 and 0.0088 standard errors on R_d and T_t, event counts to 2.3e-6. Record: `results/gpu-r0-summary.json` (pasted summary only) |
| The GPU engine (rung R1) is correct | The full validation suite on a T4 with `--rung 1`; the CPU's float32 path on identical photons | `tissuemc-validate --engine cuda --rung 1`, `tissuemc-cross-engine --b-rung 1` (Colab) | 24/24 checks pass; R_d and T_t gaps identical to R0's (R1 does not change those tallies). Absorbed weight per layer, the tally R1 changed, was first checked only through energy conservation; the per-batch comparison with the CPU was added afterwards and run with R3: total gap 0.0034 standard errors, worst batch 4.0e-05 (limits 0.25 and 5e-3). Record: `results/gpu-r3-summary.json` and `results/gpu-r1-summary.json` (pasted summary) and the printed cell outputs saved in `notebooks/colab_gpu_r1.ipynb` |
| The GPU engine (rung R3) is correct | The full validation suite on a T4 with `--rung 3`; the CPU's float32 path on identical photons, including absorbed weight; and host emulation of the lane logic with interleaved virtual lanes (one lane equals R1 bit for bit) | `tissuemc-validate --engine cuda --rung 3`, `tissuemc-cross-engine --b-rung 3` (Colab), `tissuemc_tests` | 24/24 checks pass; absorbed total gap 0.00024 standard errors, worst batch 4.1e-05. Record: `results/gpu-r3-summary.json` (pasted summary) |
| The GPU engine (rungs R5 and R6) is correct | The full validation suite on a T4 with `--rung 5` and `--rung 6`; the CPU's float32 path on identical photons including absorbed weight; and host emulation (R6 with one lane equals R5 bit for bit; per-batch equality under interleavings; the layer-change accumulator tested directly) | `tissuemc-validate --engine cuda --rung 5|6`, `tissuemc-cross-engine --b-rung 5|6` (Colab), `tissuemc_tests` | 24/24 checks pass on each; R_d, T_t and event gaps identical to R1's; absorbed total gap 0.0034 (R5) and 0.0034 (R6) standard errors, worst batch 4.0e-05 and 4.0e-05. Record: `results/gpu-r5-r6-summary.json` (pasted summary) |
| A float32 running total is not safe for absorbed weight | Energy conservation with `--gpu-tally float32` on the T4, and the host emulation | `tissuemc-validate --engine cuda --gpu-tally float32`, `tissuemc_tests` | conservation fails (z = −82.5); R_d and T_t unaffected |
| The random number generator is correct | Known-answer vectors from the Random123 distribution | `tissuemc_tests`, CI | match |

## Tests written alongside the code

These catch regressions but do not independently show correctness: unit tests of Fresnel reflectance (against the textbook formulas), the Henyey–Greenstein sampler (against its analytic moments), the step-length distribution, a purely absorbing slab (against its closed-form solution, in float64 and float32), the input parser and the engine's batching.

## Defects caught

See [BUILD_LOG.md](BUILD_LOG.md), written as the work happened. Entries to date:

- An energy-conservation test failed because of naive floating-point summation, not the physics (resolved with compensated summation).
- Exit directions at refracting boundaries were not unit vectors (fixed).
- MCML's rotation formula amplifies direction-norm error for steep directions (fixed by renormalizing after each scatter).
- Published V2 values are total, not diffuse, reflectance (reference interpretation corrected, with evidence).

## Limits and open items

- The original MCML program is not distributed here; `tissuemc-compare-mcml` needs its five source files in `third_party/mcml/`. Its checksums were recorded from the copy first compared and guard against accidental change, not against a tampered first download.
- The 10⁷-photon comparison with the original was too weak to detect a 2×10⁻⁴ difference in V3 transmittance that the 4×10⁷-photon run found. That difference is attributed to the original's random number generator through a controlled swap. The mechanism inside that generator was not investigated.
- The published V3 values come without standard errors. Their errors are estimated from this engine's per-photon variance, scaled by the photons behind each value.
- There is no first-principles reference for the layered case, so V3 rests on published values and on agreement with the original program once its generator is replaced.
- GPU results come from one free Colab T4 session; only the pasted summary was kept, not the per-file outputs. The GPU engine is validated for R0, R1, R3, R5 and R6; each later rung must pass again before its speed counts.
- All performance numbers are from one machine.
