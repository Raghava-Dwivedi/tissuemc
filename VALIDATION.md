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
| Three-layer R_d agrees | MCML and Gardner et al. (Table 6.7) | `tissuemc-validate`, CI | z = +1.26 (MCML) |
| Three-layer T_t agrees | MCML and Gardner et al. (Table 6.7) | `tissuemc-validate`, CI | **z = −2.58 (MCML); passes the 3σ rule but is the weakest result. Open.** |
| Energy is conserved | Weight balance, within standard error | every validation run | passes in all cases |
| No invalid photon state occurs | Unit-direction and finite-weight checks at every scoring call | every validation run | 0 anomalies |
| Results do not depend on thread count | Bitwise comparison at 1, 2, 3, 8, 24 and auto threads | `tissuemc_tests`, CI | identical |
| The threaded engine has no data races | ThreadSanitizer | CI | clean |
| The float32 transport path (the one the GPU runs) is correct | The full validation suite on `--precision float32`, plus float32 against float64 on identical photons (totals agree to about 1% of their standard error on V3) | `tissuemc-validate --precision float32`, `tissuemc_tests`, CI | 24/24 checks pass; V3 R_d differs by 2.8e-6 against a standard error of 3.0e-4 |
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

- **V3 transmittance** is lower than both published values (see above and `BUILD_LOG.md`). The planned check is to run the original `mcml.c` on the same input.
- The published V3 values come without standard errors. Their errors are estimated from this engine's per-photon variance, scaled by the photons behind each value.
- The GPU engine does not exist yet. Nothing here validates GPU results.
- All performance numbers are from one machine.
