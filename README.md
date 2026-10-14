# TissueMC

**Validated Monte Carlo simulation of light transport in layered biological tissue, with a measured GPU optimization study.**

> **Status: under construction.** The CPU engine is built, validated and benchmarked. The CUDA engine and the GPU optimization study are not started; no GPU results are claimed. Every number below traces to a committed file in [`results/`](results) and the command that produced it.

TissueMC reimplements the layered-tissue model of [MCML](https://omlc.org/software/mc/) (Wang, Jacques & Zheng, 1995) as:

- a multithreaded C++ CPU engine (float64), **done**, and
- a CUDA engine (float32), **planned**.

Both engines share one physics code path (`include/tissuemc/physics.hpp`). Nothing is benchmarked until it passes validation against published reference values. [`docs/DESIGN.md`](docs/DESIGN.md) has the full design.

This is a reimplementation for study and measurement, not new science: GPU photon-transport codes already exist (CUDAMCML, MCX). What this repository adds is correctness checked against published values and an independent first-principles calculation, plus a documented performance study.

## Quick start

```bash
uv sync                                  # Python analysis tools + build tools (cmake, ninja)
uv run cmake -S . -B build -G Ninja      # configure
uv run cmake --build build               # build engine + tests
uv run ctest --test-dir build --output-on-failure
uv run pytest
uv run tissuemc-validate                 # validation suite, ~20 s
./build/tissuemc run configs/validation/v1_slab.mci --photons 1000000 --threads 0
```

## Validation (CPU engine)

`uv run tissuemc-validate` runs four cases with 4 million photons each and compares them with published values and with an independent calculation ([`oracle.py`](python/tissuemc_analysis/oracle.py), Chandrasekhar's H-function). It passes if every estimate is within 3 combined standard errors of the reference plus the reference's rounding. The pass rule was fixed before the comparisons were run.

Headline results (seed 1, 4,000,000 photons, 50 batches; all 24 checks pass; full list in [`results/validation-cpu.json`](results/validation-cpu.json)):

| Case | Quantity | This engine | Reference | z | Reference source |
|---|---|---|---|---|---|
| V1 thin slab, matched boundary | R_d | 0.09741 ± 0.00011 | 0.09739 | +0.15 | van de Hulst (MCML manual, Table 6.1) |
| V1 thin slab, matched boundary | T_t | 0.66098 ± 0.00018 | 0.66096 | +0.11 | van de Hulst (Table 6.1) |
| V2 semi-infinite, n = 1.5 | R_d + R_sp | 0.25986 ± 0.00014 | 0.26000 | −0.99 | Giovanelli (Table 6.2) |
| V2 semi-infinite, n = 1.5 | R_d | 0.21986 ± 0.00014 | 0.21970 | +1.06 | H-function calculation |
| O1 semi-infinite, matched | R_d | 0.41488 ± 0.00018 | 0.41495 | −0.37 | H-function calculation |
| V3 three layers | R_d | 0.23796 ± 0.00016 | 0.23750 | +1.26 | MCML (Table 6.7) |
| V3 three layers | **T_t** | **0.09600 ± 0.00009** | 0.09650 | **−2.58** | MCML (Table 6.7) |

Each run also checks that energy is conserved and that no photon ends with a non-finite weight or a non-unit direction.

**Known issue: V3 transmittance is low against the published values.** It passes the pre-set 3σ rule, but at z between −2.1 and −2.9 across seeds against both published values (0.0965 and 0.0974). Over 8 seeds and 3.2×10⁷ photons the engine gives T_t = 0.09599 ± 0.00003. The published values' errors are estimated, not stated. I do not yet know whether this is chance in the old published runs or a small real difference; the planned check is to run the original `mcml.c` on the same input. Details are in [`BUILD_LOG.md`](BUILD_LOG.md).

## CPU scaling

Workload: [`configs/bench/tissue_g09.mci`](configs/bench/tissue_g09.mci), a tissue-like semi-infinite medium (μa = 0.1, μs = 100 cm⁻¹, g = 0.9, n = 1.37), about 1,550 absorption/scattering events per photon. 300,000 photons in 240 batches; median of 5 runs after 1 warm-up. The build was validated first, and results were identical at every thread count (the engine guarantees this).

| Threads | Million events/s | Speedup | Efficiency |
|---|---|---|---|
| 1 | 33.4 | 1.00 | 100% |
| 2 | 63.4 | 1.90 | 95% |
| 4 | 123.0 | 3.68 | 92% |
| 8 | 232.7 | 6.97 | 87% |
| 12 | 253.5 | 7.59 | 63% |

![CPU scaling](docs/img/cpu-scaling.png)

Hardware and build: Apple M4 Pro (8 performance + 4 efficiency cores, 24 GB), Apple clang 17.0.0, `-O3`, commit `d354c78`. Raw data: [`results/cpu-scaling.json`](results/cpu-scaling.json). Reproduce with `uv run tissuemc-bench`.

Caveats, as recorded in the result file:

- macOS cannot pin threads to cores. Threads beyond 8 also run on the slower efficiency cores, which is why 12 threads gain only 9% over 8.
- About 1.1 cores of unrelated background activity were running throughout (total background CPU use was 110% before and 104% after), so the 12-thread figure is somewhat pessimistic.
- One machine, one compiler, one workload.

## Development process

The author designs and specifies the project, writes each prediction before it is tested, runs the benchmarks, and reviews the results. Implementation is AI-assisted, under that review. [`BUILD_LOG.md`](BUILD_LOG.md) records the defects found and fixed along the way, as they happened. [`VALIDATION.md`](VALIDATION.md) maps each claim to the independent check behind it.

## License

MIT. See [LICENSE](LICENSE).
