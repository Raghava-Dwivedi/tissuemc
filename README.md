# TissueMC

**Validated Monte Carlo simulation of light transport in layered biological tissue, with a measured GPU optimization study.**

> **Status: under construction.** The CPU engine is built, validated, cross-checked against the original MCML and benchmarked. The first CUDA rung (R0, the naive baseline) has run on a T4 and is validated. Rung R1 (fewer atomics) is validated on the GPU and about 10% faster than R0; the hardware counters point at warp divergence as the main loss. Rung R3 (photon relaunch) raised the active lanes per warp from 4.74 to 21.61 of 32 but was about 6% slower than R1; a diagnostic run then put the wait on per-thread local memory. Rungs R5 and R6, which keep that state in registers, took 10.06 s and 10.05 s, the same as R1's 10.06 s: registers removed R3's loss but did not make the kernel faster than R1, so what limits R1 is not yet identified. R6's counters then pointed at the depth-profile atomics, and a diagnostic that removes them ([`docs/diagnostics.md`](docs/diagnostics.md)) is built but has not been run. Every number below traces to a committed file in [`results/`](results) and the command that produced it.

TissueMC reimplements the layered-tissue model of [MCML](https://omlc.org/software/mc/) (Wang, Jacques & Zheng, 1995) as:

- a multithreaded C++ CPU engine (float64), **done**, and
- a CUDA engine (float32 transport), **rung R0 done; later rungs in progress** (see [`docs/rungs/R0.md`](docs/rungs/R0.md) and [`LOG.md`](LOG.md)).

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

**V3 transmittance sits 2.6σ below the printed Table 6.7 value.** The cross-check below explains this: the original program itself, run at high statistics, gives a value above this engine's, and a controlled experiment points to the original's random number generator.

## Cross-check against the original MCML

The original program (Wang and Jacques, 1992–93) is not distributed here: its page states no license. Download the five source files from <https://omlc.org/software/mc/mcml/mcml-src/> into `third_party/mcml/` and run `uv run tissuemc-compare-mcml`. The tool checks the files against pinned SHA-256 values, compiles them, runs many independent runs inside one launch, checks that the program read the input as sent (its 1993 input reader returns the address of a local buffer, so every output is verified), and compares with this engine at equal photon counts. The pass rule was fixed before any comparison: |z| < 3 for every quantity.

**10 million photons per case: all 10 comparisons pass** ([`results/mcml-crosscheck.json`](results/mcml-crosscheck.json)).

| Case | Quantity | This engine | Original MCML | z |
|---|---|---|---|---|
| V1 thin slab | R_d | 0.097543 ± 0.000071 | 0.097420 ± 0.000071 | +1.23 |
| V1 thin slab | T_t | 0.660837 ± 0.000102 | 0.660890 ± 0.000100 | −0.37 |
| V1 thin slab | A | 0.241620 ± 0.000053 | 0.241690 ± 0.000051 | −0.94 |
| V2 semi-infinite, n = 1.5 | R_d | 0.219952 ± 0.000084 | 0.220049 ± 0.000093 | −0.77 |
| V2 semi-infinite, n = 1.5 | A | 0.740048 ± 0.000084 | 0.739951 ± 0.000093 | +0.77 |
| O1 semi-infinite, matched | R_d | 0.415034 ± 0.000087 | 0.414935 ± 0.000101 | +0.74 |
| O1 semi-infinite, matched | A | 0.584966 ± 0.000087 | 0.585065 ± 0.000101 | −0.74 |
| V3 three layers | R_d | 0.237732 ± 0.000104 | 0.237852 ± 0.000102 | −0.82 |
| V3 three layers | T_t | 0.096095 ± 0.000064 | 0.096238 ± 0.000060 | −1.64 |
| V3 three layers | A | 0.641800 ± 0.000096 | 0.641537 ± 0.000086 | +2.04 |

**That pass was too weak to count as agreement on V3.** With four times the photons, transmittance fails the same rule:

| Quantity | This engine | Original MCML, own generator (z) | Original MCML, generator swapped (z) |
|---|---|---|---|
| R_d | 0.237847 ± 0.000049 | 0.237933 ± 0.000051 (−1.23) | 0.237869 ± 0.000054 (−0.30) |
| T_t | 0.096055 ± 0.000027 | 0.096261 ± 0.000029 (−5.20) | 0.096076 ± 0.000030 (−0.53) |
| A | 0.641726 ± 0.000048 | 0.641434 ± 0.000047 (+4.34) | 0.641683 ± 0.000048 (+0.63) |

40,000,000 photons each ([`original generator`](results/mcml-crosscheck-v3-extended.json), [`swapped generator`](results/mcml-crosscheck-v3-xoshiro.json)). The three quantities are one difference seen three ways: weight leaving the bottom versus absorbed. What was established:

- The original gave the same T_t in two separate launches (0.096238 ± 0.000060 at 10⁷ photons and 0.096261 ± 0.000029 at 4×10⁷), so this is not a seed fluke.
- Splitting a homogeneous layer into two or three identical sublayers leaves this engine's results unchanged (z within ±1.2 at 10⁷ photons), so its layer-crossing logic is self-consistent.
- Rounding this engine's random draws to single precision, as the original's generator does, changes the totals by under 10⁻⁷ (4×10⁻⁸ on T_t), thousands of times smaller than the 2×10⁻⁴ gap. Precision alone is not the cause.
- Replacing **only** the random number generator in a copy of the original (`--original-rng xoshiro`, with xoshiro256++) gives T_t = 0.096076 ± 0.000030. That agrees with this engine and differs from the unmodified original at z = −4.5.

So the difference comes from the original program's random number generator, not from this engine's physics. **Why** that generator biases the result is not established. The 10⁶-photon run behind the MCML manual's Table 6.7 has a standard error of about 1.9×10⁻⁴ (estimated), so its published T_t of 0.0965 is consistent with that program's own long-run value plus ordinary noise. Result files without an `original_rng` field used the original generator.

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

## GPU, rung R0 (Tesla T4)

R0 is the deliberately naive baseline: one thread per photon, float32 transport, global atomic tallies (running totals in `double`). Design: [`docs/rungs/R0.md`](docs/rungs/R0.md). It ran on a free Colab Tesla T4 (CUDA 13.0) at commit `a3e9804`; the record is [`results/gpu-r0-summary.json`](results/gpu-r0-summary.json), the pasted final summary of [`notebooks/colab_gpu.ipynb`](notebooks/colab_gpu.ipynb). The per-file outputs from that session were not saved. Log: [`LOG.md`](LOG.md), which is measurement-only; **no prediction was recorded before this run**.

- **Validation:** all 24 checks pass on the GPU engine, the same published values and independent oracle as the CPU engine.
- **Agreement with the CPU on identical photons** (CPU float32 against GPU, same seed and batches; limits fixed before any GPU run):

| Gap | Measured | Limit |
|---|---|---|
| R_d total, in standard errors | 0.0013 | 0.25 |
| T_t total, in standard errors | 0.0088 | 0.25 |
| R_d worst per-batch relative gap | 0.000080 | 0.005 |
| T_t worst per-batch relative gap | 0.000243 | 0.005 |
| Event count, relative gap | 0.0000023 | 0.001 |

- **Float32 running totals, as a demonstration:** R_d and T_t still pass, but energy conservation fails (z = −82.5), so a float32 total is not safe for the absorbed weight. Measured on the host first: fine up to about 10⁴ photons per batch and 1.6% low at 4×10⁵.
- **Throughput**, 4,000,000 photons of the tissue-like workload in 240 batches, median kernel time 11.60 s over 5 repeats: **533.6 million events/s** (0.345 million photons/s). Kernel time excludes setup and copies.

The CPU figures from [`results/cpu-scaling.json`](results/cpu-scaling.json), on a different machine (Apple M4 Pro), are 33.4 million events/s on one thread and 253.5 on 12 threads, so R0 is about 2.1× the 12-thread CPU. That ratio is context, not a like-for-like speedup.

Caveats: one Colab session; the five kernel times rise steadily from 11.33 s to 11.90 s (+5%), cause not checked (GPU clock throttling is one possibility); no comparison with MCX yet; R0 is the reference point for later rungs, not a good kernel.

## GPU, rung R1 (fewer atomics)

R1 accumulates absorbed weight per layer inside each thread and adds it to the global total once per photon, instead of two global atomic adds per absorption event. That **halves** the global atomics per photon (the depth profile still adds one per event). Design: [`docs/rungs/R1.md`](docs/rungs/R1.md). Same Colab T4 session ran R0, then R1, then R0 again, so the GPU's own drift can be told apart from a real change; the record is [`results/gpu-r1-summary.json`](results/gpu-r1-summary.json) (the pasted summary of [`notebooks/colab_gpu_r1.ipynb`](notebooks/colab_gpu_r1.ipynb); that notebook, saved from Colab, also keeps the printed output of its validation, cross-check, benchmark and profiling cells; the per-file JSON results were not saved). **No prediction was recorded before this run**; see [`LOG.md`](LOG.md).

- **Validation:** all 24 checks pass on rung 1.
- **Throughput**, 4×10⁶ photons of the tissue-like workload, median kernel time of 5 repeats:

| Run | Median kernel time | Million events/s |
|---|---|---|
| R0 (first) | 11.274 s | 549.0 |
| **R1** | **10.263 s** | **603.1** |
| R0 (again) | 11.338 s | 545.9 |

**R1 is about 10% faster than R0** (11.306 s average of the two R0 runs over 10.263 s). R0 moved by 0.6% between its two runs, so the gain is well outside the session drift. The earlier R0 session measured 533.6 million events/s, 2 to 3% below this session's R0 runs, so absolute numbers vary between Colab sessions; only same-session comparisons are used for speedups.

- **Hardware counters** (Nsight Compute, 200,000 photons; ratios and percentages only, because profiling locks the GPU clocks):

| Counter | R0 | R1 |
|---|---|---|
| Avg. active threads per warp (of 32) | 4.75 | 4.75 |
| Compute (SM) throughput | 29.55 % | 33.34 % |
| Memory throughput | 10.98 % | 13.95 % |
| L2 cache throughput | 9.34 % | 11.98 % |
| Warp cycles per issued instruction | 22.83 cycle | 20.10 cycle |
| Achieved occupancy | 85.67 % | 86.05 % |

**What the counters say** (my reading, not something the profiler states): only **4.75 of 32 lanes are active per warp instruction (15%)**, the same in both rungs. That matches the 16% I estimated beforehand from the host-side distribution of steps per photon (mean 1,537, median 226, 90th percentile 7,386), which was an estimate and is now measured. Memory and L2 throughput are low (at most 14% and 12%), so global atomics are not what limits this kernel, which fits R1's modest gain. Occupancy is high (86%). Each warp waits about 20 cycles between instructions it issues. That pointed at divergence as the main loss, which the next rung (R3, below) tests; the reading did not hold up. The depth-profile privatization rung (R2) is deferred because memory and L2 are not busy.

## GPU, rung R3 (photon relaunch)

R3 keeps R1's scoring and changes how threads are used: a fixed set of resident threads, here 40,960 (the T4's maximum, 40 multiprocessors × 1,024), each claim photons from a shared counter and take the next one as soon as theirs ends, one step per loop iteration. Design: [`docs/rungs/R3.md`](docs/rungs/R3.md). The record is [`results/gpu-r3-summary.json`](results/gpu-r3-summary.json), the pasted summary of [`notebooks/colab_gpu_r3.ipynb`](notebooks/colab_gpu_r3.ipynb); the per-file JSON results were not saved. [`LOG.md`](LOG.md) is measurement-only, so no prediction was recorded.

- **Validation:** all 24 checks pass on rung 3.
- **Agreement with the CPU on identical photons** (2×10⁶ photons, float32 CPU against the GPU, limits fixed before any GPU run): both R1 and R3 pass. This run is the first to compare **absorbed weight**, the tally R1 changed: total gap 0.0034 standard errors for R1 and 0.00024 for R3, worst per-batch relative gap 4.0e-05 and 4.1e-05 (limits 0.25 and 5×10⁻³).
- **Throughput**, same session, 4×10⁶ photons, median kernel time of 5 repeats:

| Run | Median kernel time | Million events/s |
|---|---|---|
| R1 (first) | 10.639 s | 581.7 |
| **R3** | **11.338 s** | **545.9** |
| R1 (again) | 10.823 s | 571.8 |

**R3 was not faster than R1: it was about 5.7% slower** (11.338 s against 10.731 s, the average of the two R1 runs). All ten R1 repeats (10.35 to 11.11 s) were faster than all five R3 repeats (11.33 to 11.43 s), and R1 ran both before and after R3. R1's own first run spread by 7% across its five repeats, so the size of the gap is rough; that R3 gave no speedup is not.

- **Hardware counters** (Nsight Compute, 2×10⁶ photons rather than the 2×10⁵ used earlier, because R3's 40,960 lanes would otherwise get only about five photons each; ratios and percentages only, because profiling locks the GPU clocks):

| Counter | R1 | R3 |
|---|---|---|
| Threads launched | 2,000,000 | 40,960 |
| Registers per thread | 48 | 37 |
| Avg. active threads per warp (of 32) | 4.74 | 21.61 |
| Warp cycles per issued instruction | 19.95 cycle | 98.09 cycle |
| Compute (SM) throughput | 34.80 % | 8.12 % |
| Memory throughput | 14.48 % | 16.81 % |
| L2 cache throughput | 12.39 % | 11.65 % |
| Achieved occupancy | 86.97 % | 99.78 % |

**What the counters say** (my reading, not something the profiler states): R3 did what it was built to do, raising the active lanes per warp from 4.74 to 21.61 of 32 (15% to 68%). Run time did not follow. Each warp now waited 98 cycles between issued instructions instead of 20 (4.9×), which roughly cancels the 4.6× more lanes working in each instruction; under the profiler's locked clocks the two kernels took 8,924,395,584 and 8,962,183,072 ns, within 0.4% of each other (that run uses half the photons at lower, locked clocks, so it is not a second timing of the benchmark). Compute throughput fell from 34.80 % to 8.12 %, and memory and L2 stay under 17%. So lane utilization, memory bandwidth and atomic throughput did not explain R3's loss (the atomics matter a little: halving them gave R1's 10%). This also overturns my reading after R1 that divergence was the main loss: divergence was real, and R3 removed most of it, but removing it did not make the kernel faster. The diagnostic run below shows why each warp waits so long.

### Why each R3 warp waits so long

The diagnostic run collected every Nsight Compute counter for R1 and R3 on 500,000 photons ([`results/gpu-r1-r3-diagnosis.json`](results/gpu-r1-r3-diagnosis.json), the printed output of [`notebooks/colab_gpu_diagnose.ipynb`](notebooks/colab_gpu_diagnose.ipynb)). Ratios and percentages only, as before. The waits break down like this:

| Per issued instruction, warps stalled on | R1 | R3 |
|---|---|---|
| Total wait (average warp latency, cycles) | 19.93 | 82.90 |
| Long scoreboard (waiting for a load from local, global, surface or texture memory) | 12.39 | 62.42 |
| LG throttle (the local/global load-store queue is full) | 2.08 | 14.69 |
| Every other reason together | 5.50 | 5.83 |

| Counter | R1 | R3 |
|---|---|---|
| Warp instructions executed (billions) | 72.33 | 20.17 |
| Average active lanes per instruction | 4.73 | 18.16 |
| Issue slots used (% of peak while active) | 34.88 | 9.59 |
| Eligible warps per cycle | 0.51 | 0.15 |
| Global or shared loads and stores | 0 | 0 |
| Local-memory load instructions (billions) | 0.91 | 0.94 |
| Local-memory store instructions (billions) | 1.50 | 0.79 |
| Local-memory sectors requested (billions) | 8.38 | 8.76 |
| L1 hit rate, local loads (%) | 89.95 | 29.02 |
| L1 hit rate, local stores (%) | 95.66 | 66.92 |
| Register-spill requests | 0 | 0 |

**What the counters say** (my reading):

- **The extra wait is almost entirely the load/store path.** Long scoreboard plus LG throttle make up 73% of R1's wait and 93% of R3's, and account for 99.5% of the 63.0-cycle increase. Arithmetic, branches and the other stall reasons barely moved.
- **That load/store traffic is the threads' own local memory.** Neither rung issues any global or shared loads or stores; the tissue geometry is read from constant memory and the tallies go through atomics, which fell with the instruction count (to 0.23×). The profiler counts no register spills, so this is not spilled registers: it is per-thread arrays or structures that the compiler keeps in stack memory.
- **R3 does not reduce that traffic, and it misses the L1 cache far more.** R3 executes 0.28× the warp instructions of R1, but the same number of local loads and a similar volume of local sectors, so local memory is 3.7× as large a share of what each warp does. The L1 hit rate for local loads fell from 89.95% to 29.02%, so about 10× as many local-load sectors went on to L2 (0.36 to 3.60 billion). A miss waits for L2 instead of L1, which is a long scoreboard stall.
- **The earlier assumption that R3 does the same work per photon holds to about 7%.** Warp instructions times active lanes gives 1.07× the thread-instructions for R3, and instructions times wait per instruction divided by resident warps predicts the two kernels' run times within 2% of each other, which matches the profiler's near-equal durations at 2×10⁶ photons.

What the counters do **not** say: which variables live in local memory (I suspect the random generator's four-word buffer, which is indexed at run time, and the per-layer accumulator array that R1 and R3 index by layer; that is from reading the code, not from a measurement), or why the L1 hit rate fell. Rungs R5 and R6 ([`docs/rungs/R5.md`](docs/rungs/R5.md), results in the next section) tested the first of these directly.

## GPU, rungs R5 and R6 (state in registers)

R5 (one photon per thread) and R6 (persistent lanes that relaunch) keep the random generator's buffer and the per-layer accumulator in registers instead of run-time-indexed arrays, which the compiler had placed in per-thread local memory. With R1 and R3 they form a two-by-two comparison: arrays against registers, one photon per thread against relaunch. Design: [`docs/rungs/R5.md`](docs/rungs/R5.md). The record is [`results/gpu-r5-r6-summary.json`](results/gpu-r5-r6-summary.json), the pasted summary of [`notebooks/colab_gpu_r5.ipynb`](notebooks/colab_gpu_r5.ipynb); its profiling output was not pasted, so there are no counters for these rungs yet. [`LOG.md`](LOG.md) is measurement-only, so no prediction was recorded.

- **Validation:** all 24 checks pass on rung 5 and on rung 6.
- **Agreement with the CPU on identical photons** (2×10⁶ photons): both pass. R_d, T_t and event-count gaps are identical to R1's to every printed digit, as expected from the same arithmetic; absorbed weight, scored differently, differs from the CPU by 0.0034 standard errors in total (R5) and 0.0034 (R6), with a worst per-batch gap of 4.0e-05 and 4.0e-05 (limits 0.25 and 5×10⁻³).
- **Throughput**, same session, 4×10⁶ photons, median kernel time of 5 repeats:

| Run | Median kernel time | Million events/s |
|---|---|---|
| R1 (first) | 10.060 s | 615.2 |
| R3 | 10.764 s | 575.0 |
| **R5** | **10.059 s** | **615.3** |
| **R6** | **10.052 s** | **615.7** |
| R1 (again) | 10.486 s | 590.2 |

**Keeping the state in registers did not make the kernel faster, and it undid R3's loss.** R5 took 10.059 s against R1's 10.060 (-0.01%), and R6 took 10.052 s (-0.08%). R3 was 7.0% slower than R1, and R6, which is R3 with its state in registers, is 7.1% faster than R3. R1 repeated at the end of the session was 4.2% slower than at the start, so the GPU itself drifts by about that much; R5 and R6 ran in the same stretch as the first R1, and the five repeats of R1, R5 and R6 each span only 0.02%, 0.09% and 0.02%, so the three equalities are tighter than the drift.

**What this means** (my reading):

- **R3's slowdown was caused by local memory, as the diagnostic run indicated.** Moving its state into registers (R6) brings it back to R1's time.
- **Local memory was not what limited R1.** With it gone (R5), the time did not move. My earlier statement that local-memory latency was the limit on this kernel was true of R3's *extra* wait and wrong as a statement about the kernel.
- **Something R1, R5 and R6 share sets their run time.** They differ in local-memory use, in how many warp instructions they issue (R6 issues far fewer than R1, from the R3 counters) and in whether they relaunch, yet they take the same time to within 0.1%. What they share is not identified.
- **Candidates, untested:** the per-event atomic add into the depth profile (about 6×10⁹ per run, the same in all three; the hot bins near the surface receive many of them, and a profile's average L2 busy percentage would not show one hot address serializing); other per-event work that is the same in every kernel. R1's reading that atomics were not the limit rested on L2 being only about 12% busy, which does not measure that, so the question is open again. R6's counters, in the next subsection, point at the first.

### What R6's counters show

The saved profiles of R3 and R6 (500,000 photons, every counter; [`results/gpu-r3-r6-digest.json`](results/gpu-r3-r6-digest.json)) say where R6's time goes now that its local memory is gone:

| Counter | R3 | R6 |
|---|---|---|
| Local-memory load instructions (billions) | 0.94 | 0.00 |
| Local-memory store instructions (billions) | 0.79 | 0.00 |
| DRAM bytes written (gigabytes) | 70.7 | 0.000007 |
| Warp instructions executed (billions) | 20.19 | 18.62 |
| Average warp latency per issued instruction (cycles) | 85.28 | 89.84 |
| Waiting for room in the load/store queue (LG throttle) | 15.22 | 70.36 |
| Waiting for a load result (long scoreboard) | 64.31 | 10.48 |
| Atomic add instructions (billions) | 0.035 | 0.035 |
| Busiest L2 slice, throughput (% of peak) | 63.64 | 63.78 |
| Average over L2 slices, throughput (% of peak) | 12.49 | 2.08 |
| Busiest L2 slice, atomic unit busy (%) | 33.43 | 33.96 |
| Kernel duration at locked clocks (seconds) | 2.297 | 2.258 |

**What the counters say** (my reading):

- **R6's local memory is gone, and the wait did not shrink; it moved.** Local loads and stores are zero and DRAM traffic vanished, yet the wait per instruction went from 85.28 to 89.84 cycles. The long-scoreboard wait fell by 53.8 cycles and the LG-throttle wait rose by 55.1: warps now wait for room in the load/store queue, which is 78% of their wait.
- **The only load/store traffic left is atomic adds.** R6 issues no global or shared loads or stores, and its atomic count (0.035 billion warp-level instructions, now compiled as reductions) equals R3's (0.035 billion). By construction almost all of them are the per-event depth-profile adds: about 1,500 per photon, against about 6 other atomic adds per photon.
- **One part of L2 is far busier than the rest, and removing the local memory did not change it.** The busiest L2 slice runs at 63.78% of peak while the average over slices is 2.08% (31×), with its atomic unit busy 33.96% of the time; in R3 the busiest slice was at 63.64% even though R3 sent 8.8 billion extra sectors to L2. Something that did not change between the kernels sets that slice's load, and the atomics are what did not change.
- **Taken together:** the run time looks set by many atomic adds to a few depth bins, serialized at one L2 slice. That would explain why R1, R5 and R6 take the same time whatever their local-memory use or instruction count, and why R3's extra local-memory traffic cost it a further 7%. The streaming multiprocessors are evenly loaded (the busiest was active 1.1% longer than the average), so this is not a tail effect.

This is a reading of counters that agree with each other, not a test. The test is to remove the atomics and see whether the time drops; that is the next experiment.

## Development process

The author designs and specifies the project, runs the benchmarks, and reviews the results. [`LOG.md`](LOG.md) is measurement-only: for each GPU rung it records what was measured and what the data suggest, with hypotheses labelled as such and checked by the next run. No forecasts are recorded; R0 and R1 were run without one, and the author chose measurement-only from then on. Implementation is AI-assisted, under that review. [`BUILD_LOG.md`](BUILD_LOG.md) records the defects found and fixed along the way, as they happened. [`VALIDATION.md`](VALIDATION.md) maps each claim to the independent check behind it.

## License

MIT. See [LICENSE](LICENSE).
