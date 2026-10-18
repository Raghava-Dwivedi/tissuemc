# Optimization log

This log is **measurement-only**: for each rung of the GPU study it records what changed, what was measured, and what the data show or suggest. No forecast is written before a run. The first plan was to record a prediction before each run; none was recorded for R0 or R1, and on 2026-10-18 the author chose measurement-only instead. The R0 and R1 entries below keep their original wording.

A hypothesis written after one run is labelled as one, and the next run says whether it held. Nothing is edited after the fact; corrections go in as new dated notes.

## R0: naive baseline

*Design:* [`docs/rungs/R0.md`](docs/rungs/R0.md) · *Run:* free Colab T4, 2026-10-14, commit `a3e9804` · *Data:* [`results/gpu-r0-summary.json`](results/gpu-r0-summary.json)

**Prediction (written before the run):** **none was recorded.** The run was made before a prediction was written down, so there is nothing to compare the result with, and none is written afterwards: a prediction made after seeing the number would not be one. From R1 on, the prediction goes in before each run and is informed by the measurements of the rung before.

**Result:**
- Validation on the GPU: 24 of 24 checks pass.
- On identical photons the GPU agrees with the CPU's float32 path to 0.0013 standard errors on R_d and 0.0088 on T_t, worst per-batch gap 8×10⁻⁵ and 2.4×10⁻⁴, event counts to 2.3×10⁻⁶. The limits (0.25, 5×10⁻³, 10⁻³) were fixed before any GPU run.
- Float32 running totals, as a demonstration: R_d and T_t still pass, but energy conservation fails (z = −82.5), as the host emulation predicted.
- Throughput, 4×10⁶ photons of the tissue-like workload, median kernel time 11.60 s over 5 repeats: **533.6 million events/s** (0.345 million photons/s). The CPU numbers on a different machine, for context only: 33.4 on one thread and 253.5 on 12 (M4 Pro).
- The five kernel times rise steadily, 11.33 → 11.90 s (+5%). The cause was not checked; GPU clock throttling as the card heats is one possibility.

**Did the prediction hold, and why?** Not applicable (no prediction). What the data suggest, as hypotheses for the next rungs and not as findings:
- R0 issues two global atomic adds per absorption event. At 6.19×10⁹ events in 11.6 s that is about 10⁹ atomic adds per second, which may be limiting. Absorbed weight per layer needs only one add per photon if it is accumulated in the thread first: rung R1.
- Photons differ hugely in length, so warps probably idle a lot: rung R3.

## R1: fewer atomics

*Design:* [`docs/rungs/R1.md`](docs/rungs/R1.md) · *Notebook:* [`notebooks/colab_gpu_r1.ipynb`](notebooks/colab_gpu_r1.ipynb) · *Record:* [`results/gpu-r1-summary.json`](results/gpu-r1-summary.json)

**Prediction (written before the run):** **none was recorded.** As with R0, the run was made before a prediction was written down, and none is written afterwards.

**Result:**
- Validation on the GPU, rung 1: 24 of 24 checks pass.
- Same photons through the CPU (float32) and rung 1: gaps of 0.0013 standard errors on R_d and 0.0088 on T_t, worst per-batch 8×10⁻⁵ and 2.4×10⁻⁴, event counts 2.3×10⁻⁶. These are **identical to every printed digit** to R0's gaps. R1 did not touch the R_d or T_t tallies, so this comparison says nothing about the one tally R1 changed (absorbed weight per layer). That was a gap in the check, found after the run; the check now also compares absorbed weight per batch. It has not yet been run on a GPU. Until it is, R1's absorbed weight is supported only by the energy-conservation checks in the validation suite.
- Throughput, same session, 4×10⁶ photons, median kernel time of 5 repeats: R0 11.274 s (549.0 million events/s), **R1 10.263 s (603.1)**, R0 again 11.338 s (545.9). R1 is about 10% faster than the mean of the two R0 runs; R0 itself moved 0.6% between its runs.
- Hardware counters (Nsight Compute, 200,000 photons): active threads per warp 4.75 of 32 in both rungs; memory throughput 11% (R0) and 14% (R1); L2 9% and 12%; compute throughput 30% and 33%; warp cycles per issued instruction 22.8 and 20.1; achieved occupancy 86% in both.

**Did the prediction hold, and why?** Not applicable (no prediction). What the data show and suggest:
- Measured: R1 halves the global atomics per photon (about 3,000 to about 1,500: the per-layer total is now one add per photon, the depth profile is still one add per event) and the kernel is about 10% faster.
- My hypothesis after R0 was that global atomics, at about 10⁹ per second, might be limiting. The counters weigh against that as the main limit: memory and L2 are at most 14% and 12% busy. Halving the atomics helped somewhat, so they cost something, but not most of the time.
- Measured: only 4.75 of 32 lanes are active per warp instruction (15%). I had estimated about 16% beforehand from the distribution of steps per photon (mean 1,537, median 226, 90th percentile 7,386, maximum 18,406), because a warp runs as long as its longest photon. This is the largest loss the counters show, and the same in R0 and R1.
- Hypothesis for the next rung, not a finding: photon relaunch (R3) should recover much of that. It is therefore built before R2 (shared-memory histograms), which targets memory traffic the counters show is not busy. R2 is not dropped, only deferred.

## R3: photon relaunch (built before R2)

*Design:* [`docs/rungs/R3.md`](docs/rungs/R3.md) · *Notebook:* [`notebooks/colab_gpu_r3.ipynb`](notebooks/colab_gpu_r3.ipynb) · *Record:* [`results/gpu-r3-summary.json`](results/gpu-r3-summary.json)

**Prediction:** none, by design (measurement-only; see the note at the top).

**Why this rung:** R1's counters showed memory and L2 at most 14% and 12% busy and only 4.75 of 32 lanes active per warp, so the largest measured loss looked like idle lanes, not memory traffic. The hypothesis recorded after R1 was that relaunch should recover much of that.

**Result:**
- Validation on the GPU, rung 3: 24 of 24 checks pass.
- Same photons through the CPU (float32), 2×10⁶ photons, now including absorbed weight: R1 passes (absorbed total gap 0.0034 standard errors, worst per-batch 4.0e-05) and R3 passes (0.00024, 4.1e-05); R_d, T_t and event counts agree within the limits fixed before any GPU run (event gap 2.3e-06 for R1, 2.2e-06 for R3). **This closes the gap noted in the R1 entry:** R1's absorbed weight is now compared with the CPU per batch, not only through energy conservation. R3's gaps differ slightly from R1's; why was not checked (different generated code rounding some operations differently is one possibility).
- Throughput, same session, 4×10⁶ photons, median kernel time of 5 repeats: R1 10.639 s (581.7 million events/s), **R3 11.338 s (545.9)**, R1 again 10.823 s (571.8). R3 is about 5.7% slower than the mean of the two R1 runs. All ten R1 repeats (10.35 to 11.11 s) beat all five R3 repeats (11.33 to 11.43 s); R1's first run alone spread by 7%, so the size of the gap is rough.
- Hardware counters (Nsight Compute, 2×10⁶ photons): threads launched 2,000,000 (R1) and 40,960 (R3, the T4's maximum of 40 × 1,024); registers per thread 48 and 37; active threads per warp 4.74 and 21.61 of 32; warp cycles per issued instruction 19.95 and 98.09; compute throughput 34.80 % and 8.12 %; memory 14.48 % and 16.81 %; L2 12.39 % and 11.65 %; occupancy 86.97 % and 99.78 %; kernel duration at locked clocks 8,924,395,584 and 8,962,183,072 ns (+0.4%).

**What the data show and suggest:**
- Measured: R3 raised active lanes per warp 4.6× (15% to 68% of 32) and the kernel did not get faster. The hypothesis recorded after R1 held for lane utilization and **did not hold for run time**.
- Measured: each warp waited 4.9× longer between issued instructions, which roughly cancels the 4.6× more lanes per instruction. If the work per photon is unchanged (assumed, not measured), R3 issues about 4.6× fewer warp instructions that each take about 4.9× longer.
- Measured: compute throughput fell to 8.12 % and memory and L2 stay under 17%, so no throughput limit is being hit in either rung. Occupancy is already near its maximum, so more resident warps are not available.
- Not known: why each warp waits so long. The counters collected here do not include stall reasons. One candidate, untested: per-thread state held in local memory (R3 uses 37 registers per thread against R1's 48, which could mean more state is kept in local memory; not checked).
- This overturns the reading in the R1 entry that divergence was the main loss.
- Next measurement, not another rung: collect warp stall reasons and local-memory counters for R1 and R3 (`notebooks/colab_gpu_diagnose.ipynb`). R2 stays deferred.
