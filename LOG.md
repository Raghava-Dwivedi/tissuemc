# Optimization log

For each rung of the GPU study, the **prediction is written before the run**: what is expected to change, by roughly how much, and why. After the run the measurement goes next to it, with whether the prediction held and what explains the difference.

A prediction that was wrong is as useful as one that was right. Nothing is edited after the fact; corrections go in as new dated notes.

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
