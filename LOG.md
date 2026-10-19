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

## R3 diagnosis: why each warp waits so long (recorded 2026-10-18)

*Notebook:* [`notebooks/colab_gpu_diagnose.ipynb`](notebooks/colab_gpu_diagnose.ipynb) · *Record:* [`results/gpu-r1-r3-diagnosis.json`](results/gpu-r1-r3-diagnosis.json)

**Prediction:** none (measurement-only). This follows the R3 entry, which left the cause of the long waits open.

**Result** (every Nsight Compute counter, R1 against R3, 500,000 photons, locked clocks): average warp latency 19.93 to 82.9 cycles per issued instruction; long scoreboard 12.39 to 62.42 and LG throttle 2.08 to 14.69 warps stalled per issue; warp instructions 72.33 to 20.17 billion; active lanes 4.73 to 18.16; issue slots used 34.88% to 9.59%; global and shared loads and stores 0 in both; local loads 0.91 to 0.94 billion and local stores 1.50 to 0.79 billion instructions; L1 hit rate for local loads 89.95% to 29.02%; register-spill requests 0 in both.

**What the data show and suggest:**
- Measured: long scoreboard plus LG throttle are 72.6% of R1's wait and 93.0% of R3's, and 99.5% of the increase. The extra wait is the load/store path.
- Measured: the only data traffic through that path is per-thread local memory (plus atomics, which fell with the instruction count). The profiler reports no register spilling, so the local memory is not spilled registers.
- Measured: R3 issues 0.28× the warp instructions but the same number of local-load instructions, and local-load sectors that missed L1 rose about 10× (0.36 to 3.60 billion). The collapse in L1 hit rate (local loads 89.95% to 29.02%) turns L1 hits into L2 round trips, which is what long scoreboard measures.
- Derived: thread-instructions (warp instructions times active lanes) are 1.07× R1's, so the work per photon is unchanged to about 7%, and instructions times wait over resident warps predicts run times within 2% of each other, matching the profiler's near-equal durations at 2×10⁶ photons. This checks the assumption stated in the R3 entry.
- This supports the candidate named in the R3 entry in one respect (the waits are local-memory waits) and not in another: the argument from register counts (37 against 48) is not supported, since nothing is spilled.
- Not known: which variables are in local memory, and why the L1 hit rate fell. Candidates from reading the code, untested: the random generator's four-word buffer, indexed at run time, and the per-layer accumulator array, indexed by layer (R1 and R3 only).
- Next: a rung that removes the run-time-indexed per-thread arrays from the hot loop, judged by local-memory instruction counts, stall reasons and run time, not by run time alone.

## R5 and R6: per-thread state in registers

*Design:* [`docs/rungs/R5.md`](docs/rungs/R5.md) · *Notebook:* [`notebooks/colab_gpu_r5.ipynb`](notebooks/colab_gpu_r5.ipynb)

**Prediction:** none (measurement-only).

**Why these rungs:** the diagnosis above put the extra wait on per-thread local memory. Reading the code, two things are indexed at run time and so live in local memory: the random generator's four-word buffer and the per-layer accumulator array (R1 and R3). R5 (one photon per thread) and R6 (persistent lanes) keep both in registers. With R1 and R3 they form a two-by-two comparison of arrays against registers and one photon per thread against relaunch.

**Before any GPU run, from the continuous-integration build only:** the compiler's stack frame fell from 224 and 272 bytes to 32, and the static count of local-memory instructions from 149 and 197 to 10 (CUDA 12.0 toolkit; static, so it shows what was emitted, not what runs).

**Result:**
- Validation on the GPU: 24 of 24 checks pass on rung 5 and 24 of 24 on rung 6.
- Same photons through the CPU (float32), 2×10⁶ photons: both pass. R_d, T_t and event-count gaps are identical to R1's to every printed digit; absorbed weight total gap 0.0034 (R5) and 0.0034 (R6) standard errors, worst per-batch 4.0e-05 and 4.0e-05.
- Throughput, same session, 4×10⁶ photons, median kernel time of 5 repeats: R1 10.060 s (615.2 million events/s), R3 10.764 s (575.0), **R5 10.059 s (615.3)**, **R6 10.052 s (615.7)**, R1 again 10.486 s (590.2). R5 is -0.01% and R6 -0.08% against the first R1; R3 is +7.0% and R1 again +4.2% (the GPU's own drift). The five repeats span 0.02% (R1), 0.09% (R5) and 0.02% (R6), and 2.5% for R3.
- The notebook's profiling cells (every counter for R3 and R6) were not part of what was pasted: no counters yet for R5 or R6.

**Did the data support the diagnosis?** In part:
- Supported: R3's slowdown was local memory. R6 is R3 with its state in registers and is 7.1% faster than R3, equal to R1.
- Not supported as a statement about the kernel: local memory was not what limited R1. With it removed (R5) the time did not change (-0.01%). The diagnosis explained R3's *extra* wait; I then described it as the limit on the kernel, which was too broad.
- Measured: R1, R5 and R6 take the same time to 0.1%, with repeats that agree to 0.02 to 0.09%, although they differ in local-memory use, in warp instructions issued and in relaunch. Something they share sets the run time. It is not identified.
- Hypotheses, untested: the per-event atomic add into the depth profile (about 6×10⁹ per run, the same in all three; hot bins near the surface may serialize at L2, which an average L2 busy percentage does not show); other per-event work that is identical in every kernel. This reopens a question the R1 entry closed too early: that entry read atomics as not the main limit because L2 was only about 12% busy, which measures bandwidth, not contention on one address.
- Next: read R6's stall reasons and its atomic and L2 counters (the digest cell of the same notebook, or the saved `.ncu.csv` files if the session still has them); then decide between an ablation that removes the depth-profile atomic and the deferred rung R2 (shared-memory histograms), which tests the same thing and would also be the fix.

## R6 counters: the wait moved to the atomic path (recorded 2026-10-18)

*Record:* [`results/gpu-r3-r6-digest.json`](results/gpu-r3-r6-digest.json)

**Prediction:** none (measurement-only). This follows the R5 and R6 entry, which left open what R1, R5 and R6 share.

**Result** (every counter, R3 against R6, 500,000 photons, locked clocks): local loads and stores 0.94 and 0.79 billion to zero; DRAM bytes written 70.7 gigabytes to 6656 bytes; warp instructions 20.19 to 18.62 billion; wait per issued instruction 85.28 to 89.84 cycles, of which LG throttle 15.22 to 70.36 and long scoreboard 64.31 to 10.48; atomic instructions 0.035 and 0.035 billion; busiest L2 slice 63.64% and 63.78% of peak against averages of 12.49% and 2.08%; kernel duration at locked clocks 2.297 and 2.258 seconds.

**What the data show and suggest:**
- Measured: the local memory is gone and the wait per instruction did not fall (85.28 to 89.84 cycles). The long-scoreboard wait (53.8 cycles) was replaced by an almost equal LG-throttle wait (55.1 cycles): LG throttle is now 78% of the wait.
- Measured: the only load/store instructions R6 issues are atomic adds, and their count equals R3's. By construction nearly all are the per-event depth-profile adds (about 1,500 per photon against about 6 others).
- Measured: the busiest L2 slice is at 63.78% of peak against an average of 2.08%, and was at 63.64% in R3 too, so the local-memory traffic did not set it.
- Suggested, not tested: the run time of R1, R5 and R6 is set by the depth-profile atomics serializing at one L2 slice. That explains their equal times and R3's further 7%.
- This corrects the R1 entry's reading that atomics were not the main limit: that rested on average L2 busy percentages, which a hot slice does not show. It also means the earlier decision to defer R2 (shared-memory histograms) was made on the wrong evidence.
- Next: remove the depth and radius profile atomics from R6 as a diagnostic (the profiles come out as zeros, everything else unchanged) and time it. If the time drops, the atomics are the floor, and R2 or a spread-out copy of the profile is the fix; if not, the reading above is wrong.

## Diagnostic: R6 without the depth and radius profile atomics

*Design:* [`docs/diagnostics.md`](docs/diagnostics.md) · *Notebook:* [`notebooks/colab_gpu_ablation.ipynb`](notebooks/colab_gpu_ablation.ipynb)

**Prediction:** none (measurement-only). This is a diagnostic, not a rung: its numbers are not a result.

**Why:** the R6 counters entry suggested that the per-event depth-profile atomic sets the run time of R1, R5 and R6. This removes that atomic (and the radius profile's) from R6 and times it, which tests the suggestion instead of assuming it.

**How the result will be read:** if the time drops substantially, the atomics are the floor and the fix is to spread them out; if it does not move, the suggestion was wrong and the diagnostic's own counters say where the time is going.

**Result:**
- The comparison with the CPU on identical photons (2×10⁶) passes; R_d, T_t and event-count gaps are identical to R6's to every printed digit, absorbed weight 0.0034 standard errors, worst per-batch 4.0e-05. Event rate times kernel time gives 6.19 billion events for R6 and 6.19 for the diagnostic: the same work.
- Throughput, same session, 4×10⁶ photons, median kernel time of 5 repeats: R6 10.053 s (615.7 million events/s), **R6 without the profile atomics 0.823 s (7518.7)**, R1 10.644 s (581.5), R6 again 10.053 s (615.6). The diagnostic is 12.2 times faster than R6; R6 took 10.052 s in the previous session as well.
- The diagnostic is not a result: it was not validated, and the profiles are zeros. The notebook's profiling cells were not part of what was pasted.

**Was the reading right?** Yes, by intervention:
- Measured: removing the depth and radius profile atomics takes the time from 10.053 s to 0.823 s. They are 91.8% of R6's run time.
- This is the floor R1, R5 and R6 shared (R1 +5.9% against R6 in this session, equal to it in the last). R6's time is steady to 0.01% across sessions, as a fixed hardware rate would be.
- Consistent with the counters, not separately measured: the benchmark's depth profile has 40 bins, 320 bytes, about three cache lines, so all of about 6×10⁹ adds serialize on the same lines.
- Not shown: that a real kernel can reach 0.82 s. The diagnostic is the target, not the result.
- Next: a rung that keeps the profiles but stops serializing their adds: per-block histograms in shared memory (the planned privatization rung, now applied to R6), judged against both R6 and this diagnostic.

## R7: the depth and radius profiles in shared memory

*Design:* [`docs/rungs/R7.md`](docs/rungs/R7.md) · *Notebook:* [`notebooks/colab_gpu_r7.ipynb`](notebooks/colab_gpu_r7.ipynb)

**Prediction:** none (measurement-only).

**Why this rung:** the diagnostic showed that the per-event depth-profile atomics are 92% of R6's time: without them R6 took 0.823 s instead of 10.053 s. R7 keeps the profiles and removes the contention: a private copy per warp in shared memory, flushed once per block. The diagnostic is the target, not the result: R7 cannot beat it, and how near it gets is what R7 measures.

**Before any GPU run, from the continuous-integration build and the host tests only:** the host emulation shows the profiles equal R6's to double rounding (bit for bit with one lane) and agree with the CPU engine; the GPU comparison against the CPU now also checks the profiles, with limits fixed before this run from the host emulation and from the CPU's float32 against float64 gaps on V3.

**Result:**
- Validation on the GPU: 24 of 24 checks pass.
- Same photons through the CPU (float32), 2×10⁶ photons, now including the depth and radius profiles: R6 and R7 both pass. Profile gaps for R7: depth total 1.22e-06 and shape 1.75e-06, radius total 1.29e-06 and shape 6.40e-06, against limits of 1e-4 and 1e-3 fixed before the run (R6's are the same to nine digits). R_d, T_t and event-count gaps identical to R6's; absorbed weight 0.0034 standard errors. This is the first time the profiles are compared on the GPU for any rung.
- Throughput, same session, 4×10⁶ photons, median kernel time of 5 repeats: R6 10.052 s (615.7 million events/s), **R7 1.961 s (3156.4)**, the profiles-off diagnostic 0.850 s (7277.3), R7 again 2.016 s (3070.6). R7 is 5.13 times faster than R6 (4.99 on the repeat) for the same 6.19 billion events. Repeats span 1.55% for R7 and 0.04% for R6; R7's second run is 2.8% slower than its first.
- The notebook's profiling cells (counters for the diagnostic and R7) were not part of what was pasted.

**Did the design work?** Largely:
- Measured: R7 is 5.1 times faster than R6 with the profiles kept and correct, and takes 2.31 times the diagnostic's time: it closes 87.9% of the gap between R6 (10.052 s) and the target (0.850 s).
- Measured: the diagnostic took 0.850 s this session and 0.823 s in the last (+3.3%), while R6 took 10.052 s in both. R7 varies more than R6 did.
- Suggested, not tested: R7's time follows the multiprocessors' clock, where R6's followed a fixed-rate atomic unit; that would explain the larger run-to-run and session-to-session variation.
- Not known: why R7 is still 1.11 s above the target. Candidates from reading the code, untested: compare-and-swap loops for the `double` adds in shared memory, retries among lanes of a warp that hit the same bin, the flush, and R7's higher register count (52 against 47 to 49 in the compiler's output).
- Not claimed: anything about inputs with other profile sizes. The cost removed comes from this benchmark's 40-bin depth profile.
- Next: read R7's counters against the diagnostic's (the digest the notebook prints, or the saved profiles in the session) before changing anything.

