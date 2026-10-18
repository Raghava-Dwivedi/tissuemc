# Diagnostic switches

A diagnostic changes what a kernel computes in order to measure what a part of it costs. Its output is **not a result** and is labelled so everywhere it appears.

## `--no-profiles` (GPU rung 6 only)

`tissuemc run ... --engine cuda --rung 6 --no-profiles` skips the depth and radius profile tallies, `A(z)` and `R_d(r)`. They come out as zeros. Everything else (R_d, T_t, absorbed weight per layer, the event count, the random numbers and the photon paths) is exactly what rung 6 produces: the host emulation test compares them bit for bit with one lane, and `tissuemc-cross-engine --b-no-profiles` checks them against the CPU engine on identical photons and refuses a run that still reports a profile.

**Why it exists.** Rungs R1, R5 and R6 all take the same time although they differ in local memory, instruction count and relaunch. R6's counters point at one thing they share: every absorption event adds to the depth profile with a global atomic, about 1,500 per photon against about 6 other atomic adds, and one L2 slice runs at 64% of peak against a 2% average. Removing those adds tells what they cost.

**How the result is read.** If the time drops substantially, the depth-profile atomics are the floor and the fix is to spread them out (shared-memory histograms per block, or copies of the profile). If the time does not move, that reading is wrong and the counters of the diagnostic say where the time is going.

**How it is labelled.** The result JSON carries `"profiles": false`; the benchmark file is named `gpu-r6-noprofiles.json`, records `"validated_before_measuring": false`, and prints "DIAGNOSTIC" when it runs; the profiler and cross-engine tools accept the same flag. The CPU engine refuses it, and so does every GPU rung but 6.

## Outcome (the first diagnostic)

R6 took 10.053 s with the profiles and 0.823 s without (12.2× faster), for the same 6.19 billion events and with R_d, T_t, absorbed weight and counts agreeing with the CPU. The depth and radius profile atomics are 92% of R6's run time, which is the floor that R1, R5 and R6 shared. The diagnostic's time is the target for a real fix, not a result. See [`LOG.md`](../LOG.md).

