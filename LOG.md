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

*Design:* [`docs/rungs/R1.md`](docs/rungs/R1.md) · *Notebook:* [`notebooks/colab_gpu_r1.ipynb`](notebooks/colab_gpu_r1.ipynb)

**Prediction (written before the run):** not yet recorded.

**Result:** not run yet.

**Did the prediction hold, and why?** Not run yet.
