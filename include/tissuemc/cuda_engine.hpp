#pragma once
// GPU engine. Declared always, defined only when built with -DTISSUEMC_ENABLE_CUDA=ON.

#include "tissuemc/cpu_engine.hpp"

namespace tissuemc {

// True if a CUDA device is present and usable.
bool cuda_available();

// Rung R0: one thread per photon, float32 transport, global atomic tallies. Photon i uses the same
// random stream as on the CPU, and batch b covers the same photon ids, so per-batch results can be
// compared directly with the CPU engine.
RunResult run_cuda(const RunSpec &run, const EngineOptions &options);

} // namespace tissuemc
