#pragma once
// Multi-batch CPU engine (float64). The reference that the GPU engine is checked against.

#include <cstdint>

#include "tissuemc/result.hpp"

namespace tissuemc {

struct EngineOptions {
    uint64_t seed = 1;
    int batches = 10;
    int threads = 1;      // 0 = one per hardware thread
    uint64_t photons = 0; // if nonzero, overrides the photon count in the run description
};

// Photon i always uses the random stream keyed by (seed, i), so the result depends only on
// (run, seed, photons, batches), never on thread count or scheduling.
RunResult run_cpu(const RunSpec &run, const EngineOptions &options);

} // namespace tissuemc
