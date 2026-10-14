#pragma once
// Multi-batch CPU engine (float64). The reference that the GPU engine is checked against.

#include <cstdint>

#include "tissuemc/result.hpp"

namespace tissuemc {

// Precision of the photon transport. Tallies are always accumulated in double. float32 runs the
// same physics code as the GPU engine, so it is the local stand-in for checking the GPU path.
enum class Precision { Float64, Float32 };

struct EngineOptions {
    uint64_t seed = 1;
    int batches = 10;
    int threads = 1;      // 0 = one per hardware thread
    uint64_t photons = 0; // if nonzero, overrides the photon count in the run description
    Precision precision = Precision::Float64;
};

// Photon i always uses the random stream keyed by (seed, i), so the result depends only on
// (run, seed, photons, batches), never on thread count or scheduling.
RunResult run_cpu(const RunSpec &run, const EngineOptions &options);

} // namespace tissuemc
