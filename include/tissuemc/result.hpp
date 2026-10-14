#pragma once
// Results of one simulated run, independent of which engine produced them.

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "tissuemc/config.hpp"

namespace tissuemc {

// One batch of consecutive photon ids. Batches are the unit of statistics: the spread of the
// per-batch means gives the standard error, because packet weights are not Bernoulli trials.
struct BatchResult {
    uint64_t first_photon = 0;
    uint64_t photons = 0;
    double rd = 0;                // summed weight leaving through the top surface (diffuse)
    double tt = 0;                // summed weight leaving through the bottom, unscattered included
    std::vector<double> absorbed; // summed weight absorbed in each layer
    uint64_t events = 0;          // absorption/scattering events (one per hop-drop-spin)
    uint64_t anomalies = 0;       // non-finite weights or non-unit directions seen while scoring
};

struct Estimate {
    double mean = 0;
    double se = 0; // NaN with fewer than two batches
};

struct RunResult {
    RunSpec spec;
    std::string engine;    // "cpu"
    std::string precision; // "float64"
    uint64_t seed = 0;
    int threads = 1;
    double specular = 0; // specular reflectance as a fraction of launched weight
    std::vector<BatchResult> batches;
    std::vector<double> a_z;  // absorbed weight per depth bin (summed over all photons)
    std::vector<double> rd_r; // diffuse reflectance weight per radial bin
    double wall_seconds = 0;
    double kernel_seconds = 0;               // GPU kernel time alone (0 for the CPU engine)
    std::string device;                      // GPU model, empty for the CPU engine
    std::string tally_precision = "float64"; // precision of the running totals

    uint64_t photons() const;
    uint64_t events() const;
    uint64_t anomalies() const;

    // Mean fraction of launched photon weight per photon, and its standard error across batches.
    Estimate rd() const;
    Estimate tt() const;
    Estimate absorbed() const; // all layers
    // rd + tt + absorbed + specular; equals 1 in expectation (roulette is unbiased), so the
    // returned estimate of (total - 1) should be within a few standard errors of zero.
    Estimate conservation_residual() const;
};

void write_json(const RunResult &run, std::ostream &out);

} // namespace tissuemc
