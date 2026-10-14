#include <doctest.h>

#include <cmath>
#include <string>
#include <vector>

#include "tissuemc/batching.hpp"
#include "tissuemc/config.hpp"
#include "tissuemc/cpu_engine.hpp"
#include "tissuemc/gpu_tally.hpp"
#include "tissuemc/mci.hpp"

using namespace tissuemc;

TEST_CASE("batch plan partitions photons and inverts first_of") {
    struct Case {
        uint64_t photons, batches;
    };
    for (const Case c : {Case{1000, 7}, Case{10, 10}, Case{1, 1}, Case{1001, 1000}, Case{240, 24},
                         Case{5003, 50}}) {
        CAPTURE(c.photons);
        CAPTURE(c.batches);
        const BatchPlan plan = BatchPlan::make(c.photons, c.batches);
        uint64_t total = 0;
        for (uint64_t b = 0; b < c.batches; ++b) {
            CHECK(plan.first_of(b) == total);
            total += plan.count_of(b);
        }
        CHECK(total == c.photons);
        for (uint64_t id = 0; id < c.photons; ++id) {
            const uint64_t b = plan.batch_of(id);
            REQUIRE(b < c.batches);
            REQUIRE(plan.first_of(b) <= id);
            REQUIRE(id < plan.first_of(b) + plan.count_of(b));
        }
    }
}

TEST_CASE("the CPU engine uses the shared batch plan") {
    const std::string dir = TISSUEMC_CONFIG_DIR;
    const RunSpec run = parse_mci_file(dir + "/validation/v3_three_layer.mci").at(0);
    EngineOptions o;
    o.photons = 1003;
    o.batches = 7;
    const RunResult r = run_cpu(run, o);
    const BatchPlan plan = BatchPlan::make(1003, 7);
    for (uint64_t b = 0; b < 7; ++b) {
        CHECK(r.batches[b].first_photon == plan.first_of(b));
        CHECK(r.batches[b].photons == plan.count_of(b));
    }
}

namespace {

struct Emulated {
    std::vector<double> rd, tt, absorbed, az;
    std::vector<unsigned long long> events, anomalies;
};

// Runs the kernel body for every photon on the host, with tallies of precision T.
template <class T>
Emulated emulate(const RunSpec &spec, uint64_t photons, uint64_t batches, uint64_t seed) {
    const Geometry<float> geo = make_geometry<float>(spec);
    const BatchPlan plan = BatchPlan::make(photons, batches);
    const size_t layers = static_cast<size_t>(geo.num_layers);
    std::vector<T> rd(batches), tt(batches), absorbed(batches * layers);
    std::vector<T> az(static_cast<size_t>(geo.nz)), rr(static_cast<size_t>(geo.nr));
    Emulated out;
    out.events.assign(batches, 0);
    out.anomalies.assign(batches, 0);
    GpuTallyArrays<T> arrays{rd.data(), tt.data(),         absorbed.data(),     az.data(),
                             rr.data(), out.events.data(), out.anomalies.data()};
    for (uint64_t id = 0; id < photons; ++id) {
        run_photon_r0(geo, seed, id, static_cast<int>(plan.batch_of(id)), arrays);
    }
    out.rd.assign(rd.begin(), rd.end());
    out.tt.assign(tt.begin(), tt.end());
    out.absorbed.assign(absorbed.begin(), absorbed.end());
    out.az.assign(az.begin(), az.end());
    return out;
}

RunResult cpu_float32(const RunSpec &spec, uint64_t photons, uint64_t batches, uint64_t seed) {
    EngineOptions o;
    o.photons = photons;
    o.batches = static_cast<int>(batches);
    o.seed = seed;
    o.precision = Precision::Float32;
    return run_cpu(spec, o);
}

RunSpec v3() {
    const std::string dir = TISSUEMC_CONFIG_DIR;
    return parse_mci_file(dir + "/validation/v3_three_layer.mci").at(0);
}

} // namespace

TEST_CASE("the GPU kernel body with double tallies matches the CPU engine to rounding") {
    // This executes the same function the CUDA kernel calls for each photon, with the atomic add
    // replaced by an ordinary add. What it leaves untested is the launch and the atomics. Same
    // photons and the same float32 transport give identical paths, so identical event counts, and
    // sums that differ only by the order of double additions.
    const RunSpec spec = v3();
    const uint64_t photons = 40'000, batches = 8;
    const Emulated mine = emulate<double>(spec, photons, batches, 11);
    const RunResult cpu = cpu_float32(spec, photons, batches, 11);
    const size_t layers = spec.layers.size();
    for (uint64_t b = 0; b < batches; ++b) {
        CAPTURE(b);
        CHECK(mine.events[b] == cpu.batches[b].events);
        CHECK(mine.anomalies[b] == 0);
        CHECK(std::fabs(mine.rd[b] - cpu.batches[b].rd) < 1e-9 * cpu.batches[b].rd);
        CHECK(std::fabs(mine.tt[b] - cpu.batches[b].tt) < 1e-9 * cpu.batches[b].tt);
        for (size_t l = 0; l < layers; ++l) {
            const double expected = cpu.batches[b].absorbed[l];
            CHECK(std::fabs(mine.absorbed[b * layers + l] - expected) < 1e-9 * expected);
        }
    }
    // The depth profile is binned with a float division here and a double division in the CPU
    // engine, so a photon sitting exactly on a bin edge can land in the neighbouring bin. The
    // total over all bins is unaffected; individual bins agree to a part in a thousand.
    double sum_mine = 0, sum_cpu = 0;
    for (size_t i = 0; i < mine.az.size(); ++i) {
        sum_mine += mine.az[i];
        sum_cpu += cpu.a_z[i];
        CHECK(std::fabs(mine.az[i] - cpu.a_z[i]) <= 1e-3 * (1.0 + cpu.a_z[i]));
    }
    CHECK(std::fabs(sum_mine - sum_cpu) < 1e-9 * sum_cpu);
}

TEST_CASE("float32 tallies lose absorbed weight once batches are large") {
    // The reason R0 does not use float32 tallies. Each batch's running total grows, and an
    // addition smaller than half the total's last digit vanishes. Absorption is the worst case:
    // many small additions into a large total. Transport is unchanged, so the event counts still
    // match exactly; only the sums are wrong.
    const RunSpec spec = v3();
    const uint64_t photons = 400'000, batches = 2; // 200,000 photons per batch
    const Emulated f32 = emulate<float>(spec, photons, batches, 11);
    const Emulated f64 = emulate<double>(spec, photons, batches, 11);
    double absorbed32 = 0, absorbed64 = 0;
    for (size_t i = 0; i < f64.absorbed.size(); ++i) {
        absorbed32 += f32.absorbed[i];
        absorbed64 += f64.absorbed[i];
    }
    CHECK(f32.events == f64.events);
    CHECK(f32.anomalies == f64.anomalies);
    const double lost = (absorbed64 - absorbed32) / absorbed64;
    INFO("fraction of absorbed weight lost by float32 tallies: " << lost);
    CHECK(lost > 1e-3); // a visible bias, thousands of times the float64 rounding
    CHECK(lost < 0.5);  // but not catastrophic: the total is not simply wrong
    // Reflectance and transmittance are single large additions per photon, so they survive.
    for (size_t b = 0; b < batches; ++b) {
        CHECK(std::fabs(f32.rd[b] - f64.rd[b]) < 1e-3 * f64.rd[b]);
        CHECK(std::fabs(f32.tt[b] - f64.tt[b]) < 1e-3 * f64.tt[b]);
    }
}
