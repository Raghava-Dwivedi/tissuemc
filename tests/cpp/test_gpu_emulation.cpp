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
    std::vector<double> rd, tt, absorbed, az, rr;
    std::vector<unsigned long long> events, anomalies;
};

// Sets up the tally arrays, lets `run` fill them, and collects the result. `run` is the host
// stand-in for a kernel launch: it receives the geometry, the batch plan and the arrays.
template <class T, class Runner>
Emulated emulate_with(const RunSpec &spec, uint64_t photons, uint64_t batches, Runner &&run) {
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
    run(geo, plan, arrays);
    out.rd.assign(rd.begin(), rd.end());
    out.tt.assign(tt.begin(), tt.end());
    out.absorbed.assign(absorbed.begin(), absorbed.end());
    out.az.assign(az.begin(), az.end());
    out.rr.assign(rr.begin(), rr.end());
    return out;
}

// Runs the kernel body for every photon on the host, with tallies of precision T.
template <class T>
Emulated emulate(const RunSpec &spec, uint64_t photons, uint64_t batches, uint64_t seed,
                 int rung = 0) {
    return emulate_with<T>(
        spec, photons, batches,
        [&](const Geometry<float> &geo, const BatchPlan &plan, const GpuTallyArrays<T> &arrays) {
            for (uint64_t id = 0; id < photons; ++id) {
                run_photon(rung, geo, seed, id, static_cast<int>(plan.batch_of(id)), arrays);
            }
        });
}

// Rung 3 on the host: `lanes` virtual threads, each advanced one step at a time. The order in
// which lanes are advanced is round-robin, or pseudo-random when `shuffled`; the real GPU's order
// is neither, which is the point: the result must not depend on it. `claimed` receives the final
// value of the shared photon counter.
template <class T, template <class> class Lane = PersistentLane>
Emulated emulate_persistent(const RunSpec &spec, uint64_t photons, uint64_t batches, uint64_t seed,
                            size_t lanes, bool shuffled, unsigned long long *claimed = nullptr) {
    return emulate_with<T>(
        spec, photons, batches,
        [&](const Geometry<float> &geo, const BatchPlan &plan, const GpuTallyArrays<T> &arrays) {
            std::vector<Lane<T>> pool(lanes, Lane<T>(arrays, geo.num_layers));
            std::vector<char> live(lanes, 1);
            size_t remaining = lanes;
            unsigned long long next = 0;
            uint64_t lcg = 0x9e3779b97f4a7c15ull;
            size_t i = 0;
            while (remaining > 0) {
                if (shuffled) {
                    lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
                    i = static_cast<size_t>((lcg >> 33) % lanes);
                } else {
                    i = (i + 1) % lanes;
                }
                if (live[i] && !pool[i].advance(geo, seed, plan, 0, photons, &next)) {
                    live[i] = 0;
                    --remaining;
                }
            }
            if (claimed != nullptr) {
                *claimed = next;
            }
        });
}

// Rung 7 on the host: rung 6's lanes, grouped `lanes_per_group` to a group like the warps of a
// block, each group with its own private copy of the depth and radius profiles. The groups' copies
// are summed into the global profiles at the end, as the kernel's flush does.
template <class T>
Emulated emulate_grouped(const RunSpec &spec, uint64_t photons, uint64_t batches, uint64_t seed,
                         size_t lanes, size_t lanes_per_group, bool shuffled,
                         unsigned long long *claimed = nullptr) {
    return emulate_with<T>(
        spec, photons, batches,
        [&](const Geometry<float> &geo, const BatchPlan &plan, const GpuTallyArrays<T> &arrays) {
            const size_t groups = (lanes + lanes_per_group - 1) / lanes_per_group;
            const size_t nz = static_cast<size_t>(geo.nz), nr = static_cast<size_t>(geo.nr);
            std::vector<std::vector<T>> az(groups, std::vector<T>(nz, T(0)));
            std::vector<std::vector<T>> rr(groups, std::vector<T>(nr, T(0)));
            std::vector<PersistentLaneRegs<T>> pool;
            for (size_t i = 0; i < lanes; ++i) {
                GpuTallyArrays<T> mine = arrays;
                mine.az = az[i / lanes_per_group].data();
                mine.rr = rr[i / lanes_per_group].data();
                pool.emplace_back(mine, geo.num_layers);
            }
            std::vector<char> live(lanes, 1);
            size_t remaining = lanes;
            unsigned long long next = 0;
            uint64_t lcg = 0x9e3779b97f4a7c15ull;
            size_t i = 0;
            while (remaining > 0) {
                if (shuffled) {
                    lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
                    i = static_cast<size_t>((lcg >> 33) % lanes);
                } else {
                    i = (i + 1) % lanes;
                }
                if (live[i] && !pool[i].advance(geo, seed, plan, 0, photons, &next)) {
                    live[i] = 0;
                    --remaining;
                }
            }
            for (size_t g = 0; g < groups; ++g) { // the flush
                for (size_t b = 0; b < nz; ++b) {
                    arrays.az[b] += az[g][b];
                }
                for (size_t b = 0; b < nr; ++b) {
                    arrays.rr[b] += rr[g][b];
                }
            }
            if (claimed != nullptr) {
                *claimed = next;
            }
        });
}

// Rung 5 on the host: one photon at a time, with the register-resident tally and generator.
template <class T>
Emulated emulate_r5(const RunSpec &spec, uint64_t photons, uint64_t batches, uint64_t seed) {
    return emulate_with<T>(
        spec, photons, batches,
        [&](const Geometry<float> &geo, const BatchPlan &plan, const GpuTallyArrays<T> &arrays) {
            for (uint64_t id = 0; id < photons; ++id) {
                run_photon_r5(geo, seed, id, static_cast<int>(plan.batch_of(id)), arrays);
            }
        });
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

TEST_CASE("rung 1 (per-photon accumulation) matches the CPU engine, with float per-photon sums") {
    // Same photons and transport, so identical events and identical R_d and T_t. Absorbed weight
    // is summed in float within each photon before being published, so it differs from the CPU's
    // compensated double by float rounding: measured about 5e-8 relative; the bound leaves 20x.
    const RunSpec spec = v3();
    const uint64_t photons = 40'000, batches = 8;
    const Emulated r1 = emulate<double>(spec, photons, batches, 11, 1);
    const Emulated r0 = emulate<double>(spec, photons, batches, 11, 0);
    const RunResult cpu = cpu_float32(spec, photons, batches, 11);
    const size_t layers = spec.layers.size();
    for (uint64_t b = 0; b < batches; ++b) {
        CAPTURE(b);
        CHECK(r1.events[b] == cpu.batches[b].events);
        CHECK(r1.anomalies[b] == 0);
        CHECK(r1.rd[b] == r0.rd[b]);
        CHECK(r1.tt[b] == r0.tt[b]);
        for (size_t l = 0; l < layers; ++l) {
            const double expected = cpu.batches[b].absorbed[l];
            CHECK(std::fabs(r1.absorbed[b * layers + l] - expected) < 1e-6 * expected);
        }
    }
    // The depth profile is untouched by this rung.
    CHECK(r1.az == r0.az);
}

TEST_CASE("rung 3 with a single lane is rung 1, bit for bit") {
    // One lane claims photons 0, 1, 2, ... in order and scores them exactly as rung 1 does, so
    // every double addition happens in the same order and the results are identical, not close.
    const RunSpec spec = v3();
    const uint64_t photons = 20'000, batches = 8;
    const Emulated r1 = emulate<double>(spec, photons, batches, 11, 1);
    const Emulated r3 = emulate_persistent<double>(spec, photons, batches, 11, 1, false);
    CHECK(r3.events == r1.events);
    CHECK(r3.anomalies == r1.anomalies);
    CHECK(r3.rd == r1.rd);
    CHECK(r3.tt == r1.tt);
    CHECK(r3.absorbed == r1.absorbed);
    CHECK(r3.az == r1.az);
}

TEST_CASE("rung 3 gives the same per-batch results whatever the lanes' interleaving") {
    // Many lanes, advanced round-robin and in a pseudo-random order. Which lane runs which photon
    // changes, and so does the order of the double additions, but not the photons in each batch:
    // events must match exactly, and the sums to double rounding.
    const RunSpec spec = v3();
    const uint64_t photons = 20'000, batches = 8;
    const Emulated r1 = emulate<double>(spec, photons, batches, 11, 1);
    const size_t layers = spec.layers.size();
    struct Case {
        size_t lanes;
        bool shuffled;
    };
    for (const Case c : {Case{7, false}, Case{64, false}, Case{37, true}, Case{500, true}}) {
        CAPTURE(c.lanes);
        CAPTURE(c.shuffled);
        const Emulated r3 =
            emulate_persistent<double>(spec, photons, batches, 11, c.lanes, c.shuffled);
        for (uint64_t b = 0; b < batches; ++b) {
            CAPTURE(b);
            CHECK(r3.events[b] == r1.events[b]);
            CHECK(r3.anomalies[b] == 0);
            CHECK(std::fabs(r3.rd[b] - r1.rd[b]) < 1e-12 * r1.rd[b]);
            CHECK(std::fabs(r3.tt[b] - r1.tt[b]) < 1e-12 * r1.tt[b]);
            for (size_t l = 0; l < layers; ++l) {
                const double expected = r1.absorbed[b * layers + l];
                CHECK(std::fabs(r3.absorbed[b * layers + l] - expected) < 1e-12 * expected);
            }
        }
        for (size_t i = 0; i < r1.az.size(); ++i) {
            CHECK(std::fabs(r3.az[i] - r1.az[i]) <= 1e-12 * (1.0 + r1.az[i]));
        }
    }
}

TEST_CASE("rung 3 agrees with the CPU engine on identical photons") {
    // The same check as for rungs 0 and 1, against the CPU's float32 path, with an interleaving
    // that looks nothing like the CPU's order.
    const RunSpec spec = v3();
    const uint64_t photons = 20'000, batches = 8;
    const Emulated r3 = emulate_persistent<double>(spec, photons, batches, 11, 61, true);
    const RunResult cpu = cpu_float32(spec, photons, batches, 11);
    const size_t layers = spec.layers.size();
    for (uint64_t b = 0; b < batches; ++b) {
        CAPTURE(b);
        CHECK(r3.events[b] == cpu.batches[b].events);
        CHECK(r3.anomalies[b] == 0);
        CHECK(std::fabs(r3.rd[b] - cpu.batches[b].rd) < 1e-9 * cpu.batches[b].rd);
        CHECK(std::fabs(r3.tt[b] - cpu.batches[b].tt) < 1e-9 * cpu.batches[b].tt);
        for (size_t l = 0; l < layers; ++l) {
            const double expected = cpu.batches[b].absorbed[l];
            CHECK(std::fabs(r3.absorbed[b * layers + l] - expected) < 1e-6 * expected);
        }
    }
}

TEST_CASE("rung 3 claims every photon once and each lane stops after one failed claim") {
    // The counter ends at photons + lanes exactly when no photon was claimed twice or skipped and
    // every lane stopped the first time it found nothing left. It includes the case of more lanes
    // than photons, where some lanes never run anything.
    const RunSpec spec = v3();
    struct Case {
        uint64_t photons, batches;
        size_t lanes;
    };
    for (const Case c : {Case{1, 1, 1}, Case{1, 1, 9}, Case{50, 5, 128}, Case{1003, 7, 32}}) {
        CAPTURE(c.photons);
        CAPTURE(c.lanes);
        unsigned long long claimed = 0;
        const Emulated r3 =
            emulate_persistent<double>(spec, c.photons, c.batches, 3, c.lanes, true, &claimed);
        CHECK(claimed == c.photons + c.lanes);
        const Emulated r1 = emulate<double>(spec, c.photons, c.batches, 3, 1);
        CHECK(r3.events == r1.events);
    }
}

namespace {

RunSpec bench_tissue() {
    const std::string dir = TISSUEMC_CONFIG_DIR;
    return parse_mci_file(dir + "/bench/tissue_g09.mci").at(0);
}

} // namespace

TEST_CASE("rung 5's accumulator publishes a partial sum at every layer change") {
    // The one new piece of logic in rung 5: absorbed weight is summed in a register for the layer
    // the packet is in, and published when it moves to another layer and when it ends. Layers 0, 1,
    // 0 again, then end: layer 0 must receive both its partial sums.
    const RunSpec spec = v3();
    const Geometry<float> geo = make_geometry<float>(spec);
    const size_t layers = static_cast<size_t>(geo.num_layers);
    REQUIRE(layers >= 2);
    std::vector<double> rd(1), tt(1), absorbed(layers), az(static_cast<size_t>(geo.nz)),
        rr(static_cast<size_t>(geo.nr));
    std::vector<unsigned long long> events(1), anomalies(1);
    GpuTallyArrays<double> arrays{rd.data(), tt.data(),     absorbed.data(), az.data(),
                                  rr.data(), events.data(), anomalies.data()};
    GpuTallyR5<double> tally(arrays, 0, geo.num_layers);
    Photon<float> p{};
    p.ux = 0.0f;
    p.uy = 0.0f;
    p.uz = 1.0f;
    p.z = 0.01f;
    const float w0a = 0.25f, w0b = 0.125f, w1 = 0.5f, w0c = 0.0625f;
    p.layer = 0;
    tally.score_absorption(p, geo, w0a);
    tally.score_absorption(p, geo, w0b);
    p.layer = 1;
    tally.score_absorption(p, geo, w1);
    p.layer = 0;
    tally.score_absorption(p, geo, w0c);
    CHECK(absorbed[0] == 0.375); // layer 0's first partial sum was published at the move to layer 1
    CHECK(absorbed[1] == 0.5);   // layer 1's, at the move back to layer 0
    tally.flush_counters();
    CHECK(absorbed[0] == 0.375 + 0.0625); // the last partial sum, at the end
    CHECK(absorbed[1] == 0.5);
    CHECK(events[0] == 4);
    CHECK(anomalies[0] == 0);
    // A new photon on the same lane starts clean.
    tally.begin_photon(0);
    tally.flush_counters();
    CHECK(absorbed[0] == 0.375 + 0.0625);
    CHECK(events[0] == 4);
}

TEST_CASE("rung 5 matches rung 1 where it should and the CPU engine where it can") {
    // Same photons, same random numbers, same transport: events are identical, and so are R_d, T_t
    // and the depth profile, which are scored exactly as in rung 1. Absorbed weight is published in
    // partial sums instead of one sum per layer, so it differs from the CPU's compensated double by
    // float rounding of at most a photon's weight: the bound is the one used for rung 1.
    const RunSpec spec = v3();
    const uint64_t photons = 40'000, batches = 8;
    const Emulated r1 = emulate<double>(spec, photons, batches, 11, 1);
    const Emulated r5 = emulate_r5<double>(spec, photons, batches, 11);
    const RunResult cpu = cpu_float32(spec, photons, batches, 11);
    const size_t layers = spec.layers.size();
    CHECK(r5.events == r1.events);
    CHECK(r5.rd == r1.rd);
    CHECK(r5.tt == r1.tt);
    CHECK(r5.az == r1.az);
    for (uint64_t b = 0; b < batches; ++b) {
        CAPTURE(b);
        CHECK(r5.events[b] == cpu.batches[b].events);
        CHECK(r5.anomalies[b] == 0);
        for (size_t l = 0; l < layers; ++l) {
            const double expected = cpu.batches[b].absorbed[l];
            CHECK(std::fabs(r5.absorbed[b * layers + l] - expected) < 1e-6 * expected);
        }
    }
}

TEST_CASE("rung 5 on the single-layer benchmark tissue") {
    // No layer changes except the exit: the accumulator publishes once per photon.
    const RunSpec spec = bench_tissue();
    const uint64_t photons = 2'000, batches = 4;
    const Emulated r5 = emulate_r5<double>(spec, photons, batches, 5);
    const RunResult cpu = cpu_float32(spec, photons, batches, 5);
    for (uint64_t b = 0; b < batches; ++b) {
        CAPTURE(b);
        CHECK(r5.events[b] == cpu.batches[b].events);
        CHECK(r5.anomalies[b] == 0);
        CHECK(std::fabs(r5.rd[b] - cpu.batches[b].rd) < 1e-9 * cpu.batches[b].rd);
        const double expected = cpu.batches[b].absorbed[0];
        CHECK(std::fabs(r5.absorbed[b] - expected) < 1e-6 * expected);
    }
}

TEST_CASE("rung 6 with a single lane is rung 5, bit for bit") {
    const RunSpec spec = v3();
    const uint64_t photons = 20'000, batches = 8;
    const Emulated r5 = emulate_r5<double>(spec, photons, batches, 11);
    const Emulated r6 =
        emulate_persistent<double, PersistentLaneRegs>(spec, photons, batches, 11, 1, false);
    CHECK(r6.events == r5.events);
    CHECK(r6.anomalies == r5.anomalies);
    CHECK(r6.rd == r5.rd);
    CHECK(r6.tt == r5.tt);
    CHECK(r6.absorbed == r5.absorbed);
    CHECK(r6.az == r5.az);
}

TEST_CASE("rung 6 gives the same per-batch results whatever the lanes' interleaving") {
    const RunSpec spec = v3();
    const uint64_t photons = 20'000, batches = 8;
    const Emulated r5 = emulate_r5<double>(spec, photons, batches, 11);
    const size_t layers = spec.layers.size();
    struct Case {
        size_t lanes;
        bool shuffled;
    };
    for (const Case c : {Case{7, false}, Case{64, false}, Case{37, true}, Case{500, true}}) {
        CAPTURE(c.lanes);
        CAPTURE(c.shuffled);
        unsigned long long claimed = 0;
        const Emulated r6 = emulate_persistent<double, PersistentLaneRegs>(
            spec, photons, batches, 11, c.lanes, c.shuffled, &claimed);
        CHECK(claimed == photons + c.lanes); // every photon claimed once, every lane stops once
        for (uint64_t b = 0; b < batches; ++b) {
            CAPTURE(b);
            CHECK(r6.events[b] == r5.events[b]);
            CHECK(r6.anomalies[b] == 0);
            CHECK(std::fabs(r6.rd[b] - r5.rd[b]) < 1e-12 * r5.rd[b]);
            CHECK(std::fabs(r6.tt[b] - r5.tt[b]) < 1e-12 * r5.tt[b]);
            for (size_t l = 0; l < layers; ++l) {
                const double expected = r5.absorbed[b * layers + l];
                CHECK(std::fabs(r6.absorbed[b * layers + l] - expected) < 1e-12 * expected);
            }
        }
    }
}

TEST_CASE("the profile switch skips only the depth and radius tallies") {
    // The diagnostic behind --no-profiles: with Profiles = false a tally must still score R_d, T_t,
    // absorbed weight and the counters, and leave the depth and radius arrays untouched.
    const RunSpec spec = v3();
    const Geometry<float> geo = make_geometry<float>(spec);
    const size_t layers = static_cast<size_t>(geo.num_layers);
    auto run = [&](auto tally_tag, bool expect_profiles) {
        using Tally = decltype(tally_tag);
        std::vector<double> rd(1), tt(1), absorbed(layers), az(static_cast<size_t>(geo.nz)),
            rr(static_cast<size_t>(geo.nr));
        std::vector<unsigned long long> events(1), anomalies(1);
        GpuTallyArrays<double> arrays{rd.data(), tt.data(),     absorbed.data(), az.data(),
                                      rr.data(), events.data(), anomalies.data()};
        Tally tally(arrays, 0, geo.num_layers);
        Photon<float> p{};
        p.uz = 1.0f;
        p.z = 0.05f;
        p.layer = 0;
        tally.score_absorption(p, geo, 0.25f);
        p.x = 0.03f;
        tally.score_reflectance(p, geo, 0.125f);
        tally.score_transmittance(p, geo, 0.0625f);
        tally.flush_counters();
        CHECK(absorbed[0] == 0.25);
        CHECK(rd[0] == 0.125);
        CHECK(tt[0] == 0.0625);
        CHECK(events[0] == 1);
        double az_sum = 0, rr_sum = 0;
        for (double v : az) {
            az_sum += v;
        }
        for (double v : rr) {
            rr_sum += v;
        }
        CHECK(az_sum == (expect_profiles ? 0.25 : 0.0));
        CHECK(rr_sum == (expect_profiles ? 0.125 : 0.0));
    };
    run(GpuTallyR5<double, true>(GpuTallyArrays<double>{}, 0, 1), true);
    run(GpuTallyR5<double, false>(GpuTallyArrays<double>{}, 0, 1), false);
}

TEST_CASE("rung 6 without profiles scores everything else exactly as rung 6 does") {
    // One lane, so the order of every addition is the same: the totals must be identical, not
    // close, and only the two profile arrays differ (zeros against real profiles).
    const RunSpec spec = v3();
    const uint64_t photons = 20'000, batches = 8;
    const Emulated with =
        emulate_persistent<double, PersistentLaneRegs>(spec, photons, batches, 11, 1, false);
    const Emulated without = emulate_persistent<double, PersistentLaneRegsNoProfiles>(
        spec, photons, batches, 11, 1, false);
    CHECK(without.events == with.events);
    CHECK(without.anomalies == with.anomalies);
    CHECK(without.rd == with.rd);
    CHECK(without.tt == with.tt);
    CHECK(without.absorbed == with.absorbed);
    double az_with = 0, az_without = 0, rr_with = 0, rr_without = 0;
    for (double v : with.az)
        az_with += v;
    for (double v : without.az)
        az_without += v;
    for (double v : with.rr)
        rr_with += v;
    for (double v : without.rr)
        rr_without += v;
    CHECK(az_with > 0);
    CHECK(rr_with > 0);
    CHECK(az_without == 0);
    CHECK(rr_without == 0);
}

TEST_CASE("rung 7 with one lane is rung 6, bit for bit") {
    // One lane in one group: the group's copy starts at zero and receives every add in rung 6's
    // order, so adding it to the zeroed global arrays reproduces rung 6 exactly.
    const RunSpec spec = v3();
    const uint64_t photons = 20'000, batches = 8;
    const Emulated r6 =
        emulate_persistent<double, PersistentLaneRegs>(spec, photons, batches, 11, 1, false);
    const Emulated r7 = emulate_grouped<double>(spec, photons, batches, 11, 1, 32, false);
    CHECK(r7.events == r6.events);
    CHECK(r7.rd == r6.rd);
    CHECK(r7.tt == r6.tt);
    CHECK(r7.absorbed == r6.absorbed);
    CHECK(r7.az == r6.az);
    CHECK(r7.rr == r6.rr);
}

TEST_CASE("rung 7's per-warp profile copies sum to rung 6's profiles") {
    // The same lanes advanced in the same order score the same adds, so R_d, T_t, absorbed weight
    // and counts are identical to rung 6's. Only the profiles are accumulated differently (in
    // groups, then summed), so they agree to double rounding, per bin.
    const RunSpec spec = v3();
    const uint64_t photons = 20'000, batches = 8;
    struct Case {
        size_t lanes, per_group;
        bool shuffled;
    };
    for (const Case c :
         {Case{64, 32, false}, Case{37, 8, true}, Case{500, 32, true}, Case{3, 32, false}}) {
        CAPTURE(c.lanes);
        CAPTURE(c.per_group);
        const Emulated r6 = emulate_persistent<double, PersistentLaneRegs>(spec, photons, batches,
                                                                           11, c.lanes, c.shuffled);
        unsigned long long claimed = 0;
        const Emulated r7 = emulate_grouped<double>(spec, photons, batches, 11, c.lanes,
                                                    c.per_group, c.shuffled, &claimed);
        CHECK(claimed == photons + c.lanes);
        CHECK(r7.events == r6.events);
        CHECK(r7.rd == r6.rd);
        CHECK(r7.tt == r6.tt);
        CHECK(r7.absorbed == r6.absorbed);
        double az_sum = 0, rr_sum = 0;
        for (size_t i = 0; i < r6.az.size(); ++i) {
            CHECK(std::fabs(r7.az[i] - r6.az[i]) <= 1e-12 * (1.0 + r6.az[i]));
            az_sum += r6.az[i];
        }
        for (size_t i = 0; i < r6.rr.size(); ++i) {
            CHECK(std::fabs(r7.rr[i] - r6.rr[i]) <= 1e-12 * (1.0 + r6.rr[i]));
            rr_sum += r6.rr[i];
        }
        CHECK(az_sum > 0); // the comparison is not trivially between two empty profiles
        CHECK(rr_sum > 0);
    }
}

TEST_CASE("rung 7's profiles agree with the CPU engine's") {
    // The depth profile is binned with a float division here and a double division in the CPU
    // engine, so a photon on a bin edge can land in a neighbouring bin: bins agree to a part in a
    // thousand and the totals to rounding, as for rung 0.
    const RunSpec spec = v3();
    const uint64_t photons = 20'000, batches = 8;
    const Emulated r7 = emulate_grouped<double>(spec, photons, batches, 11, 61, 32, true);
    const RunResult cpu = cpu_float32(spec, photons, batches, 11);
    double sum7 = 0, sum_cpu = 0, rsum7 = 0, rsum_cpu = 0;
    for (size_t i = 0; i < r7.az.size(); ++i) {
        CHECK(std::fabs(r7.az[i] - cpu.a_z[i]) <= 1e-3 * (1.0 + cpu.a_z[i]));
        sum7 += r7.az[i];
        sum_cpu += cpu.a_z[i];
    }
    for (size_t i = 0; i < r7.rr.size(); ++i) {
        rsum7 += r7.rr[i];
        rsum_cpu += cpu.rd_r[i];
    }
    CHECK(std::fabs(sum7 - sum_cpu) < 1e-6 * sum_cpu);
    CHECK(std::fabs(rsum7 - rsum_cpu) < 1e-6 * rsum_cpu);
}
