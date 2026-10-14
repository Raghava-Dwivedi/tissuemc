#include <doctest.h>

#include <cmath>
#include <string>

#include "tissuemc/cpu_engine.hpp"
#include "tissuemc/mci.hpp"

using namespace tissuemc;

namespace {

RunSpec load(const char *name) {
    const std::string dir = TISSUEMC_CONFIG_DIR;
    return parse_mci_file(dir + "/validation/" + name).at(0);
}

RunSpec absorbing_slab() {
    RunSpec run;
    run.photons = 1;
    run.layers = {{1.5, 5.0, 0.0, 0.0, 0.1}};
    return run;
}

} // namespace

TEST_CASE("batches partition the photons with sizes differing by at most one") {
    EngineOptions opt;
    opt.batches = 7;
    opt.photons = 1000;
    const RunResult r = run_cpu(absorbing_slab(), opt);
    REQUIRE(r.batches.size() == 7);
    uint64_t total = 0, expected_first = 0;
    for (const auto &b : r.batches) {
        CHECK(b.photons >= 142);
        CHECK(b.photons <= 143);
        CHECK(b.first_photon == expected_first);
        expected_first += b.photons;
        total += b.photons;
    }
    CHECK(total == 1000);
    CHECK(r.photons() == 1000);
}

TEST_CASE("results are reproducible and independent of how photons are batched") {
    const RunSpec run = load("v3_three_layer.mci");
    EngineOptions a, b, c;
    a.photons = b.photons = c.photons = 20'000;
    a.batches = 1;
    b.batches = 4;
    c.batches = 7;
    const RunResult ra = run_cpu(run, a), rb = run_cpu(run, b), rc = run_cpu(run, c);
    // Photon i uses the stream (seed, i), so every partition simulates the same photons. Only the
    // order of floating-point additions differs, hence a tight relative tolerance, not equality.
    const double rd = ra.rd().mean * 20'000, tt = ra.tt().mean * 20'000;
    CHECK(std::fabs(rb.rd().mean * 20'000 - rd) < 1e-9 * rd);
    CHECK(std::fabs(rc.rd().mean * 20'000 - rd) < 1e-9 * rd);
    CHECK(std::fabs(rb.tt().mean * 20'000 - tt) < 1e-9 * tt);
    CHECK(std::fabs(rc.tt().mean * 20'000 - tt) < 1e-9 * tt);
    CHECK(ra.events() == rb.events());
    CHECK(ra.events() == rc.events());

    const RunResult again = run_cpu(run, b);
    CHECK(again.rd().mean == rb.rd().mean); // same options -> bitwise identical
    CHECK(again.events() == rb.events());

    EngineOptions other = b;
    other.seed = 2;
    CHECK(run_cpu(run, other).rd().mean != rb.rd().mean); // a different seed is a different run
}

TEST_CASE("engine invariants hold on the three-layer case") {
    EngineOptions opt;
    opt.photons = 200'000;
    opt.batches = 20;
    const RunResult r = run_cpu(load("v3_three_layer.mci"), opt);
    CHECK(r.anomalies() == 0);
    CHECK(r.events() > r.photons()); // every photon scatters at least once on average
    // Energy conservation holds in expectation (roulette is unbiased): within 5 standard errors.
    const Estimate c = r.conservation_residual();
    CHECK(std::fabs(c.mean) < 5 * c.se);
    // The depth profile accounts for all absorbed weight.
    double az = 0, in_layers = 0;
    for (double v : r.a_z)
        az += v;
    for (const auto &b : r.batches)
        for (double v : b.absorbed)
            in_layers += v;
    CHECK(std::fabs(az - in_layers) < 1e-9 * in_layers);
    // The radial profile accounts for all diffuse reflectance.
    double rr = 0, rd = 0;
    for (double v : r.rd_r)
        rr += v;
    for (const auto &b : r.batches)
        rd += b.rd;
    CHECK(std::fabs(rr - rd) < 1e-9 * rd);
}

TEST_CASE("coarse physics check on the thin slab (rigorous validation is in the Python suite)") {
    EngineOptions opt;
    opt.photons = 400'000;
    opt.batches = 20;
    const RunResult r = run_cpu(load("v1_slab.mci"), opt);
    CHECK(r.anomalies() == 0);
    CHECK(std::fabs(r.rd().mean - 0.09739) < 5 * r.rd().se);
    CHECK(std::fabs(r.tt().mean - 0.66096) < 5 * r.tt().se);
}

TEST_CASE("bad engine options are rejected with clear errors") {
    const RunSpec run = absorbing_slab();
    EngineOptions o;
    o.photons = 10;
    o.batches = 0;
    CHECK_THROWS_AS(run_cpu(run, o), ConfigError);
    o.batches = 11;
    CHECK_THROWS_AS(run_cpu(run, o), ConfigError);
    o.batches = 2;
    o.threads = -1;
    CHECK_THROWS_AS(run_cpu(run, o), ConfigError);
    o.threads = 5000;
    CHECK_THROWS_AS(run_cpu(run, o), ConfigError);
    RunSpec invalid = run;
    invalid.layers[0].g = 1.0;
    o.threads = 1;
    CHECK_THROWS_AS(run_cpu(invalid, o), ConfigError);
}

TEST_CASE("results are bitwise identical for any number of threads") {
    const RunSpec run = load("v3_three_layer.mci");
    EngineOptions base;
    base.photons = 60'000;
    base.batches = 24;
    base.threads = 1;
    const RunResult ref = run_cpu(run, base);
    for (int threads : {2, 3, 8, 24, 0}) {
        EngineOptions o = base;
        o.threads = threads;
        const RunResult r = run_cpu(run, o);
        CAPTURE(threads);
        REQUIRE(r.batches.size() == ref.batches.size());
        for (size_t b = 0; b < ref.batches.size(); ++b) {
            CHECK(r.batches[b].rd == ref.batches[b].rd);
            CHECK(r.batches[b].tt == ref.batches[b].tt);
            CHECK(r.batches[b].absorbed == ref.batches[b].absorbed);
            CHECK(r.batches[b].events == ref.batches[b].events);
        }
        CHECK(r.a_z == ref.a_z);
        CHECK(r.rd_r == ref.rd_r);
    }
}

TEST_CASE("the thread count in the result is the number actually used") {
    EngineOptions o;
    o.photons = 1000;
    o.batches = 3;
    o.threads = 8; // never more workers than batches
    CHECK(run_cpu(absorbing_slab(), o).threads == 3);
    o.threads = 0;
    const int used = run_cpu(absorbing_slab(), o).threads;
    CHECK(used >= 1);
    CHECK(used <= 3);
}

TEST_CASE("float32 transport follows the same photons as float64 far more closely than noise") {
    // Both precisions draw the same random numbers for photon i, so a photon almost always takes
    // the same path in either; only a tiny fraction flip an outcome through rounding. The totals
    // must therefore agree much better than their statistical error. Measured at 1e6 photons on
    // this input: total R_d differs by 2.8e-6 against a standard error of 3.0e-4. The bounds
    // below leave about a tenfold margin so that other compilers' rounding does not trip them.
    const RunSpec run = load("v3_three_layer.mci");
    EngineOptions o64, o32;
    o64.photons = o32.photons = 200'000;
    o64.batches = o32.batches = 20;
    o32.precision = Precision::Float32;
    const RunResult a = run_cpu(run, o64), b = run_cpu(run, o32);
    CHECK(a.precision == "float64");
    CHECK(b.precision == "float32");
    CHECK(b.anomalies() == 0);
    CHECK(std::fabs(a.rd().mean - b.rd().mean) < 0.1 * a.rd().se);
    CHECK(std::fabs(a.tt().mean - b.tt().mean) < 0.1 * a.tt().se);
    for (size_t i = 0; i < a.batches.size(); ++i) {
        CHECK(std::fabs(a.batches[i].rd - b.batches[i].rd) < 5e-3 * a.batches[i].rd);
        CHECK(std::fabs(a.batches[i].tt - b.batches[i].tt) < 5e-3 * a.batches[i].tt);
    }
    const double events_a = static_cast<double>(a.events()),
                 events_b = static_cast<double>(b.events());
    CHECK(std::fabs(events_a - events_b) < 1e-3 * events_a);
}
