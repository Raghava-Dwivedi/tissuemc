#include <doctest.h>

#include <sstream>
#include <string>

#include "tissuemc/mci.hpp"

using namespace tissuemc;

namespace {

// The three-layer input from the MCML manual (Table 6.6), with the original comments.
const char *kThreeLayer = R"(1.0 # file version
1 # number of runs
### Specify data
comp.mco A # output filename, ASCII/Binary
1000000 # No. of photons
.01 .01 # dz, dr
40 50 1 # No. of dz, dr & da.
3 # No. of layers
# n mua mus g d # One line for each layer
1.0 # n for medium above.
1.37 1 100 0.90 0.1 # layer 1
1.37 1 10 0 0.1 # layer 2
1.37 2 10 0.70 0.2 # layer 3
1.0 # n for medium below.
)";

std::vector<RunSpec> parse(const std::string &text) {
    std::istringstream in(text);
    return parse_mci(in);
}

// Error text of a failed parse, or "" if it unexpectedly succeeded.
std::string parse_error(const std::string &text) {
    try {
        parse(text);
    } catch (const ConfigError &e) {
        return e.what();
    }
    return "";
}

// Replace the first occurrence of `from` with `to` in the reference input.
std::string mutate(const std::string &from, const std::string &to) {
    std::string t = kThreeLayer;
    const size_t at = t.find(from);
    REQUIRE(at != std::string::npos);
    t.replace(at, from.size(), to);
    return t;
}

} // namespace

TEST_CASE("parses the manual's three-layer example") {
    const auto runs = parse(kThreeLayer);
    REQUIRE(runs.size() == 1);
    const RunSpec &r = runs[0];
    CHECK(r.output_name == "comp.mco");
    CHECK(r.photons == 1'000'000);
    CHECK(r.dz == 0.01);
    CHECK(r.dr == 0.01);
    CHECK(r.nz == 40);
    CHECK(r.nr == 50);
    CHECK(r.na == 1);
    CHECK(r.n_above == 1.0);
    CHECK(r.n_below == 1.0);
    REQUIRE(r.layers.size() == 3);
    CHECK(r.layers[0].n == 1.37);
    CHECK(r.layers[0].mua == 1.0);
    CHECK(r.layers[0].mus == 100.0);
    CHECK(r.layers[0].g == 0.9);
    CHECK(r.layers[0].d == 0.1);
    CHECK(r.layers[2].mua == 2.0);
    CHECK(r.layers[2].g == 0.7);
    CHECK(r.layers[2].d == 0.2);
}

TEST_CASE("parses several runs and scientific notation") {
    std::string text = "1.0\n2\n";
    text += "a.mco A\n100\n0.1 0.1\n1 1 1\n1\n1.0\n1.0 1 9 0 1e8\n1.0\n";
    text += "b.mco A\n200\n0.1 0.1\n1 1 1\n1\n1.5\n1.4 1 9 0.5 2E-1\n1.5\n";
    const auto runs = parse(text);
    REQUIRE(runs.size() == 2);
    CHECK(runs[0].layers[0].d == 1e8);
    CHECK(runs[1].photons == 200);
    CHECK(runs[1].n_above == 1.5);
    CHECK(runs[1].layers[0].d == doctest::Approx(0.2));
}

TEST_CASE("errors name the offending line") {
    CHECK(parse_error(mutate("1.0 # file version", "2.0 # file version")).find("line 1") !=
          std::string::npos);
    CHECK(parse_error(mutate("comp.mco A", "comp.mco B")).find("line 4") != std::string::npos);
    CHECK(parse_error(mutate("1000000 #", "1000000.5 #")).find("line 5") != std::string::npos);
    // Layer 1 is on line 11 and layer 2 on line 12 of the reference input.
    CHECK(parse_error(mutate("1.37 1 10 0 0.1", "1.37 1 10 0")).find("line 12") !=
          std::string::npos);
    CHECK(parse_error(mutate("1.37 1 100 0.90 0.1", "1.37 1 100 abc 0.1")).find("line 11") !=
          std::string::npos);
}

TEST_CASE("rejects physically invalid and unsupported input") {
    CHECK(!parse_error(mutate("1.37 1 100 0.90 0.1", "1.37 1 100 1.0 0.1")).empty());  // g = 1
    CHECK(!parse_error(mutate("1.37 1 100 0.90 0.1", "1.37 -1 100 0.9 0.1")).empty()); // mua < 0
    CHECK(!parse_error(mutate("1.37 1 100 0.90 0.1", "1.37 1 100 0.9 0")).empty());    // d = 0
    CHECK(!parse_error(mutate("1.37 1 100 0.90 0.1", "0.9 1 100 0.9 0.1")).empty());   // n < 1
    CHECK(!parse_error(mutate("1.37 1 100 0.90 0.1", "1.37 0 0 0 0.1")).empty());      // clear
    CHECK(!parse_error(mutate("1000000 #", "0 #")).empty());                           // 0 photons
    CHECK(!parse_error(mutate("3 # No. of layers", "17 # No. of layers")).empty());    // too many
}

TEST_CASE("rejects truncated and over-long files") {
    CHECK(!parse_error("").empty());
    CHECK(!parse_error("1.0\n1\ncomp.mco A\n").empty());
    CHECK(!parse_error(std::string(kThreeLayer) + "extra line\n").empty());
    // The header promises two runs but only one is present.
    CHECK(!parse_error(mutate("1 # number of runs", "2 # number of runs")).empty());
}

TEST_CASE("missing file is reported with its name") {
    try {
        parse_mci_file("/nonexistent/definitely_missing.mci");
        FAIL("expected ConfigError");
    } catch (const ConfigError &e) {
        CHECK(std::string(e.what()).find("definitely_missing.mci") != std::string::npos);
    }
}

TEST_CASE("bundled validation inputs all parse") {
    const std::string dir = TISSUEMC_CONFIG_DIR;
    for (const char *name : {"v1_slab.mci", "v2_semi_infinite.mci", "v3_three_layer.mci"}) {
        CAPTURE(name);
        CHECK(parse_mci_file(dir + "/validation/" + name).size() == 1);
    }
}
