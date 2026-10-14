#include <doctest.h>

#include <cmath>
#include <set>

#include "tissuemc/philox.hpp"

using tissuemc::PhiloxKey;
using tissuemc::PhotonRng;
using tissuemc::U32x4;

namespace {
void check_kat(U32x4 ctr, PhiloxKey key, U32x4 expected) {
    const U32x4 out = tissuemc::philox4x32_10(ctr, key);
    for (int i = 0; i < 4; ++i) {
        CHECK(out.v[i] == expected.v[i]);
    }
}
} // namespace

// Known-answer vectors for philox4x32 with 10 rounds, from the Random123 distribution
// (kat_vectors). Matching all three rules out a wrong constant, round order or key schedule.
TEST_CASE("philox4x32-10 matches Random123 known-answer vectors") {
    check_kat({{0, 0, 0, 0}}, {{0, 0}}, {{0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}});
    check_kat({{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}}, {{0xffffffffu, 0xffffffffu}},
              {{0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}});
    check_kat({{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}}, {{0xa4093822u, 0x299f31d0u}},
              {{0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}});
}

TEST_CASE("unit-interval conversion never returns 0 or 1") {
    for (uint32_t x : {0u, 1u, 511u, 512u, 0x7fffffffu, 0xfffffffeu, 0xffffffffu}) {
        const double d = tissuemc::to_unit_open<double>(x);
        const float f = tissuemc::to_unit_open<float>(x);
        CHECK(d > 0.0);
        CHECK(d < 1.0);
        CHECK(f > 0.0f);
        CHECK(f < 1.0f);
    }
}

TEST_CASE("float and double draws agree to float resolution") {
    PhotonRng a(42, 7), b(42, 7);
    for (int i = 0; i < 1000; ++i) {
        const double d = tissuemc::uniform<double>(a);
        const float f = tissuemc::uniform<float>(b);
        // The float draw keeps 23 of the 32 bits, so the two differ by less than 2^-23.
        CHECK(std::fabs(d - static_cast<double>(f)) < 2.5e-7);
    }
}

TEST_CASE("photon streams are reproducible and distinct") {
    PhotonRng a(1234, 99), b(1234, 99), c(1234, 100), d(1235, 99);
    std::set<uint32_t> first_c, first_d;
    bool all_same_ab = true;
    for (int i = 0; i < 64; ++i) {
        const uint32_t xa = a.next_u32(), xb = b.next_u32();
        all_same_ab = all_same_ab && (xa == xb);
        first_c.insert(c.next_u32());
        first_d.insert(d.next_u32());
    }
    CHECK(all_same_ab);
    PhotonRng a2(1234, 99);
    int shared_with_c = 0, shared_with_d = 0;
    for (int i = 0; i < 64; ++i) {
        const uint32_t x = a2.next_u32();
        shared_with_c += static_cast<int>(first_c.count(x));
        shared_with_d += static_cast<int>(first_d.count(x));
    }
    CHECK(shared_with_c == 0); // different photon id -> independent stream
    CHECK(shared_with_d == 0); // different seed -> independent stream
}

TEST_CASE("uniform deviates have mean 1/2 and variance 1/12") {
    PhotonRng rng(2026, 0);
    const int n = 1'000'000;
    double sum = 0, sum2 = 0;
    for (int i = 0; i < n; ++i) {
        const double u = tissuemc::uniform<double>(rng);
        sum += u;
        sum2 += u * u;
    }
    const double mean = sum / n;
    const double var = sum2 / n - mean * mean;
    // Standard error of the mean is sqrt(1/12 / n) ~ 2.9e-4; allow 5 sigma.
    CHECK(std::fabs(mean - 0.5) < 1.5e-3);
    CHECK(std::fabs(var - 1.0 / 12.0) < 5e-4);
}
