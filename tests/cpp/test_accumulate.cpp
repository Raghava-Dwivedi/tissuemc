#include <doctest.h>

#include <cmath>
#include <limits>

#include "tissuemc/accumulate.hpp"

using tissuemc::CompensatedSum;

TEST_CASE("compensated sum recovers a term that naive summation loses") {
    // Classic Neumaier case: the 1.0s vanish next to 1e100 in a plain running sum.
    const double xs[] = {1.0, 1e100, 1.0, -1e100};
    double naive = 0;
    CompensatedSum<double> comp;
    for (double x : xs) {
        naive += x;
        comp.add(x);
    }
    CHECK(naive == 0.0);
    CHECK(comp.value() == 2.0);
}

TEST_CASE("compensated sum of ten million equal increments is within 2 ulp of the exact total") {
    const double x = 0.96;
    const long n = 10'000'000;
    CompensatedSum<double> comp;
    for (long i = 0; i < n; ++i) {
        comp.add(x);
    }
    // x * n is a single correctly rounded product, i.e. within half an ulp of the exact sum of n
    // copies of the stored double x.
    const double expected = x * static_cast<double>(n);
    const double ulp = std::nextafter(expected, std::numeric_limits<double>::infinity()) - expected;
    CHECK(std::fabs(comp.value() - expected) <= 2 * ulp);
}
