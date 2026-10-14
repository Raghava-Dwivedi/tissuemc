#pragma once
// Neumaier compensated summation (an improved Kahan sum).
//
// Tallies add hundreds of millions of small weights to a large running total. Naive summation
// rounds the same way on every add when the increments are similar, so the error grows steadily:
// measured on a one-million-photon run, the deviation from exact energy conservation reached
// 5.7e-12 with a plain double accumulator and 0 with this one (see BUILD_LOG.md).
//
// Do not build with -ffast-math / -Ofast: reassociation optimizes the compensation term away.

#include "tissuemc/hd.hpp"

namespace tissuemc {

template <class Real> struct CompensatedSum {
    Real sum = Real(0);
    Real comp = Real(0); // running low-order correction

    TMC_HD void add(Real x) {
        const Real t = sum + x;
        if (m::fabs(sum) >= m::fabs(x)) {
            comp += (sum - t) + x;
        } else {
            comp += (x - t) + sum;
        }
        sum = t;
    }

    TMC_HD Real value() const { return sum + comp; }
};

} // namespace tissuemc
