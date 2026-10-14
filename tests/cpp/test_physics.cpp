#include <doctest.h>

#include <cmath>

#include "tissuemc/accumulate.hpp"
#include "tissuemc/config.hpp"
#include "tissuemc/physics.hpp"

using namespace tissuemc;

namespace {

// Minimal tally: totals only. Engines provide richer tallies; physics tests only need conservation.
// Compensated sums keep the conservation check exact; plain doubles drift by ~1e-11 over 1e6
// photons.
struct TotalsTally {
    CompensatedSum<double> rd, tt, a;
    template <class Real>
    void score_absorption(const Photon<Real> &, const Geometry<Real> &, Real w) {
        a.add(w);
    }
    template <class Real>
    void score_reflectance(const Photon<Real> &, const Geometry<Real> &, Real w) {
        rd.add(w);
    }
    template <class Real>
    void score_transmittance(const Photon<Real> &, const Geometry<Real> &, Real w) {
        tt.add(w);
    }
};

RunSpec single_layer(double n, double mua, double mus, double g, double d, double n_amb = 1.0) {
    RunSpec run;
    run.photons = 1;
    run.n_above = run.n_below = n_amb;
    run.layers = {{n, mua, mus, g, d}};
    return run;
}

// Textbook Fresnel equations for unpolarized light (average of s and p polarization).
double textbook_fresnel(double n1, double n2, double cos_i) {
    const double sin_t = n1 / n2 * std::sqrt(1 - cos_i * cos_i);
    if (sin_t >= 1)
        return 1.0;
    const double cos_t = std::sqrt(1 - sin_t * sin_t);
    const double rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
    const double rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t);
    return 0.5 * (rs * rs + rp * rp);
}

} // namespace

TEST_CASE("specular reflectance matches ((n0 - n1) / (n0 + n1))^2") {
    const auto geo = make_geometry<double>(single_layer(1.5, 1, 1, 0, 1));
    CHECK(specular_reflectance(geo) == doctest::Approx(0.04));
    const auto matched = make_geometry<double>(single_layer(1.0, 1, 1, 0, 1));
    CHECK(specular_reflectance(matched) == 0.0);
}

TEST_CASE("MCML-form Fresnel equals the textbook s/p formulas") {
    for (auto [n1, n2] :
         {std::pair{1.0, 1.5}, std::pair{1.5, 1.0}, std::pair{1.37, 1.0}, std::pair{1.33, 1.37}}) {
        for (double cos_i = 0.05; cos_i < 0.999; cos_i += 0.05) {
            double ca2 = 0;
            const double r = fresnel(n1, n2, cos_i, ca2);
            CAPTURE(n1);
            CAPTURE(n2);
            CAPTURE(cos_i);
            CHECK(r == doctest::Approx(textbook_fresnel(n1, n2, cos_i)).epsilon(1e-10));
            if (r < 1.0) { // Snell's law for the transmitted angle
                const double sin_i = std::sqrt(1 - cos_i * cos_i);
                const double sin_t = std::sqrt(1 - ca2 * ca2);
                CHECK(n1 * sin_i == doctest::Approx(n2 * sin_t).epsilon(1e-10));
            }
        }
    }
}

TEST_CASE("Fresnel edge cases: matched index, normal incidence, total internal reflection") {
    double ca2 = 0;
    CHECK(fresnel(1.37, 1.37, 0.3, ca2) == 0.0);
    CHECK(ca2 == 0.3);
    CHECK(fresnel(1.0, 1.5, 1.0, ca2) == doctest::Approx(0.04));
    // From glass (1.5) to air the critical angle is asin(1/1.5) ~ 41.8 deg, cos ~ 0.745.
    CHECK(fresnel(1.5, 1.0, 0.70, ca2) == 1.0);
    CHECK(ca2 == 0.0);
    CHECK(critical_cos(1.5, 1.0) == doctest::Approx(std::sqrt(1 - 1 / 2.25)));
    CHECK(critical_cos(1.0, 1.5) == 0.0);
}

TEST_CASE("Henyey-Greenstein sampler reproduces its first two moments") {
    // For HG, E[cos] = g and E[cos^2] = (1 + 2 g^2) / 3.
    for (double g : {0.0, 0.5, 0.9, -0.3}) {
        PhotonRng rng(7, static_cast<uint64_t>(g * 1000 + 1000));
        const int n = 1'000'000;
        double s1 = 0, s2 = 0;
        for (int i = 0; i < n; ++i) {
            const double c = sample_hg_cos(g, rng);
            REQUIRE(c >= -1.0);
            REQUIRE(c <= 1.0);
            s1 += c;
            s2 += c * c;
        }
        CAPTURE(g);
        // The variance of cos is at most 1/3, so the standard error is < 5.8e-4; allow ~5 sigma.
        CHECK(std::fabs(s1 / n - g) < 3e-3);
        CHECK(std::fabs(s2 / n - (1 + 2 * g * g) / 3) < 3e-3);
    }
}

TEST_CASE("spin keeps the direction a unit vector and scatters by mean cosine g") {
    const double g = 0.75;
    PhotonRng rng(11, 0);
    Photon<double> p{};
    const int n = 400'000;
    double sum_cos = 0, worst_norm_err = 0;
    p.ux = 0, p.uy = 0, p.uz = 1; // start vertical to exercise the near-vertical branch
    for (int i = 0; i < n; ++i) {
        const double ux = p.ux, uy = p.uy, uz = p.uz;
        spin(p, g, rng);
        sum_cos += ux * p.ux + uy * p.uy + uz * p.uz;
        const double norm = std::sqrt(p.ux * p.ux + p.uy * p.uy + p.uz * p.uz);
        worst_norm_err = std::fmax(worst_norm_err, std::fabs(norm - 1));
        if (i % 1000 == 0) { // periodically renormalize-free restart to also test vertical input
            p.ux = 0, p.uy = 0, p.uz = -1;
        }
    }
    CHECK(worst_norm_err < 1e-9);
    CHECK(std::fabs(sum_cos / n - g) < 5e-3);
}

// Regression guard for the direction-norm drift described in spin(): a long unbroken chain of
// scattering events must not let the direction vector walk away from unit length.
template <class Real> double worst_norm_error_over_long_walk(Real g, long events) {
    PhotonRng rng(5, 1);
    Photon<Real> p{};
    p.ux = Real(0.6), p.uy = Real(0), p.uz = Real(0.8);
    double worst = 0;
    for (long i = 0; i < events; ++i) {
        spin(p, g, rng);
        const double ux = p.ux, uy = p.uy, uz = p.uz;
        worst = std::fmax(worst, std::fabs(std::sqrt(ux * ux + uy * uy + uz * uz) - 1.0));
    }
    return worst;
}

TEST_CASE("direction stays a unit vector over millions of consecutive scatterings") {
    // g = 0.99 keeps the direction close to its previous value, which is the regime where the
    // unnormalized rotation formula was seen to amplify error.
    CHECK(worst_norm_error_over_long_walk<double>(0.0, 4'000'000) < 1e-13);
    CHECK(worst_norm_error_over_long_walk<double>(0.99, 4'000'000) < 1e-13);
    CHECK(worst_norm_error_over_long_walk<float>(0.0f, 4'000'000) < 5e-7);
    CHECK(worst_norm_error_over_long_walk<float>(0.99f, 4'000'000) < 5e-7);
}

TEST_CASE("step lengths are exponential with mean 1 / mu_t") {
    const auto geo = make_geometry<double>(single_layer(1.0, 10, 90, 0.75, 1e6));
    PhotonRng rng(3, 0);
    Photon<double> p{};
    launch(p, geo);
    const int n = 1'000'000;
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        p.sleft = 0;
        step_size(p, geo, rng);
        sum += p.s;
    }
    // Mean 0.01 cm with standard deviation 0.01 cm, so the standard error is 1e-5; allow 5 sigma.
    CHECK(std::fabs(sum / n - 0.01) < 5e-5);
}

// With mu_s = 0 a packet travels straight down and back, and is absorbed completely at its first
// interaction, so each outcome is a binomial event. Summing the geometric series of internal
// reflections (R = 0.04 at each glass-air face) gives exact reflectance and transmittance.
template <class Real> void check_absorbing_slab() {
    const double n = 1.5, mua = 5.0, d = 0.1, tau = mua * d;
    const auto geo = make_geometry<Real>(single_layer(n, mua, 0.0, 0.0, d));
    const double rsp = 0.04, rt = 0.04, rb = 0.04;
    const double loop = 1.0 / (1.0 - rt * rb * std::exp(-2 * tau));
    const double t_exact = (1 - rsp) * std::exp(-tau) * (1 - rb) * loop;
    const double rd_exact = (1 - rsp) * std::exp(-2 * tau) * rb * (1 - rt) * loop;

    TotalsTally tally;
    const int n_photons = 1'000'000;
    for (int i = 0; i < n_photons; ++i) {
        PhotonRng rng(99, static_cast<uint64_t>(i));
        simulate_photon(geo, rng, tally);
    }
    const double rd_total = tally.rd.value(), tt_total = tally.tt.value(),
                 a_total = tally.a.value();
    const double t_sim = tt_total / n_photons, rd_sim = rd_total / n_photons;
    // Each packet deposits all of its (1 - Rsp) weight into exactly one outcome, so the counts are
    // binomial: SE = (1 - Rsp) * sqrt(p (1 - p) / N).
    auto se = [&](double x) {
        const double p = x / (1 - rsp);
        return (1 - rsp) * std::sqrt(p * (1 - p) / n_photons);
    };
    CHECK(std::fabs(t_sim - t_exact) < 5 * se(t_exact));
    CHECK(std::fabs(rd_sim - rd_exact) < 5 * se(rd_exact));
    // Energy conservation: every packet's weight ends up somewhere. In double the residual is
    // summation rounding only (0 observed at 4e6 photons; plain double accumulation gave 5.7e-12 at
    // 1e6). In float the per-packet weight itself carries ~1e-7 rounding, so allow 1e-5.
    const double tol = sizeof(Real) == sizeof(double) ? 1e-13 : 1e-5;
    CHECK(std::fabs(rsp + (rd_total + tt_total + a_total) / n_photons - 1.0) < tol);
}

TEST_CASE("purely absorbing slab matches the closed-form multiple-reflection solution (double)") {
    check_absorbing_slab<double>();
}

TEST_CASE("purely absorbing slab matches the closed-form multiple-reflection solution (float)") {
    // The GPU engine runs this same code in float, so the float instantiation needs its own check.
    check_absorbing_slab<float>();
}
