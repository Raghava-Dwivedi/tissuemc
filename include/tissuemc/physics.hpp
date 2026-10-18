#pragma once
// Photon-packet transport through layered tissue, following MCML.
//
// Reference: L. Wang, S. L. Jacques, "Monte Carlo Modeling of Light Transport in Multi-layered
// Tissues in Standard C" (1992), chapters 3-5. Section numbers in comments refer to that manual.
//
// Every function is TMC_HD and templated on the floating-point type, so the float64 CPU engine and
// the float32 GPU engine execute the same code. The random generator is a template parameter too:
// PhotonRng everywhere except the GPU rungs R5 and R6, which use the register-friendly
// RegPhotonRng. Scoring is delegated to a Tally type supplied by
// each engine (see score_* calls), because CPU and GPU accumulate results differently.

#include "tissuemc/geometry.hpp"
#include "tissuemc/hd.hpp"
#include "tissuemc/philox.hpp"

namespace tissuemc {

template <class Real> struct Constants;

template <> struct Constants<double> {
    static constexpr double kPi = 3.14159265358979323846;
    static constexpr double kWeightThreshold = 1e-4; // MCML WEIGHT (Table 5.1)
    static constexpr double kChance = 0.1;           // MCML CHANCE (Table 5.1)
    static constexpr double kCosZero = 1.0 - 1e-12;  // MCML COSZERO
    static constexpr double kCos90D = 1e-6;          // MCML COS90D
};

// float cannot represent 1 - 1e-12 (it rounds to 1), so the near-vertical cutoff is widened to the
// largest value float distinguishes reliably. Documented as a precision difference between engines.
template <> struct Constants<float> {
    static constexpr float kPi = 3.14159265358979323846f;
    static constexpr float kWeightThreshold = 1e-4f;
    static constexpr float kChance = 0.1f;
    static constexpr float kCosZero = 1.0f - 1e-6f;
    static constexpr float kCos90D = 1e-6f;
};

template <class Real> struct Photon {
    Real x, y, z;    // position [cm]
    Real ux, uy, uz; // direction cosines (unit vector)
    Real w;          // packet weight
    Real s;          // current step length [cm]
    Real sleft;      // unused step carried across a boundary, dimensionless (in mean free paths)
    int layer;       // index of the current layer (0 = top)
    bool dead;
};

// Specular reflectance at the top surface for a normally incident pencil beam (MCML 3.1).
template <class Real> TMC_HD Real specular_reflectance(const Geometry<Real> &geo) {
    const Real n0 = geo.n_above;
    const Real n1 = geo.layers[0].n;
    const Real r = (n0 - n1) / (n0 + n1);
    return r * r;
}

// Launch a packet at the origin travelling straight down (MCML 3.1).
template <class Real> TMC_HD void launch(Photon<Real> &p, const Geometry<Real> &geo) {
    p.x = p.y = p.z = Real(0);
    p.ux = p.uy = Real(0);
    p.uz = Real(1);
    p.w = Real(1) - specular_reflectance(geo);
    p.s = p.sleft = Real(0);
    p.layer = 0;
    p.dead = false;
}

// Fresnel reflectance for unpolarized light going from index n1 into n2, with ca1 = cos(incidence)
// >= 0. Writes the cosine of the transmission angle to ca2. Same algebraic form as MCML's
// RFresnel (MCML 3.5); tests compare it against the textbook s/p-polarization formulas.
template <class Real> TMC_HD Real fresnel(Real n1, Real n2, Real ca1, Real &ca2) {
    using C = Constants<Real>;
    if (n1 == n2) {
        ca2 = ca1;
        return Real(0);
    }
    if (ca1 > C::kCosZero) { // normal incidence
        ca2 = ca1;
        const Real r = (n2 - n1) / (n2 + n1);
        return r * r;
    }
    if (ca1 < C::kCos90D) { // grazing incidence
        ca2 = Real(0);
        return Real(1);
    }
    const Real sa1 = m::sqrt(Real(1) - ca1 * ca1);
    const Real sa2 = n1 * sa1 / n2;
    if (sa2 >= Real(1)) { // total internal reflection
        ca2 = Real(0);
        return Real(1);
    }
    ca2 = m::sqrt(Real(1) - sa2 * sa2);
    const Real cap = ca1 * ca2 - sa1 * sa2; // cos(a1 + a2)
    const Real cam = ca1 * ca2 + sa1 * sa2; // cos(a1 - a2)
    const Real sap = sa1 * ca2 + ca1 * sa2; // sin(a1 + a2)
    const Real sam = sa1 * ca2 - ca1 * sa2; // sin(a1 - a2)
    return Real(0.5) * sam * sam * (cam * cam + cap * cap) / (sap * sap * cam * cam);
}

// Sample cos(theta) from the Henyey-Greenstein phase function by inverting its CDF (MCML 3.4).
template <class Real, class Rng> TMC_HD Real sample_hg_cos(Real g, Rng &rng) {
    const Real xi = uniform<Real>(rng);
    if (g == Real(0)) {
        return Real(2) * xi - Real(1);
    }
    const Real t = (Real(1) - g * g) / (Real(1) - g + Real(2) * g * xi);
    Real c = (Real(1) + g * g - t * t) / (Real(2) * g);
    if (c < Real(-1))
        c = Real(-1);
    if (c > Real(1))
        c = Real(1);
    return c;
}

// Rotate the direction by polar angle theta (cos given) and a uniform azimuth (MCML 3.4).
template <class Real, class Rng> TMC_HD void spin(Photon<Real> &p, Real g, Rng &rng) {
    using C = Constants<Real>;
    const Real cost = sample_hg_cos(g, rng);
    Real sint = Real(1) - cost * cost;
    sint = sint > Real(0) ? m::sqrt(sint) : Real(0);
    const Real psi = Real(2) * C::kPi * uniform<Real>(rng);
    const Real cosp = m::cos(psi);
    const Real sinp = m::sin(psi);
    if (m::fabs(p.uz) > C::kCosZero) { // nearly vertical: use the simple form
        p.ux = sint * cosp;
        p.uy = sint * sinp;
        p.uz = p.uz >= Real(0) ? cost : -cost;
    } else {
        const Real temp = m::sqrt(Real(1) - p.uz * p.uz);
        const Real ux = p.ux, uy = p.uy, uz = p.uz;
        p.ux = sint * (ux * uz * cosp - uy * sinp) / temp + ux * cost;
        p.uy = sint * (uy * uz * cosp + ux * sinp) / temp + uy * cost;
        p.uz = -sint * cosp * temp + uz * cost;
    }
    // MCML's rotation formula uses sqrt(1 - uz^2) in place of sqrt(ux^2 + uy^2). That is exact for
    // a unit vector but does not preserve the length of a slightly non-unit one, and for |uz|
    // above ~0.6 it amplifies the error instead of damping it. In double the effect is invisible
    // (worst measured |norm^2 - 1| was 6.6e-9 over 1.8e8 events), but float32 rounding is ~1e9
    // times larger, so renormalize after every event. One sqrt; the model is unchanged.
    const Real inv_norm = Real(1) / m::sqrt(p.ux * p.ux + p.uy * p.uy + p.uz * p.uz);
    p.ux *= inv_norm;
    p.uy *= inv_norm;
    p.uz *= inv_norm;
}

// Choose the next step. A fresh step is s = -ln(xi) / mu_t; a step interrupted at a boundary
// resumes with its dimensionless remainder in the new layer (MCML 3.2, 3.9).
template <class Real, class Rng>
TMC_HD void step_size(Photon<Real> &p, const Geometry<Real> &geo, Rng &rng) {
    const Layer<Real> &L = geo.layers[p.layer];
    const Real mut = L.mua + L.mus;
    if (p.sleft == Real(0)) {
        p.s = -m::log(uniform<Real>(rng)) / mut;
    } else {
        p.s = p.sleft / mut;
        p.sleft = Real(0);
    }
}

// If the step would cross the current layer's boundary, shorten it to stop on the boundary and
// keep the remainder in p.sleft. Returns true if a boundary is hit (MCML 3.9).
template <class Real> TMC_HD bool hit_boundary(Photon<Real> &p, const Geometry<Real> &geo) {
    const Layer<Real> &L = geo.layers[p.layer];
    Real dl_b;
    if (p.uz > Real(0)) {
        dl_b = (L.z1 - p.z) / p.uz;
    } else if (p.uz < Real(0)) {
        dl_b = (L.z0 - p.z) / p.uz;
    } else {
        return false;
    }
    if (p.s > dl_b) {
        p.sleft = (p.s - dl_b) * (L.mua + L.mus);
        p.s = dl_b;
        return true;
    }
    return false;
}

template <class Real> TMC_HD void hop(Photon<Real> &p) {
    p.x += p.s * p.ux;
    p.y += p.s * p.uy;
    p.z += p.s * p.uz;
}

// Deposit the absorbed fraction mu_a / mu_t of the packet's weight (MCML 3.3).
template <class Real, class Tally>
TMC_HD void drop(Photon<Real> &p, const Geometry<Real> &geo, Tally &tally) {
    const Layer<Real> &L = geo.layers[p.layer];
    const Real dwa = p.w * L.mua / (L.mua + L.mus);
    p.w -= dwa;
    tally.score_absorption(p, geo, dwa);
}

// At a boundary: reflect with the Fresnel probability, otherwise transmit into the neighbouring
// layer or out of the stack, where the weight is scored as reflectance (top) or transmittance
// (bottom) and the packet ends. MCML's default non-partial scheme (MCML 3.6-3.7).
template <class Real, class Tally, class Rng>
TMC_HD void cross_or_not(Photon<Real> &p, const Geometry<Real> &geo, Rng &rng, Tally &tally) {
    const Layer<Real> &L = geo.layers[p.layer];
    const Real ni = L.n;
    Real uz1 = Real(0);
    Real r;
    if (p.uz < Real(0)) { // moving up, toward the top boundary
        const Real nt = p.layer == 0 ? geo.n_above : geo.layers[p.layer - 1].n;
        r = (-p.uz <= L.cos_crit_top) ? Real(1) : fresnel(ni, nt, -p.uz, uz1);
        if (uniform<Real>(rng) > r) {
            if (p.layer == 0) {
                // The exit direction is refracted like any other crossing, so scale the lateral
                // components too and keep (ux, uy, uz) a unit vector at the moment of scoring.
                p.ux *= ni / nt;
                p.uy *= ni / nt;
                p.uz = -uz1;
                tally.score_reflectance(p, geo, p.w);
                p.dead = true;
            } else {
                --p.layer;
                p.ux *= ni / nt;
                p.uy *= ni / nt;
                p.uz = -uz1;
            }
        } else {
            p.uz = -p.uz;
        }
    } else { // moving down, toward the bottom boundary
        const bool last = p.layer == geo.num_layers - 1;
        const Real nt = last ? geo.n_below : geo.layers[p.layer + 1].n;
        r = (p.uz <= L.cos_crit_bottom) ? Real(1) : fresnel(ni, nt, p.uz, uz1);
        if (uniform<Real>(rng) > r) {
            if (last) {
                p.ux *= ni / nt;
                p.uy *= ni / nt;
                p.uz = uz1;
                tally.score_transmittance(p, geo, p.w);
                p.dead = true;
            } else {
                ++p.layer;
                p.ux *= ni / nt;
                p.uy *= ni / nt;
                p.uz = uz1;
            }
        } else {
            p.uz = -p.uz;
        }
    }
}

// Russian roulette for low-weight packets: survive with probability CHANCE and weight / CHANCE,
// otherwise die. Unbiased: the expected weight is unchanged (MCML 3.8).
template <class Real, class Rng> TMC_HD void roulette(Photon<Real> &p, Rng &rng) {
    using C = Constants<Real>;
    if (p.w == Real(0)) {
        p.dead = true;
    } else if (uniform<Real>(rng) < C::kChance) {
        p.w /= C::kChance;
    } else {
        p.dead = true;
    }
}

// One hop-drop-spin cycle plus roulette (MCML HopDropSpin). Exposed separately from
// simulate_photon so a GPU thread can interleave packets (photon relaunch).
template <class Real, class Tally, class Rng>
TMC_HD void step_once(Photon<Real> &p, const Geometry<Real> &geo, Rng &rng, Tally &tally) {
    step_size(p, geo, rng);
    if (hit_boundary(p, geo)) {
        hop(p);
        cross_or_not(p, geo, rng, tally);
    } else {
        hop(p);
        drop(p, geo, tally);
        spin(p, geo.layers[p.layer].g, rng);
    }
    if (!p.dead && p.w < Constants<Real>::kWeightThreshold) {
        roulette(p, rng);
    }
}

// Transport one packet from launch until it leaves the stack or loses roulette.
template <class Real, class Tally, class Rng>
TMC_HD void simulate_photon(const Geometry<Real> &geo, Rng &rng, Tally &tally) {
    Photon<Real> p;
    launch(p, geo);
    while (!p.dead) {
        step_once(p, geo, rng, tally);
    }
}

} // namespace tissuemc
