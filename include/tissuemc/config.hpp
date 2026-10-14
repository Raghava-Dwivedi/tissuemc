#pragma once
// Host-side run description (one MCML "run") and its conversion to the device-friendly Geometry.

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "tissuemc/geometry.hpp"

namespace tissuemc {

struct LayerSpec {
    double n;   // refractive index
    double mua; // absorption coefficient [1/cm]
    double mus; // scattering coefficient [1/cm]
    double g;   // anisotropy
    double d;   // thickness [cm]
};

struct RunSpec {
    std::string output_name; // as given in the .mci file (informational)
    uint64_t photons = 0;
    double dz = 0.01, dr = 0.01; // grid bin widths [cm]
    int nz = 1, nr = 1, na = 1;  // grid bin counts (na is accepted for MCML compatibility)
    double n_above = 1.0, n_below = 1.0;
    std::vector<LayerSpec> layers;
};

class ConfigError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

// Throws ConfigError naming the offending field if the run is not physically valid or is outside
// what this implementation supports (e.g. clear "glass" layers with mua = mus = 0).
inline void validate(const RunSpec &run) {
    auto fail = [](const std::string &msg) { throw ConfigError(msg); };
    if (run.photons == 0)
        fail("number of photons must be > 0");
    if (!(run.dz > 0) || !(run.dr > 0))
        fail("dz and dr must be > 0");
    if (run.nz < 1 || run.nr < 1 || run.na < 1)
        fail("grid bin counts must be >= 1");
    if (!(run.n_above >= 1.0) || !(run.n_below >= 1.0))
        fail("ambient refractive indices must be >= 1");
    if (run.layers.empty())
        fail("at least one layer is required");
    if (run.layers.size() > static_cast<size_t>(kMaxLayers))
        fail("at most " + std::to_string(kMaxLayers) + " layers are supported");
    for (size_t i = 0; i < run.layers.size(); ++i) {
        const LayerSpec &L = run.layers[i];
        const std::string where = "layer " + std::to_string(i + 1) + ": ";
        if (!(L.n >= 1.0))
            fail(where + "n must be >= 1");
        if (!(L.mua >= 0) || !(L.mus >= 0))
            fail(where + "mua and mus must be >= 0");
        if (!(L.mua + L.mus > 0))
            fail(where + "clear layers (mua = mus = 0) are not supported");
        if (!(L.g > -1.0 && L.g < 1.0))
            fail(where + "g must be in (-1, 1)");
        if (!(L.d > 0) || !std::isfinite(L.d))
            fail(where + "thickness must be finite and > 0");
    }
}

// Critical-angle cosine when going from index ni into nt: total internal reflection happens for
// cos(incidence) <= this value. Zero when ni <= nt (no critical angle). MCML 3.6.
inline double critical_cos(double ni, double nt) {
    return ni > nt ? std::sqrt(1.0 - (nt * nt) / (ni * ni)) : 0.0;
}

template <class Real> Geometry<Real> make_geometry(const RunSpec &run) {
    validate(run);
    Geometry<Real> geo{};
    geo.num_layers = static_cast<int>(run.layers.size());
    geo.n_above = static_cast<Real>(run.n_above);
    geo.n_below = static_cast<Real>(run.n_below);
    double z = 0.0;
    for (size_t i = 0; i < run.layers.size(); ++i) {
        const LayerSpec &s = run.layers[i];
        const double n_top = i == 0 ? run.n_above : run.layers[i - 1].n;
        const double n_bot = i + 1 == run.layers.size() ? run.n_below : run.layers[i + 1].n;
        Layer<Real> &L = geo.layers[i];
        L.n = static_cast<Real>(s.n);
        L.mua = static_cast<Real>(s.mua);
        L.mus = static_cast<Real>(s.mus);
        L.g = static_cast<Real>(s.g);
        L.z0 = static_cast<Real>(z);
        z += s.d;
        L.z1 = static_cast<Real>(z);
        L.cos_crit_top = static_cast<Real>(critical_cos(s.n, n_top));
        L.cos_crit_bottom = static_cast<Real>(critical_cos(s.n, n_bot));
    }
    geo.dz = static_cast<Real>(run.dz);
    geo.dr = static_cast<Real>(run.dr);
    geo.nz = run.nz;
    geo.nr = run.nr;
    return geo;
}

} // namespace tissuemc
