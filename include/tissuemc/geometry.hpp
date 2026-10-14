#pragma once
// Device-friendly description of a stack of planar tissue layers.
//
// Plain fixed-size arrays (no std::vector) so the same struct can be copied as-is into GPU
// constant memory. Built on the host from a parsed .mci run by make_geometry() in config.hpp.

#include "tissuemc/hd.hpp"

namespace tissuemc {

inline constexpr int kMaxLayers = 16;

template <class Real> struct Layer {
    Real n;               // refractive index
    Real mua;             // absorption coefficient [1/cm]
    Real mus;             // scattering coefficient [1/cm]
    Real g;               // anisotropy (mean cosine of the scattering angle)
    Real z0;              // depth of the top boundary [cm]
    Real z1;              // depth of the bottom boundary [cm]
    Real cos_crit_top;    // cosine of the critical angle at the top boundary (0 if none)
    Real cos_crit_bottom; // cosine of the critical angle at the bottom boundary (0 if none)
};

template <class Real> struct Geometry {
    int num_layers;
    Real n_above; // refractive index of the medium above the stack
    Real n_below; // refractive index of the medium below the stack
    Layer<Real> layers[kMaxLayers];
    Real dz; // depth bin width for the absorption profile A(z) [cm]
    Real dr; // radial bin width for the reflectance profile Rd(r) [cm]
    int nz;  // number of depth bins
    int nr;  // number of radial bins
};

} // namespace tissuemc
