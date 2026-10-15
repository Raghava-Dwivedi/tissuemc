#pragma once
// The GPU engine's scoring, written so that it also compiles for the host.
//
// Rung R0 is the deliberately naive baseline: every photon adds its results straight into global
// float32 arrays with atomicAdd. Under a plain C++ compiler the atomic becomes an ordinary add, so
// the exact kernel body can be run on the CPU and compared with the CPU engine (see
// tests/cpp/test_gpu_emulation.cpp). Only the launch and the atomics are left to test on a GPU.
//
// The tallies are generic over their precision T. R0 uses double (hardware atomicAdd on double
// needs compute capability 6.0, which any T4 has): naive, but correct. T = float is kept as a
// measured demonstration of why a float32 total is not safe: once a running total is large, an
// addition smaller than half its last digit is silently lost, so absorbed weight (many small
// additions into a large total) comes out low and energy conservation fails.

#include <stdint.h>

#include "tissuemc/geometry.hpp"
#include "tissuemc/hd.hpp"
#include "tissuemc/physics.hpp"

#if defined(__CUDA_ARCH__)
#define TMC_ATOMIC_ADD(ptr, val) atomicAdd((ptr), (val))
#else
#define TMC_ATOMIC_ADD(ptr, val) (*(ptr) += (val))
#endif

namespace tissuemc {

// Device (or host) arrays that the tallies accumulate into.
template <class T> struct GpuTallyArrays {
    T *rd;                      // [batches]           diffuse reflectance weight
    T *tt;                      // [batches]           transmitted weight
    T *absorbed;                // [batches * layers]  absorbed weight per layer
    T *az;                      // [nz]                absorbed weight per depth bin (all photons)
    T *rr;                      // [nr]                diffuse reflectance per radial bin
    unsigned long long *events; // [batches]
    unsigned long long *anomalies; // [batches]
};

template <class T> struct GpuTally {
    GpuTallyArrays<T> out;
    int batch;
    int layers;
    unsigned long long events;
    unsigned long long anomalies;

    TMC_HD GpuTally(const GpuTallyArrays<T> &arrays, int batch_index, int num_layers)
        : out(arrays), batch(batch_index), layers(num_layers), events(0), anomalies(0) {}

    TMC_HD void score_absorption(const Photon<float> &p, const Geometry<float> &geo, float w) {
        ++events;
        check(p, w);
        TMC_ATOMIC_ADD(&out.absorbed[batch * layers + p.layer], static_cast<T>(w));
        TMC_ATOMIC_ADD(&out.az[bin(p.z, geo.dz, geo.nz)], static_cast<T>(w));
    }

    TMC_HD void score_reflectance(const Photon<float> &p, const Geometry<float> &geo, float w) {
        check(p, w);
        TMC_ATOMIC_ADD(&out.rd[batch], static_cast<T>(w));
        const float r = m::sqrt(p.x * p.x + p.y * p.y);
        TMC_ATOMIC_ADD(&out.rr[bin(r, geo.dr, geo.nr)], static_cast<T>(w));
    }

    TMC_HD void score_transmittance(const Photon<float> &p, const Geometry<float> &, float w) {
        check(p, w);
        TMC_ATOMIC_ADD(&out.tt[batch], static_cast<T>(w));
    }

    // Called once per photon after it ends: publish the thread-local counters.
    TMC_HD void flush_counters() {
        TMC_ATOMIC_ADD(&out.events[batch], events);
        TMC_ATOMIC_ADD(&out.anomalies[batch], anomalies);
    }

  private:
    // Bins clamp at both ends: anything deeper (or farther) than the grid lands in the last bin.
    TMC_HD static int bin(float x, float width, int count) {
        if (!(x > 0.0f)) {
            return 0;
        }
        const float i = x / width;
        return i >= static_cast<float>(count) ? count - 1 : static_cast<int>(i);
    }

    // The same invariants the CPU engine checks: finite weight and a unit direction vector.
    TMC_HD void check(const Photon<float> &p, float w) {
        const float norm2 = p.ux * p.ux + p.uy * p.uy + p.uz * p.uz;
        const float dev = m::fabs(norm2 - 1.0f);
        // NaN fails every comparison, so test with a negated "<=" to catch it.
        if (!(w == w) || !(dev <= 1e-5f)) {
            ++anomalies;
        }
    }
};

// Rung R1: absorbed weight per layer is accumulated in the thread and published once per photon,
// instead of one global atomic add per absorption event. The depth and radius profiles are still
// added event by event; they are the subject of the next rung.
//
// The per-photon accumulator is float: one photon's total is at most 1, so a float sum is exact
// enough, and unlike a shared running total it cannot grow large. Validation checks this.
template <class T> struct GpuTallyR1 {
    GpuTallyArrays<T> out;
    int batch;
    int layers;
    unsigned long long events;
    unsigned long long anomalies;
    float absorbed_local[kMaxLayers];

    TMC_HD GpuTallyR1(const GpuTallyArrays<T> &arrays, int batch_index, int num_layers)
        : out(arrays), batch(batch_index), layers(num_layers), events(0), anomalies(0) {
        for (int i = 0; i < kMaxLayers; ++i) {
            absorbed_local[i] = 0.0f;
        }
    }

    TMC_HD void score_absorption(const Photon<float> &p, const Geometry<float> &geo, float w) {
        ++events;
        check(p, w);
        absorbed_local[p.layer] += w;
        TMC_ATOMIC_ADD(&out.az[bin(p.z, geo.dz, geo.nz)], static_cast<T>(w));
    }

    TMC_HD void score_reflectance(const Photon<float> &p, const Geometry<float> &geo, float w) {
        check(p, w);
        TMC_ATOMIC_ADD(&out.rd[batch], static_cast<T>(w));
        const float r = m::sqrt(p.x * p.x + p.y * p.y);
        TMC_ATOMIC_ADD(&out.rr[bin(r, geo.dr, geo.nr)], static_cast<T>(w));
    }

    TMC_HD void score_transmittance(const Photon<float> &p, const Geometry<float> &, float w) {
        check(p, w);
        TMC_ATOMIC_ADD(&out.tt[batch], static_cast<T>(w));
    }

    // Publish the per-photon accumulators and counters: at most layers + 2 atomic adds.
    TMC_HD void flush_counters() {
        for (int l = 0; l < layers; ++l) {
            if (absorbed_local[l] != 0.0f) {
                TMC_ATOMIC_ADD(&out.absorbed[batch * layers + l],
                               static_cast<T>(absorbed_local[l]));
            }
        }
        TMC_ATOMIC_ADD(&out.events[batch], events);
        TMC_ATOMIC_ADD(&out.anomalies[batch], anomalies);
    }

  private:
    TMC_HD static int bin(float x, float width, int count) {
        if (!(x > 0.0f)) {
            return 0;
        }
        const float i = x / width;
        return i >= static_cast<float>(count) ? count - 1 : static_cast<int>(i);
    }

    TMC_HD void check(const Photon<float> &p, float w) {
        const float norm2 = p.ux * p.ux + p.uy * p.uy + p.uz * p.uz;
        const float dev = m::fabs(norm2 - 1.0f);
        if (!(w == w) || !(dev <= 1e-5f)) {
            ++anomalies;
        }
    }
};

template <class T>
TMC_HD inline void run_photon_r1(const Geometry<float> &geo, uint64_t seed, uint64_t id, int batch,
                                 const GpuTallyArrays<T> &arrays) {
    GpuTallyR1<T> tally(arrays, batch, geo.num_layers);
    PhotonRng rng(seed, id);
    simulate_photon(geo, rng, tally);
    tally.flush_counters();
}

// Dispatch on the rung number, shared by the kernel and the host emulation.
template <class T>
TMC_HD inline void run_photon(int rung, const Geometry<float> &geo, uint64_t seed, uint64_t id,
                              int batch, const GpuTallyArrays<T> &arrays) {
    if (rung == 1) {
        run_photon_r1(geo, seed, id, batch, arrays);
    } else {
        run_photon_r0(geo, seed, id, batch, arrays);
    }
}

// One photon, exactly as the kernel runs it. `batch` is the batch that owns photon `id`.
template <class T>
TMC_HD inline void run_photon_r0(const Geometry<float> &geo, uint64_t seed, uint64_t id, int batch,
                                 const GpuTallyArrays<T> &arrays) {
    GpuTally<T> tally(arrays, batch, geo.num_layers);
    PhotonRng rng(seed, id);
    simulate_photon(geo, rng, tally);
    tally.flush_counters();
}

} // namespace tissuemc
