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

#include "tissuemc/batching.hpp"
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

    // Prepare for the next photon on a lane that is reused (rung R3): clear the per-photon
    // accumulators and counters and point them at the new photon's batch.
    TMC_HD void begin_photon(int batch_index) {
        batch = batch_index;
        events = 0;
        anomalies = 0;
        for (int i = 0; i < kMaxLayers; ++i) {
            absorbed_local[i] = 0.0f;
        }
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

// Rung R5: R1's scoring with the per-layer accumulator held in registers.
//
// The profiler showed that the kernels' time goes to per-thread local memory, so R5 removes the
// run-time-indexed arrays from the hot loop. R1 keeps one accumulator per layer in an array and
// indexes it by the packet's layer, which the compiler places in local memory. Here the packet's
// absorbed weight is summed in a single register for the layer it is currently in; when the packet
// changes layer, that partial sum is published and a new one starts. The global atomics are
// therefore one per layer change plus one at the end, which for a photon that stays in one layer
// is still one, and the depth and radius profiles are unchanged. The random generator is
// RegPhotonRng, for the same reason.
//
// A layer's total is now published in several partial sums rather than one. Each is a float sum of
// at most a photon's weight, so the rounding is no worse than R1's (validation checks it).
template <class T> struct GpuTallyR5 {
    GpuTallyArrays<T> out;
    int batch;
    int layers;
    unsigned long long events;
    unsigned long long anomalies;
    float run_sum;
    int run_layer;

    TMC_HD GpuTallyR5(const GpuTallyArrays<T> &arrays, int batch_index, int num_layers)
        : out(arrays), batch(batch_index), layers(num_layers), events(0), anomalies(0),
          run_sum(0.0f), run_layer(0) {}

    TMC_HD void score_absorption(const Photon<float> &p, const Geometry<float> &geo, float w) {
        ++events;
        check(p, w);
        if (p.layer != run_layer) {
            publish();
            run_layer = p.layer;
        }
        run_sum += w;
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

    // Start the next photon on a lane that is reused (rung R6).
    TMC_HD void begin_photon(int batch_index) {
        batch = batch_index;
        events = 0;
        anomalies = 0;
        run_sum = 0.0f;
        run_layer = 0;
    }

    // Publish what is left of the running sum and the counters when the photon ends.
    TMC_HD void flush_counters() {
        publish();
        TMC_ATOMIC_ADD(&out.events[batch], events);
        TMC_ATOMIC_ADD(&out.anomalies[batch], anomalies);
    }

  private:
    TMC_HD void publish() {
        if (run_sum != 0.0f) {
            TMC_ATOMIC_ADD(&out.absorbed[batch * layers + run_layer], static_cast<T>(run_sum));
            run_sum = 0.0f;
        }
    }

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
TMC_HD inline void run_photon_r5(const Geometry<float> &geo, uint64_t seed, uint64_t id, int batch,
                                 const GpuTallyArrays<T> &arrays) {
    GpuTallyR5<T> tally(arrays, batch, geo.num_layers);
    RegPhotonRng rng(seed, id);
    simulate_photon(geo, rng, tally);
    tally.flush_counters();
}

// Rung R3: persistent lanes that relaunch.
//
// In R0 and R1 a thread runs one photon to its end, so a warp is held up by its longest photon
// while the other 31 lanes sit idle (the profiler measured 4.75 of 32 lanes active). Here a thread
// is a lane that claims photons from a shared counter: as soon as its photon ends it claims the
// next and carries on, so lanes stay busy until the photons run out. One call to advance() does at
// most one hop-drop-spin step, so the host emulation can interleave many lanes in any order and
// exercise the state machine the kernel runs.
//
// Photon i still uses the stream (seed, i) and counts toward batch_of(i) whichever lane runs it,
// so every batch holds the same photons as in R0 and R1. The scoring is R1's: per-photon sums,
// published once when the photon ends.

// Returns the value the counter held before the increment (the next unclaimed photon number).
TMC_HD inline unsigned long long claim_next(unsigned long long *counter) {
#if defined(__CUDA_ARCH__)
    return atomicAdd(counter, 1ULL);
#else
    return (*counter)++;
#endif
}

template <class T, class Tally, class Rng> class PersistentLaneOf {
  public:
    TMC_HD PersistentLaneOf(const GpuTallyArrays<T> &arrays, int num_layers)
        : tally_(arrays, 0, num_layers), p_{}, rng_(0, 0), has_photon_(false) {}

    // Advance by at most one step. A lane without a photon first claims photon first + n for the
    // n-th claim; once that reaches `end` there is nothing left and the lane stops (false).
    TMC_HD bool advance(const Geometry<float> &geo, uint64_t seed, const BatchPlan &plan,
                        uint64_t first, uint64_t end, unsigned long long *next) {
        if (!has_photon_) {
            const uint64_t id = first + claim_next(next);
            if (id >= end) {
                return false;
            }
            tally_.begin_photon(static_cast<int>(plan.batch_of(id)));
            rng_ = Rng(seed, id);
            launch(p_, geo);
            has_photon_ = true;
        }
        step_once(p_, geo, rng_, tally_);
        if (p_.dead) {
            tally_.flush_counters();
            has_photon_ = false;
        }
        return true;
    }

  private:
    Tally tally_;
    Photon<float> p_;
    Rng rng_;
    bool has_photon_;
};

// Rung R3 and rung R6: the same lane, with R1's array accumulator and generator, or with R5's
// register-resident ones.
template <class T> using PersistentLane = PersistentLaneOf<T, GpuTallyR1<T>, PhotonRng>;
template <class T> using PersistentLaneRegs = PersistentLaneOf<T, GpuTallyR5<T>, RegPhotonRng>;

// Dispatch on the rung number, shared by the kernel and the host emulation for rungs 0 and 1. The
// other rungs do not go through here: rungs 3 and 6 are not one photon per call (see
// PersistentLaneOf), and rung 5 has its own kernel so that R0 and R1 stay exactly what their
// recorded results measured.
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
