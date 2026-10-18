#pragma once
// Philox4x32-10 counter-based random number generator.
//
// Reference: J. K. Salmon, M. A. Moraes, R. O. Dror, D. E. Shaw, "Parallel random numbers: as easy
// as 1, 2, 3", SC'11 (2011). Constants and round structure follow the Random123 reference
// implementation; tests/cpp/test_philox.cpp checks its published known-answer vectors.
//
// Why counter-based: the output is a pure function of (counter, key). A photon's random stream is
// fully determined by (seed, photon id), with no per-thread state to store or seed. Any CPU thread
// or GPU thread can simulate any photon and get the same random numbers.

#include <stdint.h>

#include "tissuemc/hd.hpp"

namespace tissuemc {

struct U32x4 {
    uint32_t v[4];
};

struct PhiloxKey {
    uint32_t k[2];
};

namespace philox_detail {

inline constexpr uint32_t kM0 = 0xD2511F53u;
inline constexpr uint32_t kM1 = 0xCD9E8D57u;
inline constexpr uint32_t kW0 = 0x9E3779B9u; // golden ratio
inline constexpr uint32_t kW1 = 0xBB67AE85u; // sqrt(3) - 1

TMC_HD inline void mulhilo(uint32_t a, uint32_t b, uint32_t &hi, uint32_t &lo) {
    const uint64_t p = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    hi = static_cast<uint32_t>(p >> 32);
    lo = static_cast<uint32_t>(p);
}

TMC_HD inline U32x4 round(const U32x4 &c, const PhiloxKey &key) {
    uint32_t hi0, lo0, hi1, lo1;
    mulhilo(kM0, c.v[0], hi0, lo0);
    mulhilo(kM1, c.v[2], hi1, lo1);
    return U32x4{{hi1 ^ c.v[1] ^ key.k[0], lo1, hi0 ^ c.v[3] ^ key.k[1], lo0}};
}

} // namespace philox_detail

// One Philox4x32 block with 10 rounds: 128 bits of output per (counter, key).
TMC_HD inline U32x4 philox4x32_10(U32x4 ctr, PhiloxKey key) {
    ctr = philox_detail::round(ctr, key);
    for (int i = 1; i < 10; ++i) {
        key.k[0] += philox_detail::kW0;
        key.k[1] += philox_detail::kW1;
        ctr = philox_detail::round(ctr, key);
    }
    return ctr;
}

// The random stream for one photon packet. Counter words 0-1 hold the 64-bit photon id, word 2
// counts 128-bit blocks drawn so far, and word 3 is reserved (always 0). The key is the 64-bit
// run seed.
class PhotonRng {
  public:
    TMC_HD PhotonRng(uint64_t seed, uint64_t photon_id)
        : ctr_{{static_cast<uint32_t>(photon_id), static_cast<uint32_t>(photon_id >> 32), 0u, 0u}},
          key_{{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)}},
          buf_{{0, 0, 0, 0}}, idx_(4) {}

    TMC_HD uint32_t next_u32() {
        if (idx_ == 4) {
            buf_ = philox4x32_10(ctr_, key_);
            ++ctr_.v[2];
            idx_ = 0;
        }
        return buf_.v[idx_++];
    }

  private:
    U32x4 ctr_;
    PhiloxKey key_;
    U32x4 buf_;
    int idx_;
};

// The same stream as PhotonRng, laid out for the GPU's registers (rungs R5 and R6). PhotonRng keeps
// the four words of the current block in an array and reads them at a run-time index, which the
// compiler places in per-thread local memory. Here the unread words are four scalars that shift
// down as they are used, so nothing is indexed at run time and the whole generator can stay in
// registers. The output is word for word the same; tests/cpp/test_philox.cpp checks it. PhotonRng
// is left as it was so that the earlier rungs remain what their recorded results measured.
class RegPhotonRng {
  public:
    TMC_HD RegPhotonRng(uint64_t seed, uint64_t photon_id)
        : ctr0_(static_cast<uint32_t>(photon_id)), ctr1_(static_cast<uint32_t>(photon_id >> 32)),
          ctr2_(0u), k0_(static_cast<uint32_t>(seed)), k1_(static_cast<uint32_t>(seed >> 32)),
          w0_(0u), w1_(0u), w2_(0u), w3_(0u), left_(0) {}

    TMC_HD uint32_t next_u32() {
        if (left_ == 0) {
            const U32x4 block =
                philox4x32_10(U32x4{{ctr0_, ctr1_, ctr2_, 0u}}, PhiloxKey{{k0_, k1_}});
            ++ctr2_;
            w0_ = block.v[0];
            w1_ = block.v[1];
            w2_ = block.v[2];
            w3_ = block.v[3];
            left_ = 4;
        }
        const uint32_t out = w0_;
        w0_ = w1_;
        w1_ = w2_;
        w2_ = w3_;
        --left_;
        return out;
    }

  private:
    uint32_t ctr0_, ctr1_, ctr2_;
    uint32_t k0_, k1_;
    uint32_t w0_, w1_, w2_, w3_;
    int left_;
};

// Uniform deviate strictly inside (0, 1); never 0 (log is safe) and never 1. Every draw consumes
// exactly one 32-bit word in both precisions, so a float engine and a double engine given the same
// (seed, photon id) see the same sequence of draws.
template <class Real> TMC_HD Real to_unit_open(uint32_t x);

template <> TMC_HD inline double to_unit_open<double>(uint32_t x) {
    return (static_cast<double>(x) + 0.5) * 0x1p-32; // max (2^32 - 0.5) / 2^32 < 1
}

template <> TMC_HD inline float to_unit_open<float>(uint32_t x) {
    // A float holds 24 significant bits, so keep 23 random bits; (2^23 - 0.5) / 2^23 < 1 exactly.
    return (static_cast<float>(x >> 9) + 0.5f) * 0x1p-23f;
}

template <class Real, class Rng> TMC_HD inline Real uniform(Rng &rng) {
    return to_unit_open<Real>(rng.next_u32());
}

} // namespace tissuemc
