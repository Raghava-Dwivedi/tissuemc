#pragma once
// How photons are split into batches. Shared by the CPU engine and the GPU kernel so that batch b
// covers the same photon ids in both, which makes per-batch results directly comparable.

#include <stdint.h>

#include "tissuemc/hd.hpp"

namespace tissuemc {

struct BatchPlan {
    uint64_t base;  // photons in every batch, before the remainder is spread out
    uint64_t extra; // the first `extra` batches get one more photon
    uint64_t batches;

    static BatchPlan make(uint64_t photons, uint64_t batches) {
        return BatchPlan{photons / batches, photons % batches, batches};
    }

    TMC_HD uint64_t count_of(uint64_t b) const { return base + (b < extra ? 1 : 0); }

    TMC_HD uint64_t first_of(uint64_t b) const { return b * base + (b < extra ? b : extra); }

    // The batch that contains photon `id`; the inverse of first_of.
    TMC_HD uint64_t batch_of(uint64_t id) const {
        const uint64_t boundary = extra * (base + 1); // ids below this are in the larger batches
        return id < boundary ? id / (base + 1) : extra + (id - boundary) / base;
    }
};

} // namespace tissuemc
