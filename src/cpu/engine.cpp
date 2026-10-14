#include "tissuemc/cpu_engine.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <mutex>
#include <thread>

#include "tissuemc/accumulate.hpp"
#include "tissuemc/config.hpp"
#include "tissuemc/physics.hpp"

namespace tissuemc {
namespace {

// Scores one batch. Every accumulator is compensated; see accumulate.hpp for why.
class CpuTally {
  public:
    explicit CpuTally(const Geometry<double> &geo)
        : absorbed_(static_cast<size_t>(geo.num_layers)), az_(static_cast<size_t>(geo.nz)),
          rr_(static_cast<size_t>(geo.nr)) {}

    void score_absorption(const Photon<double> &p, const Geometry<double> &geo, double w) {
        ++events;
        check(p, w);
        absorbed_[static_cast<size_t>(p.layer)].add(w);
        az_[bin(p.z, geo.dz, geo.nz)].add(w);
    }

    void score_reflectance(const Photon<double> &p, const Geometry<double> &geo, double w) {
        check(p, w);
        rd.add(w);
        rr_[bin(std::sqrt(p.x * p.x + p.y * p.y), geo.dr, geo.nr)].add(w);
    }

    void score_transmittance(const Photon<double> &p, const Geometry<double> &, double w) {
        check(p, w);
        tt.add(w);
    }

    CompensatedSum<double> rd, tt;
    uint64_t events = 0;
    uint64_t anomalies = 0;

    std::vector<double> absorbed_values() const { return values(absorbed_); }
    std::vector<double> az_values() const { return values(az_); }
    std::vector<double> rr_values() const { return values(rr_); }

  private:
    // Bins clamp at both ends: anything deeper (or farther) than the grid lands in the last bin.
    static size_t bin(double x, double width, int count) {
        if (!(x > 0)) {
            return 0;
        }
        const double i = x / width;
        return i >= static_cast<double>(count) ? static_cast<size_t>(count - 1)
                                               : static_cast<size_t>(i);
    }

    // A finite weight and a unit direction vector are invariants of correct transport.
    void check(const Photon<double> &p, double w) {
        const double norm2 = p.ux * p.ux + p.uy * p.uy + p.uz * p.uz;
        if (!std::isfinite(w) || !std::isfinite(norm2) || std::fabs(norm2 - 1.0) > 1e-9) {
            ++anomalies;
        }
    }

    static std::vector<double> values(const std::vector<CompensatedSum<double>> &v) {
        std::vector<double> out;
        out.reserve(v.size());
        for (const auto &s : v) {
            out.push_back(s.value());
        }
        return out;
    }

    std::vector<CompensatedSum<double>> absorbed_, az_, rr_;
};

struct BatchOutput {
    BatchResult result;
    std::vector<double> az, rr;
};

BatchOutput run_batch(const Geometry<double> &geo, uint64_t seed, uint64_t first, uint64_t count) {
    CpuTally tally(geo);
    for (uint64_t id = first; id < first + count; ++id) {
        PhotonRng rng(seed, id);
        simulate_photon(geo, rng, tally);
    }
    BatchOutput out;
    out.result.first_photon = first;
    out.result.photons = count;
    out.result.rd = tally.rd.value();
    out.result.tt = tally.tt.value();
    out.result.absorbed = tally.absorbed_values();
    out.result.events = tally.events;
    out.result.anomalies = tally.anomalies;
    out.az = tally.az_values();
    out.rr = tally.rr_values();
    return out;
}

} // namespace

RunResult run_cpu(const RunSpec &run_in, const EngineOptions &opt) {
    RunSpec run = run_in;
    if (opt.photons != 0) {
        run.photons = opt.photons;
    }
    const Geometry<double> geo = make_geometry<double>(run); // validates the run
    if (opt.batches < 1) {
        throw ConfigError("batches must be >= 1");
    }
    if (static_cast<uint64_t>(opt.batches) > run.photons) {
        throw ConfigError("batches cannot exceed the number of photons");
    }
    constexpr int kMaxThreads = 1024;
    if (opt.threads < 0 || opt.threads > kMaxThreads) {
        throw ConfigError("threads must be in [0, " + std::to_string(kMaxThreads) +
                          "] (0 = one per hardware thread)");
    }
    int threads = opt.threads;
    if (threads == 0) {
        threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }
    threads = std::min(threads, opt.batches); // a thread with no batch would only idle

    const auto t0 = std::chrono::steady_clock::now();

    const uint64_t nb = static_cast<uint64_t>(opt.batches);
    const uint64_t base = run.photons / nb;
    const uint64_t extra = run.photons % nb; // the first `extra` batches get one more photon

    RunResult result;
    result.spec = run;
    result.engine = "cpu";
    result.precision = "float64";
    result.seed = opt.seed;
    result.threads = threads;
    result.specular = specular_reflectance(geo);
    result.a_z.assign(static_cast<size_t>(geo.nz), 0.0);
    result.rd_r.assign(static_cast<size_t>(geo.nr), 0.0);

    // Batch b covers photon ids [first_of(b), first_of(b) + count_of(b)).
    auto count_of = [&](uint64_t b) { return base + (b < extra ? 1 : 0); };
    auto first_of = [&](uint64_t b) { return b * base + std::min(b, extra); };

    // Batches are independent: each has its own tally and its photons use streams keyed by
    // (seed, photon id). Workers claim batches from a shared counter in any order, and the
    // results are merged below in batch order, so the output does not depend on the thread count
    // or on scheduling.
    std::vector<BatchOutput> outs(nb);
    std::atomic<uint64_t> next_batch{0};
    std::exception_ptr failure;
    std::mutex failure_mutex;
    auto worker = [&]() {
        try {
            for (;;) {
                const uint64_t b = next_batch.fetch_add(1, std::memory_order_relaxed);
                if (b >= nb) {
                    return;
                }
                outs[b] = run_batch(geo, opt.seed, first_of(b), count_of(b));
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure) {
                failure = std::current_exception();
            }
            next_batch.store(nb, std::memory_order_relaxed); // stop the other workers early
        }
    };
    {
        std::vector<std::thread> pool;
        pool.reserve(static_cast<size_t>(threads - 1));
        for (int t = 1; t < threads; ++t) {
            pool.emplace_back(worker);
        }
        worker(); // the calling thread works too
        for (auto &th : pool) {
            th.join();
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }

    for (uint64_t b = 0; b < nb; ++b) {
        for (size_t i = 0; i < outs[b].az.size(); ++i) {
            result.a_z[i] += outs[b].az[i];
        }
        for (size_t i = 0; i < outs[b].rr.size(); ++i) {
            result.rd_r[i] += outs[b].rr[i];
        }
        result.batches.push_back(std::move(outs[b].result));
    }

    result.wall_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return result;
}

} // namespace tissuemc
