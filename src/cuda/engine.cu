#include "tissuemc/cuda_engine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include "tissuemc/batching.hpp"
#include "tissuemc/config.hpp"
#include "tissuemc/gpu_tally.hpp"

namespace tissuemc {
namespace {

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        const cudaError_t tmc_err = (call);                                                        \
        if (tmc_err != cudaSuccess) {                                                              \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" +              \
                                     std::to_string(__LINE__) + ": " +                             \
                                     cudaGetErrorString(tmc_err));                                 \
        }                                                                                          \
    } while (0)

constexpr int kBlockSize = 128;
constexpr uint64_t kPhotonsPerLaunch = uint64_t(1) << 26;

// Rungs 0 and 1: every thread simulates one photon from launch to death. Rung 0 adds each result
// straight into the global tallies; rung 1 publishes per-photon sums once (see gpu_tally.hpp).
template <class T>
__global__ void photon_kernel(Geometry<float> geo, BatchPlan plan, uint64_t seed, uint64_t first,
                              uint64_t count, GpuTallyArrays<T> arrays, int rung) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const uint64_t id = first + i;
    run_photon(rung, geo, seed, id, static_cast<int>(plan.batch_of(id)), arrays);
}

// Rung 3: a fixed set of resident threads. Each is a lane that relaunches a new photon whenever its
// own ends, until the shared counter says the photons have run out (see PersistentLane).
template <class T>
__global__ void persistent_kernel(Geometry<float> geo, BatchPlan plan, uint64_t seed,
                                  uint64_t first, uint64_t end, GpuTallyArrays<T> arrays,
                                  unsigned long long *next) {
    PersistentLane<T> lane(arrays, geo.num_layers);
    while (lane.advance(geo, seed, plan, first, end, next)) {
    }
}

// Rung 5: one photon per thread, exactly as rungs 0 and 1, but with the generator and the per-layer
// accumulator in registers (see GpuTallyR5). A kernel of its own, so that photon_kernel, which
// rungs 0 and 1 share, stays what their recorded results measured.
template <class T>
__global__ void regs_kernel(Geometry<float> geo, BatchPlan plan, uint64_t seed, uint64_t first,
                            uint64_t count, GpuTallyArrays<T> arrays) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const uint64_t id = first + i;
    run_photon_r5(geo, seed, id, static_cast<int>(plan.batch_of(id)), arrays);
}

// Rung 6: rung 3's persistent lanes, with rung 5's register-resident state.
template <class T>
__global__ void persistent_regs_kernel(Geometry<float> geo, BatchPlan plan, uint64_t seed,
                                       uint64_t first, uint64_t end, GpuTallyArrays<T> arrays,
                                       unsigned long long *next) {
    PersistentLaneRegs<T> lane(arrays, geo.num_layers);
    while (lane.advance(geo, seed, plan, first, end, next)) {
    }
}

// Rung 7: rung 6 with the depth and radius profiles privatized in shared memory.
//
// Every absorption event adds to the depth profile, and the benchmark's profile has 40 bins, so
// about 6e9 global atomic adds serialized on a few cache lines and were 92% of rung 6's time. Here
// each warp keeps its own copy of the two profiles in shared memory, so the adds only contend with
// the other lanes of the same warp and never leave the multiprocessor; at the end of the kernel
// one thread per bin sums the block's copies and makes a single global add. The accumulators have
// the tally's precision T (double by default), because a float total in shared memory would lose
// small additions just as a global one does. The summation order differs from rung 6's, so the
// profiles agree to rounding, not bit for bit; the totals and event counts do not change.
template <class T>
__global__ void persistent_shared_kernel(Geometry<float> geo, BatchPlan plan, uint64_t seed,
                                         uint64_t first, uint64_t end, GpuTallyArrays<T> arrays,
                                         unsigned long long *next) {
    extern __shared__ unsigned char tmc_shared[];
    T *copies = reinterpret_cast<T *>(tmc_shared);
    const int stride = geo.nz + geo.nr; // one warp's copy: depth bins, then radius bins
    const int warps = blockDim.x / 32;
    for (int i = threadIdx.x; i < warps * stride; i += blockDim.x) {
        copies[i] = T(0);
    }
    __syncthreads();
    GpuTallyArrays<T> mine = arrays;
    mine.az = copies + (threadIdx.x / 32) * stride;
    mine.rr = mine.az + geo.nz;
    PersistentLaneRegs<T> lane(mine, geo.num_layers);
    while (lane.advance(geo, seed, plan, first, end, next)) {
    }
    __syncthreads();
    for (int i = threadIdx.x; i < stride; i += blockDim.x) {
        T sum = T(0);
        for (int w = 0; w < warps; ++w) {
            sum += copies[w * stride + i];
        }
        if (sum != T(0)) {
            atomicAdd(i < geo.nz ? &arrays.az[i] : &arrays.rr[i - geo.nz], sum);
        }
    }
}

// Rung 8: rung 7 with the depth-profile adds of a warp's lanes combined by bin before the shared
// add (see aggregated_add). A kernel of its own, so persistent_shared_kernel stays what the
// recorded rung 7 results measured.
template <class T>
__global__ void persistent_shared_aggregated_kernel(Geometry<float> geo, BatchPlan plan,
                                                    uint64_t seed, uint64_t first, uint64_t end,
                                                    GpuTallyArrays<T> arrays,
                                                    unsigned long long *next) {
    extern __shared__ unsigned char tmc_shared[];
    T *copies = reinterpret_cast<T *>(tmc_shared);
    const int stride = geo.nz + geo.nr;
    const int warps = blockDim.x / 32;
    for (int i = threadIdx.x; i < warps * stride; i += blockDim.x) {
        copies[i] = T(0);
    }
    __syncthreads();
    GpuTallyArrays<T> mine = arrays;
    mine.az = copies + (threadIdx.x / 32) * stride;
    mine.rr = mine.az + geo.nz;
    PersistentLaneRegsAggregated<T> lane(mine, geo.num_layers);
    while (lane.advance(geo, seed, plan, first, end, next)) {
    }
    __syncthreads();
    for (int i = threadIdx.x; i < stride; i += blockDim.x) {
        T sum = T(0);
        for (int w = 0; w < warps; ++w) {
            sum += copies[w * stride + i];
        }
        if (sum != T(0)) {
            atomicAdd(i < geo.nz ? &arrays.az[i] : &arrays.rr[i - geo.nz], sum);
        }
    }
}

// The diagnostic: rung 6 without the depth and radius profile tallies. A kernel of its own, so
// persistent_regs_kernel stays what the recorded rung 6 results measured.
template <class T>
__global__ void persistent_regs_noprofiles_kernel(Geometry<float> geo, BatchPlan plan,
                                                  uint64_t seed, uint64_t first, uint64_t end,
                                                  GpuTallyArrays<T> arrays,
                                                  unsigned long long *next) {
    PersistentLaneRegsNoProfiles<T> lane(arrays, geo.num_layers);
    while (lane.advance(geo, seed, plan, first, end, next)) {
    }
}

template <class T> class DeviceArray {
  public:
    explicit DeviceArray(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr_), count_ * sizeof(T)));
        CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
    }
    ~DeviceArray() { cudaFree(ptr_); }
    DeviceArray(const DeviceArray &) = delete;
    DeviceArray &operator=(const DeviceArray &) = delete;

    T *get() const { return ptr_; }

    std::vector<T> to_host() const {
        std::vector<T> out(count_);
        CUDA_CHECK(cudaMemcpy(out.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
        return out;
    }

  private:
    T *ptr_ = nullptr;
    size_t count_;
};

template <class T> std::vector<double> as_double(const std::vector<T> &v) {
    return std::vector<double>(v.begin(), v.end());
}

// Runs the whole simulation with running totals of precision T.
template <class T>
RunResult run_with_tallies(const RunSpec &run, const EngineOptions &opt, const Geometry<float> &geo,
                           const cudaDeviceProp &prop) {
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t nb = static_cast<uint64_t>(opt.batches);
    const BatchPlan plan = BatchPlan::make(run.photons, nb);
    const size_t layers = static_cast<size_t>(geo.num_layers);

    DeviceArray<T> d_rd(nb), d_tt(nb), d_abs(nb * layers);
    DeviceArray<T> d_az(static_cast<size_t>(geo.nz)), d_rr(static_cast<size_t>(geo.nr));
    DeviceArray<unsigned long long> d_events(nb), d_anomalies(nb);
    const GpuTallyArrays<T> arrays{d_rd.get(), d_tt.get(),     d_abs.get(),      d_az.get(),
                                   d_rr.get(), d_events.get(), d_anomalies.get()};

    DeviceArray<unsigned long long> d_next(1); // rung 3: the next unclaimed photon number

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    if (opt.rung == 3 || opt.rung == 6 || opt.rung == 7 || opt.rung == 8) {
        // As many blocks as can be resident at once, so every lane stays busy until the photons
        // run out; never more lanes than photons.
        int per_sm = 0;
        // Rung 7: one copy of the depth and radius profiles per warp, in shared memory.
        const size_t shared_bytes =
            static_cast<size_t>(kBlockSize / 32) * static_cast<size_t>(geo.nz + geo.nr) * sizeof(T);
        if (opt.rung == 3) {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, persistent_kernel<T>,
                                                                     kBlockSize, 0));
        } else if (opt.rung == 7) {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &per_sm, persistent_shared_kernel<T>, kBlockSize, shared_bytes));
        } else if (opt.rung == 8) {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &per_sm, persistent_shared_aggregated_kernel<T>, kBlockSize, shared_bytes));
        } else if (opt.profiles) {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &per_sm, persistent_regs_kernel<T>, kBlockSize, 0));
        } else {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &per_sm, persistent_regs_noprofiles_kernel<T>, kBlockSize, 0));
        }
        const uint64_t resident = static_cast<uint64_t>(std::max(per_sm, 1)) *
                                  static_cast<uint64_t>(prop.multiProcessorCount);
        const uint64_t needed = (run.photons + kBlockSize - 1) / kBlockSize;
        const unsigned blocks = static_cast<unsigned>(std::min(resident, needed));
        if (opt.rung == 3) {
            persistent_kernel<T>
                <<<blocks, kBlockSize>>>(geo, plan, opt.seed, 0, run.photons, arrays, d_next.get());
        } else if (opt.rung == 7) {
            persistent_shared_kernel<T><<<blocks, kBlockSize, shared_bytes>>>(
                geo, plan, opt.seed, 0, run.photons, arrays, d_next.get());
        } else if (opt.rung == 8) {
            persistent_shared_aggregated_kernel<T><<<blocks, kBlockSize, shared_bytes>>>(
                geo, plan, opt.seed, 0, run.photons, arrays, d_next.get());
        } else if (opt.profiles) {
            persistent_regs_kernel<T>
                <<<blocks, kBlockSize>>>(geo, plan, opt.seed, 0, run.photons, arrays, d_next.get());
        } else {
            persistent_regs_noprofiles_kernel<T>
                <<<blocks, kBlockSize>>>(geo, plan, opt.seed, 0, run.photons, arrays, d_next.get());
        }
        CUDA_CHECK(cudaGetLastError());
    } else {
        for (uint64_t first = 0; first < run.photons; first += kPhotonsPerLaunch) {
            const uint64_t count = std::min(kPhotonsPerLaunch, run.photons - first);
            const unsigned blocks = static_cast<unsigned>((count + kBlockSize - 1) / kBlockSize);
            if (opt.rung == 5) {
                regs_kernel<T><<<blocks, kBlockSize>>>(geo, plan, opt.seed, first, count, arrays);
            } else {
                photon_kernel<T>
                    <<<blocks, kBlockSize>>>(geo, plan, opt.seed, first, count, arrays, opt.rung);
            }
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const std::vector<double> rd = as_double(d_rd.to_host());
    const std::vector<double> tt = as_double(d_tt.to_host());
    const std::vector<double> absorbed = as_double(d_abs.to_host());
    const std::vector<unsigned long long> events = d_events.to_host();
    const std::vector<unsigned long long> anomalies = d_anomalies.to_host();

    RunResult result;
    result.spec = run;
    result.engine = "cuda";
    result.rung = opt.rung;
    result.profiles = opt.profiles;
    result.precision = "float32";
    result.tally_precision = sizeof(T) == sizeof(double) ? "float64" : "float32";
    result.seed = opt.seed;
    result.threads = 1;
    result.specular = specular_reflectance(make_geometry<double>(run));
    result.device = std::string(prop.name) + " (sm_" + std::to_string(prop.major) +
                    std::to_string(prop.minor) + ")";
    result.a_z = as_double(d_az.to_host());
    result.rd_r = as_double(d_rr.to_host());
    for (uint64_t b = 0; b < nb; ++b) {
        BatchResult br;
        br.first_photon = plan.first_of(b);
        br.photons = plan.count_of(b);
        br.rd = rd[b];
        br.tt = tt[b];
        br.absorbed.assign(absorbed.begin() + static_cast<std::ptrdiff_t>(b * layers),
                           absorbed.begin() + static_cast<std::ptrdiff_t>((b + 1) * layers));
        br.events = events[b];
        br.anomalies = anomalies[b];
        result.batches.push_back(std::move(br));
    }
    result.kernel_seconds = static_cast<double>(ms) / 1000.0;
    result.wall_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return result;
}

} // namespace

bool cuda_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

RunResult run_cuda(const RunSpec &run_in, const EngineOptions &opt) {
    RunSpec run = run_in;
    if (opt.photons != 0) {
        run.photons = opt.photons;
    }
    const Geometry<float> geo = make_geometry<float>(run); // validates the run
    if (opt.batches < 1) {
        throw ConfigError("batches must be >= 1");
    }
    if (static_cast<uint64_t>(opt.batches) > run.photons) {
        throw ConfigError("batches cannot exceed the number of photons");
    }
    if (opt.rung != 0 && opt.rung != 1 && opt.rung != 3 && opt.rung != 5 && opt.rung != 6 &&
        opt.rung != 7 && opt.rung != 8) {
        throw ConfigError("rung must be 0, 1, 3, 5, 6, 7 or 8 (rungs 2 and 4 are not built yet)");
    }
    if (opt.rung == 7 || opt.rung == 8) {
        // One copy of the profiles per warp, with the widest accumulator (double): 48 KB is the
        // shared memory a block may use without opting in.
        const size_t needed = static_cast<size_t>(kBlockSize / 32) *
                              static_cast<size_t>(geo.nz + geo.nr) * sizeof(double);
        if (needed > 48 * 1024) {
            throw ConfigError("rung 7 keeps the depth and radius profiles in shared memory, which "
                              "needs fewer bins than this input has");
        }
    }
    if (!opt.profiles && opt.rung != 6) {
        throw ConfigError("--no-profiles is a diagnostic for rung 6 only");
    }
    if (!cuda_available()) {
        throw std::runtime_error("no usable CUDA device found");
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return opt.gpu_tally == TallyPrecision::Float32 ? run_with_tallies<float>(run, opt, geo, prop)
                                                    : run_with_tallies<double>(run, opt, geo, prop);
}

} // namespace tissuemc
