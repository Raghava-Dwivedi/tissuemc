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

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        const cudaError_t tmc_err = (call);                                                     \
        if (tmc_err != cudaSuccess) {                                                           \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" +           \
                                     std::to_string(__LINE__) + ": " +                          \
                                     cudaGetErrorString(tmc_err));                              \
        }                                                                                       \
    } while (0)

constexpr int kBlockSize = 128;
constexpr uint64_t kPhotonsPerLaunch = uint64_t(1) << 26;

// R0: every thread simulates one photon from launch to death and adds its results straight into
// the global tallies.
template <class T>
__global__ void r0_kernel(Geometry<float> geo, BatchPlan plan, uint64_t seed, uint64_t first,
                          uint64_t count, GpuTallyArrays<T> arrays) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const uint64_t id = first + i;
    run_photon_r0(geo, seed, id, static_cast<int>(plan.batch_of(id)), arrays);
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
    const GpuTallyArrays<T> arrays{d_rd.get(),     d_tt.get(),     d_abs.get(),       d_az.get(),
                                   d_rr.get(),     d_events.get(), d_anomalies.get()};

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (uint64_t first = 0; first < run.photons; first += kPhotonsPerLaunch) {
        const uint64_t count = std::min(kPhotonsPerLaunch, run.photons - first);
        const unsigned blocks = static_cast<unsigned>((count + kBlockSize - 1) / kBlockSize);
        r0_kernel<T><<<blocks, kBlockSize>>>(geo, plan, opt.seed, first, count, arrays);
        CUDA_CHECK(cudaGetLastError());
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
    if (!cuda_available()) {
        throw std::runtime_error("no usable CUDA device found");
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return opt.gpu_tally == TallyPrecision::Float32
               ? run_with_tallies<float>(run, opt, geo, prop)
               : run_with_tallies<double>(run, opt, geo, prop);
}

} // namespace tissuemc
