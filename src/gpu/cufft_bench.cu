#include "cufft_bench.cuh"
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <iostream>

// ── Error helpers ──────────────────────────────────────────────────────────────
#define CUDA_CHECK(expr) do {                                              \
    cudaError_t _e = (expr);                                               \
    if (_e != cudaSuccess)                                                 \
        throw std::runtime_error(std::string("CUDA: ") +                   \
                                 cudaGetErrorString(_e) +                  \
                                 " (" #expr ")");                          \
} while(0)

#define CUFFT_CHECK(expr) do {                                             \
    cufftResult _r = (expr);                                               \
    if (_r != CUFFT_SUCCESS)                                               \
        throw std::runtime_error("cuFFT error " +                          \
                                 std::to_string(static_cast<int>(_r)) +   \
                                 " (" #expr ")");                          \
} while(0)

// ── Timing helpers ─────────────────────────────────────────────────────────────
struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void record_start(cudaStream_t s) { cudaEventRecord(start, s); }
    void record_stop (cudaStream_t s) { cudaEventRecord(stop,  s); }
    float elapsed_ms() const {
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

static double vec_median_gpu(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? (v[n/2-1] + v[n/2]) / 2.0 : v[n/2];
}

// ── Compute-only wrapper (used by correctness verification) ───────────────────
void compute_cufft(const cf32* h_input, cf32* h_output, size_t N, size_t batch)
{
    size_t bytes = N * batch * sizeof(cf32);
    cufftComplex* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_input, bytes, cudaMemcpyHostToDevice));

    cufftHandle plan;
    int n_arr[1] = { static_cast<int>(N) };
    CUFFT_CHECK(cufftPlanMany(&plan, 1, n_arr,
                              nullptr, 1, static_cast<int>(N),
                              nullptr, 1, static_cast<int>(N),
                              CUFFT_C2C, static_cast<int>(batch)));
    CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUFFT_CHECK(cufftDestroy(plan));

    CUDA_CHECK(cudaMemcpy(h_output, d_data, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
}

// ── Main benchmark ─────────────────────────────────────────────────────────────
BenchResult run_cufft_bench(
    const cf32*        h_input,
    size_t             fft_size,
    size_t             batch_size,
    const std::string& dataset,
    const std::string& platform,
    int                num_repeats,
    int                warmup)
{
    // Halve batch_size until it fits in GPU memory (workspace may push past limit)
    while (batch_size > 1) {
        size_t probe_bytes = fft_size * batch_size * sizeof(cf32);
        cufftComplex* probe = nullptr;
        if (cudaMalloc(&probe, probe_bytes) == cudaSuccess) {
            cudaFree(probe);
            break;
        }
        std::cerr << "  [cuFFT] OOM at batch=" << batch_size
                  << " (" << probe_bytes/1024/1024 << " MB), halving\n";
        batch_size /= 2;
    }

    size_t total = fft_size * batch_size;
    size_t bytes = total * sizeof(cf32);

    // Single device buffer — in-place transform halves peak memory vs. out-of-place
    cufftComplex* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    // Host output buffer for D2H timing (separate from h_input)
    std::vector<cf32> h_output(total);

    // cuFFT plan: batched 1-D C2C
    cufftHandle plan;
    int n_arr[1] = { static_cast<int>(fft_size) };
    // Retry plan creation with smaller batch if workspace causes OOM
    cufftResult plan_result;
    while (batch_size > 1) {
        plan_result = cufftPlanMany(
            &plan, 1, n_arr,
            nullptr, 1, static_cast<int>(fft_size),
            nullptr, 1, static_cast<int>(fft_size),
            CUFFT_C2C, static_cast<int>(batch_size)
        );
        if (plan_result == CUFFT_SUCCESS) break;
        std::cerr << "  [cuFFT] Plan OOM at batch=" << batch_size << ", halving\n";
        batch_size /= 2;
        total = fft_size * batch_size;
        bytes  = total * sizeof(cf32);
        cudaFree(d_data);
        CUDA_CHECK(cudaMalloc(&d_data, bytes));
        h_output.resize(total);
    }
    if (plan_result != CUFFT_SUCCESS)
        throw std::runtime_error("cuFFT plan failed after batch reduction");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUFFT_CHECK(cufftSetStream(plan, stream));

    GpuTimer t_h2d, t_ker, t_d2h;

    std::vector<double> h2d_v, ker_v, d2h_v;
    h2d_v.reserve(num_repeats);
    ker_v.reserve(num_repeats);
    d2h_v.reserve(num_repeats);

    for (int i = 0; i < num_repeats + warmup; i++) {
        // H2D ─────────────────────────────────────────────────────────────────
        t_h2d.record_start(stream);
        CUDA_CHECK(cudaMemcpyAsync(d_data, h_input, bytes,
                                   cudaMemcpyHostToDevice, stream));
        t_h2d.record_stop(stream);

        // Kernel (in-place) ───────────────────────────────────────────────────
        t_ker.record_start(stream);
        CUFFT_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
        t_ker.record_stop(stream);

        // D2H ─────────────────────────────────────────────────────────────────
        t_d2h.record_start(stream);
        CUDA_CHECK(cudaMemcpyAsync(h_output.data(), d_data, bytes,
                                   cudaMemcpyDeviceToHost, stream));
        t_d2h.record_stop(stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (i >= warmup) {
            h2d_v.push_back(t_h2d.elapsed_ms());
            ker_v.push_back(t_ker.elapsed_ms());
            d2h_v.push_back(t_d2h.elapsed_ms());
        }
    }

    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_data));

    double med_h2d = vec_median_gpu(h2d_v);
    double med_ker = vec_median_gpu(ker_v);
    double med_d2h = vec_median_gpu(d2h_v);

    BenchResult r{};
    r.impl           = "cufft";
    r.dataset        = dataset;
    r.platform       = platform;
    r.fft_size       = fft_size;
    r.batch_size     = batch_size;
    r.num_threads    = 0;
    r.h2d_ms         = med_h2d;
    r.kernel_ms      = med_ker;
    r.d2h_ms         = med_d2h;
    r.total_wall_ms  = med_h2d + med_ker + med_d2h;
    r.gflops         = fft_gflops(fft_size, batch_size, med_ker);
    r.bandwidth_gbs  = fft_bandwidth_gbs(fft_size, batch_size, med_ker);
    return r;
}
