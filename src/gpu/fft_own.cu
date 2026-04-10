#include "fft_own.cuh"
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>

// ── Error helper ───────────────────────────────────────────────────────────────
#define CUDA_CHECK(expr) do {                                              \
    cudaError_t _e = (expr);                                               \
    if (_e != cudaSuccess)                                                 \
        throw std::runtime_error(std::string("CUDA: ") +                   \
                                 cudaGetErrorString(_e) +                  \
                                 " (" #expr ")");                          \
} while(0)

static constexpr float MY_PI = 3.14159265358979323846f;

// ── Bit-reversal helper (device) ──────────────────────────────────────────────
__device__ __forceinline__ int bit_rev(int x, int log2_n) {
    int r = 0;
    for (int i = 0; i < log2_n; i++, x >>= 1)
        r = (r << 1) | (x & 1);
    return r;
}

// ── Kernel 1: shared-memory FFT (N ≤ 2048) ────────────────────────────────────
// Launch config:  grid = (batch_size, 1), block = (N/2, 1)
// Dynamic smem  = N * sizeof(cuFloatComplex)  bytes
//
// Each thread handles exactly one butterfly element-pair per stage.
// All N elements reside in shared memory throughout; zero mid-computation
// global traffic.
__global__ void fft_shared_kernel(cuFloatComplex* __restrict__ data,
                                   int N, int log2_N)
{
    extern __shared__ cuFloatComplex smem[];

    const int tid    = threadIdx.x;   // 0 .. N/2 - 1
    const int half_N = N >> 1;
    cuFloatComplex* g = data + (size_t)blockIdx.x * N;

    // Load input into bit-reversed positions so DIT stages need no permutation.
    smem[bit_rev(tid,          log2_N)] = g[tid];
    smem[bit_rev(tid + half_N, log2_N)] = g[tid + half_N];
    __syncthreads();

    // log2_N butterfly stages: group size doubles each stage (2, 4, 8, …, N).
    for (int s = 1; s <= log2_N; s++) {
        const int len  = 1 << s;
        const int half = len >> 1;

        // This thread's butterfly pair within the current stage.
        const int j_w   = tid % half;   // position inside the group
        const int group = tid / half;   // which group
        const int idx0  = group * len + j_w;
        const int idx1  = idx0 + half;

        const float angle = -2.0f * MY_PI * (float)j_w / (float)len;
        const cuFloatComplex w = make_cuFloatComplex(cosf(angle), sinf(angle));

        cuFloatComplex u = smem[idx0];
        cuFloatComplex v = cuCmulf(smem[idx1], w);
        smem[idx0] = cuCaddf(u, v);
        smem[idx1] = cuCsubf(u, v);
        __syncthreads();
    }

    // Write result back to global memory.
    g[tid]          = smem[tid];
    g[tid + half_N] = smem[tid + half_N];
}

// ── Kernel 2a: global-memory bit-reversal (N > 2048) ─────────────────────────
// Each thread swaps element i with bit_rev(i) when i < bit_rev(i),
// so every pair is swapped exactly once.
__global__ void bit_reverse_kernel(cuFloatComplex* data,
                                    int N, int log2_N, int batch_size)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= batch_size * N) return;

    int batch = tid / N;
    int i     = tid % N;
    int j     = bit_rev(i, log2_N);

    if (i < j) {
        cuFloatComplex* base = data + (size_t)batch * N;
        cuFloatComplex tmp = base[i];
        base[i] = base[j];
        base[j] = tmp;
    }
}

// ── Kernel 2b: global-memory butterfly stage (N > 2048) ──────────────────────
// Called once per DIT stage s = 1 .. log2_N (after bit-reversal).
// Each thread handles one butterfly pair.
__global__ void fft_butterfly_stage(cuFloatComplex* data,
                                     int N, int batch_size, int s)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int half_N = N >> 1;
    if (tid >= batch_size * half_N) return;

    int batch = tid / half_N;
    int pair  = tid % half_N;

    int len   = 1 << s;
    int half  = len >> 1;
    int j_w   = pair % half;    // position within butterfly group
    int group = pair / half;    // which group

    int idx0 = (int)((size_t)batch * N) + group * len + j_w;
    int idx1 = idx0 + half;

    float angle = -2.0f * MY_PI * (float)j_w / (float)len;
    cuFloatComplex w = make_cuFloatComplex(cosf(angle), sinf(angle));

    cuFloatComplex u = data[idx0];
    cuFloatComplex v = cuCmulf(data[idx1], w);
    data[idx0] = cuCaddf(u, v);
    data[idx1] = cuCsubf(u, v);
}

// ── Compute-only wrapper (used by correctness verification) ───────────────────
void compute_own_gpu(const cf32* h_input, cf32* h_output, size_t N, size_t batch)
{
    int log2_N = 0;
    for (size_t t = N; t > 1; t >>= 1) log2_N++;

    size_t bytes = N * batch * sizeof(cuFloatComplex);
    cuFloatComplex* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_input, bytes, cudaMemcpyHostToDevice));

    const bool use_shared = (N <= 2048);
    if (use_shared) {
        fft_shared_kernel<<<(int)batch, (int)(N/2), N * sizeof(cuFloatComplex)>>>(
            d_data, (int)N, log2_N);
    } else {
        const int BLOCK      = 256;
        int total_elem  = (int)(batch * N);
        int total_pairs = (int)(batch * N / 2);
        int grid_elem   = (total_elem  + BLOCK - 1) / BLOCK;
        int grid_pairs  = (total_pairs + BLOCK - 1) / BLOCK;
        bit_reverse_kernel<<<grid_elem, BLOCK>>>(d_data, (int)N, log2_N, (int)batch);
        for (int s = 1; s <= log2_N; s++)
            fft_butterfly_stage<<<grid_pairs, BLOCK>>>(d_data, (int)N, (int)batch, s);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_data, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
}

// ── GPU event timer ────────────────────────────────────────────────────────────
struct OwnGpuTimer {
    cudaEvent_t start, stop;
    OwnGpuTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~OwnGpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void record_start(cudaStream_t s) { cudaEventRecord(start, s); }
    void record_stop (cudaStream_t s) { cudaEventRecord(stop,  s); }
    float elapsed_ms() const {
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

static double vec_median_own(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? (v[n/2-1] + v[n/2]) / 2.0 : v[n/2];
}

// ── Main benchmark function ────────────────────────────────────────────────────
BenchResult run_own_gpu_bench(
    const cf32*        h_input,
    size_t             fft_size,
    size_t             batch_size,
    const std::string& dataset,
    const std::string& platform,
    int                num_repeats,
    int                warmup)
{
    if (fft_size == 0 || (fft_size & (fft_size - 1)))
        throw std::runtime_error("own_gpu: fft_size must be a power of 2");

    const int N      = (int)fft_size;
    int log2_N = 0;
    for (int t = N; t > 1; t >>= 1) log2_N++;

    size_t bytes = fft_size * batch_size * sizeof(cuFloatComplex);

    cuFloatComplex* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    std::vector<cf32> h_output(fft_size * batch_size);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    OwnGpuTimer t_h2d, t_ker, t_d2h;
    std::vector<double> h2d_v, ker_v, d2h_v;
    h2d_v.reserve(num_repeats);
    ker_v.reserve(num_repeats);
    d2h_v.reserve(num_repeats);

    // N ≤ 2048: shared-memory kernel (one block per FFT, N/2 threads/block)
    // N > 2048: global-memory multi-pass (bit-reversal + log2_N stage kernels)
    const bool use_shared = (N <= 2048);
    const int  BLOCK      = 256;

    for (int i = 0; i < num_repeats + warmup; i++) {
        // H2D
        t_h2d.record_start(stream);
        CUDA_CHECK(cudaMemcpyAsync(d_data, h_input, bytes,
                                   cudaMemcpyHostToDevice, stream));
        t_h2d.record_stop(stream);

        // Kernel
        t_ker.record_start(stream);
        if (use_shared) {
            int    threads    = N / 2;
            size_t smem_bytes = (size_t)N * sizeof(cuFloatComplex);
            fft_shared_kernel<<<(int)batch_size, threads, smem_bytes, stream>>>(
                d_data, N, log2_N);
        } else {
            int total_elem  = (int)(batch_size) * N;
            int total_pairs = (int)(batch_size) * (N / 2);
            int grid_elem   = (total_elem  + BLOCK - 1) / BLOCK;
            int grid_pairs  = (total_pairs + BLOCK - 1) / BLOCK;

            bit_reverse_kernel<<<grid_elem, BLOCK, 0, stream>>>(
                d_data, N, log2_N, (int)batch_size);

            for (int s = 1; s <= log2_N; s++) {
                fft_butterfly_stage<<<grid_pairs, BLOCK, 0, stream>>>(
                    d_data, N, (int)batch_size, s);
            }
        }
        t_ker.record_stop(stream);

        // D2H
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

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_data));

    double med_h2d = vec_median_own(h2d_v);
    double med_ker = vec_median_own(ker_v);
    double med_d2h = vec_median_own(d2h_v);

    BenchResult r{};
    r.impl          = "own_gpu";
    r.dataset       = dataset;
    r.platform      = platform;
    r.fft_size      = fft_size;
    r.batch_size    = batch_size;
    r.num_threads   = 0;
    r.h2d_ms        = med_h2d;
    r.kernel_ms     = med_ker;
    r.d2h_ms        = med_d2h;
    r.total_wall_ms = med_h2d + med_ker + med_d2h;
    r.gflops        = fft_gflops(fft_size, batch_size, med_ker);
    r.bandwidth_gbs = fft_bandwidth_gbs(fft_size, batch_size, med_ker);
    return r;
}
