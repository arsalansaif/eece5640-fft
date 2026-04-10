#include "fft_own.hpp"
#include "common/timer.hpp"
#include "common/types.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <omp.h>

// ── Iterative Cooley-Tukey Radix-2 DIT FFT (in-place) ─────────────────────────
// Input x must have length N (power of 2).
// Forward DFT: X[k] = sum_{n=0}^{N-1} x[n] * exp(-2πi·nk/N)
static void fft_r2_dit(cf32* x, int N)
{
    // ── Bit-reversal permutation ──────────────────────────────────────────────
    // Iterate i from 1 to N-1; maintain j as the bit-reverse of i.
    // Swap x[i] and x[j] only when i < j to avoid double-swapping.
    for (int i = 1, j = 0; i < N; i++) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(x[i], x[j]);
    }

    // ── Butterfly stages ──────────────────────────────────────────────────────
    // len is the current butterfly group size: 2, 4, 8, ..., N.
    // wlen is the primitive len-th root of unity: exp(-2πi/len).
    for (int len = 2; len <= N; len <<= 1) {
        float angle = -2.0f * float(M_PI) / float(len);
        cf32  wlen(std::cos(angle), std::sin(angle));

        for (int i = 0; i < N; i += len) {
            cf32 w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; j++) {
                cf32 u = x[i + j];
                cf32 v = x[i + j + len / 2] * w;
                x[i + j]           = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// ── Compute-only wrapper ──────────────────────────────────────────────────────
void compute_own_cpu(const cf32* input, cf32* output, size_t N, size_t batch)
{
    std::copy(input, input + N * batch, output);
    for (size_t b = 0; b < batch; b++)
        fft_r2_dit(output + b * N, (int)N);
}

// ── Benchmark harness ─────────────────────────────────────────────────────────
BenchResult run_own_cpu_bench(
    const cf32*        input,
    size_t             fft_size,
    size_t             batch_size,
    const std::string& dataset,
    const std::string& platform,
    int                num_threads,
    int                num_repeats,
    int                warmup)
{
    if (fft_size == 0 || (fft_size & (fft_size - 1)))
        throw std::runtime_error("own_cpu: fft_size must be a power of 2");

    omp_set_num_threads(num_threads);

    // Working buffer — refilled from input before each timed rep so we measure
    // only FFT computation, not side effects of the previous run's in-place result.
    std::vector<cf32> buf(fft_size * batch_size);

    std::vector<double> times;
    times.reserve(num_repeats);

    CpuTimer timer;

    for (int rep = 0; rep < num_repeats + warmup; rep++) {
        std::copy(input, input + fft_size * batch_size, buf.begin());

        timer.start();
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < (int)batch_size; b++)
            fft_r2_dit(buf.data() + (size_t)b * fft_size, (int)fft_size);
        timer.stop();

        if (rep >= warmup)
            times.push_back(timer.elapsed_ms());
    }

    double med = vec_median(times);

    BenchResult r{};
    r.impl          = "own_cpu";
    r.dataset       = dataset;
    r.platform      = platform;
    r.fft_size      = fft_size;
    r.batch_size    = batch_size;
    r.num_threads   = num_threads;
    r.h2d_ms        = 0.0;
    r.kernel_ms     = med;
    r.d2h_ms        = 0.0;
    r.total_wall_ms = med;
    r.gflops        = fft_gflops(fft_size, batch_size, med);
    r.bandwidth_gbs = fft_bandwidth_gbs(fft_size, batch_size, med);
    return r;
}
