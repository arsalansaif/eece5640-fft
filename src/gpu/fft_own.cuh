#pragma once
#include "common/types.hpp"
#include "common/result.hpp"
#include <string>

// Custom Cooley-Tukey Radix-2 DIT FFT — GPU implementation.
//
// Kernel strategy (chosen at runtime based on fft_size):
//   N ≤ 2048 — shared-memory kernel: one CUDA block per FFT, N/2 threads per
//              block, all N elements live in shared memory for the entire
//              computation (zero global-memory traffic during butterflies).
//   N > 2048 — global-memory multi-pass: a separate kernel launch for each of
//              the log2(N) DIT butterfly stages, preceded by one bit-reversal
//              pass.  Less memory-access efficient but handles arbitrary sizes.
//
// Timing follows the same H2D / kernel / D2H breakdown as cufft_bench.cu.
BenchResult run_own_gpu_bench(
    const cf32*        input,
    size_t             fft_size,
    size_t             batch_size,
    const std::string& dataset,
    const std::string& platform,
    int                num_repeats = 20,
    int                warmup      = 3
);

// Compute-only wrapper (no timing). output must be pre-allocated N*batch elements.
void compute_own_gpu(const cf32* h_input, cf32* h_output, size_t N, size_t batch);
