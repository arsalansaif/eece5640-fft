#pragma once
#include "common/types.hpp"
#include "common/result.hpp"
#include <string>

// Custom Cooley-Tukey Radix-2 DIT FFT — reference CPU implementation.
// Parallelised over the batch dimension with OpenMP.
// fft_size must be a power of 2.
BenchResult run_own_cpu_bench(
    const cf32*        input,
    size_t             fft_size,
    size_t             batch_size,
    const std::string& dataset,
    const std::string& platform,
    int                num_threads = 16,
    int                num_repeats = 20,
    int                warmup      = 3
);

// Compute-only wrapper (no timing). output must be pre-allocated N*batch elements.
void compute_own_cpu(const cf32* input, cf32* output, size_t N, size_t batch);
