#pragma once
#include "common/types.hpp"
#include "common/result.hpp"
#include <string>

// Run batched 1-D forward FFT with cuFFT (CUFFT_C2C, in-place on device).
// Times H2D transfer, kernel, and D2H transfer separately via cudaEvent_t.
// Returns median timing over measured repeats.
BenchResult run_cufft_bench(
    const cf32*        h_input,
    size_t             fft_size,
    size_t             batch_size,
    const std::string& dataset,
    const std::string& platform,
    int                num_repeats = 20,
    int                warmup      = 3
);

// Compute-only wrapper (no timing). output must be pre-allocated N*batch elements.
void compute_cufft(const cf32* h_input, cf32* h_output, size_t N, size_t batch);
