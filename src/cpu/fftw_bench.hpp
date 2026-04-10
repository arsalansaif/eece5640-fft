#pragma once
#include "common/types.hpp"
#include "common/result.hpp"
#include <string>

// Run batched 1-D forward FFT with FFTW3 (single precision, OpenMP threads).
// Repeats num_repeats times after warmup; returns median timing.
BenchResult run_fftw_bench(
    const cf32*        input,
    size_t             fft_size,
    size_t             batch_size,
    const std::string& dataset,
    const std::string& platform,
    int                num_threads = 16,
    int                num_repeats = 20,
    int                warmup      = 3
);
