#pragma once
#include "common/types.hpp"
#include "common/result.hpp"
#include <string>

// Run batched 1-D forward FFT using Intel MKL DFTI (single precision).
// Threading is controlled by MKL's internal OpenMP pool (mkl_set_num_threads).
BenchResult run_mkl_bench(
    const cf32*        input,
    size_t             fft_size,
    size_t             batch_size,
    const std::string& dataset,
    const std::string& platform,
    int                num_threads = 16,
    int                num_repeats = 20,
    int                warmup      = 3
);
