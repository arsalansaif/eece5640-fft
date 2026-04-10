#include "mkl_bench.hpp"
#include "common/timer.hpp"
#include <mkl_dfti.h>
#include <mkl.h>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <iostream>

// MKL_Complex8 == { float re; float im; } — same layout as std::complex<float>
static_assert(sizeof(MKL_Complex8) == sizeof(cf32),
              "MKL_Complex8 and cf32 size mismatch");

BenchResult run_mkl_bench(
    const cf32*        input,
    size_t             fft_size,
    size_t             batch_size,
    const std::string& dataset,
    const std::string& platform,
    int                num_threads,
    int                num_repeats,
    int                warmup)
{
    // MKL uses its own OpenMP thread pool
    mkl_set_num_threads(num_threads);

    size_t total = fft_size * batch_size;

    // Allocate aligned buffers (MKL_malloc gives 64-byte alignment for AVX-512)
    MKL_Complex8* in  = static_cast<MKL_Complex8*>(
        MKL_malloc(total * sizeof(MKL_Complex8), 64));
    MKL_Complex8* out = static_cast<MKL_Complex8*>(
        MKL_malloc(total * sizeof(MKL_Complex8), 64));
    if (!in || !out) {
        MKL_free(in); MKL_free(out);
        throw std::runtime_error("MKL_malloc failed");
    }

    std::memcpy(in, input, total * sizeof(MKL_Complex8));

    // ── Build DFTI descriptor for batched 1-D single-precision C2C ───────────
    DFTI_DESCRIPTOR_HANDLE desc = nullptr;
    MKL_LONG n = static_cast<MKL_LONG>(fft_size);

    MKL_LONG status = DftiCreateDescriptor(&desc, DFTI_SINGLE, DFTI_COMPLEX, 1, n);
    if (status != DFTI_NO_ERROR)
        throw std::runtime_error("DftiCreateDescriptor failed: " +
                                 std::to_string(status));

    // Batching: transforms are contiguous in memory (distance = fft_size)
    DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS, static_cast<MKL_LONG>(batch_size));
    DftiSetValue(desc, DFTI_INPUT_DISTANCE,  n);
    DftiSetValue(desc, DFTI_OUTPUT_DISTANCE, n);
    DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(desc, DFTI_FORWARD_SCALE, 1.0f);

    status = DftiCommitDescriptor(desc);
    if (status != DFTI_NO_ERROR) {
        DftiFreeDescriptor(&desc);
        throw std::runtime_error("DftiCommitDescriptor failed: " +
                                 std::to_string(status));
    }

    // ── Timing loop ───────────────────────────────────────────────────────────
    std::vector<double> times;
    times.reserve(num_repeats);
    CpuTimer timer;

    for (int i = 0; i < num_repeats + warmup; i++) {
        timer.start();
        status = DftiComputeForward(desc, in, out);
        timer.stop();
        if (status != DFTI_NO_ERROR)
            std::cerr << "[MKL] DftiComputeForward warning: " << status << '\n';
        if (i >= warmup)
            times.push_back(timer.elapsed_ms());
    }

    DftiFreeDescriptor(&desc);
    MKL_free(in);
    MKL_free(out);

    double med_ms = vec_median(times);

    BenchResult r{};
    r.impl          = "mkl";
    r.dataset       = dataset;
    r.platform      = platform;
    r.fft_size      = fft_size;
    r.batch_size    = batch_size;
    r.num_threads   = num_threads;
    r.h2d_ms        = 0.0;
    r.kernel_ms     = med_ms;
    r.d2h_ms        = 0.0;
    r.total_wall_ms = med_ms;
    r.gflops        = fft_gflops(fft_size, batch_size, med_ms);
    r.bandwidth_gbs = fft_bandwidth_gbs(fft_size, batch_size, med_ms);
    return r;
}
