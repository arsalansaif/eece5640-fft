#include "fftw_bench.hpp"
#include "common/timer.hpp"
#include <fftw3.h>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <iostream>

BenchResult run_fftw_bench(
    const cf32*        input,
    size_t             fft_size,
    size_t             batch_size,
    const std::string& dataset,
    const std::string& platform,
    int                num_threads,
    int                num_repeats,
    int                warmup)
{
    // One-time thread initialisation (safe to call repeatedly; idempotent).
#ifdef HAVE_FFTW3_THREADS
    static bool init_done = false;
    if (!init_done) {
        if (fftwf_init_threads() == 0)
            throw std::runtime_error("fftwf_init_threads() failed");
        init_done = true;
    }
    fftwf_plan_with_nthreads(num_threads);
#else
    num_threads = 1;
    std::cerr << "[FFTW3] Warning: compiled without thread library, running single-threaded\n";
#endif

    size_t total = fft_size * batch_size;

    // FFTW-aligned allocation for best SIMD performance
    fftwf_complex* in  = fftwf_alloc_complex(total);
    fftwf_complex* out = fftwf_alloc_complex(total);
    if (!in || !out) {
        fftwf_free(in); fftwf_free(out);
        throw std::runtime_error("fftwf_alloc_complex failed");
    }

    // Copy input once; FFTW out-of-place does not modify 'in'
    static_assert(sizeof(cf32) == sizeof(fftwf_complex), "float complex size mismatch");
    std::memcpy(in, input, total * sizeof(fftwf_complex));

    // Build plan — FFTW_MEASURE auto-tunes the plan (may take a few seconds
    // for large transforms; use FFTW_ESTIMATE if startup time is critical)
    int n_arr[1] = { static_cast<int>(fft_size) };
    fftwf_plan plan = fftwf_plan_many_dft(
        /*rank*/   1, n_arr, static_cast<int>(batch_size),
        in,  nullptr, 1, static_cast<int>(fft_size),
        out, nullptr, 1, static_cast<int>(fft_size),
        FFTW_FORWARD, FFTW_MEASURE
    );
    if (!plan) {
        fftwf_free(in); fftwf_free(out);
        throw std::runtime_error("fftwf_plan_many_dft failed");
    }

    // fftwf_plan_many_dft with FFTW_MEASURE may overwrite 'in'; restore
    std::memcpy(in, input, total * sizeof(fftwf_complex));

    std::vector<double> times;
    times.reserve(num_repeats);
    CpuTimer timer;

    for (int i = 0; i < num_repeats + warmup; i++) {
        timer.start();
        fftwf_execute(plan);
        timer.stop();
        if (i >= warmup)
            times.push_back(timer.elapsed_ms());
    }

    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);

    double med_ms = vec_median(times);

    BenchResult r{};
    r.impl           = "fftw3";
    r.dataset        = dataset;
    r.platform       = platform;
    r.fft_size       = fft_size;
    r.batch_size     = batch_size;
    r.num_threads    = num_threads;
    r.h2d_ms         = 0.0;
    r.kernel_ms      = med_ms;
    r.d2h_ms         = 0.0;
    r.total_wall_ms  = med_ms;
    r.gflops         = fft_gflops(fft_size, batch_size, med_ms);
    r.bandwidth_gbs  = fft_bandwidth_gbs(fft_size, batch_size, med_ms);
    return r;
}
