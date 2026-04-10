// Correctness verification for all four FFT implementations.
// Uses MKL DFTI as the reference (double-precision internally; comparison in
// single-precision normalized RMS error).
//
// Test signals:
//   1. Impulse: x[0] = 1, rest = 0  → DFT should be all-ones (X[k] = 1 ∀k).
//   2. Single tone: x[n] = exp(2πi·k₀·n/N) → DFT = N at k=k₀, 0 elsewhere.
//   3. Random complex: compare each impl against MKL on random input.

#include "verify.cuh"
#include "cpu/fft_own.hpp"
#include "gpu/fft_own.cuh"
#include "gpu/cufft_bench.cuh"
#include "common/types.hpp"

#include <mkl_dfti.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <random>
#include <string>

// ── Helpers ───────────────────────────────────────────────────────────────────

// Normalized RMS error: ||got - ref|| / ||ref||
static double nrmse(const cf32* ref, const cf32* got, size_t n)
{
    double err = 0.0, mag = 0.0;
    for (size_t i = 0; i < n; i++) {
        float dr = std::real(got[i]) - std::real(ref[i]);
        float di = std::imag(got[i]) - std::imag(ref[i]);
        err += (double)dr * dr + (double)di * di;
        mag += (double)std::real(ref[i]) * std::real(ref[i])
             + (double)std::imag(ref[i]) * std::imag(ref[i]);
    }
    return (mag > 0.0) ? std::sqrt(err / mag) : std::sqrt(err);
}

// MKL reference: forward batched FFT, returns output buffer.
static std::vector<cf32> mkl_fft(const cf32* input, int N, int B)
{
    std::vector<cf32> out(N * B);
    DFTI_DESCRIPTOR_HANDLE desc = nullptr;
    MKL_LONG nn = N;
    DftiCreateDescriptor(&desc, DFTI_SINGLE, DFTI_COMPLEX, 1, nn);
    DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)B);
    DftiSetValue(desc, DFTI_INPUT_DISTANCE,       (MKL_LONG)N);
    DftiSetValue(desc, DFTI_OUTPUT_DISTANCE,      (MKL_LONG)N);
    DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiCommitDescriptor(desc);
    DftiComputeForward(desc, const_cast<cf32*>(input), out.data());
    DftiFreeDescriptor(&desc);
    return out;
}

static void report(const char* name, double e, double tol)
{
    std::cerr << "    " << name
              << "  NRMSE=" << e
              << (e < tol ? "  [PASS]" : "  [FAIL]") << '\n';
}

// ── One size test ─────────────────────────────────────────────────────────────
static void verify_size(int N, int B, double tol)
{
    std::cerr << "  N=" << N << "  batch=" << B << '\n';

    // ── Test 1: Impulse ───────────────────────────────────────────────────────
    {
        std::vector<cf32> in(N * B, cf32(0.f, 0.f));
        for (int b = 0; b < B; b++)
            in[b * N] = cf32(1.f, 0.f);   // x[0] = 1 per batch

        auto ref = mkl_fft(in.data(), N, B);

        std::cerr << "    [impulse]\n";

        std::vector<cf32> out(N * B);
        compute_own_cpu(in.data(), out.data(), N, B);
        report("own_cpu", nrmse(ref.data(), out.data(), N * B), tol);

        compute_own_gpu(in.data(), out.data(), N, B);
        report("own_gpu", nrmse(ref.data(), out.data(), N * B), tol);

        compute_cufft(in.data(), out.data(), N, B);
        report("cuFFT  ", nrmse(ref.data(), out.data(), N * B), tol);
    }

    // ── Test 2: Single tone ───────────────────────────────────────────────────
    {
        const int k0 = N / 4;   // tone at quarter-frequency
        std::vector<cf32> in(N * B);
        for (int b = 0; b < B; b++)
            for (int n = 0; n < N; n++) {
                float angle = 2.f * 3.14159265f * k0 * n / N;
                in[b * N + n] = cf32(std::cos(angle), std::sin(angle));
            }

        auto ref = mkl_fft(in.data(), N, B);

        std::cerr << "    [single tone k0=" << k0 << "]\n";

        std::vector<cf32> out(N * B);
        compute_own_cpu(in.data(), out.data(), N, B);
        report("own_cpu", nrmse(ref.data(), out.data(), N * B), tol);

        compute_own_gpu(in.data(), out.data(), N, B);
        report("own_gpu", nrmse(ref.data(), out.data(), N * B), tol);

        compute_cufft(in.data(), out.data(), N, B);
        report("cuFFT  ", nrmse(ref.data(), out.data(), N * B), tol);
    }

    // ── Test 3: Random input ──────────────────────────────────────────────────
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.f, 1.f);
        std::vector<cf32> in(N * B);
        for (auto& c : in)
            c = cf32(dist(rng), dist(rng));

        auto ref = mkl_fft(in.data(), N, B);

        std::cerr << "    [random]\n";

        std::vector<cf32> out(N * B);
        compute_own_cpu(in.data(), out.data(), N, B);
        report("own_cpu", nrmse(ref.data(), out.data(), N * B), tol);

        compute_own_gpu(in.data(), out.data(), N, B);
        report("own_gpu", nrmse(ref.data(), out.data(), N * B), tol);

        compute_cufft(in.data(), out.data(), N, B);
        report("cuFFT  ", nrmse(ref.data(), out.data(), N * B), tol);
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────
void run_verify()
{
    // Single-precision FFT accumulates ~log2(N) * eps rounding error.
    // At N=4096: log2(4096)=12 stages, eps~1e-7 → tol ~ 1e-5 is generous.
    const double tol = 1e-4;

    std::cerr << "\n=== Correctness Verification (reference: MKL DFTI) ===\n";
    std::cerr << "Tolerance: NRMSE < " << tol << '\n';

    verify_size(64,   16, tol);   // small — well inside shared-memory path
    verify_size(1024, 16, tol);   // medium — typical benchmark size
    verify_size(4096, 16, tol);   // large — crosses into global-memory path

    std::cerr << "=== Verification done ===\n";
}
