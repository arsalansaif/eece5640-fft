#pragma once
#include <complex>
#include <vector>
#include <cstddef>
#include <cmath>

using cf32     = std::complex<float>;
using DataBatch = std::vector<cf32>;

// 5 N log2(N) flops per complex DFT of length N
inline double fft_gflops(size_t fft_size, size_t batch_size, double time_ms) {
    double ops = 5.0 * static_cast<double>(batch_size) *
                 static_cast<double>(fft_size) *
                 std::log2(static_cast<double>(fft_size));
    return ops / (time_ms * 1e6);   // ops / (ms * 1e-3 * 1e9)
}

// Read input + write output, each batch_size*fft_size complex<float> (8 bytes)
inline double fft_bandwidth_gbs(size_t fft_size, size_t batch_size, double time_ms) {
    double bytes = 2.0 * static_cast<double>(batch_size) *
                   static_cast<double>(fft_size) * sizeof(cf32);
    return bytes / (time_ms * 1e6);
}
