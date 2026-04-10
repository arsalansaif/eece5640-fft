#pragma once
#include "common/types.hpp"
#include <random>
#include <cstddef>

// Generate fft_size * batch_size random complex<float> samples.
inline DataBatch generate_synthetic(size_t fft_size, size_t batch_size,
                                    unsigned seed = 42) {
    DataBatch data(fft_size * batch_size);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& s : data)
        s = cf32(dist(rng), dist(rng));
    return data;
}

// Compute how many transforms fit in budget_bytes (input only, in-place).
inline size_t batch_for_budget(size_t fft_size, size_t budget_bytes) {
    size_t bytes_per = fft_size * sizeof(cf32);
    return std::max<size_t>(1, budget_bytes / bytes_per);
}
