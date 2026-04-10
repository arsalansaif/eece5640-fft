#pragma once
#include "common/types.hpp"
#include <string>

// Read raw interleaved complex float32 samples from a .sigmf-data file.
// max_samples == 0 means read the entire file.
DataBatch read_sigmf_data(const std::string& path, size_t max_samples = 0);

// Concatenate every .sigmf-data file found under dir_path (recursive).
DataBatch read_sigmf_dir(const std::string& dir_path, size_t max_samples = 0);

// Trim samples to a whole number of fft_size-point frames.
DataBatch segment_to_frames(const DataBatch& samples, size_t fft_size);
