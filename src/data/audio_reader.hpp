#pragma once
#include "common/types.hpp"
#include <string>

// Read a single WAV file (PCM-16 or IEEE-float-32, mono or stereo).
// Channel 0 is taken; samples are normalised to [-1, 1] and stored as
// complex<float> with imaginary part = 0 (real baseband).
DataBatch read_wav_file(const std::string& path, size_t max_samples = 0);

// Walk dir_path recursively, concatenate all WAV files until max_samples.
DataBatch read_wav_dir(const std::string& dir_path, size_t max_samples = 0);

// Alias for segment_to_frames — shared logic lives in sigmf_reader
#include "sigmf_reader.hpp"
// Use segment_to_frames() from sigmf_reader for audio segmentation too.
