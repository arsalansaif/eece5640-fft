#include "sigmf_reader.hpp"
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <iostream>
#include <complex>
#include <cstring>

namespace fs = std::filesystem;

// ORACLE dataset note: despite the .sigmf-meta claiming cf32, the binary files
// are actually interleaved complex128 (two 64-bit doubles per sample).
// We read as complex<double> and downcast to complex<float>.
DataBatch read_sigmf_data(const std::string& path, size_t max_samples) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("Cannot open SigMF data file: " + path);

    auto file_bytes = static_cast<size_t>(f.tellg());
    f.seekg(0);

    using cf64 = std::complex<double>;

    // Determine sample count based on complex128 (16 bytes per sample)
    size_t n = file_bytes / sizeof(cf64);
    if (max_samples > 0 && n > max_samples)
        n = max_samples;

    // Read as complex128
    std::vector<cf64> raw(n);
    f.read(reinterpret_cast<char*>(raw.data()), n * sizeof(cf64));
    if (!f)
        throw std::runtime_error("Read error: " + path);

    // Downcast to complex64 for FFT processing
    DataBatch data(n);
    for (size_t i = 0; i < n; i++)
        data[i] = cf32(static_cast<float>(raw[i].real()),
                       static_cast<float>(raw[i].imag()));

    std::cerr << "[SigMF] " << n << " complex128→float samples from " << path << '\n';
    return data;
}

DataBatch read_sigmf_dir(const std::string& dir_path, size_t max_samples) {
    DataBatch all;
    for (const auto& entry : fs::recursive_directory_iterator(dir_path)) {
        if (entry.path().extension() == ".sigmf-data" ||
            entry.path().extension() == ".dat") {
            size_t remaining = (max_samples > 0) ? max_samples - all.size() : 0;
            try {
                auto chunk = read_sigmf_data(entry.path().string(), remaining);
                all.insert(all.end(), chunk.begin(), chunk.end());
            } catch (const std::exception& e) {
                std::cerr << "[SigMF] Skipping " << entry.path()
                          << ": " << e.what() << '\n';
            }
            if (max_samples > 0 && all.size() >= max_samples)
                break;
        }
    }
    std::cerr << "[SigMF] Total: " << all.size() << " complex samples\n";
    return all;
}

DataBatch segment_to_frames(const DataBatch& samples, size_t fft_size) {
    size_t num_frames = samples.size() / fft_size;
    if (num_frames == 0)
        throw std::runtime_error("Not enough samples for a single FFT frame of size "
                                 + std::to_string(fft_size));
    DataBatch out(num_frames * fft_size);
    std::memcpy(out.data(), samples.data(), out.size() * sizeof(cf32));
    return out;
}
