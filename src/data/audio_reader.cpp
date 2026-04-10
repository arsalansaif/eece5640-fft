#include "audio_reader.hpp"
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

// ── Minimal WAV parser ────────────────────────────────────────────────────────
// Handles PCM-16 (audio_format=1) and IEEE-float-32 (audio_format=3),
// mono or stereo (only channel 0 is used).

namespace {

// Read a little-endian integer of T bytes from stream
template<typename T>
T read_le(std::ifstream& f) {
    T val{};
    f.read(reinterpret_cast<char*>(&val), sizeof(T));
    return val;
}

struct WavInfo {
    uint16_t audio_format;   // 1 = PCM, 3 = float
    uint16_t num_channels;
    uint32_t sample_rate;
    uint16_t bits_per_sample;
    std::streampos data_offset;
    uint32_t data_bytes;
};

WavInfo parse_wav_header(std::ifstream& f, const std::string& path) {
    // RIFF chunk
    char riff[4]; f.read(riff, 4);
    if (std::strncmp(riff, "RIFF", 4) != 0)
        throw std::runtime_error("Not a RIFF file: " + path);
    read_le<uint32_t>(f);           // file size (ignored)
    char wave[4]; f.read(wave, 4);
    if (std::strncmp(wave, "WAVE", 4) != 0)
        throw std::runtime_error("Not a WAVE file: " + path);

    WavInfo info{};
    bool found_fmt  = false;
    bool found_data = false;

    while (f && !found_data) {
        char id[4];
        f.read(id, 4);
        if (!f) break;
        uint32_t chunk_size = read_le<uint32_t>(f);

        if (std::strncmp(id, "fmt ", 4) == 0) {
            info.audio_format    = read_le<uint16_t>(f);
            info.num_channels    = read_le<uint16_t>(f);
            info.sample_rate     = read_le<uint32_t>(f);
            read_le<uint32_t>(f);  // byte_rate
            read_le<uint16_t>(f);  // block_align
            info.bits_per_sample = read_le<uint16_t>(f);
            // skip any extension bytes
            if (chunk_size > 16)
                f.seekg(chunk_size - 16, std::ios::cur);
            found_fmt = true;
        } else if (std::strncmp(id, "data", 4) == 0) {
            info.data_offset = f.tellg();
            info.data_bytes  = chunk_size;
            found_data = true;
        } else {
            f.seekg(chunk_size, std::ios::cur);
        }
    }

    if (!found_fmt)  throw std::runtime_error("No fmt  chunk: " + path);
    if (!found_data) throw std::runtime_error("No data chunk: " + path);
    if (info.audio_format != 1 && info.audio_format != 3)
        throw std::runtime_error("Unsupported audio format (need PCM-16 or float-32): " + path);

    return info;
}

} // namespace

DataBatch read_wav_file(const std::string& path, size_t max_samples) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open: " + path);

    WavInfo info = parse_wav_header(f, path);
    f.seekg(info.data_offset);

    size_t bytes_per_sample_ch = info.bits_per_sample / 8;
    size_t bytes_per_frame     = bytes_per_sample_ch * info.num_channels;
    size_t total_frames        = info.data_bytes / bytes_per_frame;
    if (max_samples > 0 && total_frames > max_samples)
        total_frames = max_samples;

    DataBatch result;
    result.reserve(total_frames);

    for (size_t i = 0; i < total_frames; i++) {
        float sample = 0.f;

        if (info.audio_format == 3) {
            // IEEE float-32
            f.read(reinterpret_cast<char*>(&sample), 4);
            // skip remaining channels
            f.seekg(4 * (info.num_channels - 1), std::ios::cur);
        } else {
            // PCM-16
            int16_t s16 = 0;
            f.read(reinterpret_cast<char*>(&s16), 2);
            sample = static_cast<float>(s16) / 32768.f;
            f.seekg(2 * (info.num_channels - 1), std::ios::cur);
        }

        if (!f) break;
        result.emplace_back(sample, 0.f);  // real-valued; im = 0
    }

    return result;
}

DataBatch read_wav_dir(const std::string& dir_path, size_t max_samples) {
    DataBatch all;
    size_t file_count = 0;

    for (const auto& entry : fs::recursive_directory_iterator(dir_path)) {
        if (entry.path().extension() == ".wav") {
            size_t remaining = (max_samples > 0) ? max_samples - all.size() : 0;
            try {
                auto chunk = read_wav_file(entry.path().string(), remaining);
                all.insert(all.end(), chunk.begin(), chunk.end());
                ++file_count;
            } catch (const std::exception& e) {
                std::cerr << "[WAV] Skipping " << entry.path()
                          << ": " << e.what() << '\n';
            }
            if (max_samples > 0 && all.size() >= max_samples)
                break;
        }
    }

    std::cerr << "[DNS] " << all.size() << " samples from "
              << file_count << " WAV files in " << dir_path << '\n';
    return all;
}
