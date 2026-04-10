// fft_bench — batched 1-D FFT benchmarking harness
// Experiments:
//   1  CPU (MKL/FFTW3) vs GPU (cuFFT) speedup sweep over FFT sizes
//   2  Batch-depth scaling at fixed FFT size (1024 by default)
//
// Usage:
//   ./fft_bench --experiment <1|2|all> \
//               --dataset   <synthetic|oracle|dns> \
//               --platform  <v100|p100>            \
//               --data-path <path-to-dataset>      \
//               --output    <file.csv>             \
//               [--threads  <n>]                   \
//               [--repeats  <n>]                   \
//               [--warmup   <n>]                   \
//               [--fft-size <n>]   # fix size for exp 2
//               [--max-gb   <f>]   # max GB of data loaded for real datasets

#include "common/types.hpp"
#include "common/result.hpp"
#include "common/timer.hpp"
#include "cpu/mkl_bench.hpp"
#include "cpu/fft_own.hpp"
#ifdef HAVE_FFTW3
#  include "cpu/fftw_bench.hpp"
#endif
#include "gpu/cufft_bench.cuh"
#include "gpu/fft_own.cuh"
#include "verify.cuh"
#include "data/sigmf_reader.hpp"
#include "data/audio_reader.hpp"
#include "data/synthetic_gen.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <cmath>

// ── CLI config ─────────────────────────────────────────────────────────────────
struct Config {
    int         experiment  = 0;          // 0 = all, 1, 2
    std::string dataset     = "synthetic"; // synthetic | oracle | dns
    std::string platform    = "v100";
    std::string data_path   = "";
    std::string output_file = "";
    int         threads     = 16;
    int         repeats     = 20;
    int         warmup      = 3;
    size_t      fft_size_fixed = 1024;    // experiment 2
    double      max_gb      = 2.0;        // max GB to load for real datasets
    double      budget_gb   = 0.25;       // synthetic data budget per FFT size
};

static void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << "\n"
        << "  --experiment <1|2|all>          (default: all)\n"
        << "  --dataset    <synthetic|oracle|dns> (default: synthetic)\n"
        << "  --platform   <name>             tag in output CSV, e.g. v100\n"
        << "  --data-path  <path>             path to dataset directory\n"
        << "  --output     <file.csv>         (default: stdout)\n"
        << "  --threads    <n>                CPU OpenMP threads (default: 16)\n"
        << "  --repeats    <n>                timing repeats (default: 20)\n"
        << "  --warmup     <n>                warmup iters  (default: 3)\n"
        << "  --fft-size   <n>                fixed FFT size for exp 2 (default: 1024)\n"
        << "  --max-gb     <f>                max GB to load from real datasets (default: 2.0)\n"
        << "  --budget-gb  <f>                synthetic data budget per FFT size (default: 0.25)\n"
        << "                                  use 8.0 for roofline sweep filling GPU memory\n";
}

static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + a);
            return argv[++i];
        };
        if      (a == "--experiment") { auto v = next(); cfg.experiment = (v == "all") ? 0 : (v == "verify") ? 4 : std::stoi(v); }
        else if (a == "--dataset")    cfg.dataset      = next();
        else if (a == "--platform")   cfg.platform     = next();
        else if (a == "--data-path")  cfg.data_path    = next();
        else if (a == "--output")     cfg.output_file  = next();
        else if (a == "--threads")    cfg.threads      = std::stoi(next());
        else if (a == "--repeats")    cfg.repeats      = std::stoi(next());
        else if (a == "--warmup")     cfg.warmup       = std::stoi(next());
        else if (a == "--fft-size")   cfg.fft_size_fixed = std::stoull(next());
        else if (a == "--max-gb")     cfg.max_gb       = std::stod(next());
        else if (a == "--budget-gb")  cfg.budget_gb    = std::stod(next());
        else if (a == "--help" || a == "-h") { usage(argv[0]); std::exit(0); }
        else { std::cerr << "Unknown flag: " << a << '\n'; usage(argv[0]); std::exit(1); }
    }
    return cfg;
}

// ── Data loading ───────────────────────────────────────────────────────────────
// Returns framed samples (size == whole multiple of fft_size).
// batch_size = returned_size / fft_size.
static DataBatch load_data(const Config& cfg, size_t fft_size) {
    size_t max_samples = static_cast<size_t>(
        cfg.max_gb * 1024.0 * 1024.0 * 1024.0 / sizeof(cf32));

    if (cfg.dataset == "synthetic") {
        size_t budget_bytes = static_cast<size_t>(cfg.budget_gb * 1024.0 * 1024.0 * 1024.0);
        size_t batch = std::max<size_t>(64, budget_bytes / (fft_size * sizeof(cf32)));
        return generate_synthetic(fft_size, batch);
    }

    if (cfg.dataset == "oracle") {
        if (cfg.data_path.empty())
            throw std::runtime_error("--data-path required for oracle dataset");
        auto raw = read_sigmf_dir(cfg.data_path, max_samples);
        return segment_to_frames(raw, fft_size);
    }

    if (cfg.dataset == "dns") {
        if (cfg.data_path.empty())
            throw std::runtime_error("--data-path required for dns dataset");
        auto raw = read_wav_dir(cfg.data_path, max_samples);
        return segment_to_frames(raw, fft_size);
    }

    throw std::runtime_error("Unknown dataset: " + cfg.dataset);
}

// ── Experiment 1: CPU vs GPU speedup sweep ────────────────────────────────────
// FFT sizes 2^7 … 2^20; for real datasets only a subset of relevant sizes.
static void run_experiment1(const Config& cfg, std::ostream& out) {
    std::cerr << "\n=== Experiment 1: CPU vs GPU speedup sweep ===\n";

    // Full sweep for synthetic; domain-relevant subset for real data
    std::vector<size_t> sizes;
    if (cfg.dataset == "synthetic") {
        for (int e = 7; e <= 20; e++)
            sizes.push_back(1UL << e);
    } else if (cfg.dataset == "oracle") {
        // ORACLE: native 1024-point segmentation
        sizes = { 512, 1024, 2048 };
    } else if (cfg.dataset == "dns") {
        // DNS Challenge 4: explicit STFT frame sizes from proposal
        sizes = { 256, 512, 1024 };
    }

    for (size_t N : sizes) {
        std::cerr << "\n--- FFT size " << N << " ---\n";
        DataBatch data;
        try {
            data = load_data(cfg, N);
        } catch (const std::exception& e) {
            std::cerr << "[SKIP] " << e.what() << '\n';
            continue;
        }

        size_t batch = data.size() / N;
        std::cerr << "  batch=" << batch
                  << "  data=" << data.size() * sizeof(cf32) / (1024*1024) << " MB\n";

        // CPU — own Cooley-Tukey Radix-2 ─────────────────────────────────────
        try {
            auto r = run_own_cpu_bench(data.data(), N, batch,
                                       cfg.dataset, cfg.platform,
                                       cfg.threads, cfg.repeats, cfg.warmup);
            write_csv_row(out, r);
            out.flush();
            std::cerr << "  own_cpu " << r.gflops << " GFlops  "
                      << r.kernel_ms << " ms\n";
        } catch (const std::exception& e) {
            std::cerr << "  [own_cpu error] " << e.what() << '\n';
        }

        // CPU — MKL ───────────────────────────────────────────────────────────
        try {
            auto r = run_mkl_bench(data.data(), N, batch,
                                   cfg.dataset, cfg.platform,
                                   cfg.threads, cfg.repeats, cfg.warmup);
            write_csv_row(out, r);
            out.flush();
            std::cerr << "  MKL    " << r.gflops << " GFlops  "
                      << r.kernel_ms << " ms\n";
        } catch (const std::exception& e) {
            std::cerr << "  [MKL error] " << e.what() << '\n';
        }

#ifdef HAVE_FFTW3
        // CPU — FFTW3 (optional, built from source) ───────────────────────────
        try {
            auto r = run_fftw_bench(data.data(), N, batch,
                                    cfg.dataset, cfg.platform,
                                    cfg.threads, cfg.repeats, cfg.warmup);
            write_csv_row(out, r);
            out.flush();
            std::cerr << "  FFTW3  " << r.gflops << " GFlops  "
                      << r.kernel_ms << " ms\n";
        } catch (const std::exception& e) {
            std::cerr << "  [FFTW3 error] " << e.what() << '\n';
        }
#endif

        // GPU — own Cooley-Tukey Radix-2 ──────────────────────────────────────
        try {
            auto r = run_own_gpu_bench(data.data(), N, batch,
                                       cfg.dataset, cfg.platform,
                                       cfg.repeats, cfg.warmup);
            write_csv_row(out, r);
            out.flush();
            std::cerr << "  own_gpu " << r.gflops << " GFlops  "
                      << "ker=" << r.kernel_ms << " ms  "
                      << "h2d=" << r.h2d_ms    << " ms  "
                      << "d2h=" << r.d2h_ms    << " ms\n";
        } catch (const std::exception& e) {
            std::cerr << "  [own_gpu error] " << e.what() << '\n';
        }

        // GPU — cuFFT ─────────────────────────────────────────────────────────
        try {
            auto r = run_cufft_bench(data.data(), N, batch,
                                     cfg.dataset, cfg.platform,
                                     cfg.repeats, cfg.warmup);
            write_csv_row(out, r);
            out.flush();
            std::cerr << "  cuFFT  " << r.gflops << " GFlops  "
                      << "ker=" << r.kernel_ms << " ms  "
                      << "h2d=" << r.h2d_ms    << " ms  "
                      << "d2h=" << r.d2h_ms    << " ms\n";
        } catch (const std::exception& e) {
            std::cerr << "  [cuFFT error] " << e.what() << '\n';
        }
    }
}

// ── Experiment 2: Batch-depth scaling at fixed FFT size ───────────────────────
static void run_experiment2(const Config& cfg, std::ostream& out) {
    size_t N = cfg.fft_size_fixed;
    std::cerr << "\n=== Experiment 2: Batch scaling (FFT size=" << N << ") ===\n";

    // Batch sizes: powers of 2 from 1 up to 1 M, plus a few non-power-of-2
    std::vector<size_t> batches;
    for (size_t b = 1; b <= 1000000UL; b *= 2)
        batches.push_back(b);
    // Add 500 K if not already present
    if (batches.back() != 1000000UL)
        batches.push_back(1000000UL);

    // Pre-generate the largest dataset once; slice for smaller batches
    size_t max_batch = batches.back();
    DataBatch big;
    try {
        big = generate_synthetic(N, max_batch);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << '\n';
        return;
    }

    for (size_t batch : batches) {
        // Use prefix of big to avoid re-allocation
        const cf32* ptr = big.data();

        std::cerr << "  batch=" << batch << "  "
                  << batch * N * sizeof(cf32) / (1024*1024) << " MB\n";

        // CPU — own Cooley-Tukey
        try {
            auto r = run_own_cpu_bench(ptr, N, batch,
                                       "synthetic", cfg.platform,
                                       cfg.threads, cfg.repeats, cfg.warmup);
            write_csv_row(out, r);
            out.flush();
        } catch (const std::exception& e) {
            std::cerr << "    [own_cpu] " << e.what() << '\n';
        }

        // CPU — MKL
        try {
            auto r = run_mkl_bench(ptr, N, batch,
                                   "synthetic", cfg.platform,
                                   cfg.threads, cfg.repeats, cfg.warmup);
            write_csv_row(out, r);
            out.flush();
        } catch (const std::exception& e) {
            std::cerr << "    [MKL] " << e.what() << '\n';
        }

#ifdef HAVE_FFTW3
        // CPU — FFTW3 (optional)
        try {
            auto r = run_fftw_bench(ptr, N, batch,
                                    "synthetic", cfg.platform,
                                    cfg.threads, cfg.repeats, cfg.warmup);
            write_csv_row(out, r);
            out.flush();
        } catch (const std::exception& e) {
            std::cerr << "    [FFTW3] " << e.what() << '\n';
        }
#endif

        // GPU — own Cooley-Tukey
        try {
            auto r = run_own_gpu_bench(ptr, N, batch,
                                       "synthetic", cfg.platform,
                                       cfg.repeats, cfg.warmup);
            write_csv_row(out, r);
            out.flush();
        } catch (const std::exception& e) {
            std::cerr << "    [own_gpu] " << e.what() << '\n';
        }

        // GPU — cuFFT
        try {
            auto r = run_cufft_bench(ptr, N, batch,
                                     "synthetic", cfg.platform,
                                     cfg.repeats, cfg.warmup);
            write_csv_row(out, r);
            out.flush();
        } catch (const std::exception& e) {
            std::cerr << "    [cuFFT] " << e.what() << '\n';
        }
    }
}

// ── Experiment 3: Thread scaling (strong + weak) for own_cpu ─────────────────
// Strong scaling: fixed N and fixed total batch, vary threads 1→2→4→8→16.
//   Ideal: time halves each time; deviations reveal Amdahl overhead.
// Weak scaling: fixed N, batch grows proportionally with thread count.
//   Ideal: time stays constant; deviations reveal communication/sync cost.
static void run_experiment3(const Config& cfg, std::ostream& out) {
    size_t N           = cfg.fft_size_fixed;   // default 1024
    size_t base_batch  = 32768;                // work for 1 thread in weak scaling
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};

    std::cerr << "\n=== Experiment 3: Thread scaling (N=" << N << ") ===\n";

    // Pre-generate the largest dataset (weak scaling needs base_batch * 16 frames)
    size_t max_batch = base_batch * thread_counts.back();
    DataBatch big    = generate_synthetic(N, max_batch);

    for (int t : thread_counts) {
        // ── Strong scaling: fixed total work = base_batch * max_threads ───────
        {
            size_t batch = max_batch;   // same total work for all thread counts
            std::cerr << "  strong  threads=" << t
                      << "  batch=" << batch << '\n';
            try {
                auto r = run_own_cpu_bench(big.data(), N, batch,
                                           "synthetic_strong", cfg.platform,
                                           t, cfg.repeats, cfg.warmup);
                write_csv_row(out, r);
                out.flush();
                std::cerr << "    own_cpu " << r.gflops << " GFlops  "
                          << r.kernel_ms << " ms\n";
            } catch (const std::exception& e) {
                std::cerr << "    [own_cpu] " << e.what() << '\n';
            }
            // MKL for reference
            try {
                auto r = run_mkl_bench(big.data(), N, batch,
                                       "synthetic_strong", cfg.platform,
                                       t, cfg.repeats, cfg.warmup);
                write_csv_row(out, r);
                out.flush();
            } catch (const std::exception& e) {
                std::cerr << "    [mkl] " << e.what() << '\n';
            }
        }

        // ── Weak scaling: work per thread stays constant = base_batch ─────────
        {
            size_t batch = base_batch * static_cast<size_t>(t);
            std::cerr << "  weak    threads=" << t
                      << "  batch=" << batch << '\n';
            try {
                auto r = run_own_cpu_bench(big.data(), N, batch,
                                           "synthetic_weak", cfg.platform,
                                           t, cfg.repeats, cfg.warmup);
                write_csv_row(out, r);
                out.flush();
                std::cerr << "    own_cpu " << r.gflops << " GFlops  "
                          << r.kernel_ms << " ms\n";
            } catch (const std::exception& e) {
                std::cerr << "    [own_cpu] " << e.what() << '\n';
            }
            try {
                auto r = run_mkl_bench(big.data(), N, batch,
                                       "synthetic_weak", cfg.platform,
                                       t, cfg.repeats, cfg.warmup);
                write_csv_row(out, r);
                out.flush();
            } catch (const std::exception& e) {
                std::cerr << "    [mkl] " << e.what() << '\n';
            }
        }
    }
}

// ── Entry point ────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);

    // Validate dataset
    if (cfg.dataset != "synthetic" && cfg.dataset != "oracle" && cfg.dataset != "dns") {
        std::cerr << "Unknown dataset: " << cfg.dataset << '\n';
        return 1;
    }

    // Open output
    std::ofstream fout;
    std::ostream* out = &std::cout;
    if (!cfg.output_file.empty()) {
        fout.open(cfg.output_file, std::ios::app);
        if (!fout) {
            std::cerr << "Cannot open output file: " << cfg.output_file << '\n';
            return 1;
        }
        out = &fout;
    }

    // Write CSV header only when creating a new file (or going to stdout)
    bool write_header = cfg.output_file.empty() || fout.tellp() == 0;
    if (write_header)
        write_csv_header(*out);

    std::cerr << "Platform: " << cfg.platform
              << "  Dataset: " << cfg.dataset
              << "  Threads: " << cfg.threads
              << "  Repeats: " << cfg.repeats
              << "  Warmup: "  << cfg.warmup << '\n';

    if (cfg.experiment == 4) {
        run_verify();
        return 0;
    }
    if (cfg.experiment == 0 || cfg.experiment == 1)
        run_experiment1(cfg, *out);
    if (cfg.experiment == 0 || cfg.experiment == 2)
        run_experiment2(cfg, *out);
    if (cfg.experiment == 3)
        run_experiment3(cfg, *out);

    return 0;
}
