#pragma once
#include <string>
#include <ostream>

struct BenchResult {
    std::string impl;        // "fftw3", "cufft", "mkl"
    std::string dataset;     // "synthetic", "oracle", "dns"
    std::string platform;    // "cpu", "v100", "p100"  (set by --platform flag)
    size_t      fft_size;
    size_t      batch_size;
    int         num_threads; // CPU only; 0 for GPU

    // All times in milliseconds, median over measured repeats
    double h2d_ms;           // GPU H2D transfer; 0 for CPU
    double kernel_ms;        // pure FFT execution (GPU event or CPU chrono)
    double d2h_ms;           // GPU D2H transfer; 0 for CPU
    double total_wall_ms;    // h2d + kernel + d2h (== kernel_ms for CPU)

    double gflops;           // based on kernel_ms only
    double bandwidth_gbs;    // based on kernel_ms only
};

inline void write_csv_header(std::ostream& out) {
    out << "impl,dataset,platform,fft_size,batch_size,num_threads,"
           "h2d_ms,kernel_ms,d2h_ms,total_wall_ms,gflops,bandwidth_gbs\n";
}

inline void write_csv_row(std::ostream& out, const BenchResult& r) {
    out << r.impl        << ','
        << r.dataset     << ','
        << r.platform    << ','
        << r.fft_size    << ','
        << r.batch_size  << ','
        << r.num_threads << ','
        << r.h2d_ms      << ','
        << r.kernel_ms   << ','
        << r.d2h_ms      << ','
        << r.total_wall_ms << ','
        << r.gflops      << ','
        << r.bandwidth_gbs << '\n';
}
