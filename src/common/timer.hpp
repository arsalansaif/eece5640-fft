#pragma once
#include <chrono>
#include <vector>
#include <algorithm>

class CpuTimer {
public:
    void start() { t0_ = clock_t::now(); }
    void stop()  { t1_ = clock_t::now(); }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(t1_ - t0_).count();
    }
private:
    using clock_t = std::chrono::high_resolution_clock;
    clock_t::time_point t0_, t1_;
};

inline double vec_median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? (v[n/2 - 1] + v[n/2]) / 2.0 : v[n/2];
}
