cmake_minimum_required(VERSION 3.20)
project(fft_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for both P100 (sm_60) and V100 (sm_70)
set(CMAKE_CUDA_ARCHITECTURES "60;70")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-relaxed-constexpr")

# ── CUDA Toolkit ───────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── OpenMP ────────────────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)

# ── Intel oneMKL (required — provides DFT + optional FFTW3 wrappers) ──────────
# module load intel/mkl-2025.0  sets MKLROOT
set(_MKL_HINTS
    "$ENV{MKLROOT}"
    "$ENV{MKL_ROOT}"
    "$ENV{ONEAPI_ROOT}/mkl/latest"
    "/shared/EL9/explorer/software/intel/mkl-2025.0"
)

find_library(MKL_LP64_LIB  NAMES mkl_intel_lp64  HINTS ${_MKL_HINTS} PATH_SUFFIXES lib/intel64 lib REQUIRED)
find_library(MKL_CORE_LIB  NAMES mkl_core         HINTS ${_MKL_HINTS} PATH_SUFFIXES lib/intel64 lib REQUIRED)
find_library(MKL_GNU_LIB   NAMES mkl_gnu_thread   HINTS ${_MKL_HINTS} PATH_SUFFIXES lib/intel64 lib REQUIRED)
find_path(MKL_INCLUDE_DIR  NAMES mkl_dfti.h        HINTS ${_MKL_HINTS} PATH_SUFFIXES include REQUIRED)

message(STATUS "MKL LP64:    ${MKL_LP64_LIB}")
message(STATUS "MKL core:    ${MKL_CORE_LIB}")
message(STATUS "MKL thread:  ${MKL_GNU_LIB}")
message(STATUS "MKL include: ${MKL_INCLUDE_DIR}")

# ── FFTW3 (optional — build from source with build_fftw3.sh) ─────────────────
# If you run build_fftw3.sh, set -DFFTW3_ROOT=$HOME/local when configuring.
set(_FFTW_HINTS
    "${FFTW3_ROOT}"
    "$ENV{FFTW3_ROOT}"
    "$ENV{FFTW_ROOT}"
    "$ENV{HOME}/local"
)

find_library(FFTW3F_LIB     NAMES fftw3f     HINTS ${_FFTW_HINTS} PATH_SUFFIXES lib lib64)
find_library(FFTW3F_OMP_LIB NAMES fftw3f_omp HINTS ${_FFTW_HINTS} PATH_SUFFIXES lib lib64)
find_path(FFTW3_INCLUDE_DIR  NAMES fftw3.h   HINTS ${_FFTW_HINTS} PATH_SUFFIXES include)

if(FFTW3F_LIB AND FFTW3_INCLUDE_DIR)
    set(HAS_FFTW3 TRUE)
    message(STATUS "FFTW3 found: ${FFTW3F_LIB}")
else()
    set(HAS_FFTW3 FALSE)
    message(STATUS "FFTW3 not found — CPU benchmark will use MKL only")
    message(STATUS "  Run scripts/build_fftw3.sh to build it, then reconfigure with -DFFTW3_ROOT=\$HOME/local")
endif()

# ── Sources ────────────────────────────────────────────────────────────────────
set(CPU_SOURCES
    src/cpu/mkl_bench.cpp
    src/cpu/fft_own.cpp
)
if(HAS_FFTW3)
    list(APPEND CPU_SOURCES src/cpu/fftw_bench.cpp)
endif()

add_executable(fft_bench
    src/main.cpp
    ${CPU_SOURCES}
    src/gpu/cufft_bench.cu
    src/gpu/fft_own.cu
    src/verify.cu
    src/data/sigmf_reader.cpp
    src/data/audio_reader.cpp
)

set_source_files_properties(
    src/gpu/cufft_bench.cu
    src/gpu/fft_own.cu
    src/verify.cu
    PROPERTIES LANGUAGE CUDA
)

target_include_directories(fft_bench PRIVATE
    src
    ${MKL_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_options(fft_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
)

target_compile_definitions(fft_bench PRIVATE HAVE_MKL=1)

target_link_libraries(fft_bench PRIVATE
    ${MKL_LP64_LIB}
    ${MKL_CORE_LIB}
    ${MKL_GNU_LIB}
    CUDA::cufft
    CUDA::cudart
    OpenMP::OpenMP_CXX
    gomp pthread m dl
)

if(HAS_FFTW3)
    target_compile_definitions(fft_bench PRIVATE HAVE_FFTW3=1)
    target_include_directories(fft_bench PRIVATE ${FFTW3_INCLUDE_DIR})
    target_link_libraries(fft_bench PRIVATE ${FFTW3F_LIB})
    if(FFTW3F_OMP_LIB)
        target_compile_definitions(fft_bench PRIVATE HAVE_FFTW3_THREADS=1)
        target_link_libraries(fft_bench PRIVATE ${FFTW3F_OMP_LIB})
    endif()
endif()
