#!/bin/bash
# Build script for Explorer cluster.
#
# Usage:
#   bash build.sh          # system GCC host compiler (default)
#   bash build.sh icpx     # Intel icpx (Clang-based) host compiler

set -euo pipefail

HOST_COMPILER="${1:-gcc}"
BUILD_DIR=build

# Detect build directory name so GCC and icpx builds don't collide
if [[ "$HOST_COMPILER" == "icpx" ]]; then
    BUILD_DIR=build_icpx
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

if [[ "$HOST_COMPILER" == "icpx" ]]; then
    HOSTCC=$(command -v icpx 2>/dev/null)
    if [[ -z "$HOSTCC" ]]; then
        echo "icpx not found — load intel/compilers-2025.0.4 first"
        exit 1
    fi
    echo "Host compiler: $HOSTCC"
    cmake .. \
        -DCMAKE_CUDA_HOST_COMPILER="$HOSTCC" \
        -DCMAKE_BUILD_TYPE=Release
else
    echo "Host compiler: system g++"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release
fi

cmake --build . --parallel 4
echo ""
echo "Build complete: $(pwd)/fft_bench"
