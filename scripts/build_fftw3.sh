#!/bin/bash
# Build FFTW3 single-precision with OpenMP + AVX2 into ~/local.
# Run once on a login node. Then rebuild fft_bench with -DFFTW3_ROOT=$HOME/local

VERSION=3.3.10
PREFIX=$HOME/local
TMPDIR=$(mktemp -d)

cd $TMPDIR
curl -fsSL http://www.fftw.org/fftw-${VERSION}.tar.gz -o fftw.tar.gz
tar xf fftw.tar.gz
cd fftw-${VERSION}

./configure --prefix=$PREFIX --enable-float --enable-avx2 --enable-openmp --enable-shared --disable-static
make -j $(nproc)
make install

rm -rf $TMPDIR
