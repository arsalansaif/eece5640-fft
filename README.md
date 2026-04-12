# Batched 1D FFT Benchmark

A performance study of batched 1D complex FFT on CPU and GPU. The project implements the Cooley-Tukey radix-2 Decimation-in-Time (DIT) algorithm from scratch in both C++ (CPU) and CUDA (GPU), then benchmarks it against industry-standard libraries — Intel MKL on CPU and NVIDIA cuFFT on GPU.

The goal is to understand how close a hand-written implementation can get to a production library, where the bottlenecks are (memory bandwidth vs. compute), and how performance scales with batch size, FFT size, and thread count.

Experiments run on NVIDIA P100 (PCIe, 549 GB/s HBM2) and V100 (SXM2, 900 GB/s HBM2) GPUs using three datasets: synthetic random data, real-world WiFi RF captures (ORACLE), and noisy speech audio (DNS Challenge).

## Implementations

| Implementation | Description |
|---|---|
| `own_cpu` | Hand-written Cooley-Tukey radix-2 DIT FFT parallelized across the batch with OpenMP |
| `mkl` | Intel MKL DFTI — CPU reference |
| `own_gpu` | Hand-written CUDA FFT: shared-memory kernel for N≤2048, global-memory multi-pass for N>2048 |
| `cuFFT` | NVIDIA cuFFT — GPU reference |

### CPU Implementation

The CPU FFT uses the standard iterative Cooley-Tukey algorithm:
1. Bit-reversal permutation to reorder input
2. log2(N) butterfly stages, each combining pairs of complex values with twiddle factors

OpenMP parallelizes across the batch — each thread handles a separate FFT independently, giving near-linear scaling up to 16 threads.

### GPU Implementation

Two CUDA kernels depending on FFT size:

- **N ≤ 2048 (shared-memory kernel)** — one CUDA block per FFT, all N elements loaded into shared memory. The entire computation happens in shared memory with no global memory traffic, then results are written back once. Launched with N/2 threads per block.

- **N > 2048 (global-memory multi-pass)** — shared memory is too small, so data stays in global memory. A separate `bit_reverse_kernel` permutes the input, then `fft_butterfly_stage` is called once per DIT stage (log2(N) launches total).

## Experiments

| Experiment | Description |
|---|---|
| Speedup sweep | GFlops vs FFT size (64 to 65536) across all four implementations |
| Batch scaling | Throughput vs batch size (1 to 524288) at fixed N=1024 |
| Thread scaling | Strong and weak scaling on CPU across 1–16 threads |
| Verification | Correctness check: NRMSE vs MKL reference across 3 signal types × 3 sizes |

## Key Results

- Own GPU reaches **1,469 GFlops** on V100 vs cuFFT's **2,490 GFlops** (59% of reference)
- V100 is **2.7× faster** than P100 for own_gpu (exceeds the 1.64× bandwidth ratio — benefits from higher occupancy)
- CPU strong scaling: **15.7× speedup** on 16 threads (98% parallel efficiency)
- PCIe bottleneck dominates at large batch sizes — transfer time is 24× the kernel time at batch=524288
- Both GPUs are memory-bound: roofline analysis shows arithmetic intensity well below the compute ridge point

## Requirements

- CUDA 12.3, Intel MKL 2025.0, CMake 3.20+, OpenMP
- FFTW3 (optional)

On Explorer cluster:

```bash
module load cuda/12.3.0
module load intel/mkl-2025.0
module load cmake/3.30.2
```

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel 4
```

With FFTW3 (build from source first with `bash scripts/build_fftw3.sh`):

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DFFTW3_ROOT=$HOME/local
cmake --build . --parallel 4
```

## Run

```
./build/fft_bench --experiment <1|2|3|verify> --platform <p100|v100> [options]
```

- `--experiment` — `1` speedup sweep, `2` batch scaling, `3` thread scaling, `verify` correctness check
- `--dataset` — `synthetic`, `oracle`, `dns`
- `--fft-size` — FFT length N (power of 2)
- `--batch` — batch size
- `--threads` — OpenMP thread count
- `--repeats` / `--warmup` — timing repetitions
- `--output` — CSV output path
- `--data-path` — path to dataset directory
- `--max-gb` — dataset size cap (default: 2.0 GB)

### Examples

```bash
# Speedup sweep on P100
./build/fft_bench --experiment 1 --dataset synthetic --platform p100 --threads 16 --output results/exp1_p100_synthetic.csv

# Batch scaling on V100
./build/fft_bench --experiment 2 --dataset synthetic --platform v100 --fft-size 1024 --threads 16 --output results/exp2_v100.csv

# Correctness verification
./build/fft_bench --experiment verify
```

## Cluster Submission

```bash
sbatch scripts/submit_p100.sh
sbatch scripts/submit_v100.sh
sbatch scripts/submit_thread_scaling.sh
sbatch scripts/profile_cpu.sh    # Intel VTune
sbatch scripts/profile_gpu.sh    # Nsight Systems + Nsight Compute
```

## Datasets

**Synthetic** — generated in memory, no setup needed.

**ORACLE RF** — real WiFi RF captures (Northeastern Genesys Lab). Request access at https://www.genesys-lab.org/oracle, paste download links into `scripts/download_oracle.sh`, then:

```bash
bash scripts/download_oracle.sh ~/datasets/oracle
ORACLE_PATH=~/datasets/oracle sbatch scripts/submit_p100.sh
```

**DNS Challenge** — noisy speech recordings from Microsoft DNS Challenge 4 (ICASSP 2022):

```bash
bash scripts/download_dns.sh ~/datasets/dns
DNS_PATH=~/datasets/dns sbatch scripts/submit_p100.sh
```

## Figures

```bash
python3 scripts/plot_results.py
```

Pre-generated results are in `results/`, figures in `figures/`.

## Report

See `report.pdf` for full implementation details, experiments, profiling, roofline analysis, and discussion.
