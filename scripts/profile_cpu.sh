#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=fft_vtune
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:p100:1
#SBATCH --output=fft_vtune.log
#SBATCH --partition=courses-gpu

module load cuda/12.3.0
module load intel/mkl-2025.0
module load VTune/2025.0

vtune -collect hpc-performance -result-dir results/vtune/hpc_perf \
      -- ./build/fft_bench --experiment 1 --dataset synthetic --platform p100 --threads 16 --repeats 5 --warmup 1

vtune -collect memory-access -result-dir results/vtune/mem_access \
      -- ./build/fft_bench --experiment 1 --dataset synthetic --platform p100 --threads 16 --repeats 5 --warmup 1
