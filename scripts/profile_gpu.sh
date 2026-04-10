#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=fft_nsight
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:p100:1
#SBATCH --output=fft_nsight.log
#SBATCH --partition=courses-gpu

module load cuda/12.3.0
module load intel/mkl-2025.0
module load Nsight/2024.7.1

nsys profile --trace=cuda,osrt --output=results/nsight/nsys_timeline --force-overwrite=true \
     ./build/fft_bench --experiment 2 --dataset synthetic --platform p100 --fft-size 1024 --repeats 5 --warmup 1

ncu --set basic --kernel-name regex:"fft_shared_kernel|fft_butterfly_stage|bit_reverse_kernel" \
    --launch-count 5 --export results/nsight/ncu_own_gpu --force-overwrite \
    ./build/fft_bench --experiment 1 --dataset synthetic --platform p100 --repeats 3 --warmup 1

ncu --set basic --kernel-name regex:"volta_fp32" \
    --launch-count 5 --export results/nsight/ncu_cufft --force-overwrite \
    ./build/fft_bench --experiment 1 --dataset synthetic --platform p100 --repeats 3 --warmup 1
