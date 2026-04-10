#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=fft_threads
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=fft_threads.log
#SBATCH --partition=courses-gpu

module load cuda/12.3.0
module load intel/mkl-2025.0

./build/fft_bench --experiment 3 --platform v100 --fft-size 1024 --repeats 20 --warmup 3 --output results/exp3_thread_scaling_v100.csv
./build/fft_bench --experiment 3 --platform p100 --fft-size 1024 --repeats 20 --warmup 3 --output results/exp3_thread_scaling_p100.csv
