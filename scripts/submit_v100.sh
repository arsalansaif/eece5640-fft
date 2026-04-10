#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --job-name=fft_v100
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=fft_v100.log
#SBATCH --partition=courses-gpu

module load cuda/12.3.0
module load intel/mkl-2025.0

BIN=./build/fft_bench
PLATFORM=v100
THREADS=16

$BIN --experiment 1 --dataset synthetic --platform $PLATFORM --threads $THREADS --output results/exp1_${PLATFORM}_synthetic.csv
[ -d "${ORACLE_PATH:-}" ] && $BIN --experiment 1 --dataset oracle --platform $PLATFORM --data-path "$ORACLE_PATH" --max-gb 2.0 --threads $THREADS --output results/exp1_${PLATFORM}_oracle.csv
[ -d "${DNS_PATH:-}" ]    && $BIN --experiment 1 --dataset dns    --platform $PLATFORM --data-path "$DNS_PATH"    --max-gb 2.0 --threads $THREADS --output results/exp1_${PLATFORM}_dns.csv
$BIN --experiment 2 --dataset synthetic --platform $PLATFORM --fft-size 1024 --threads $THREADS --output results/exp2_${PLATFORM}.csv
