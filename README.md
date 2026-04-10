# Batched 1D FFT Benchmark                                                                                                                                                            
                                                                                                                                                                                        
  Benchmarks four implementations of a batched 1D complex FFT on CPU and GPU:
                                                                                                                                                                                        
  | Implementation | Description |                                                                                                                                                      
  |---------------|-------------|                                                                                                                                                       
  | `own_cpu` | Hand-written Cooley-Tukey radix-2 DIT FFT with OpenMP batch parallelism |
  | `mkl` | Intel MKL DFTI (CPU reference) |                                                                                                                                            
  | `own_gpu` | Hand-written CUDA FFT (shared-memory for N≤2048, global-memory multi-pass for N>2048) |                                                                                 
  | `cuFFT` | NVIDIA cuFFT (GPU reference) |                                                                                                                                            
                                                                                                                                                                                        
  Tested on NVIDIA P100 and V100 GPUs using synthetic, ORACLE RF, and DNS Challenge datasets.                                                                                           
                                                                                                                                                                                        
  # Requirements                                                                                                                                                                                            
  - CUDA 12.3+                                                                                                                                                                          
  - Intel MKL 2025.0                                                                                                                                                                    
  - CMake 3.20+                               
  - OpenMP                                
  - FFTW3 (optional)
                                                                                                                                                                                        
  On the Explorer cluster:                
  module load cuda/12.3.0                                                                                                                                                               
  module load intel/mkl-2025.0                                                                                                                                                          
  module load cmake/3.30.2                
                                                                                                                                                                                        
  ## Build                                                                                                                                            

   
  ```bash                                                                                                                                                                               
  mkdir build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release         
  cmake --build . --parallel 4            

  With FFTW3:                                                                                                                                                                           
  cmake .. -DCMAKE_BUILD_TYPE=Release -DFFTW3_ROOT=$HOME/local
                                                                                                                                                                                        
  To build FFTW3 from source first:                                                                                                                                                     
  bash scripts/build_fftw3.sh             
                                                                                                                                                                                        
  Run                                                                                                                                                                                   
                                          
  ./build/fft_bench --experiment <N> --platform <p100|v100> [options]                                                                                                                   
                                                                                                                                                                                        
  ┌──────────────┬────────────────────────────────────────────────────────────────────────────────┐
  │     Flag     │                                  Description                                   │                                                                                     
  ├──────────────┼────────────────────────────────────────────────────────────────────────────────┤                                                                                     
  │ --experiment │ 1 = speedup sweep, 2 = batch scaling, 3 = thread scaling, verify = correctness │
  ├──────────────┼────────────────────────────────────────────────────────────────────────────────┤                                                                                     
  │ --dataset    │ synthetic, oracle, dns                                                         │
  ├──────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ --platform   │ Label written to CSV (e.g. p100, v100)                                         │
  ├──────────────┼────────────────────────────────────────────────────────────────────────────────┤                                                                                     
  │ --fft-size   │ FFT length N (must be power of 2, default: sweep)                              │                                                                                     
  ├──────────────┼────────────────────────────────────────────────────────────────────────────────┤                                                                                     
  │ --batch      │ Batch size                                                                     │                                                                                     
  ├──────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ --threads    │ OpenMP thread count                                                            │                                                                                     
  ├──────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ --repeats    │ Timed repetitions (default: 20)                                                │                                                                                     
  ├──────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ --warmup     │ Warmup reps excluded from timing (default: 3)                                  │                                                                                     
  ├──────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ --output     │ CSV output path                                                                │                                                                                     
  ├──────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ --data-path  │ Path to dataset directory                                                      │                                                                                     
  ├──────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ --max-gb     │ Cap on dataset size loaded (default: 2.0)                                      │
  └──────────────┴────────────────────────────────────────────────────────────────────────────────┘
                                          
  Examples
                                                                                                                                                                                        
  # Synthetic sweep on P100
  ./build/fft_bench --experiment 1 --dataset synthetic --platform p100 --threads 16 --output results/exp1_p100_synthetic.csv                                                            
                                              
  # Batch scaling on V100                 
  ./build/fft_bench --experiment 2 --dataset synthetic --platform v100 --fft-size 1024 --threads 16 --output results/exp2_v100.csv
                                                                                                                                                                                        
  # Correctness verification              
  ./build/fft_bench --experiment verify                                                                                                                                                 
                                                                                                                                                                                        
  Cluster Submission
                                                                                                                                                                                        
  sbatch scripts/submit_p100.sh
  sbatch scripts/submit_v100.sh
  sbatch scripts/submit_thread_scaling.sh     
                                          
  For profiling:
  sbatch scripts/profile_cpu.sh   # Intel VTune                                                                                                                                         
  sbatch scripts/profile_gpu.sh   # Nsight Systems + Nsight Compute
                                                                                                                                                                                        
  Datasets                                    
                                          
  Synthetic — generated in memory, always available, no setup needed.
                                                                                                                                                                                        
  ORACLE RF — WiFi captures from the ORACLE dataset (Northeastern Genesys Lab).
  Request access at https://www.genesys-lab.org/oracle, then:                                                                                                                           
  # Edit URLS array in the script with your download links
  bash scripts/download_oracle.sh ~/datasets/oracle                                                                                                                                     
  ORACLE_PATH=~/datasets/oracle sbatch scripts/submit_p100.sh
                                                                                                                                                                                        
  DNS Challenge — noisy speech from Microsoft DNS Challenge 4 (ICASSP 2022).
  bash scripts/download_dns.sh ~/datasets/dns
  DNS_PATH=~/datasets/dns sbatch scripts/submit_p100.sh
                                                                                                                                                                                        
  Results and Figures                     
                                                                                                                                                                                        
  After running experiments, generate all figures:                                                                                                                                      
  python3 scripts/plot_results.py
                                                                                                                                                                                        
  Figures are saved as PDF to figures/. Pre-generated results from P100 and V100 runs are in results/.
                                              
  Correctness Verification                

  Tests all implementations against MKL DFTI using 3 signal types × 3 FFT sizes = 27 tests. All must pass NRMSE < 1e-4.                                                                 
                                          
  ./build/fft_bench --experiment verify                                                                                                                                                 
                                                                        
