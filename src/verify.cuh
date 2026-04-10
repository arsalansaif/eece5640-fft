#pragma once

// Run correctness verification: computes FFT with all four implementations on
// known test inputs and reports normalized RMS error vs. MKL reference.
// Tests N = {64, 1024, 4096} to cover both shared-memory and global-memory
// GPU kernel paths.
void run_verify();
