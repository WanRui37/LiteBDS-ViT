# Usage

This directory contains only the BDS-related kernels and benchmark outputs for quickly validating INT4 BDS computation under different matrix sizes.

## Quick Start

```bash
# INT1/INT4/INT8 dense GEMM benchmarks implemented with CUTLASS
make bench_gemm.bin
python run-gemm.py

# Matrix sizes greater than or equal to 256
make int4_mma_compute_bound.bin
./int4_mma_compute_bound.bin

# Matrix sizes smaller than 256 (e.g., 128 × 128 × 128)
make int4_mma_compute_bound_128x128x128.bin
./int4_mma_compute_bound_128x128x128.bin
```