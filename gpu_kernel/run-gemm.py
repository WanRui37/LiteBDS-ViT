#!/usr/bin/env python
import os

B = 64

N_K_list = [
    128,
    256,
    512,
    1024,
    2048,
]

for N_K in N_K_list:
    # os.system("./bench_gemm.bin {} {} {}".format(B, N_K, N_K))
    os.system("./bench_gemm.bin {} {} {}".format(N_K, N_K, N_K))