// int4_mma_compute_bound_v2.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(1);                                                   \
    }                                                              \
} while (0)

constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

constexpr int G = 4;
constexpr int TAU = 1;
constexpr int IG = K / G;   // 256
constexpr int OG = N / G;   // 256

#ifndef INNER_REPEAT
#define INNER_REPEAT 1
#endif

static_assert(M % 8 == 0, "M must be multiple of 8");
static_assert(N % 8 == 0, "N must be multiple of 8");
static_assert(K % 32 == 0, "K must be multiple of 32");
static_assert(G == 4, "This demo assumes G=4");
static_assert(IG % 32 == 0, "IG must be multiple of 32");
static_assert(OG % 32 == 0, "OG must be multiple of 32 for BDS_v3");

__host__ __device__ __forceinline__
uint32_t pack_s4(int val) {
    uint32_t nibble = static_cast<uint32_t>(val) & 0xFu;
    uint32_t packed = 0;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        packed |= (nibble << (4 * i));
    }

    return packed;
}

__device__ __forceinline__ void mma_m8n8k32_s4(
    int &d0,
    int &d1,
    unsigned a,
    unsigned b,
    int c0,
    int c1
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    asm volatile(
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%4, %5};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a), "r"(b), "r"(c0), "r"(c1)
    );
#else
    d0 = c0;
    d1 = c1;
#endif
}

// ============================================================
// Dense_v1
// 一个 block = 一个 warp
// 一个 block 计算一个 8x8 tile
// grid = (N/8, M/8)
// ============================================================
__global__ void dense_v1_kernel(int *C) {
    int lane = threadIdx.x & 31;

    int tile_n = blockIdx.x * 8;
    int tile_m = blockIdx.y * 8;

    unsigned a_frag = 0x11111111u;
    unsigned b_frag = 0x11111111u;

    int acc0 = 0;
    int acc1 = 0;

#pragma unroll 1
    for (int r = 0; r < INNER_REPEAT; ++r) {
#pragma unroll
        for (int kk = 0; kk < K; kk += 32) {
            int d0, d1;
            mma_m8n8k32_s4(d0, d1, a_frag, b_frag, acc0, acc1);
            acc0 = d0;
            acc1 = d1;
        }
    }

    int row = lane / 4;
    int col_pair = (lane % 4) * 2;

    int global_row = tile_m + row;
    int global_col0 = tile_n + col_pair;
    int global_col1 = global_col0 + 1;

    C[global_row * N + global_col0] = acc0;
    C[global_row * N + global_col1] = acc1;
}

// ============================================================
// BDS_v3
// G = 4
// 一个 block = 4 个 warp
// 每个 warp 仍然计算一个 8x8 tile
// 一个 block 计算 group 内连续 4 个 8x8 tile，即 8x32
//
// grid = (OG/32, M/8, G)
// block = 128 threads
// ============================================================
__global__ void bds_v3_kernel(int *Y) {
    int tid = threadIdx.x;
    int warp_id = tid >> 5;     // 0,1,2,3
    int lane = tid & 31;

    int gout = blockIdx.z;
    int tile_m = blockIdx.y * 8;

    int tile_n_in_group = blockIdx.x * 32 + warp_id * 8;

    int gin = (gout + TAU) & 3;

    int x_val = gin + 1;
    int w_val = gout + 1;

    unsigned a_frag = pack_s4(x_val);
    unsigned b_frag = pack_s4(w_val);

    int acc0 = 0;
    int acc1 = 0;

#pragma unroll 1
    for (int r = 0; r < INNER_REPEAT; ++r) {
#pragma unroll
        for (int kk = 0; kk < IG; kk += 32) {
            int d0, d1;
            mma_m8n8k32_s4(d0, d1, a_frag, b_frag, acc0, acc1);
            acc0 = d0;
            acc1 = d1;
        }
    }

    int row = lane / 4;
    int col_pair = (lane % 4) * 2;

    int global_row = tile_m + row;
    int global_col0 = gout * OG + tile_n_in_group + col_pair;
    int global_col1 = global_col0 + 1;

    Y[global_row * N + global_col0] = acc0;
    Y[global_row * N + global_col1] = acc1;
}

float benchmark_dense_v1(int *d_C, int repeat_launch) {
    dim3 block(32);
    dim3 grid(N / 8, M / 8);

    for (int i = 0; i < 10; ++i) {
        dense_v1_kernel<<<grid, block>>>(d_C);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat_launch; ++i) {
        dense_v1_kernel<<<grid, block>>>(d_C);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / repeat_launch;
}

float benchmark_bds_v3(int *d_Y, int repeat_launch) {
    dim3 block(128);
    dim3 grid(OG / 32, M / 8, G);

    for (int i = 0; i < 10; ++i) {
        bds_v3_kernel<<<grid, block>>>(d_Y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat_launch; ++i) {
        bds_v3_kernel<<<grid, block>>>(d_Y);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / repeat_launch;
}

bool check_dense(const int *h_C) {
    int expected = K * INNER_REPEAT;

    for (int i = 0; i < M * N; ++i) {
        if (h_C[i] != expected) {
            printf("[Dense_v1] Mismatch at %d: got %d, expected %d\n",
                   i, h_C[i], expected);
            return false;
        }
    }

    return true;
}

bool check_bds_v3(const int *h_Y) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int gout = n / OG;
            int gin = (gout + TAU) & 3;

            int x_val = gin + 1;
            int w_val = gout + 1;

            int expected = IG * x_val * w_val * INNER_REPEAT;
            int got = h_Y[m * N + n];

            if (got != expected) {
                printf("[BDS_v3] Mismatch at Y[%d, %d]: got %d, expected %d, gout=%d, gin=%d\n",
                       m, n, got, expected, gout, gin);
                return false;
            }
        }
    }

    return true;
}

int main() {
    printf("INT4 MMA Direct Intrinsics Dense vs BDS_v3\n");
    printf("M=%d, N=%d, K=%d, G=%d, IG=%d, OG=%d, TAU=%d, INNER_REPEAT=%d\n",
           M, N, K, G, IG, OG, TAU, INNER_REPEAT);

    printf("\nDense_v1 config:\n");
    printf("  block = 32 threads, 1 warp/block\n");
    printf("  grid  = (%d, %d)\n", N / 8, M / 8);
    printf("  total blocks = %d\n", (N / 8) * (M / 8));

    printf("\nBDS_v3 config:\n");
    printf("  block = 128 threads, 4 warps/block\n");
    printf("  grid  = (%d, %d, %d)\n", OG / 32, M / 8, G);
    printf("  total blocks = %d\n", (OG / 32) * (M / 8) * G);

    size_t bytes = size_t(M) * N * sizeof(int);

    int *d_C = nullptr;
    int *d_Y = nullptr;
    int *h_C = nullptr;
    int *h_Y = nullptr;

    h_C = reinterpret_cast<int *>(malloc(bytes));
    h_Y = reinterpret_cast<int *>(malloc(bytes));

    if (!h_C || !h_Y) {
        fprintf(stderr, "Host malloc failed\n");
        return -1;
    }

    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMalloc(&d_Y, bytes));

    CUDA_CHECK(cudaMemset(d_C, 0, bytes));
    CUDA_CHECK(cudaMemset(d_Y, 0, bytes));

    int repeat_launch = 200;

    float dense_ms = benchmark_dense_v1(d_C, repeat_launch);
    float bds_ms = benchmark_bds_v3(d_Y, repeat_launch);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, bytes, cudaMemcpyDeviceToHost));

    bool dense_pass = check_dense(h_C);
    bool bds_pass = check_bds_v3(h_Y);

    double dense_ops = 2.0 * double(M) * double(N) * double(K) * double(INNER_REPEAT);
    double bds_real_ops = dense_ops / double(G);

    double dense_tops = dense_ops / (dense_ms / 1000.0) / 1e12;
    double bds_real_tops = bds_real_ops / (bds_ms / 1000.0) / 1e12;
    double bds_dense_equiv_tops = dense_ops / (bds_ms / 1000.0) / 1e12;

    double speedup = dense_ms / bds_ms;

    printf("\n================ Results ================\n");

    printf("\nDense_v1 INT4 MMA\n");
    printf("Average latency: %.6f ms\n", dense_ms);
    printf("Dense TOPS: %.2f\n", dense_tops);
    printf("Check: %s\n", dense_pass ? "PASSED" : "FAILED");

    printf("\nBDS_v3 INT4 MMA\n");
    printf("Average latency: %.6f ms\n", bds_ms);
    printf("BDS real TOPS: %.2f\n", bds_real_tops);
    printf("BDS dense-equivalent TOPS: %.2f\n", bds_dense_equiv_tops);
    printf("Dense_v1 / BDS_v3 latency speedup: %.2fx\n", speedup);
    printf("Check: %s\n", bds_pass ? "PASSED" : "FAILED");

    printf("\nInterpretation:\n");
    printf("  Dense_v1 computes full K=1024 per output tile.\n");
    printf("  BDS_v3 computes IG=K/G=256 per output tile with G=4.\n");
    printf("  BDS_v3 also reduces block count by using 4 warps per CTA.\n");

    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_Y));

    free(h_C);
    free(h_Y);

    return (dense_pass && bds_pass) ? 0 : -1;
}