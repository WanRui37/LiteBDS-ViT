// int4_mma_compute_bound_v2.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(1);                                                   \
    }                                                              \
} while (0)

#define CUTLASS_CHECK(call) do {                                   \
    cutlass::Status status = call;                                 \
    if (status != cutlass::Status::kSuccess) {                     \
        fprintf(stderr, "CUTLASS error %s:%d: %d\n",             \
                __FILE__, __LINE__, int(status));                  \
        exit(1);                                                   \
    }                                                              \
} while (0)

constexpr int M = 128;
constexpr int N = 128;
constexpr int K = 128;

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
static_assert(K % 8 == 0, "K must be multiple of 8 for int4 packing");

constexpr int K_PACK = K / 8;

using PackedInt4 = uint32_t;

using CutlassElementInputA = cutlass::int4b_t;
using CutlassElementInputB = cutlass::int4b_t;
using CutlassElementOutput = int32_t;
using CutlassElementAccumulator = int32_t;
using CutlassElementCompute = int32_t;

using CutlassLayoutA = cutlass::layout::RowMajor;
using CutlassLayoutB = cutlass::layout::ColumnMajor;
using CutlassLayoutC = cutlass::layout::RowMajor;

// 4 warps per CTA: (64x64) / (32x32) => 2x2 warps
using CutlassGemm = cutlass::gemm::device::Gemm<
    CutlassElementInputA,
    CutlassLayoutA,
    CutlassElementInputB,
    CutlassLayoutB,
    CutlassElementOutput,
    CutlassLayoutC,
    CutlassElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<32, 32, 128>,
    cutlass::gemm::GemmShape<8, 8, 32>,
    cutlass::epilogue::thread::LinearCombinationClamp<
        CutlassElementOutput,
        128 / cutlass::sizeof_bits<CutlassElementOutput>::value,
        CutlassElementAccumulator,
        CutlassElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2>;

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

static inline PackedInt4 pack_int4x8(const int8_t *src) {
    PackedInt4 packed = 0;
    for (int i = 0; i < 8; ++i) {
        PackedInt4 nibble = static_cast<PackedInt4>(src[i]) & 0xFu;
        packed |= (nibble << (4 * i));
    }
    return packed;
}

// ============================================================
// BDS_v4
// G = 4
// 一个 block = 4 个 warp
// 每个 warp 仍然计算一个 8x8 tile
// 一个 block 计算 group 内连续 4 个 8x8 tile，即 8x32
//
// grid = (OG/32, M/8, G)
// block = 128 threads
// ============================================================
__global__ void bds_v4_kernel(const PackedInt4 *A_packed, const PackedInt4 *B_packed, int *Y) {
    int tid = threadIdx.x;
    int warp_id = tid >> 5;     // 0,1,2,3
    int lane = tid & 31;

    int gout = blockIdx.z;
    int tile_m = blockIdx.y * 8;

    int tile_n_in_group = blockIdx.x * 32 + warp_id * 8;

    int acc0 = 0;
    int acc1 = 0;

#pragma unroll 1
    for (int r = 0; r < INNER_REPEAT; ++r) {
#pragma unroll
        for (int kk = 0; kk < IG; kk += 32) {
            int row = lane / 4;
            int col_pair = (lane % 4) * 2;
            int col_group = lane & 3;
            int row_group = lane & 3;

            int a_pack_idx = (tile_m + row) * K_PACK + (kk >> 3) + col_group;
            int b_pack_idx = (gout * OG + tile_n_in_group + col_pair) * K_PACK + (kk >> 3) + row_group;

            unsigned a_frag = A_packed[a_pack_idx];
            unsigned b_frag = B_packed[b_pack_idx];
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

float benchmark_bds_v4(const PackedInt4 *d_A, const PackedInt4 *d_B, int *d_Y, int repeat_launch) {
    dim3 block(128);
    dim3 grid(OG / 32, M / 8, G);

    for (int i = 0; i < 10; ++i) {
        bds_v4_kernel<<<grid, block>>>(d_A, d_B, d_Y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat_launch; ++i) {
        bds_v4_kernel<<<grid, block>>>(d_A, d_B, d_Y);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / repeat_launch;
}

bool check_bds_v4(const int *h_Y) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int expected = IG * INNER_REPEAT;
            int got = h_Y[m * N + n];

            if (got != expected) {
                printf("[BDS_v4] Mismatch at Y[%d, %d]: got %d, expected %d\n",
                       m, n, got, expected);
                return false;
            }
        }
    }

    return true;
}

int main() {
        printf("INT4 MMA Direct Intrinsics BDS_v4\n");
    printf("M=%d, N=%d, K=%d, G=%d, IG=%d, OG=%d, TAU=%d, INNER_REPEAT=%d\n",
           M, N, K, G, IG, OG, TAU, INNER_REPEAT);

    printf("\nBDS_v4 config:\n");
    printf("  block = 128 threads, 4 warps/block\n");
    printf("  grid  = (%d, %d, %d)\n", OG / 32, M / 8, G);
    printf("  total blocks = %d\n", (OG / 32) * (M / 8) * G);


    size_t bytes = size_t(M) * N * sizeof(int);

    int *d_Y = nullptr;
    PackedInt4 *d_A = nullptr;
    PackedInt4 *d_B = nullptr;
    int *h_Y = nullptr;
    PackedInt4 *h_A = nullptr;
    PackedInt4 *h_B = nullptr;

    h_Y = reinterpret_cast<int *>(malloc(bytes));
    h_A = reinterpret_cast<PackedInt4 *>(malloc(size_t(M) * K_PACK * sizeof(PackedInt4)));
    h_B = reinterpret_cast<PackedInt4 *>(malloc(size_t(N) * K_PACK * sizeof(PackedInt4)));

    if (!h_Y || !h_A || !h_B) {
        fprintf(stderr, "Host malloc failed\n");
        return -1;
    }

    // Pack A as row-major int4 tiles: A_packed[m, kpack]
    for (int m = 0; m < M; ++m) {
        for (int kp = 0; kp < K_PACK; ++kp) {
            int8_t tmp[8];
            for (int i = 0; i < 8; ++i) {
                tmp[i] = 1;
            }
            h_A[m * K_PACK + kp] = pack_int4x8(tmp);
        }
    }
    // Pack B as column-major int4 tiles: B_packed[n, kpack]
    for (int n = 0; n < N; ++n) {
        for (int kp = 0; kp < K_PACK; ++kp) {
            int8_t tmp[8];
            for (int i = 0; i < 8; ++i) {
                tmp[i] = 1;
            }
            h_B[n * K_PACK + kp] = pack_int4x8(tmp);
        }
    }

    CUDA_CHECK(cudaMalloc(&d_Y, bytes));
    CUDA_CHECK(cudaMalloc(&d_A, size_t(M) * K_PACK * sizeof(PackedInt4)));
    CUDA_CHECK(cudaMalloc(&d_B, size_t(N) * K_PACK * sizeof(PackedInt4)));

    CUDA_CHECK(cudaMemset(d_Y, 0, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_t(M) * K_PACK * sizeof(PackedInt4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_t(N) * K_PACK * sizeof(PackedInt4), cudaMemcpyHostToDevice));

    int repeat_launch = 20;

    float bds_ms = benchmark_bds_v4(d_A, d_B, d_Y, repeat_launch);

    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, bytes, cudaMemcpyDeviceToHost));
    bool bds_pass = check_bds_v4(h_Y);

    double dense_ops = 2.0 * double(M) * double(N) * double(K) * double(INNER_REPEAT);
    double bds_real_ops = dense_ops / double(G);
    double bds_real_tops = bds_real_ops / (bds_ms / 1000.0) / 1e12;
    double bds_dense_equiv_tops = dense_ops / (bds_ms / 1000.0) / 1e12;

    printf("\n================ Results ================\n");

    printf("\nBDS_v4 INT4 MMA\n");
    printf("Average latency: %.6f ms\n", bds_ms);
    printf("BDS real TOPS: %.2f\n", bds_real_tops);
    printf("BDS dense-equivalent TOPS: %.2f\n", bds_dense_equiv_tops);
    printf("Check: %s\n", bds_pass ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    free(h_Y);
    free(h_A);
    free(h_B);

    return bds_pass ? 0 : -1;
}