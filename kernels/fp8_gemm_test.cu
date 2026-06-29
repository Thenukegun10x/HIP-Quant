#include <hip/hip_runtime.h>
#include "../hip_quant_util.h"

// FP8 GEMM micro-benchmark using raw WMMA intrinsic on gfx12 (RDNA4).
//
// Uses __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12 directly,
// bypassing rocWMMA's fragment template layer.
//
// VGPR layout per thread (gfx12 WMMA, Wave32):
//   lane_wrapped = tid % 16  -> column for C/B, row for A
//   lane_group   = tid / 16  -> low/high half of the 8 packed K/result values
//
// Store layout:
//   each thread writes 8 C elements:
//     row = lane_group * 8 + i
//     col = lane_wrapped
//
// A operand packing:
//   A row = tile_m + lane_wrapped
//   A col = tile_k + lane_group * 8 + i
//
// B operand packing:
//   B row = tile_k + lane_group * 8 + i
//   B col = tile_n + lane_wrapped
//
// Input:  two FP8 E4M3 matrices A (MxK row-major) and B (KxN row-major)
// Output: float32 C (MxN row-major) = A * B
// Block:  1 wave (32 threads) per 16x16 output tile

typedef float v8f __attribute__((ext_vector_type(8)));
typedef int   v2i __attribute__((ext_vector_type(2)));

extern "C" __global__
__launch_bounds__(32, 1)
void fp8_gemm_wmma_kernel(
    const uint8_t * __restrict__ A_fp8,
    const uint8_t * __restrict__ B_fp8,
    float * __restrict__ C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc
) {
    int row_block = blockIdx.x;
    int col_block = blockIdx.y;
    int m = row_block * 16;
    int n = col_block * 16;
    if (m >= M || n >= N) return;

    int tid = threadIdx.x;
    if (tid >= 32) return;

    int lane_wrapped = tid & 15;    // column index 0..15
    int lane_group   = tid >> 4;    // row half 0..1

    v8f acc = (v8f){};

    for (int k = 0; k < K; k += 16) {
        int a_packed[2] = {0, 0};
        int b_packed[2] = {0, 0};

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int k_offset = lane_group * 8 + i;

            // A: WMMA packs A so lane_wrapped maps to M, element index maps to K.
            int a_row = m + lane_wrapped;
            int a_col = k + k_offset;
            ((uint8_t*)a_packed)[i] =
                (a_row < M && a_col < K) ? A_fp8[a_row * lda + a_col] : 0;

            // B: lane_wrapped maps to N, element index maps to K.
            int b_row = k + k_offset;
            int b_col = n + lane_wrapped;
            ((uint8_t*)b_packed)[i] =
                (b_row < K && b_col < N) ? B_fp8[b_row * ldb + b_col] : 0;
        }

        v2i a_vec = (v2i){ a_packed[0], a_packed[1] };
        v2i b_vec = (v2i){ b_packed[0], b_packed[1] };

        acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_vec, b_vec, acc);
    }

    // Store 8 result elements per thread (same VGPR layout)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int r = lane_group * 8 + i;
        int c = lane_wrapped;
        int g_row = m + r;
        int g_col = n + c;
        if (g_row < M && g_col < N) {
            C[g_row * ldc + g_col] = acc[i];
        }
    }
}
