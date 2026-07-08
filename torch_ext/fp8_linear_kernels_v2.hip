// torch_ext/fp8_linear_kernels_v2.hip
//
// Phase 1: Cooperative LDS staging with multi-K-tile amortised synchronisation.
//
//   blockDim  = 512  (WAVES_PER_BLOCK × 32)
//   A_lds     =  2 KB  [16][128+4]       (4 B row-pad avoids bank conflicts)
//   B_lds     = 34 KB  [256][128+4]
//   total LDS = 36 KB  < 64 KB → 1 block / CU (wave-limited anyway)
//
//   K_TILES_PER_STAGE = 8  →  128 K-elements per __syncthreads() pair
//   For K=4096: 32 stages × 2 syncs = 64 barriers (v1 had 0 barriers)
//
//   Savings: A loaded once per stage (16× less global traffic for A);
//            B accessed in coalesced 4-byte loads instead of 1-byte loads.
//
// Convention (identical to fp8_linear_kernels.hip):
//   A_fp8 : [M, K]  uint8_t    (activations, row-major)
//   B_fp8 : [N, K]  uint8_t    (weights,     row-major)
//   C     : [M, N]  float/half (output)

#include <hip/hip_runtime.h>
#include "../hip_quant_util.h"

#define TILE_M  16
#define TILE_N  16
#define TILE_K  16
#define WAVES_PER_BLOCK_V2 16
#define K_TILES_PER_STAGE  8
#define K_STAGE_ELEMS      (K_TILES_PER_STAGE * TILE_K)   // 128

#define HIP_QUANT_DTYPE_F32  0
#define HIP_QUANT_DTYPE_F16  1
#define HIP_QUANT_DTYPE_BF16 2

typedef float v8f __attribute__((ext_vector_type(8)));
typedef int   v2i __attribute__((ext_vector_type(2)));

static __device__ inline float load_float_input(const void* src, int dtype, int64_t idx) {
    if (dtype == HIP_QUANT_DTYPE_F16) {
        return fp16_to_fp32(((const uint16_t*)src)[idx]);
    }
    if (dtype == HIP_QUANT_DTYPE_BF16) {
        return bf16_to_fp32(((const uint16_t*)src)[idx]);
    }
    return ((const float*)src)[idx];
}

static __device__ inline void store_float_output(void* dst, int dtype, int64_t idx, float value) {
    if (dtype == HIP_QUANT_DTYPE_F16) {
        ((uint16_t*)dst)[idx] = fp32_to_fp16(value);
    } else if (dtype == HIP_QUANT_DTYPE_BF16) {
        ((uint16_t*)dst)[idx] = fp32_to_bf16(value);
    } else {
        ((float*)dst)[idx] = value;
    }
}

// LDS row stride = 128 + 4 = 132 bytes.
// The extra 4 bytes per row rotate the bank mapping so that consecutive
// rows land in consecutive LDS banks, avoiding 16-way bank conflicts
// when all 16 lanes of a wave read the same K-offset from different rows.
#define LDS_ROW_STRIDE  (K_STAGE_ELEMS + 4)

__global__
__launch_bounds__(WAVES_PER_BLOCK_V2 * 32, 2)
void fp8_gemm_v2_input_weight_fp8_kernel(
    const uint8_t * __restrict__ A_fp8,
    const uint8_t * __restrict__ B_fp8,
    void          * __restrict__ C,
    const void    * __restrict__ bias,
    int M, int N, int K,
    float out_scale,
    int c_dtype, int bias_dtype, bool has_bias
) {
    int wave = threadIdx.x >> 5;
    int tid  = threadIdx.x & 31;
    if (wave >= WAVES_PER_BLOCK_V2) return;

    int m = blockIdx.y * TILE_M;
    int n_tile = blockIdx.x * WAVES_PER_BLOCK_V2 + wave;
    int n = n_tile * TILE_N;
    bool valid_wave = (m < M && n < N);

    int lane_wrapped = tid & 15;
    int lane_group   = tid >> 4;

    v8f acc = (v8f){};

    __shared__ uint8_t A_lds[TILE_M][LDS_ROW_STRIDE];
    __shared__ uint8_t B_lds[WAVES_PER_BLOCK_V2 * TILE_N][LDS_ROW_STRIDE];

    int linear_tid = threadIdx.x;

    for (int k_start = 0; k_start < K; k_start += K_STAGE_ELEMS) {

        // ---------- cooperative load  A  ----------
        // 512 threads  ×  4 bytes = 2048 bytes = 16 rows × 128 K-elements
        {
            int a_row = linear_tid * 4 / K_STAGE_ELEMS;         // 0..15
            int a_col = (linear_tid * 4) % K_STAGE_ELEMS;       // 0,4,8,...,124
            int global_row = m + a_row;
            unsigned int a_val = 0;
            if (global_row < M) {
                int global_off = global_row * K + k_start + a_col;
                int limit = K - k_start - a_col;
                if (limit > 4) limit = 4;
                for (int b = 0; b < limit; b++) {
                    ((uint8_t*)&a_val)[b] = A_fp8[global_off + b];
                }
            }
            *((unsigned int*)&A_lds[a_row][a_col]) = a_val;
        }

        // ---------- cooperative load  B  ----------
        // 512 threads × 32 bytes (8 × uint32)  =  16,384 bytes per pass
        // 2 passes needed for 32,768 bytes (256 rows × 128 K)
        // Each thread loads 16 uint32 in a loop → 512 × 16 = 8192 uint32
        // = 32,768 bytes
        {
            for (int pass = 0; pass < 16; pass++) {
                int b_idx = pass * 512 + linear_tid;            // 0..8191
                int b_row = b_idx / (K_STAGE_ELEMS / 4);       // 0..255
                int b_col = (b_idx % (K_STAGE_ELEMS / 4)) * 4; // 0,4,...,124

                int wave_idx = b_row >> 4;                     // b_row / 16
                int n_off    = b_row & 15;                     // b_row % 16
                int global_n = (blockIdx.x * WAVES_PER_BLOCK_V2 + wave_idx) * TILE_N + n_off;
                unsigned int b_val = 0;
                if (global_n < N) {
                    int global_off = global_n * K + k_start + b_col;
                    int limit = K - k_start - b_col;
                    if (limit > 4) limit = 4;
                    for (int b = 0; b < limit; b++) {
                        ((uint8_t*)&b_val)[b] = B_fp8[global_off + b];
                    }
                }
                ((unsigned int*)&B_lds[b_row][b_col])[0] = b_val;
            }
        }

        __syncthreads();

        // ---------- compute  8  WMMA tiles from LDS ----------
        if (valid_wave) {
        for (int kt = 0; kt < K_TILES_PER_STAGE; kt++) {
            int a_packed[2] = {0, 0};
            int b_packed[2] = {0, 0};

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int k_offset = lane_group * 8 + i;
                ((uint8_t*)a_packed)[i] = A_lds[lane_wrapped][kt * TILE_K + k_offset];
                ((uint8_t*)b_packed)[i] = B_lds[wave * TILE_N + lane_wrapped][kt * TILE_K + k_offset];
            }

            v2i a_vec = (v2i){ a_packed[0], a_packed[1] };
            v2i b_vec = (v2i){ b_packed[0], b_packed[1] };
            acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_vec, b_vec, acc);
        }
        }

        __syncthreads();
    }

    // ---------- store  C  ----------
    if (valid_wave) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = m + lane_group * 8 + i;
        int col = n + lane_wrapped;
        if (row < M && col < N) {
            float value = acc[i] * out_scale;
            if (has_bias) value += load_float_input(bias, bias_dtype, col);
            store_float_output(C, c_dtype, (int64_t)row * N + col, value);
        }
    }
    }
}


// ---------------------------------------------------------------------------
// C-linkage launch helpers (called from pytorch_bindings_hip.cpp)
// ---------------------------------------------------------------------------
extern "C" {

void launch_fp8_linear_forward_v2_input_weight(
    const uint8_t* A_fp8, const uint8_t* B_fp8, void* C, const void* bias,
    int M, int N, int K,
    float input_scale, float weight_inv_scale,
    int c_dtype, int bias_dtype, bool has_bias,
    hipStream_t stream
) {
    dim3 block(WAVES_PER_BLOCK_V2 * 32);
    dim3 grid((N + TILE_N * WAVES_PER_BLOCK_V2 - 1) / (TILE_N * WAVES_PER_BLOCK_V2),
              (M + TILE_M - 1) / TILE_M);
    float out_scale = weight_inv_scale / input_scale;
    hipLaunchKernelGGL(fp8_gemm_v2_input_weight_fp8_kernel, grid, block, 0, stream,
                       A_fp8, B_fp8, C, bias, M, N, K, out_scale, c_dtype,
                       bias_dtype, has_bias);
}

} // extern "C"
