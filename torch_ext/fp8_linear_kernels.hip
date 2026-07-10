// torch_ext/fp8_linear_kernels.hip
//
// Phase 4: FP8 Linear forward / backward kernels.
//
// Current forward implementation: one wave computes one 16x16 output tile
// using gfx12 FP8 WMMA.  Inputs are float32 master tensors; each loaded value
// is quantized to E4M3 in registers, packed into the WMMA operand layout, and
// accumulated in float32.
//
// Convention:
//   A   : [M, K]  float32   (activation or grad_output)
//   B   : [N, K]  float32   (weight, stored row-major so B.T is [K, N])
//   C   : [M, N]  float32   (output)
//
// Backward uses gfx12 BF8/E5M2 WMMA for both operands.  gfx12 exposes FP8/FP8
// and BF8/BF8 WMMA variants, not a mixed BF8/FP8 WMMA variant.

#include <hip/hip_runtime.h>
#include "../hip_quant_util.h"

#define HIP_QUANT_DTYPE_F32  0
#define HIP_QUANT_DTYPE_F16  1
#define HIP_QUANT_DTYPE_BF16 2

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16
#define WAVES_PER_BLOCK_N 16

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

// ---------------------------------------------------------------------------
// WMMA GEMM: C = A * B^T  (A:[M,K], B:[N,K], C:[M,N])
// With on-the-fly FP8 E4M3 quantization in registers.
// ---------------------------------------------------------------------------
__global__
__launch_bounds__(32, 1)
void fp8_gemm_e4m3_kernel(
    const void  * __restrict__ A,   // [M, K]
    const void  * __restrict__ B,   // [N, K]  (weight, compute A @ B.T)
    void        * __restrict__ C,   // [M, N]
    const void  * __restrict__ bias,
    int M, int N, int K,
    float a_scale, float b_scale, float out_scale,
    int a_dtype, int b_dtype, int c_dtype, int bias_dtype, bool has_bias
) {
    int m = blockIdx.y * TILE_M;
    int n = blockIdx.x * TILE_N;
    if (m >= M || n >= N) return;

    int tid = threadIdx.x;
    if (tid >= 32) return;

    int lane_wrapped = tid & 15;
    int lane_group   = tid >> 4;

    v8f acc = (v8f){};

    for (int k = 0; k < K; k += TILE_K) {
        int a_packed[2] = {0, 0};
        int b_packed[2] = {0, 0};

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int k_offset = lane_group * 8 + i;

            int a_row = m + lane_wrapped;
            int a_col = k + k_offset;
            ((uint8_t*)a_packed)[i] =
                (a_row < M && a_col < K) ? fp32_to_fp8_e4m3(load_float_input(A, a_dtype, a_row * K + a_col) * a_scale) : 0;

            int b_row = k + k_offset;
            int b_col = n + lane_wrapped;
            ((uint8_t*)b_packed)[i] =
                (b_row < K && b_col < N) ? fp32_to_fp8_e4m3(load_float_input(B, b_dtype, b_col * K + b_row) * b_scale) : 0;
        }

        v2i a_vec = (v2i){ a_packed[0], a_packed[1] };
        v2i b_vec = (v2i){ b_packed[0], b_packed[1] };
        acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_vec, b_vec, acc);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = m + lane_group * 8 + i;
        int col = n + lane_wrapped;
        if (row < M && col < N) {
            float value = acc[i] * out_scale;
            if (has_bias) value += load_float_input(bias, bias_dtype, col);
            store_float_output(C, c_dtype, row * N + col, value);
        }
    }
}

// Same forward GEMM, but B is already E4M3 bytes produced from weight * weight_scale.
__global__
__launch_bounds__(32, 1)
void fp8_gemm_e4m3_weight_fp8_kernel(
    const void    * __restrict__ A,       // [M, K]
    const uint8_t * __restrict__ B_fp8,   // [N, K]
    void          * __restrict__ C,       // [M, N]
    const void    * __restrict__ bias,
    int M, int N, int K,
    float a_scale, float out_scale,
    int a_dtype, int c_dtype, int bias_dtype, bool has_bias
) {
    int m = blockIdx.y * TILE_M;
    int n = blockIdx.x * TILE_N;
    if (m >= M || n >= N) return;

    int tid = threadIdx.x;
    if (tid >= 32) return;

    int lane_wrapped = tid & 15;
    int lane_group   = tid >> 4;

    v8f acc = (v8f){};

    for (int k = 0; k < K; k += TILE_K) {
        int a_packed[2] = {0, 0};
        int b_packed[2] = {0, 0};

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int k_offset = lane_group * 8 + i;

            int a_row = m + lane_wrapped;
            int a_col = k + k_offset;
            ((uint8_t*)a_packed)[i] =
                (a_row < M && a_col < K) ? fp32_to_fp8_e4m3(load_float_input(A, a_dtype, a_row * K + a_col) * a_scale) : 0;

            int b_row = k + k_offset;
            int b_col = n + lane_wrapped;
            ((uint8_t*)b_packed)[i] =
                (b_row < K && b_col < N) ? B_fp8[b_col * K + b_row] : 0;
        }

        v2i a_vec = (v2i){ a_packed[0], a_packed[1] };
        v2i b_vec = (v2i){ b_packed[0], b_packed[1] };
        acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_vec, b_vec, acc);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = m + lane_group * 8 + i;
        int col = n + lane_wrapped;
        if (row < M && col < N) {
            float value = acc[i] * out_scale;
            if (has_bias) value += load_float_input(bias, bias_dtype, col);
            store_float_output(C, c_dtype, row * N + col, value);
        }
    }
}

// Same forward GEMM, but A is already scaled E4M3 bytes and B is a master tensor.
__global__
__launch_bounds__(32, 1)
void fp8_gemm_e4m3_input_fp8_kernel(
    const uint8_t * __restrict__ A_fp8,   // [M, K]
    const void    * __restrict__ B,       // [N, K]
    void          * __restrict__ C,       // [M, N]
    const void    * __restrict__ bias,
    int M, int N, int K,
    float b_scale, float out_scale,
    int b_dtype, int c_dtype, int bias_dtype, bool has_bias
) {
    int m = blockIdx.y * TILE_M;
    int n = blockIdx.x * TILE_N;
    if (m >= M || n >= N) return;

    int tid = threadIdx.x;
    if (tid >= 32) return;

    int lane_wrapped = tid & 15;
    int lane_group   = tid >> 4;

    v8f acc = (v8f){};

    for (int k = 0; k < K; k += TILE_K) {
        int a_packed[2] = {0, 0};
        int b_packed[2] = {0, 0};

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int k_offset = lane_group * 8 + i;

            int a_row = m + lane_wrapped;
            int a_col = k + k_offset;
            ((uint8_t*)a_packed)[i] =
                (a_row < M && a_col < K) ? A_fp8[a_row * K + a_col] : 0;

            int b_row = k + k_offset;
            int b_col = n + lane_wrapped;
            ((uint8_t*)b_packed)[i] =
                (b_row < K && b_col < N) ? fp32_to_fp8_e4m3(load_float_input(B, b_dtype, b_col * K + b_row) * b_scale) : 0;
        }

        v2i a_vec = (v2i){ a_packed[0], a_packed[1] };
        v2i b_vec = (v2i){ b_packed[0], b_packed[1] };
        acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_vec, b_vec, acc);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = m + lane_group * 8 + i;
        int col = n + lane_wrapped;
        if (row < M && col < N) {
            float value = acc[i] * out_scale;
            if (has_bias) value += load_float_input(bias, bias_dtype, col);
            store_float_output(C, c_dtype, row * N + col, value);
        }
    }
}

// Correctness-first fused block-scaled FP8 linear:
// C[M,N] = dequant(A_fp8[M,K], A_scales[M,KB]) @ dequant(B_fp8[N,K], B_scales[N,KB]).T + bias
// This intentionally avoids gfx12 WMMA so the block-scale API can be validated
// independently before an optimized tiled implementation is added.
__global__
__launch_bounds__(256, 4)
void fp8_gemm_e4m3_blockwise_naive_kernel(
    const uint8_t * __restrict__ A_fp8,
    const float   * __restrict__ A_scales,
    const uint8_t * __restrict__ B_fp8,
    const float   * __restrict__ B_scales,
    void          * __restrict__ C,
    const void    * __restrict__ bias,
    int M, int N, int K,
    int blocks_per_row,
    int block_size,
    int c_dtype, int bias_dtype, bool has_bias
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)M * N;
    if (idx >= total) return;

    int m = (int)(idx / N);
    int n = (int)(idx - (int64_t)m * N);
    float acc = 0.0f;

    for (int kb = 0; kb < blocks_per_row; ++kb) {
        int k_start = kb * block_size;
        int k_end = k_start + block_size;
        if (k_end > K) k_end = K;
        float scale = A_scales[m * blocks_per_row + kb] * B_scales[n * blocks_per_row + kb];
        float partial = 0.0f;
        for (int k = k_start; k < k_end; ++k) {
            float av = fp8_e4m3_to_fp32(A_fp8[m * K + k]);
            float bv = fp8_e4m3_to_fp32(B_fp8[n * K + k]);
            partial += av * bv;
        }
        acc += partial * scale;
    }

    if (has_bias) acc += load_float_input(bias, bias_dtype, n);
    store_float_output(C, c_dtype, idx, acc);
}

// Same forward GEMM, but both A and B are already scaled E4M3 bytes.
__global__
__launch_bounds__(32 * WAVES_PER_BLOCK_N, 2)
void fp8_gemm_e4m3_input_weight_fp8_kernel(
    const uint8_t * __restrict__ A_fp8,   // [M, K]
    const uint8_t * __restrict__ B_fp8,   // [N, K]
    void          * __restrict__ C,       // [M, N]
    const void    * __restrict__ bias,
    int M, int N, int K,
    float out_scale,
    int c_dtype, int bias_dtype, bool has_bias
) {
    int wave = threadIdx.x >> 5;
    int tid = threadIdx.x & 31;
    if (wave >= WAVES_PER_BLOCK_N) return;

    int m = blockIdx.y * TILE_M;
    int n = (blockIdx.x * WAVES_PER_BLOCK_N + wave) * TILE_N;
    if (m >= M || n >= N) return;

    int lane_wrapped = tid & 15;
    int lane_group   = tid >> 4;

    v8f acc = (v8f){};

    for (int k = 0; k < K; k += TILE_K) {
        int a_packed[2] = {0, 0};
        int b_packed[2] = {0, 0};

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int k_offset = lane_group * 8 + i;

            int a_row = m + lane_wrapped;
            int a_col = k + k_offset;
            ((uint8_t*)a_packed)[i] =
                (a_row < M && a_col < K) ? A_fp8[a_row * K + a_col] : 0;

            int b_row = k + k_offset;
            int b_col = n + lane_wrapped;
            ((uint8_t*)b_packed)[i] =
                (b_row < K && b_col < N) ? B_fp8[b_col * K + b_row] : 0;
        }

        v2i a_vec = (v2i){ a_packed[0], a_packed[1] };
        v2i b_vec = (v2i){ b_packed[0], b_packed[1] };
        acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_vec, b_vec, acc);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = m + lane_group * 8 + i;
        int col = n + lane_wrapped;
        if (row < M && col < N) {
            float value = acc[i] * out_scale;
            if (has_bias) value += load_float_input(bias, bias_dtype, col);
            store_float_output(C, c_dtype, row * N + col, value);
        }
    }
}

__global__
__launch_bounds__(256, 4)
void pack_fp8_weight_wmma_kernel(
    const uint8_t * __restrict__ B_fp8,      // [N, K]
    uint8_t       * __restrict__ B_packed,   // [NT, KT, lane_group, lane, byte]
    int N, int K, int n_tiles, int k_tiles
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)n_tiles * k_tiles * TILE_K * TILE_N;
    if (idx >= total) return;

    int byte = (int)(idx & 7);
    int lane = (int)((idx >> 3) & 15);
    int lane_group = (int)((idx >> 7) & 1);
    int64_t tile = idx >> 8;
    int k_tile = (int)(tile % k_tiles);
    int n_tile = (int)(tile / k_tiles);

    int n = n_tile * TILE_N + lane;
    int k = k_tile * TILE_K + lane_group * 8 + byte;
    B_packed[idx] = (n < N && k < K) ? B_fp8[n * K + k] : 0;
}

// ---------------------------------------------------------------------------
// Backward input gradient: grad_input = grad_output * weight
// grad_output:[M,N] (E5M2), weight:[N,K] (E5M2), grad_input:[M,K]
// Computes:  grad_input[m,k] = sum_n grad_output[m,n] * weight[n,k]
// ---------------------------------------------------------------------------
__global__
__launch_bounds__(32, 1)
void fp8_gemm_backward_input_kernel(
    const void  * __restrict__ grad_output, // [M, N]
    const void  * __restrict__ weight,      // [N, K]
    void        * __restrict__ grad_input,  // [M, K]
    int M, int N, int K,
    float weight_scale, float out_scale,
    int grad_output_dtype, int weight_dtype, int grad_input_dtype
) {
    int m = blockIdx.y * TILE_M;
    int k_tile = blockIdx.x * TILE_N;
    if (m >= M || k_tile >= K) return;

    int tid = threadIdx.x;
    if (tid >= 32) return;

    int lane_wrapped = tid & 15;
    int lane_group   = tid >> 4;

    v8f acc = (v8f){};

    for (int n_tile = 0; n_tile < N; n_tile += TILE_K) {
        int a_packed[2] = {0, 0};
        int b_packed[2] = {0, 0};

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int n_offset = lane_group * 8 + i;

            int a_row = m + lane_wrapped;
            int a_col = n_tile + n_offset;
            ((uint8_t*)a_packed)[i] =
                (a_row < M && a_col < N) ? fp32_to_fp8_e5m2(load_float_input(grad_output, grad_output_dtype, a_row * N + a_col)) : 0;

            int b_row = n_tile + n_offset;
            int b_col = k_tile + lane_wrapped;
            ((uint8_t*)b_packed)[i] =
                (b_row < N && b_col < K) ? fp32_to_fp8_e5m2(load_float_input(weight, weight_dtype, b_row * K + b_col) * weight_scale) : 0;
        }

        v2i a_vec = (v2i){ a_packed[0], a_packed[1] };
        v2i b_vec = (v2i){ b_packed[0], b_packed[1] };
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a_vec, b_vec, acc);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = m + lane_group * 8 + i;
        int col = k_tile + lane_wrapped;
        if (row < M && col < K) {
            store_float_output(grad_input, grad_input_dtype, row * K + col, acc[i] * out_scale);
        }
    }
}

// Backward input gradient with grad_output already quantized to E5M2/BF8 bytes.
__global__
__launch_bounds__(32, 1)
void fp8_gemm_backward_input_fp8_grad_kernel(
    const uint8_t * __restrict__ grad_output_fp8, // [M, N]
    const void    * __restrict__ weight,          // [N, K]
    void          * __restrict__ grad_input,      // [M, K]
    int M, int N, int K,
    float weight_scale, float out_scale,
    int weight_dtype, int grad_input_dtype
) {
    int m = blockIdx.y * TILE_M;
    int k_tile = blockIdx.x * TILE_N;
    if (m >= M || k_tile >= K) return;

    int tid = threadIdx.x;
    if (tid >= 32) return;

    int lane_wrapped = tid & 15;
    int lane_group   = tid >> 4;

    v8f acc = (v8f){};

    for (int n_tile = 0; n_tile < N; n_tile += TILE_K) {
        int a_packed[2] = {0, 0};
        int b_packed[2] = {0, 0};

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int n_offset = lane_group * 8 + i;

            int a_row = m + lane_wrapped;
            int a_col = n_tile + n_offset;
            ((uint8_t*)a_packed)[i] =
                (a_row < M && a_col < N) ? grad_output_fp8[a_row * N + a_col] : 0;

            int b_row = n_tile + n_offset;
            int b_col = k_tile + lane_wrapped;
            ((uint8_t*)b_packed)[i] =
                (b_row < N && b_col < K) ? fp32_to_fp8_e5m2(load_float_input(weight, weight_dtype, b_row * K + b_col) * weight_scale) : 0;
        }

        v2i a_vec = (v2i){ a_packed[0], a_packed[1] };
        v2i b_vec = (v2i){ b_packed[0], b_packed[1] };
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a_vec, b_vec, acc);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = m + lane_group * 8 + i;
        int col = k_tile + lane_wrapped;
        if (row < M && col < K) {
            store_float_output(grad_input, grad_input_dtype, row * K + col, acc[i] * out_scale);
        }
    }
}

// ---------------------------------------------------------------------------
// Backward weight gradient: grad_weight = grad_output^T * input
// grad_output:[M,N] (E5M2), input:[M,K] (E5M2), grad_weight:[N,K]
// Computes:  grad_weight[n,k] = sum_m grad_output[m,n] * input[m,k]
// ---------------------------------------------------------------------------
__global__
__launch_bounds__(32, 1)
void fp8_gemm_backward_weight_kernel(
    const void  * __restrict__ grad_output, // [M, N]
    const void  * __restrict__ input,       // [M, K]
    void        * __restrict__ grad_weight, // [N, K]
    int M, int N, int K,
    float input_scale, float out_scale,
    int grad_output_dtype, int input_dtype, int grad_weight_dtype
) {
    int n = blockIdx.y * TILE_M;
    int k_tile = blockIdx.x * TILE_N;
    if (n >= N || k_tile >= K) return;

    int tid = threadIdx.x;
    if (tid >= 32) return;

    int lane_wrapped = tid & 15;
    int lane_group   = tid >> 4;

    v8f acc = (v8f){};

    for (int m_tile = 0; m_tile < M; m_tile += TILE_K) {
        int a_packed[2] = {0, 0};
        int b_packed[2] = {0, 0};

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int m_offset = lane_group * 8 + i;

            int a_row = n + lane_wrapped;
            int a_col = m_tile + m_offset;
            ((uint8_t*)a_packed)[i] =
                (a_row < N && a_col < M) ? fp32_to_fp8_e5m2(load_float_input(grad_output, grad_output_dtype, a_col * N + a_row)) : 0;

            int b_row = m_tile + m_offset;
            int b_col = k_tile + lane_wrapped;
            ((uint8_t*)b_packed)[i] =
                (b_row < M && b_col < K) ? fp32_to_fp8_e5m2(load_float_input(input, input_dtype, b_row * K + b_col) * input_scale) : 0;
        }

        v2i a_vec = (v2i){ a_packed[0], a_packed[1] };
        v2i b_vec = (v2i){ b_packed[0], b_packed[1] };
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a_vec, b_vec, acc);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = n + lane_group * 8 + i;
        int col = k_tile + lane_wrapped;
        if (row < N && col < K) {
            store_float_output(grad_weight, grad_weight_dtype, row * K + col, acc[i] * out_scale);
        }
    }
}

// Backward weight gradient with grad_output already quantized to E5M2/BF8 bytes.
__global__
__launch_bounds__(32, 1)
void fp8_gemm_backward_weight_fp8_grad_kernel(
    const uint8_t * __restrict__ grad_output_fp8, // [M, N]
    const void    * __restrict__ input,           // [M, K]
    void          * __restrict__ grad_weight,     // [N, K]
    int M, int N, int K,
    float input_scale, float out_scale,
    int input_dtype, int grad_weight_dtype
) {
    int n = blockIdx.y * TILE_M;
    int k_tile = blockIdx.x * TILE_N;
    if (n >= N || k_tile >= K) return;

    int tid = threadIdx.x;
    if (tid >= 32) return;

    int lane_wrapped = tid & 15;
    int lane_group   = tid >> 4;

    v8f acc = (v8f){};

    for (int m_tile = 0; m_tile < M; m_tile += TILE_K) {
        int a_packed[2] = {0, 0};
        int b_packed[2] = {0, 0};

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int m_offset = lane_group * 8 + i;

            int a_row = n + lane_wrapped;
            int a_col = m_tile + m_offset;
            ((uint8_t*)a_packed)[i] =
                (a_row < N && a_col < M) ? grad_output_fp8[a_col * N + a_row] : 0;

            int b_row = m_tile + m_offset;
            int b_col = k_tile + lane_wrapped;
            ((uint8_t*)b_packed)[i] =
                (b_row < M && b_col < K) ? fp32_to_fp8_e5m2(load_float_input(input, input_dtype, b_row * K + b_col) * input_scale) : 0;
        }

        v2i a_vec = (v2i){ a_packed[0], a_packed[1] };
        v2i b_vec = (v2i){ b_packed[0], b_packed[1] };
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a_vec, b_vec, acc);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = n + lane_group * 8 + i;
        int col = k_tile + lane_wrapped;
        if (row < N && col < K) {
            store_float_output(grad_weight, grad_weight_dtype, row * K + col, acc[i] * out_scale);
        }
    }
}

// ---------------------------------------------------------------------------
// C-linkage launch helpers called from pytorch_bindings.cpp
// ---------------------------------------------------------------------------
extern "C" {

void launch_fp8_linear_forward(
    const void* A, const void* B, void* C, const void* bias,
    int M, int N, int K,
    int a_dtype, int b_dtype, int c_dtype, int bias_dtype, bool has_bias,
    hipStream_t stream
) {
    dim3 block(32);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    hipLaunchKernelGGL(fp8_gemm_e4m3_kernel, grid, block, 0, stream,
                       A, B, C, bias, M, N, K, 1.0f, 1.0f, 1.0f,
                       a_dtype, b_dtype, c_dtype, bias_dtype, has_bias);
}

void launch_fp8_linear_forward_scaled(
    const void* A, const void* B, void* C, const void* bias,
    int M, int N, int K,
    float input_scale, float weight_scale,
    int a_dtype, int b_dtype, int c_dtype, int bias_dtype, bool has_bias,
    hipStream_t stream
) {
    dim3 block(32);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    float out_scale = 1.0f / (input_scale * weight_scale);
    hipLaunchKernelGGL(fp8_gemm_e4m3_kernel, grid, block, 0, stream,
                       A, B, C, bias, M, N, K, input_scale, weight_scale, out_scale,
                       a_dtype, b_dtype, c_dtype, bias_dtype, has_bias);
}

void launch_fp8_linear_forward_fp8_weight(
    const void* A, const uint8_t* B_fp8, void* C, const void* bias,
    int M, int N, int K,
    float input_scale, float weight_inv_scale,
    int a_dtype, int c_dtype, int bias_dtype, bool has_bias,
    hipStream_t stream
) {
    dim3 block(32);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    float out_scale = weight_inv_scale / input_scale;
    hipLaunchKernelGGL(fp8_gemm_e4m3_weight_fp8_kernel, grid, block, 0, stream,
                       A, B_fp8, C, bias, M, N, K, input_scale, out_scale,
                       a_dtype, c_dtype, bias_dtype, has_bias);
}

void launch_fp8_linear_forward_fp8_input(
    const uint8_t* A_fp8, const void* B, void* C, const void* bias,
    int M, int N, int K,
    float input_scale, float weight_scale,
    int b_dtype, int c_dtype, int bias_dtype, bool has_bias,
    hipStream_t stream
) {
    dim3 block(32);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    float out_scale = 1.0f / (input_scale * weight_scale);
    hipLaunchKernelGGL(fp8_gemm_e4m3_input_fp8_kernel, grid, block, 0, stream,
                       A_fp8, B, C, bias, M, N, K, weight_scale, out_scale,
                       b_dtype, c_dtype, bias_dtype, has_bias);
}

void launch_fp8_linear_forward_fp8_input_weight(
    const uint8_t* A_fp8, const uint8_t* B_fp8, void* C, const void* bias,
    int M, int N, int K,
    float input_scale, float weight_inv_scale,
    int c_dtype, int bias_dtype, bool has_bias,
    hipStream_t stream
) {
    dim3 block(32 * WAVES_PER_BLOCK_N);
    dim3 grid((N + TILE_N * WAVES_PER_BLOCK_N - 1) / (TILE_N * WAVES_PER_BLOCK_N),
              (M + TILE_M - 1) / TILE_M);
    float out_scale = weight_inv_scale / input_scale;
    hipLaunchKernelGGL(fp8_gemm_e4m3_input_weight_fp8_kernel, grid, block, 0, stream,
                       A_fp8, B_fp8, C, bias, M, N, K, out_scale, c_dtype,
                       bias_dtype, has_bias);
}

void launch_pack_fp8_weight_for_wmma(
    const uint8_t* B_fp8, uint8_t* B_packed,
    int N, int K, int n_tiles, int k_tiles,
    hipStream_t stream
) {
    int64_t total = (int64_t)n_tiles * k_tiles * TILE_K * TILE_N;
    if (total == 0) return;
    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(pack_fp8_weight_wmma_kernel,
                       dim3((uint32_t)blocks), dim3(threads), 0, stream,
                       B_fp8, B_packed, N, K, n_tiles, k_tiles);
}

void launch_fp8_linear_forward_blockwise(
    const uint8_t* A_fp8, const float* A_scales,
    const uint8_t* B_fp8, const float* B_scales,
    void* C, const void* bias,
    int M, int N, int K, int blocks_per_row, int block_size,
    int c_dtype, int bias_dtype, bool has_bias,
    hipStream_t stream
) {
    int64_t total = (int64_t)M * N;
    if (total == 0) return;
    int threads = 256;
    int64_t blocks64 = (total + threads - 1) / threads;
    hipLaunchKernelGGL(fp8_gemm_e4m3_blockwise_naive_kernel,
                       dim3((uint32_t)blocks64), dim3(threads), 0, stream,
                       A_fp8, A_scales, B_fp8, B_scales, C, bias,
                       M, N, K, blocks_per_row, block_size,
                       c_dtype, bias_dtype, has_bias);
}

void launch_fp8_linear_backward_input(
    const void* grad_output, const void* weight, void* grad_input,
    int M, int N, int K,
    int grad_output_dtype, int weight_dtype, int grad_input_dtype,
    hipStream_t stream
) {
    dim3 block(32);
    dim3 grid((K + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    hipLaunchKernelGGL(fp8_gemm_backward_input_kernel, grid, block, 0, stream,
                       grad_output, weight, grad_input, M, N, K, 1.0f, 1.0f,
                       grad_output_dtype, weight_dtype, grad_input_dtype);
}

void launch_fp8_linear_backward_input_scaled(
    const void* grad_output, const void* weight, void* grad_input,
    int M, int N, int K,
    float weight_scale,
    int grad_output_dtype, int weight_dtype, int grad_input_dtype,
    hipStream_t stream
) {
    dim3 block(32);
    dim3 grid((K + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    float out_scale = 1.0f / weight_scale;
    hipLaunchKernelGGL(fp8_gemm_backward_input_kernel, grid, block, 0, stream,
                       grad_output, weight, grad_input, M, N, K, weight_scale, out_scale,
                       grad_output_dtype, weight_dtype, grad_input_dtype);
}

void launch_fp8_linear_backward_weight(
    const void* grad_output, const void* input, void* grad_weight,
    int M, int N, int K,
    int grad_output_dtype, int input_dtype, int grad_weight_dtype,
    hipStream_t stream
) {
    dim3 block(32);
    dim3 grid((K + TILE_N - 1) / TILE_N, (N + TILE_M - 1) / TILE_M);
    hipLaunchKernelGGL(fp8_gemm_backward_weight_kernel, grid, block, 0, stream,
                       grad_output, input, grad_weight, M, N, K, 1.0f, 1.0f,
                       grad_output_dtype, input_dtype, grad_weight_dtype);
}

void launch_fp8_linear_backward_weight_scaled(
    const void* grad_output, const void* input, void* grad_weight,
    int M, int N, int K,
    float input_scale,
    int grad_output_dtype, int input_dtype, int grad_weight_dtype,
    hipStream_t stream
) {
    dim3 block(32);
    dim3 grid((K + TILE_N - 1) / TILE_N, (N + TILE_M - 1) / TILE_M);
    float out_scale = 1.0f / input_scale;
    hipLaunchKernelGGL(fp8_gemm_backward_weight_kernel, grid, block, 0, stream,
                       grad_output, input, grad_weight, M, N, K, input_scale, out_scale,
                       grad_output_dtype, input_dtype, grad_weight_dtype);
}

void launch_fp8_linear_backward_input_fp8_grad(
    const uint8_t* grad_output_fp8, const void* weight, void* grad_input,
    int M, int N, int K,
    float weight_scale,
    int weight_dtype, int grad_input_dtype,
    hipStream_t stream
) {
    dim3 block(32);
    dim3 grid((K + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    float out_scale = 1.0f / weight_scale;
    hipLaunchKernelGGL(fp8_gemm_backward_input_fp8_grad_kernel, grid, block, 0, stream,
                       grad_output_fp8, weight, grad_input, M, N, K, weight_scale, out_scale,
                       weight_dtype, grad_input_dtype);
}

void launch_fp8_linear_backward_weight_fp8_grad(
    const uint8_t* grad_output_fp8, const void* input, void* grad_weight,
    int M, int N, int K,
    float input_scale,
    int input_dtype, int grad_weight_dtype,
    hipStream_t stream
) {
    dim3 block(32);
    dim3 grid((K + TILE_N - 1) / TILE_N, (N + TILE_M - 1) / TILE_M);
    float out_scale = 1.0f / input_scale;
    hipLaunchKernelGGL(fp8_gemm_backward_weight_fp8_grad_kernel, grid, block, 0, stream,
                       grad_output_fp8, input, grad_weight, M, N, K, input_scale, out_scale,
                       input_dtype, grad_weight_dtype);
}

} // extern "C"
