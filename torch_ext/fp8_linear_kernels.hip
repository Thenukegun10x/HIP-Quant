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
    int M, int N, int K,
    float a_scale, float b_scale, float out_scale,
    int a_dtype, int b_dtype, int c_dtype
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
            store_float_output(C, c_dtype, row * N + col, acc[i] * out_scale);
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
    int M, int N, int K,
    float a_scale, float out_scale,
    int a_dtype, int c_dtype
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
            store_float_output(C, c_dtype, row * N + col, acc[i] * out_scale);
        }
    }
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

// ---------------------------------------------------------------------------
// C-linkage launch helpers called from pytorch_bindings.cpp
// ---------------------------------------------------------------------------
extern "C" {

void launch_fp8_linear_forward(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    int a_dtype, int b_dtype, int c_dtype,
    hipStream_t stream
) {
    dim3 block(32);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    hipLaunchKernelGGL(fp8_gemm_e4m3_kernel, grid, block, 0, stream,
                       A, B, C, M, N, K, 1.0f, 1.0f, 1.0f, a_dtype, b_dtype, c_dtype);
}

void launch_fp8_linear_forward_scaled(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float input_scale, float weight_scale,
    int a_dtype, int b_dtype, int c_dtype,
    hipStream_t stream
) {
    dim3 block(32);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    float out_scale = 1.0f / (input_scale * weight_scale);
    hipLaunchKernelGGL(fp8_gemm_e4m3_kernel, grid, block, 0, stream,
                       A, B, C, M, N, K, input_scale, weight_scale, out_scale, a_dtype, b_dtype, c_dtype);
}

void launch_fp8_linear_forward_fp8_weight(
    const void* A, const uint8_t* B_fp8, void* C,
    int M, int N, int K,
    float input_scale, float weight_inv_scale,
    int a_dtype, int c_dtype,
    hipStream_t stream
) {
    dim3 block(32);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    float out_scale = weight_inv_scale / input_scale;
    hipLaunchKernelGGL(fp8_gemm_e4m3_weight_fp8_kernel, grid, block, 0, stream,
                       A, B_fp8, C, M, N, K, input_scale, out_scale, a_dtype, c_dtype);
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

} // extern "C"
