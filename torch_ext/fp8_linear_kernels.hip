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

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

typedef float v8f __attribute__((ext_vector_type(8)));
typedef int   v2i __attribute__((ext_vector_type(2)));

// ---------------------------------------------------------------------------
// WMMA GEMM: C = A * B^T  (A:[M,K], B:[N,K], C:[M,N])
// With on-the-fly FP8 E4M3 quantization in registers.
// ---------------------------------------------------------------------------
__global__
__launch_bounds__(32, 1)
void fp8_gemm_e4m3_kernel(
    const float * __restrict__ A,   // [M, K]
    const float * __restrict__ B,   // [N, K]  (weight, compute A @ B.T)
    float       * __restrict__ C,   // [M, N]
    int M, int N, int K
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
                (a_row < M && a_col < K) ? fp32_to_fp8_e4m3(A[a_row * K + a_col]) : 0;

            int b_row = k + k_offset;
            int b_col = n + lane_wrapped;
            ((uint8_t*)b_packed)[i] =
                (b_row < K && b_col < N) ? fp32_to_fp8_e4m3(B[b_col * K + b_row]) : 0;
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
            C[row * N + col] = acc[i];
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
    const float * __restrict__ grad_output, // [M, N]
    const float * __restrict__ weight,      // [N, K]
    float       * __restrict__ grad_input,  // [M, K]
    int M, int N, int K
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
                (a_row < M && a_col < N) ? fp32_to_fp8_e5m2(grad_output[a_row * N + a_col]) : 0;

            int b_row = n_tile + n_offset;
            int b_col = k_tile + lane_wrapped;
            ((uint8_t*)b_packed)[i] =
                (b_row < N && b_col < K) ? fp32_to_fp8_e5m2(weight[b_row * K + b_col]) : 0;
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
            grad_input[row * K + col] = acc[i];
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
    const float * __restrict__ grad_output, // [M, N]
    const float * __restrict__ input,       // [M, K]
    float       * __restrict__ grad_weight, // [N, K]
    int M, int N, int K
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
                (a_row < N && a_col < M) ? fp32_to_fp8_e5m2(grad_output[a_col * N + a_row]) : 0;

            int b_row = m_tile + m_offset;
            int b_col = k_tile + lane_wrapped;
            ((uint8_t*)b_packed)[i] =
                (b_row < M && b_col < K) ? fp32_to_fp8_e5m2(input[b_row * K + b_col]) : 0;
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
            grad_weight[row * K + col] = acc[i];
        }
    }
}

// ---------------------------------------------------------------------------
// C-linkage launch helpers called from pytorch_bindings.cpp
// ---------------------------------------------------------------------------
extern "C" {

void launch_fp8_linear_forward(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    hipStream_t stream
) {
    dim3 block(32);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    hipLaunchKernelGGL(fp8_gemm_e4m3_kernel, grid, block, 0, stream,
                       A, B, C, M, N, K);
}

void launch_fp8_linear_backward_input(
    const float* grad_output, const float* weight, float* grad_input,
    int M, int N, int K,
    hipStream_t stream
) {
    dim3 block(32);
    dim3 grid((K + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    hipLaunchKernelGGL(fp8_gemm_backward_input_kernel, grid, block, 0, stream,
                       grad_output, weight, grad_input, M, N, K);
}

void launch_fp8_linear_backward_weight(
    const float* grad_output, const float* input, float* grad_weight,
    int M, int N, int K,
    hipStream_t stream
) {
    dim3 block(32);
    dim3 grid((K + TILE_N - 1) / TILE_N, (N + TILE_M - 1) / TILE_M);
    hipLaunchKernelGGL(fp8_gemm_backward_weight_kernel, grid, block, 0, stream,
                       grad_output, input, grad_weight, M, N, K);
}

} // extern "C"
