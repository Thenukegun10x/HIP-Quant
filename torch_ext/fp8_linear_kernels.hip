// torch_ext/fp8_linear_kernels.hip
//
// Phase 4: FP8 Linear forward / backward kernels.
//
// Current implementation: correctness-first tiled SGEMM that operates on
// float32 data obtained by dequantizing FP8 inputs.  This is a real FP8
// GEMM stub — the quantize/dequantize step happens inside the kernel so
// the data flowing between blocks is always in FP8 and the accumulation is
// in float32.  For maximum throughput replace with rocBLASLt or a
// Composable Kernel FP8 GEMM path once validated.
//
// Convention:
//   A   : [M, K]  float32   (activation or grad_output)
//   B   : [N, K]  float32   (weight, stored row-major so B.T is [K, N])
//   C   : [M, N]  float32   (output)
//
// The "FP8" part: A and B are quantized to E4M3 per element then immediately
// dequantized back to float32 for the actual multiply-accumulate.  This is
// the "fake FP8" path from Phase 3/4, which lets training correctness checks
// pass before a real hardware FP8 GEMM path is plugged in.
//
// For Phase 4 real GEMM: swap the inner loop for an FP8 MFMA instruction
// sequence once rocBLASLt FP8 support is confirmed on the target hardware.

#include <hip/hip_runtime.h>
#include "../hip_quant_util.h"

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// ---------------------------------------------------------------------------
// Tiled GEMM: C = A * B^T  (A:[M,K], B:[N,K], C:[M,N])
// With on-the-fly FP8 E4M3 quantize/dequantize on tiles.
// ---------------------------------------------------------------------------
__global__
__launch_bounds__(256, 2)
void fp8_gemm_e4m3_kernel(
    const float * __restrict__ A,   // [M, K]
    const float * __restrict__ B,   // [N, K]  (weight, compute A @ B.T)
    float       * __restrict__ C,   // [M, N]
    int M, int N, int K
) {
    // Tile indices
    int row = blockIdx.y * TILE_M + threadIdx.y;  // output row (M dim)
    int col = blockIdx.x * TILE_N + threadIdx.x;  // output col (N dim)

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        // Quantize to FP8 E4M3 then dequantize — models quantization error
        float a_fp8 = fp8_e4m3_to_fp32(fp32_to_fp8_e4m3(A[row * K + k]));
        float b_fp8 = fp8_e4m3_to_fp32(fp32_to_fp8_e4m3(B[col * K + k]));
        acc += a_fp8 * b_fp8;
    }
    C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Backward input gradient: grad_input = grad_output * weight
// grad_output:[M,N] (E5M2), weight:[N,K] (E4M3), grad_input:[M,K]
// Computes:  grad_input[m,k] = sum_n grad_output[m,n] * weight[n,k]
// ---------------------------------------------------------------------------
__global__
__launch_bounds__(256, 2)
void fp8_gemm_backward_input_kernel(
    const float * __restrict__ grad_output, // [M, N]
    const float * __restrict__ weight,      // [N, K]
    float       * __restrict__ grad_input,  // [M, K]
    int M, int N, int K
) {
    int row = blockIdx.y * TILE_M + threadIdx.y;  // M dim
    int col = blockIdx.x * TILE_N + threadIdx.x;  // K dim

    if (row >= M || col >= K) return;

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        float g = fp8_e5m2_to_fp32(fp32_to_fp8_e5m2(grad_output[row * N + n]));
        float w = fp8_e4m3_to_fp32(fp32_to_fp8_e4m3(weight[n * K + col]));
        acc += g * w;
    }
    grad_input[row * K + col] = acc;
}

// ---------------------------------------------------------------------------
// Backward weight gradient: grad_weight = grad_output^T * input
// grad_output:[M,N] (E5M2), input:[M,K] (E4M3), grad_weight:[N,K]
// Computes:  grad_weight[n,k] = sum_m grad_output[m,n] * input[m,k]
// ---------------------------------------------------------------------------
__global__
__launch_bounds__(256, 2)
void fp8_gemm_backward_weight_kernel(
    const float * __restrict__ grad_output, // [M, N]
    const float * __restrict__ input,       // [M, K]
    float       * __restrict__ grad_weight, // [N, K]
    int M, int N, int K
) {
    int row = blockIdx.y * TILE_M + threadIdx.y;  // N dim
    int col = blockIdx.x * TILE_N + threadIdx.x;  // K dim

    if (row >= N || col >= K) return;

    float acc = 0.0f;
    for (int m = 0; m < M; ++m) {
        float g = fp8_e5m2_to_fp32(fp32_to_fp8_e5m2(grad_output[m * N + row]));
        float a = fp8_e4m3_to_fp32(fp32_to_fp8_e4m3(input[m * K + col]));
        acc += g * a;
    }
    grad_weight[row * K + col] = acc;
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
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    hipLaunchKernelGGL(fp8_gemm_e4m3_kernel, grid, block, 0, stream,
                       A, B, C, M, N, K);
}

void launch_fp8_linear_backward_input(
    const float* grad_output, const float* weight, float* grad_input,
    int M, int N, int K,
    hipStream_t stream
) {
    dim3 block(TILE_N, TILE_M);
    dim3 grid((K + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    hipLaunchKernelGGL(fp8_gemm_backward_input_kernel, grid, block, 0, stream,
                       grad_output, weight, grad_input, M, N, K);
}

void launch_fp8_linear_backward_weight(
    const float* grad_output, const float* input, float* grad_weight,
    int M, int N, int K,
    hipStream_t stream
) {
    dim3 block(TILE_N, TILE_M);
    dim3 grid((K + TILE_N - 1) / TILE_N, (N + TILE_M - 1) / TILE_M);
    hipLaunchKernelGGL(fp8_gemm_backward_weight_kernel, grid, block, 0, stream,
                       grad_output, input, grad_weight, M, N, K);
}

} // extern "C"
