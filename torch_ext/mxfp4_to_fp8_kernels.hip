// torch_ext/mxfp4_to_fp8_kernels.hip
//
// GPU MXFP4 compatibility decoder for devices without native FP4 matrix
// instructions.  This implements the OCP MX E2M1 + UE8M0 layout:
//
//   * 32 logical FP4 values per block
//   * two values packed per byte (even index in the low nibble)
//   * one unsigned E8M0 (power-of-two) scale byte per block
//
// It deliberately expands to E4M3 rather than attempting to issue an FP4
// instruction.  The resulting E4M3 tensor is intended for hipBLASLt's normal
// FP8 GEMM path on gfx12 and other supported FP8 targets.

#include <hip/hip_runtime.h>
#include <math.h>
#include "../hip_quant_util.h"

constexpr int MXFP4_VALUES_PER_BLOCK = 32;
constexpr int MXFP4_VALUES_PER_BYTE = 2;
constexpr int MXFP4_VALUES_PER_THREAD = 4;

// OCP FP4 E2M1: sign | exponent[2] | mantissa[1].  E=0,M=0 is signed
// zero; E=0,M=1 is the 0.5 subnormal.  The remaining finite magnitudes are
// 1, 1.5, 2, 3, 4, and 6.  Decode explicitly so this remains a software
// compatibility path even when hip_fp4.h is unavailable for the target.
__device__ __forceinline__ float mxfp4_e2m1_to_fp32(uint8_t code) {
    const uint8_t sign = code >> 3;
    const uint8_t exponent = (code >> 1) & 0x3;
    const uint8_t mantissa = code & 0x1;

    float magnitude;
    if (exponent == 0) {
        magnitude = mantissa ? 0.5f : 0.0f;
    } else {
        // (1 + mantissa / 2) * 2^(exponent - 1)
        magnitude = ldexpf(1.0f + 0.5f * (float)mantissa, (int)exponent - 1);
    }
    return sign ? -magnitude : magnitude;
}

// UE8M0 is an unsigned, zero-mantissa FP8 scale: 2^(encoded_exponent - 127).
// 0xFF is its sole NaN encoding. Applying a finite scale before E4M3
// conversion gives hipBLASLt a fully dequantized FP8 matrix with a unit GEMM
// scale. Values beyond E4M3's finite range saturate in fp32_to_fp8_e4m3,
// exactly as the rest of this extension's E4M3 API does.
__device__ __forceinline__ float mxfp4_ue8m0_scale(uint8_t encoded_exponent) {
    if (encoded_exponent == 0xFF) {
        return NAN;
    }
    return ldexpf(1.0f, (int)encoded_exponent - 127);
}

__global__ __launch_bounds__(256, 4)
void dequant_mxfp4_to_e4m3_kernel(
    const uint8_t* __restrict__ packed_values,
    const uint8_t* __restrict__ block_scales,
    uint8_t* __restrict__ output,
    int64_t nrows,
    int n_per_row
) {
    const int64_t first = ((int64_t)blockIdx.x * 256 + threadIdx.x) * MXFP4_VALUES_PER_THREAD;
    const int packed_bytes_per_row = n_per_row / MXFP4_VALUES_PER_BYTE;
    const int blocks_per_row = n_per_row / MXFP4_VALUES_PER_BLOCK;
    const int64_t total = nrows * (int64_t)n_per_row;

    #pragma unroll
    for (int offset = 0; offset < MXFP4_VALUES_PER_THREAD; ++offset) {
        const int64_t linear = first + offset;
        if (linear >= total) {
            continue;
        }

        const int64_t row = linear / n_per_row;
        const int col = (int)(linear - row * n_per_row);
        const uint8_t packed = packed_values[row * packed_bytes_per_row + col / 2];
        const uint8_t fp4 = (col & 1) ? (packed >> 4) : (packed & 0x0F);
        const uint8_t scale_bits = block_scales[row * blocks_per_row + col / MXFP4_VALUES_PER_BLOCK];
        const float value = mxfp4_e2m1_to_fp32(fp4) * mxfp4_ue8m0_scale(scale_bits);
        output[linear] = fp32_to_fp8_e4m3(value);
    }
}

extern "C" void launch_dequant_mxfp4_to_fp8(
    const uint8_t* packed_values,
    const uint8_t* block_scales,
    uint8_t* output,
    int64_t nrows,
    int n_per_row,
    hipStream_t stream
) {
    if (nrows == 0 || n_per_row == 0) {
        return;
    }
    const int64_t total = nrows * (int64_t)n_per_row;
    const int64_t values_per_block = 256 * MXFP4_VALUES_PER_THREAD;
    const int64_t blocks = (total + values_per_block - 1) / values_per_block;
    hipLaunchKernelGGL(dequant_mxfp4_to_e4m3_kernel, dim3((unsigned int)blocks), dim3(256), 0, stream,
        packed_values, block_scales, output, nrows, n_per_row);
}
