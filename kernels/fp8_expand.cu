#include <hip/hip_runtime.h>
#include "../hip_quant_util.h"

// Expand FP8 E4M3 source data to float32 on-device.
// Used by quantize_tensor_fp8_input() to support quantizing
// from FP8 input, halving host memory and transfer bandwidth.

extern "C" __global__
__launch_bounds__(256, 4)
void fp8_to_f32_expand_kernel(
    const uint8_t * __restrict__ src_fp8,
    float * __restrict__ dst_f32,
    int64_t total_elements
) {
    int64_t idx = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (idx >= total_elements) return;
    dst_f32[idx] = fp8_e4m3_to_fp32(src_fp8[idx]);
}
