#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

// F8_E4M3: 32-element blocks, 32 raw FP8 E4M3 values (no scale factor).
// Each float is directly converted to FP8 E4M3 format.
// No block scaling needed since FP8 is already floating-point.
// 8 bits per weight, zero overhead.

extern "C" __global__
__launch_bounds__(256, 4)
void quantize_f8_e4m3_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)nrows * n_per_row;
    if (idx >= total) return;

    dst[idx] = fp32_to_fp8_e4m3(src[idx]);
}
