#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

// F8_E5M2: 32-element blocks, 32 raw FP8 E5M2 values (no scale factor).
// Intended for gradients/backward-pass data where dynamic range matters more
// than mantissa precision.

extern "C" __global__
__launch_bounds__(256, 4)
void quantize_f8_e5m2_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)nrows * n_per_row;
    if (idx >= total) return;

    dst[idx] = fp32_to_fp8_e5m2(src[idx]);
}
